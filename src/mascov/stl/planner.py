"""Adapter around the authors' PWLPlan.py (Sun, Chen, Mitra, Fan, RA-L 2022).

What it gives the rest of the project:
  * STLPlanner.plan_visits(): "eventually reach each goal box within tmax, always stay out of every
    (inflated) obstacle" for ONE drone -> timed waypoints (a Plan), found by the authors' MILP encoding.
  * verify_plan(): an independent re-check of any returned plan, so we never just trust the solver.

Things about PWLPlan.py this adapter works around (found by reading it):
  * plan() writes "test.lp" into the working directory on every call -> we run it in a temp directory.
  * it prints a lot and Gurobi logs to stdout -> silenced; stdout is parsed for the binary count.
  * it swallows solver exceptions and returns [None] -> reported as STLResult(ok=False).
  * goal regions are shrunk by a hard-coded 0.1 (not by the tracking error) -> exposed as `goal_shrink`.
  * the speed limit and inter-agent separation use the 1-norm, so they are conservative.
Obstacles are avoided with margin  bloat + size  = eps_track + drone_radius, i.e. exactly cfg.clearance.
The guarantee is conditional on the real tracking error staying below eps_track (see README).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, box

from mascov.common.types import Plan
from mascov.geometry.polygons import halfspaces

_MODULE = None


class STLUnavailable(RuntimeError):
    pass


def _find_repo() -> Path:
    candidates = [os.environ.get("STL_PLANNING_PATH"),
                  Path(__file__).resolve().parents[3] / "third_party" / "STLPlanning"]
    for c in candidates:
        if c and (Path(c) / "PWLPlan.py").exists():
            return Path(c)
    raise STLUnavailable("PWLPlan.py not found. Run scripts/setup_stlplanning.sh or set STL_PLANNING_PATH.")


def load_pwlplan():
    """Import the authors' module by file path (cached)."""
    global _MODULE
    if _MODULE is None:
        path = _find_repo() / "PWLPlan.py"
        try:
            import gurobipy  # noqa: F401
        except ImportError as e:
            raise STLUnavailable("gurobipy is not installed (pip install gurobipy)") from e
        spec = importlib.util.spec_from_file_location("PWLPlan", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULE = mod
    return _MODULE


def stl_available() -> bool:
    try:
        load_pwlplan()
        return True
    except STLUnavailable:
        return False


@dataclass
class STLResult:
    ok: bool
    waypoints: list = field(default_factory=list)     # [(t, x, y), ...]
    n_segs: int = 0
    solve_s: float = 0.0
    n_binaries: int | None = None
    message: str = ""

    def to_plan(self, t0: float = 0.0) -> Plan:
        t = [w[0] + t0 for w in self.waypoints]
        return Plan(t, [[w[1], w[2]] for w in self.waypoints])


@dataclass
class TeamResult:
    ok: bool
    paths: list = field(default_factory=list)         # per agent: [(t, x, y), ...]
    assignment: list = field(default_factory=list)    # goal index -> agent that reaches it (-1 if none)
    n_segs: int = 0
    solve_s: float = 0.0
    n_binaries: int | None = None
    message: str = ""

    def to_plans(self, t0: float = 0.0) -> list[Plan]:
        return [Plan([w[0] + t0 for w in path], [[w[1], w[2]] for w in path]) for path in self.paths]


def _box_halfspaces(center, half_width: float):
    cx, cy = center
    A = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    b = np.array([-(cx - half_width), cx + half_width, -(cy - half_width), cy + half_width])
    return A, b


class STLPlanner:
    def __init__(self, cfg, goal_half_width: float = 0.25, goal_shrink: float = 0.1,
                 time_limit: float = 20.0, mip_gap: float = 1e-2, big_m: float = 1e3):
        self.cfg = cfg
        self.goal_half_width = goal_half_width
        self.goal_shrink = goal_shrink
        self.time_limit, self.mip_gap, self.big_m = time_limit, mip_gap, big_m

    def plan_visits(self, start, goals, obstacles, *, tmax: float, max_segs: int = 8,
                    window=None, bloat: float | None = None, num_segs: int | None = None) -> STLResult:
        """Plan one drone: eventually reach every goal box within tmax while avoiding all obstacles.

        num_segs : solve for exactly this many segments instead of searching 1..max_segs
        window : ((xmin, xmax), (ymin, ymax)) the waypoints are confined to (obstacles outside it can be
                 ignored, since segments between waypoints stay inside the convex window).
        """
        mod = load_pwlplan()
        import gurobipy as gp
        cfg = self.cfg
        eps = cfg.eps_track if bloat is None else bloat
        start = np.asarray(start, dtype=float)
        goals = np.asarray(goals, dtype=float).reshape(-1, 2)
        if window is None:
            c = cfg.clearance
            window = ((c, cfg.width - c), (c, cfg.height - c))
        (x0, x1), (y0, y1) = window
        win_box = box(x0 - cfg.clearance, y0 - cfg.clearance, x1 + cfg.clearance, y1 + cfg.clearance)
        near = [o for o in obstacles if o.intersects(win_box)]

        avoid = [mod.Node("negmu", info={"A": H, "b": b}) for H, b in (halfspaces(o) for o in near)]
        goal_nodes = []
        for g in goals:
            A, b = _box_halfspaces(g, self.goal_half_width)
            goal_nodes.append(mod.Node("F", deps=[mod.Node("mu", info={"A": A, "b": b})], info={"int": [0, tmax]}))
        deps = list(goal_nodes)
        if avoid:
            deps.insert(0, mod.Node("A", deps=[mod.Node("and", deps=avoid)], info={"int": [0, tmax]}))
        spec = mod.Node("and", deps=deps)

        # contain the library's quirks: hard-coded goal shrink, big-M, global Gurobi params, and the
        # exceptions it swallows (record them so a licence/size problem is not mistaken for infeasibility)
        errors: list[str] = []

        class _RecordingModel(gp.Model):
            def optimize(self, *a, **k):
                try:
                    return super().optimize(*a, **k)
                except Exception as e:
                    errors.append(str(e))
                    raise

        orig_mu, orig_M, orig_Model = mod.mu, mod.M, mod.Model
        mod.Model = _RecordingModel
        mod.mu = lambda i, PWL, bloat_factor, A, b: orig_mu(i, PWL, self.goal_shrink, A, b)
        mod.setM(self.big_m)
        gp.setParam("OutputFlag", 0)
        gp.setParam("TimeLimit", self.time_limit)
        buf = io.StringIO()
        t0 = time.perf_counter()
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                with contextlib.redirect_stdout(buf):
                    out = mod.plan([start.tolist()], [spec], bloat=eps, limits=window,
                                   num_segs=num_segs, max_segs=None if num_segs else max_segs,
                                   vmax=cfg.plan_speed, MIPGap=self.mip_gap, tmax=tmax, size=cfg.drone_radius)
        except Exception as e:                       # e.g. Gurobi licence size limit
            return STLResult(False, solve_s=time.perf_counter() - t0, message=f"{type(e).__name__}: {e}")
        finally:
            os.chdir(cwd)
            mod.mu, mod.M, mod.Model = orig_mu, orig_M, orig_Model
        solve_s = time.perf_counter() - t0
        text = buf.getvalue()
        segs = [int(m) for m in re.findall(r"num_segs (\d+)", text)]
        nbin = [int(m) for m in re.findall(r"NumBinVars: (\d+)", text)]
        if out is None or out[0] is None:
            return STLResult(False, n_segs=segs[-1] if segs else 0, solve_s=solve_s,
                             n_binaries=nbin[-1] if nbin else None,
                             message=errors[-1] if errors else "infeasible or time limit")
        wps = [(float(p[1]), float(p[0][0]), float(p[0][1])) for p in out[0]]
        return STLResult(True, wps, len(wps) - 1, solve_s, nbin[-1] if nbin else None)

    def plan_team(self, starts, goals, obstacles, *, tmax: float, max_segs: int = 8, num_segs: int | None = None,
                  window=None, makespan: bool = True) -> TeamResult:
        """Joint plan for N drones: every goal is reached by AT LEAST ONE drone (the solver assigns them),
        all drones avoid the obstacles, and drones keep their planned paths apart.

        Three deliberate departures from the upstream code (patched in only for the duration of the call):
          * equal end times: upstream stops enforcing separation once a drone reaches its last waypoint (it
            "vanishes"), but a real drone stays put there. Making every drone end at the same time T fixes that.
          * separation margin: upstream keeps reference paths 2*eps apart, which lets bodies touch when both
            drift by eps. We use bloat + drone_radius, i.e. >= 2*eps + d_safe between references.
          * objective: upstream minimises the SUM of finishing times, which gives no incentive to balance
            work; we minimise the makespan (max finishing time) with 0.01 x the sum as a tie-break.
        All drones use the same number of segments, and they all start at t=0 (a synchronous window).
        """
        mod = load_pwlplan()
        import gurobipy as gp
        cfg = self.cfg
        eps = cfg.eps_track
        starts = np.asarray(starts, dtype=float).reshape(-1, 2)
        goals = np.asarray(goals, dtype=float).reshape(-1, 2)
        n = len(starts)
        if window is None:
            c = cfg.clearance
            window = ((c, cfg.width - c), (c, cfg.height - c))
        (x0, x1), (y0, y1) = window
        win_box = box(x0 - cfg.clearance, y0 - cfg.clearance, x1 + cfg.clearance, y1 + cfg.clearance)
        near = [o for o in obstacles if o.intersects(win_box)]

        def avoid_spec():
            nodes = [mod.Node("negmu", info={"A": H, "b": b}) for H, b in (halfspaces(o) for o in near)]
            if nodes:
                return mod.Node("A", deps=[mod.Node("and", deps=nodes)], info={"int": [0, tmax]})
            A, b = _box_halfspaces(((x0 + x1) / 2, (y0 + y1) / 2), max(x1 - x0, y1 - y0))      # trivially true
            return mod.Node("and", deps=[mod.Node("F", deps=[mod.Node("mu", info={"A": A, "b": b})], info={"int": [0, tmax]})])

        specs = [avoid_spec() for _ in range(n)]
        tasks = []
        for _ in range(n):
            row = []
            for g in goals:
                A, b = _box_halfspaces(g, self.goal_half_width)
                row.append(mod.Node("F", deps=[mod.Node("mu", info={"A": A, "b": b})], info={"int": [0, tmax]}))
            tasks.append(row)

        errors: list[str] = []
        state: dict = {}

        class _TeamModel(gp.Model):
            def optimize(self, *a, **k):
                try:
                    return super().optimize(*a, **k)
                except Exception as e:
                    errors.append(str(e))
                    raise

            def setObjective(self, obj, sense=None):
                if makespan and "T" in state:
                    obj = state["T"] + 0.01 * obj
                return super().setObjective(obj, sense if sense is not None else gp.GRB.MINIMIZE)

        orig_mu, orig_M, orig_Model = mod.mu, mod.M, mod.Model
        orig_mutual = mod.add_mutual_clearance_constraints

        def mutual(m, PWLs, bloat):
            if makespan:
                T = m.addVar(lb=0.0)
                m.update()
                for PWL in PWLs:
                    # all drones end at the same time T. Upstream drops a drone from the separation
                    # constraints after its last waypoint, yet in reality it stays there; forcing equal end
                    # times keeps every drone present (waiting if need be) for the whole window.
                    m.addConstr(PWL[-1][1] == T)
                state["T"] = T
            return orig_mutual(m, PWLs, bloat + cfg.drone_radius)

        mod.mu = lambda i, PWL, bloat_factor, A, b: orig_mu(i, PWL, self.goal_shrink, A, b)
        mod.Model, mod.add_mutual_clearance_constraints = _TeamModel, mutual
        mod.setM(self.big_m)
        gp.setParam("OutputFlag", 0)
        gp.setParam("TimeLimit", self.time_limit)
        buf = io.StringIO()
        t0 = time.perf_counter()
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                with contextlib.redirect_stdout(buf):
                    out = mod.plan(starts.tolist(), specs, bloat=eps, limits=window, tasks=tasks, num_segs=num_segs,
                                   max_segs=None if num_segs else max_segs, vmax=cfg.plan_speed,
                                   MIPGap=self.mip_gap, tmax=tmax, size=cfg.drone_radius)
        except Exception as e:
            return TeamResult(False, solve_s=time.perf_counter() - t0, message=f"{type(e).__name__}: {e}")
        finally:
            os.chdir(cwd)
            mod.mu, mod.M, mod.Model = orig_mu, orig_M, orig_Model
            mod.add_mutual_clearance_constraints = orig_mutual
        solve_s = time.perf_counter() - t0
        text = buf.getvalue()
        segs = [int(m) for m in re.findall(r"num_segs (\d+)", text)]
        nbin = [int(m) for m in re.findall(r"NumBinVars: (\d+)", text)]
        if out is None or out[0] is None:
            return TeamResult(False, n_segs=segs[-1] if segs else 0, solve_s=solve_s,
                              n_binaries=nbin[-1] if nbin else None,
                              message=errors[-1] if errors else "infeasible or time limit")
        paths = [[(float(p[1]), float(p[0][0]), float(p[0][1])) for p in agent] for agent in out]
        assignment = []
        for g in goals:
            gb = box(g[0] - self.goal_half_width, g[1] - self.goal_half_width, g[0] + self.goal_half_width, g[1] + self.goal_half_width)
            hits = [(LineString([(w[1], w[2]) for w in path]).intersects(gb), i) for i, path in enumerate(paths)]
            assignment.append(next((i for ok, i in hits if ok), -1))
        return TeamResult(True, paths, assignment, len(paths[0]) - 1, solve_s, nbin[-1] if nbin else None)


def verify_plan(waypoints, start, goals, obstacles, cfg, *, tmax: float, goal_half_width: float = 0.25,
                tol: float = 1e-4) -> dict:
    """Independent check of a returned plan against the specification (no trust in the solver)."""
    P = np.array([[w[1], w[2]] for w in waypoints])
    T = np.array([w[0] for w in waypoints])
    line = LineString(P) if len(P) > 1 else Point(P[0])
    report = {"starts_at_start": bool(np.allclose(P[0], start, atol=1e-6)),
              "times_increase": bool(np.all(np.diff(T) > 0)),
              "within_tmax": bool(T[-1] <= tmax + 1e-6)}
    dmin = min((line.distance(o) for o in obstacles), default=np.inf)
    report["min_obstacle_distance"] = float(dmin)
    report["obstacle_clearance_ok"] = bool(dmin >= cfg.clearance - tol)
    reached = []
    for g in np.asarray(goals, dtype=float).reshape(-1, 2):
        reached.append(bool(line.intersects(box(g[0] - goal_half_width, g[1] - goal_half_width,
                                                g[0] + goal_half_width, g[1] + goal_half_width))))
    report["goals_reached"] = reached
    seg = np.diff(P, axis=0)
    l1 = np.abs(seg).sum(axis=1)
    report["max_l1_speed"] = float((l1 / np.maximum(np.diff(T), 1e-9)).max()) if len(seg) else 0.0
    report["speed_ok"] = bool(report["max_l1_speed"] <= cfg.plan_speed + 1e-6)
    report["in_workspace"] = bool(P.min() >= -1e-6 and P[:, 0].max() <= cfg.width + 1e-6 and P[:, 1].max() <= cfg.height + 1e-6)
    report["all_ok"] = all([report["starts_at_start"], report["times_increase"], report["within_tmax"],
                            report["obstacle_clearance_ok"], all(reached), report["speed_ok"], report["in_workspace"]])
    return report


def verify_team(paths, starts, goals, obstacles, cfg, *, tmax: float, goal_half_width: float = 0.25,
                tol: float = 1e-3) -> dict:
    """Independent check of a joint plan: each agent individually, every goal reached by someone, and the
    synchronised distance between every pair of reference paths (must be >= 2*eps + d_safe)."""
    per_agent = [verify_plan(p, starts[i], np.zeros((0, 2)), obstacles, cfg, tmax=tmax) for i, p in enumerate(paths)]
    report = {"agents_ok": all(r["obstacle_clearance_ok"] and r["speed_ok"] and r["in_workspace"] and r["times_increase"]
                               and r["starts_at_start"] and r["within_tmax"] for r in per_agent)}
    goals = np.asarray(goals, dtype=float).reshape(-1, 2)
    lines = [LineString([(w[1], w[2]) for w in p]) for p in paths]
    reached = [any(l.intersects(box(g[0] - goal_half_width, g[1] - goal_half_width,
                                    g[0] + goal_half_width, g[1] + goal_half_width)) for l in lines) for g in goals]
    report["goals_reached"] = reached
    t_end = max(p[-1][0] for p in paths)
    ts = np.arange(0.0, t_end + 0.05, 0.05)
    pos = [np.c_[np.interp(ts, [w[0] for w in p], [w[1] for w in p]),
                 np.interp(ts, [w[0] for w in p], [w[2] for w in p])] for p in paths]
    dmin = min((np.linalg.norm(pos[i] - pos[j], axis=1).min() for i in range(len(paths)) for j in range(i + 1, len(paths))),
               default=np.inf)
    need = 2 * cfg.eps_track + cfg.d_safe
    report["min_pair_reference_distance"] = float(dmin)
    report["required_pair_distance"] = need
    report["separation_ok"] = bool(dmin >= need - tol)
    report["all_ok"] = bool(report["agents_ok"] and all(reached) and report["separation_ok"])
    return report
