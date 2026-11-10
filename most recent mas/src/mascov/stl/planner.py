"""Planner for the coverage specification: a MILP encoding (stl/encoding.py) solved with HiGHS.

What it gives the rest of the project:
  * STLPlanner.plan_visits(): "eventually reach each goal box within tmax, always stay out of every
    (inflated) obstacle" for ONE drone -> timed waypoints (a Plan).
  * STLPlanner.plan_team(): the same for N drones jointly (the solver assigns goals to drones), with a
    minimum separation between drones.
  * verify_plan() / verify_team(): independent re-checks of any returned plan, so we never just trust the solver.

The solver is HiGHS, an open-source MILP solver with no size limit and no licence, reached through
scipy.optimize.milp (scipy >= 1.9). There is no Gurobi and no third-party planner code: the encoding is ours,
following the piecewise-linear / big-M scheme of Sun, Chen, Mitra and Fan (RA-L 2022). See stl/encoding.py.

Properties of the encoding that matter downstream (all documented there):
  * the speed limit uses the 1-norm (conservative); separation uses the infinity-norm (conservative);
  * a goal counts as reached when a WAYPOINT lies in the goal box shrunk by ``goal_shrink``;
  * all drones of a joint plan share the waypoint times, so they all end together and none vanishes;
  * obstacles are avoided with margin ``bloat + drone_radius``, i.e. exactly cfg.clearance when bloat=eps_track.
The guarantee is conditional on the real tracking error staying below eps_track (see README).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, box

from mascov.common.types import Plan
from mascov.geometry.polygons import halfspaces
from mascov.stl.encoding import PathsSolution, plan_paths
from mascov.stl.milp import highs_available


class STLUnavailable(RuntimeError):
    pass


def stl_available() -> bool:
    """True if the HiGHS solver (scipy >= 1.9) can be imported."""
    return highs_available()


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
                 time_limit: float = 20.0, mip_gap: float = 1e-2):
        """time_limit: seconds per MILP solve (one solve per tried segment count)."""
        if goal_shrink >= goal_half_width:
            raise ValueError("goal_shrink must be smaller than goal_half_width")
        self.cfg = cfg
        self.goal_half_width = goal_half_width
        self.goal_shrink = goal_shrink
        self.time_limit, self.mip_gap = time_limit, mip_gap

    # ------------------------------------------------------------------ shared machinery
    def _window(self, window):
        if window is not None:
            return window
        c = self.cfg.clearance
        return ((c, self.cfg.width - c), (c, self.cfg.height - c))

    def _near(self, obstacles, window):
        """Obstacles that can matter inside the window (segments between waypoints stay in the convex window)."""
        (x0, x1), (y0, y1) = window
        c = self.cfg.clearance
        win_box = box(x0 - c, y0 - c, x1 + c, y1 + c)
        return [o for o in obstacles if o.intersects(win_box)]

    def _search(self, starts, goals, near, window, tmax, rho, sep, max_segs, num_segs):
        """Solve for the smallest number of segments (1, 2, ...) that is feasible, or exactly ``num_segs``."""
        if not highs_available():
            raise STLUnavailable("the HiGHS solver is not available: pip install 'scipy>=1.9'")
        obstacles = [halfspaces(o) for o in near]
        ks = [num_segs] if num_segs else range(1, max_segs + 1)
        t0 = time.perf_counter()
        last, tried = PathsSolution(False, message="no segment count was tried"), 0
        for k in ks:
            last, tried = plan_paths(starts, goals, obstacles, n_segs=k, tmax=tmax, vmax=self.cfg.plan_speed,
                                     window=window, rho=rho, goal_hw=self.goal_half_width - self.goal_shrink,
                                     sep=sep, time_limit=self.time_limit, mip_gap=self.mip_gap), k
            if last.ok:
                break
        return last, tried, time.perf_counter() - t0

    # ------------------------------------------------------------------ one drone
    def plan_visits(self, start, goals, obstacles, *, tmax: float, max_segs: int = 8,
                    window=None, bloat: float | None = None, num_segs: int | None = None) -> STLResult:
        """Plan one drone: eventually reach every goal box within tmax while avoiding all obstacles.

        num_segs : solve for exactly this many segments instead of searching 1..max_segs
        window : ((xmin, xmax), (ymin, ymax)) the waypoints are confined to (obstacles outside it can be
                 ignored, since segments between waypoints stay inside the convex window).
        bloat : tracking-error margin (default cfg.eps_track); obstacles are kept ``bloat + drone_radius`` away.
        """
        cfg = self.cfg
        eps = cfg.eps_track if bloat is None else bloat
        starts = np.asarray(start, dtype=float).reshape(1, 2)
        goals = np.asarray(goals, dtype=float).reshape(-1, 2)
        window = self._window(window)
        sol, k, solve_s = self._search(starts, goals, self._near(obstacles, window), window, tmax,
                                       eps + cfg.drone_radius, None, max_segs, num_segs)
        if not sol.ok:
            return STLResult(False, n_segs=k, solve_s=solve_s, n_binaries=sol.n_binaries, message=sol.message)
        return STLResult(True, list(sol.paths[0]), k, solve_s, sol.n_binaries)

    # ------------------------------------------------------------------ N drones, jointly
    def plan_team(self, starts, goals, obstacles, *, tmax: float, max_segs: int = 8, num_segs: int | None = None,
                  window=None, makespan: bool = True) -> TeamResult:
        """Joint plan for N drones: every goal is reached by AT LEAST ONE drone (the solver assigns them),
        all drones avoid the obstacles, and drones keep their planned paths at least 2 * eps + d_safe apart
        (infinity-norm).

        All drones use the same number of segments and the same waypoint times, so they all start at t=0 and
        end together at the makespan T (a synchronous window); a drone that finishes early waits in place and
        stays in the separation constraints. The objective is T (with a tiny path-length tie-break).
        ``makespan`` is kept for API compatibility: equal end times are inherent to this encoding.
        """
        cfg = self.cfg
        eps = cfg.eps_track
        starts = np.asarray(starts, dtype=float).reshape(-1, 2)
        goals = np.asarray(goals, dtype=float).reshape(-1, 2)
        window = self._window(window)
        sol, k, solve_s = self._search(starts, goals, self._near(obstacles, window), window, tmax,
                                       eps + cfg.drone_radius, 2.0 * eps + cfg.d_safe, max_segs, num_segs)
        if not sol.ok:
            return TeamResult(False, n_segs=k, solve_s=solve_s, n_binaries=sol.n_binaries, message=sol.message)
        paths = [list(p) for p in sol.paths]
        assignment = []
        for g in goals:
            gb = box(g[0] - self.goal_half_width, g[1] - self.goal_half_width,
                     g[0] + self.goal_half_width, g[1] + self.goal_half_width)
            hits = [(LineString([(w[1], w[2]) for w in path]).intersects(gb), i) for i, path in enumerate(paths)]
            assignment.append(next((i for ok, i in hits if ok), -1))
        return TeamResult(True, paths, assignment, k, solve_s, sol.n_binaries)


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
    # speed is checked as a displacement with 2 cm of slack: solver tolerances (~1e-6 m) are a visible fraction of
    # the very short (0.01 s) segments a MILP may use, so a ratio test would fail correct plans
    report["speed_ok"] = bool(np.all(l1 <= cfg.plan_speed * np.diff(T) + 0.02)) if len(seg) else True
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
