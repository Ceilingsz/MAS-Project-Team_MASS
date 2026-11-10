"""STL-team: N drones planned JOINTLY by the authors' multi-agent STL MILP, in synchronous windows.

Each window (all drones idle):
  1. greedily pick up to N * goals_per_drone viewpoints (scored by the nearest drone's travel time);
  2. ONE joint MILP: every goal is reached by at least one drone (the solver assigns and orders them),
     drones avoid obstacles and keep reference paths >= 2*eps + d_safe apart, minimising the MAKESPAN;
  3. all drones execute their timed waypoints; the next window starts when ALL of them have finished.
If the joint problem is infeasible or too large for the solver licence it is retried with fewer goals; if
that still fails, each drone is sent to its own nearest unclaimed pick by a plain shortest path (NO
separation guarantee, counted in stats as team_fallbacks).

Honest costs of this design: windows are synchronous, so drones that finish early wait for the slowest, and
the joint MILP is the centralised formulation (no distributed claims or messages). Needs gurobipy + the
STLPlanning checkout (third_party/README.md) and, in practice, WorldConfig(eps_track=0.45) and drone starts
at least ~2 m apart.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from mascov.common.model import CentralModel, register
from mascov.common.types import Command, Plan
from mascov.planning.greedy import choose_viewpoint
from mascov.stl.planner import STLPlanner, verify_team


@register("stl_team")
class STLTeam(CentralModel):
    def __init__(self, goals_per_drone: int = 2, delta: float = 1.0, time_limit: float = 20.0,
                 goal_half_width: float = 0.25, window_margin: float = 2.0):
        self.gpd, self.delta, self.time_limit = goals_per_drone, delta, time_limit
        self.goal_half_width, self.window_margin = goal_half_width, window_margin

    def reset(self, world, agent_ids):
        self.w, self.ids = world, list(agent_ids)
        self.vp = world.viewpoints
        self.stl = STLPlanner(world.cfg, goal_half_width=self.goal_half_width, time_limit=self.time_limit)
        self.tries: dict[int, int] = {}
        self.s = dict(team_windows=0, team_fallbacks=0, team_degraded=0, team_solve_s=0.0,
                      team_binaries_max=0, team_verify_fail=0, team_goals=0)

    # ---------------------------------------------------------------- goal selection
    def _pick(self, positions, covered, count) -> list[int]:
        vp, cfg = self.vp, self.w.cfg
        travel = np.vstack([self.w.dist_to_viewpoints(p) for p in positions]).min(axis=0) / cfg.v_max
        claimed, picks = covered.copy(), []
        for _ in range(count):
            exclude = {u for u, c in self.tries.items() if c >= 2} | set(picks)
            v = choose_viewpoint(vp, travel, claimed, exclude, self.delta)
            if v is None:
                break
            picks.append(v)
            claimed |= vp.sees[v]
        return picks

    def _window(self, pts):
        cfg, m = self.w.cfg, self.window_margin
        c = cfg.clearance
        return ((max(c, pts[:, 0].min() - m), min(cfg.width - c, pts[:, 0].max() + m)),
                (max(c, pts[:, 1].min() - m), min(cfg.height - c, pts[:, 1].max() + m)))

    # ---------------------------------------------------------------- one window
    def step(self, obs):
        if not all(o.plan_finished for o in obs.values()):
            return {}
        cfg, planner = self.w.cfg, self.w.planner
        ids = sorted(obs)
        covered = np.zeros(len(self.vp.coverable), dtype=bool)
        for o in obs.values():
            covered |= o.belief.covered
        pos = np.array([obs[i].pos for i in ids])
        picks = self._pick(pos, covered, len(ids) * self.gpd)
        if not picks:
            return {i: Command(done=True) for i in ids}
        t = obs[ids[0]].t
        starts = np.array([planner.snap(p, inset=0.03) for p in pos])
        lead = np.linalg.norm(starts - pos, axis=1) / cfg.plan_speed

        sizes = sorted({len(picks), max(len(ids), len(picks) // 2), min(len(ids), len(picks)), 1}, reverse=True)
        res, goals, tmax, used = None, None, None, 0
        for attempt, m in enumerate(sizes):
            if m > len(picks):
                continue
            goals = self.vp.positions[picks[:m]]
            route_pts, longest, n_corners = [s for s in starts], 0.0, 0
            for g in goals:                                            # route from the nearest drone to each goal
                best = min(range(len(ids)), key=lambda k: planner.distance(starts[k], g))
                path = planner.path(starts[best], g)
                route_pts.extend(path[1:])
                n_corners = max(n_corners, max(0, len(path) - 2))
                longest += sum(float(np.linalg.norm(path[k + 1] - path[k])) for k in range(len(path) - 1))
            window = self._window(np.asarray(route_pts))
            tmax = 2.0 * longest / cfg.plan_speed + 20.0
            per = math.ceil(m / len(ids))
            res = self.stl.plan_team(starts, goals, self.w.obstacles, tmax=tmax, window=window,
                                     max_segs=2 * per + n_corners + 2)
            self.s["team_solve_s"] += res.solve_s
            if res.n_binaries:
                self.s["team_binaries_max"] = max(self.s["team_binaries_max"], res.n_binaries)
            if res.ok:
                used = m
                self.s["team_degraded"] += int(attempt > 0)
                break
        self.s["team_windows"] += 1

        if res is not None and res.ok:
            rep = verify_team(res.paths, starts, goals, self.w.obstacles, cfg, tmax=tmax,
                              goal_half_width=self.goal_half_width)
            self.s["team_verify_fail"] += int(not rep["all_ok"])        # never trust the solver blindly
            self.s["team_goals"] += used
            cmds = {}
            for k, i in enumerate(ids):
                stl_plan = res.to_plans(t0=t + lead[k])[k]
                cmds[i] = Command(plan=Plan(np.r_[t, stl_plan.t], np.vstack([pos[k][None], stl_plan.p])))
            for v in picks[:used]:
                self.tries[v] = self.tries.get(v, 0) + 1
            return cmds

        # fallback: each drone to its own nearest unclaimed pick, plain shortest paths, no joint guarantee
        self.s["team_fallbacks"] += 1
        cost = np.array([[planner.distance(pos[k], self.vp.positions[v]) for v in picks] for k in range(len(ids))])
        n_use = min(len(ids), len(picks))
        best = min(itertools.permutations(range(len(picks)), n_use), key=lambda perm: max(cost[k, perm[k]] for k in range(n_use)))
        cmds = {}
        for k, j in enumerate(best):
            route = planner.path(pos[k], self.vp.positions[picks[j]])
            cmds[ids[k]] = Command(plan=Plan.from_polyline(route, t, cfg.plan_speed))
            self.tries[picks[j]] = self.tries.get(picks[j], 0) + 1
        return cmds

    def stats(self):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.s.items()}
