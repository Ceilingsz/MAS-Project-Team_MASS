"""STL-single: ONE drone, windowed formal planning with the authors' STL MILP.

Every time the drone has finished its plan:
  1. pick the next M viewpoints greedily (each scored from the previous pick, so they form a tour);
  2. ask the STL planner for timed waypoints satisfying
         eventually-reach each of the M goal boxes within tmax  AND  always avoid every obstacle
     (the MILP decides the visiting ORDER and the timing, with an obstacle margin of r_b + eps);
  3. execute it; repeat on what is left (receding horizon).
If the MILP is infeasible / too slow / too large, fall back to a plain shortest path to the first pick and
count it (stats: stl_windows, stl_fallbacks, stl_degraded, stl_solve_s). A window that is infeasible or too
large for the solver licence is retried with fewer goals first.

This is the formal counterpart of B0 (single-drone greedy) and the first rung of the STL ladder.
Needs gurobipy and the STLPlanning checkout (see third_party/README.md).
"""
from __future__ import annotations

import numpy as np

from mascov.common.model import CentralModel, register
from mascov.common.types import Command, Plan
from mascov.planning.greedy import choose_viewpoint
from mascov.stl.planner import STLPlanner, verify_plan


@register("stl_single")
class STLSingle(CentralModel):
    def __init__(self, window_goals: int = 4, delta: float = 1.0, time_limit: float = 15.0,
                 goal_half_width: float = 0.25, window_margin: float = 2.0):
        self.M, self.delta, self.time_limit = window_goals, delta, time_limit
        self.goal_half_width, self.window_margin = goal_half_width, window_margin

    def reset(self, world, agent_ids):
        if len(agent_ids) != 1:
            raise ValueError("stl_single is a single-drone planner")
        self.w = world
        self.vp = world.viewpoints
        self.stl = STLPlanner(world.cfg, goal_half_width=self.goal_half_width, time_limit=self.time_limit)
        self.tries: dict[int, int] = {}                          # viewpoint -> times we have headed there
        self.s = dict(stl_windows=0, stl_fallbacks=0, stl_solve_s=0.0, stl_binaries_max=0, stl_goals=0, stl_verify_fail=0, stl_degraded=0)

    def _pick(self, pos, covered) -> list[int]:
        """Greedy tour of up to M viewpoints: each pick is scored from the previous one."""
        vp, cfg = self.vp, self.w.cfg
        claimed, here, picks = covered.copy(), np.asarray(pos, dtype=float), []
        for _ in range(self.M):
            travel = self.w.dist_to_viewpoints(here) / cfg.v_max
            exclude = {u for u, c in self.tries.items() if c >= 2} | set(picks)
            v = choose_viewpoint(vp, travel, claimed, exclude, self.delta)
            if v is None:
                break
            picks.append(v)
            claimed |= vp.sees[v]
            here = vp.positions[v]
        return picks

    def _window(self, pts):
        """Bounding box of the geodesic route start->g1->g2..., expanded by a margin. Including the route
        (not just the endpoints) guarantees a feasible detour, e.g. through a door, lies inside the window."""
        cfg, m = self.w.cfg, self.window_margin
        c = cfg.clearance
        return ((max(c, pts[:, 0].min() - m), min(cfg.width - c, pts[:, 0].max() + m)),
                (max(c, pts[:, 1].min() - m), min(cfg.height - c, pts[:, 1].max() + m)))

    def step(self, obs):
        o = obs[0]
        if not o.plan_finished:
            return {}
        cfg = self.w.cfg
        picks = self._pick(o.pos, o.belief.covered)
        if not picks:
            return {0: Command(done=True)}
        goals = self.vp.positions[picks]
        start = self.w.planner.snap(o.pos, inset=0.03)           # tracking error may leave the free space
        lead = float(np.linalg.norm(start - o.pos)) / cfg.plan_speed
        res, used = None, 0
        sizes = sorted({len(picks), max(1, len(picks) // 2), 1}, reverse=True)
        for attempt, m in enumerate(sizes):                      # degrade: fewer goals if too big / infeasible
            goals = self.vp.positions[picks[:m]]
            route_pts, n_corners, order_len = [start], 0, 0.0
            for a_, b_ in zip([start, *goals[:-1]], goals):
                path = self.w.planner.path(a_, b_)
                route_pts.extend(path[1:])
                n_corners += max(0, len(path) - 2)
                order_len += sum(float(np.linalg.norm(path[k + 1] - path[k])) for k in range(len(path) - 1))
            window = self._window(np.asarray(route_pts))
            tmax = 2.0 * order_len / cfg.plan_speed + 15.0
            res = self.stl.plan_visits(start, goals, self.w.obstacles, tmax=tmax, window=window,
                                       max_segs=2 * m + n_corners + 2)
            self.s["stl_solve_s"] += res.solve_s
            if res.n_binaries:
                self.s["stl_binaries_max"] = max(self.s["stl_binaries_max"], res.n_binaries)
            if res.ok:
                used = m
                self.s["stl_degraded"] += int(attempt > 0)
                break
        self.s["stl_windows"] += 1
        self.s["stl_goals"] += used if res.ok else 0
        if res.ok:
            rep = verify_plan(res.waypoints, start, goals, self.w.obstacles, cfg, tmax=tmax,
                              goal_half_width=self.goal_half_width)
            self.s["stl_verify_fail"] += int(not rep["all_ok"])   # never trust the solver blindly
            stl_plan = res.to_plan(t0=o.t + lead)
            plan = Plan(np.r_[o.t, stl_plan.t], np.vstack([o.pos[None], stl_plan.p]))
            for v in picks[:used]:
                self.tries[v] = self.tries.get(v, 0) + 1
        else:                                                    # fall back to a plain route to the first pick
            self.s["stl_fallbacks"] += 1
            route = self.w.planner.path(o.pos, self.vp.positions[picks[0]])
            plan = Plan.from_polyline(route, o.t, cfg.plan_speed)
            self.tries[picks[0]] = self.tries.get(picks[0], 0) + 1
        return {0: Command(plan=plan)}

    def stats(self):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.s.items()}
