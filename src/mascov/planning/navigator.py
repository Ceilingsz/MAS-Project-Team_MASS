"""Local navigation for ONE drone: follow a route to a goal while avoiding other drones.

This is the collision rule used by the heuristic baselines (it is *not* a safety guarantee):

  * Neighbours that are (nearly) stationary are treated as obstacles: the route is re-planned
    around a disc of radius ``conflict_dist`` centred on them.
  * Neighbours that are moving and have right of way ("higher priority") are predicted over a short
    horizon. If my intended trajectory would come within ``conflict_dist`` of theirs, I yield:
    first by holding in place, or, if holding would still be in the way, by sidestepping to the
    nearest viewpoint that is clear. I resume as soon as the route is clear again.
  * If I have been waiting longer than ``patience`` I push forward (stopping short of any
    stationary neighbour), which makes an idle or lower-priority neighbour move out of my way.

Right of way is a total order supplied by the caller (e.g. lower agent id first).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from mascov.common.types import Plan
from mascov.geometry.pathing import PathPlanner

STATIONARY_SPEED = 0.1      # m/s below which an extrapolated neighbour counts as stationary


def plan_positions(plan: Plan, ts: np.ndarray) -> np.ndarray:
    """Reference positions of a plan at absolute times ts (holds before/after the ends)."""
    return np.c_[np.interp(ts, plan.t, plan.p[:, 0]), np.interp(ts, plan.t, plan.p[:, 1])]


@dataclass
class Neighbor:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    higher: bool                       # has right of way over me
    plan: Plan | None = None           # known plan (centralised models); else constant velocity

    def traj(self, ts: np.ndarray) -> np.ndarray:
        if self.plan is not None:
            return plan_positions(self.plan, ts)
        return self.pos[None, :] + self.vel[None, :] * (ts - ts[0])[:, None]

    def is_static(self, tr: np.ndarray) -> bool:
        still = np.linalg.norm(tr[-1] - tr[0]) < 0.05
        return bool(still and (self.plan is not None or np.linalg.norm(self.vel) < STATIONARY_SPEED))


class Navigator:
    def __init__(self, cfg, planner: PathPlanner, candidates: np.ndarray, horizon: float = 2.5,
                 step: float = 0.2, margin: float = 0.5, static_margin: float = 0.3, patience: float = 4.0,
                 max_candidates: int = 25, sidestep_radius: float = 5.0, seed: int = 0):
        self.cfg, self.planner = cfg, planner
        self.cands = np.asarray(candidates, dtype=float)
        self.horizon, self.step = horizon, step
        self.conflict_dist = cfg.d_safe + margin            # keep-away from MOVING higher-priority drones
        self.static_dist = cfg.d_safe + static_margin       # keep-away from STATIONARY drones (route around)
        self.rng = np.random.default_rng(seed)
        self.near_horizon = 1.0                             # seconds of short-range look-ahead for lower-priority drones
        self.patience, self.max_candidates, self.sidestep_radius = patience, max_candidates, sidestep_radius
        self.goal: np.ndarray | None = None
        self.mode = "idle"                       # idle | go | wait
        self.plan: Plan | None = None            # the last plan this navigator issued
        self.arrived = False
        self._wait_since: float | None = None
        self._last_t: float | None = None
        self._hold_pending = False
        self._tmp: dict[tuple, PathPlanner] = {}
        self.stats = dict(yields=0, wait_s=0.0, sidesteps=0, pushes=0, reroutes=0, relocations=0)

    # ------------------------------------------------------------------ public API
    def set_goal(self, goal) -> None:
        self._hold_pending = goal is None and self.mode in ("go", "wait")
        self.goal = None if goal is None else np.asarray(goal, dtype=float)
        self.arrived, self.plan, self._wait_since = False, None, None
        self.mode = "idle" if goal is None else "go"

    def update(self, t: float, pos, vel, plan_finished: bool, nbrs: list[Neighbor],
               idle: bool = False) -> Plan | None:
        """Return a new Plan to command, or None to keep the current one.

        idle=True means "I have no work": I yield to every neighbour.
        """
        pos, vel = np.asarray(pos, dtype=float), np.asarray(vel, dtype=float)
        if self._last_t is not None and self.mode == "wait":
            self.stats["wait_s"] += t - self._last_t
        self._last_t = t
        ts = t + self.step * np.arange(int(self.horizon / self.step) + 1)

        moving_higher, static_tr, static_pts, near_lower = [], [], [], []
        k_near = int(round(self.near_horizon / self.step)) + 1
        for n in nbrs:
            tr = n.traj(ts)
            if n.is_static(tr):
                static_tr.append((tr, self.static_dist))
                static_pts.append(tr[-1])
            elif n.higher or idle:
                moving_higher.append((tr, self.conflict_dist))
            else:
                # lower priority: they should yield to me, but I must never be blind to a drone
                # right ahead (e.g. one that is still braking), so keep a short-range check.
                near_lower.append((tr[:k_near], self.static_dist))
        relevant = moving_higher + static_tr + near_lower

        if self.goal is None:
            return self._update_idle(t, pos, vel, ts, moving_higher, relevant)

        if self.mode == "go":
            if self.plan is not None and plan_finished and np.linalg.norm(pos - self.goal) < 0.25:
                self.arrived = True
                return None
            if self.plan is None:
                new = self._try_go(t, pos, ts, relevant, static_pts)
                return self._issue(new) if new is not None else self._start_wait(t, pos, vel, ts, relevant)
            if not self._hits(plan_positions(self.plan, ts), relevant):
                return None
            new = self._try_go(t, pos, ts, relevant, static_pts)
            if new is not None:
                self.stats["reroutes"] += 1
                return self._issue(new)
            return self._start_wait(t, pos, vel, ts, relevant)

        # mode == "wait"
        new = self._try_go(t, pos, ts, relevant, static_pts)
        if new is not None:
            self.mode, self._wait_since = "go", None
            return self._issue(new)
        if self._wait_since is not None and t - self._wait_since >= self.patience:
            push = self._push_plan(t, pos, ts, moving_higher, static_pts)
            if push is not None:
                self.stats["pushes"] += 1
                self._wait_since = t
                return self._issue(push)
            reloc = self._random_relocation(t, pos, ts, relevant)      # break symmetric deadlocks
            if reloc is not None:
                self.stats["relocations"] += 1
                self._wait_since = t
                return self._issue(reloc)
        cur = plan_positions(self.plan, ts) if self.plan is not None else np.repeat(pos[None], len(ts), 0)
        if self._hits(cur, moving_higher):
            side = self._sidestep(t, pos, ts, relevant)
            if side is not None:
                self.stats["sidesteps"] += 1
                return self._issue(side)
        return None

    # ------------------------------------------------------------------ internals
    def _issue(self, plan: Plan) -> Plan:
        self.plan = plan
        return plan

    @staticmethod
    def _hits(mine: np.ndarray, others) -> bool:
        """Would my trajectory come within the keep-away distance of any other, while approaching it?
        others: list of (trajectory, keep-away distance)."""
        for tr, dist in others:
            k = min(len(mine), len(tr))
            d = np.linalg.norm(mine[:k] - tr[:k], axis=1)
            if d.min() < dist and d.min() < d[0] - 1e-3:
                return True
        return False

    def _hold_plan(self, t: float, pos: np.ndarray, vel: np.ndarray) -> Plan:
        stop = pos + vel * np.linalg.norm(vel) / (2.0 * self.cfg.a_max)      # braking distance
        return Plan([t, t + 0.5], [pos, stop])

    def _update_idle(self, t, pos, vel, ts, moving_higher, relevant) -> Plan | None:
        self.mode = "idle"
        if self._hold_pending:
            self._hold_pending = False
            return self._issue(self._hold_plan(t, pos, vel))
        mine = plan_positions(self.plan, ts) if self.plan is not None else np.repeat(pos[None], len(ts), 0)
        if self._hits(mine, moving_higher):
            side = self._sidestep(t, pos, ts, relevant)
            if side is not None:
                self.stats["sidesteps"] += 1
                return self._issue(side)
        return None

    def _start_wait(self, t, pos, vel, ts, relevant) -> Plan:
        self.mode, self._wait_since = "wait", t
        self.stats["yields"] += 1
        hold = self._hold_plan(t, pos, vel)
        if self._hits(plan_positions(hold, ts), relevant):
            side = self._sidestep(t, pos, ts, relevant)
            if side is not None:
                self.stats["sidesteps"] += 1
                return self._issue(side)
        return self._issue(hold)

    def _try_go(self, t, pos, ts, relevant, static_pts) -> Plan | None:
        route = self._route(pos, self.goal, static_pts)
        if route is None:
            return None
        plan = Plan.from_polyline(route, t, self.cfg.plan_speed)
        return None if self._hits(plan_positions(plan, ts), relevant) else plan

    def _route(self, a, b, static_pts) -> list[np.ndarray] | None:
        base = self.planner.path(a, b)
        static_pts = [p for p in static_pts if np.linalg.norm(p - a) > self.static_dist + 0.02]
        if base is None or not static_pts:
            return base
        line = LineString(np.asarray(base))
        if all(line.distance(Point(p)) >= self.static_dist for p in static_pts):
            return base
        key = tuple(sorted((round(float(p[0]), 1), round(float(p[1]), 1)) for p in static_pts))
        tp = self._tmp.get(key)
        if tp is None:
            # 8-sided polygon circumscribed about the circle, so its edges sit at >= static_dist
            rad = self.static_dist / np.cos(np.pi / 8)
            discs = unary_union([Point(p).buffer(rad, quad_segs=2) for p in static_pts])
            region = self.planner.allowed.difference(discs)
            if region.is_empty:
                return None
            tp = PathPlanner(region)
            if len(self._tmp) > 48:
                self._tmp.pop(next(iter(self._tmp)))
            self._tmp[key] = tp
        return tp.path(a, b)

    def _sidestep(self, t, pos, ts, relevant) -> Plan | None:
        d = np.linalg.norm(self.cands - pos, axis=1)
        order = np.argsort(d)
        order = [k for k in order if 0.5 < d[k] < self.sidestep_radius][: self.max_candidates]
        for k in order:
            route = self.planner.path(pos, self.cands[k])
            if route is None:
                continue
            plan = Plan.from_polyline(route, t, self.cfg.plan_speed)
            if not self._hits(plan_positions(plan, ts), relevant):
                return plan
        return None

    def _random_relocation(self, t, pos, ts, relevant) -> Plan | None:
        """Move to a random clear viewpoint nearby: random choice breaks symmetric wait cycles."""
        d = np.linalg.norm(self.cands - pos, axis=1)
        ids = np.flatnonzero((d > 0.8) & (d < self.sidestep_radius))
        self.rng.shuffle(ids)
        for k in ids[: self.max_candidates]:
            route = self.planner.path(pos, self.cands[k])
            if route is None:
                continue
            plan = Plan.from_polyline(route, t, self.cfg.plan_speed)
            if not self._hits(plan_positions(plan, ts), relevant):
                return plan
        return None

    def _push_plan(self, t, pos, ts, moving_higher, static_pts) -> Plan | None:
        """Head for the goal ignoring stationary neighbours, but stop short of them."""
        route = self.planner.path(pos, self.goal)
        if route is None:
            return None
        plan = Plan.from_polyline(route, t, self.cfg.plan_speed)
        fine = np.arange(plan.t_start, plan.t_end + 0.05, 0.05)
        pts = plan_positions(plan, fine)
        stop_at = len(fine) - 1
        for p in static_pts:
            near = np.flatnonzero(np.linalg.norm(pts - p, axis=1) < self.static_dist + 0.05)
            if near.size:
                stop_at = min(stop_at, max(0, int(near[0]) - 1))
        if stop_at < 2:
            return None
        cut = Plan(fine[: stop_at + 1], pts[: stop_at + 1])
        return None if self._hits(plan_positions(cut, ts), moving_higher) else cut
