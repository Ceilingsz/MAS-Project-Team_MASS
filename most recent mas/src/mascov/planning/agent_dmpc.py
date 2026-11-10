"""Greedy goals + D-MPC trajectories: one drone's policy (known map).

Layer 1 (where to go): the same greedy rule as the uncoordinated baseline: the viewpoint maximising
(new targets in this drone's belief) / (travel time + delta). Subclasses can replace ``select_goal`` (CBBA later).

Layer 2 (how to get there): every decision (default every control step, dt) the drone solves its own D-MPC
problem (``mascov.planning.dmpc``): follow a reference sampled along the shortest path to the goal, stay
clear of walls and obstacles, and, with ``coupling=True``, stay ``d_min`` away from each neighbour's
predicted path. The predicted path (H+1 points) is broadcast to drones within r_comm and arrives at the next
decision. Neighbours that are sensed (within r_neighbor) but whose prediction has not arrived are predicted at
constant velocity. With ``coupling=False`` drones ignore each other and send nothing (the ablation).

The navigator's yield/sidestep heuristic is NOT used: D-MPC replaces it.

Deadlocks (drones pushing against each other, e.g. a symmetric crossing): a drone that has moved less than
0.15 m per second for ``stuck_time`` seconds sidesteps for ``detour_time`` seconds (to the right of its way,
with a small random tilt to break symmetry, so facing drones pass each other), then resumes. After
``max_nudges`` sidesteps for the same goal it gives that goal up for ``ban_time`` seconds.
"""
from __future__ import annotations

import math

import numpy as np

from mascov.common.model import AgentPolicy
from mascov.common.types import Command, Message, Plan
from mascov.planning.dmpc import (DMPCParams, obstacle_halfspaces, project_on_polyline, sample_polyline,
                                  solve_dmpc)
from mascov.planning.greedy import choose_viewpoint, remaining_gain

KIND = "dmpc_pred"


class DMPCAgentPolicy(AgentPolicy):
    decision_period = 0.1

    def __init__(self, coupling: bool = True, horizon: int = 10, margin: float = 0.1, share: bool = False,
                 delta: float = 1.0, seed: int = 0, arrive_tol: float = 0.1, replan_every: float = 1.0,
                 stuck_time: float = 2.0, detour_time: float = 1.5, max_nudges: int = 3, ban_time: float = 10.0,
                 **dmpc_kw):
        self.coupling, self.H, self.margin, self.share = coupling, int(horizon), margin, share
        self.delta, self.seed, self.arrive_tol, self.replan_every = delta, seed, arrive_tol, replan_every
        self.stuck_time, self.detour_time, self.max_nudges, self.ban_time = stuck_time, detour_time, max_nudges, ban_time
        self.dmpc_kw = dmpc_kw

    # ------------------------------------------------------------------ setup
    def reset(self, world, agent_id: int, n_agents: int) -> None:
        self.w, self.id, self.cfg = world, agent_id, world.cfg
        c = world.cfg
        self.rng = np.random.default_rng(self.seed * 1000 + agent_id)
        self.prm = DMPCParams(horizon=self.H, dt=c.dt, v_cap=0.95 * c.v_max, clearance=c.clearance,
                              d_min=c.d_safe + self.margin, width=c.width, height=c.height, **self.dmpc_kw)
        self._setup_map(world)
        self.visited: set[int] = set()
        self.banned: dict[int, float] = {}            # viewpoint -> time the ban ends
        self.goal_vp: int | None = None
        self.goal: np.ndarray | None = None
        self.path: np.ndarray | None = None
        self.path_t = -math.inf
        self.s = 0.0
        self.hold: np.ndarray | None = None
        self.done = False
        self.pred: np.ndarray | None = None           # last solution, (H+1, 2), times t_pred + k dt
        self.t_pred = 0.0
        self.heard: dict[int, tuple[float, np.ndarray]] = {}   # neighbour -> (t sent, (H+1, 2) prediction)
        self.check_t, self.check_p = 0.0, None
        self.stuck_since: float | None = None
        self.nudges, self.detour_until, self.detour_dir = 0, -math.inf, np.zeros(2)
        self.st = dict(dmpc_solves=0, dmpc_failures=0, dmpc_slack_steps=0, dmpc_deadlock_nudges=0,
                       dmpc_abandoned=0, dmpc_max_slack=0.0, belief_merged=0)

    # ------------------------------------------------------------------ map hooks (overridden for unknown maps)
    def _setup_map(self, world) -> None:
        """Known map: goals are the viewpoints, the planner and obstacles are the true ones."""
        self.vp = world.viewpoints
        self.goal_pos = self.vp.positions
        self.planner = world.planner
        self.obs_hs = obstacle_halfspaces(world.obstacles)

    def _sync_map(self, obs) -> None:
        """Called every decision after merging messages: take in newly known obstacles (nothing to do here)."""

    def _goal_useful(self, obs, v: int) -> bool:
        """False once the goal would no longer add anything (its targets were seen en route)."""
        return remaining_gain(self.vp, v, obs.belief.covered) > 0

    # ------------------------------------------------------------------ layer 1: goals
    def _static_blocked(self, obs) -> set[int]:
        """Viewpoints too close to a neighbour that is standing still."""
        pts = [p for j, p in obs.neighbors.items() if np.linalg.norm(obs.neighbor_vels.get(j, np.zeros(2))) < 0.1]
        if not pts:
            return set()
        d = np.linalg.norm(self.goal_pos[:, None, :] - np.asarray(pts)[None], axis=2).min(axis=1)
        return set(np.flatnonzero(d < self.prm.d_min + 0.1).tolist())

    def select_goal(self, obs) -> int | None:
        covered = obs.belief.covered
        travel = self.w.dist_to_viewpoints(obs.pos) / self.cfg.v_max
        excl = self.visited | {v for v, t_end in self.banned.items() if t_end > obs.t}
        if self.coupling:
            excl |= self._static_blocked(obs)
        return choose_viewpoint(self.vp, travel, covered, excl, self.delta, self.rng)

    def _set_goal(self, v: int | None, obs) -> None:
        self.goal_vp = v
        self.goal = None if v is None else self.goal_pos[v].copy()
        self.path, self.path_t, self.s = None, -math.inf, 0.0
        self.stuck_since, self.check_t, self.check_p = None, obs.t, obs.pos.copy()
        self.nudges, self.detour_until = 0, -math.inf

    def _update_goal(self, obs) -> None:
        if self.goal_vp is not None:
            if np.linalg.norm(obs.pos - self.goal) <= self.arrive_tol:
                self.visited.add(self.goal_vp)
                self._set_goal(None, obs)
            elif not self._goal_useful(obs, self.goal_vp):
                self._set_goal(None, obs)                       # seen en route (or, unknown map, now inside an obstacle)
        if self.goal_vp is None and not self.done:
            v = self.select_goal(obs)
            if v is None:
                self.done = True
                self.hold = obs.pos.copy()
            else:
                self._set_goal(v, obs)

    # ------------------------------------------------------------------ reference
    def _reference(self, obs) -> np.ndarray:
        H, dt, v = self.H, self.cfg.dt, self.cfg.plan_speed
        if self.goal is None:
            pt = self.hold if self.hold is not None else obs.pos
            return np.repeat(np.asarray(pt, float)[None], H, axis=0)
        far = self.path is not None and (np.linalg.norm(sample_polyline(self.path, np.array([self.s]))[0] - obs.pos) > 0.5)
        if self.path is None or obs.t - self.path_t >= self.replan_every or far:
            path = self.planner.path(obs.pos, self.goal)
            if path is None:                                  # unreachable: drop it
                self.visited.add(self.goal_vp)
                self._set_goal(None, obs)
                self._update_goal(obs)
                return self._reference(obs)
            self.path, self.path_t, self.s = np.asarray(path, float), obs.t, 0.0
        self.s = project_on_polyline(self.path, obs.pos, self.s - 0.2, self.s + 1.5)
        ref = sample_polyline(self.path, self.s + v * dt * np.arange(1, H + 1))
        if obs.t < self.detour_until:
            # deadlock sidestep: move sideways (to the right of the way to the goal) instead of towards the goal
            ref = obs.pos[None] + self.detour_dir[None] * v * dt * np.arange(1, H + 1)[:, None]
        elif self.detour_until > -math.inf:
            self.detour_until = -math.inf
            self.path = None                                  # back from a sidestep: fresh shortest path next step
        return ref

    def _start_detour(self, obs) -> None:
        d = self.goal - obs.pos
        if self.path is not None:
            ahead = sample_polyline(self.path, np.array([self.s + 0.5]))[0] - obs.pos
            if np.linalg.norm(ahead) > 1e-6:
                d = ahead
        nd = float(np.linalg.norm(d))
        right = np.array([d[1], -d[0]]) / nd if nd > 1e-6 else np.array([1.0, 0.0])
        a = self.rng.uniform(-0.4, 0.4)                       # break exact symmetry between drones
        c, s_ = math.cos(a), math.sin(a)
        self.detour_dir = np.array([c * right[0] - s_ * right[1], s_ * right[0] + c * right[1]])
        self.detour_until = obs.t + self.detour_time
        self.nudges += 1
        self.st["dmpc_deadlock_nudges"] += 1

    def _progress_check(self, obs) -> None:
        """Detect being stuck: moved less than 0.15 m over the last second while having a goal."""
        if self.goal is None:
            self.stuck_since = None
            return
        if obs.t - self.check_t >= 1.0:
            moved = float(np.linalg.norm(obs.pos - self.check_p)) if self.check_p is not None else math.inf
            if moved < 0.15:
                if self.stuck_since is None:
                    self.stuck_since = self.check_t
            else:
                self.stuck_since = None
            self.check_t, self.check_p = obs.t, obs.pos.copy()
        if self.stuck_since is not None and obs.t >= self.detour_until                 and obs.t - self.stuck_since >= self.stuck_time:
            if self.nudges >= self.max_nudges:               # sidestepping did not help: give the goal up for a while
                self.banned[self.goal_vp] = obs.t + self.ban_time
                self.st["dmpc_abandoned"] += 1
                self._set_goal(None, obs)
                self._update_goal(obs)
            else:
                self._start_detour(obs)
                self.stuck_since = None

    # ------------------------------------------------------------------ neighbours
    def _at_times(self, t_sent: float, pts: np.ndarray, times: np.ndarray) -> np.ndarray:
        idx = (times - t_sent) / self.cfg.dt
        k = np.arange(len(pts))
        return np.c_[np.interp(idx, k, pts[:, 0]), np.interp(idx, k, pts[:, 1])]

    def _neighbour_predictions(self, obs, times: np.ndarray) -> list[np.ndarray]:
        out = []
        horizon_s = self.H * self.cfg.dt
        for j, p in obs.neighbors.items():
            h = self.heard.get(j)
            if h is not None and obs.t - h[0] <= horizon_s / 2:
                q = self._at_times(h[0], h[1], times)
                q = q + (p - self._at_times(h[0], h[1], np.array([obs.t]))[0])[None]   # re-anchor at the sensed position
            else:
                v = obs.neighbor_vels.get(j, np.zeros(2))
                q = p[None] + v[None] * (times - obs.t)[:, None]
            out.append(q)
        return out

    # ------------------------------------------------------------------ step
    def step(self, obs) -> Command:
        for m in obs.inbox:
            if m.kind == KIND and m.sender != self.id:
                self.heard[m.sender] = (m.payload["t"], m.payload["pts"])
                if self.share and m.payload.get("belief") is not None:
                    self.st["belief_merged"] += obs.belief.merge(m.payload["belief"])
        self._sync_map(obs)

        self._update_goal(obs)
        self._progress_check(obs)
        ref = self._reference(obs)

        H, dt = self.H, self.cfg.dt
        times = obs.t + dt * np.arange(1, H + 1)
        prev = None
        if self.pred is not None:
            prev = self._at_times(self.t_pred, self.pred, times)
            prev = prev + (obs.pos - self._at_times(self.t_pred, self.pred, np.array([obs.t]))[0])[None]
        nbrs = self._neighbour_predictions(obs, times) if self.coupling else []

        res = solve_dmpc(obs.pos, obs.vel, ref, prev, self.obs_hs, nbrs, self.prm)
        self.st["dmpc_solves"] += 1
        if res.info.get("failed"):
            self.st["dmpc_failures"] += 1
        elif res.slack > 1e-3:
            self.st["dmpc_slack_steps"] += 1
            self.st["dmpc_max_slack"] = max(self.st["dmpc_max_slack"], round(res.slack, 4))
        self.pred, self.t_pred = res.points, obs.t

        # Plan.ref holds still AT its first time stamp, and this plan is replaced at the next control step, so
        # start it a hair earlier: the tracker then applies the first predicted velocity right away.
        tt = obs.t + dt * np.arange(H + 1)
        tt[0] -= 1e-6
        plan = Plan(tt, res.points)
        msgs = []
        if self.coupling or self.share:
            payload = dict(t=obs.t, pts=res.points.copy())
            if self.share:
                payload["belief"] = obs.belief.copy()
            msgs.append(Message(self.id, KIND, payload))
        return Command(plan=plan, messages=msgs, done=self.done)

    def stats(self) -> dict:
        return dict(self.st)
