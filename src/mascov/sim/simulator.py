"""Simulation loop.

Each tick (dt):
  1. every drone senses (disc + line of sight) and updates its OWN belief
  2. on decision ticks the model is called with per-agent Observations -> Plans/Messages
  3. the shared tracker turns each drone's plan into an acceleration; dynamics advance
  4. metrics: true coverage, collisions, tracking error

The simulator, not the model, owns ground truth. Agents only learn about each other's coverage
through messages the model sends, delivered to agents within communication range.
"""
from __future__ import annotations

import math
import time as _time

import numpy as np
import shapely

from mascov.common.config import SimConfig
from mascov.common.model import CentralModel
from mascov.common.types import Command, Message, Observation, Plan
from mascov.metrics.logger import MetricsLogger, RunResult
from mascov.safety.bvc import bvc_filter
from mascov.sim.dynamics import step_double_integrator
from mascov.sim.tracker import PlanTracker
from mascov.world.belief import AgentBelief
from mascov.world.world import World


class Simulator:
    def __init__(self, world: World, model: CentralModel, sim_cfg: SimConfig | None = None):
        self.world = world
        self.model = model
        self.sc = sim_cfg or SimConfig()

    def run(self, record_paths: bool = False) -> RunResult:
        w, c, sc = self.world, self.world.cfg, self.sc
        N, M, dt = w.n_agents, w.n_targets, c.dt
        coverable = w.coverable
        sensor = w.sensor
        tracker = PlanTracker(sc.tracker_omega, c.a_max)

        pos = w.starts.copy()
        vel = np.zeros((N, 2))
        beliefs = [AgentBelief(M) for _ in range(N)]
        plans: list[Plan | None] = [None] * N
        hold = [pos[i].copy() for i in range(N)]
        inbox: list[list[Message]] = [[] for _ in range(N)]
        newly = [[] for _ in range(N)]
        new_obs = [[] for _ in range(N)]
        last_sense = [None] * N
        believed_done: list[float | None] = [None] * N

        true_cov = np.zeros(M, dtype=bool)
        found: dict[int, float] = {}
        log = MetricsLogger()
        self.paths = [[p.copy()] for p in pos] if record_paths else None

        max_err = np.zeros(N)
        path_len = np.zeros(N)
        min_pair, min_clear = math.inf, math.inf
        events, in_violation = 0, False
        model_time, n_dec, n_msgs = 0.0, 0, 0
        coverage_time = None
        n_filtered = 0

        n_ticks = int(round(sc.max_time / dt))
        dec_every = max(1, int(round(sc.decision_period / dt)))
        self.model.reset(w.view(known_map=sc.known_map), list(range(N)))
        t = 0.0

        for tick in range(n_ticks + 1):
            t = tick * dt

            # 1. sense -------------------------------------------------------------------
            for i in range(N):
                if last_sense[i] is None or np.linalg.norm(pos[i] - last_sense[i]) > 1e-3:
                    ids = sensor.visible_target_ids(pos[i])
                    if ids.size:
                        fresh = ids[~beliefs[i].covered[ids]]
                        if fresh.size:
                            newly[i].extend(fresh.tolist())
                        beliefs[i].mark(ids)
                        true_cov[ids] = True
                    if w.obstacles:
                        seen = sensor.visible_obstacle_ids(pos[i])
                        fresh_o = seen - beliefs[i].known_obstacles
                        if fresh_o:
                            new_obs[i].extend(sorted(fresh_o))
                            beliefs[i].known_obstacles |= fresh_o
                    if len(w.victims):
                        for k in np.flatnonzero(sensor.visible_points_mask(pos[i], w.victims)):
                            found.setdefault(int(k), t)
                    last_sense[i] = pos[i].copy()

            n_cov = int(true_cov[coverable].sum())
            frac = n_cov / max(1, int(coverable.sum()))
            if coverage_time is None and n_cov == int(coverable.sum()):
                coverage_time = t

            # 4a. metrics (before moving) -----------------------------------------------
            if N > 1:
                diff = pos[:, None, :] - pos[None, :, :]
                dist = np.linalg.norm(diff, axis=2)
                dist[np.diag_indices(N)] = math.inf
                pair = float(dist.min())
            else:
                pair = math.inf
            walls = np.minimum.reduce([pos[:, 0], c.width - pos[:, 0], pos[:, 1], c.height - pos[:, 1]])
            obs_d = (shapely.distance(w.obstacle_union, shapely.points(pos))
                     if w.obstacle_union is not None else np.full(N, math.inf))
            clear = float(np.minimum(walls, obs_d).min())
            min_pair, min_clear = min(min_pair, pair), min(min_clear, clear)
            violating = pair < c.d_safe - 1e-9 or clear < c.drone_radius - 1e-9
            if violating and not in_violation:
                events += 1
            in_violation = violating
            stop = ((coverage_time is not None and sc.stop_when_covered)
                    or all(b is not None for b in believed_done) or tick == n_ticks)
            if tick % sc.log_every == 0 or stop:
                row = dict(t=round(t, 3), coverage=round(frac, 5), min_pair=None if not math.isfinite(pair) else round(pair, 3),
                           clearance=round(clear, 3))
                for i in range(N):
                    row[f"believed_{i}"] = round(beliefs[i].fraction(coverable), 5)
                log.log(**row)

            if stop:
                break

            # 2. decide ------------------------------------------------------------------
            if tick % dec_every == 0:
                obs = {}
                for i in range(N):
                    near = [j for j in range(N) if j != i and np.linalg.norm(pos[j] - pos[i]) <= c.r_comm]
                    nb = {j: pos[j].copy() for j in near}
                    nb_v = {j: vel[j].copy() for j in near}
                    p = plans[i]
                    obs[i] = Observation(
                        t=t, agent_id=i, pos=pos[i].copy(), vel=vel[i].copy(), belief=beliefs[i],
                        newly_covered=np.asarray(newly[i], dtype=int), new_obstacles=list(new_obs[i]),
                        neighbors=nb, neighbor_vels=nb_v, inbox=inbox[i],
                        plan_finished=tracker.finished(t, pos[i], vel[i], p),
                        plan_time_left=0.0 if p is None else max(0.0, p.t_end - t))
                    inbox[i] = []
                    newly[i], new_obs[i] = [], []
                t0 = _time.perf_counter()
                cmds = self.model.step(obs)
                model_time += _time.perf_counter() - t0
                n_dec += 1
                for i, cmd in (cmds or {}).items():
                    if cmd is None:
                        continue
                    if cmd.plan is not None:
                        plans[i] = cmd.plan
                    if cmd.done and believed_done[i] is None:
                        believed_done[i] = t        # the agent keeps being stepped (it may still dodge others)
                    for m in cmd.messages:
                        n_msgs += 1
                        targets = ([m.to] if m.to is not None else
                                   [j for j in range(N) if j != i])
                        for j in targets:
                            if j != i and np.linalg.norm(pos[j] - pos[i]) <= c.r_comm:
                                inbox[j].append(m)

            # 3. control (tracker -> optional safety filter, using a synchronous snapshot) + dynamics
            acc = np.zeros((N, 2))
            for i in range(N):
                p = plans[i]
                acc[i], err = tracker.accel(t, pos[i], vel[i], p, hold[i])
                if p is not None and p.t_start <= t <= p.t_end:
                    max_err[i] = max(max_err[i], err)
            if sc.safety == "bvc" and N > 1:
                snap_p, snap_v, safe = pos.copy(), vel.copy(), np.empty_like(acc)
                for i in range(N):
                    near = [j for j in range(N) if j != i and np.linalg.norm(snap_p[j] - snap_p[i]) <= c.r_comm]
                    safe[i], changed = bvc_filter(snap_p[i], snap_v[i], acc[i], snap_p[near], snap_v[near], dt=dt,
                                                  a_max=c.a_max, d_safe=c.d_safe, brake_frac=sc.brake_frac)
                    n_filtered += int(changed)
                acc = safe
            for i in range(N):
                new_p, new_v = step_double_integrator(pos[i], vel[i], acc[i], dt, c.v_max, c.a_max)
                path_len[i] += float(np.linalg.norm(new_p - pos[i]))
                pos[i], vel[i] = new_p, new_v
                if self.paths is not None:
                    self.paths[i].append(new_p.copy())

        return RunResult(
            model=getattr(self.model, "name", type(self.model).__name__), world=w.name, n_agents=N,
            completed=coverage_time is not None, coverage_time=coverage_time, final_time=t,
            believed_done=believed_done, min_pair_dist=min_pair, min_clearance=min_clear,
            collision_events=events, max_tracking_error=max_err.tolist(), path_length=path_len.tolist(),
            victims_found=found, model_time=model_time, n_decisions=n_dec, n_messages=n_msgs, log=log,
            extra={**self.model.stats(), **({"safety_interventions": n_filtered} if sc.safety != "none" else {})})
