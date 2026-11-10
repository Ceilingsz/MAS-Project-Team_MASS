"""Simulation loop.

Each tick (dt):
  1. every drone senses (disc + line of sight) and updates its OWN belief
  2. on decision ticks the model is called with per-agent Observations -> Plans/Messages
  3. the shared tracker turns each drone's plan into a velocity command; dynamics advance
  4. metrics: true coverage, collisions, tracking error

The simulator, not the model, owns ground truth. Agents only learn about each other's coverage
through messages the model sends, delivered to agents within communication range.

Two interfaces (``SimConfig.known_map``):
  * known map   : models get a WorldView and a target-indexed AgentBelief.
  * unknown map : models get an UnknownWorldView and a lattice-indexed MapBelief. Seeing a free cell marks it
                  FREE; seeing any part of an obstacle reveals its WHOLE polygon (marking the cells inside it).
                  Revealing only the visible boundary would be the harder, more realistic variant (not done).
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
from mascov.sim.dynamics import step_single_integrator
from mascov.sim.tracker import PlanTracker
from mascov.world.belief import FREE, AgentBelief, MapBelief
from mascov.world.world import World


class Simulator:
    def __init__(self, world: World, model: CentralModel, sim_cfg: SimConfig | None = None):
        self.world = world
        self.model = model
        self.sc = sim_cfg or SimConfig()
        want = "known" if self.sc.known_map else "unknown"
        have = getattr(model, "map_interface", "known")
        if have != want:
            hint = (" Run it with SimConfig(known_map=False, reveal_obstacles_at_start=True) to give it the map."
                    if have == "unknown" else "")
            raise ValueError(f"model '{getattr(model, 'name', type(model).__name__)}' is written for the {have}-map "
                             f"interface but SimConfig.known_map={self.sc.known_map}.{hint}")

    def run(self, record_paths: bool = False) -> RunResult:
        w, c, sc = self.world, self.world.cfg, self.sc
        N, M, dt = w.n_agents, w.n_targets, c.dt
        coverable = w.coverable
        sensor = w.sensor
        tracker = PlanTracker(sc.tracker_gain, c.v_max)

        pos = w.starts.copy()
        vel = np.zeros((N, 2))
        known = sc.known_map
        if known:
            beliefs = [AgentBelief(M) for _ in range(N)]
            view = w.view(known_map=not sc.hide_obstacles)
            cov_cells = None
        else:
            view = w.unknown_view()
            beliefs = [MapBelief(view.lattice) for _ in range(N)]
            if sc.reveal_obstacles_at_start:
                for b in beliefs:
                    for k, poly in enumerate(w.obstacles):
                        b.add_obstacle(k, poly)
            cov_cells = w.target_cell[coverable]        # lattice cells whose coverage defines "done"

        self.beliefs = beliefs                          # kept so callers can inspect / plot what each agent believed

        def believed_fraction(i: int) -> float:
            return beliefs[i].fraction(coverable) if known else beliefs[i].fraction_free(cov_cells)

        plans: list[Plan | None] = [None] * N
        hold = [pos[i].copy() for i in range(N)]
        inbox: list[list[Message]] = [[] for _ in range(N)]
        newly = [[] for _ in range(N)]
        new_obs = [[] for _ in range(N)]
        last_sense = [None] * N
        believed_done: list[float | None] = [None] * N

        true_cov = np.zeros(M, dtype=bool)
        in_view = np.zeros((N, M), dtype=bool)          # which targets each drone currently has in view
        visits = np.zeros(M, dtype=int)                 # per target: separate stretches of being in view, all drones
        n_coverable = max(1, int(coverable.sum()))
        milestones: dict[float, float] = {}             # coverage fraction -> first time true coverage reached it
        found: dict[int, float] = {}
        log = MetricsLogger()
        self.paths = [[p.copy()] for p in pos] if record_paths else None

        def team_believed() -> float:
            """Fraction of coverable targets that at least one drone believes covered (the team's pooled belief)."""
            if known:
                return float(np.logical_or.reduce([b.covered for b in beliefs])[coverable].mean()) if coverable.any() else 1.0
            pooled = np.logical_or.reduce([b.state[cov_cells] == FREE for b in beliefs])
            return float(pooled.mean()) if pooled.size else 1.0

        max_err = np.zeros(N)
        path_len = np.zeros(N)
        min_pair, min_clear = math.inf, math.inf
        events, pair_events, obstacle_events = 0, 0, 0
        in_violation, in_pair, in_obstacle = False, False, False
        model_time, n_dec, n_msgs, n_deliveries = 0.0, 0, 0, 0
        coverage_time = None
        n_filtered = 0
        frac = 0.0

        n_ticks = int(round(sc.max_time / dt))
        dec_period = getattr(self.model, "decision_period", None) or sc.decision_period
        dec_every = max(1, int(round(dec_period / dt)))
        self.model.reset(view, list(range(N)))
        t = 0.0

        for tick in range(n_ticks + 1):
            t = tick * dt

            # 1. sense -------------------------------------------------------------------
            for i in range(N):
                if last_sense[i] is None or np.linalg.norm(pos[i] - last_sense[i]) > 1e-3:
                    ids = sensor.visible_target_ids(pos[i])
                    now_in_view = np.zeros(M, dtype=bool)
                    now_in_view[ids] = True
                    visits += now_in_view & ~in_view[i]          # a visit starts when a target comes into view
                    in_view[i] = now_in_view
                    if ids.size:
                        if known:
                            fresh = ids[~beliefs[i].covered[ids]]
                            if fresh.size:
                                newly[i].extend(fresh.tolist())
                            beliefs[i].mark(ids)
                        else:
                            cells = w.target_cell[ids]
                            fresh = cells[beliefs[i].state[cells] != FREE]
                            if fresh.size:
                                newly[i].extend(fresh.tolist())
                            beliefs[i].mark_free(cells)
                        true_cov[ids] = True
                    if w.obstacles:
                        seen = sensor.visible_obstacle_ids(pos[i])
                        if known:
                            fresh_o = seen - beliefs[i].known_obstacles
                            if fresh_o:
                                new_obs[i].extend(sorted(fresh_o))
                                beliefs[i].known_obstacles |= fresh_o
                        else:
                            fresh_o = sorted(k for k in seen if k not in beliefs[i].obstacles)
                            for k in fresh_o:
                                beliefs[i].add_obstacle(k, w.obstacles[k])
                            new_obs[i].extend(fresh_o)
                    if len(w.victims):
                        for k in np.flatnonzero(sensor.visible_points_mask(pos[i], w.victims)):
                            found.setdefault(int(k), t)
                    last_sense[i] = pos[i].copy()

            n_cov = int(true_cov[coverable].sum())
            frac = n_cov / n_coverable
            for f in sc.milestones:
                if f not in milestones and n_cov >= f * n_coverable - 1e-9:
                    milestones[f] = t
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
            pair_bad = pair < c.d_safe - 1e-9
            obstacle_bad = clear < c.drone_radius - 1e-9
            violating = pair_bad or obstacle_bad
            if violating and not in_violation:
                events += 1
            if pair_bad and not in_pair:
                pair_events += 1                                 # drone-drone episodes
            if obstacle_bad and not in_obstacle:
                obstacle_events += 1                             # drone-obstacle / wall episodes
            in_violation, in_pair, in_obstacle = violating, pair_bad, obstacle_bad
            stop = ((coverage_time is not None and sc.stop_when_covered)
                    or all(b is not None for b in believed_done) or tick == n_ticks)
            if tick % sc.log_every == 0 or stop:
                row = dict(t=round(t, 3), coverage=round(frac, 5), min_pair=None if not math.isfinite(pair) else round(pair, 3),
                           clearance=round(clear, 3), team_believed=round(team_believed(), 5))
                for i in range(N):
                    row[f"believed_{i}"] = round(believed_fraction(i), 5)
                log.log(**row)

            if stop:
                break

            # 2. decide ------------------------------------------------------------------
            if tick % dec_every == 0:
                obs = {}
                for i in range(N):
                    gaps = {j: float(np.linalg.norm(pos[j] - pos[i])) for j in range(N) if j != i}
                    near = [j for j, g in gaps.items() if g <= c.neighbor_radius]      # sensed (position + velocity)
                    comm = [j for j, g in gaps.items() if g <= c.r_comm]               # reachable by messages
                    nb = {j: pos[j].copy() for j in near}
                    nb_v = {j: vel[j].copy() for j in near}
                    p = plans[i]
                    obs[i] = Observation(
                        t=t, agent_id=i, pos=pos[i].copy(), vel=vel[i].copy(), belief=beliefs[i],
                        newly_covered=np.asarray(newly[i], dtype=int), new_obstacles=list(new_obs[i]),
                        neighbors=nb, neighbor_vels=nb_v, comm_neighbors=comm, inbox=inbox[i],
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
                                n_deliveries += 1

            # 3. control (tracker -> optional safety filter, using a synchronous snapshot) + dynamics
            cmd = np.zeros((N, 2))
            for i in range(N):
                p = plans[i]
                cmd[i], err = tracker.velocity(t, pos[i], p, hold[i])
                if p is not None and p.t_start <= t <= p.t_end:
                    max_err[i] = max(max_err[i], err)
            if sc.safety == "bvc" and N > 1:
                snap_p, safe = pos.copy(), np.empty_like(cmd)
                for i in range(N):
                    near = [j for j in range(N) if j != i and np.linalg.norm(snap_p[j] - snap_p[i]) <= c.neighbor_radius]
                    safe[i], changed = bvc_filter(snap_p[i], cmd[i], snap_p[near], dt=dt, v_max=c.v_max,
                                                  d_safe=c.d_safe)
                    n_filtered += int(changed)
                cmd = safe
            for i in range(N):
                new_p, new_v = step_single_integrator(pos[i], cmd[i], dt, c.v_max)
                path_len[i] += float(np.linalg.norm(new_p - pos[i]))
                pos[i], vel[i] = new_p, new_v
                if self.paths is not None:
                    self.paths[i].append(new_p.copy())

        seen = coverable & (visits > 0)
        return RunResult(
            model=getattr(self.model, "name", type(self.model).__name__), world=w.name, n_agents=N,
            completed=coverage_time is not None, coverage_time=coverage_time, final_time=t,
            believed_done=believed_done, min_pair_dist=min_pair, min_clearance=min_clear,
            collision_events=events, max_tracking_error=max_err.tolist(), path_length=path_len.tolist(),
            victims_found=found, model_time=model_time, n_decisions=n_dec, n_messages=n_msgs, log=log,
            extra={**self.model.stats(), **({"safety_interventions": n_filtered} if sc.safety != "none" else {})},
            final_coverage=frac, time_to=dict(milestones),
            mean_visits=float(visits[seen].mean()) if seen.any() else 0.0,
            frac_revisited=float((visits[seen] > 1).mean()) if seen.any() else 0.0,
            target_visits=visits, pair_events=pair_events, obstacle_events=obstacle_events,
            n_deliveries=n_deliveries)
