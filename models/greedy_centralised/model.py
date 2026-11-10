"""B2: centralised greedy for N drones.

One planner sees every drone's belief (their union is the global coverage) and hands out
viewpoints like a list scheduler: whichever drone needs work picks the viewpoint with the best
(new targets) / (travel time + delta), counting only targets that are neither covered nor already
claimed by another drone's current viewpoint. A claim is dropped if the targets it was chosen for
get seen en route.

Collisions: the navigator's priority rule. Working drones outrank idle ones, then lower id first.
"""
from __future__ import annotations

import numpy as np

from mascov.common.model import CentralModel, register
from mascov.common.types import Command
from mascov.planning.greedy import choose_viewpoint
from mascov.planning.navigator import Navigator, Neighbor


@register("greedy_centralised")
class GreedyCentralised(CentralModel):
    def __init__(self, delta: float = 1.0, margin: float = 0.5):
        self.delta, self.margin = delta, margin

    def reset(self, world, agent_ids):
        self.w, self.ids = world, list(agent_ids)
        self.vp = world.viewpoints
        self.nav = {i: Navigator(world.cfg, world.planner, self.vp.positions, margin=self.margin, seed=i) for i in self.ids}
        self.claim: dict[int, int | None] = {i: None for i in self.ids}
        self.visited: set[int] = set()
        self.done = False

    def _priority(self, i: int) -> tuple:
        return (0 if self.claim[i] is not None else 1, i)

    def _claimed_targets(self, exclude: int) -> np.ndarray:
        m = np.zeros(len(self.vp.coverable), dtype=bool)
        for j, v in self.claim.items():
            if j != exclude and v is not None:
                m |= self.vp.sees[v]
        return m

    def _neighbors(self, i, obs) -> list[Neighbor]:
        pi = self._priority(i)
        return [Neighbor(j, obs[j].pos, obs[j].vel, higher=self._priority(j) < pi, plan=self.nav[j].plan)
                for j in self.ids if j != i and j in obs]

    def _navigate(self, i, obs):
        o = obs[i]
        return self.nav[i].update(o.t, o.pos, o.vel, o.plan_finished, self._neighbors(i, obs),
                                  idle=self.claim[i] is None)

    def step(self, obs):
        vp, cfg = self.vp, self.w.cfg
        covered = np.zeros(len(vp.coverable), dtype=bool)
        for o in obs.values():
            covered |= o.belief.covered
        plans = {i: self._navigate(i, obs) for i in obs}              # pass 1: detects arrivals

        needs = []
        for i in sorted(obs):
            v, nav = self.claim[i], self.nav[i]
            if v is not None and nav.arrived:
                self.visited.add(v)
                self.claim[i] = None
            elif v is not None:
                mine = vp.sees[v] & vp.coverable & ~covered & ~self._claimed_targets(i)
                if not mine.any():
                    self.claim[i] = None                              # obsolete: seen en route or claimed
            if self.claim[i] is None:
                needs.append(i)

        for i in needs:                                               # list scheduling, in id order
            o = obs[i]
            statics = [obs[j].pos for j in self.ids if j != i and j in obs and np.linalg.norm(obs[j].vel) < 0.1]
            blocked = []
            if statics:
                d = np.linalg.norm(vp.positions[:, None, :] - np.asarray(statics)[None], axis=2).min(axis=1)
                blocked = list(np.flatnonzero(d < self.nav[i].static_dist + 0.1))
            travel = self.w.dist_to_viewpoints(o.pos) / cfg.v_max
            v = choose_viewpoint(vp, travel, covered | self._claimed_targets(i),
                                 self.visited | set(blocked), self.delta)
            self.claim[i] = v
            self.nav[i].set_goal(None if v is None else vp.positions[v])
            plans[i] = self._navigate(i, obs)                         # pass 2: route to the new goal

        working = [i for i in self.ids if self.claim[i] is not None]
        if not working:        # finished when nobody has a claim and no viewpoint offers any new target
            self.done = choose_viewpoint(vp, np.zeros(len(vp.positions)), covered, self.visited) is None
        return {i: Command(plan=plans.get(i), done=self.done) for i in obs}

    def stats(self):
        out: dict = {}
        for n in self.nav.values():
            for k, v in n.stats.items():
                out[k] = out.get(k, 0) + v
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}
