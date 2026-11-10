"""B0: single-drone greedy. Also the end-to-end smoke test of the environment.

Repeatedly flies to the viewpoint with the best (new targets seen) / (travel time + delta).
"""
from __future__ import annotations

import numpy as np

from mascov.common.model import CentralModel, register
from mascov.common.types import Command, Plan


@register("greedy_single")
class GreedySingle(CentralModel):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def reset(self, world, agent_ids):
        if len(agent_ids) != 1:
            raise ValueError("greedy_single is a single-drone baseline")
        self.world = world
        self.visited: set[int] = set()

    def step(self, obs):
        o = obs.get(0)
        if o is None or not o.plan_finished:
            return {}
        vp, cfg = self.world.viewpoints, self.world.cfg
        gain = vp.gain(o.belief.covered).astype(float)
        gain[list(self.visited)] = 0.0
        travel = self.world.planner.dist_matrix(o.pos, vp.positions)[0] / cfg.v_max
        score = np.where(np.isfinite(travel) & (gain > 0), gain / (travel + self.delta), -1.0)
        best = int(np.argmax(score))
        if score[best] <= 0:
            return {0: Command(done=True)}
        self.visited.add(best)
        route = self.world.planner.path(o.pos, vp.positions[best])
        return {0: Command(plan=Plan.from_polyline(route, t0=o.t, speed=cfg.plan_speed))}
