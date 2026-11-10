"""A greedy agent that decides from its OWN belief only (shared by B3 and B4).

It never sends messages. It only learns about other drones through relative-state sensing
(positions and velocities of neighbours within communication range), which it uses for
collision avoidance, not for deciding what to cover.
"""
from __future__ import annotations

import numpy as np

from mascov.common.model import AgentPolicy
from mascov.common.types import Command
from mascov.planning.greedy import choose_viewpoint, remaining_gain
from mascov.planning.navigator import Navigator, Neighbor


class GreedyAgentPolicy(AgentPolicy):
    def __init__(self, delta: float = 1.0, region_mask: np.ndarray | None = None, seed: int = 0,
                 margin: float = 0.5):
        self.delta, self.region_mask, self.seed, self.margin = delta, region_mask, seed, margin

    def reset(self, world, agent_id: int, n_agents: int) -> None:
        self.w, self.id = world, agent_id
        self.vp = world.viewpoints
        self.nav = Navigator(world.cfg, world.planner, self.vp.positions, margin=self.margin,
                             seed=self.seed * 1000 + agent_id)
        self.rng = np.random.default_rng(self.seed * 1000 + agent_id)   # random tie-breaking
        self.visited: set[int] = set()
        self.goal_vp: int | None = None
        self.done = False

    def _neighbors(self, obs) -> list[Neighbor]:
        if not self.w.cfg.yield_to_neighbors:
            return []                                    # drones ignore each other (the "no safety" baseline)
        return [Neighbor(j, p, obs.neighbor_vels.get(j, np.zeros(2)), higher=j < self.id)
                for j, p in obs.neighbors.items()]

    def _static_blocked(self, nbrs) -> list[int]:
        """Viewpoints too close to a stationary neighbour to stand on."""
        pts = [n.pos for n in nbrs if np.linalg.norm(n.vel) < 0.1]
        if not pts:
            return []
        d = np.linalg.norm(self.vp.positions[:, None, :] - np.asarray(pts)[None], axis=2).min(axis=1)
        return list(np.flatnonzero(d < self.nav.static_dist + 0.1))

    def step(self, obs) -> Command:
        nbrs = self._neighbors(obs)
        covered = obs.belief.covered
        idle = self.goal_vp is None
        plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished, nbrs, idle=idle)

        if self.nav.arrived and self.goal_vp is not None:
            self.visited.add(self.goal_vp)
            self.goal_vp = None
        if self.goal_vp is not None and remaining_gain(self.vp, self.goal_vp, covered, self.region_mask) == 0:
            self.goal_vp = None                                   # what I came for has been seen en route
        if self.goal_vp is None:
            travel = self.w.dist_to_viewpoints(obs.pos) / self.w.cfg.v_max
            v = choose_viewpoint(self.vp, travel, covered, self.visited | set(self._static_blocked(nbrs)),
                                 self.delta, self.rng, self.region_mask)
            if v is None:
                self.done = True
                self.nav.set_goal(None)
            else:
                self.goal_vp = v
                self.nav.set_goal(self.vp.positions[v])
            plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished, nbrs, idle=self.goal_vp is None)
        return Command(plan=plan, done=self.done)

    def stats(self) -> dict:
        return dict(self.nav.stats)
