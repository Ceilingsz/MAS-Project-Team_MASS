"""Frontier goals + D-MPC trajectories for the UNKNOWN-map interface (one instance per drone).

Layer 1 is the frontier explorer's rule (``agent_frontier.choose_frontier``): among candidate standing positions,
the one maximising (UNKNOWN cells it would reveal, line of sight past KNOWN obstacles) / (travel time + delta),
travel time through known free space with unknown space treated as free. Layer 2 is unchanged D-MPC
(``agent_dmpc.DMPCAgentPolicy``), but its obstacle constraints and its reference path use only the obstacles
this drone KNOWS. When it learns of a new obstacle (by sensing it, or from a neighbour's belief when share=True)
it rebuilds both and re-plans; a goal that turns out to lie inside an obstacle is dropped.

Optimism is safe here as long as an obstacle is seen before it matters: R_s = 1.5 m against a clearance of
r_b + eps_track, and the MPC's obstacle constraints are re-linearised every step.

Run with SimConfig(known_map=False). With reveal_obstacles_at_start=True every belief starts with all
obstacles: the same algorithm with the map given, the reference for "what does not knowing the map cost".
"""
from __future__ import annotations

import numpy as np

from mascov.planning.agent_dmpc import DMPCAgentPolicy
from mascov.planning.agent_frontier import choose_frontier
from mascov.planning.dmpc import obstacle_halfspaces


class FrontierDMPCAgentPolicy(DMPCAgentPolicy):
    map_interface = "unknown"

    def _setup_map(self, world) -> None:
        self.cands = world.candidates
        self.goal_pos = self.cands
        self.key = None
        self.n_rebuilds = 0
        self._use_space(world.space_for({}), {})

    def _use_space(self, space, obstacles: dict) -> None:
        self.space = space
        self.planner = space.planner
        self.obs_hs = obstacle_halfspaces(obstacles.values())
        self.key = frozenset(obstacles)

    def _sync_map(self, obs) -> None:
        key = frozenset(obs.belief.obstacles)
        if key != self.key:
            self._use_space(self.w.space_for(obs.belief.obstacles), dict(obs.belief.obstacles))
            self.n_rebuilds += 1
            self.path = None                                  # re-plan around what was just learned

    def _goal_useful(self, obs, v: int) -> bool:
        return bool(self.space.valid[v]) and self.space.gains(obs.belief.unknown)[v] > 0

    def select_goal(self, obs) -> int | None:
        gain = self.space.gains(obs.belief.unknown)
        usable = self.space.valid.copy()
        excl = self.visited | {v for v, t_end in self.banned.items() if t_end > obs.t}
        if self.coupling:
            excl |= self._static_blocked(obs)
        if excl:
            usable[list(excl)] = False
        travel = self.space.distances(obs.pos) / self.cfg.v_max
        return choose_frontier(gain, travel, usable, self.delta, self.rng)

    def stats(self) -> dict:
        return dict(super().stats(), map_rebuilds=self.n_rebuilds)
