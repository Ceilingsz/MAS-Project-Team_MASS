"""Greedy explorer (baseline). One instance per drone, unknown-map interface.

Each decision a drone goes to the standing position that maximises

    (UNKNOWN cells it would reveal) / (travel time + delta)

"Reveal" counts only cells still unknown to THIS drone, seen from the spot within sensing range and line of sight
past the obstacles it already knows. Travel time is the shortest path through known free space, with unknown space
treated as free. When it senses an obstacle the route is re-planned; a goal that turns out to be inside an obstacle,
or that was seen on the way, is dropped. A drone is done when no reachable, unvisited spot would reveal anything.

It uses nothing but its own belief (and, with share=True, beliefs received from drones within r_comm). It does not
look at other drones at all: no collision avoidance and no deconfliction of goals. That is the baseline weakness.

Known vs unknown map is a property of the RUN, not of this code:
  SimConfig(known_map=False)                                   obstacles are learned by sensing
  SimConfig(known_map=False, reveal_obstacles_at_start=True)   the same code with the map given (coverage still unknown)
"""
from __future__ import annotations

import numpy as np

from mascov.common.model import AgentPolicy, DecentralisedModel, register
from mascov.common.types import Command, Message
from mascov.planning.navigator import Navigator
from mascov.planning.spots import choose_spot  # noqa: F401  (re-exported)


class GreedyExplorerPolicy(AgentPolicy):
    map_interface = "unknown"

    def __init__(self, delta: float = 1.0, seed: int = 0, share: bool = False):
        self.delta, self.seed, self.share = delta, seed, share

    def reset(self, world, agent_id: int, n_agents: int) -> None:
        self.w, self.id, self.cfg = world, agent_id, world.cfg
        self.spots = world.candidates
        self.known = frozenset()
        self.space = world.space_for({})
        self.nav = Navigator(self.cfg, self.space.planner, self.spots)
        self.rng = np.random.default_rng(self.seed * 1000 + agent_id)
        self.visited: set[int] = set()
        self.goal: int | None = None
        self.done = False
        self.n_rebuilds = self.n_dropped = self.n_merged = 0

    def _sync_space(self, belief) -> None:
        key = frozenset(belief.obstacles)
        if key != self.known:
            self.known, self.space = key, self.w.space_for(belief.obstacles)
            self.nav.set_planner(self.space.planner)
            self.n_rebuilds += 1

    def step(self, obs) -> Command:
        belief = obs.belief
        for m in obs.inbox:
            if m.kind == "belief":
                self.n_merged += belief.merge(m.payload)
        self._sync_space(belief)
        gain = self.space.gains(belief.unknown)
        plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished)

        if self.goal is not None:
            if self.nav.arrived or self.nav.mode == "blocked":
                self.visited.add(self.goal)
                self.goal = None
            elif not self.space.valid[self.goal] or gain[self.goal] == 0:
                self.goal = None
                self.n_dropped += 1
        if self.goal is None:
            usable = self.space.valid.copy()
            usable[list(self.visited)] = False
            travel = self.space.distances(obs.pos) / self.cfg.plan_speed
            g = choose_spot(gain, travel, usable, self.delta, self.rng)
            self.done = g is None
            self.goal = g
            self.nav.set_goal(None if g is None else self.spots[g])
            plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished)

        msgs = [Message(self.id, "belief", belief.copy())] if self.share and obs.comm_neighbors else []
        return Command(plan=plan, messages=msgs, done=self.done)

    def stats(self) -> dict:
        return dict(self.nav.stats, map_rebuilds=self.n_rebuilds, goals_dropped=self.n_dropped,
                    cells_merged=self.n_merged)


@register("greedy_explorer")
def GreedyExplorer(delta: float = 1.0, seed: int = 0, share: bool = False):
    return DecentralisedModel(GreedyExplorerPolicy, name="greedy_explorer", delta=delta, seed=seed, share=share)
