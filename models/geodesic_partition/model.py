"""Geodesic Voronoi partition (decentralised). One instance per drone, unknown-map interface.

Idea: split the free space between the drones by SHORTEST-PATH (geodesic) distance, so each drone owns the spots
that it can reach before any teammate, then explore your own cell greedily. Euclidean Voronoi would hand a drone
space behind a wall that a teammate reaches first; geodesic distance uses the map to avoid that.

Each decision a drone
  1. merges what it heard: teammates' positions (and, with share=True, their beliefs),
  2. computes shortest-path distances, on the obstacles IT knows (unknown space counts as free), from itself and from
     every teammate whose position it heard in the last ``ttl`` seconds,
  3. owns every spot where it is the closest (ties go to the lowest id),
  4. picks the owned spot with the best (unknown cells revealed) / (travel time + delta); if its cell has nothing
     left to reveal it helps out and picks the best spot anywhere (counted as ``fallback_picks``),
  5. keeps that goal until it arrives or the goal is seen / blocked (a committed goal is not re-partitioned).

The partition is recomputed from where drones are NOW, so the cells move as the drones do, and it sharpens as the
map is learned (in an unknown map the first partition is made with optimistic distances). Drones broadcast their
position every decision (one message each, within r_comm); a drone that hears nobody is a plain greedy explorer.

Known vs unknown map is a property of the RUN (see models/greedy_explorer): SimConfig(known_map=False) versus
SimConfig(known_map=False, reveal_obstacles_at_start=True).
"""
from __future__ import annotations

import numpy as np

from mascov.common.model import AgentPolicy, DecentralisedModel, register
from mascov.common.types import Command, Message
from mascov.planning.navigator import Navigator
from mascov.planning.spots import choose_spot


def owners(dist: np.ndarray, ids) -> np.ndarray:
    """Voronoi owner of every spot. dist: (n_drones, n_spots) path distances; ids: agent id per row.
    Ties go to the lowest id."""
    ids = np.asarray(ids)
    order = np.argsort(ids, kind="stable")
    return ids[order][np.argmin(dist[order], axis=0)]


class GeodesicPartitionPolicy(AgentPolicy):
    map_interface = "unknown"

    def __init__(self, delta: float = 1.0, seed: int = 0, share: bool = False, ttl: float = 3.0):
        self.delta, self.seed, self.share, self.ttl = delta, seed, share, ttl

    def reset(self, world, agent_id: int, n_agents: int) -> None:
        self.w, self.id, self.cfg = world, agent_id, world.cfg
        self.spots = world.candidates
        self.known = frozenset()
        self.space = world.space_for({})
        self.nav = Navigator(self.cfg, self.space.planner, self.spots)
        self.rng = np.random.default_rng(self.seed * 1000 + agent_id)
        self.visited: set[int] = set()
        self.mates: dict[int, tuple[float, np.ndarray]] = {}
        self.goal: int | None = None
        self.done = False
        self.n_rebuilds = self.n_dropped = self.n_merged = self.n_fallback = 0

    def _sync_space(self, belief) -> None:
        key = frozenset(belief.obstacles)
        if key != self.known:
            self.known, self.space = key, self.w.space_for(belief.obstacles)
            self.nav.set_planner(self.space.planner)
            self.n_rebuilds += 1

    def _my_cell(self, t: float, pos, dist_self: np.ndarray) -> np.ndarray:
        ids, rows = [self.id], [dist_self]
        for j, (heard, p) in self.mates.items():
            if t - heard <= self.ttl:
                ids.append(j)
                rows.append(self.space.distances(p))
        if len(ids) == 1:
            return np.ones(len(self.spots), dtype=bool)
        return owners(np.vstack(rows), ids) == self.id

    def step(self, obs) -> Command:
        belief = obs.belief
        for m in obs.inbox:
            if m.kind == "state":
                self.mates[m.sender] = (obs.t, np.asarray(m.payload["pos"], dtype=float))
                if m.payload.get("belief") is not None:
                    self.n_merged += belief.merge(m.payload["belief"])
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
            dist = self.space.distances(obs.pos)
            travel = dist / self.cfg.plan_speed
            g = choose_spot(gain, travel, usable & self._my_cell(obs.t, obs.pos, dist), self.delta, self.rng)
            if g is None:
                g = choose_spot(gain, travel, usable, self.delta, self.rng)
                self.n_fallback += g is not None
            self.done = g is None
            self.goal = g
            self.nav.set_goal(None if g is None else self.spots[g])
            plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished)

        msgs = []
        if obs.comm_neighbors:
            msgs.append(Message(self.id, "state", {"pos": obs.pos.copy(),
                                                   "belief": belief.copy() if self.share else None}))
        return Command(plan=plan, messages=msgs, done=self.done)

    def stats(self) -> dict:
        return dict(self.nav.stats, map_rebuilds=self.n_rebuilds, goals_dropped=self.n_dropped,
                    cells_merged=self.n_merged, fallback_picks=self.n_fallback)


@register("geodesic_partition")
def GeodesicPartition(delta: float = 1.0, seed: int = 0, share: bool = False, ttl: float = 3.0):
    return DecentralisedModel(GeodesicPartitionPolicy, name="geodesic_partition", delta=delta, seed=seed,
                              share=share, ttl=ttl)
