"""Greedy explorer: per-drone greedy with optional goal deconfliction, unknown-map interface.

Each decision a drone goes to the standing position that maximises

    (UNKNOWN cells it would reveal) / (travel time + delta)

Two coordination modes controlled by ``coordinate``:

  * ``False`` (default): pure greedy, no deconfliction. Drones may pick the same spot.
  * ``True``: each drone broadcasts its current goal. When picking a new spot, it excludes
    spots claimed by a teammate it heard in the last ``claim_ttl`` seconds. This is cheap
    coordination — one extra int per message, no bidding, no consensus rounds.

With ``share=True`` (independent of ``coordinate``), drones also exchange coverage beliefs so
they don't go to spots that a teammate has already revealed.

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

    def __init__(self, delta: float = 1.0, seed: int = 0, share: bool = False,
                 coordinate: bool = False, claim_ttl: float = 3.0):
        self.delta, self.seed, self.share = delta, seed, share
        self.coordinate, self.claim_ttl = coordinate, claim_ttl

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
        self.claims: dict[int, tuple[float, int | None]] = {}   # teammate id -> (time heard, their goal spot index)
        self.n_rebuilds = self.n_dropped = self.n_merged = self.n_deconf = 0

    def _sync_space(self, belief) -> None:
        key = frozenset(belief.obstacles)
        if key != self.known:
            self.known, self.space = key, self.w.space_for(belief.obstacles)
            self.nav.set_planner(self.space.planner)
            self.n_rebuilds += 1

    def _claimed_spots(self, t: float) -> set[int]:
        """Spots currently claimed by teammates heard recently."""
        if not self.coordinate:
            return set()
        out = set()
        for j, (heard, g) in self.claims.items():
            if g is not None and t - heard <= self.claim_ttl:
                out.add(g)
        return out

    def step(self, obs) -> Command:
        belief = obs.belief
        for m in obs.inbox:
            if m.kind == "belief":
                self.n_merged += belief.merge(m.payload)
            elif m.kind == "greedy_state":
                if m.payload.get("belief") is not None:
                    self.n_merged += belief.merge(m.payload["belief"])
                self.claims[m.sender] = (obs.t, m.payload.get("goal"))
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
            claimed = self._claimed_spots(obs.t)
            if claimed:
                usable[list(claimed)] = False
                self.n_deconf += 1
            travel = self.space.distances(obs.pos) / self.cfg.plan_speed
            g = choose_spot(gain, travel, usable, self.delta, self.rng)
            if g is None and claimed:
                # all usable spots are claimed — ignore claims and pick anyway
                usable[list(claimed)] = True
                g = choose_spot(gain, travel, usable, self.delta, self.rng)
            self.done = g is None
            self.goal = g
            self.nav.set_goal(None if g is None else self.spots[g])
            plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished)

        # Build outgoing message
        msgs = []
        if obs.comm_neighbors:
            if self.coordinate:
                msgs.append(Message(self.id, "greedy_state", {
                    "goal": self.goal,
                    "belief": belief.copy() if self.share else None,
                }))
            elif self.share:
                msgs.append(Message(self.id, "belief", belief.copy()))
        return Command(plan=plan, messages=msgs, done=self.done)

    def stats(self) -> dict:
        return dict(self.nav.stats, map_rebuilds=self.n_rebuilds, goals_dropped=self.n_dropped,
                    cells_merged=self.n_merged, deconflictions=self.n_deconf)


@register("greedy_explorer")
def GreedyExplorer(delta: float = 1.0, seed: int = 0, share: bool = False,
                   coordinate: bool = False, claim_ttl: float = 3.0):
    return DecentralisedModel(GreedyExplorerPolicy, name="greedy_explorer", delta=delta, seed=seed,
                              share=share, coordinate=coordinate, claim_ttl=claim_ttl)
