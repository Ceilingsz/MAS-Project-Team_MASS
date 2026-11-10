"""Frontier-greedy explorer for the UNKNOWN-map interface (one instance per drone).

Each decision it picks, among candidate standing positions, the one maximising

    (unknown cells it would reveal) / (travel time + delta)

where "reveal" counts only UNKNOWN cells within sensing range and line of sight past KNOWN obstacles, and
travel time is the shortest path through known free space with UNKNOWN SPACE TREATED AS FREE. When it senses
an obstacle it re-plans: the route is recomputed and a goal that now lies inside an obstacle is dropped.

share=False : decides from its own belief only (the floor, like B3).
share=True  : additionally broadcasts its whole belief to neighbours within r_comm and merges what it hears
              (union merge, so lost or repeated messages cannot hurt). No intent sharing yet, so drones
              that know the same things pick the same frontier: that is the next mechanism to add.

A drone declares itself done when no reachable, unvisited candidate would still reveal an unknown cell.
"""
from __future__ import annotations

import numpy as np

from mascov.common.model import AgentPolicy
from mascov.common.types import Command, Message
from mascov.planning.navigator import Navigator, Neighbor


def choose_frontier(gain: np.ndarray, travel_s: np.ndarray, usable: np.ndarray, delta: float,
                    rng: np.random.Generator | None = None) -> int | None:
    """argmax of gain / (travel time + delta) over usable, reachable candidates with gain > 0."""
    ok = usable & np.isfinite(travel_s) & (gain > 0)
    if not ok.any():
        return None
    score = np.where(ok, gain / (travel_s + delta), -1.0)
    ties = np.flatnonzero(score >= score.max() * (1 - 1e-9))
    return int(rng.choice(ties)) if rng is not None else int(ties[0])


class FrontierAgentPolicy(AgentPolicy):
    map_interface = "unknown"

    def __init__(self, delta: float = 1.0, seed: int = 0, margin: float = 0.5, share: bool = False):
        self.delta, self.seed, self.margin, self.share = delta, seed, margin, share

    def reset(self, world, agent_id: int, n_agents: int) -> None:
        self.w, self.id = world, agent_id
        self.cfg = world.cfg
        self.cands = world.candidates
        self.key = frozenset()
        self.space = world.space_for({})
        self.nav = Navigator(self.cfg, self.space.planner, self.cands[self.space.valid], margin=self.margin,
                             seed=self.seed * 1000 + agent_id)
        self.rng = np.random.default_rng(self.seed * 1000 + agent_id)
        self.visited: set[int] = set()
        self.goal: int | None = None
        self.done = False
        self.n_rebuilds = 0
        self.n_goals_dropped = 0
        self.n_merged_cells = 0

    # ------------------------------------------------------------------ helpers
    def _sync_space(self, belief) -> None:
        """Re-plan on the known obstacles if the set changed (own discovery or a merge)."""
        key = frozenset(belief.obstacles)
        if key != self.key:
            self.key = key
            self.space = self.w.space_for(belief.obstacles)
            self.nav.set_planner(self.space.planner, self.cands[self.space.valid])
            self.n_rebuilds += 1

    def _neighbors(self, obs) -> list[Neighbor]:
        if not self.cfg.yield_to_neighbors:
            return []                                    # drones ignore each other (the "no safety" baseline)
        return [Neighbor(j, p, obs.neighbor_vels.get(j, np.zeros(2)), higher=j < self.id)
                for j, p in obs.neighbors.items()]

    def _static_blocked(self, nbrs) -> np.ndarray:
        """Candidates too close to a stationary neighbour to stand on."""
        out = np.zeros(len(self.cands), dtype=bool)
        pts = [n.pos for n in nbrs if np.linalg.norm(n.vel) < 0.1]
        if pts:
            d = np.linalg.norm(self.cands[:, None, :] - np.asarray(pts)[None], axis=2).min(axis=1)
            out = d < self.nav.static_dist + 0.1
        return out

    # ------------------------------------------------------------------ decision
    def step(self, obs) -> Command:
        belief = obs.belief
        for m in obs.inbox:
            if m.kind == "belief":
                self.n_merged_cells += belief.merge(m.payload)
        self._sync_space(belief)

        nbrs = self._neighbors(obs)
        gain = self.space.gains(belief.unknown)
        plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished, nbrs, idle=self.goal is None)

        if self.nav.arrived and self.goal is not None:
            self.visited.add(self.goal)
            self.goal = None
        if self.goal is not None and (not self.space.valid[self.goal] or gain[self.goal] == 0):
            self.goal = None                                  # seen en route, or now inside a known obstacle
            self.n_goals_dropped += 1
        if self.goal is None:
            usable = self.space.valid & ~self._static_blocked(nbrs)
            usable[list(self.visited)] = False
            travel = self.space.distances(obs.pos) / self.cfg.v_max
            g = choose_frontier(gain, travel, usable, self.delta, self.rng)
            if g is None:
                self.done = True
                self.nav.set_goal(None)
            else:
                self.goal = g
                self.nav.set_goal(self.cands[g])
            plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished, nbrs, idle=self.goal is None)

        msgs = []
        if self.share and obs.neighbors:
            msgs.append(Message(self.id, "belief", belief.copy()))
        return Command(plan=plan, messages=msgs, done=self.done)

    def stats(self) -> dict:
        return dict(self.nav.stats, map_rebuilds=self.n_rebuilds, goals_dropped=self.n_goals_dropped,
                    cells_merged=self.n_merged_cells)
