"""``marl_explorer``: a goal layer learned with MAPPO, on the unknown-map interface.

One policy instance per drone (``DecentralisedModel``): it decides from its own Observation and the messages it
hears, nothing else. Execution is decentralised; only training used a centralised critic (``train.py``).

What a drone does each decision (every 0.5 s), as in ``greedy_explorer``:
  * merge what it heard (map beliefs with share=True; every message also carries the sender's position and goal);
  * keep its goal until it arrives, the Navigator reports it blocked, the spot became invalid (inside an obstacle it
    has since learned) or would reveal nothing any more;
  * otherwise choose a new goal: the learned actor scores every usable candidate standing position
    (``features.py``) and the drone takes the best (``stochastic=False``, deterministic: same seed, same run) or
    samples (``stochastic=True``, used while training);
  * done = no reachable, unvisited spot would reveal an unknown cell (the explorers' rule), so ``knows_when_done``
    behaves like the other sharing explorers.

Flying is the shared Navigator (``marl_explorer``) or D-MPC (``marl_explorer+dmpc`` in models/mix_dmpc). Safety is
not learned: add ``SimConfig(safety="bvc")`` or use the D-MPC variant.

Weights: ``checkpoint`` (default ``models/marl_explorer/checkpoints/marl_explorer.pt``). Without a checkpoint the
actor is the untrained residual network, whose argmax is the greedy spot: the model then behaves like
``greedy_explorer(share=True)`` plus position/goal messages.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import torch

from mascov.common.model import AgentPolicy, DecentralisedModel, register
from mascov.common.types import Command, Message
from mascov.planning.navigator import Navigator
from models.marl_explorer.features import candidate_features, statics_for
from models.marl_explorer.nets import Actor

KIND = "marl"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "marl_explorer.pt"

_ACTORS: dict[str, Actor] = {}


def load_actor(checkpoint: str | os.PathLike | None = None) -> Actor:
    """Load (and cache) the actor for inference on the CPU."""
    path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    key = str(path)
    if key not in _ACTORS:
        torch.set_num_threads(1)
        if path.exists():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            actor = Actor(**blob.get("actor_kwargs", {}))
            actor.load_state_dict(blob["actor"])
        else:
            warnings.warn(f"marl_explorer: no checkpoint at {path}; using the untrained (greedy-equivalent) actor")
            actor = Actor()
        actor.eval()
        _ACTORS[key] = actor
    return _ACTORS[key]


class MARLExplorerPolicy(AgentPolicy):
    map_interface = "unknown"

    def __init__(self, checkpoint=None, share: bool = True, delta: float = 1.0, seed: int = 0,
                 claim_ttl: float = 3.0, stochastic: bool = False, actor: Actor | None = None, recorder=None):
        self.checkpoint, self.share, self.delta, self.seed = checkpoint, share, delta, seed
        self.claim_ttl, self.stochastic = claim_ttl, stochastic
        self._actor, self.recorder = actor, recorder

    def reset(self, world, agent_id: int, n_agents: int) -> None:
        self.w, self.id, self.cfg, self.n_agents = world, agent_id, world.cfg, n_agents
        self.actor = self._actor if self._actor is not None else load_actor(self.checkpoint)
        self.spots = world.candidates
        self.statics = statics_for(world)
        self.known = frozenset()
        self.space = world.space_for({})
        self.nav = Navigator(self.cfg, self.space.planner, self.spots)
        self.rng = np.random.default_rng(self.seed * 1000 + agent_id)
        self.visited: set[int] = set()
        self.goal: int | None = None
        self.done = False
        self.team: dict[int, tuple[float, np.ndarray]] = {}        # teammate -> (time, last known position)
        self.claims: dict[int, tuple[float, int | None]] = {}      # teammate -> (time heard, goal spot)
        self.n_rebuilds = self.n_merged = self.n_decisions = self.n_dropped = 0

    # ------------------------------------------------------------------ helpers
    def _sync_space(self, belief) -> None:
        key = frozenset(belief.obstacles)
        if key != self.known:
            self.known, self.space = key, self.w.space_for(belief.obstacles)
            self.nav.set_planner(self.space.planner)
            self.n_rebuilds += 1

    def _listen(self, obs) -> None:
        for m in obs.inbox:
            if m.kind != KIND:
                continue
            p = m.payload
            if p.get("belief") is not None:
                self.n_merged += obs.belief.merge(p["belief"])
            self.team[m.sender] = (obs.t, np.asarray(p["pos"], dtype=float))
            self.claims[m.sender] = (obs.t, p.get("goal"))
        for j, q in obs.neighbors.items():                          # sensed positions are fresher
            self.team[j] = (obs.t, np.asarray(q, dtype=float))

    def _team(self, t: float):
        pos = [q for (heard, q) in self.team.values() if t - heard <= self.claim_ttl]
        goals = [g for (heard, g) in self.claims.values() if g is not None and t - heard <= self.claim_ttl]
        return pos, goals

    def _choose(self, obs) -> int | None:
        usable = self.space.valid.copy()
        if self.visited:
            usable[list(self.visited)] = False
        travel = self.space.distances(obs.pos) / self.cfg.plan_speed
        team_pos, team_goals = self._team(obs.t)
        feats, ctx, mask, base, _ = candidate_features(
            space=self.space, statics=self.statics, unknown=obs.belief.unknown, pos=np.asarray(obs.pos, float),
            travel=travel, usable=usable, candidates=self.spots, team_pos=team_pos, team_goals=team_goals,
            n_agents=self.n_agents, t=obs.t, r_comm=self.cfg.r_comm, r_sense=self.cfg.r_sense, delta=self.delta)
        if not mask.any():
            return None
        with torch.no_grad():
            logits = self.actor(torch.from_numpy(feats)[None], torch.from_numpy(ctx)[None],
                                torch.from_numpy(mask)[None], torch.from_numpy(base)[None])[0].double().numpy()
        z = logits[mask] - logits[mask].max()
        p = np.exp(z)
        p /= p.sum()
        idx = np.flatnonzero(mask)
        if self.stochastic:
            k = int(self.rng.choice(len(idx), p=p))
        else:
            k = int(np.argmax(p))
        a = int(idx[k])
        self.n_decisions += 1
        if self.recorder is not None:
            self.recorder(self.id, obs.t, feats, ctx, mask, base, a, float(np.log(max(p[k], 1e-300))))
        return a

    # ------------------------------------------------------------------ AgentPolicy
    def step(self, obs) -> Command:
        belief = obs.belief
        self._listen(obs)
        self._sync_space(belief)
        plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished)
        if self.goal is not None:
            if self.nav.arrived or self.nav.mode == "blocked":
                self.visited.add(self.goal)
                self.goal = None
            elif not self.space.valid[self.goal] or self.space.gains(belief.unknown)[self.goal] == 0:
                self.goal = None
                self.n_dropped += 1
        if self.goal is None:
            g = self._choose(obs)
            self.done = g is None
            self.goal = g
            self.nav.set_goal(None if g is None else self.spots[g])
            plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished)
        msgs = []
        if obs.comm_neighbors:
            msgs.append(Message(self.id, KIND, {"pos": np.asarray(obs.pos, float).copy(), "goal": self.goal,
                                                "belief": belief.copy() if self.share else None}))
        return Command(plan=plan, messages=msgs, done=self.done)

    def stats(self) -> dict:
        return dict(self.nav.stats, map_rebuilds=self.n_rebuilds, cells_merged=self.n_merged,
                    marl_decisions=self.n_decisions, goals_dropped=self.n_dropped)


@register("marl_explorer")
def MARLExplorer(checkpoint=None, share: bool = True, delta: float = 1.0, seed: int = 0, claim_ttl: float = 3.0,
                 stochastic: bool = False):
    return DecentralisedModel(MARLExplorerPolicy, name="marl_explorer", checkpoint=checkpoint, share=share,
                              delta=delta, seed=seed, claim_ttl=claim_ttl, stochastic=stochastic)
