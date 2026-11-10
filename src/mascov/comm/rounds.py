"""Several rounds of message exchange inside ONE decision step.

The simulator delivers a message at the next decision, which is one hop per decision period. Algorithms that
negotiate (CBBA's consensus, ADMM's iterations) need many exchanges before they can act. This module gives
them that, still without any central planner and with the same range limit as every other message:

  * every round all drones send simultaneously (synchronous rounds), nothing is delivered mid-round;
  * a message reaches only drones within the communication radius r_comm at the decision time, so
    information travels one hop per round and a drone cut off from the rest negotiates only with its group;
  * every transmission and every delivery is counted (``neg_transmissions``, ``neg_deliveries``) and the number
    of rounds used is recorded (``neg_rounds``, ``neg_rounds_max``), all merged into RunResult.extra.

Write a policy by subclassing :class:`NegotiatingPolicy` and wrap it in :class:`NegotiatingModel`::

    class MyPolicy(NegotiatingPolicy):
        max_rounds = 10
        def reset(self, world, agent_id, n_agents): ...
        def begin(self, obs):            # decision starts: read obs, build the local proposal
        def outgoing(self, r):           # round r: messages to send (Message(self.id, kind, payload[, to=j]))
        def incoming(self, r, msgs):     # round r: messages received from neighbours
        def converged(self):             # True when another round would change nothing here
        def finish(self):                # return the Command (plan, final messages, done)

    register("my_model")(lambda **kw: NegotiatingModel(MyPolicy, name="my_model", **kw))

Rounds stop when every drone reports converged() after a round, when nobody sends anything, or at max_rounds.
Messages put in the final Command still use the normal one-decision-late delivery.
"""
from __future__ import annotations

from abc import abstractmethod

from mascov.common.model import AgentPolicy, DecentralisedModel, sum_stats
from mascov.common.types import Command, Message, Observation


class NegotiatingPolicy(AgentPolicy):
    """An AgentPolicy that can exchange messages with its neighbours several times per decision."""

    max_rounds: int = 1

    @abstractmethod
    def begin(self, obs: Observation) -> None:
        """Start of a decision: take in the observation (belief, neighbours, inbox) and form a local proposal."""

    def outgoing(self, round_idx: int) -> list[Message]:
        """Messages to send in this round (default: none)."""
        return []

    def incoming(self, round_idx: int, msgs: list[Message]) -> None:
        """Messages that reached this drone in this round, in sender-id order (default: ignored)."""

    def converged(self) -> bool:
        """True if this drone would not change anything with another round (default: after one round)."""
        return True

    @abstractmethod
    def finish(self) -> Command:
        """End of the decision: the plan (and any final messages) this drone commits to."""

    def step(self, obs: Observation) -> Command:
        """Stand-alone use (no negotiation): begin, then finish."""
        self.begin(obs)
        return self.finish()


class NegotiatingModel(DecentralisedModel):
    """Decentralised model whose policies negotiate in rounds. Still no central planner: each round a drone
    only sees what its neighbours (within r_comm) sent it."""

    def __init__(self, policy_cls, name: str = "negotiating", max_rounds: int | None = None, **kwargs):
        super().__init__(policy_cls, name=name, **kwargs)
        self.max_rounds = max_rounds
        self._counters = self._fresh()

    @staticmethod
    def _fresh() -> dict:
        return dict(neg_decisions=0, neg_rounds=0, neg_rounds_max=0, neg_transmissions=0, neg_deliveries=0)

    def reset(self, world, agent_ids):
        super().reset(world, agent_ids)
        self._counters = self._fresh()

    def step(self, obs):
        pol = self.policies
        for i, o in obs.items():
            pol[i].begin(o)
        limit = self.max_rounds or max(p.max_rounds for p in pol.values())
        used = 0
        for r in range(limit):
            sent = {i: pol[i].outgoing(r) for i in sorted(obs)}
            if not any(sent.values()):
                break                                   # nothing to say: the negotiation is over
            inbox = {i: [] for i in obs}
            for i, msgs in sent.items():
                reach = set(obs[i].comm_neighbors)
                for m in msgs:
                    self._counters["neg_transmissions"] += 1
                    for j in ([m.to] if m.to is not None else sorted(reach)):
                        if j in reach and j in inbox:
                            inbox[j].append(m)
                            self._counters["neg_deliveries"] += 1
            for i in sorted(obs):
                pol[i].incoming(r, inbox[i])
            used += 1
            if all(pol[i].converged() for i in obs):
                break
        self._counters["neg_decisions"] += 1
        self._counters["neg_rounds"] += used
        self._counters["neg_rounds_max"] = max(self._counters["neg_rounds_max"], used)
        return {i: pol[i].finish() for i in sorted(obs)}

    def stats(self):
        out = sum_stats(p.stats() for p in self.policies.values())
        out.update(self._counters)
        return out
