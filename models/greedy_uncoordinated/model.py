"""B3: uncoordinated greedy: every drone runs the single-drone greedy on its OWN belief.

No messages, no shared coverage, so drones duplicate each other's work. A drone only declares
itself done when its own belief is complete, which (beliefs being subsets of the truth) can never
happen before the true completion time. Exact ties are broken at random, otherwise drones that
start together would move in lockstep and N drones would behave like one.
"""
from __future__ import annotations

from mascov.common.model import DecentralisedModel, register
from mascov.planning.agent_greedy import GreedyAgentPolicy


@register("greedy_uncoordinated")
def GreedyUncoordinated(delta: float = 1.0, seed: int = 0, margin: float = 0.5):
    return DecentralisedModel(GreedyAgentPolicy, name="greedy_uncoordinated", delta=delta, seed=seed, margin=margin)
