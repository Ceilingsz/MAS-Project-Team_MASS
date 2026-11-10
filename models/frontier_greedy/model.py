"""Frontier-greedy exploration for the UNKNOWN-map interface (see planning/agent_frontier.py).

frontier_greedy         : each drone uses only what it has sensed itself (no messages).
frontier_greedy_shared  : drones also broadcast and merge beliefs within r_comm.

Run with SimConfig(known_map=False). Add reveal_obstacles_at_start=True to give the same algorithm the map
(coverage stays unknown): the reference for "what does not knowing the map cost".
"""
from __future__ import annotations

from mascov.common.model import DecentralisedModel, register
from mascov.planning.agent_frontier import FrontierAgentPolicy


@register("frontier_greedy")
def FrontierGreedy(delta: float = 1.0, seed: int = 0, margin: float = 0.5):
    return DecentralisedModel(FrontierAgentPolicy, name="frontier_greedy", delta=delta, seed=seed,
                              margin=margin, share=False)


@register("frontier_greedy_shared")
def FrontierGreedyShared(delta: float = 1.0, seed: int = 0, margin: float = 0.5):
    return DecentralisedModel(FrontierAgentPolicy, name="frontier_greedy_shared", delta=delta, seed=seed,
                              margin=margin, share=True)
