"""Greedy + D-MPC (known map) and Frontier + D-MPC (unknown map): goals from each drone's own belief,
trajectories by distributed MPC.

Each drone solves its own QP every control step and broadcasts its predicted path; neighbours' predictions
enter as linearised separation constraints (``coupling=True``). ``coupling=False`` is the ablation in which
drones ignore each other. See ``mascov.planning.agent_dmpc`` and ``mascov.planning.dmpc``.

Options: coupling (bool), horizon (steps), margin (extra separation over d_safe, m), share (also broadcast and
merge coverage beliefs), decision_period (s, default the control step 0.1 s), seed, plus DMPCParams weights.
"""
from __future__ import annotations

from mascov.common.model import DecentralisedModel, register
from mascov.planning.agent_dmpc import DMPCAgentPolicy
from mascov.planning.agent_frontier_dmpc import FrontierDMPCAgentPolicy


@register("greedy+dmpc")
def GreedyDMPC(coupling: bool = True, horizon: int = 10, margin: float = 0.1, share: bool = False,
               decision_period: float = 0.1, seed: int = 0, **kw):
    m = DecentralisedModel(DMPCAgentPolicy, name="greedy+dmpc", coupling=coupling, horizon=horizon,
                           margin=margin, share=share, seed=seed, **kw)
    m.decision_period = decision_period
    return m


@register("frontier+dmpc")
def FrontierDMPC(coupling: bool = True, horizon: int = 10, margin: float = 0.1, share: bool = False,
                 decision_period: float = 0.1, seed: int = 0, **kw):
    """Unknown-map interface: frontier goals + D-MPC over the obstacles each drone knows. share=True also
    broadcasts and merges map beliefs. Run with SimConfig(known_map=False)."""
    m = DecentralisedModel(FrontierDMPCAgentPolicy, name="frontier+dmpc", coupling=coupling, horizon=horizon,
                           margin=margin, share=share, seed=seed, **kw)
    m.decision_period = decision_period
    return m

