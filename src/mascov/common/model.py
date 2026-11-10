"""The contract every algorithm in models/ implements, plus a name registry.

Two flavours:
  * CentralModel     - one object sees all observations (centralised baselines and planners).
  * AgentPolicy      - one instance per drone, sees only its own observation and inbox.
                       Wrap a policy class in DecentralisedModel to run it: the wrapper hands
                       each policy only its own Observation, so "no central planner" is enforced.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Type

from mascov.common.types import Command, Observation


class CentralModel(ABC):
    name: str = "unnamed"
    centralised: bool = True
    # Which world interface the model is written for. "known": WorldView + AgentBelief (target-indexed).
    # "unknown": UnknownWorldView + MapBelief (lattice-indexed, obstacles learned by sensing).
    # The Simulator checks it matches SimConfig.known_map.
    map_interface: str = "known"
    # Seconds between calls to step(). None: use SimConfig.decision_period. A receding-horizon controller
    # that must re-plan every control step sets this to the simulation step (0.1 s).
    decision_period: float | None = None

    @abstractmethod
    def reset(self, world, agent_ids: list[int]) -> None:
        """world is a WorldView (known map) or an UnknownWorldView (see map_interface)."""

    @abstractmethod
    def step(self, obs: dict[int, Observation]) -> dict[int, Command]:
        """Called every decision period. Return commands for any agents that need one."""

    def stats(self) -> dict:
        """Optional model-specific counters, merged into RunResult.extra."""
        return {}


class AgentPolicy(ABC):
    map_interface: str = "known"        # see CentralModel.map_interface
    decision_period: float | None = None   # see CentralModel.decision_period

    @abstractmethod
    def reset(self, world, agent_id: int, n_agents: int) -> None: ...

    @abstractmethod
    def step(self, obs: Observation) -> Command: ...

    def stats(self) -> dict:
        return {}


class DecentralisedModel(CentralModel):
    centralised = False

    def __init__(self, policy_cls: Type[AgentPolicy], name: str = "decentralised", **kwargs):
        self.policy_cls, self.kwargs, self.name = policy_cls, kwargs, name
        self.map_interface = getattr(policy_cls, "map_interface", "known")
        self.decision_period = getattr(policy_cls, "decision_period", None)
        self.policies: dict[int, AgentPolicy] = {}

    def reset(self, world, agent_ids):
        self.policies = {i: self.policy_cls(**self.kwargs) for i in agent_ids}
        for i, p in self.policies.items():
            p.reset(world, i, len(agent_ids))

    def step(self, obs):
        return {i: self.policies[i].step(o) for i, o in obs.items()}

    def stats(self):
        return sum_stats(p.stats() for p in self.policies.values())


def sum_stats(dicts) -> dict:
    out: dict = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}


_REGISTRY: dict[str, Callable[..., CentralModel]] = {}


def register(name: str):
    def deco(factory):
        _REGISTRY[name] = factory
        try:
            factory.name = name
        except AttributeError:
            pass
        return factory
    return deco


def get_model(name: str, **kwargs) -> CentralModel:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def registered_models() -> list[str]:
    return sorted(_REGISTRY)
