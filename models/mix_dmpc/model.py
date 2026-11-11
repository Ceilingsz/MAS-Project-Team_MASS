"""Mix and match: any navigator-based goal layer, flown by D-MPC instead of the Navigator.

``greedy_explorer+dmpc``, ``geodesic_partition+dmpc``, ``cbba_explorer+dmpc``: the goal layer is the UNCHANGED
policy of ``greedy_explorer``, ``geodesic_partition`` or ``cbba_explorer``; only the flying changes. With the
plain models this gives a 3 x 2 grid (goal rule x flying layer) in which the goal decisions come from the same code.

How it is wired (one instance per drone):
* The goal policy is created as usual, then its Navigator is replaced by a ``GoalRecorder``: the policy still calls
  ``set_goal`` / ``update`` / ``arrived`` / ``mode``, but the recorder only notes the target and reports arrival
  (within the Navigator's 0.1 m) or ``mode == "blocked"``; it returns no plan.
* D-MPC (the unknown-map ``FrontierDMPCAgentPolicy``: obstacles learned by sensing, neighbours' predicted paths)
  flies to the recorded target every 0.1 s. When D-MPC gives a target up (stuck after its sidesteps, or no path
  through the known free space), the recorder reports it blocked, and the goal policy treats it exactly as it
  treats a blocked Navigator goal (visited, pick another).
* The goal policy keeps its own cadence (``goal_period``, default 0.5 s = SimConfig.decision_period, as when it
  runs on its own) and receives every message of its own kinds that arrived since its last decision. D-MPC
  messages (predicted paths) go out every 0.1 s; the goal policy's messages go out when it decides, as before.
* Done = the goal policy says done (it decides that from its belief, as before).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from mascov.common.model import DecentralisedModel, register
from mascov.common.types import Command
from mascov.planning.agent_dmpc import KIND as DMPC_KIND
from mascov.planning.agent_frontier_dmpc import FrontierDMPCAgentPolicy


class GoalRecorder:
    """Stands in for the Navigator of a goal policy: records the target, reports arrival or a block."""

    def __init__(self, cfg, planner, candidates=None, arrival_tolerance: float = 0.1, **_):
        self.cfg, self.planner, self.arrival_tolerance = cfg, planner, arrival_tolerance
        self.goal: np.ndarray | None = None
        self.arrived, self.mode, self.blocked = False, "idle", False
        self.static_dist = 0.0
        self.stats = {"goals_set": 0, "blocked_by_dmpc": 0}

    def set_planner(self, planner, candidates=None) -> None:
        self.planner = planner

    def set_goal(self, goal) -> None:
        self.goal = None if goal is None else np.asarray(goal, dtype=float)
        self.arrived, self.blocked = False, False
        self.mode = "idle" if goal is None else "go"
        self.stats["goals_set"] += goal is not None

    def block(self) -> None:
        self.blocked, self.mode = True, "blocked"
        self.stats["blocked_by_dmpc"] += 1

    def update(self, t, pos, vel, plan_finished, nbrs=None, idle=False):
        if self.goal is None:
            self.mode = "idle"
        elif self.blocked:
            self.mode = "blocked"
        elif np.linalg.norm(np.asarray(pos) - self.goal) <= self.arrival_tolerance:
            self.arrived, self.mode = True, "arrived"
        return None                                       # D-MPC produces the plan


class GoalDMPCPolicy(FrontierDMPCAgentPolicy):
    """D-MPC flying whatever goals an existing navigator-based policy chooses."""

    def __init__(self, goal_cls=None, goal_kwargs=None, goal_period: float = 0.5, **dmpc_kw):
        dmpc_kw.setdefault("share", False)                 # the goal policy shares beliefs, not D-MPC
        super().__init__(**dmpc_kw)
        self.goal_cls, self.goal_kwargs, self.goal_period = goal_cls, dict(goal_kwargs or {}), goal_period

    def reset(self, world, agent_id: int, n_agents: int) -> None:
        super().reset(world, agent_id, n_agents)
        self.inner = self.goal_cls(**self.goal_kwargs)
        self.inner.reset(world, agent_id, n_agents)
        nav = self.inner.nav
        self.rec = GoalRecorder(self.cfg, getattr(nav, "planner", None))
        self.inner.nav = self.rec
        self.next_goal_t = 0.0
        self.buffer: list = []
        self.inner_done = False
        self._last: np.ndarray | None = None

    # ------------------------------------------------------------------ goal layer = the inner policy
    def _hold(self, obs) -> None:
        if self.goal is not None:
            self._set_goal(None, obs)
        if self.hold is None:
            self.hold = obs.pos.copy()

    def _update_goal(self, obs) -> None:
        tgt = self.rec.goal
        if self.goal is None and self._last is not None and tgt is not None and np.array_equal(tgt, self._last):
            self.rec.block()                              # D-MPC gave this target up (stuck, or no path)
            self._last = None
        if tgt is None or self.rec.blocked:
            self._hold(obs)
            self.done = self.inner_done
            return
        if self.goal is None or not np.array_equal(tgt, self.goal):
            idx = np.flatnonzero(np.all(self.goal_pos == tgt, axis=1))
            if len(idx) == 0 or self.planner.path(obs.pos, tgt) is None:
                self.rec.block()
                self._hold(obs)
                return
            self._set_goal(int(idx[0]), obs)
            self._last, self.hold = tgt.copy(), None
        self.done = False

    def step(self, obs) -> Command:
        self.buffer.extend(m for m in obs.inbox if m.kind != DMPC_KIND)
        inner_msgs = []
        if obs.t + 1e-9 >= self.next_goal_t:
            cmd = self.inner.step(dataclasses.replace(obs, inbox=self.buffer))
            self.buffer = []
            self.next_goal_t = obs.t + self.goal_period
            self.inner_done = bool(cmd.done)
            inner_msgs = list(cmd.messages)
        cmd = super().step(obs)
        return Command(plan=cmd.plan, messages=inner_msgs + list(cmd.messages), done=self.inner_done)

    def stats(self) -> dict:
        out = dict(super().stats())
        out.update({f"goal_{k}": v for k, v in self.inner.stats().items() if isinstance(v, (int, float))})
        out.update({f"rec_{k}": v for k, v in self.rec.stats.items()})
        return out


def _factory(name, goal_cls):
    def make(seed: int = 0, coupling: bool = True, horizon: int = 10, margin: float = 0.1,
             goal_period: float = 0.5, **goal_kw):
        m = DecentralisedModel(GoalDMPCPolicy, name=name, goal_cls=goal_cls, goal_kwargs=dict(seed=seed, **goal_kw),
                               goal_period=goal_period, seed=seed, coupling=coupling, horizon=horizon, margin=margin)
        m.decision_period = 0.1
        return m
    make.__name__ = name
    return make


from models.cbba.explorer import CBBAExplorerPolicy  # noqa: E402
from models.geodesic_partition.model import GeodesicPartitionPolicy  # noqa: E402
from models.greedy_explorer.model import GreedyExplorerPolicy  # noqa: E402

GreedyExplorerDMPC = register("greedy_explorer+dmpc")(_factory("greedy_explorer+dmpc", GreedyExplorerPolicy))
GeodesicPartitionDMPC = register("geodesic_partition+dmpc")(_factory("geodesic_partition+dmpc", GeodesicPartitionPolicy))
CBBAExplorerDMPC = register("cbba_explorer+dmpc")(_factory("cbba_explorer+dmpc", CBBAExplorerPolicy))

try:  # the learned goal layer needs PyTorch
    from models.marl_explorer.model import MARLExplorerPolicy  # noqa: E402
    MARLExplorerDMPC = register("marl_explorer+dmpc")(_factory("marl_explorer+dmpc", MARLExplorerPolicy))
except ImportError:  # pragma: no cover
    pass
