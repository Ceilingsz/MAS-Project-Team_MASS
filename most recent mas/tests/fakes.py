"""A tiny numpy-only open-space world and scripted models, for testing simulator bookkeeping (metrics, message
counts, neighbour ranges) without any geometry. It implements only what Simulator.run touches."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mascov.common.config import WorldConfig
from mascov.common.model import CentralModel
from mascov.common.types import Command, Plan


class FakeSensor:
    def __init__(self, targets, radius):
        self.targets, self.radius = targets, radius

    def visible_target_ids(self, pos, radius=None):
        d = np.linalg.norm(self.targets - np.asarray(pos), axis=1)
        return np.flatnonzero(d <= (radius or self.radius))

    def visible_obstacle_ids(self, pos):
        return set()

    def visible_points_mask(self, pos, pts):
        return np.zeros(len(pts), dtype=bool)


class FakeWorld:
    """Open 12 x 12 m field with the given targets and start positions, no obstacles."""

    def __init__(self, targets, starts, cfg: WorldConfig | None = None, name="fake"):
        self.cfg = cfg or WorldConfig()
        self.name = name
        self.targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        self.starts = np.asarray(starts, dtype=float).reshape(-1, 2)
        self.coverable = np.ones(len(self.targets), dtype=bool)
        self.victims = np.zeros((0, 2))
        self.obstacles = []
        self.obstacle_union = None
        self.sensor = FakeSensor(self.targets, self.cfg.r_sense)

    @property
    def n_agents(self):
        return len(self.starts)

    @property
    def n_targets(self):
        return len(self.targets)

    def view(self):
        return SimpleNamespace(cfg=self.cfg)


class ScriptedModel(CentralModel):
    """Every drone follows a fixed list of waypoints at plan speed; optionally broadcasts messages."""
    name = "scripted"

    def __init__(self, routes=None, broadcast_from=(), decision_period=None):
        self.routes, self.broadcast_from = routes or {}, set(broadcast_from)
        self.decision_period = decision_period
        self.first_obs, self.calls = None, 0

    def reset(self, world, agent_ids):
        self.cfg, self.sent = world.cfg, False
        self.first_obs, self.calls = None, 0

    def step(self, obs):
        from mascov.common.types import Message
        self.calls += 1
        if self.first_obs is None:
            self.first_obs = obs
        out = {}
        for i, o in obs.items():
            cmd = Command()
            if i in self.routes and not self.sent:
                cmd.plan = Plan.from_polyline([o.pos, *self.routes[i]], o.t, self.cfg.plan_speed)
            if i in self.broadcast_from and not self.sent:
                cmd.messages = [Message(i, "hello")]
            out[i] = cmd
        self.sent = True
        return out
