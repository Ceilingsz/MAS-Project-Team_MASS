"""Data types that cross layer boundaries: plans, commands, observations, messages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from mascov.world.belief import AgentBelief, MapBelief


class Plan:
    """A piecewise-linear timed reference path: the single output format of every planner.

    Waypoints are (t, x, y). Before the first waypoint the reference holds at it; after the
    last waypoint it holds at the last one.
    """

    def __init__(self, times: Sequence[float], points: Sequence[Sequence[float]]):
        t = np.asarray(times, dtype=float)
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(t) != len(p) or len(t) == 0:
            raise ValueError("Plan needs matching, non-empty times and points")
        order = np.argsort(t, kind="stable")
        self.t, self.p = t[order], p[order]

    @classmethod
    def from_polyline(cls, points: Sequence[Sequence[float]], t0: float, speed: float) -> "Plan":
        """Constant-speed traversal of a polyline starting at time t0."""
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        t = t0 + np.concatenate([[0.0], np.cumsum(seg)]) / max(speed, 1e-9)
        return cls(t, p)

    @classmethod
    def hold(cls, point: Sequence[float], t0: float = 0.0) -> "Plan":
        return cls([t0], [point])

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    @property
    def end_point(self) -> np.ndarray:
        return self.p[-1]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.p, axis=0), axis=1).sum())

    def ref(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Reference position and velocity at time t."""
        if t <= self.t[0] or len(self.t) == 1:
            return self.p[0].copy(), np.zeros(2)
        if t >= self.t[-1]:
            return self.p[-1].copy(), np.zeros(2)
        k = int(np.searchsorted(self.t, t, side="right")) - 1
        dt = self.t[k + 1] - self.t[k]
        v = (self.p[k + 1] - self.p[k]) / dt
        return self.p[k] + v * (t - self.t[k]), v

    def waypoints(self) -> list[tuple[float, float, float]]:
        return [(float(t), float(x), float(y)) for t, (x, y) in zip(self.t, self.p)]


@dataclass
class Message:
    sender: int
    kind: str
    payload: Any = None
    to: Optional[int] = None   # None = broadcast to everyone within communication range


@dataclass
class Command:
    """What a model returns for one agent at a decision step."""
    plan: Optional[Plan] = None                 # replaces the agent's current plan if given
    messages: list[Message] = field(default_factory=list)
    done: bool = False                          # agent believes the mission is complete


@dataclass
class Observation:
    """Everything one agent is allowed to know at a decision step."""
    t: float
    agent_id: int
    pos: np.ndarray
    vel: np.ndarray
    belief: "AgentBelief | MapBelief"           # the agent's own memory (it may merge into it)
    newly_covered: np.ndarray                   # ids this agent saw since the last decision: target ids (known map)
                                                # or lattice cell ids (unknown map)
    new_obstacles: list[int]                    # obstacle ids this agent discovered since the last decision
    neighbors: dict[int, np.ndarray]            # id -> position, agents within the neighbour-sensing radius
    inbox: list[Message]
    plan_finished: bool                         # tracker is at the end of the plan (or has none)
    plan_time_left: float                       # seconds until the current plan ends (0 if none)
    neighbor_vels: dict[int, np.ndarray] = field(default_factory=dict)   # id -> velocity (relative-state sensing)
    comm_neighbors: list[int] = field(default_factory=list)              # ids within the COMMUNICATION radius r_comm
                                                                         # (``neighbors`` uses the sensing radius r_neighbor)
