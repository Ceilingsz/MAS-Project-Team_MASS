"""Simple single-agent waypoint navigation.

The frontier policy chooses *where* an agent should go.  This module only
computes a path through the currently known free space and turns that path into
a timed Plan.  It deliberately does not perform inter-agent collision
avoidance; ``nbrs`` and the legacy constructor options are accepted only to
keep the existing caller interface compatible.

A newly discovered obstacle is handled by ``set_planner``: the old path is
invalidated and a new path to the same goal is computed on the next ``update``.
"""
from __future__ import annotations

import numpy as np

from mascov.common.types import Plan
from mascov.geometry.pathing import PathPlanner


class Neighbor:
    """Legacy neighbour record retained for callers; navigation ignores it."""

    def __init__(self, id, pos, vel, higher=False, plan=None):
        self.id = int(id)
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.higher = bool(higher)
        self.plan = plan


class Navigator:
    """Follow one waypoint using the current known-map PathPlanner.

    Public methods/attributes used by ``FrontierAgentPolicy`` are preserved:
    ``set_planner``, ``set_goal``, ``update``, ``goal``, ``mode``, ``plan``,
    ``arrived`` and ``stats``.
    """

    def __init__(
        self,
        cfg,
        planner: PathPlanner,
        candidates: np.ndarray,
        horizon: float = 2.5,
        step: float = 0.2,
        margin: float = 0.5,
        static_margin: float = 0.3,
        patience: float = 4.0,
        max_candidates: int = 25,
        sidestep_radius: float = 5.0,
        seed: int = 0,
        arrival_tolerance: float = 0.25,
    ):
        self.cfg = cfg
        self.planner = planner
        self.cands = np.asarray(candidates, dtype=float)
        self.arrival_tolerance = float(arrival_tolerance)
        # Compatibility attribute read by frontier/greedy policies when filtering
        # candidate points. Zero disables meaningful neighbour-based filtering.
        self.static_dist = 0.0

        self.goal: np.ndarray | None = None
        self.mode = "idle"  # idle | go | blocked | arrived
        self.plan: Plan | None = None
        self.arrived = False

        self._needs_replan = False
        self._stop_pending = False
        self.stats = {
            # Keep legacy metric keys so existing CSV/reporting code still works.
            "yields": 0,
            "wait_s": 0.0,
            "sidesteps": 0,
            "pushes": 0,
            "reroutes": 0,
            "relocations": 0,
            # New navigation counters.
            "replans": 0,
            "blocked_goals": 0,
            "arrivals": 0,
        }

    def set_planner(
        self, planner: PathPlanner, candidates: np.ndarray | None = None
    ) -> None:
        """Install an updated known map and invalidate any path based on the old map."""
        self.planner = planner
        if candidates is not None:
            self.cands = np.asarray(candidates, dtype=float)
        if self.goal is not None:
            self.plan = None
            self._needs_replan = True
            self.mode = "go"

    def set_goal(self, goal) -> None:
        """Set (or clear) the waypoint to follow."""
        had_goal_or_plan = self.goal is not None or self.plan is not None
        self.goal = None if goal is None else np.asarray(goal, dtype=float)
        self.arrived = False
        self.plan = None
        self._needs_replan = goal is not None
        self._stop_pending = goal is None and had_goal_or_plan
        self.mode = "idle" if goal is None else "go"

    def update(
        self,
        t: float,
        pos,
        vel,
        plan_finished: bool,
        nbrs=None,
        idle: bool = False,
    ) -> Plan | None:
        """Return a replacement Plan, or ``None`` to retain the current plan.

        ``nbrs`` and ``idle`` remain in the signature for compatibility but are
        intentionally ignored: this navigator has no inter-drone logic.
        """
        del vel, nbrs, idle
        pos = np.asarray(pos, dtype=float)

        # Clearing a goal must cancel the previous command rather than leave the
        # drone following a stale plan.
        if self.goal is None:
            self.mode = "idle"
            if self._stop_pending:
                self._stop_pending = False
                self.plan = Plan.hold(pos, t)
                return self.plan
            return None

        # Reaching the waypoint is based on actual position, not only plan time.
        if np.linalg.norm(pos - self.goal) <= self.arrival_tolerance:
            if not self.arrived:
                self.stats["arrivals"] += 1
            self.arrived = True
            self.mode = "arrived"
            self.plan = Plan.hold(pos, t)
            self._needs_replan = False
            return self.plan

        # A completed plan that did not reach its endpoint (e.g. tracking error)
        # is recomputed from the agent's current position.
        if plan_finished and self.plan is not None:
            self._needs_replan = True

        if self.plan is not None and not self._needs_replan:
            return None

        route = self.planner.path(pos, self.goal)
        self._needs_replan = False
        if route is None or len(route) == 0:
            self.mode = "blocked"
            self.stats["blocked_goals"] += 1
            self.plan = Plan.hold(pos, t)
            return self.plan

        self.mode = "go"
        self.stats["replans"] += 1
        if len(route) == 1 or np.linalg.norm(np.asarray(route[-1]) - pos) < 1e-9:
            self.plan = Plan.hold(pos, t)
        else:
            speed = float(getattr(self.cfg, "plan_speed", None) or self.cfg.v_max)
            self.plan = Plan.from_polyline(route, t, speed)
        return self.plan

    def notify_collision(self) -> None:
        """Invalidate the current route after the caller has handled a collision.

        The caller must first update the belief/planner if the collision revealed
        new obstacle geometry.  The next ``update`` then replans to the same goal.
        """
        if self.goal is not None:
            self.plan = None
            self._needs_replan = True
            self.mode = "go"