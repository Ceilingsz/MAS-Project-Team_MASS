"""Run results and a simple time-series logger.

Measurables (what the experiments observe), all computed by the simulator from ground truth:

  speed         coverage_time (full coverage), time_to[f] / t50 ... t99 (first time a fraction f was covered)
  completeness  final_coverage
  safety        collision_events (= pair_events + obstacle_events, counted as separate episodes),
                min_pair_dist, min_clearance, safe_complete (full coverage AND no collision)
  effort        mean_visits (average number of visits per target; a visit is one continuous stretch
                during which one drone has the target in view), frac_revisited, path_length
  coordination  n_messages / n_deliveries (plus the negotiation rounds a model reports in ``extra``),
                msgs_per_drone_per_s, solve_ms_per_decision
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0) if len(x) > 1 else 0.0


def milestone_key(f: float) -> str:
    """0.95 -> 't95', 1.0 -> 't100'."""
    return f"t{int(round(100 * f))}"


class MetricsLogger:
    """Collects rows of (time, fields...) and writes them as CSV."""

    def __init__(self):
        self.rows: list[dict] = []

    def log(self, **row):
        self.rows.append(row)

    def column(self, key: str) -> np.ndarray:
        return np.asarray([r[key] for r in self.rows], dtype=float)

    def to_csv(self, path):
        if not self.rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


@dataclass
class RunResult:
    model: str
    world: str
    n_agents: int
    completed: bool                      # true coverage of all coverable targets reached
    coverage_time: float | None          # ground-truth time of full coverage
    final_time: float
    believed_done: list[float | None]    # per agent: when it believed the mission was done
    min_pair_dist: float                 # smallest centre distance between two drones (inf if N=1)
    min_clearance: float                 # smallest centre distance to an obstacle or wall
    collision_events: int                # number of separate violation episodes (drone-drone or drone-obstacle)
    max_tracking_error: list[float]      # per agent: worst |position - reference| => measured eps
    path_length: list[float]
    victims_found: dict[int, float]
    model_time: float                    # wall-clock seconds spent inside model.step
    n_decisions: int
    n_messages: int                      # messages sent through Command.messages (one per send, however many receive it)
    log: MetricsLogger = field(default_factory=MetricsLogger, repr=False)
    extra: dict = field(default_factory=dict)   # model-specific counters (yields, sidesteps, neg_*, ...)
    final_coverage: float = 0.0          # true fraction of coverable targets seen when the run ended
    time_to: dict = field(default_factory=dict)  # coverage fraction -> first time it was reached (missing: never)
    mean_visits: float = 0.0             # average number of visits per target, over the targets that were seen
    frac_revisited: float = 0.0          # share of seen targets that were visited more than once
    target_visits: np.ndarray | None = field(default=None, repr=False)   # per target visit counts
    pair_events: int = 0                 # episodes of two drones closer than d_safe
    obstacle_events: int = 0             # episodes of a drone closer than its radius to an obstacle or wall
    n_deliveries: int = 0                # messages received (a broadcast heard by k drones counts k)

    # ------------------------------------------------------------------ derived measurables
    @property
    def safe_complete(self) -> bool:
        """The reliability measurable: full coverage and no collision of any kind."""
        return bool(self.completed and self.collision_events == 0)

    @property
    def n_transmissions(self) -> int:
        """Messages sent, including those exchanged inside multi-round negotiations."""
        return int(self.n_messages + self.extra.get("neg_transmissions", 0))

    @property
    def n_received(self) -> int:
        return int(self.n_deliveries + self.extra.get("neg_deliveries", 0))

    @property
    def msgs_per_drone_per_s(self) -> float:
        return self.n_transmissions / max(1e-9, self.n_agents * self.final_time)

    @property
    def solve_ms_per_decision(self) -> float:
        """Wall-clock milliseconds the model spent per call of step() (planner / solver cost)."""
        return 1000.0 * self.model_time / max(1, self.n_decisions)

    def time_to_fraction(self, f: float) -> float | None:
        """First time the true coverage reached fraction f (None if it never did)."""
        if f in self.time_to:
            return self.time_to[f]
        if not self.log.rows:
            return None
        t, c = self.log.column("t"), self.log.column("coverage")
        hit = np.flatnonzero(c >= f - 1e-9)
        return float(t[hit[0]]) if hit.size else None

    def coverage_curve(self, max_points: int = 400, column: str = "coverage") -> tuple[np.ndarray, np.ndarray]:
        """(t, true coverage) from the log, thinned to at most max_points points (column="team_believed": the
        team's pooled belief instead of the truth)."""
        t, c = self.log.column("t"), self.log.column(column)
        if len(t) > max_points:
            keep = np.unique(np.linspace(0, len(t) - 1, max_points).round().astype(int))
            t, c = t[keep], c[keep]
        return t, c

    def coverage_auc(self, horizon: float | None = None) -> float:
        """Mean true coverage over [0, horizon]: 1.0 means everything was covered at time 0.

        The curve is held at its last value after the run ends, so runs that stop at full coverage are
        credited with 1.0 afterwards. Use the SAME horizon for every run you compare.
        """
        t, c = self.log.column("t"), self.log.column("coverage")
        horizon = float(horizon if horizon is not None else self.final_time)
        if horizon <= 0 or len(t) == 0:
            return 0.0
        if t[0] > 0:
            t, c = np.r_[0.0, t], np.r_[0.0, c]
        grid = np.r_[t[t < horizon], horizon]
        vals = np.interp(grid, t, c)                   # np.interp holds the last value beyond the end
        return _trapezoid(vals, grid) / horizon

    # ------------------------------------------------------------------ reporting
    def metrics(self) -> dict:
        """Flat scalar measurables for experiment tables (None means 'not reached' / not applicable)."""
        d = dict(
            completed=self.completed, safe_complete=self.safe_complete,
            coverage_time=self.coverage_time, final_time=self.final_time, final_coverage=self.final_coverage)
        for f in sorted(set(self.time_to) | {0.5, 0.9, 0.95, 1.0}):
            d[milestone_key(f)] = self.time_to_fraction(f)
        d.update(
            collisions=self.collision_events, pair_events=self.pair_events, obstacle_events=self.obstacle_events,
            min_pair_dist=None if not math.isfinite(self.min_pair_dist) else self.min_pair_dist,
            min_clearance=self.min_clearance,
            mean_visits=self.mean_visits, frac_revisited=self.frac_revisited,
            path_len_total=float(sum(self.path_length)), path_len_mean=float(np.mean(self.path_length)),
            transmissions=self.n_transmissions, deliveries=self.n_received,
            msgs_per_drone_per_s=self.msgs_per_drone_per_s,
            decisions=self.n_decisions, solve_ms=self.solve_ms_per_decision,
            max_tracking_error=float(max(self.max_tracking_error)) if self.max_tracking_error else 0.0)
        return d

    def summary(self) -> dict:
        return dict(
            model=self.model, world=self.world, n_agents=self.n_agents,
            completed=self.completed, safe_complete=self.safe_complete,
            coverage_time=None if self.coverage_time is None else round(self.coverage_time, 2),
            final_time=round(self.final_time, 2), final_coverage=round(self.final_coverage, 4),
            **{milestone_key(f): None if t is None else round(t, 2)
               for f, t in ((f, self.time_to_fraction(f)) for f in sorted(set(self.time_to) | {0.5, 0.9, 0.95, 1.0}))},
            believed_done=[None if t is None else round(t, 2) for t in self.believed_done],
            min_pair_dist=None if not np.isfinite(self.min_pair_dist) else round(self.min_pair_dist, 3),
            min_clearance=round(self.min_clearance, 3),
            collision_events=self.collision_events, pair_events=self.pair_events,
            obstacle_events=self.obstacle_events,
            mean_visits=round(self.mean_visits, 3), frac_revisited=round(self.frac_revisited, 3),
            max_tracking_error=[round(e, 3) for e in self.max_tracking_error],
            path_length=[round(p, 2) for p in self.path_length],
            victims_found=len(self.victims_found),
            model_time_s=round(self.model_time, 3),
            n_decisions=self.n_decisions, n_messages=self.n_messages, n_deliveries=self.n_deliveries,
            msgs_per_drone_per_s=round(self.msgs_per_drone_per_s, 4), **self.extra)

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(self.summary(), indent=2))
        self.log.to_csv(d / "timeseries.csv")
