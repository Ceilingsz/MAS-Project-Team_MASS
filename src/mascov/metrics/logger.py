"""Run results and a simple time-series logger."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


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
    collision_events: int                # number of separate violation episodes
    max_tracking_error: list[float]      # per agent: worst |position - reference| => measured eps
    path_length: list[float]
    victims_found: dict[int, float]
    model_time: float                    # wall-clock seconds spent inside model.step
    n_decisions: int
    n_messages: int
    log: MetricsLogger = field(default_factory=MetricsLogger, repr=False)
    extra: dict = field(default_factory=dict)   # model-specific counters (yields, sidesteps, ...)

    def summary(self) -> dict:
        return dict(
            model=self.model, world=self.world, n_agents=self.n_agents,
            completed=self.completed,
            coverage_time=None if self.coverage_time is None else round(self.coverage_time, 2),
            final_time=round(self.final_time, 2),
            believed_done=[None if t is None else round(t, 2) for t in self.believed_done],
            min_pair_dist=None if not np.isfinite(self.min_pair_dist) else round(self.min_pair_dist, 3),
            min_clearance=round(self.min_clearance, 3),
            collision_events=self.collision_events,
            max_tracking_error=[round(e, 3) for e in self.max_tracking_error],
            path_length=[round(p, 2) for p in self.path_length],
            victims_found=len(self.victims_found),
            model_time_s=round(self.model_time, 3),
            n_decisions=self.n_decisions, n_messages=self.n_messages, **self.extra)

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(self.summary(), indent=2))
        self.log.to_csv(d / "timeseries.csv")
