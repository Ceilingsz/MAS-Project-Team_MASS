"""Scoring functions for CBBA viewpoint allocation."""
from __future__ import annotations

import numpy as np


class CBBAScorer:
    """Computes marginal scores for bidding on viewpoints in CBBA."""

    def __init__(self, world_view, delta: float = 1.0):
        self.w = world_view
        self.vp = world_view.viewpoints
        self.cfg = world_view.cfg
        self.delta = delta
        # Precompute inter-viewpoint distance matrix for fast travel time lookups (known map)
        positions = self.vp.positions
        self.vp_dist = self.w.planner.dist_matrix(positions, positions)
        self.vp_travel = self.vp_dist / self.cfg.v_max

    def get_bundle_covered(self, bundle: list[int]) -> np.ndarray:
        """Returns boolean mask of targets covered by the viewpoints in bundle."""
        if not bundle:
            return np.zeros(len(self.vp.coverable), dtype=bool)
        return np.bitwise_or.reduce(self.vp.sees[bundle], axis=0)

    def marginal_gain(self, v: int, covered: np.ndarray, bundle: list[int] | None = None) -> int:
        """Number of new targets seen by v, not covered by existing belief or bundle."""
        bundle_cov = self.get_bundle_covered(bundle) if bundle else np.zeros_like(covered)
        todo = self.vp.coverable & ~covered & ~bundle_cov
        return int((self.vp.sees[v] & todo).sum())

    def make_validator(self, covered: np.ndarray):
        """Creates a sequential validator closure for validate_bundle.
        Each item in the bundle must add positive marginal gain given earlier items."""
        cum_covered = covered.copy()

        def validator(task_id: int) -> int:
            todo = self.vp.coverable & ~cum_covered
            gain = int((self.vp.sees[task_id] & todo).sum())
            if gain > 0:
                cum_covered[:] = cum_covered | self.vp.sees[task_id]
            return gain

        return validator

    def compute_score(self, v: int, current_pos: np.ndarray, covered: np.ndarray,
                      bundle: list[int], planner=None, bound=None) -> float:
        """Marginal score for appending viewpoint v to bundle."""
        gain = self.marginal_gain(v, covered, bundle)
        if gain <= 0:
            return 0.0

        p = planner if planner is not None else self.w.planner
        b = bound if bound is not None else self.w._vp_bound

        if not bundle:
            # Distance from drone's current position to viewpoint v
            d = p.dist_to(current_pos[None, :], b)[0, v]
            travel_time = d / self.cfg.v_max
        else:
            if planner is None:
                last_vp = bundle[-1]
                travel_time = self.vp_travel[last_vp, v]
            else:
                last_vp = bundle[-1]
                d = p.distance(self.vp.positions[last_vp], self.vp.positions[v])
                travel_time = d / self.cfg.v_max

        if not np.isfinite(travel_time):
            return 0.0

        return float(gain / (travel_time + self.delta))
