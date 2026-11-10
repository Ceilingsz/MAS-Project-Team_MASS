"""Choosing a standing position from per-spot gain and travel time (shared by the explorer models)."""
from __future__ import annotations

import numpy as np


def choose_spot(gain: np.ndarray, travel_s: np.ndarray, usable: np.ndarray, delta: float,
                rng: np.random.Generator | None = None) -> int | None:
    """argmax of gain / (travel time + delta) over usable, reachable spots with gain > 0; exact ties at random."""
    ok = usable & np.isfinite(travel_s) & (gain > 0)
    if not ok.any():
        return None
    score = np.where(ok, gain / (travel_s + delta), -1.0)
    ties = np.flatnonzero(score >= score.max() * (1 - 1e-9))
    return int(rng.choice(ties)) if rng is not None else int(ties[0])
