"""Greedy viewpoint selection shared by all heuristic baselines."""
from __future__ import annotations

import numpy as np


def choose_viewpoint(vp, travel_s: np.ndarray, covered: np.ndarray, exclude=(), delta: float = 1.0,
                     rng: np.random.Generator | None = None,
                     mask: np.ndarray | None = None) -> int | None:
    """Pick argmax of  (new targets seen) / (travel time + delta).

    covered : targets that no longer count (covered, or claimed by someone else)
    mask    : if given, only targets inside the mask count (a drone's own region)
    exclude : viewpoint indices that may not be chosen
    rng     : if given, exact ties are broken at random (otherwise the lowest index wins)
    """
    todo = vp.coverable & ~covered
    if mask is not None:
        todo = todo & mask
    gain = (vp.sees & todo[None, :]).sum(axis=1).astype(float)
    ex = list(exclude)
    if ex:
        gain[ex] = 0.0
    ok = np.isfinite(travel_s) & (gain > 0)
    if not ok.any():
        return None
    score = np.where(ok, gain / (travel_s + delta), -1.0)
    best = score.max()
    ties = np.flatnonzero(score >= best * (1 - 1e-9))
    return int(rng.choice(ties)) if rng is not None else int(ties[0])


def remaining_gain(vp, v: int, covered: np.ndarray, mask: np.ndarray | None = None) -> int:
    todo = vp.coverable & ~covered
    if mask is not None:
        todo = todo & mask
    return int((vp.sees[v] & todo).sum())
