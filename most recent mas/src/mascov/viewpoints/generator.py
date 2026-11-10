"""Candidate viewpoints and, for each one, the set of targets it sees."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from mascov.geometry.visibility import clear_los_pairs


@dataclass
class ViewpointSet:
    positions: np.ndarray        # (V, 2) candidate standing positions
    sees: np.ndarray             # (V, M) bool: viewpoint v sees target m
    coverable: np.ndarray        # (M,) bool: at least one viewpoint sees the target

    def gain(self, covered: np.ndarray) -> np.ndarray:
        """Number of still-uncovered coverable targets each viewpoint would reveal."""
        todo = self.coverable & ~covered
        return (self.sees & todo[None, :]).sum(axis=1)

    def targets_seen(self, v: int) -> np.ndarray:
        return np.flatnonzero(self.sees[v])


def build_viewpoints(world) -> ViewpointSet:
    cfg = world.cfg
    s = cfg.viewpoint_spacing
    xs = np.arange(s / 2, cfg.width, s)
    ys = np.arange(s / 2, cfg.height, s)
    gx, gy = np.meshgrid(xs, ys)
    cand = np.c_[gx.ravel(), gy.ravel()]
    ok = shapely.contains_xy(world.reachable, cand[:, 0], cand[:, 1])
    pos = np.vstack([cand[ok], world.starts])
    pos = np.unique(np.round(pos, 9), axis=0)

    T = world.targets
    r_eff = cfg.r_sense - cfg.sense_slack
    d = np.linalg.norm(pos[:, None, :] - T[None, :, :], axis=2)
    vi, ti = np.nonzero(d <= r_eff)
    sees = np.zeros((len(pos), len(T)), dtype=bool)
    union = world.obstacle_union if cfg.use_los else None
    step = 200_000
    for k in range(0, len(vi), step):
        v, t = vi[k:k + step], ti[k:k + step]
        sees[v, t] = clear_los_pairs(pos[v], T[t], union)
    return ViewpointSet(positions=pos, sees=sees, coverable=sees.any(axis=0))
