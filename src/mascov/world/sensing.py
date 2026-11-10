"""Sensing: what a drone at a position sees. Disc of radius R_s, blocked by obstacles."""
from __future__ import annotations

import numpy as np

from mascov.geometry.polygons import densify_boundary
from mascov.geometry.visibility import clear_los_mask


class Sensor:
    def __init__(self, world):
        self.cfg = world.cfg
        self.targets = world.targets
        self.union = world.obstacle_union if world.cfg.use_los else None
        self.victims = world.victims
        self._world = world
        pts, owner = [], []
        for k, poly in enumerate(world.obstacles):
            p = densify_boundary(poly, world.cfg.target_spacing)
            pts.append(p)
            owner.extend([k] * len(p))
        self._obs_pts = np.vstack(pts) if pts else np.zeros((0, 2))
        self._obs_owner = np.asarray(owner, dtype=int)

    def visible_target_ids(self, pos, radius: float | None = None) -> np.ndarray:
        r = self.cfg.r_sense if radius is None else radius
        d = np.linalg.norm(self.targets - pos, axis=1)
        idx = np.flatnonzero(d <= r)
        if idx.size and self.union is not None:
            idx = idx[clear_los_mask(pos, self.targets[idx], self.union)]
        return idx

    def visible_points_mask(self, pos, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        mask = np.linalg.norm(pts - pos, axis=1) <= self.cfg.r_sense
        if mask.any() and self.union is not None:
            sub = np.flatnonzero(mask)
            mask[sub] = clear_los_mask(pos, pts[sub], self.union)
        return mask

    def visible_obstacle_ids(self, pos) -> set[int]:
        """Obstacles of which any boundary point is currently visible."""
        if not len(self._obs_pts):
            return set()
        d = np.linalg.norm(self._obs_pts - pos, axis=1)
        idx = np.flatnonzero(d <= self.cfg.r_sense)
        if not idx.size:
            return set()
        if self.union is not None:
            idx = idx[clear_los_mask(pos, self._obs_pts[idx], self.union, end_shrink=1e-4)]
        return set(int(k) for k in np.unique(self._obs_owner[idx]))
