"""Line-of-sight tests against obstacle polygons.

Convention: a segment is blocked only if it passes through the *interior* of an obstacle.
Grazing a corner or running along an edge does not block.
"""
from __future__ import annotations

import numpy as np
import shapely


def prepared_union(obstacles) -> "shapely.Geometry | None":
    from shapely.ops import unary_union
    obstacles = list(obstacles)
    if not obstacles:
        return None
    u = unary_union(obstacles)
    shapely.prepare(u)
    return u


def clear_los_pairs(a: np.ndarray, b: np.ndarray, union, end_shrink: float = 0.0) -> np.ndarray:
    """Vectorised LOS between a[i] and b[i]. Returns a bool array (True = clear).

    end_shrink moves each end point towards its start by that distance. Use a small value
    (1e-4) when b[i] lies *on* an obstacle boundary, so rounding cannot push it inside.
    """
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)
    n = len(a)
    clear = np.ones(n, dtype=bool)
    if union is None or union.is_empty or n == 0:
        return clear
    d = b - a
    length = np.linalg.norm(d, axis=1)
    live = length > 1e-9
    if not live.any():
        return clear
    a_l, b_l, d_l, len_l = a[live], b[live], d[live], length[live]
    if end_shrink > 0:
        b_l = b_l - d_l / len_l[:, None] * np.minimum(end_shrink, 0.5 * len_l)[:, None]
    coords = np.stack([a_l, b_l], axis=1)                    # (m, 2, 2)
    lines = shapely.linestrings(coords)
    blocked = shapely.intersects(union, lines) & ~shapely.touches(union, lines)
    clear[np.flatnonzero(live)] = ~blocked
    return clear


def clear_los_mask(origin, pts, union, end_shrink: float = 0.0) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    origin = np.broadcast_to(np.asarray(origin, dtype=float), pts.shape)
    return clear_los_pairs(origin, pts, union, end_shrink)
