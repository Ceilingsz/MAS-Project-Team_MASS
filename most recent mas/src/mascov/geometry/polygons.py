"""Polygon helpers. Obstacles are convex shapely Polygons; planners need the H-representation."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

MITRE = dict(join_style="mitre", mitre_limit=5.0)


def rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return shapely.box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def rotated_rect(cx: float, cy: float, w: float, h: float, angle: float) -> Polygon:
    c, s = math.cos(angle), math.sin(angle)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return Polygon([(cx + c * x - s * y, cy + s * x + c * y) for x, y in corners])


def regular_polygon(cx: float, cy: float, radius: float, n: int, phase: float = 0.0) -> Polygon:
    return Polygon([(cx + radius * math.cos(phase + 2 * math.pi * k / n),
                     cy + radius * math.sin(phase + 2 * math.pi * k / n)) for k in range(n)])


def random_convex_polygon(rng: np.random.Generator, cx: float, cy: float,
                          radius: float, n_points: int = 7) -> Polygon:
    """Convex hull of random points around (cx, cy)."""
    ang = np.sort(rng.uniform(0, 2 * math.pi, n_points))
    rad = rng.uniform(0.55, 1.0, n_points) * radius
    pts = np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)]
    hull = MultiPoint(pts).convex_hull
    return hull if isinstance(hull, Polygon) else regular_polygon(cx, cy, radius * 0.6, 4)


def is_convex(poly: Polygon, tol: float = 1e-9) -> bool:
    return abs(poly.convex_hull.area - poly.area) <= tol * max(1.0, poly.area)


def halfspaces(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """H-representation of a convex polygon: {x : H @ x <= b}. Rows are unit outward normals.

    This is the form the STL layer's polytope predicates need.
    """
    if not is_convex(poly):
        raise ValueError("halfspaces() needs a convex polygon; decompose it first")
    ring = np.asarray(orient(poly, 1.0).exterior.coords)[:-1]       # counter-clockwise
    q = np.roll(ring, -1, axis=0)
    edge = q - ring
    normal = np.c_[edge[:, 1], -edge[:, 0]]                         # outward for CCW
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    b = np.einsum("ij,ij->i", normal, ring)
    return normal, b


def inflate(geom, r: float):
    """Grow a geometry by r with sharp (mitre) corners, so vertices stay vertices."""
    return geom.buffer(r, **MITRE) if r > 0 else geom


def union(polys: Iterable[Polygon]):
    polys = list(polys)
    return unary_union(polys) if polys else None


def densify_boundary(poly: Polygon, spacing: float) -> np.ndarray:
    """Points along a polygon's boundary, at most `spacing` apart (vertices included)."""
    ring = poly.exterior
    n = max(4, int(math.ceil(ring.length / spacing)))
    pts = [ring.interpolate(k * ring.length / n).coords[0] for k in range(n)]
    return np.asarray(pts, dtype=float)


def polygon_parts(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    return [g for g in shapely.get_parts(geom) if isinstance(g, Polygon) and not g.is_empty]
