"""Shortest paths in the planning free space via a visibility graph.

The planning free space is the workspace shrunk by the planning clearance with all
obstacles grown by it, so any path returned keeps a drone centre that far from everything.
Distances are Euclidean path lengths; divide by v_max for a kinematic travel-time estimate.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Point
from shapely.ops import nearest_points

from mascov.geometry.polygons import polygon_parts

INF = float("inf")


class BoundPoints:
    """Destination points with their precomputed visibility to the graph nodes."""

    def __init__(self, pts: np.ndarray, off: np.ndarray, dnodes: np.ndarray):
        self.pts, self.off, self.dnodes = pts, off, dnodes


class PathPlanner:
    def __init__(self, allowed, chunk: int = 32):
        self.allowed = allowed
        self._cover = allowed.buffer(1e-6)          # tolerance: paths may touch the boundary
        shapely.prepare(self._cover)
        self._chunk = chunk
        self.nodes = self._collect_nodes(allowed)
        self.D, self.nxt = self._all_pairs()

    @staticmethod
    def _collect_nodes(allowed) -> np.ndarray:
        pts = []
        for poly in polygon_parts(allowed):
            pts.extend(poly.exterior.coords[:-1])
            for hole in poly.interiors:
                pts.extend(hole.coords[:-1])
        if not pts:
            return np.zeros((0, 2))
        arr = np.round(np.asarray(pts, dtype=float), 9)
        return np.unique(arr, axis=0)

    def _visible_pairs(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Row-wise: is segment a[i]-b[i] inside the allowed region?"""
        coords = np.stack([a, b], axis=1)
        lines = shapely.linestrings(coords)
        return shapely.covers(self._cover, lines)

    def _snap(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Move points that lie outside the planning free space (e.g. a drone pushed into the
        clearance margin by tracking error) to the nearest allowed point.

        Returns (snapped points, snap distances).
        """
        P = np.atleast_2d(np.asarray(P, dtype=float))
        inside = shapely.covers(self._cover, shapely.points(P))
        if inside.all():
            return P, np.zeros(len(P))
        Q = P.copy()
        for k in np.flatnonzero(~inside):
            q = nearest_points(self.allowed, Point(P[k]))[0]
            Q[k] = (q.x, q.y)
        return Q, np.linalg.norm(P - Q, axis=1)

    def _all_pairs(self):
        n = len(self.nodes)
        D = np.full((n, n), INF)
        nxt = np.full((n, n), -1, dtype=int)
        if n == 0:
            return D, nxt
        iu, ju = np.triu_indices(n, k=1)
        vis = self._visible_pairs(self.nodes[iu], self.nodes[ju])
        i, j = iu[vis], ju[vis]
        w = np.linalg.norm(self.nodes[i] - self.nodes[j], axis=1)
        D[i, j] = w
        D[j, i] = w
        nxt[i, j] = j
        nxt[j, i] = i
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(nxt, np.arange(n))
        for k in range(n):                              # Floyd-Warshall with next hops
            alt = D[:, k, None] + D[None, k, :]
            better = alt < D - 1e-12
            D = np.where(better, alt, D)
            nxt = np.where(better, nxt[:, k][:, None], nxt)
        return D, nxt

    def _to_nodes(self, pts: np.ndarray) -> np.ndarray:
        """(len(pts), N) distances to visible nodes, inf where not visible."""
        n = len(self.nodes)
        out = np.full((len(pts), n), INF)
        if n == 0:
            return out
        for s in range(0, len(pts), self._chunk):
            p = pts[s:s + self._chunk]
            a = np.repeat(p, n, axis=0)
            b = np.tile(self.nodes, (len(p), 1))
            vis = self._visible_pairs(a, b).reshape(len(p), n)
            dist = np.linalg.norm(a - b, axis=1).reshape(len(p), n)
            out[s:s + self._chunk] = np.where(vis, dist, INF)
        return out

    def snap(self, p, inset: float = 0.0) -> np.ndarray:
        """Nearest point of the planning free space to p (p itself if already inside).

        inset > 0 additionally pulls points that are within `inset` of the boundary inwards (used so
        a start point is not exactly on the boundary the MILP is about to forbid with a small slack).
        """
        p = np.asarray(p, dtype=float)
        if inset <= 0:
            return self._snap(p)[0][0]
        region = self.allowed.buffer(-inset, join_style="mitre", mitre_limit=5.0)
        if region.is_empty or region.buffer(1e-9).contains(Point(p)):
            return p
        q = nearest_points(region, Point(p))[0]
        return np.array([q.x, q.y])

    def bind(self, B) -> "BoundPoints":
        """Precompute everything about a fixed set of destination points (e.g. all viewpoints),
        so repeated queries from moving drones only pay for the source side."""
        B, offB = self._snap(B)
        dB = self._to_nodes(B)
        return BoundPoints(B, offB, dB)

    def dist_to(self, A, bound: "BoundPoints") -> np.ndarray:
        """Shortest-path distances from each a in A to every bound point."""
        A, offA = self._snap(A)
        B, offB, dB = bound.pts, bound.off, bound.dnodes
        out = np.full((len(A), len(B)), INF)
        if len(self.nodes):
            dA = self._to_nodes(A)
            for s in range(0, len(A), self._chunk):
                E = (dA[s:s + self._chunk, :, None] + self.D[None, :, :]).min(axis=1)
                out[s:s + self._chunk] = (E[:, None, :] + dB[None, :, :]).min(axis=2)
        for s in range(0, len(A), self._chunk):
            a = np.repeat(A[s:s + self._chunk], len(B), axis=0)
            b = np.tile(B, (len(A[s:s + self._chunk]), 1))
            vis = self._visible_pairs(a, b).reshape(-1, len(B))
            eu = np.linalg.norm(a - b, axis=1).reshape(-1, len(B))
            out[s:s + self._chunk] = np.where(vis, eu, out[s:s + self._chunk])
        return out + offA[:, None] + offB[None, :]

    def dist_matrix(self, A, B) -> np.ndarray:
        """Shortest-path distance between every a in A and b in B (inf if unreachable)."""
        return self.dist_to(A, self.bind(B))

    def distance(self, a, b) -> float:
        return float(self.dist_matrix(a, b)[0, 0])

    def path(self, a, b) -> list[np.ndarray] | None:
        """Shortest polyline from a to b, or None if unreachable."""
        a_true = np.asarray(a, dtype=float)
        b_true = np.asarray(b, dtype=float)
        a = self._snap(a_true)[0][0]
        b = self._snap(b_true)[0][0]
        if self._visible_pairs(a[None], b[None])[0]:
            return self._dedupe([a_true, a, b, b_true])
        if len(self.nodes) == 0:
            return None
        dA, dB = self._to_nodes(a[None])[0], self._to_nodes(b[None])[0]
        tot = dA[:, None] + self.D + dB[None, :]
        i, j = np.unravel_index(int(np.argmin(tot)), tot.shape)
        if not np.isfinite(tot[i, j]):
            return None
        seq = [int(i)]
        while seq[-1] != j:
            seq.append(int(self.nxt[seq[-1], j]))
        return self._dedupe([a_true, a] + [self.nodes[k] for k in seq] + [b, b_true])

    @staticmethod
    def _dedupe(pts) -> list[np.ndarray]:
        clean = [np.asarray(pts[0], dtype=float)]
        for p in pts[1:]:
            if np.linalg.norm(p - clean[-1]) > 1e-9:
                clean.append(np.asarray(p, dtype=float))
        return clean
