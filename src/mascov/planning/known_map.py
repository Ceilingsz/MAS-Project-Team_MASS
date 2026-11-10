"""Planning over the part of the map an agent KNOWS.

Unknown space is treated as free (optimism): a route may cross a region that later turns out to hold an
obstacle, and the agent re-plans when it discovers it. The planning free space is built exactly as in
``World.allowed`` (walls shrunk, obstacles grown by ``cfg.clearance``) but only from known obstacles.

A ``KnownSpace`` is a pure function of the set of known obstacles, so it is cached and shared between
agents that happen to know the same set. Sharing the cache reveals nothing: a key exists only if some agent
legitimately knew that set.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import shapely

from mascov.geometry.pathing import PathPlanner
from mascov.geometry.polygons import inflate, rect
from mascov.geometry.visibility import clear_los_pairs, prepared_union


class KnownSpace:
    """Planner, candidate-position data and line-of-sight tables for one set of known obstacles."""

    def __init__(self, view, obstacles: dict):
        cfg, c = view.cfg, view.cfg.clearance
        self.ids = frozenset(obstacles)
        inner = rect(c, c, cfg.width - c, cfg.height - c)
        self.union = prepared_union(obstacles.values())
        self.allowed = inner if self.union is None else inner.difference(inflate(self.union, c))
        self.planner = PathPlanner(self.allowed)
        cand = view.candidates
        self.valid = shapely.contains_xy(self.allowed, cand[:, 0], cand[:, 1])   # standing position is free as far as known
        self.bound = self.planner.bind(cand)                                     # for distances from a moving drone
        # (candidate, cell) pairs within sensing range; LOS is evaluated against the KNOWN obstacles only
        r_eff = cfg.r_sense - cfg.sense_slack
        d = np.linalg.norm(cand[:, None, :] - view.lattice[None, :, :], axis=2)
        self.pair_v, self.pair_c = np.nonzero(d <= r_eff)
        los_union = self.union if cfg.use_los else None
        self.pair_clear = clear_los_pairs(cand[self.pair_v], view.lattice[self.pair_c], los_union)
        self.n_candidates = len(cand)

    def gains(self, unknown: np.ndarray) -> np.ndarray:
        """Per candidate: number of UNKNOWN cells it would reveal (occlusion by known obstacles respected)."""
        hit = unknown[self.pair_c] & self.pair_clear
        return np.bincount(self.pair_v, weights=hit, minlength=self.n_candidates)

    def distances(self, pos) -> np.ndarray:
        """Shortest-path distance from pos to every candidate (inf if unreachable given what is known)."""
        return self.planner.dist_to(np.asarray(pos, dtype=float), self.bound)[0]


class KnownSpaceCache:
    def __init__(self, view, max_size: int = 64):
        self.view, self.max_size = view, max_size
        self._d: OrderedDict[frozenset, KnownSpace] = OrderedDict()

    def get(self, obstacles: dict) -> KnownSpace:
        key = frozenset(obstacles)
        sp = self._d.get(key)
        if sp is None:
            sp = KnownSpace(self.view, dict(obstacles))
            self._d[key] = sp
            if len(self._d) > self.max_size:
                self._d.popitem(last=False)
        else:
            self._d.move_to_end(key)
        return sp
