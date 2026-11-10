"""What one agent believes. Coverage is permanent, so merging is a set union."""
from __future__ import annotations

import numpy as np


class AgentBelief:
    def __init__(self, n_targets: int):
        self.covered = np.zeros(n_targets, dtype=bool)
        self.known_obstacles: set[int] = set()

    def mark(self, ids) -> int:
        """Mark targets covered; return how many were new."""
        ids = np.asarray(ids, dtype=int)
        if ids.size == 0:
            return 0
        new = int((~self.covered[ids]).sum())
        self.covered[ids] = True
        return new

    def merge(self, other: "AgentBelief") -> int:
        """Set-union merge (commutative, idempotent). Returns number of new covered targets."""
        new = int((other.covered & ~self.covered).sum())
        self.covered |= other.covered
        self.known_obstacles |= other.known_obstacles
        return new

    def copy(self) -> "AgentBelief":
        b = AgentBelief(len(self.covered))
        b.covered = self.covered.copy()
        b.known_obstacles = set(self.known_obstacles)
        return b

    def fraction(self, mask: np.ndarray | None = None) -> float:
        if mask is None:
            return float(self.covered.mean()) if self.covered.size else 1.0
        n = int(mask.sum())
        return float(self.covered[mask].sum() / n) if n else 1.0


UNKNOWN, FREE, OBSTACLE = 0, 1, 2


class MapBelief:
    """Belief for the UNKNOWN-MAP interface: a state per lattice cell plus the obstacles seen so far.

    Cell states only ever increase (UNKNOWN -> FREE or OBSTACLE), so merging two beliefs is the
    element-wise maximum: commutative, associative and idempotent, exactly like the set-union merge of
    ``AgentBelief``. Gossip therefore converges, and the only cost of limited communication is staleness.

    * FREE     : the cell was seen, so it is free space and counts as covered
    * OBSTACLE : the cell lies inside an obstacle this agent (or someone it merged with) has seen
    * UNKNOWN  : not yet resolved

    Sensing an obstacle reveals its whole polygon (``Simulator`` calls ``add_obstacle``). Obstacle ids
    are global labels, so beliefs from different agents merge by id.
    """

    def __init__(self, lattice: np.ndarray):
        self.lattice = lattice                          # (G, 2), shared and read-only
        self.state = np.zeros(len(lattice), dtype=np.int8)
        self.obstacles: dict[int, object] = {}          # id -> shapely Polygon

    # ---------------------------------------------------------------- updates
    def mark_free(self, cells) -> int:
        """Mark cells as seen free space; return how many were new."""
        cells = np.asarray(cells, dtype=int)
        if cells.size == 0:
            return 0
        new = int((self.state[cells] == UNKNOWN).sum())
        self.state[cells] = np.maximum(self.state[cells], FREE)
        return new

    def add_obstacle(self, oid: int, poly) -> int:
        """Record an obstacle polygon and mark the cells inside it. Returns the number of new obstacle cells."""
        if oid in self.obstacles:
            return 0
        self.obstacles[oid] = poly
        import shapely
        inside = np.flatnonzero(shapely.contains_xy(poly, self.lattice[:, 0], self.lattice[:, 1]))
        new = int((self.state[inside] != OBSTACLE).sum())
        self.state[inside] = OBSTACLE
        return new

    def merge(self, other: "MapBelief") -> int:
        """Union merge. Returns the number of cells that became resolved (UNKNOWN -> FREE/OBSTACLE)."""
        new = int(((self.state == UNKNOWN) & (other.state != UNKNOWN)).sum())
        np.maximum(self.state, other.state, out=self.state)
        for k, poly in other.obstacles.items():
            self.obstacles.setdefault(k, poly)
        return new

    def copy(self) -> "MapBelief":
        b = MapBelief(self.lattice)
        b.state = self.state.copy()
        b.obstacles = dict(self.obstacles)              # polygons are immutable, so sharing them is safe
        return b

    # ---------------------------------------------------------------- queries
    @property
    def known_obstacles(self) -> set[int]:
        """API parity with ``AgentBelief``."""
        return set(self.obstacles)

    @property
    def unknown(self) -> np.ndarray:
        return self.state == UNKNOWN

    @property
    def free(self) -> np.ndarray:
        return self.state == FREE

    def n_unknown(self) -> int:
        return int((self.state == UNKNOWN).sum())

    def fraction_free(self, cells: np.ndarray) -> float:
        """Fraction of the given cell ids that are known free (i.e. covered)."""
        return float((self.state[cells] == FREE).mean()) if len(cells) else 1.0
