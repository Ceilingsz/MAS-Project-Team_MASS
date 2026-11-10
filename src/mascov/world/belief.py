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
