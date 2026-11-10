"""A minimal mixed-integer linear program builder solved with HiGHS.

HiGHS (https://highs.dev) is an open-source MILP solver with no size limit and no licence. It is reached
through ``scipy.optimize.milp`` (scipy >= 1.9 bundles HiGHS), so ``pip install scipy`` is the only
requirement.

    m = MILP()
    x = m.var(0, 10, cost=1.0)           # continuous
    z = m.bin()                          # binary
    m.constr([(x, 1.0), (z, -5.0)], lo=0.0)       # 0 <= x - 5 z
    res = m.solve(time_limit=10, mip_gap=1e-2)
    res.x[x]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

INF = math.inf


def highs_available() -> bool:
    try:
        from scipy.optimize import milp  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class MILPResult:
    x: np.ndarray | None          # a feasible solution if one was found (optimal, or best at the time limit)
    optimal: bool
    status: int                   # scipy.optimize.milp status: 0 optimal, 1 limit reached, 2 infeasible, 3 unbounded, 4 other
    message: str


class MILP:
    def __init__(self):
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integer: list[int] = []
        self.cost: list[float] = []
        self._row: list[int] = []
        self._col: list[int] = []
        self._val: list[float] = []
        self._lo: list[float] = []
        self._hi: list[float] = []

    # ------------------------------------------------------------------ building
    def var(self, lb: float = 0.0, ub: float = INF, *, integer: bool = False, cost: float = 0.0) -> int:
        self.lb.append(float(lb)), self.ub.append(float(ub)), self.integer.append(int(integer)), self.cost.append(float(cost))
        return len(self.lb) - 1

    def bin(self, cost: float = 0.0) -> int:
        return self.var(0.0, 1.0, integer=True, cost=cost)

    def constr(self, terms, lo: float = -INF, hi: float = INF) -> None:
        """lo <= sum(coef * var) <= hi. ``terms`` is an iterable of (variable index, coefficient)."""
        r = len(self._lo)
        for j, a in terms:
            self._row.append(r), self._col.append(j), self._val.append(float(a))
        self._lo.append(float(lo)), self._hi.append(float(hi))

    @property
    def n_vars(self) -> int:
        return len(self.lb)

    @property
    def n_binaries(self) -> int:
        return int(sum(self.integer))

    @property
    def n_constraints(self) -> int:
        return len(self._lo)

    # ------------------------------------------------------------------ solving
    def solve(self, time_limit: float = 20.0, mip_gap: float = 1e-2) -> MILPResult:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        n, m = self.n_vars, self.n_constraints
        constraints = []
        if m:
            A = coo_matrix((self._val, (self._row, self._col)), shape=(m, n)).tocsr()
            constraints = [LinearConstraint(A, np.array(self._lo), np.array(self._hi))]
        res = milp(c=np.array(self.cost), constraints=constraints, integrality=np.array(self.integer),
                   bounds=Bounds(np.array(self.lb), np.array(self.ub)),
                   options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_gap), "disp": False})
        x = None if res.x is None else np.asarray(res.x, dtype=float)
        return MILPResult(x, bool(res.status == 0), int(res.status), str(res.message))
