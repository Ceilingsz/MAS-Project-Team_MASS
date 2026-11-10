"""Distributed model predictive control (D-MPC) for single-integrator drones: one drone's local problem.

Every decision, each drone solves its OWN small quadratic program over a horizon of H steps of length dt:

    variables   p_1 .. p_H                       predicted positions (p_0 = current position, fixed)
                s >= 0                           one slack per soft constraint row
    minimise    sum_k w_k |p_k - r_k|^2          follow the reference r_k (heavier weight on the last step)
              + lam sum_k |u_k - u_{k-1}|^2      smooth velocities, u_k = (p_{k+1} - p_k) / dt, u_{-1} = current vel
              + rho1 sum s + rho2 sum s^2        soft-constraint penalty
    subject to  d_m . u_k <= v_cap cos(pi/8)     8 directions: an octagon inside the speed disc (hard)
                walls:      clearance <= p_k <= size - clearance                         (soft)
                obstacles:  n_ok . (p_k - c_ok) >= clearance, one line per obstacle and step (soft)
                neighbours: n_jk . (p_k - q_jk) >= d_min                                 (soft, coupling only)

Obstacle line: c_ok is the point of the (convex) obstacle closest to last step's prediction for step k, n_ok
the unit vector from c_ok to that prediction. The line through c_ok with normal n_ok supports the obstacle, so
the half-plane is a valid outer bound, and it is the one with the most room on the drone's side. (Using the
face the prediction is furthest outside of instead is NOT safe to use: past an obstacle's corner the extended face
line can run right through the drone even when the obstacle is metres away, and the drone then cannot move
towards its goal. That bug stalled drones in cluttered maps.) If the linearisation point is inside the obstacle,
the face it is least inside of is used. Neighbour constraint: q_jk is neighbour
j's predicted position at the same time (from its last broadcast, or a constant-velocity guess when no
prediction has arrived), n_jk the unit vector from q_jk to this drone's own previous prediction. This is
the linearised separation constraint of the usual D-MPC schemes (e.g. Luis & Schoellig 2019). Both drones
apply it, so each moves away from the other's plan; the guarantee is only "separation from the PREDICTED
neighbour", not a proof of collision freedom, so measure collisions (and add BVC as a backstop if needed).

Only the speed limit is hard, so the problem is always feasible (standing still satisfies it).
Solved with OSQP. ``DMPCSolver`` builds the problem sparse and keeps one OSQP object per drone, updating
only values between steps instead of setting up a new solver every step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import osqp
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - checked by dmpc_available()
    osqp = None
    sp = None


def dmpc_available() -> bool:
    return osqp is not None


_DIRS = np.array([[math.cos(a), math.sin(a)] for a in np.arange(8) * (math.pi / 4)])
_OCT = math.cos(math.pi / 8)
_WALL_A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


@dataclass
class DMPCParams:
    horizon: int = 10            # H steps
    dt: float = 0.1
    v_cap: float = 0.95          # speed limit inside the MPC [m/s]
    clearance: float = 0.45      # drone centre to obstacle / wall [m]
    d_min: float = 0.4           # drone centre to drone centre [m]
    w_track: float = 1.0
    w_terminal: float = 5.0
    w_smooth: float = 0.02
    rho1: float = 1e3
    rho2: float = 1e4
    obstacle_reach: float = 1.5  # obstacles further than (H dt v_cap + this) are ignored
    width: float = 12.0
    height: float = 12.0


@dataclass
class DMPCResult:
    points: np.ndarray           # (H+1, 2), points[0] = current position
    status: str
    slack: float                 # largest slack used (0: every soft constraint met)
    n_constraints: int = 0
    info: dict = field(default_factory=dict)


@dataclass
class ObstacleHS:
    """A convex obstacle as unit outward normals H and offsets b: inside = {x : H x < b}."""
    H: np.ndarray
    b: np.ndarray
    centre: np.ndarray
    radius: float                # circumradius about centre (for the reach test)
    verts: np.ndarray            # (F, 2) vertices, counter-clockwise; edge f runs verts[f] -> verts[f+1]


def closest_points(ob: ObstacleHS, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point in X (n, 2): the closest point of the convex polygon's boundary and the distance."""
    A = ob.verts
    B = np.roll(A, -1, axis=0)
    E = B - A                                                 # (F, 2)
    L2 = np.maximum((E * E).sum(axis=1), 1e-18)
    t = np.clip(((X[:, None, :] - A[None]) * E[None]).sum(axis=2) / L2[None], 0.0, 1.0)   # (n, F)
    C = A[None] + t[..., None] * E[None]                      # (n, F, 2)
    d = np.linalg.norm(X[:, None, :] - C, axis=2)
    j = np.argmin(d, axis=1)
    idx = np.arange(len(X))
    return C[idx, j], d[idx, j]


def obstacle_halfspaces(polys) -> list[ObstacleHS]:
    from mascov.geometry.polygons import halfspaces
    out = []
    for poly in polys:
        H, b = halfspaces(poly)
        from shapely.geometry.polygon import orient
        ring = np.asarray(orient(poly, 1.0).exterior.coords)[:-1]
        c = ring.mean(axis=0)
        out.append(ObstacleHS(H, b, c, float(np.linalg.norm(ring - c, axis=1).max()), ring))
    return out


def _soft_rows(p0, ref, prev, obstacles, neighbours, prm):
    """Every soft constraint as (step k, normal a, lower bound lo): a . p_k + s >= lo. Vectorised over steps."""
    H, dt, clr = prm.horizon, prm.dt, prm.clearance
    lin = ref if prev is None else prev
    # neighbours: with no previous solution linearise at the current position, not at the reference (a
    # reference that passes through a neighbour would put the constraint on the neighbour's far side)
    lin_n = np.repeat(p0[None], H, axis=0) if prev is None else prev
    steps = np.arange(H)
    # every group is one block of H rows, row i of a block at step i (DMPCSolver's pattern relies on it)
    K = [np.tile(steps, 4)]
    A = [np.repeat(_WALL_A, H, axis=0)]
    L = [np.repeat([clr, -(prm.width - clr), clr, -(prm.height - clr)], H)]

    reach = H * dt * prm.v_cap + prm.obstacle_reach
    n_obs = 0
    for ob in obstacles:
        if np.linalg.norm(ob.centre - p0) > reach + ob.radius:
            continue
        n_obs += 1
        sd = lin @ ob.H.T - ob.b[None, :]            # (H, F) signed distance to each face's line
        C, dist = closest_points(ob, lin)
        n = (lin - C) / np.maximum(dist, 1e-12)[:, None]   # supporting line through the closest point
        lo = (n * C).sum(axis=1) + clr
        bad = (sd.max(axis=1) <= 1e-9) | (dist < 1e-9)    # inside (or on) the obstacle: push out through the nearest face
        if bad.any():
            f = np.argmax(sd[bad], axis=1)
            n[bad] = ob.H[f]
            lo[bad] = ob.b[f] + clr
        K.append(steps)
        A.append(n)
        L.append(lo)

    n_nbr = 0
    for q in neighbours:
        q = np.asarray(q, float)
        if np.min(np.linalg.norm(q - p0, axis=1)) > 2 * H * dt * prm.v_cap + prm.d_min + 0.5:
            continue                                  # cannot meet within the horizon
        n_nbr += 1
        d = lin_n - q
        nd = np.linalg.norm(d, axis=1)
        tiny = nd < 1e-6
        if tiny.any():                                # fall back to the current offset, then to +x
            d[tiny] = p0[None] - q[tiny]
            nd[tiny] = np.linalg.norm(d[tiny], axis=1)
            still = nd < 1e-6
            d[still] = (1.0, 0.0)
            nd[still] = 1.0
        n = d / nd[:, None]
        K.append(steps)
        A.append(n)
        L.append((n * q).sum(axis=1) + prm.d_min)
    return np.concatenate(K), np.vstack(A), np.concatenate(L), n_obs, n_nbr


class DMPCSolver:
    """One drone's D-MPC problem, built sparse and solved by ONE OSQP object reused between steps.

    The sparsity pattern is fixed by a capacity of soft constraint rows (walls + obstacles + neighbours, in
    blocks of H rows, row r at step r mod H, so each row has 3 entries). Unused rows get zero coefficients and a lower bound of -inf, so they never bind and their
    slack stays 0. Each step then only updates values (q, A, bounds) and warm starts, instead of building and
    setting up a new solver. The cost matrix P never changes. If a step needs more rows than the capacity, the
    capacity grows and the solver is set up again.

    It solves the same QP as a fresh setup; results agree to the solver tolerance (tests/test_dmpc.py).
    """

    def __init__(self, prm: "DMPCParams", capacity: int = 12):
        self.prm = prm
        H, dt = prm.horizon, prm.dt
        self.n_p = 2 * H
        w = np.full(H, prm.w_track)
        w[-1] = prm.w_terminal
        self.w2 = np.repeat(2 * w, 2)                  # 2 x tracking weight per coordinate
        # velocity u = D p + d0 and its change du = E u - e0, so du = G p + g
        D = (np.eye(self.n_p) - np.eye(self.n_p, k=-2)) / dt
        self.E = np.eye(self.n_p) - np.eye(self.n_p, k=-2)
        self.G = self.E @ D
        self.Ppp = np.diag(self.w2) + 2 * prm.w_smooth * self.G.T @ self.G
        # speed: d_m . (p_k - p_{k-1}) <= vlim, 8 directions per step (p_{-1} = p0 moves into the bound)
        rows = []
        for k in range(H):
            for dm in _DIRS:
                a = np.zeros(self.n_p)
                a[2 * k:2 * k + 2] = dm
                if k:
                    a[2 * (k - 1):2 * k] = -dm
                rows.append(a)
        self.speed = sp.coo_matrix(np.asarray(rows))
        self.speed_hi = np.full(8 * H, prm.v_cap * _OCT * dt)
        self.n_setups = 0
        self._setup(max(capacity, 4) * H)

    def _setup(self, rows: int) -> None:
        """Fix the pattern for `rows` soft rows and set up OSQP once."""
        n_p, prm = self.n_p, self.prm
        S = self.rows = rows
        self.nx = n_p + S
        P = sp.block_diag((sp.csc_matrix(np.triu(self.Ppp)), 2 * prm.rho2 * sp.eye(S)), format="csc")
        # A = [speed rows | 0], soft rows (row r at step r mod H: its two position columns + its slack), then
        # s >= 0. Entry positions in the CSC data array are found once with distinct marker values, so the
        # pattern keeps its zeros and every later update writes values in the right order.
        r = np.arange(S)
        k = r % self.prm.horizon
        m_speed = self.speed.shape[0]
        row = np.concatenate([self.speed.row, m_speed + r, m_speed + r, m_speed + r, m_speed + S + r])
        col = np.concatenate([self.speed.col, 2 * k, 2 * k + 1, n_p + r, n_p + r])
        self.m = m_speed + 2 * S
        mark = sp.csc_matrix((np.arange(1, len(row) + 1, dtype=float), (row, col)), shape=(self.m, self.nx))
        mark.sort_indices()
        self.perm = mark.data.astype(int) - 1
        A = mark.copy()
        A.data = self._values(np.zeros(0, dtype=int), np.zeros((0, 2)))
        self.solver = osqp.OSQP(algebra="builtin")     # explicit: the default probes for GPU/MKL backends by import
        self.solver.setup(P, np.zeros(self.nx), A, np.full(self.m, -np.inf), np.full(self.m, np.inf),
                          verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=8000, polishing=True)
        self.n_setups += 1

    def _values(self, K: np.ndarray, Aa: np.ndarray) -> np.ndarray:
        """A's values in CSC order for soft rows at steps K with normals Aa (rows beyond len(K) are zero)."""
        S = self.rows
        assert np.array_equal(K, np.arange(len(K)) % self.prm.horizon), "soft rows must come in step-ordered blocks"
        a0 = np.zeros(S)
        a1 = np.zeros(S)
        a0[:len(K)] = Aa[:, 0]
        a1[:len(K)] = Aa[:, 1]
        vals = np.concatenate([self.speed.data, a0, a1, np.ones(S), np.ones(S)])
        return vals[self.perm]

    def solve(self, p0, v0, ref, prev, obstacles, neighbours) -> "DMPCResult":
        """One drone's D-MPC step.

        p0, v0      current position and velocity
        ref         (H, 2) reference positions for steps 1..H
        prev        (H, 2) this drone's previous prediction shifted to steps 1..H (linearisation point), or None
        neighbours  list of (H, 2) predicted neighbour positions for steps 1..H
        """
        prm, H, n_p = self.prm, self.prm.horizon, self.n_p
        p0 = np.asarray(p0, float)
        ref = np.asarray(ref, float)
        K, Aa, Lo, n_obs, n_nbr = _soft_rows(p0, ref, prev, obstacles, neighbours, prm)
        if len(K) > self.rows:
            self._setup(int(math.ceil(len(K) / H)) * H + 4 * H)
        S = self.rows

        d0 = np.zeros(n_p)
        d0[:2] = -p0 / prm.dt
        e0 = np.zeros(n_p)
        e0[:2] = np.asarray(v0, float)
        g = self.E @ d0 - e0
        q = np.empty(self.nx)
        q[:n_p] = -self.w2 * ref.ravel() + 2 * prm.w_smooth * self.G.T @ g
        q[n_p:] = prm.rho1

        hi_speed = self.speed_hi.copy()
        hi_speed[:8] += _DIRS @ p0                     # step 0 is measured from the current position
        lo = np.concatenate([np.full(8 * H, -np.inf), np.full(S, -np.inf), np.zeros(S)])
        lo[8 * H:8 * H + len(K)] = Lo
        hi = np.concatenate([hi_speed, np.full(2 * S, np.inf)])

        self.solver.update(q=q, l=lo, u=hi, Ax=self._values(K, Aa))
        x0 = np.zeros(self.nx)
        x0[:n_p] = (prev if prev is not None else np.repeat(p0[None], H, axis=0)).ravel()
        self.solver.warm_start(x=x0, y=np.zeros(self.m))
        res = self.solver.solve(raise_error=False)
        status = str(res.info.status)
        n_con = 8 * H + 2 * len(K)
        ok = res.x is not None and np.all(np.isfinite(res.x)) and "solved" in status.lower()
        if not ok:
            pts = np.repeat(p0[None], H + 1, axis=0)   # hold: always within the speed limit
            return DMPCResult(pts, status, math.inf, n_con, dict(n_obs=n_obs, n_nbr=n_nbr, failed=True))
        x = res.x
        P = np.vstack([p0[None], x[:n_p].reshape(H, 2)])
        slack = float(x[n_p:n_p + len(K)].max()) if len(K) else 0.0
        return DMPCResult(P, status, max(0.0, slack), n_con, dict(n_obs=n_obs, n_nbr=n_nbr, failed=False))


def solve_dmpc(p0: np.ndarray, v0: np.ndarray, ref: np.ndarray, prev: np.ndarray | None,
               obstacles: list[ObstacleHS], neighbours: list[np.ndarray], prm: DMPCParams) -> DMPCResult:
    """One-off D-MPC step with a fresh solver (convenient in tests; a drone keeps one DMPCSolver instead)."""
    return DMPCSolver(prm).solve(p0, v0, ref, prev, obstacles, neighbours)


def sample_polyline(path: list[np.ndarray] | np.ndarray, s: np.ndarray) -> np.ndarray:
    """Points at arc lengths s along a polyline (clamped to its ends)."""
    P = np.asarray(path, float).reshape(-1, 2)
    if len(P) == 1:
        return np.repeat(P, len(s), axis=0)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.clip(s, 0.0, cum[-1])
    return np.c_[np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])]


def project_on_polyline(path: np.ndarray, p: np.ndarray, s_lo: float = 0.0, s_hi: float = math.inf) -> float:
    """Arc length of the point of the polyline closest to p, searching only arc lengths in [s_lo, s_hi]."""
    P = np.asarray(path, float).reshape(-1, 2)
    if len(P) == 1:
        return 0.0
    seg = np.diff(P, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    best_s, best_d = 0.0, math.inf
    for i in range(len(seg)):
        if cum[i + 1] < s_lo or cum[i] > s_hi or L[i] < 1e-12:
            continue
        t = float(np.clip((p - P[i]) @ seg[i] / L[i] ** 2, 0.0, 1.0))
        s = float(np.clip(cum[i] + t * L[i], s_lo, s_hi))
        pt = sample_polyline(P, np.array([s]))[0]
        d = float(np.linalg.norm(pt - p))
        if d < best_d:
            best_s, best_d = s, d
    return best_s
