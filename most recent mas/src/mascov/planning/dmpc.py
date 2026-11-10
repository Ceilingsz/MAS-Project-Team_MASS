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
Solved with OSQP.
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


def solve_dmpc(p0: np.ndarray, v0: np.ndarray, ref: np.ndarray, prev: np.ndarray | None,
               obstacles: list[ObstacleHS], neighbours: list[np.ndarray], prm: DMPCParams) -> DMPCResult:
    """One drone's D-MPC step.

    p0, v0      current position and velocity
    ref         (H, 2) reference positions for steps 1..H
    prev        (H, 2) this drone's previous prediction shifted to steps 1..H (linearisation point), or None
    neighbours  list of (H, 2) predicted neighbour positions for steps 1..H
    """
    H, dt = prm.horizon, prm.dt
    p0 = np.asarray(p0, float)
    lin = ref if prev is None else prev
    # neighbours: with no previous solution linearise at the current position, not at the reference (a
    # reference that passes through a neighbour would put the constraint on the neighbour's far side)
    lin_n = np.repeat(p0[None], H, axis=0) if prev is None else prev
    n_p = 2 * H

    # ---- soft rows: a . p_k >= lo  ------------------------------------------------------------------
    rows_k, rows_a, rows_lo = [], [], []

    def soft(k, a, lo):
        rows_k.append(k)
        rows_a.append(a)
        rows_lo.append(lo)

    clr = prm.clearance
    for k in range(H):
        soft(k, (1.0, 0.0), clr)
        soft(k, (-1.0, 0.0), -(prm.width - clr))
        soft(k, (0.0, 1.0), clr)
        soft(k, (0.0, -1.0), -(prm.height - clr))

    reach = H * dt * prm.v_cap + prm.obstacle_reach
    n_obs = 0
    for ob in obstacles:
        if np.linalg.norm(ob.centre - p0) > reach + ob.radius:
            continue
        n_obs += 1
        sd = lin @ ob.H.T - ob.b[None, :]            # (H, F) signed distance to each face's line
        inside = sd.max(axis=1) <= 1e-9
        C, dist = closest_points(ob, lin)
        for k in range(H):
            if inside[k] or dist[k] < 1e-9:
                f = int(np.argmax(sd[k]))                # inside (or on) the obstacle: push out through the nearest face
                soft(k, ob.H[f], ob.b[f] + clr)
            else:
                n = (lin[k] - C[k]) / dist[k]            # supporting line through the closest point
                soft(k, n, float(n @ C[k]) + clr)

    n_nbr = 0
    for q in neighbours:
        q = np.asarray(q, float)
        if np.min(np.linalg.norm(q - p0, axis=1)) > 2 * H * dt * prm.v_cap + prm.d_min + 0.5:
            continue                                  # cannot meet within the horizon
        n_nbr += 1
        for k in range(H):
            d = lin_n[k] - q[k]
            nd = float(np.linalg.norm(d))
            if nd < 1e-6:
                d = p0 - q[k]
                nd = float(np.linalg.norm(d))
            if nd < 1e-6:
                d, nd = np.array([1.0, 0.0]), 1.0
            n = d / nd
            soft(k, n, float(n @ q[k]) + prm.d_min)

    S = len(rows_k)
    nx = n_p + S

    # ---- cost ---------------------------------------------------------------------------------------
    w = np.full(H, prm.w_track)
    w[-1] = prm.w_terminal
    Pd = np.zeros((nx, nx))
    q = np.zeros(nx)
    for k in range(H):
        Pd[2 * k:2 * k + 2, 2 * k:2 * k + 2] += 2 * w[k] * np.eye(2)
        q[2 * k:2 * k + 2] += -2 * w[k] * ref[k]
    # velocity u = Dp + d0 (2H x 2H)
    D = np.zeros((n_p, n_p))
    d0 = np.zeros(n_p)
    for k in range(H):
        D[2 * k:2 * k + 2, 2 * k:2 * k + 2] += np.eye(2) / dt
        if k == 0:
            d0[0:2] = -p0 / dt
        else:
            D[2 * k:2 * k + 2, 2 * (k - 1):2 * k] -= np.eye(2) / dt
    # delta u = E u - e0
    E = np.eye(n_p)
    for k in range(1, H):
        E[2 * k:2 * k + 2, 2 * (k - 1):2 * k] -= np.eye(2)
    e0 = np.zeros(n_p)
    e0[0:2] = np.asarray(v0, float)
    G = E @ D
    g = E @ d0 - e0
    Pd[:n_p, :n_p] += 2 * prm.w_smooth * G.T @ G
    q[:n_p] += 2 * prm.w_smooth * G.T @ g
    if S:
        Pd[n_p:, n_p:] += 2 * prm.rho2 * np.eye(S)
        q[n_p:] += prm.rho1

    # ---- constraints l <= A x <= u ------------------------------------------------------------------
    A_rows, lo, hi = [], [], []
    vlim = prm.v_cap * _OCT * dt
    for k in range(H):
        for dm in _DIRS:                               # d . (p_k - p_{k-1}) <= vlim
            a = np.zeros(nx)
            a[2 * k:2 * k + 2] = dm
            off = 0.0
            if k == 0:
                off = float(dm @ p0)
            else:
                a[2 * (k - 1):2 * k] = -dm
            A_rows.append(a)
            lo.append(-np.inf)
            hi.append(vlim + off)
    for r, (k, a2, l2) in enumerate(zip(rows_k, rows_a, rows_lo)):
        a = np.zeros(nx)
        a[2 * k:2 * k + 2] = a2
        a[n_p + r] = 1.0
        A_rows.append(a)
        lo.append(l2)
        hi.append(np.inf)
    if S:
        A_rows.extend(np.eye(nx)[n_p:])               # s >= 0
        lo.extend([0.0] * S)
        hi.extend([np.inf] * S)
    A = np.asarray(A_rows)

    # warm start: previous prediction (or hold) with slacks at zero
    x0 = np.zeros(nx)
    x0[:n_p] = (lin if prev is not None else np.repeat(p0[None], H, axis=0)).ravel()

    solver = osqp.OSQP()
    solver.setup(sp.csc_matrix(np.triu(Pd)), q, sp.csc_matrix(A), np.asarray(lo), np.asarray(hi),
                 verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=8000, polishing=True)
    solver.warm_start(x=x0)
    res = solver.solve(raise_error=False)
    status = str(res.info.status)
    ok = res.x is not None and np.all(np.isfinite(res.x)) and "solved" in status.lower()
    if not ok:
        pts = np.repeat(p0[None], H + 1, axis=0)       # hold: always within the speed limit
        return DMPCResult(pts, status, math.inf, len(A_rows), dict(n_obs=n_obs, n_nbr=n_nbr, failed=True))
    x = res.x
    P = np.vstack([p0[None], x[:n_p].reshape(H, 2)])
    slack = float(x[n_p:].max()) if S else 0.0
    return DMPCResult(P, status, max(0.0, slack), len(A_rows), dict(n_obs=n_obs, n_nbr=n_nbr, failed=False))


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
