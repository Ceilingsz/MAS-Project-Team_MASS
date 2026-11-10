"""MILP encoding of the coverage specification, solved with HiGHS (no Gurobi, no third-party planner code).

This is our own encoding of the specification in the proposal (after Sun, Chen, Mitra and Fan, RA-L 2022:
piecewise-linear trajectories, "outside a polytope" as a disjunction over its faces with big-M binaries):

    phi =  AND over goals g    EVENTUALLY[0,tmax]  (some drone is inside the goal box g)
       AND AND over drones i   ALWAYS[0,tmax]      NOT (p_i in O_k grown by rho)     for every obstacle O_k
       AND AND over pairs i<j  ALWAYS[0,tmax]      ||p_i - p_j||_inf >= sep

Trajectory model. Every drone follows K straight segments between waypoints p_{i,0..K} at times t_0=0 <
t_1 < ... < t_K <= tmax. **All drones share the same times** (a synchronous window, so every drone ends at
t_K, early finishers waiting in place). Consequences:
  * the relative position of two drones is linear on each segment, so separation on a segment holds iff it
    holds at both ends for one fixed direction: four binaries per (pair, segment);
  * a drone never vanishes after its last waypoint (the upstream encoding drops it from the separation
    constraints, which let later drones be routed through it), and the objective is the MAKESPAN t_K.

Constraint details
  * speed: ||p_{k+1} - p_k||_1 <= vmax (t_{k+1} - t_k). The 1-norm is conservative (it implies the Euclidean
    bound) and linear with no binaries.
  * obstacles: for each segment and obstacle, one face h.x <= b of the polygon must be violated by at least rho
    at BOTH endpoints (a half-space is convex, so the whole segment is then outside the grown polygon).
    rho is the clearance, drone_radius + tracking error. Mitred growth: Euclidean distance >= rho follows.
  * goals: some waypoint of some drone lies in the (shrunk) goal box. Reaching a box between waypoints does
    not count; the planner shrinks the box so that being inside it guarantees the targets are seen.
  * separation: L-infinity distance >= sep, so the Euclidean distance is >= sep.
  * big-M constants are computed per constraint from the window, as small as is valid, which keeps the
    relaxation tight and the numerical error low.

Only numpy and scipy are needed here (no shapely), so this module can be tested on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mascov.stl.milp import MILP


@dataclass
class PathsSolution:
    ok: bool
    paths: list = field(default_factory=list)        # per drone: [(t, x, y), ...] (K + 1 waypoints, shared times)
    n_binaries: int = 0
    n_vars: int = 0
    n_constraints: int = 0
    message: str = ""


def plan_paths(starts, goals, obstacles, *, n_segs: int, tmax: float, vmax: float, window, rho: float,
               goal_hw: float, sep: float | None = None, min_dt: float = 0.01, time_limit: float = 20.0,
               mip_gap: float = 1e-2, length_weight: float = 1e-3) -> PathsSolution:
    """Solve for exactly ``n_segs`` segments per drone.

    starts   : (A, 2) start positions (all at time 0)
    goals    : (G, 2) goal box centres; each box is the square of half-width ``goal_hw``
    obstacles: list of (H, b) with ``H x <= b`` describing a convex polygon (rows are unit outward normals)
    window   : ((xmin, xmax), (ymin, ymax)) every waypoint must lie in it
    rho      : clearance kept from every obstacle
    sep      : minimum L-infinity distance between drones (None: single drone / no separation)
    """
    starts = np.asarray(starts, dtype=float).reshape(-1, 2)
    goals = np.asarray(goals, dtype=float).reshape(-1, 2)
    A, K = len(starts), int(n_segs)
    if K < 1:
        return PathsSolution(False, message="n_segs must be >= 1")
    if min_dt * K > tmax:
        return PathsSolution(False, message="tmax is too small for this many segments")
    (x0, x1), (y0, y1) = window
    bx = (min(x0, starts[:, 0].min()), max(x1, starts[:, 0].max()))      # box that contains every waypoint
    by = (min(y0, starts[:, 1].min()), max(y1, starts[:, 1].max()))

    m = MILP()
    t = [m.var(0.0, 0.0)] + [m.var(min_dt * k, tmax, cost=1.0 if k == K else 0.0) for k in range(1, K + 1)]
    for k in range(K):
        m.constr([(t[k + 1], 1.0), (t[k], -1.0)], lo=min_dt)                                  # strictly increasing times
    X = [[m.var(*((s[0], s[0]) if k == 0 else (x0, x1))) for k in range(K + 1)] for s in starts]
    Y = [[m.var(*((s[1], s[1]) if k == 0 else (y0, y1))) for k in range(K + 1)] for s in starts]

    # speed (1-norm) and path length: ex >= |dx|, ey >= |dy|, ex + ey <= vmax * dt
    for a in range(A):
        for k in range(K):
            ex, ey = m.var(0.0, cost=length_weight), m.var(0.0, cost=length_weight)
            for e, P in ((ex, X), (ey, Y)):
                m.constr([(e, 1.0), (P[a][k + 1], -1.0), (P[a][k], 1.0)], lo=0.0)            # e >= p_{k+1} - p_k
                m.constr([(e, 1.0), (P[a][k + 1], 1.0), (P[a][k], -1.0)], lo=0.0)            # e >= p_k - p_{k+1}
            m.constr([(ex, 1.0), (ey, 1.0), (t[k + 1], -vmax), (t[k], vmax)], hi=0.0)

    # obstacles: each segment must be outside the grown polygon, through one face, at both endpoints
    for H, b in obstacles:
        H, b = np.asarray(H, dtype=float), np.asarray(b, dtype=float)
        min_hp = np.where(H[:, 0] > 0, H[:, 0] * bx[0], H[:, 0] * bx[1]) + np.where(H[:, 1] > 0, H[:, 1] * by[0], H[:, 1] * by[1])
        Mr = b + rho - min_hp                           # largest possible shortfall of h.p >= b + rho inside the box
        if np.any(Mr <= 0):
            continue                                     # one face is satisfied everywhere in the box: no restriction
        for a in range(A):
            for k in range(K):
                zs = [m.bin() for _ in range(len(b))]
                m.constr([(z, 1.0) for z in zs], lo=1.0)
                for r, z in enumerate(zs):
                    for e in (k, k + 1):
                        m.constr([(X[a][e], H[r, 0]), (Y[a][e], H[r, 1]), (z, -Mr[r])], lo=b[r] + rho - Mr[r])

    # goals: some waypoint of some drone lies in the box
    for cx, cy in goals:
        ys = []
        Mx_hi, Mx_lo = max(bx[1] - (cx + goal_hw), 0.0), max((cx - goal_hw) - bx[0], 0.0)
        My_hi, My_lo = max(by[1] - (cy + goal_hw), 0.0), max((cy - goal_hw) - by[0], 0.0)
        for a in range(A):
            for k in range(K + 1):
                y = m.bin()
                ys.append(y)
                m.constr([(X[a][k], 1.0), (y, Mx_hi)], hi=cx + goal_hw + Mx_hi)
                m.constr([(X[a][k], -1.0), (y, Mx_lo)], hi=-(cx - goal_hw) + Mx_lo)
                m.constr([(Y[a][k], 1.0), (y, My_hi)], hi=cy + goal_hw + My_hi)
                m.constr([(Y[a][k], -1.0), (y, My_lo)], hi=-(cy - goal_hw) + My_lo)
        m.constr([(y, 1.0) for y in ys], lo=1.0)

    # separation: L-infinity distance >= sep between every pair, on every segment (shared time grid)
    if sep and A > 1:
        Mx, My = (bx[1] - bx[0]) + sep, (by[1] - by[0]) + sep
        for i in range(A):
            for j in range(i + 1, A):
                for k in range(K):
                    ws = [m.bin() for _ in range(4)]
                    m.constr([(w, 1.0) for w in ws], lo=1.0)
                    for e in (k, k + 1):
                        m.constr([(X[i][e], 1.0), (X[j][e], -1.0), (ws[0], -Mx)], lo=sep - Mx)    # x_i - x_j >= sep
                        m.constr([(X[j][e], 1.0), (X[i][e], -1.0), (ws[1], -Mx)], lo=sep - Mx)    # x_j - x_i >= sep
                        m.constr([(Y[i][e], 1.0), (Y[j][e], -1.0), (ws[2], -My)], lo=sep - My)    # y_i - y_j >= sep
                        m.constr([(Y[j][e], 1.0), (Y[i][e], -1.0), (ws[3], -My)], lo=sep - My)    # y_j - y_i >= sep

    res = m.solve(time_limit=time_limit, mip_gap=mip_gap)
    info = dict(n_binaries=m.n_binaries, n_vars=m.n_vars, n_constraints=m.n_constraints)
    if res.x is None:
        return PathsSolution(False, message=_reason(res), **info)
    x = res.x
    ts = [float(x[v]) for v in t]
    paths = [[(ts[k], float(x[X[a][k]]), float(x[Y[a][k]])) for k in range(K + 1)] for a in range(A)]
    return PathsSolution(True, paths, message="optimal" if res.optimal else "feasible (time limit)", **info)


def _reason(res) -> str:
    if res.status == 2:
        return "infeasible"
    if res.status == 1:
        return "time limit reached without a feasible solution"
    return f"solver status {res.status}: {res.message}"
