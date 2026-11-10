"""Buffered-Voronoi-cell safety filter on STOPPING positions (decentralised, control rate).

Each drone i looks at neighbours j it can sense (position + velocity) and, for each, builds the
perpendicular bisector between the two drones' *stopping points*

    s = p + v |v| / (2 a_b)          (where the drone would stop if it braked at a_b < a_max)

shifted back by the buffer r_s = d_safe / 2. Drone i may only apply accelerations whose NEXT stopping
point stays on its own side of every such line. Why stopping points and not positions: a plain
position BVC can be infeasible once two drones are already closing (cannot brake in time). Braking
exactly at a_b along -v leaves the stopping point unchanged, so "brake" is always an admissible action
as long as everyone obeys the same rule (this is the recursive-feasibility argument; it is exact along
the line joining two drones and approximate when the line rotates).

The filter picks the admissible acceleration closest to the one the tracker asked for, from a finite
candidate set (the desired acceleration, zero, braking, and a polar grid). If nothing is admissible it
picks the least-violating candidate.

This is the SAFETY part of the proposal's Layer 2. It is not the C-ADMM trajectory optimisation, and it
handles drone-drone avoidance only (no obstacles).
"""
from __future__ import annotations

import numpy as np

_ANGLES = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
_MAGS = np.array([0.25, 0.5, 0.75, 1.0])
_GRID = np.concatenate([[[0.0, 0.0]],
                        np.c_[np.outer(_MAGS, np.cos(_ANGLES)).ravel(), np.outer(_MAGS, np.sin(_ANGLES)).ravel()]])


def stopping_point(p: np.ndarray, v: np.ndarray, a_brake: float) -> np.ndarray:
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    return p + v * speed / (2.0 * a_brake)


def bvc_filter(pos: np.ndarray, vel: np.ndarray, a_des: np.ndarray, nbr_pos: np.ndarray,
               nbr_vel: np.ndarray, *, dt: float, a_max: float, d_safe: float,
               brake_frac: float = 0.7, buffer_extra: float = 0.02) -> tuple[np.ndarray, bool]:
    """Return (safe acceleration, was_modified)."""
    a_des = np.asarray(a_des, dtype=float)
    if len(nbr_pos) == 0:
        return a_des, False
    a_b = brake_frac * a_max
    s_i = stopping_point(pos, vel, a_b)
    s_j = stopping_point(nbr_pos, nbr_vel, a_b)
    d = s_j - s_i
    dist = np.linalg.norm(d, axis=1)
    fallback = nbr_pos - pos                                   # degenerate: stopping points coincide
    d = np.where(dist[:, None] < 1e-9, fallback, d)
    n = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    mid = 0.5 * (s_i + s_j)
    c = np.einsum("kd,kd->k", n, mid) - (0.5 * d_safe + buffer_extra)       # require n . s_next <= c

    speed = np.linalg.norm(vel)
    brake = -vel / speed * a_b if speed > 1e-9 else np.zeros(2)
    cand = np.vstack([a_des[None], _GRID * a_max, brake[None]])
    norms = np.linalg.norm(cand, axis=1)
    over = norms > a_max
    cand[over] *= (a_max / norms[over])[:, None]

    p_next = pos + vel * dt + 0.5 * cand * dt * dt
    v_next = vel + cand * dt
    s_next = stopping_point(p_next, v_next, a_b)
    g = s_next @ n.T - c[None, :]                                # (C, K): <= 0 means admissible
    worst = g.max(axis=1)
    ok = worst <= 0.0
    if ok[0]:
        return a_des, False
    if ok.any():
        idx = np.flatnonzero(ok)
        best = idx[np.argmin(np.linalg.norm(cand[idx] - a_des, axis=1))]
    else:
        best = int(np.argmin(worst))
    return cand[best].copy(), True
