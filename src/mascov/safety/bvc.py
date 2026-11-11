"""Buffered-Voronoi-cell safety filter for single-integrator drones (decentralised, control rate).

Drone i looks at each neighbour j it can sense (position only) and builds the perpendicular bisector of
the segment p_i p_j, shifted towards i by half the minimum separation, d_safe / 2 (plus a small buffer).
Drone i may only apply a velocity whose NEXT position p_i + v dt stays on its own side of every such line.
If every drone obeys the same rule, two drones are always on opposite sides of a line that is at least
d_safe apart, so they can never come closer than d_safe.

Why this is exact for a single integrator: a drone stops instantly, so "zero velocity" is always
admissible while all neighbours are currently at least d_safe + 2 * buffer away. There is no braking
distance to account for (the double-integrator version needed stopping points for that reason).

The filter picks the admissible velocity closest to the one the tracker asked for, from a finite candidate
set: the request itself, its projection onto each neighbour's half-space, zero, and a polar grid. If
nothing is admissible it picks the least-violating candidate.

Sensing requirement: a neighbour must be sensed before it can get within d_safe in one control step, i.e.
neighbor_radius >= d_safe + 2 * v_max * dt (+ buffer), where neighbor_radius is WorldConfig.r_neighbor
(or r_comm when r_neighbor is None).

This is a safety filter, not a trajectory optimiser (that is D-MPC, planning/dmpc.py), and it
handles drone-drone avoidance only (no obstacles).
"""
from __future__ import annotations

import numpy as np

_ANGLES = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
_MAGS = np.array([0.25, 0.5, 0.75, 1.0])
_GRID = np.concatenate([[[0.0, 0.0]],
                        np.c_[np.outer(_MAGS, np.cos(_ANGLES)).ravel(), np.outer(_MAGS, np.sin(_ANGLES)).ravel()]])


def bvc_filter(pos: np.ndarray, v_des: np.ndarray, nbr_pos: np.ndarray, *, dt: float, v_max: float,
               d_safe: float, buffer_extra: float = 0.02) -> tuple[np.ndarray, bool]:
    """Return (safe velocity, was_modified)."""
    v_des = np.asarray(v_des, dtype=float)
    s = float(np.linalg.norm(v_des))
    if s > v_max:
        v_des = v_des * (v_max / s)
    if len(nbr_pos) == 0:
        return v_des, False
    d = nbr_pos - pos
    dist = np.linalg.norm(d, axis=1)
    n = d / np.maximum(dist, 1e-12)[:, None]                  # unit vectors towards each neighbour
    c = 0.5 * dist - (0.5 * d_safe + buffer_extra)            # require n . (v dt) <= c for every neighbour

    excess = (n @ v_des) * dt - c                             # > 0: the request enters that neighbour's buffer
    proj = v_des[None] - (np.maximum(excess, 0.0) / dt)[:, None] * n
    cand = np.vstack([v_des[None], proj, _GRID * v_max])
    norms = np.linalg.norm(cand, axis=1)
    over = norms > v_max
    cand[over] *= (v_max / norms[over])[:, None]

    g = (cand @ n.T) * dt - c[None, :]                        # (C, K): <= 0 means admissible
    worst = g.max(axis=1)
    ok = worst <= 1e-12
    if ok[0]:
        return v_des, False
    if ok.any():
        idx = np.flatnonzero(ok)
        best = idx[np.argmin(np.linalg.norm(cand[idx] - v_des, axis=1))]
    else:
        best = int(np.argmin(worst))
    return cand[best].copy(), True
