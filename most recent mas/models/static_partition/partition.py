"""Balanced geodesic k-means over viewpoints, then a drone <-> region assignment.

Distances are shortest-path distances around obstacles (not straight lines), so two points on
opposite sides of a wall are far apart. "Work" of a region is the number of coverable targets it
owns (a proxy for sweep time); a per-cluster additive bias is adjusted so loads even out.
"""
from __future__ import annotations

import itertools

import numpy as np


def _owner_viewpoint(view) -> np.ndarray:
    """Each coverable target is owned by the nearest viewpoint that sees it (-1 if none)."""
    vp, T = view.viewpoints, view.targets
    owner = np.full(len(T), -1, dtype=int)
    d = np.linalg.norm(vp.positions[:, None, :] - T[None, :, :], axis=2)
    d = np.where(vp.sees, d, np.inf)
    ok = np.isfinite(d.min(axis=0))
    owner[ok] = d.argmin(axis=0)[ok]
    return owner


def partition_targets(view, n: int, seed: int = 0, iters: int = 80):
    """Return (masks, info): masks[k] is the bool target mask of region k, drone-agnostic."""
    rng = np.random.default_rng(seed)
    P = view.viewpoints.positions
    D = view.planner.dist_matrix(P, P)
    D = np.where(np.isfinite(D), D, 1e6)
    owner = _owner_viewpoint(view)
    w = np.bincount(owner[owner >= 0], minlength=len(P)).astype(float)
    if n == 1:
        return [owner >= 0], dict(loads=[float(w.sum())], centers=[int(np.argmax(w))])

    # weighted farthest-first initialisation
    centers = [int(rng.choice(len(P), p=w / w.sum()))]
    while len(centers) < n:
        dmin = D[:, centers].min(axis=1)
        centers.append(int(np.argmax(dmin * (w > 0))))
    bias = np.zeros(n)
    best = None
    scale = float(np.median(D[D < 1e5]))
    for it in range(iters):
        assign = np.argmin(D[:, centers] + bias[None, :], axis=1)
        loads = np.array([w[assign == k].sum() for k in range(n)])
        imb = loads.max() / max(loads.mean(), 1e-9)
        if best is None or imb < best[0]:
            best = (imb, assign.copy(), list(centers), loads.copy())
        for k in range(n):                                   # medoid update (weighted)
            members = np.flatnonzero(assign == k)
            if members.size:
                cost = (D[np.ix_(members, members)] * w[members][None, :]).sum(axis=1)
                centers[k] = int(members[np.argmin(cost)])
        bias += 0.15 * scale * (loads - loads.mean()) / max(loads.mean(), 1e-9)   # shrink overloaded regions
    imb, assign, centers, loads = best
    masks = []
    for k in range(n):
        m = np.zeros(len(view.targets), dtype=bool)
        sel = owner >= 0
        m[sel] = assign[owner[sel]] == k
        masks.append(m)
    return masks, dict(loads=loads.tolist(), centers=centers, imbalance=float(imb))


def assign_regions(view, masks, centers) -> list[int]:
    """Region index for each drone, minimising the largest start->region-centre travel."""
    starts = view.starts
    P = view.viewpoints.positions
    cost = view.planner.dist_matrix(starts, P[centers])              # (drones, regions)
    n = len(masks)
    if n <= 7:
        best = min(itertools.permutations(range(n)), key=lambda perm: max(cost[i, perm[i]] for i in range(n)))
        return list(best)
    order, used, out = np.argsort(cost.min(axis=1))[::-1], set(), [0] * n
    for i in order:
        k = min((k for k in range(n) if k not in used), key=lambda k: cost[i, k])
        out[i] = k
        used.add(k)
    return out
