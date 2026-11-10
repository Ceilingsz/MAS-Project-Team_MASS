"""Lower bounds on coverage time: a floor no method can beat (approximately).

Both bounds ignore other drones and tracking error, so they are optimistic. The reach
bound works on the candidate-viewpoint lattice, so a sampling slack is subtracted.
"""
from __future__ import annotations

import math

import numpy as np


def reach_bound(world) -> float:
    """Slowest-to-reach target: every target needs some drone to travel to a spot that sees it."""
    cfg = world.cfg
    vp = world.viewpoints
    travel = world.planner.dist_matrix(world.starts, vp.positions) / cfg.v_max   # (N, V)
    best_for_vp = travel.min(axis=0)                                              # (V,)
    worst = 0.0
    for m in np.flatnonzero(vp.coverable):
        seen_by = vp.sees[:, m]
        worst = max(worst, float(best_for_vp[seen_by].min()))
    slack = cfg.viewpoint_spacing * 0.7072 / cfg.v_max
    return max(0.0, worst - slack)


def area_bound(world) -> float:
    """A disc of radius R_s sweeping a path of length L reveals at most 2 R_s L + pi R_s^2 of
    area, so N drones need  T >= (A/N - pi R_s^2) / (2 R_s v_max)."""
    cfg = world.cfg
    area = float(world.coverable.sum()) * cfg.target_spacing ** 2
    n = world.n_agents
    return max(0.0, (area / n - math.pi * cfg.r_sense ** 2) / (2.0 * cfg.r_sense * cfg.v_max))


def lower_bound(world) -> dict:
    r, a = reach_bound(world), area_bound(world)
    return dict(reach=r, area=a, best=max(r, a))
