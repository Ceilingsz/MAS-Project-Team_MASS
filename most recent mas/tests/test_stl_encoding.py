"""The MILP encoding and the HiGHS wrapper, tested on their own (numpy + scipy only, no shapely).

Every guarantee is checked independently of the encoder: dense sampling of the returned trajectories against the
obstacle and the other drones, and direct checks of goals, speed, times and the window.
"""
import numpy as np
import pytest

from mascov.stl.encoding import plan_paths
from mascov.stl.milp import MILP, highs_available

pytestmark = pytest.mark.skipif(not highs_available(), reason="scipy >= 1.9 (HiGHS solver) not available")

VMAX, RHO, GOAL_HW, REAL_HW = 0.6, 0.6, 0.15, 0.25          # GOAL_HW is the shrunk box the solver uses
WINDOW = ((0.6, 11.4), (0.6, 11.4))
WALL = (5.8, 0.0, 6.2, 8.0)
SEP = 2 * 0.45 + 0.3


def rect_hs(x0, y0, x1, y1):
    return np.array([[-1.0, 0], [1, 0], [0, -1], [0, 1]]), np.array([-x0, x1, -y0, y1])


def dist_to_rect(P, x0, y0, x1, y1):
    dx = np.maximum(np.maximum(x0 - P[:, 0], 0), P[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - P[:, 1], 0), P[:, 1] - y1)
    return np.hypot(dx, dy)


def sampled(path, ts):
    T = [w[0] for w in path]
    return np.c_[np.interp(ts, T, [w[1] for w in path]), np.interp(ts, T, [w[2] for w in path])]


def first_feasible(starts, goals, obstacles, max_k=8, **kw):
    for k in range(1, max_k + 1):
        sol = plan_paths(starts, goals, obstacles, n_segs=k, vmax=VMAX, window=WINDOW, rho=RHO, goal_hw=GOAL_HW,
                         time_limit=60, **kw)
        if sol.ok:
            return sol
    return None


def reaches(path, g):
    W = np.array([[w[1], w[2]] for w in path])
    return bool(np.any(np.all(np.abs(W - g) <= REAL_HW + 1e-6, axis=1)))


def test_milp_wrapper_solves_a_small_integer_program():
    m = MILP()
    x, y = m.bin(cost=-5.0), m.bin(cost=-4.0)
    z = m.var(0.0, 10.0, cost=0.1)
    m.constr([(x, 6.0), (y, 4.0)], hi=8.0)                     # only one of x, y fits
    m.constr([(z, 1.0), (x, -2.0), (y, -2.0)], lo=0.0)         # z >= 2 (x + y)
    res = m.solve()
    assert res.optimal and res.x[x] == pytest.approx(1.0) and res.x[y] == pytest.approx(0.0)
    assert res.x[z] == pytest.approx(2.0) and m.n_binaries == 2


def test_single_drone_goes_over_a_wall_and_every_guarantee_holds():
    starts, goals = np.array([[2.0, 2.0]]), np.array([[10.0, 2.0], [10.0, 10.0]])
    sol = first_feasible(starts, goals, [rect_hs(*WALL)], tmax=80.0)
    assert sol is not None and sol.n_binaries > 0
    path = sol.paths[0]
    ts = np.linspace(0, path[-1][0], 4000)
    assert dist_to_rect(sampled(path, ts), *WALL).min() >= RHO - 1e-4          # clearance along the whole path
    assert all(reaches(path, g) for g in goals)
    W, T = np.array([[w[1], w[2]] for w in path]), np.array([w[0] for w in path])
    assert np.all(np.abs(np.diff(W, axis=0)).sum(axis=1) <= VMAX * np.diff(T) + 1e-5)    # 1-norm speed limit
    assert np.all(np.diff(T) > 0) and T[-1] <= 80.0 and np.allclose(W[0], [2.0, 2.0])
    assert W.min() >= 0.6 - 1e-6 and W.max() <= 11.4 + 1e-6


def test_waypoints_stay_inside_a_tighter_window():
    window = ((1.0, 6.0), (1.0, 6.0))
    sol = plan_paths(np.array([[2.0, 2.0]]), np.array([[4.0, 4.0]]), [], n_segs=2, tmax=60.0, vmax=VMAX, window=window,
                     rho=RHO, goal_hw=GOAL_HW)
    assert sol.ok
    W = np.array([[w[1], w[2]] for w in sol.paths[0]])
    assert W.min() >= 1.0 - 1e-6 and W.max() <= 6.0 + 1e-6


def test_infeasible_instances_are_reported_not_raised():
    starts = np.array([[2.0, 2.0]])
    inside_wall_zone = plan_paths(starts, np.array([[6.0, 4.0]]), [rect_hs(*WALL)], n_segs=4, tmax=60.0, vmax=VMAX,
                                  window=WINDOW, rho=RHO, goal_hw=GOAL_HW)
    assert not inside_wall_zone.ok and inside_wall_zone.message
    too_short = plan_paths(starts, np.array([[10.0, 2.0], [10.0, 10.0]]), [], n_segs=4, tmax=3.0, vmax=VMAX,
                           window=WINDOW, rho=RHO, goal_hw=GOAL_HW)
    assert not too_short.ok
    assert not plan_paths(starts, np.zeros((0, 2)), [], n_segs=100, tmax=0.5, vmax=VMAX, window=WINDOW, rho=RHO,
                          goal_hw=GOAL_HW).ok                                   # 100 segments of >= 0.01 s do not fit


def test_team_keeps_separation_ends_together_and_reaches_every_goal():
    starts = np.array([[2.0, 2.0], [4.0, 2.0], [2.0, 4.0], [4.0, 4.0]])
    goals = np.array([[8.0, 8.0], [8.0, 4.0], [4.0, 8.0], [10.0, 10.0]])
    sol = first_feasible(starts, goals, [], tmax=150.0, sep=SEP)
    assert sol is not None
    paths = sol.paths
    ts = np.linspace(0, paths[0][-1][0], 3000)
    pos = [sampled(p, ts) for p in paths]
    dmin = min(np.linalg.norm(pos[i] - pos[j], axis=1).min() for i in range(4) for j in range(i + 1, 4))
    assert dmin >= SEP - 1e-3
    assert len({round(p[-1][0], 6) for p in paths}) == 1                        # nobody vanishes early
    assert all(any(reaches(p, g) for p in paths) for g in goals)


def test_team_around_a_wall_keeps_both_separation_and_clearance():
    starts = np.array([[2.0, 2.0], [2.0, 4.0], [2.0, 6.0]])
    goals = np.array([[10.0, 2.0], [10.0, 6.0], [10.0, 10.0]])
    sol = first_feasible(starts, goals, [rect_hs(*WALL)], max_k=6, tmax=250.0, sep=SEP)
    assert sol is not None
    ts = np.linspace(0, sol.paths[0][-1][0], 4000)
    pos = [sampled(p, ts) for p in sol.paths]
    assert min(np.linalg.norm(pos[i] - pos[j], axis=1).min() for i in range(3) for j in range(i + 1, 3)) >= SEP - 1e-3
    assert min(dist_to_rect(P, *WALL).min() for P in pos) >= RHO - 1e-4


def test_starts_closer_than_the_separation_are_infeasible():
    starts = np.array([[2.0, 2.0], [2.5, 2.5]])                                  # infinity-norm distance 0.5 < 1.2
    sol = plan_paths(starts, np.array([[6.0, 6.0]]), [], n_segs=3, tmax=100.0, vmax=VMAX, window=WINDOW, rho=RHO,
                     goal_hw=GOAL_HW, sep=SEP)
    assert not sol.ok
