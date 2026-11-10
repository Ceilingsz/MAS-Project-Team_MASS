"""Tests for the STL planner. Tests that solve a MILP need scipy >= 1.9 (HiGHS) and are skipped without it;
verify_plan and the geometry helpers are always tested. The encoding itself is tested in test_stl_encoding.py."""
import numpy as np
import pytest

import models  # noqa: F401
from mascov.common.config import SimConfig, WorldConfig
from mascov.common.model import get_model
from mascov.geometry.polygons import rect
from mascov.maps.benchmarks import make_world
from mascov.sim.simulator import Simulator
from mascov.stl.planner import STLPlanner, stl_available, verify_plan

needs_stl = pytest.mark.skipif(not stl_available(), reason="scipy >= 1.9 (HiGHS solver) not available")
CFG = WorldConfig()
WALL = [rect(5.8, 0.0, 6.2, 8.0)]


def test_verify_plan_accepts_a_good_plan_and_catches_each_kind_of_violation():
    start, goals = [2.0, 2.0], [[10.0, 2.0]]
    good = [(0.0, 2.0, 2.0), (20.0, 2.0, 9.0), (40.0, 10.0, 9.0), (60.0, 10.0, 2.0)]
    assert verify_plan(good, start, goals, WALL, CFG, tmax=100.0)["all_ok"]
    through_wall = [(0.0, 2.0, 2.0), (30.0, 10.0, 2.0)]
    assert not verify_plan(through_wall, start, goals, WALL, CFG, tmax=100.0)["obstacle_clearance_ok"]
    assert not verify_plan(good, start, [[10.0, 10.5]], WALL, CFG, tmax=100.0)["all_ok"]           # goal never reached
    fast = [(0.0, 2.0, 2.0), (1.0, 2.0, 9.0), (2.0, 10.0, 9.0), (3.0, 10.0, 2.0)]
    assert not verify_plan(fast, start, goals, WALL, CFG, tmax=100.0)["speed_ok"]
    assert not verify_plan(good, start, goals, WALL, CFG, tmax=30.0)["within_tmax"]


def test_snap_with_inset_pulls_boundary_points_inwards():
    w = make_world("empty", n_agents=1)
    c = w.cfg.clearance
    inside = np.array([6.0, 6.0])
    assert np.allclose(w.planner.snap(inside, inset=0.03), inside)
    on_boundary = np.array([c, 6.0])
    pulled = w.planner.snap(on_boundary, inset=0.03)
    assert pulled[0] >= c + 0.03 - 1e-6
    outside = np.array([0.1, 6.0])                                   # in the clearance margin
    assert w.planner.snap(outside)[0] == pytest.approx(c, abs=1e-6)


@needs_stl
def test_stl_plan_over_a_wall_is_verified_and_visits_both_goals():
    start, goals = [2.0, 2.0], [[10.0, 2.0], [10.0, 10.0]]
    res = STLPlanner(CFG).plan_visits(start, goals, WALL, tmax=80.0, max_segs=6)
    assert res.ok and res.n_binaries > 0
    rep = verify_plan(res.waypoints, start, goals, WALL, CFG, tmax=80.0)
    assert rep["all_ok"], rep
    assert rep["min_obstacle_distance"] >= CFG.clearance


@needs_stl
def test_stl_waypoints_stay_inside_the_window():
    start, goals = [2.0, 2.0], [[4.0, 4.0]]
    window = ((1.0, 6.0), (1.0, 6.0))
    res = STLPlanner(CFG).plan_visits(start, goals, [], tmax=60.0, max_segs=4, window=window)
    assert res.ok
    P = np.array([[w[1], w[2]] for w in res.waypoints])
    assert P[:, 0].min() >= 1.0 - 1e-6 and P[:, 0].max() <= 6.0 + 1e-6
    assert P[:, 1].min() >= 1.0 - 1e-6 and P[:, 1].max() <= 6.0 + 1e-6


@needs_stl
def test_stl_solver_failure_is_reported_not_swallowed():
    # a goal sealed inside the wall's inflated zone cannot be reached: must come back ok=False, not crash
    start, goals = [2.0, 2.0], [[6.0, 4.0]]
    res = STLPlanner(CFG, time_limit=5.0).plan_visits(start, goals, WALL, tmax=60.0, max_segs=4)
    assert not res.ok and res.message


@needs_stl
def test_stl_single_covers_a_map_with_verified_plans_and_no_collisions():
    w = make_world("rooms_and_door", n_agents=1, seed=0)
    r = Simulator(w, get_model("stl_single"), SimConfig(max_time=900)).run()
    assert r.completed and r.collision_events == 0
    assert r.extra["stl_verify_fail"] == 0                          # every plan re-checked independently
    assert r.extra["stl_windows"] >= 5 and r.extra["stl_fallbacks"] <= 2


# ------------------------------------------------------------------ multi-agent (joint) STL
from mascov.stl.planner import verify_team  # noqa: E402

CFG45 = WorldConfig(eps_track=0.45)


def test_verify_team_catches_a_drone_that_is_run_through_after_it_stops():
    """A drone that has finished still occupies its spot: the verifier must hold it in place."""
    starts = np.array([[2.0, 2.0], [2.0, 8.0]])
    goals = np.array([[2.0, 4.0]])
    a = [(0.0, 2.0, 2.0), (10.0, 2.0, 4.0)]                        # A drives to (2,4) and stays there
    b = [(0.0, 2.0, 8.0), (20.0, 2.0, 4.0), (30.0, 2.0, 1.0)]      # B later drives straight through it
    rep = verify_team([a, b], starts, goals, [], CFG45, tmax=100.0)
    assert not rep["separation_ok"] and rep["min_pair_reference_distance"] < 0.1
    clear = [(0.0, 2.0, 2.0), (10.0, 2.0, 4.0)]
    far = [(0.0, 8.0, 8.0), (10.0, 8.0, 7.0)]
    assert verify_team([clear, far], np.array([[2.0, 2.0], [8.0, 8.0]]), goals, [], CFG45, tmax=100.0)["separation_ok"]


@needs_stl
def test_team_plan_assigns_goals_balances_work_and_keeps_drones_apart():
    starts = np.array([[1.0, 1.0], [3.0, 1.0]])
    goals = np.array([[10.0, 2.0], [10.0, 10.0], [2.0, 10.0], [6.0, 6.0]])
    res = STLPlanner(CFG45, time_limit=30).plan_team(starts, goals, [], tmax=150.0, max_segs=7)
    assert res.ok
    assert set(res.assignment) <= {0, 1} and -1 not in res.assignment
    rep = verify_team(res.paths, starts, goals, [], CFG45, tmax=150.0)
    assert rep["all_ok"], rep
    ends = [p[-1][0] for p in res.paths]
    assert max(ends) - min(ends) < 1e-3                          # equal end times: nobody vanishes early
    assert rep["min_pair_reference_distance"] >= 2 * CFG45.eps_track + CFG45.d_safe - 1e-3


@needs_stl
def test_team_plan_around_a_wall_is_verified():
    starts = np.array([[1.0, 1.0], [3.0, 1.0]])
    goals = np.array([[10.0, 2.0], [2.0, 10.0]])
    res = STLPlanner(CFG45, time_limit=30).plan_team(starts, goals, WALL, tmax=250.0, max_segs=9)
    assert res.ok and sorted(res.assignment) == [0, 1]
    assert verify_team(res.paths, starts, goals, WALL, CFG45, tmax=250.0)["all_ok"]


@needs_stl
def test_stl_team_covers_a_map_with_no_collisions_and_verified_plans():
    w = make_world("rooms_and_door", n_agents=2, seed=0, cfg=CFG45, start_spacing=2.0, start_per_row=2)
    r = Simulator(w, get_model("stl_team"), SimConfig(max_time=900)).run()
    assert r.completed and r.collision_events == 0
    assert r.extra["team_verify_fail"] == 0
    assert r.min_pair_dist >= CFG45.d_safe
