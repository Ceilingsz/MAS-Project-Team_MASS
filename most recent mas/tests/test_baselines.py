"""Baseline building blocks and per-model contracts: greedy viewpoint choice, the shared Navigator, the static
partition. Whole-run quality (completion, safety, speed, knowing when done) is graded for every model with one
bar in tests/test_model_quality.py, not here."""
import numpy as np
import pytest

import models  # noqa: F401  (registers every model)
from mascov.common.config import SimConfig, WorldConfig
from mascov.common.model import get_model, registered_models
from mascov.common.types import Plan
from mascov.maps.benchmarks import make_world
from mascov.planning.greedy import choose_viewpoint
from mascov.planning.navigator import Navigator, Neighbor
from mascov.sim.simulator import Simulator
from models.static_partition.partition import assign_regions, partition_targets


MULTI = ["greedy_centralised", "greedy_uncoordinated", "static_partition"]


def run(name, world, **sim):
    return Simulator(world, get_model(name), SimConfig(max_time=sim.pop("max_time", 500), **sim)).run()


def test_all_baselines_registered():
    assert {"greedy_single", *MULTI} <= set(registered_models())


def test_single_drone_centralised_completes_and_is_not_slower_than_b0():
    w = make_world("sparse", n_agents=1, seed=1)
    b0 = run("greedy_single", w, max_time=1500)
    b2 = run("greedy_centralised", w, max_time=1500)
    assert b0.completed and b2.completed
    assert b2.coverage_time <= b0.coverage_time * 1.05      # B2 drops claims that were seen en route


def test_uncoordinated_sends_no_messages_and_believes_done_only_after_the_truth():
    w = make_world("empty", n_agents=3)
    r = run("greedy_uncoordinated", w, stop_when_covered=False, max_time=900)
    assert r.n_messages == 0
    assert all(b is not None for b in r.believed_done)
    # a belief is a subset of the truth, so no agent can believe it is done before the world is covered
    assert min(r.believed_done) >= r.coverage_time - 1e-6


def test_partition_is_a_balanced_disjoint_cover_of_the_coverable_targets():
    w = make_world("empty", n_agents=4)
    view = w.view()
    masks, info = partition_targets(view, 4)
    stacked = np.stack(masks)
    assert stacked.sum(axis=0).max() == 1                                   # disjoint
    assert np.array_equal(stacked.any(axis=0), view.coverable)              # complete
    assert info["imbalance"] < 1.3
    assert sorted(assign_regions(view, masks, info["centers"])) == [0, 1, 2, 3]   # a permutation


def test_partition_respects_walls():
    """Geodesic distance: with a wall splitting the room, a region should not straddle it needlessly."""
    w = make_world("rooms_and_door", n_agents=2)
    masks, _ = partition_targets(w.view(), 2)
    left = w.targets[:, 0] < w.cfg.width / 2
    purity = [max((m & left).sum(), (m & ~left).sum()) / m.sum() for m in masks]
    assert min(purity) > 0.8


def test_static_partition_drones_only_chase_their_own_region():
    w = make_world("empty", n_agents=3)
    model = get_model("static_partition")
    r = Simulator(w, model, SimConfig(max_time=500, stop_when_covered=False)).run()
    assert r.completed and r.n_messages == 0
    masks = [p.region_mask for p in model.policies.values()]
    assert np.stack(masks).sum(axis=0).max() == 1                       # regions are disjoint
    for i, p in model.policies.items():
        # each drone's own belief covers its own region completely (that is why it stopped)
        assert r.believed_done[i] is not None


def test_choose_viewpoint_tie_breaking_and_exclusion():
    w = make_world("empty", n_agents=1)
    vp = w.viewpoints
    covered = np.zeros(len(vp.coverable), dtype=bool)
    travel = np.full(len(vp.positions), 5.0)                 # all ties
    first = choose_viewpoint(vp, travel, covered)
    picks = {choose_viewpoint(vp, travel, covered, rng=np.random.default_rng(s)) for s in range(30)}
    assert len(picks) > 1                                    # random tie-breaking actually varies
    assert choose_viewpoint(vp, travel, covered, exclude=[first]) != first
    assert choose_viewpoint(vp, travel, np.ones_like(covered)) is None


def _nav(world, **kw):
    return Navigator(world.cfg, world.planner, world.viewpoints.positions, **kw)


def test_navigator_flies_the_shortest_path_at_planning_speed():
    w = make_world("rooms_and_door", n_agents=1)
    nav = _nav(w)
    start, goal = np.array([1.0, 1.0]), np.array([10.0, 10.0])
    nav.set_goal(goal)
    plan = nav.update(0.0, start, np.zeros(2), True)
    assert nav.mode == "go" and plan is not None
    assert np.allclose(plan.p[0], start) and np.allclose(plan.end_point, goal)
    assert plan.length == pytest.approx(w.planner.distance(start, goal), rel=1e-6)
    assert plan.t_end == pytest.approx(plan.length / w.cfg.plan_speed, rel=1e-6)
    assert nav.update(0.5, start, np.zeros(2), False) is None         # a valid plan is kept, not recomputed


def test_navigator_arrives_within_tolerance_and_holds():
    w = make_world("empty", n_agents=1)
    nav = _nav(w)
    goal = np.array([6.0, 6.0])
    nav.set_goal(goal)
    nav.update(0.0, np.array([2.0, 2.0]), np.zeros(2), True)
    assert not nav.arrived
    near = goal + [nav.arrival_tolerance * 1.5, 0.0]                  # just outside the tolerance
    nav.update(1.0, near, np.zeros(2), True)
    assert not nav.arrived and nav.mode == "go"                       # finished short of the goal: replans
    inside = goal + [nav.arrival_tolerance * 0.5, 0.0]
    plan = nav.update(2.0, inside, np.zeros(2), False)
    assert nav.arrived and nav.mode == "arrived" and np.allclose(plan.end_point, inside)


def test_navigator_arrival_tolerance_is_within_the_viewpoint_sensing_slack():
    """A viewpoint 'sees' a target if it is within R_s - sense_slack. Stopping further than sense_slack from
    the viewpoint can leave its edge targets unseen (0.25 m once left 0.17% of rooms_and_door uncovered)."""
    w = make_world("empty", n_agents=1)
    assert _nav(w).arrival_tolerance <= w.cfg.sense_slack + 1e-12


def test_navigator_replans_from_where_the_drone_is_when_a_plan_ends_short():
    w = make_world("empty", n_agents=1)
    nav = _nav(w)
    nav.set_goal([10.0, 3.0])
    nav.update(0.0, np.array([2.0, 3.0]), np.zeros(2), True)
    here = np.array([5.0, 4.0])                                       # knocked off the path, plan over
    plan = nav.update(20.0, here, np.zeros(2), True)
    assert plan is not None and np.allclose(plan.p[0], here) and plan.t_start == pytest.approx(20.0)


def test_navigator_holds_on_an_unreachable_goal_and_when_the_goal_is_cleared():
    from mascov.geometry.pathing import PathPlanner
    from mascov.geometry.polygons import rect
    w = make_world("empty", n_agents=1)
    split = PathPlanner(rect(0, 0, 12, 12).difference(rect(5, -1, 7, 13)))   # a wall with no door
    nav = Navigator(w.cfg, split, w.viewpoints.positions)
    nav.set_goal([10.0, 6.0])
    here = np.array([2.0, 6.0])
    plan = nav.update(0.0, here, np.zeros(2), True)
    assert nav.mode == "blocked" and nav.stats["blocked_goals"] == 1
    assert len(plan.p) == 1 and np.allclose(plan.p[0], here)
    nav.set_goal(None)
    plan = nav.update(1.0, here, np.zeros(2), False)
    assert nav.mode == "idle" and len(plan.p) == 1 and np.allclose(plan.p[0], here)


def test_navigator_has_no_drone_drone_logic():
    """Documents the contract: neighbours do not change the route. Drone-drone safety must come from a
    safety layer (SimConfig(safety="bvc")) or a trajectory planner (greedy+dmpc, stl_team). If avoidance is
    added to the navigator again, replace this test with tests of that behaviour."""
    w = make_world("empty", n_agents=1)
    plans = []
    for nbrs in ([], [Neighbor(1, np.array([6.0, 3.0]), np.zeros(2), higher=True)]):
        nav = _nav(w)
        nav.set_goal([10.0, 3.0])
        plans.append(nav.update(0.0, np.array([2.0, 3.0]), np.zeros(2), True, nbrs))
    assert np.allclose(plans[0].p, plans[1].p) and np.allclose(plans[0].t, plans[1].t)
