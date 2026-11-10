import numpy as np
import pytest

import models  # noqa: F401  (registers every model)
from mascov.common.config import SimConfig, WorldConfig
from mascov.common.model import get_model, registered_models
from mascov.common.types import Plan
from mascov.maps.benchmarks import make_world
from mascov.planning.greedy import choose_viewpoint
from mascov.planning.navigator import Navigator, Neighbor, plan_positions
from mascov.sim.simulator import Simulator
from models.static_partition.partition import assign_regions, partition_targets

MULTI = ["greedy_centralised", "greedy_uncoordinated", "static_partition"]


def run(name, world, **sim):
    return Simulator(world, get_model(name), SimConfig(max_time=sim.pop("max_time", 500), **sim)).run()


def test_all_baselines_registered():
    assert {"greedy_single", *MULTI} <= set(registered_models())


@pytest.mark.parametrize("model", MULTI)
@pytest.mark.parametrize("map_name,n", [("empty", 3), ("rooms_and_door", 3), ("corridor", 2)])
def test_multi_drone_baselines_complete_without_collisions(model, map_name, n):
    w = make_world(map_name, n_agents=n, seed=0)
    r = run(model, w)
    assert r.completed, (model, map_name)
    assert r.collision_events == 0, (model, map_name, r.min_pair_dist)
    assert r.min_pair_dist >= w.cfg.d_safe - 1e-9
    assert r.min_clearance >= w.cfg.drone_radius


def test_single_drone_centralised_completes_and_is_not_slower_than_b0():
    w = make_world("sparse", n_agents=1, seed=1)
    b0 = run("greedy_single", w, max_time=1500)
    b2 = run("greedy_centralised", w, max_time=1500)
    assert b0.completed and b2.completed
    assert b2.coverage_time <= b0.coverage_time * 1.05      # B2 drops claims that were seen en route


def test_coordination_beats_no_coordination_and_more_drones_help():
    w1 = make_world("empty", n_agents=1)
    w3 = make_world("empty", n_agents=3)
    t1 = run("greedy_centralised", w1, max_time=1500).coverage_time
    b2 = run("greedy_centralised", w3).coverage_time
    b3 = run("greedy_uncoordinated", w3).coverage_time
    assert b2 < t1 * 0.6                                     # three coordinated drones are much faster than one
    assert b2 < b3                                           # and faster than three that do not share


def test_uncoordinated_sends_no_messages_and_believes_done_only_after_the_truth():
    w = make_world("empty", n_agents=3)
    r = run("greedy_uncoordinated", w, stop_when_covered=False, max_time=900)
    assert r.n_messages == 0
    assert all(b is not None for b in r.believed_done)
    # a belief is a subset of the truth, so no agent can believe it is done before the world is covered
    assert min(r.believed_done) >= r.coverage_time - 1e-6


def test_centralised_and_partition_finish_close_to_the_truth():
    w = make_world("empty", n_agents=3)
    for name in ("greedy_centralised", "static_partition"):
        r = run(name, w, stop_when_covered=False, max_time=900)
        assert max(r.believed_done) - r.coverage_time < 15.0, name


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


def test_navigator_reroutes_around_a_stationary_neighbor():
    w = make_world("empty", n_agents=1)
    nav = _nav(w)
    nav.set_goal([10.0, 3.0])
    blocker = Neighbor(1, np.array([6.0, 3.0]), np.zeros(2), higher=True)
    plan = nav.update(0.0, np.array([2.0, 3.0]), np.zeros(2), True, [blocker])
    assert plan is not None
    pts = plan_positions(plan, np.linspace(plan.t_start, plan.t_end, 200))
    assert np.linalg.norm(pts - blocker.pos, axis=1).min() >= nav.static_dist - 1e-6
    assert np.allclose(pts[-1], [10.0, 3.0], atol=1e-6)      # still reaches the goal


def test_navigator_yields_to_higher_priority_head_on_traffic_and_resumes():
    w = make_world("empty", n_agents=1)
    nav = _nav(w)
    nav.set_goal([10.0, 3.0])
    me, vel = np.array([5.0, 3.0]), np.array([0.6, 0.0])
    plan = nav.update(0.0, me, vel, True, [])                # free road: just go
    assert nav.mode == "go" and plan is not None
    oncoming = Neighbor(0, np.array([8.0, 3.0]), np.array([-0.6, 0.0]), higher=True)
    nav.update(0.5, me + vel * 0.5, vel, False, [oncoming])
    assert nav.mode == "wait" and nav.stats["yields"] == 1   # collision course -> yield
    far = Neighbor(0, np.array([8.0, 9.0]), np.array([-0.6, 0.0]), higher=True)
    nav.update(1.0, me + vel * 0.5, np.zeros(2), True, [far])
    assert nav.mode == "go"                                   # clear again -> resume


def test_navigator_is_not_blind_to_a_lower_priority_neighbor_directly_ahead():
    w = make_world("empty", n_agents=1)
    nav = _nav(w)
    nav.set_goal([10.0, 3.0])
    me, vel = np.array([5.0, 3.0]), np.array([0.6, 0.0])
    nav.update(0.0, me, vel, True, [])
    braking = Neighbor(5, np.array([5.9, 3.0]), np.array([0.15, 0.0]), higher=False)   # lower priority, slowing
    nav.update(0.5, me, vel, False, [braking])
    assert nav.mode == "wait"
