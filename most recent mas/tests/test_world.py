import numpy as np
import pytest

from mascov.common.config import WorldConfig
from mascov.geometry.polygons import rect
from mascov.maps.benchmarks import BENCHMARKS, make_world
from mascov.metrics.bounds import area_bound, lower_bound, reach_bound
from mascov.world.belief import AgentBelief
from mascov.world.world import World


def test_targets_exclude_obstacle_interiors():
    cfg = WorldConfig()
    wall = rect(4.9, 0, 6.1, 8)                  # contains target columns x=5.25 and 5.75
    w = World(cfg, [wall], [[1, 1]])
    assert not np.any((w.targets[:, 0] > 4.9) & (w.targets[:, 0] < 6.1) & (w.targets[:, 1] < 8))
    assert w.n_targets < int(cfg.width / cfg.target_spacing) ** 2


def test_start_inside_clearance_is_rejected():
    cfg = WorldConfig()
    with pytest.raises(ValueError):
        World(cfg, [], [[0.1, 0.1]])             # closer to the wall than r_b + eps


def test_nonconvex_obstacle_is_rejected():
    from shapely.geometry import Polygon
    ell = Polygon([(2, 2), (6, 2), (6, 3), (3, 3), (3, 6), (2, 6)])
    with pytest.raises(ValueError):
        World(WorldConfig(), [ell], [[1, 1]])


def test_sensing_disc_in_empty_world():
    w = make_world("empty", n_agents=1)
    pos = np.array([6.0, 6.0])
    ids = w.sensor.visible_target_ids(pos)
    d = np.linalg.norm(w.targets[ids] - pos, axis=1)
    assert d.max() <= w.cfg.r_sense + 1e-9
    assert len(ids) == int((np.linalg.norm(w.targets - pos, axis=1) <= w.cfg.r_sense).sum())


def test_wall_blocks_sensing():
    cfg = WorldConfig()
    w = World(cfg, [rect(6.0, 0.0, 6.4, 9.0)], [[1, 1]])
    pos = np.array([5.3, 4.0])
    ids = w.sensor.visible_target_ids(pos)
    assert np.all(w.targets[ids][:, 0] < 6.0)                    # nothing behind the wall
    no_los = World(cfg.with_(use_los=False), [rect(6.0, 0.0, 6.4, 9.0)], [[1, 1]])
    assert np.any(no_los.sensor.visible_target_ids(pos) != -1) and \
        no_los.targets[no_los.sensor.visible_target_ids(pos)][:, 0].max() > 6.0


def test_obstacle_discovery_only_from_the_visible_side():
    cfg = WorldConfig()
    w = World(cfg, [rect(6.0, 0.0, 6.4, 9.0), rect(10, 10, 11, 11)], [[1, 1]])
    assert w.sensor.visible_obstacle_ids(np.array([5.0, 4.0])) == {0}
    assert w.sensor.visible_obstacle_ids(np.array([1.0, 1.0])) == set()      # out of range


def test_belief_merge_is_a_set_union():
    a, b = AgentBelief(10), AgentBelief(10)
    a.mark([1, 2, 3]); b.mark([3, 4])
    b.known_obstacles.add(7)
    assert a.merge(b) == 1
    assert a.covered.sum() == 4 and 7 in a.known_obstacles
    assert a.merge(b) == 0                                       # idempotent
    c = b.copy(); c.merge(a)
    d = a.copy(); d.merge(b)
    assert np.array_equal(c.covered, d.covered)                  # commutative


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_benchmark_maps_are_valid(name):
    w = make_world(name, n_agents=3, seed=2)
    s = w.summary()
    assert s["n_coverable"] >= 0.9 * s["n_targets"]
    assert len(w.viewpoints.positions) > 20
    # every start can reach every viewpoint
    assert np.isfinite(w.planner.dist_matrix(w.starts, w.viewpoints.positions)).all()


def test_maps_are_reproducible_per_seed():
    a, b = make_world("sparse", seed=3), make_world("sparse", seed=3)
    assert [o.wkt for o in a.obstacles] == [o.wkt for o in b.obstacles]
    assert make_world("sparse", seed=4).obstacles[0].wkt != a.obstacles[0].wkt


def test_lower_bounds_are_sane():
    w = make_world("empty", n_agents=1)
    lb1 = lower_bound(w)
    w4 = make_world("empty", n_agents=4)
    lb4 = lower_bound(w4)
    assert lb1["best"] > 0
    assert area_bound(w4) < area_bound(w)                       # more drones -> smaller bound
    assert reach_bound(w) >= 0


def test_world_view_hides_victims():
    w = World(WorldConfig(), [], [[1, 1]], victims=[[5, 5]])
    v = w.view()
    assert not hasattr(v, "victims")
