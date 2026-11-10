"""Geodesic Voronoi partition: the ownership rule (pure numpy) and end-to-end runs on real maps (need shapely)."""
import numpy as np
import pytest

import models  # noqa: F401  (registers models)
from mascov.common.config import SimConfig
from mascov.common.model import get_model
from mascov.maps.benchmarks import make_world
from mascov.sim.simulator import Simulator
from models.geodesic_partition import owners

UNKNOWN = dict(known_map=False)
KNOWN = dict(known_map=False, reveal_obstacles_at_start=True)


def test_each_spot_goes_to_the_closest_drone_and_ties_to_the_lowest_id():
    dist = np.array([[1.0, 5.0, 3.0],
                     [4.0, 2.0, 3.0]])
    assert owners(dist, [7, 3]).tolist() == [7, 3, 3]          # the tie in column 3 goes to id 3, not 7
    assert owners(dist, [3, 7]).tolist() == [3, 7, 3]          # row order does not matter, ids do


def test_a_wall_changes_who_owns_a_spot():
    """Euclidean closeness is not enough: a spot just behind a wall is far by path for the drone on the other side."""
    path = np.array([[1.0, 9.0],        # drone 0: near the spot by straight line but behind the wall (long way round)
                     [4.0, 2.0]])       # drone 1: further in a straight line, short path
    assert owners(path, [0, 1]).tolist() == [0, 1]


def _run(name, n, share=False, sim=UNKNOWN, seed=0):
    w = make_world(name, n_agents=n, seed=seed)
    r = Simulator(w, get_model("geodesic_partition", seed=seed, share=share),
                  SimConfig(max_time=1500, stop_when_covered=False, **sim)).run()
    return w, r


@pytest.mark.parametrize("name,n", [("corridor", 2), ("rooms_and_door", 3), ("sparse", 3), ("cluttered", 2)])
@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("sim", [UNKNOWN, KNOWN], ids=["unknown", "known"])
def test_covers_the_map_stays_clear_of_obstacles_and_knows_when_it_is_done(name, n, share, sim):
    w, r = _run(name, n, share, sim)
    assert r.completed
    assert all(b is not None for b in r.believed_done)
    assert r.min_clearance >= w.cfg.drone_radius, r.min_clearance


def test_it_sends_one_position_message_per_decision_and_none_when_alone():
    _, team = _run("rooms_and_door", 3)
    _, solo = _run("rooms_and_door", 1)
    assert team.n_messages > 0 and solo.n_messages == 0


def test_with_the_map_given_nothing_is_ever_discovered():
    _, r = _run("cluttered", 2, sim=KNOWN)
    assert r.extra["map_rebuilds"] == 2


def test_a_drone_that_hears_nobody_is_just_a_greedy_explorer():
    from mascov.world.world import World
    w = make_world("rooms_and_door", n_agents=2)
    w = World(w.cfg.with_(r_comm=0.1), list(w.obstacles), w.starts, name="deaf")
    r = Simulator(w, get_model("geodesic_partition"), SimConfig(max_time=1500, stop_when_covered=False, **UNKNOWN)).run()
    assert r.completed and r.n_messages == 0
