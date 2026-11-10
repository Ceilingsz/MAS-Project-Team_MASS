"""The greedy explorer: the scoring rule (pure numpy) and end-to-end runs on real maps (need shapely)."""
import numpy as np
import pytest

import models  # noqa: F401  (registers models)
from mascov.common.config import SimConfig
from mascov.common.model import get_model
from mascov.maps.benchmarks import make_world
from mascov.sim.simulator import Simulator
from models.greedy_explorer import choose_spot

UNKNOWN = dict(known_map=False)
KNOWN = dict(known_map=False, reveal_obstacles_at_start=True)       # same algorithm, obstacles given


def test_choose_spot_trades_gain_against_travel_time():
    gain, travel = np.array([10.0, 30.0, 5.0]), np.array([1.0, 20.0, 0.5])
    usable = np.ones(3, bool)
    assert choose_spot(gain, travel, usable, delta=1.0) == 0               # 10/2 = 5 beats 30/21 and 5/1.5
    assert choose_spot(gain, travel, np.array([False, True, True]), delta=1.0) == 2
    assert choose_spot(np.zeros(3), travel, usable, 1.0) is None            # nothing left to reveal
    assert choose_spot(gain, np.full(3, np.inf), usable, 1.0) is None       # nothing reachable


def test_choose_spot_breaks_exact_ties_at_random():
    gain, travel, usable = np.full(8, 4.0), np.ones(8), np.ones(8, bool)
    picks = {choose_spot(gain, travel, usable, 1.0, np.random.default_rng(s)) for s in range(40)}
    assert len(picks) > 1


def _run(name, n, share=False, sim=UNKNOWN, seed=0, **kw):
    w = make_world(name, n_agents=n, seed=seed)
    r = Simulator(w, get_model("greedy_explorer", seed=seed, share=share),
                  SimConfig(max_time=1500, stop_when_covered=False, **sim, **kw)).run()
    return w, r


@pytest.mark.parametrize("name,n", [("corridor", 2), ("rooms_and_door", 3), ("sparse", 3), ("cluttered", 2)])
@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("sim", [UNKNOWN, KNOWN], ids=["unknown", "known"])
def test_covers_the_map_stays_clear_of_obstacles_and_knows_when_it_is_done(name, n, share, sim):
    w, r = _run(name, n, share, sim)
    assert r.completed
    assert all(b is not None for b in r.believed_done)        # every drone concludes on its own that it is finished
    assert r.min_clearance >= w.cfg.drone_radius, r.min_clearance


def test_with_the_map_given_nothing_is_ever_discovered():
    w, r = _run("cluttered", 2, sim=KNOWN)
    assert r.extra["map_rebuilds"] == 2                       # one per drone: loading the prior, no later discovery


def test_without_the_map_obstacles_are_discovered():
    w, r = _run("cluttered", 2)
    assert r.extra["map_rebuilds"] > 2


def test_sharing_sends_messages_and_merges_beliefs_but_the_floor_does_not():
    _, floor = _run("rooms_and_door", 2)
    _, shared = _run("rooms_and_door", 2, share=True)
    assert floor.n_messages == 0 and floor.extra["cells_merged"] == 0
    assert shared.n_messages > 0 and shared.extra["cells_merged"] > 0


def test_no_messages_are_delivered_beyond_the_communication_radius():
    w = make_world("empty", n_agents=2, cfg=None)
    from mascov.world.world import World
    w = World(w.cfg.with_(r_comm=0.1), [], w.starts, name="far")
    r = Simulator(w, get_model("greedy_explorer", share=True), SimConfig(max_time=30, **UNKNOWN)).run()
    assert r.n_messages == 0 and r.extra["cells_merged"] == 0     # nobody within r_comm, so nobody even sends
