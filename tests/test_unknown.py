"""Unknown-map interface: no information leaks, belief algebra, simulator wiring, frontier explorer."""
import numpy as np
import pytest

import models  # noqa: F401  (registers models)
from mascov.common.config import SimConfig, WorldConfig
from mascov.common.model import AgentPolicy, CentralModel, DecentralisedModel, get_model
from mascov.common.types import Command
from mascov.geometry.polygons import rect
from mascov.maps.benchmarks import BENCHMARKS, make_world
from mascov.sim.simulator import Simulator
from mascov.world.belief import FREE, OBSTACLE, UNKNOWN, MapBelief
from mascov.world.world import World

UNKNOWN_SIM = dict(known_map=False)


# ---------------------------------------------------------------- the world and what an agent may be told
@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_targets_are_the_free_cells_of_the_lattice(name):
    w = make_world(name, n_agents=2, seed=1)
    assert np.array_equal(w.targets, w.lattice[w.cell_free])
    assert np.array_equal(w.lattice[w.target_cell], w.targets)
    assert len(w.lattice) == int(w.cfg.width / w.cfg.target_spacing) * int(w.cfg.height / w.cfg.target_spacing)


def test_unknown_view_does_not_expose_the_map():
    v = make_world("cluttered", n_agents=2).unknown_view()
    for leak in ("obstacles", "targets", "viewpoints", "coverable", "planner", "obstacle_union", "victims"):
        assert not hasattr(v, leak), leak


def test_unknown_view_is_independent_of_the_map():
    """Same config and starts -> identical view, whatever the obstacles are: nothing about them leaks."""
    a = make_world("empty", n_agents=3).unknown_view()
    b = make_world("cluttered", n_agents=3, seed=5).unknown_view()
    assert np.array_equal(a.lattice, b.lattice) and np.array_equal(a.candidates, b.candidates)
    assert np.array_equal(a.starts, b.starts)


# ---------------------------------------------------------------- belief algebra
def _lattice():
    return World(WorldConfig(), [], [[1.0, 1.0]]).unknown_view().lattice


def test_map_belief_marks_and_never_regresses():
    L = _lattice()
    b = MapBelief(L)
    assert b.mark_free([3, 4, 5]) == 3 and b.mark_free([4, 5, 6]) == 1
    wall = rect(5.0, 5.0, 5.6, 7.0)
    n = b.add_obstacle(0, wall)
    inside = np.flatnonzero(b.state == OBSTACLE)
    assert n == len(inside) > 0 and np.all(wall.contains(__import__("shapely").points(L[inside])))
    assert b.add_obstacle(0, wall) == 0                       # idempotent
    assert (b.state[[3, 4, 5, 6]] == FREE).all()
    assert b.n_unknown() == len(L) - 4 - n


def test_map_belief_merge_is_a_join():
    L = _lattice()
    a, b, c = MapBelief(L), MapBelief(L), MapBelief(L)
    a.mark_free([1, 2]); a.add_obstacle(0, rect(5, 5, 5.6, 7))
    b.mark_free([2, 3]); b.add_obstacle(1, rect(8, 1, 9, 2))
    c.mark_free([10])
    ab, ba = a.copy(), b.copy()
    ab.merge(b); ba.merge(a)
    assert np.array_equal(ab.state, ba.state) and set(ab.obstacles) == set(ba.obstacles) == {0, 1}   # commutative
    again = ab.copy()
    assert again.merge(b) == 0 and np.array_equal(again.state, ab.state)                              # idempotent
    l, r = ab.copy(), a.copy()
    l.merge(c); r.merge(b); r.merge(c)
    assert np.array_equal(l.state, r.state)                                                           # associative
    assert (ab.state >= a.state).all() and (ab.state >= b.state).all()                                # monotone
    cp = a.copy(); cp.mark_free([50])
    assert a.state[50] == UNKNOWN                                                                     # copies are independent


# ---------------------------------------------------------------- planning on what is known
def test_unknown_space_is_treated_as_free_until_the_obstacle_is_known():
    w = World(WorldConfig(), [rect(5.8, 0.0, 6.2, 9.0)], [[1.0, 1.0]])
    v = w.unknown_view()
    a, b = np.array([3.0, 4.0]), np.array([9.0, 4.0])
    blind = v.space_for({}).planner.distance(a, b)
    informed = v.space_for({0: w.obstacles[0]}).planner.distance(a, b)
    assert blind == pytest.approx(6.0)                        # straight through the (unseen) wall
    assert informed > 8.0                                     # must go round the top once it is known
    assert v.space_for({}) is v.space_for({})                 # cached, shared


def test_gain_respects_known_occlusion_and_is_zero_once_seen():
    w = World(WorldConfig(), [rect(5.8, 3.0, 6.2, 6.0)], [[1.0, 1.0]])
    v = w.unknown_view()
    all_unknown = np.ones(len(v.lattice), dtype=bool)
    free = v.space_for({}).gains(all_unknown)
    walled = v.space_for({0: w.obstacles[0]}).gains(all_unknown)
    assert (walled <= free + 1e-9).all() and (walled < free).any()
    assert v.space_for({}).gains(np.zeros(len(v.lattice), dtype=bool)).sum() == 0


# ---------------------------------------------------------------- simulator wiring
class _Spy(AgentPolicy):
    map_interface = "unknown"
    seen_view = None

    def reset(self, world, agent_id, n_agents):
        self.id, type(self).seen_view = agent_id, world
        self.obs = []

    def step(self, obs):
        self.obs.append(obs)
        return Command()


def test_simulator_rejects_a_model_written_for_the_other_interface():
    w = make_world("empty", n_agents=1)
    with pytest.raises(ValueError, match="unknown-map"):
        Simulator(w, get_model("frontier_greedy"), SimConfig())                       # known_map=True
    with pytest.raises(ValueError, match="known-map"):
        Simulator(w, get_model("greedy_single"), SimConfig(known_map=False))


def test_agent_learns_obstacles_only_by_sensing_them():
    wall_near, wall_far = rect(3.0, 0.5, 3.4, 3.5), rect(10.0, 10.0, 11.0, 11.0)
    w = World(WorldConfig(), [wall_near, wall_far], [[2.0, 2.0]])
    model = DecentralisedModel(_Spy, name="spy")
    Simulator(w, model, SimConfig(max_time=2.0, **UNKNOWN_SIM)).run()
    assert type(_Spy.seen_view).__name__ == "UnknownWorldView"
    pol = model.policies[0]
    first, last = pol.obs[0], pol.obs[-1]
    assert isinstance(last.belief, MapBelief)
    assert set(last.belief.obstacles) == {0}                  # the near wall only: the far one is out of range
    assert 0 in first.new_obstacles                           # reported as a discovery
    st = last.belief.state
    assert (st == FREE).any() and (st == OBSTACLE).any() and (st == UNKNOWN).any()
    assert w.cell_free[st == FREE].all() and not w.cell_free[st == OBSTACLE].any()   # belief never contradicts the truth


def test_reveal_obstacles_at_start_gives_the_map_but_not_the_coverage():
    w = make_world("rooms_and_door", n_agents=2)
    model = DecentralisedModel(_Spy, name="spy")
    Simulator(w, model, SimConfig(max_time=1.0, reveal_obstacles_at_start=True, **UNKNOWN_SIM)).run()
    b = model.policies[0].obs[0].belief
    assert set(b.obstacles) == set(range(len(w.obstacles)))
    assert (b.state == OBSTACLE).sum() == int((~w.cell_free).sum())
    assert model.policies[0].obs[-1].new_obstacles == []     # nothing left to discover
    assert (b.state == FREE).sum() < 0.3 * w.n_targets        # coverage is still unknown


# ---------------------------------------------------------------- the frontier explorer, end to end
def _run(name, n, model, safety="none", seed=0, **cfg):
    w = make_world(name, n_agents=n, seed=seed)
    r = Simulator(w, get_model(model), SimConfig(max_time=1500, stop_when_covered=False, safety=safety,
                                                  **UNKNOWN_SIM, **cfg)).run()
    return w, r


@pytest.mark.parametrize("name,n", [("corridor", 2), ("rooms_and_door", 3), ("sparse", 3), ("cluttered", 2)])
@pytest.mark.parametrize("model", ["frontier_greedy", "frontier_greedy_shared"])
def test_explorer_covers_the_map_stays_clear_of_obstacles_and_knows_when_it_is_done(name, n, model):
    w, r = _run(name, n, model)
    assert r.completed
    assert all(b is not None for b in r.believed_done)        # every drone concludes on its own that it is finished
    # regression: a sidestep towards a candidate inside a KNOWN obstacle once drove a drone into it
    assert r.min_clearance >= w.cfg.drone_radius, r.min_clearance


def test_explorer_with_a_known_map_never_discovers_anything():
    w, r = _run("cluttered", 2, "frontier_greedy", reveal_obstacles_at_start=True)
    assert r.completed and r.min_clearance >= w.cfg.drone_radius
    assert r.extra["map_rebuilds"] == 2                       # one per drone: loading the prior, no later discovery


def test_unknown_map_explorer_agrees_with_the_known_map_greedy_on_an_empty_map():
    """With no obstacles there is nothing to be ignorant of, so it must match the known-map implementation."""
    w = make_world("empty", n_agents=1)
    known = Simulator(w, get_model("greedy_centralised"), SimConfig(max_time=1500)).run()
    unk = Simulator(w, get_model("frontier_greedy"), SimConfig(max_time=1500, **UNKNOWN_SIM)).run()
    assert unk.coverage_time == pytest.approx(known.coverage_time, rel=0.05)


def test_sharing_sends_messages_and_merges_beliefs_but_the_floor_does_not():
    _, floor = _run("rooms_and_door", 2, "frontier_greedy")
    _, shared = _run("rooms_and_door", 2, "frontier_greedy_shared")
    assert floor.n_messages == 0 and floor.extra["cells_merged"] == 0
    assert shared.n_messages > 0 and shared.extra["cells_merged"] > 0
    assert shared.coverage_time < floor.coverage_time         # information pays here (not a general law)


def test_no_messages_are_delivered_beyond_the_communication_radius():
    w = make_world("empty", n_agents=2)
    far = w.cfg.with_(r_comm=0.1)
    w = World(far, [], w.starts, name="far")
    r = Simulator(w, get_model("frontier_greedy_shared"), SimConfig(max_time=30, **UNKNOWN_SIM)).run()
    assert r.extra["cells_merged"] == 0
