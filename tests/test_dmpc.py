"""D-MPC (greedy+dmpc): the local QP on its own, then the policy in the simulator."""
import math

import numpy as np
import pytest

pytest.importorskip("osqp")

import models  # noqa: F401,E402
from mascov.common.config import SimConfig, WorldConfig  # noqa: E402
from mascov.common.model import CentralModel, get_model, registered_models  # noqa: E402
from mascov.common.types import Command  # noqa: E402
from mascov.geometry.polygons import rect  # noqa: E402
from mascov.maps.benchmarks import make_world  # noqa: E402
from mascov.planning.agent_dmpc import DMPCAgentPolicy  # noqa: E402
from mascov.planning.dmpc import DMPCParams, DMPCSolver, obstacle_halfspaces, solve_dmpc  # noqa: E402
from mascov.sim.simulator import Simulator  # noqa: E402
from mascov.world.world import World  # noqa: E402


def straight_ref(p0, direction, speed=0.6, H=10, dt=0.1):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    return np.asarray(p0, float)[None] + d[None] * speed * dt * np.arange(1, H + 1)[:, None]


def test_registered():
    assert "greedy+dmpc" in registered_models()


def test_qp_follows_reference_within_speed_limit():
    prm = DMPCParams()
    p0 = np.array([2.0, 2.0])
    r = solve_dmpc(p0, np.zeros(2), straight_ref(p0, [1, 1]), None, [], [], prm)
    assert r.slack < 1e-4 and not r.info["failed"]
    v = np.linalg.norm(np.diff(r.points, axis=0), axis=1) / prm.dt
    assert v.max() <= prm.v_cap + 1e-3
    assert np.linalg.norm(r.points[-1] - p0) > 0.3                 # it moves towards the reference


def test_qp_respects_walls_and_obstacles():
    prm = DMPCParams()
    obs = obstacle_halfspaces([rect(3.0, 1.0, 4.0, 3.0)])
    p0 = np.array([2.4, 2.0])
    r = solve_dmpc(p0, np.zeros(2), straight_ref(p0, [1, 0], speed=1.0), None, obs, [], prm)
    assert r.points[:, 0].max() <= 3.0 - prm.clearance + 1e-3      # stops at the inflated face
    p0 = np.array([0.6, 6.0])
    r = solve_dmpc(p0, np.zeros(2), straight_ref(p0, [-1, 0], speed=1.0), None, [], [], prm)
    assert r.points[:, 0].min() >= prm.clearance - 1e-3


def _random_case(rng):
    from mascov.geometry.polygons import random_convex_polygon
    polys = [random_convex_polygon(rng, *rng.uniform(1.5, 10.5, 2), float(rng.uniform(0.3, 1.0)), 6)
             for _ in range(int(rng.integers(0, 9)))]
    p0 = rng.uniform(1, 11, 2)
    d = rng.normal(size=2)
    ref = p0[None] + d[None] / np.linalg.norm(d) * rng.uniform(0.03, 0.1) * np.arange(1, 11)[:, None]
    prev = None if rng.random() < 0.3 else ref + rng.normal(scale=0.05, size=ref.shape)
    nbrs = [p0[None] + rng.uniform(-1.5, 1.5, 2)[None] + rng.normal(scale=0.05, size=(10, 2))
            for _ in range(int(rng.integers(0, 6)))]
    return p0, rng.uniform(-0.6, 0.6, 2), ref, prev, obstacle_halfspaces(polys), nbrs


def test_reused_solver_agrees_with_a_fresh_one_and_grows_when_needed():
    """One OSQP object per drone, values updated every step, must give the answers a fresh setup gives (to the
    solver tolerance), also when the number of constraints changes from step to step or exceeds the capacity."""
    rng = np.random.default_rng(1)
    prm = DMPCParams()
    reused = DMPCSolver(prm, capacity=6)               # small on purpose: 8 obstacles + 5 neighbours do not fit
    compared = 0
    for _ in range(150):
        case = _random_case(rng)
        a = solve_dmpc(*case, prm)
        b = reused.solve(*case)
        if a.info["failed"]:
            continue        # a few random cases (5 neighbours packed around the drone) hit OSQP's iteration
                            # limit in a fresh solve too, as they did with the pre-reuse solver
        assert not b.info["failed"], "the reused solver failed where a fresh one succeeded"
        compared += 1
        assert np.abs(a.points - b.points).max() < 1e-2
        assert b.slack == pytest.approx(a.slack, abs=1e-2)
    assert compared >= 140
    assert reused.n_setups > 1                         # the capacity grew at least once


def test_unused_constraint_rows_never_bind():
    """Spare capacity rows (zero coefficients, lower bound -inf) must not change the solution."""
    rng = np.random.default_rng(2)
    prm = DMPCParams()
    small, big = DMPCSolver(prm, capacity=4), DMPCSolver(prm, capacity=40)
    for _ in range(40):
        case = _random_case(rng)
        a, b = small.solve(*case), big.solve(*case)
        if a.info["failed"] or b.info["failed"]:
            continue
        assert np.abs(a.points - b.points).max() < 1e-2


def test_qp_keeps_distance_from_predicted_neighbour():
    prm = DMPCParams()
    p0 = np.array([2.0, 6.0])
    nbr = np.repeat(np.array([[2.8, 6.0]]), prm.horizon, axis=0)    # a drone hovering in the way
    r = solve_dmpc(p0, np.zeros(2), straight_ref(p0, [1, 0], speed=1.0), None, [], [nbr], prm)
    d = np.linalg.norm(r.points[1:] - nbr, axis=1)
    assert d.min() >= prm.d_min - 1e-3


class Swap(CentralModel):
    """Drones fly to fixed goals with the D-MPC trajectory layer (goals instead of greedy viewpoints)."""
    name = "swap"
    decision_period = 0.1

    def __init__(self, goals, coupling):
        self.goals, self.coupling = goals, coupling

    def reset(self, world, ids):
        self.pol = {}
        for i in ids:
            p = DMPCAgentPolicy(coupling=self.coupling)
            p.reset(world, i, len(ids))
            p.select_goal = lambda obs, p=p: None              # no greedy choice: the goal is fixed below
            p._update_goal = lambda obs: None                   # keep the fixed goal
            p.goal_vp, p.goal = -1, np.asarray(self.goals[i], float)
            self.pol[i] = p

    def step(self, obs):
        return {i: self.pol[i].step(o) for i, o in obs.items()}


def run_swap(starts, goals, coupling):
    w = World(WorldConfig(), [], starts, name="swap")
    return Simulator(w, Swap(goals, coupling), SimConfig(max_time=25, stop_when_covered=False)).run()


RING = [[6 + 3 * math.cos(a), 6 + 3 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 5)[:-1]]
RING_OPP = [[6 - 3 * math.cos(a), 6 - 3 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 5)[:-1]]


@pytest.mark.parametrize("starts,goals", [
    ([[3.0, 6.0], [9.0, 6.0]], [[9.0, 6.0], [3.0, 6.0]]),            # head-on
    (RING, RING_OPP),                                                 # four drones through the centre
])
def test_coupling_prevents_collisions_and_off_collides(starts, goals):
    on = run_swap(starts, goals, coupling=True)
    off = run_swap(starts, goals, coupling=False)
    assert on.pair_events == 0 and on.min_pair_dist >= WorldConfig().d_safe
    assert off.pair_events > 0                                        # the coupling is what keeps them apart
    # everyone still gets there
    w = World(WorldConfig(), [], starts, name="swap")
    sim = Simulator(w, Swap(goals, True), SimConfig(max_time=25, stop_when_covered=False))
    sim.run(record_paths=True)
    ends = np.array([p[-1] for p in sim.paths])
    assert np.linalg.norm(ends - np.asarray(goals), axis=1).max() < 0.2


def test_policy_never_fails_a_solve_and_broadcasts_its_prediction():
    """Contract of greedy+dmpc (whole-run quality is graded in test_model_quality.py)."""
    w = make_world("rooms_and_door", n_agents=2, seed=0)
    m = get_model("greedy+dmpc")
    r = Simulator(w, m, SimConfig(max_time=30, stop_when_covered=False)).run()
    assert r.extra["dmpc_solves"] == 2 * r.n_decisions and r.extra["dmpc_failures"] == 0
    assert r.n_messages == r.extra["dmpc_solves"]                       # one prediction per drone per step
    off = Simulator(w, get_model("greedy+dmpc", coupling=False), SimConfig(max_time=5, stop_when_covered=False)).run()
    assert off.n_messages == 0                                          # the ablation sends nothing



# ------------------------------------------------------------------ unknown map: frontier + D-MPC
def _unknown(map_name, n, prior_known=False, **kw):
    w = make_world(map_name, n_agents=n, seed=0)
    r = Simulator(w, get_model("frontier+dmpc", **kw),
                  SimConfig(max_time=40, stop_when_covered=False, known_map=False,
                            reveal_obstacles_at_start=prior_known)).run()
    return w, r


def test_frontier_dmpc_is_an_unknown_map_model():
    assert get_model("frontier+dmpc").map_interface == "unknown"
    w = make_world("empty", n_agents=1)
    with pytest.raises(ValueError):
        Simulator(w, get_model("frontier+dmpc"), SimConfig(known_map=True)).run()


def test_frontier_dmpc_learns_obstacles_only_by_sensing_and_stays_clear():
    w, r = _unknown("cluttered", 2)
    assert r.extra["map_rebuilds"] > 2                     # obstacles discovered one by one while flying
    assert r.obstacle_events == 0 and r.min_clearance >= w.cfg.drone_radius
    assert r.extra["dmpc_failures"] == 0


def test_frontier_dmpc_with_the_map_given_never_rebuilds():
    _, r = _unknown("cluttered", 2, prior_known=True)
    assert r.extra["map_rebuilds"] == 2                    # one per drone: loading the prior, no later discovery


def test_frontier_dmpc_shares_beliefs_only_when_asked():
    _, private = _unknown("rooms_and_door", 2)
    _, shared = _unknown("rooms_and_door", 2, share=True)
    assert private.n_messages == shared.n_messages         # one prediction per drone per step either way
    assert private.extra["belief_merged"] == 0              # predictions only: nothing to merge
    assert shared.extra["belief_merged"] > 0                # beliefs travel with the prediction and are merged
