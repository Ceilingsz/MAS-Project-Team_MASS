import math

import numpy as np
import pytest

import models  # noqa: F401
from mascov.common.config import SimConfig, WorldConfig
from mascov.common.model import CentralModel, get_model
from mascov.common.types import Command, Plan
from mascov.maps.benchmarks import make_world
from mascov.safety.bvc import bvc_filter
from mascov.sim.simulator import Simulator
from mascov.world.world import World


class Scripted(CentralModel):
    """Fixed straight-line plans, ignoring everyone: the worst case for a safety filter."""
    name = "scripted"

    def __init__(self, goals):
        self.goals = goals

    def reset(self, world, ids):
        self.sent, self.speed = False, world.cfg.plan_speed

    def step(self, obs):
        if self.sent:
            return {}
        self.sent = True
        return {i: Command(plan=Plan.from_polyline([o.pos, self.goals[i]], 0.0, self.speed)) for i, o in obs.items()}


def run(starts, goals, safety, r_comm=math.inf):
    w = World(WorldConfig(r_comm=r_comm), [], starts, name="adv")
    return Simulator(w, Scripted(goals), SimConfig(max_time=40, stop_when_covered=False, safety=safety)).run()


RING = [[6 + 4 * math.cos(a), 6 + 4 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 7)[:-1]]
RING_OPP = [[6 - 4 * math.cos(a), 6 - 4 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 7)[:-1]]
SCENARIOS = {
    "head_on": ([[2, 6], [10, 6]], [[10, 6], [2, 6]]),
    "crossing": ([[2, 6], [6, 2]], [[10, 6], [6, 10]]),
    "four_swap": ([[2, 2], [10, 2], [10, 10], [2, 10]], [[10, 10], [2, 10], [2, 2], [10, 2]]),
    "four_converge": ([[2, 6], [10, 6], [6, 2], [6, 10]], [[6, 6]] * 4),
    "six_ring": (RING, RING_OPP),
}


def test_filter_leaves_safe_commands_alone():
    v, changed = bvc_filter(np.zeros(2), np.array([1.0, 0.0]), np.zeros((0, 2)), dt=0.1, v_max=1.0, d_safe=0.3)
    assert not changed and np.allclose(v, [1.0, 0.0])
    far, changed = bvc_filter(np.zeros(2), np.array([1.0, 0.0]), np.array([[20.0, 0.0]]), dt=0.1, v_max=1.0, d_safe=0.3)
    assert not changed and np.allclose(far, [1.0, 0.0])


def test_filter_limits_the_step_towards_a_close_neighbor():
    # neighbour 0.5 m ahead: each drone may close at most (0.5 - d_safe) / 2 = 0.1 m this step (less the buffer)
    v, changed = bvc_filter(np.zeros(2), np.array([1.0, 0.0]), np.array([[0.5, 0.0]]), dt=0.1, v_max=1.0, d_safe=0.3)
    assert changed and 0.0 <= v[0] * 0.1 <= 0.1


def test_filter_never_commands_a_step_that_leaves_the_cell():
    rng = np.random.default_rng(0)
    d_safe, dt = 0.3, 0.1
    for _ in range(300):
        k = int(rng.integers(1, 5))
        ang, dist = rng.uniform(0, 2 * math.pi, k), rng.uniform(0.4, 1.5, k)        # every neighbour currently >= d_safe away
        nbrs = np.c_[dist * np.cos(ang), dist * np.sin(ang)]
        v, _ = bvc_filter(np.zeros(2), rng.uniform(-1, 1, 2), nbrs, dt=dt, v_max=1.0, d_safe=d_safe)
        assert np.linalg.norm(v) <= 1.0 + 1e-9
        n = nbrs / np.linalg.norm(nbrs, axis=1, keepdims=True)
        assert np.all(n @ (v * dt) <= 0.5 * np.linalg.norm(nbrs, axis=1) - 0.5 * d_safe + 1e-9)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_forced_collisions_happen_without_the_filter_and_not_with_it(name):
    starts, goals = SCENARIOS[name]
    assert run(starts, goals, "none").collision_events >= 1       # the scenario really is adversarial
    r = run(starts, goals, "bvc")
    assert r.collision_events == 0
    assert r.min_pair_dist >= WorldConfig().d_safe


def test_no_collisions_when_the_communication_radius_covers_one_step_of_closing_speed():
    cfg = WorldConfig()
    needed = cfg.d_safe + 2 * cfg.v_max * cfg.dt + 0.02       # a neighbour must be sensed before it can get within d_safe in one step
    for name in ("head_on", "four_swap"):
        starts, goals = SCENARIOS[name]
        assert run(starts, goals, "bvc", r_comm=needed + 0.05).collision_events == 0, name


def test_filter_makes_a_colliding_model_collision_free():
    """greedy_uncoordinated has no drone-drone avoidance and collides here; the filter alone must fix that."""
    w = make_world("sparse", n_agents=4, seed=0)
    raw = Simulator(w, get_model("greedy_uncoordinated"), SimConfig(max_time=600)).run()
    r = Simulator(w, get_model("greedy_uncoordinated"), SimConfig(max_time=600, safety="bvc")).run()
    assert raw.pair_events > 0                                    # the scenario really is unsafe without it
    assert r.completed and r.collision_events == 0 and r.min_pair_dist >= w.cfg.d_safe


def test_filter_costs_little_time_when_drones_rarely_conflict():
    w = make_world("empty", n_agents=3)
    off = Simulator(w, get_model("greedy_centralised"), SimConfig(max_time=600)).run()
    on = Simulator(w, get_model("greedy_centralised"), SimConfig(max_time=600, safety="bvc")).run()
    assert on.completed and on.coverage_time <= off.coverage_time * 1.1
