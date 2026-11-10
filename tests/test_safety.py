import math

import numpy as np
import pytest

import models  # noqa: F401
from mascov.common.config import SimConfig, WorldConfig
from mascov.common.model import CentralModel, get_model
from mascov.common.types import Command, Plan
from mascov.maps.benchmarks import make_world
from mascov.safety.bvc import bvc_filter, stopping_point
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


def test_stopping_point_formula():
    s = stopping_point(np.array([1.0, 1.0]), np.array([0.6, 0.0]), 1.4)
    assert np.allclose(s, [1.0 + 0.36 / 2.8, 1.0])


def test_filter_leaves_safe_commands_alone():
    a, changed = bvc_filter(np.zeros(2), np.zeros(2), np.array([1.0, 0.0]), np.zeros((0, 2)), np.zeros((0, 2)),
                            dt=0.1, a_max=2.0, d_safe=0.3)
    assert not changed and np.allclose(a, [1.0, 0.0])
    far, _ = bvc_filter(np.zeros(2), np.zeros(2), np.array([1.0, 0.0]), np.array([[20.0, 0.0]]), np.zeros((1, 2)),
                        dt=0.1, a_max=2.0, d_safe=0.3)
    assert np.allclose(far, [1.0, 0.0])


def test_filter_blocks_acceleration_into_a_close_neighbor():
    a, changed = bvc_filter(np.zeros(2), np.array([0.6, 0.0]), np.array([2.0, 0.0]),
                            np.array([[0.5, 0.0]]), np.zeros((1, 2)), dt=0.1, a_max=2.0, d_safe=0.3)
    assert changed and a[0] < 0                                   # must brake, not accelerate


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_forced_collisions_happen_without_the_filter_and_not_with_it(name):
    starts, goals = SCENARIOS[name]
    assert run(starts, goals, "none").collision_events >= 1       # the scenario really is adversarial
    r = run(starts, goals, "bvc")
    assert r.collision_events == 0
    assert r.min_pair_dist >= WorldConfig().d_safe


def test_minimum_communication_radius_matches_the_analytic_estimate():
    cfg = WorldConfig()
    v = cfg.plan_speed
    needed = cfg.d_safe + 2 * v * v / (0.7 * cfg.a_max) + 2 * v * cfg.dt        # ~0.68 m at plan speed
    starts, goals = SCENARIOS["head_on"]
    assert run(starts, goals, "bvc", r_comm=needed + 0.15).collision_events == 0
    assert run(starts, goals, "bvc", r_comm=0.5).collision_events >= 1          # too short to react in time


def test_filter_removes_the_collisions_the_heuristic_rule_could_not():
    w = make_world("sparse", n_agents=4, seed=0)                  # B3 hit 0.035 m here without the filter
    r = Simulator(w, get_model("greedy_uncoordinated"), SimConfig(max_time=600, safety="bvc")).run()
    assert r.completed and r.collision_events == 0 and r.min_pair_dist >= w.cfg.d_safe


def test_filter_costs_little_time_when_the_heuristic_already_avoids_conflicts():
    w = make_world("empty", n_agents=3)
    off = Simulator(w, get_model("greedy_centralised"), SimConfig(max_time=600)).run()
    on = Simulator(w, get_model("greedy_centralised"), SimConfig(max_time=600, safety="bvc")).run()
    assert on.completed and on.coverage_time <= off.coverage_time * 1.1
