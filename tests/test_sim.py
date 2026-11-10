import numpy as np
import pytest

from mascov.common.config import SimConfig, WorldConfig
from mascov.common.model import CentralModel, DecentralisedModel, AgentPolicy, get_model, registered_models
from mascov.common.types import Command, Message, Plan
from mascov.maps.benchmarks import make_world
from mascov.metrics.bounds import lower_bound
from mascov.sim.dynamics import step_double_integrator
from mascov.sim.simulator import Simulator
from mascov.world.world import World
from models.greedy_single import GreedySingle


def test_plan_reference_and_polyline():
    p = Plan.from_polyline([[0, 0], [3, 0], [3, 4]], t0=2.0, speed=1.0)
    assert p.t_start == 2.0 and p.t_end == pytest.approx(9.0) and p.length == pytest.approx(7.0)
    pos, vel = p.ref(3.5)
    assert np.allclose(pos, [1.5, 0]) and np.allclose(vel, [1, 0])
    pos, vel = p.ref(7.0)
    assert np.allclose(pos, [3, 2]) and np.allclose(vel, [0, 1])
    assert np.allclose(p.ref(100)[0], [3, 4]) and np.allclose(p.ref(100)[1], 0)


def test_double_integrator_exact_and_limits():
    p, v = step_double_integrator(np.zeros(2), np.zeros(2), np.array([1.0, 0.0]), 1.0, 10, 10)
    assert np.allclose(p, [0.5, 0]) and np.allclose(v, [1, 0])
    p, v = step_double_integrator(np.zeros(2), np.zeros(2), np.array([50.0, 0.0]), 1.0, 10, 2.0)
    assert np.allclose(v, [2, 0])                                 # acceleration clipped to a_max
    p, v = step_double_integrator(np.zeros(2), np.array([5.0, 0]), np.array([2.0, 0.0]), 1.0, 1.0, 2.0)
    assert np.linalg.norm(v) <= 1.0 + 1e-12                       # speed clipped to v_max


def test_registry():
    assert "greedy_single" in registered_models()
    assert isinstance(get_model("greedy_single"), GreedySingle)
    with pytest.raises(KeyError):
        get_model("nope")


def test_greedy_covers_every_benchmark_map():
    for name in ["empty", "sparse", "corridor", "rooms_and_door", "cluttered"]:
        w = make_world(name, n_agents=1, seed=1)
        r = Simulator(w, GreedySingle(), SimConfig(max_time=2500)).run()
        assert r.completed, name
        assert r.collision_events == 0, name
        assert r.coverage_time >= lower_bound(w)["best"] - 1e-6, name     # a lower bound is a floor
        # the measured tracking error must respect the eps planners are told to assume
        assert r.max_tracking_error[0] <= w.cfg.eps_track, (name, r.max_tracking_error)


def test_coverage_is_monotone_and_ends_complete():
    w = make_world("sparse", n_agents=1, seed=0)
    r = Simulator(w, GreedySingle(), SimConfig(max_time=2500)).run()
    cov = r.log.column("coverage")
    assert np.all(np.diff(cov) >= -1e-12) and cov[-1] == pytest.approx(1.0)


def test_planner_free_space_keeps_drone_away_from_walls():
    w = make_world("corridor", n_agents=1)
    r = Simulator(w, GreedySingle(), SimConfig(max_time=2500)).run()
    assert r.min_clearance >= w.cfg.drone_radius


class _Stationary(CentralModel):
    name = "stationary"
    def reset(self, world, agent_ids): self.n = len(agent_ids)
    def step(self, obs): return {}


def test_collision_is_detected():
    cfg = WorldConfig()
    w = World(cfg, [], [[3.0, 3.0], [3.2, 3.0]], name="close")      # 0.2 m apart < d_safe = 0.3
    r = Simulator(w, _Stationary(), SimConfig(max_time=2.0)).run()
    assert r.collision_events == 1 and r.min_pair_dist == pytest.approx(0.2)


def test_stationary_drones_cover_only_what_they_see():
    w = make_world("empty", n_agents=2)
    r = Simulator(w, _Stationary(), SimConfig(max_time=3.0)).run()
    assert not r.completed
    assert 0 < r.log.column("coverage")[-1] < 0.3


class _Chatty(AgentPolicy):
    def reset(self, world, agent_id, n_agents): self.id = agent_id; self.heard = []
    def step(self, obs):
        self.heard += [(m.sender, m.payload) for m in obs.inbox]
        return Command(messages=[Message(self.id, "hello", obs.t)])


def test_messages_respect_communication_radius():
    cfg = WorldConfig(r_comm=2.0)
    w = World(cfg, [], [[1.0, 1.0], [2.0, 1.0], [9.0, 9.0]], name="comm")   # drones 0,1 close; 2 far
    model = DecentralisedModel(_Chatty, name="chatty")
    Simulator(w, model, SimConfig(max_time=3.0)).run()
    heard = {i: {s for s, _ in p.heard} for i, p in model.policies.items()}
    assert heard[0] == {1} and heard[1] == {0} and heard[2] == set()


def test_decentralised_model_gets_only_own_observation():
    seen = {}
    class Spy(AgentPolicy):
        def reset(self, world, agent_id, n_agents): self.id = agent_id
        def step(self, obs):
            seen.setdefault(self.id, set()).add(obs.agent_id)
            return Command()
    w = make_world("empty", n_agents=3)
    Simulator(w, DecentralisedModel(Spy), SimConfig(max_time=2.0)).run()
    assert all(v == {k} for k, v in seen.items())


def test_victims_are_detected_when_seen():
    cfg = WorldConfig()
    w = World(cfg, [], [[1.0, 1.0]], victims=[[1.5, 1.5], [10.0, 10.0]], name="victims")
    r = Simulator(w, _Stationary(), SimConfig(max_time=1.0)).run()
    assert set(r.victims_found) == {0}
