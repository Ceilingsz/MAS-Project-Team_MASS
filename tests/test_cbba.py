import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

import numpy as np

import models

from mascov.common.config import SimConfig
from mascov.common.model import get_model
from mascov.maps.benchmarks import make_world
from mascov.sim.simulator import Simulator
from models.cbba.cbba_core import CBBAState


def test_cbba_state_consensus_higher_bid():
    st = CBBAState(agent_id=0, n_tasks=5, n_agents=2, max_bundle_size=3)
    # Agent 0 has bundle [0, 1]
    st.bundle = [0, 1]
    st.y[0] = 0
    st.z[0] = 5.0
    st.y[1] = 0
    st.z[1] = 4.0

    # Agent 1 sends message outbidding agent 0 on task 0
    y_1 = np.array([1, -1, -1, -1, -1])
    z_1 = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    s_1 = np.array([0.0, 1.0])

    st.process_message(sender_id=1, y_k=y_1, z_k=z_1, s_k=s_1)

    # Task 0 should be won by Agent 1
    assert st.y[0] == 1
    assert st.z[0] == 10.0
    # Cascade drop: since task 0 was outbid, task 0 and task 1 (appended after 0) should be dropped
    assert st.bundle == []
    # And task 1 should be reset to unassigned
    assert st.y[1] == -1
    assert st.z[1] == 0.0


def test_cbba_state_consensus_lower_bid_ignored():
    st = CBBAState(agent_id=0, n_tasks=5, n_agents=2, max_bundle_size=3)
    st.bundle = [0]
    st.y[0] = 0
    st.z[0] = 5.0

    # Agent 1 sends message with lower bid on task 0
    y_1 = np.array([1, -1, -1, -1, -1])
    z_1 = np.array([3.0, 0.0, 0.0, 0.0, 0.0])
    s_1 = np.array([0.0, 1.0])

    st.process_message(sender_id=1, y_k=y_1, z_k=z_1, s_k=s_1)

    # Agent 0 keeps task 0
    assert st.y[0] == 0
    assert st.z[0] == 5.0
    assert st.bundle == [0]


def test_cbba_state_bundle_construction():
    st = CBBAState(agent_id=0, n_tasks=5, n_agents=2, max_bundle_size=2)

    scores = {0: 1.0, 1: 5.0, 2: 3.0, 3: 0.0, 4: 2.0}

    def score_fn(task, cur_bundle):
        return scores.get(task, 0.0)

    st.construct_bundle(score_fn)
    # Should pick task 1 (score 5.0) first, then task 2 (score 3.0) second, hitting max_bundle_size=2
    assert st.bundle == [1, 2]
    assert st.y[1] == 0
    assert st.y[2] == 0
    assert st.z[1] == 5.0
    assert st.z[2] == 3.0


def test_cbba_simulation_empty_2drones():
    world = make_world("empty", n_agents=2, size=12.0, seed=0)
    model = get_model("cbba", delta=1.0, max_bundle_size=4)
    sc = SimConfig(max_time=200.0, stop_when_covered=True, safety="none")

    sim = Simulator(world, model, sc)
    res = sim.run()

    assert res.completed
    assert res.coverage_time is not None
    assert res.coverage_time < 150.0  # Should be fast and coordinated
    assert res.n_messages > 0  # Should have communicated


def test_cbba_simulation_sparse_known():
    world = make_world("sparse", n_agents=2, size=12.0, seed=0)
    model = get_model("cbba", delta=1.0, max_bundle_size=4)
    sc = SimConfig(max_time=250.0, stop_when_covered=True, safety="none", known_map=True)

    sim = Simulator(world, model, sc)
    res = sim.run()

    assert res.completed
    assert res.coverage_time is not None
    assert res.obstacle_events == 0  # drone-drone safety is graded in test_model_quality.py: the shared Navigator has no avoidance


def test_cbba_simulation_sparse_unknown():
    world = make_world("sparse", n_agents=2, size=12.0, seed=0)
    model = get_model("cbba", delta=1.0, max_bundle_size=4)
    sc = SimConfig(max_time=300.0, stop_when_covered=True, safety="none", hide_obstacles=True)

    sim = Simulator(world, model, sc)
    res = sim.run()

    assert res.completed
    assert res.coverage_time is not None


if __name__ == "__main__":
    print("Running test_cbba_state_consensus_higher_bid...")
    test_cbba_state_consensus_higher_bid()
    print("Running test_cbba_state_consensus_lower_bid_ignored...")
    test_cbba_state_consensus_lower_bid_ignored()
    print("Running test_cbba_state_bundle_construction...")
    test_cbba_state_bundle_construction()
    print("Running test_cbba_simulation_empty_2drones...")
    test_cbba_simulation_empty_2drones()
    print("Running test_cbba_simulation_sparse_known...")
    test_cbba_simulation_sparse_known()
    print("Running test_cbba_simulation_sparse_unknown...")
    test_cbba_simulation_sparse_unknown()
    print("\n>>> ALL CBBA TESTS PASSED! <<<")
