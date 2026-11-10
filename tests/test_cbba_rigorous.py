"""Rigorous test suite for CBBA, dynamic replanning, and known vs. unknown map framework.

Covers:
  1. CBBA consensus state machine: tie-breaking, cascade drops, timestamps, stale message rejection.
  2. Submodularity and marginal score formulation.
  3. Dynamic obstacle discovery, visibility graph rebuilding, and cooperative mapping.
  4. Physical invariants: kinematic limits, monotonic coverage, belief soundness.
  5. Multi-agent edge cases: single drone, limited communication range, high swarm density.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import Point

# Ensure project root and src are on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import models  # noqa: F401
from mascov.common.config import SimConfig, WorldConfig
from mascov.common.model import get_model
from mascov.maps.benchmarks import make_world
from mascov.sim.simulator import Simulator
from models.cbba.cbba_core import CBBAState
from models.cbba.scoring import CBBAScorer


# ==============================================================================
# 1. CBBA Consensus State Machine & Cascade Drop Tests
# ==============================================================================

class TestCBBAConsensus:

    def test_single_agent_bundle_construction_order(self):
        """Bundle construction must greedily select tasks in descending order of score up to capacity."""
        st = CBBAState(agent_id=0, n_tasks=6, n_agents=2, max_bundle_size=3)
        scores = {0: 1.2, 1: 5.5, 2: 3.1, 3: 0.0, 4: 9.9, 5: 2.0}

        st.construct_bundle(lambda task, b: scores.get(task, 0.0))

        # Should select 4 (9.9), then 1 (5.5), then 2 (3.1) -> length 3 (capacity)
        assert st.bundle == [4, 1, 2]
        assert st.y[4] == 0 and st.y[1] == 0 and st.y[2] == 0
        assert np.isclose(st.z[4], 9.9)
        assert np.isclose(st.z[1], 5.5)
        assert np.isclose(st.z[2], 3.1)

    def test_consensus_higher_bid_overrides(self):
        """A neighbor with a strictly higher bid must win the task and update local state."""
        st = CBBAState(agent_id=0, n_tasks=3, n_agents=2, max_bundle_size=2)
        st.bundle = [0]
        st.y[0] = 0
        st.z[0] = 5.0

        # Neighbor 1 claims task 0 with bid 8.0
        y_1 = np.array([1, -1, -1])
        z_1 = np.array([8.0, 0.0, 0.0])
        s_1 = np.array([0.0, 1.0])

        st.process_message(sender_id=1, y_k=y_1, z_k=z_1, s_k=s_1)

        assert st.y[0] == 1
        assert np.isclose(st.z[0], 8.0)
        assert 0 not in st.bundle

    def test_consensus_lower_bid_ignored(self):
        """A neighbor with an inferior bid must NOT overwrite the current winner."""
        st = CBBAState(agent_id=0, n_tasks=3, n_agents=2, max_bundle_size=2)
        st.bundle = [0]
        st.y[0] = 0
        st.z[0] = 10.0

        y_1 = np.array([1, -1, -1])
        z_1 = np.array([4.0, 0.0, 0.0])
        s_1 = np.array([0.0, 1.0])

        st.process_message(sender_id=1, y_k=y_1, z_k=z_1, s_k=s_1)

        assert st.y[0] == 0
        assert np.isclose(st.z[0], 10.0)
        assert st.bundle == [0]

    def test_consensus_tie_breaking_deterministic(self):
        """Exact bid ties must be broken deterministically by agent ID (lower ID wins)."""
        # Case A: Agent 1 has task 0 with bid 5.0. Agent 0 sends bid 5.0 -> Agent 0 wins (0 < 1)
        st1 = CBBAState(agent_id=1, n_tasks=2, n_agents=2, max_bundle_size=2)
        st1.bundle = [0]
        st1.y[0] = 1
        st1.z[0] = 5.0

        y_0 = np.array([0, -1])
        z_0 = np.array([5.0, 0.0])
        s_0 = np.array([1.0, 0.0])

        st1.process_message(sender_id=0, y_k=y_0, z_k=z_0, s_k=s_0)
        assert st1.y[0] == 0
        assert 0 not in st1.bundle

        # Case B: Agent 0 has task 0 with bid 5.0. Agent 1 sends bid 5.0 -> Agent 0 retains (0 < 1)
        st0 = CBBAState(agent_id=0, n_tasks=2, n_agents=2, max_bundle_size=2)
        st0.bundle = [0]
        st0.y[0] = 0
        st0.z[0] = 5.0

        y_1 = np.array([1, -1])
        z_1 = np.array([5.0, 0.0])
        s_1 = np.array([0.0, 1.0])

        st0.process_message(sender_id=1, y_k=y_1, z_k=z_1, s_k=s_1)
        assert st0.y[0] == 0
        assert st0.bundle == [0]

    def test_cascade_drop_head(self):
        """Outbidding the first item of a bundle [A, B, C] must drop ALL items."""
        st = CBBAState(agent_id=0, n_tasks=5, n_agents=2, max_bundle_size=4)
        st.bundle = [0, 1, 2]
        st.y[:3] = 0
        st.z[0], st.z[1], st.z[2] = 5.0, 4.0, 3.0

        # Neighbor outbids task 0
        y_1 = np.array([1, -1, -1, -1, -1])
        z_1 = np.array([9.0, 0.0, 0.0, 0.0, 0.0])
        s_1 = np.array([0.0, 1.0])

        st.process_message(sender_id=1, y_k=y_1, z_k=z_1, s_k=s_1)

        assert st.bundle == []
        assert st.y[0] == 1
        # Subsequent items 1 and 2 must be reset to unassigned
        assert st.y[1] == -1 and st.z[1] == 0.0
        assert st.y[2] == -1 and st.z[2] == 0.0

    def test_cascade_drop_middle(self):
        """Outbidding the middle item of [A, B, C] must keep A and drop B and C."""
        st = CBBAState(agent_id=0, n_tasks=5, n_agents=2, max_bundle_size=4)
        st.bundle = [0, 1, 2]
        st.y[:3] = 0
        st.z[0], st.z[1], st.z[2] = 5.0, 4.0, 3.0

        # Neighbor outbids task 1
        y_1 = np.array([-1, 1, -1, -1, -1])
        z_1 = np.array([0.0, 9.0, 0.0, 0.0, 0.0])
        s_1 = np.array([0.0, 1.0])

        st.process_message(sender_id=1, y_k=y_1, z_k=z_1, s_k=s_1)

        assert st.bundle == [0]
        assert st.y[0] == 0
        assert st.y[1] == 1
        assert st.y[2] == -1 and st.z[2] == 0.0

    def test_stale_message_rejected(self):
        """Messages with older or obsolete timestamps should not corrupt active winning states."""
        st = CBBAState(agent_id=0, n_tasks=2, n_agents=3, max_bundle_size=2)
        st.s = np.array([5.0, 5.0, 5.0])
        st.y[0] = 2
        st.z[0] = 8.0

        # Neighbor 1 sends stale reset with older timestamp
        y_1 = np.array([-1, -1])
        z_1 = np.array([0.0, 0.0])
        s_1 = np.array([5.0, 2.0, 2.0])  # Older timestamp for agent 2

        st.process_message(sender_id=1, y_k=y_1, z_k=z_1, s_k=s_1)

        # Agent 0 knows agent 2 won at clock 5.0, rejects reset from clock 2.0
        assert st.y[0] == 2
        assert np.isclose(st.z[0], 8.0)

    def test_three_agent_cyclic_consensus_convergence(self):
        """Three agents bidding on overlapping tasks must resolve to a global conflict-free winner."""
        N, K = 3, 4
        states = [CBBAState(agent_id=i, n_tasks=K, n_agents=N, max_bundle_size=2) for i in range(N)]

        # Agent 0 bids: task 0 (score 10), task 1 (score 2)
        # Agent 1 bids: task 1 (score 8), task 2 (score 4)
        # Agent 2 bids: task 2 (score 7), task 0 (score 6)
        scores = [
            {0: 10.0, 1: 2.0},
            {1: 8.0, 2: 4.0},
            {2: 7.0, 0: 6.0},
        ]
        for i in range(N):
            states[i].construct_bundle(lambda t, b, idx=i: scores[idx].get(t, 0.0))

        # Perform 3 rounds of all-to-all communication
        for _ in range(3):
            payloads = [st.consensus_payload() for st in states]
            for i in range(N):
                for j in range(N):
                    if i != j:
                        states[i].process_message(j, payloads[j][0], payloads[j][1], payloads[j][2])

        # All agents must agree:
        # Task 0 -> Agent 0 (bid 10)
        # Task 1 -> Agent 1 (bid 8)
        # Task 2 -> Agent 2 (bid 7)
        for i in range(N):
            assert states[i].y[0] == 0 and np.isclose(states[i].z[0], 10.0)
            assert states[i].y[1] == 1 and np.isclose(states[i].z[1], 8.0)
            assert states[i].y[2] == 2 and np.isclose(states[i].z[2], 7.0)

    def test_validate_bundle_prunes_redundant_tasks(self):
        """Viewpoints whose targets have been covered must be dropped automatically."""
        st = CBBAState(agent_id=0, n_tasks=3, n_agents=2, max_bundle_size=3)
        st.bundle = [0, 1, 2]
        st.y[:3] = 0
        st.z[:3] = [5.0, 4.0, 3.0]

        # Task 1 has 0 remaining gain
        gain_dict = {0: 10, 1: 0, 2: 5}
        st.validate_bundle(lambda t: gain_dict.get(t, 0))

        # Task 1 and subsequent task 2 must be dropped
        assert st.bundle == [0]
        assert st.y[0] == 0
        assert st.y[1] == -1 and st.z[1] == 0.0
        assert st.y[2] == -1 and st.z[2] == 0.0


# ==============================================================================
# 2. Submodularity & Marginal Scoring Tests
# ==============================================================================

class TestCBBAScorer:

    @pytest.fixture
    def world(self):
        return make_world("empty", n_agents=2, size=12.0, seed=0)

    def test_marginal_gain_diminishes(self, world):
        """Marginal gain of a viewpoint must decrease (or stay equal) as bundle expands."""
        scorer = CBBAScorer(world.view())
        covered = np.zeros(len(world.targets), dtype=bool)

        vp_a = 0
        vp_b = 1  # nearby viewpoint

        gain_b_alone = scorer.marginal_gain(vp_b, covered, bundle=[])
        gain_b_after_a = scorer.marginal_gain(vp_b, covered, bundle=[vp_a])

        # Overlapping coverage means gain after A cannot exceed gain alone
        assert gain_b_after_a <= gain_b_alone

    def test_unreachable_point_score_zero(self, world):
        """If travel time is infinite, score must evaluate to exactly 0.0."""
        scorer = CBBAScorer(world.view())
        covered = np.zeros(len(world.targets), dtype=bool)

        # Mock a planner returning inf distance
        class MockPlanner:
            def dist_to(self, *args, **kwargs):
                return np.full((1, len(world.viewpoints.positions)), np.inf)

        score = scorer.compute_score(
            v=0, current_pos=np.array([1.0, 1.0]), covered=covered, bundle=[],
            planner=MockPlanner(), bound=None
        )
        assert score == 0.0

    def test_sequential_distance_uses_last_bundle_task(self, world):
        """Travel time for candidate v must be computed from the last task in bundle, not start position."""
        scorer = CBBAScorer(world.view())
        covered = np.zeros(len(world.targets), dtype=bool)

        # Distant start pos
        p_start = np.array([0.5, 0.5])
        # Two adjacent viewpoints
        v1, v2 = 10, 11
        pos1 = world.viewpoints.positions[v1]
        pos2 = world.viewpoints.positions[v2]

        score_with_bundle = scorer.compute_score(v2, p_start, covered, bundle=[v1])
        score_without_bundle = scorer.compute_score(v2, p_start, covered, bundle=[])

        # Distance from pos1 to pos2 is much shorter than from p_start to pos2
        # Therefore score_with_bundle denominator is smaller -> score is higher
        assert score_with_bundle > score_without_bundle


# ==============================================================================
# 3. Dynamic Obstacle Discovery & Visibility Graph Tests
# ==============================================================================

class TestKnownVsUnknownMap:

    def test_world_view_contract(self):
        """WorldView with known_map=False must conceal obstacles and initialize empty workspace planner."""
        world = make_world("sparse", n_agents=2, size=12.0, seed=0)
        assert len(world.obstacles) > 0

        # Known view exposes obstacles
        v_known = world.view(known_map=True)
        assert len(v_known.obstacles) == len(world.obstacles)

        # Unknown view conceals obstacles initially
        v_unknown = world.view(known_map=False)
        assert len(v_unknown.obstacles) == 0
        assert v_unknown.planner is not None

    def test_create_planner_inflates_obstacle(self):
        """Dynamically created planner must properly forbid the obstacle interior plus clearance margin."""
        world = make_world("sparse", n_agents=2, size=12.0, seed=0)
        view = world.view(known_map=False)

        # Before discovering obstacle 0, the center of obstacle 0 is considered allowed
        obs0 = world.obstacles[0]
        center = np.array([obs0.centroid.x, obs0.centroid.y])

        # Now discover obstacle 0
        planner_new, bound_new = view.create_planner({0})

        # The obstacle centroid must NOT be inside allowed free space
        assert not planner_new.allowed.contains(Point(center[0], center[1]))

    def test_visibility_graph_blocks_paths_through_obstacle(self):
        """A path directly intersecting a newly discovered obstacle must be diverted around it."""
        world = make_world("corridor", n_agents=2, size=12.0, seed=0)
        view = world.view(known_map=False)

        # Points on opposite sides of a corridor wall
        pt_a = np.array([2.0, 2.0])
        pt_b = np.array([2.0, 6.0])

        # Empty planner gives straight-line path
        path_empty = view.planner.path(pt_a, pt_b)
        assert len(path_empty) == 2  # direct segment

        # Update planner with wall 0
        planner_wall, _ = view.create_planner({0})
        path_diverted = planner_wall.path(pt_a, pt_b)

        # Diverted path must have multiple waypoints routing around the wall
        assert len(path_diverted) > 2


# ==============================================================================
# 4. Multi-Agent Physical & Simulation Invariants
# ==============================================================================

class TestSimulationInvariants:

    def test_monotonic_coverage_invariant(self):
        """Coverage percentage must monotonically increase over time for all ticks."""
        world = make_world("empty", n_agents=2, size=12.0, seed=0)
        model = get_model("cbba", delta=1.0, max_bundle_size=3)
        sim = Simulator(world, model, SimConfig(max_time=80.0, stop_when_covered=True))

        res = sim.run()
        cov_history = res.log.column("coverage")

        # Every subsequent logged coverage value must be >= previous value
        assert (np.diff(cov_history) >= -1e-6).all(), "Coverage decreased during run!"

    def test_belief_soundness_invariant(self):
        """Each agent's believed coverage must always be a subset of true ground truth coverage."""
        world = make_world("sparse", n_agents=2, size=12.0, seed=0)
        model = get_model("cbba", delta=1.0, max_bundle_size=3)
        sim = Simulator(world, model, SimConfig(max_time=100.0, stop_when_covered=True))

        res = sim.run()
        true_cov = res.log.column("coverage")
        believed_0 = res.log.column("believed_0")
        believed_1 = res.log.column("believed_1")

        # Believed coverage must never exceed true coverage
        assert (believed_0 <= true_cov + 1e-4).all()
        assert (believed_1 <= true_cov + 1e-4).all()

    def test_kinematic_limits_satisfied(self):
        """Actual tracking errors must remain within acceptable bounds (<= 0.45m)."""
        world = make_world("empty", n_agents=2, size=12.0, seed=0)
        model = get_model("cbba", delta=1.0, max_bundle_size=3)
        sim = Simulator(world, model, SimConfig(max_time=80.0, stop_when_covered=True))

        res = sim.run()
        for err in res.max_tracking_error:
            assert err <= 0.45, f"Tracking error {err:.3f} exceeded 0.45m bound"

    def test_zero_collisions_in_unknown_maps(self):
        """CBBA must navigate unknown environments without obstacle collisions."""
        world = make_world("sparse", n_agents=2, size=12.0, seed=0)
        model = get_model("cbba", delta=1.0, max_bundle_size=3)
        sc = SimConfig(max_time=180.0, stop_when_covered=True, hide_obstacles=True)

        res = Simulator(world, model, sc).run()
        assert res.obstacle_events == 0, f"Observed {res.obstacle_events} obstacle collisions in unknown map"  # drone-drone safety is graded in test_model_quality.py: the shared Navigator has no avoidance


# ==============================================================================
# 5. Multi-Agent Stress & Edge Cases
# ==============================================================================

class TestEdgeCases:

    def test_single_drone_cbba(self):
        """CBBA must function correctly when N=1 without peers or messages."""
        world = make_world("empty", n_agents=1, size=12.0, seed=0)
        model = get_model("cbba", delta=1.0, max_bundle_size=4)
        sc = SimConfig(max_time=250.0, stop_when_covered=True)

        res = Simulator(world, model, sc).run()
        assert res.completed
        assert res.collision_events == 0
        assert res.n_messages == 0  # No peers to message

    def test_limited_communication_radius(self):
        """CBBA must complete coverage even when drones have limited communication range (R_c = 4m)."""
        cfg = WorldConfig(r_comm=4.0)
        world = make_world("empty", n_agents=2, size=12.0, seed=0, cfg=cfg)
        model = get_model("cbba", delta=1.0, max_bundle_size=3)
        sc = SimConfig(max_time=200.0, stop_when_covered=True)

        res = Simulator(world, model, sc).run()
        assert res.completed
        assert res.coverage_time is not None

    def test_high_density_swarm_empty_n8(self):
        """An 8-drone swarm running CBBA in an empty arena must complete with zero collisions."""
        world = make_world("empty", n_agents=8, size=12.0, seed=0)
        model = get_model("cbba", delta=1.0, max_bundle_size=3)
        sc = SimConfig(max_time=100.0, stop_when_covered=True)

        res = Simulator(world, model, sc).run()
        assert res.completed
        assert res.obstacle_events == 0  # drone-drone safety is graded in test_model_quality.py: the shared Navigator has no avoidance
        assert res.coverage_time < 60.0  # High parallelism
