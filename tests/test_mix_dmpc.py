"""Goal layer x D-MPC flying (models/mix_dmpc): the goal policy runs unchanged, D-MPC flies its goals."""
import numpy as np
import pytest

pytest.importorskip("osqp")

import models  # noqa: F401,E402
from mascov.common.config import SimConfig  # noqa: E402
from mascov.common.model import get_model  # noqa: E402
from mascov.maps.benchmarks import make_world  # noqa: E402
from mascov.sim.simulator import Simulator  # noqa: E402
from models.cbba.explorer import CBBAExplorerPolicy  # noqa: E402
from models.geodesic_partition.model import GeodesicPartitionPolicy  # noqa: E402
from models.greedy_explorer.model import GreedyExplorerPolicy  # noqa: E402
from models.mix_dmpc.model import GoalRecorder  # noqa: E402

COMBOS = {"greedy_explorer+dmpc": GreedyExplorerPolicy, "geodesic_partition+dmpc": GeodesicPartitionPolicy,
          "cbba_explorer+dmpc": CBBAExplorerPolicy}


def _run(name, map_name="rooms_and_door", n=3, max_time=600, **kw):
    w = make_world(map_name, n_agents=n, seed=0)
    m = get_model(name, share=True, seed=0, **kw)
    r = Simulator(w, m, SimConfig(max_time=max_time, known_map=False, stop_when_covered=False)).run()
    return w, m, r


@pytest.mark.parametrize("name,cls", COMBOS.items())
def test_the_goal_layer_is_the_unchanged_policy_and_dmpc_flies_it(name, cls):
    w, m, r = _run(name)
    assert m.map_interface == "unknown" and m.decision_period == 0.1
    for p in m.policies.values():
        assert type(p.inner) is cls and isinstance(p.inner.nav, GoalRecorder)
    assert r.completed and all(b is not None for b in r.believed_done)
    assert r.obstacle_events == 0 and r.min_clearance >= w.cfg.drone_radius
    assert r.pair_events == 0                                       # D-MPC keeps drones apart
    assert r.extra["dmpc_failures"] == 0 and r.extra["rec_goals_set"] > 0


def test_goal_decisions_keep_their_own_cadence():
    """The goal policy decides every 0.5 s (as on its own); D-MPC every 0.1 s."""
    _, m, r = _run("geodesic_partition+dmpc", max_time=10)
    p = m.policies[0]
    assert r.extra["dmpc_solves"] >= 3 * 99                          # 3 drones x ~100 control steps
    assert abs(p.next_goal_t - 10.0) <= 0.5 + 1e-9


def test_a_target_dmpc_gives_up_is_reported_blocked():
    rec = GoalRecorder(None, None)
    rec.set_goal([3.0, 3.0])
    rec.update(0.0, np.array([1.0, 1.0]), np.zeros(2), False)
    assert rec.mode == "go" and not rec.arrived
    rec.block()
    rec.update(0.1, np.array([1.0, 1.0]), np.zeros(2), False)
    assert rec.mode == "blocked"                                    # the goal policy drops it, like a blocked Navigator goal
    rec.set_goal([5.0, 5.0])
    assert rec.mode == "go" and not rec.blocked
    rec.update(0.2, np.array([5.05, 5.0]), np.zeros(2), False)
    assert rec.arrived                                              # within the Navigator's 0.1 m


def test_both_layers_messages_go_out():
    _, _, r = _run("cbba_explorer+dmpc", n=2, max_time=5)
    assert r.extra["goal_cbba_msgs_in"] > 0                          # CBBA consensus still reaches teammates
    assert r.n_messages > 2 * 49                                     # plus a predicted path every 0.1 s
