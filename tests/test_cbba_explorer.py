"""CBBA / CBAA for the unknown-map interface (models/cbba/explorer.py)."""
import numpy as np
import pytest

import models  # noqa: F401
from mascov.common.config import SimConfig
from mascov.common.model import get_model
from mascov.maps.benchmarks import make_world
from mascov.sim.simulator import Simulator

UNKNOWN = dict(known_map=False)


def _run(name, map_name="rooms_and_door", n=3, seed=0, max_time=600, record=False, sim=None, **kw):
    w = make_world(map_name, n_agents=n, seed=seed)
    m = get_model(name, seed=seed, **kw)
    sim = Simulator(w, m, SimConfig(max_time=max_time, stop_when_covered=False, seed=seed, **UNKNOWN, **(sim or {})))
    return w, m, sim, sim.run(record_paths=record)


@pytest.mark.parametrize("name", ["cbba_explorer", "cbaa_explorer"])
def test_it_is_an_unknown_map_model_and_the_runner_runs_it_on_unknown_maps(name):
    from mascov.experiments.runner import filter_jobs
    from mascov.experiments.spec import Job
    assert get_model(name).map_interface == "unknown"
    job = Job(exp="t", label="x", model=name, kwargs={}, role="method", map="empty", n_agents=2, seed=0, size=12.0,
              max_time=10.0, map_prior="unknown", levers={}, world={}, sim={})
    ok, skipped = filter_jobs([job])
    assert ok and not skipped


@pytest.mark.parametrize("name,share", [("cbba_explorer", False), ("cbba_explorer", True),
                                        ("cbaa_explorer", False), ("cbaa_explorer", True)])
@pytest.mark.parametrize("map_name", ["rooms_and_door", "cluttered"])
def test_covers_the_map_stays_clear_of_obstacles_and_every_drone_knows_when_it_is_done(name, share, map_name):
    w, _, _, r = _run(name, map_name, share=share)
    assert r.completed
    assert all(b is not None for b in r.believed_done)
    assert max(r.believed_done) >= r.coverage_time - 1e-6           # the team never stops before the truth
    assert r.obstacle_events == 0 and r.min_clearance >= w.cfg.drone_radius


def test_cbaa_holds_at_most_one_task_and_cbba_more():
    _, cbaa, _, _ = _run("cbaa_explorer", max_time=20)
    _, cbba, _, _ = _run("cbba_explorer", max_time=20)
    assert all(p.state.max_bundle_size == 1 and len(p.state.bundle) <= 1 for p in cbaa.policies.values())
    assert all(p.state.max_bundle_size == 4 for p in cbba.policies.values())


def test_drones_in_contact_agree_on_disjoint_tasks():
    """After consensus no task is in two bundles, and a drone only believes in another drone's claim if that drone's
    last report includes the task (the guard against the stale-claim deadlock; messages lag one decision, so the
    owner's CURRENT bundle may already have moved on, which the next decision resolves)."""
    _, m, _, _ = _run("cbba_explorer", "empty", n=4, max_time=15)
    pol = list(m.policies.values())
    bundles = [set(p.state.bundle) for p in pol]
    for i in range(len(bundles)):
        for j in range(i + 1, len(bundles)):
            assert not bundles[i] & bundles[j]
    for p in pol:
        for v in np.flatnonzero((p.state.y >= 0) & (p.state.y != p.id)):
            owner = int(p.state.y[v])
            assert int(v) in p.heard[owner][1]


def test_a_claim_that_its_winner_no_longer_holds_is_released():
    _, m, _, _ = _run("cbba_explorer", "empty", n=2, max_time=1)
    p = m.policies[0]
    p.state.y[5], p.state.z[5] = 1, 9.9          # drone 0 believes drone 1 holds task 5 ...
    p.heard[1] = (1.0, frozenset())               # ... but drone 1's last message says it holds nothing
    p._release_stale_claims(1.0)
    assert p.state.y[5] == -1
    p.state.y[6], p.state.z[6] = 1, 9.9
    p.heard[1] = (1.0, frozenset({6}))            # held: kept while drone 1 is heard from ...
    p._release_stale_claims(2.0)
    assert p.state.y[6] == 1
    p._release_stale_claims(1.0 + p.claim_ttl + 0.5)   # ... and released once it falls silent
    assert p.state.y[6] == -1


def test_obstacles_are_learned_by_sensing_and_not_known_with_the_map_given_prior():
    _, unk, _, r1 = _run("cbba_explorer", "cluttered", n=2, max_time=60)
    _, kn, _, r2 = _run("cbba_explorer", "cluttered", n=2, max_time=60, sim=dict(reveal_obstacles_at_start=True))
    assert r1.extra["map_rebuilds"] > 2                    # discovered one by one
    assert r2.extra["map_rebuilds"] == 2                   # one per drone: the prior, nothing new


def test_messages_carry_beliefs_only_when_sharing_and_never_beyond_range():
    _, _, _, private = _run("cbba_explorer", "rooms_and_door", n=2, max_time=30, share=False)
    _, _, _, shared = _run("cbba_explorer", "rooms_and_door", n=2, max_time=30, share=True)
    assert private.n_messages > 0 and private.extra["cells_merged"] == 0
    assert shared.extra["cells_merged"] > 0
    w = make_world("empty", n_agents=2)
    from mascov.world.world import World
    far = World(w.cfg.with_(r_comm=0.1), [], w.starts, name="far")   # starts are 0.8 m apart
    r = Simulator(far, get_model("cbba_explorer", share=True), SimConfig(max_time=10, **UNKNOWN)).run()
    assert r.n_messages == 0 and r.extra["cbba_msgs_in"] == 0


def test_a_single_drone_needs_no_messages():
    _, _, _, r = _run("cbba_explorer", "sparse", n=1, max_time=600)
    assert r.completed and r.n_messages == 0
