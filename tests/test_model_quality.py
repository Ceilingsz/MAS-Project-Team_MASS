"""Model quality suite: every registered model, the same scenarios, the same bar.

These tests grade MODELS, not code paths. Each criterion below is something a good multi-drone coverage method
should achieve; a model that falls short FAILS that criterion. A shortfall that is understood and accepted for
now is listed in KNOWN_GAPS with its measured numbers and runs as a strict xfail:

  * the suite stays green while the gap is known, and the report lists every gap ("x" / XFAIL);
  * if the model improves and the criterion now passes, the strict xfail turns into a FAILURE (XPASS(strict)),
    so the table cannot go stale: delete the entry;
  * ``pytest tests/test_model_quality.py --runxfail`` ignores the table and shows every gap as a plain failure;
  * ``pytest -m "not quality"`` skips the suite (it simulates every model, several minutes).

Do not loosen a threshold to make a model pass: record the gap instead. Change a threshold only if the
criterion itself was wrong for every model.

Criteria
  completes          the true coverage reaches 100% within the time budget
  obstacle_safe      no drone ever comes within its body radius of an obstacle or wall
  drone_safe         no two drones ever come closer than d_safe (multi-drone scenarios)
  tracking           the tracking error stays within eps_track, the margin every planner's clearance assumes
  knows_when_done    every drone declares itself done; the team does not stop before the world is covered (the
                     last declaration is no earlier than full coverage); the last comes within DONE_LAG s of it.
                     A single drone may finish earlier: in a partition, a drone is done once its region is.
  efficient          coverage time <= EFFICIENCY x the centralised greedy reference on the same scenario
  team_speedup       three drones cover the two-room map in <= SPEEDUP x the time one drone of the same model needs
  reproducible       the same scenario and seed give the identical trajectory

Scenarios (tests/quality_runs.py): solo (1 drone, rooms_and_door), team (3 drones, rooms_and_door),
clutter (3 drones, cluttered seed 1), low_comm (team with messages and neighbour sensing limited to 1.5 m).
"""
from __future__ import annotations

import numpy as np
import pytest

import quality_runs as Q
from mascov.common.model import registered_models

pytestmark = pytest.mark.quality

DONE_LAG = 20.0          # s between the world being covered and the last drone noticing
EFFICIENCY = 1.5         # x the reference's coverage time
SPEEDUP = 0.6            # three drones vs one (perfect sharing would be 1/3)
REPRO_TIME = 40.0        # s simulated twice for the reproducibility check

# (criterion, model, scenario) -> what was measured and why it is accepted for now. Measured 2026-11-10.
_NAV = "the shared Navigator has no drone-drone avoidance"
_NOSHARE = "drones do not share coverage, so each must see everything itself before it knows"
KNOWN_GAPS: dict[tuple[str, str, str], str] = {
    # drone_safe: every model built on the Navigator collides when paths cross
    ("drone_safe", "greedy_centralised", "clutter"): f"1 episode, closest 0.050 m; {_NAV}",
    ("drone_safe", "static_partition", "clutter"): f"1 episode, closest 0.163 m; {_NAV}",
    ("drone_safe", "greedy_uncoordinated", "team"): f"6 episodes, closest 0.026 m; {_NAV}",
    ("drone_safe", "greedy_uncoordinated", "clutter"): f"9 episodes, closest 0.009 m; {_NAV}",
    ("drone_safe", "greedy_uncoordinated", "low_comm"): f"6 episodes, closest 0.026 m; {_NAV}",
    ("drone_safe", "frontier_greedy", "team"): f"3 episodes, closest 0.077 m; {_NAV}",
    ("drone_safe", "frontier_greedy", "clutter"): f"2 episodes, closest 0.032 m; {_NAV}",
    ("drone_safe", "frontier_greedy", "low_comm"): f"3 episodes, closest 0.077 m; {_NAV}",
    ("drone_safe", "frontier_greedy_shared", "team"): f"1 episode, closest 0.039 m; {_NAV}",
    ("drone_safe", "frontier_greedy_shared", "clutter"): f"1 episode, closest 0.000 m; {_NAV}",
    ("drone_safe", "frontier_greedy_shared", "low_comm"): f"1 episode, closest 0.000 m; {_NAV}",
    ("drone_safe", "cbba", "team"): f"1 episode, closest 0.053 m; {_NAV}",
    ("drone_safe", "cbba", "clutter"): f"2 episodes, closest 0.115 m; {_NAV}",
    ("drone_safe", "cbba", "low_comm"): f"8 episodes, closest 0.017 m; {_NAV}",
    # completes
    ("completes", "cbba", "clutter"): "stalls at 97.6% true coverage for the whole 900 s (cause not investigated)",
    # knows_when_done: the last drone notices late
    ("knows_when_done", "greedy_uncoordinated", "team"): f"last drone 46.3 s late; {_NOSHARE}",
    ("knows_when_done", "greedy_uncoordinated", "clutter"): f"last drone 27.2 s late; {_NOSHARE}",
    ("knows_when_done", "greedy_uncoordinated", "low_comm"): f"last drone 46.3 s late; {_NOSHARE}",
    ("knows_when_done", "frontier_greedy", "team"): f"last drone 95.9 s late; {_NOSHARE}",
    ("knows_when_done", "frontier_greedy", "clutter"): f"last drone 74.7 s late; {_NOSHARE}",
    ("knows_when_done", "frontier_greedy", "low_comm"): f"last drone 95.9 s late; {_NOSHARE}",
    ("knows_when_done", "greedy+dmpc", "team"): f"last drone 21.4 s late; {_NOSHARE} (default share=False)",
    ("knows_when_done", "greedy+dmpc", "clutter"): f"last drone 25.6 s late; {_NOSHARE} (default share=False)",
    ("knows_when_done", "greedy+dmpc", "low_comm"): f"last drone 24.4 s late; {_NOSHARE} (default share=False)",
    ("knows_when_done", "frontier+dmpc", "team"): f"last drone 38.5 s late; {_NOSHARE} (default share=False)",
    ("knows_when_done", "frontier+dmpc", "clutter"): f"last drone 73.0 s late; {_NOSHARE} (default share=False)",
    ("knows_when_done", "cbba", "clutter"): "never covers the map (see completes), so no drone can conclude it is done",
    ("knows_when_done", "cbba", "low_comm"): "map covered at 106.9 s but drones 1 and 2 never declare done (messages reach only 1.5 m)",
    ("knows_when_done", "frontier+dmpc", "low_comm"): f"last drone 38.5 s late; {_NOSHARE} (default share=False)",
    # efficient: slower than 1.5 x centralised greedy
    ("efficient", "greedy_uncoordinated", "team"): "141.7 s = 2.57 x reference; drones duplicate each other's work",
    ("efficient", "greedy_uncoordinated", "clutter"): "163.3 s = 2.30 x reference; drones duplicate each other's work",
    ("efficient", "greedy+dmpc", "team"): "160.8 s = 2.91 x reference; same uncoordinated goals as greedy_uncoordinated",
    ("efficient", "greedy+dmpc", "clutter"): "161.4 s = 2.28 x reference; same uncoordinated goals as greedy_uncoordinated",
    ("efficient", "cbba", "clutter"): "does not complete (see completes)",
    ("efficient", "frontier+dmpc", "team"): "159.2 s = 2.88 x reference; uncoordinated frontier goals, no sharing",
    ("efficient", "frontier+dmpc", "clutter"): "106.8 s = 1.51 x reference; uncoordinated frontier goals, no sharing",
    ("efficient", "frontier_greedy", "team"): "104.6 s = 1.89 x reference; no sharing, map unknown",
    ("efficient", "frontier_greedy", "clutter"): "118.8 s = 1.68 x reference; no sharing, map unknown",
    ("efficient", "frontier_greedy_shared", "team"): "94.7 s = 1.72 x reference; beliefs shared but no intent sharing",
    ("efficient", "frontier_greedy_shared", "clutter"): "111.8 s = 1.58 x reference; beliefs shared but no intent sharing",
    # team_speedup: three drones barely faster than one
    ("team_speedup", "greedy_uncoordinated", "team"): "3 drones 141.7 s vs 1 drone 187.9 s = 0.75; duplicated work",
    ("team_speedup", "frontier+dmpc", "team"): "3 drones 159.2 s vs 1 drone 165.1 s = 0.96; duplicated work (share=True "
                                               "makes N=4 0.61 x faster in experiment E3D)",
    ("team_speedup", "greedy+dmpc", "team"): "3 drones 160.8 s vs 1 drone 177.0 s = 0.91; duplicated work (not investigated "
                                             "further: earlier, 4 drones at eps_track 0.3 took 131 s)",
}


def _cases(criterion, scenarios, multi_only=False, skip_models=()):
    out = []
    for m in Q.MODELS:
        for s in scenarios:
            if Q.applies(m, s) is not None or m in skip_models:
                continue
            if multi_only and Q.SCENARIOS[s].n < 2:
                continue
            marks = []
            why = Q.available(m)
            if why:
                marks.append(pytest.mark.skip(reason=why))
            gap = KNOWN_GAPS.get((criterion, m, s))
            if gap:
                marks.append(pytest.mark.xfail(strict=True, reason=f"known gap: {gap}"))
            out.append(pytest.param(m, s, id=f"{m}-{s}", marks=marks))
    return out


ALL = list(Q.SCENARIOS)
GRADED = ["solo", "team", "clutter"]          # the scenarios the efficiency reference also runs


def _r(model, scenario):
    return Q.run(model, scenario).result


def test_every_model_is_graded():
    """A new model in models/ must be added to quality_runs.MODELS, so it cannot avoid the suite. (Other test
    files register throwaway models of their own; only factories defined in the models package count.)"""
    from mascov.common.model import _REGISTRY
    real = {n for n in registered_models() if getattr(_REGISTRY[n], "__module__", "").startswith("models.")}
    missing = real - set(Q.MODELS)
    assert not missing, f"add these models to tests/quality_runs.py MODELS: {sorted(missing)}"


def test_known_gaps_refer_to_real_cases():
    crit = {"completes", "obstacle_safe", "drone_safe", "tracking", "knows_when_done", "efficient",
            "team_speedup", "reproducible"}
    for (c, m, s) in KNOWN_GAPS:
        assert c in crit and m in Q.MODELS and s in Q.SCENARIOS and Q.applies(m, s) is None, (c, m, s)


@pytest.mark.parametrize("model,scenario", _cases("completes", ALL))
def test_completes(model, scenario):
    r = _r(model, scenario)
    assert r.completed, f"true coverage {r.final_coverage:.2%} after {r.final_time:.0f} s ({Q.summary(r)})"


@pytest.mark.parametrize("model,scenario", _cases("obstacle_safe", ALL))
def test_obstacle_safe(model, scenario):
    run = Q.run(model, scenario)
    r = run.result
    assert r.obstacle_events == 0 and r.min_clearance >= run.world.cfg.drone_radius - 1e-9, \
        f"{r.obstacle_events} obstacle/wall episodes, closest centre distance {r.min_clearance:.3f} m"


@pytest.mark.parametrize("model,scenario", _cases("drone_safe", ALL, multi_only=True))
def test_drone_safe(model, scenario):
    run = Q.run(model, scenario)
    r = run.result
    assert r.pair_events == 0 and r.min_pair_dist >= run.world.cfg.d_safe - 1e-9, \
        f"{r.pair_events} drone-drone episodes, closest {r.min_pair_dist:.3f} m (limit {run.world.cfg.d_safe} m)"


@pytest.mark.parametrize("model,scenario", _cases("tracking", ALL))
def test_tracking(model, scenario):
    run = Q.run(model, scenario)
    worst = max(run.result.max_tracking_error)
    assert worst <= run.world.cfg.eps_track, f"tracking error {worst:.3f} m > eps_track {run.world.cfg.eps_track} m"


@pytest.mark.parametrize("model,scenario", _cases("knows_when_done", ALL))
def test_knows_when_done(model, scenario):
    r = _r(model, scenario)
    done = r.believed_done
    assert r.completed, "world never covered, so 'done' cannot be judged"
    assert all(b is not None for b in done), f"drones that never declared done: {[i for i, b in enumerate(done) if b is None]}"
    assert max(done) >= r.coverage_time - 1e-6, f"every drone stopped by {max(done):.1f} s, before coverage at {r.coverage_time:.1f} s"
    lag = max(done) - r.coverage_time
    assert lag <= DONE_LAG, f"last drone noticed {lag:.1f} s after the world was covered (limit {DONE_LAG} s)"


@pytest.mark.parametrize("model,scenario", _cases("efficient", GRADED, skip_models=(Q.REFERENCE,)))
def test_efficient(model, scenario):
    if Q.available(Q.REFERENCE):
        pytest.skip("reference model unavailable")
    ref = _r(Q.REFERENCE, scenario)
    r = _r(model, scenario)
    assert ref.completed, "the reference did not complete"
    assert r.completed, "did not complete"
    ratio = r.coverage_time / ref.coverage_time
    assert ratio <= EFFICIENCY, f"{r.coverage_time:.1f} s = {ratio:.2f} x reference {ref.coverage_time:.1f} s (limit {EFFICIENCY} x)"


@pytest.mark.parametrize("model,scenario", _cases("team_speedup", ["team"]))
def test_team_speedup(model, scenario):
    one, team = _r(model, "solo"), _r(model, scenario)
    assert one.completed and team.completed, "a run did not complete"
    ratio = team.coverage_time / one.coverage_time
    assert ratio <= SPEEDUP, f"3 drones {team.coverage_time:.1f} s vs 1 drone {one.coverage_time:.1f} s = {ratio:.2f} (limit {SPEEDUP})"


@pytest.mark.parametrize("model,scenario", _cases("reproducible", ["team"]))
def test_reproducible(model, scenario):
    a = Q.run(model, scenario, max_time=REPRO_TIME, record_paths=True)
    Q._CACHE.pop((model, scenario, REPRO_TIME, True))
    b = Q.run(model, scenario, max_time=REPRO_TIME, record_paths=True)
    pa, pb = np.asarray(a.paths), np.asarray(b.paths)
    assert pa.shape == pb.shape and np.array_equal(pa, pb), \
        f"trajectories differ by up to {np.abs(pa - pb).max() if pa.shape == pb.shape else 'shape'} m"
