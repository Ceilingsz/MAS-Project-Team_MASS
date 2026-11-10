"""Simulator bookkeeping (metrics, ranges, message counts), multi-round negotiation, and the experiment framework."""
import math
from pathlib import Path

import numpy as np
import pytest

from mascov.common.config import SimConfig, WorldConfig
from mascov.common.model import register
from mascov.common.types import Command, Message, Observation
from mascov.comm.rounds import NegotiatingModel, NegotiatingPolicy
from mascov.experiments import analysis as an
from mascov.experiments.runner import filter_jobs, run_all
from mascov.experiments.spec import Experiment, ModelSpec, expand
from mascov.sim.simulator import Simulator

from tests.fakes import FakeWorld, ScriptedModel


def sim(world, model, **kw):
    return Simulator(world, model, SimConfig(**{"max_time": 40.0, "stop_when_covered": False, **kw})).run()


# ----------------------------------------------------------------------------------------------- metrics
def test_visits_count_separate_passes():
    # one drone goes out and comes back past a target: two visits; a far target is never seen
    w = FakeWorld([[5.0, 6.0], [5.0, 11.0]], [[1.0, 6.0]])
    r = sim(w, ScriptedModel({0: [[10.0, 6.0], [1.0, 6.0]]}), max_time=60.0)
    assert r.target_visits.tolist() == [2, 0]
    assert r.mean_visits == pytest.approx(2.0) and r.frac_revisited == pytest.approx(1.0)
    assert not r.completed and r.final_coverage == pytest.approx(0.5)


def test_time_to_milestones_are_ordered_and_match_coverage_time():
    w = FakeWorld([[3.0, 6.0], [5.0, 6.0], [7.0, 6.0], [9.0, 6.0]], [[1.0, 6.0]])
    r = sim(w, ScriptedModel({0: [[11.0, 6.0]]}), stop_when_covered=True)
    t = [r.time_to_fraction(f) for f in (0.5, 0.9, 1.0)]
    assert all(x is not None for x in t) and t == sorted(t)
    assert r.coverage_time == pytest.approx(t[-1]) and r.safe_complete
    assert r.coverage_auc(horizon=r.final_time) < 1.0 and r.coverage_auc(horizon=1000) > r.coverage_auc(horizon=r.final_time)
    m = r.metrics()
    assert m["t100"] == pytest.approx(r.coverage_time) and m["safe_complete"] is True


def test_collisions_are_split_into_pair_and_obstacle_episodes():
    w = FakeWorld([[6.0, 6.0]], [[4.0, 6.0], [8.0, 6.0]])
    r = sim(w, ScriptedModel({0: [[6.0, 6.0]], 1: [[6.0, 6.0]]}), max_time=30.0)
    assert r.pair_events == 1 and r.obstacle_events == 0 and r.collision_events == 1 and not r.safe_complete
    w = FakeWorld([[6.0, 6.0]], [[2.0, 6.0]])
    r = sim(w, ScriptedModel({0: [[0.05, 6.0]]}), max_time=30.0)       # into the wall, closer than the drone radius
    assert r.obstacle_events == 1 and r.pair_events == 0


# ----------------------------------------------------------------------------------------------- ranges, messages
def test_neighbour_sensing_and_message_ranges_are_separate():
    cfg = WorldConfig(r_comm=2.0, r_neighbor=5.0)
    m = ScriptedModel()
    sim(FakeWorld([[6, 6]], [[2.0, 6.0], [5.0, 6.0]], cfg), m, max_time=1.0)
    o = m.first_obs[0]
    assert 1 in o.neighbors and o.comm_neighbors == []                  # sensed at 3 m, but out of message reach
    cfg = WorldConfig(r_comm=5.0, r_neighbor=2.0)
    m = ScriptedModel()
    sim(FakeWorld([[6, 6]], [[2.0, 6.0], [5.0, 6.0]], cfg), m, max_time=1.0)
    o = m.first_obs[0]
    assert o.neighbors == {} and o.comm_neighbors == [1]
    assert WorldConfig(r_comm=3.0).neighbor_radius == 3.0               # default: same as r_comm


def test_message_counts_transmissions_and_deliveries():
    cfg = WorldConfig(r_comm=2.0)
    w = FakeWorld([[6, 6]], [[2.0, 6.0], [3.0, 6.0], [10.0, 6.0]], cfg)
    r = sim(w, ScriptedModel(broadcast_from=[0]), max_time=3.0)
    assert r.n_messages == 1 and r.n_deliveries == 1                    # drone 1 hears it, drone 2 is out of range
    assert r.msgs_per_drone_per_s == pytest.approx(1 / (3 * r.final_time))


def test_model_decision_period_overrides_the_simulator_default():
    w = FakeWorld([[6, 6]], [[2.0, 6.0]])
    slow = sim(w, ScriptedModel(), max_time=5.0)
    fast = sim(w, ScriptedModel(decision_period=0.1), max_time=5.0)
    assert slow.n_decisions == 10 and fast.n_decisions == 50           # one per period; the last tick only ends the run


# ----------------------------------------------------------------------------------------------- negotiation rounds
class MaxConsensus(NegotiatingPolicy):
    max_rounds = 10

    def reset(self, world, agent_id, n_agents):
        self.id, self.value, self.changed = agent_id, float(agent_id), True

    def begin(self, obs):
        self.changed = True

    def outgoing(self, r):
        return [Message(self.id, "v", self.value)]

    def incoming(self, r, msgs):
        new = max([self.value] + [m.payload for m in msgs])
        self.changed, self.value = new != self.value, new

    def converged(self):
        return not self.changed

    def finish(self):
        return Command(done=True)


def _chain_obs(n=4):
    comm = lambda i: [j for j in (i - 1, i + 1) if 0 <= j < n]
    return {i: Observation(t=0.0, agent_id=i, pos=np.array([float(i), 0.0]), vel=np.zeros(2), belief=None,
                           newly_covered=np.zeros(0, dtype=int), new_obstacles=[], neighbors={}, inbox=[],
                           plan_finished=True, plan_time_left=0.0, comm_neighbors=comm(i)) for i in range(n)}


def test_negotiation_spreads_information_hop_by_hop_and_counts_messages():
    model = NegotiatingModel(MaxConsensus, name="maxc")
    model.reset(None, [0, 1, 2, 3])
    model.step(_chain_obs())
    assert [p.value for p in model.policies.values()] == [3.0] * 4      # the maximum crossed the whole chain
    s = model.stats()
    assert s["neg_rounds"] == 4 and s["neg_rounds_max"] == 4            # 3 hops + 1 round to see nothing change
    assert s["neg_transmissions"] == 16 and s["neg_deliveries"] == 24   # 4 senders and 6 directed links per round


def test_negotiation_respects_the_communication_radius():
    obs = _chain_obs()
    for i in obs:
        obs[i].comm_neighbors = []                                      # nobody in reach
    model = NegotiatingModel(MaxConsensus, name="maxc")
    model.reset(None, [0, 1, 2, 3])
    model.step(obs)
    assert [p.value for p in model.policies.values()] == [0.0, 1.0, 2.0, 3.0]
    assert model.stats()["neg_deliveries"] == 0


# ----------------------------------------------------------------------------------------------- experiment framework
def _exp(**kw):
    base = dict(name="T", title="t", question="?", models=[ModelSpec("A", "a"), ModelSpec("B", "b", ignores=("world.r_comm",))],
                maps=["m"], n_agents=[2], seeds=[0, 1], levers={"world.r_comm": [2.0, 4.0, math.inf]})
    base.update(kw)
    return Experiment(**base)


def test_expand_builds_the_grid_and_collapses_ignored_levers():
    jobs = expand(_exp())
    a = [j for j in jobs if j.label == "A"]
    b = [j for j in jobs if j.label == "B"]
    assert len(a) == 3 * 2 and len(b) == 1 * 2                          # B runs once per seed, not once per radius
    assert {j.world["r_comm"] for j in a} == {2.0, 4.0, math.inf}
    assert all("r_comm" not in j.world and j.levers["world.r_comm"] is None for j in b)
    assert len({j.job_id for j in jobs}) == len(jobs)                   # ids are unique
    assert [j.job_id for j in jobs] == [j.job_id for j in expand(_exp())]   # and stable
    with pytest.raises(KeyError):
        expand(_exp(levers={"bogus": [1]}))


def test_runner_end_to_end_with_resume_and_skips(tmp_path):
    register("test_scripted")(lambda **kw: ScriptedModel({0: [[11.0, 6.0]]}))
    exp = _exp(models=[ModelSpec("S", "test_scripted"), ModelSpec("Missing", "no_such_model"),
                       ModelSpec("S2", "test_scripted", role="reference")],
               levers={"world.r_comm": [3.0, math.inf]}, seeds=[0, 1], max_time=30.0)
    jobs = expand(exp)
    ok, skipped = filter_jobs(jobs)
    assert len(skipped) == 4 and all("not registered" in why for _, why in skipped)
    builder = lambda name, n_agents, size, seed, cfg: FakeWorld([[3, 6], [6, 6], [9, 6]], [[1.0, 6.0]] * 1, cfg)
    info = run_all(jobs, tmp_path, n_jobs=1, progress=False, world_builder=builder)
    assert info["ran"] == 8 and info["errors"] == 0 and info["skipped"] == 4
    rows = an.load_rows(tmp_path)
    assert len(rows) == 8 and all(r["status"] == "ok" and r["safe_complete"] for r in rows)
    assert {"t95", "mean_visits", "msgs_per_drone_per_s", "world_r_comm", "label"} <= set(rows[0])
    assert len(an.load_curves(tmp_path)) == 8 and (tmp_path / "runs.csv").exists()
    again = run_all(jobs, tmp_path, n_jobs=1, progress=False, world_builder=builder)
    assert again["ran"] == 0                                            # resumed: nothing repeated


def test_analysis_numbers():
    p, lo, hi = an.wilson(8, 10)
    assert p == 0.8 and lo < 0.8 < hi and 0.0 <= lo and hi <= 1.0
    s = an.mean_ci([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["ci"] == pytest.approx(4.30 * 1.0 / math.sqrt(3))
    rows = [dict(label="A", map="m", n_agents=2, seed=s, t95=10.0 + s, status="ok") for s in range(3)]
    rows += [dict(label="B", map="m", n_agents=2, seed=s, t95=20.0 + s, status="ok") for s in range(3)]
    rows += [dict(label="B", map="m", n_agents=2, seed=5, t95=None, final_time=100.0, status="ok"),
             dict(label="A", map="m", n_agents=2, seed=5, t95=12.0, status="ok")]
    d = an.paired(rows, "A", "B", "t95")
    assert d["n"] == 4 and d["mean_diff"] < 0                           # A is faster; the censored run counts at 100 s
    g = an.group_stats(rows, ["label"], "t95")
    assert {x["label"]: x["censored"] for x in g}["B"] == pytest.approx(0.25)


def test_every_defined_experiment_expands_and_every_lever_key_is_valid():
    from mascov.experiments.definitions import EXPERIMENTS
    for e in EXPERIMENTS.values():
        jobs = expand(e)
        assert jobs, e.name
        assert len({j.job_id for j in jobs}) == len(jobs), e.name


def test_figures_are_drawn_from_rows(tmp_path):
    from mascov.experiments.definitions import EXPERIMENTS
    from mascov.experiments.plots import make_figures
    from mascov.experiments.spec import PlotSpec
    rows, curves = [], {}
    for lab, role, base in (("CBBA", "method", 20.0), ("Greedy", "method", 30.0), ("Static partition", "reference", 40.0)):
        for n in (2, 4):
            for seed in range(3):
                jid = f"{lab}{n}{seed}"
                rows.append(dict(job_id=jid, label=lab, role=role, map="m", n_agents=n, seed=seed, map_prior="known",
                                 t95=base / n + seed, collisions=seed % 2, safe_complete=bool(seed), world_r_comm=None if role == "reference" else (2.0 if seed else math.inf),
                                 status="ok", final_time=base / n + 5, mean_visits=1.5, msgs_per_drone_per_s=0.2))
                curves[jid] = dict(t=[0, 5, 10], coverage=[0.0, 0.6, 1.0])
    specs = [PlotSpec("sweep", "a", "sweep", x="n_agents", y="t95"),
             PlotSpec("rate", "b", "rate", x="n_agents", y="safe_complete"),
             PlotSpec("sweep", "c", "radius", x="world_r_comm", y="t95"),
             PlotSpec("curves", "d", "curves"),
             PlotSpec("tradeoff", "e", "tradeoff", x="t95", y="collisions", per=None),
             PlotSpec("paired", "f", "paired", y="t95", per=None, options=dict(pairs=[("CBBA", "Greedy")]))]
    for s in specs:
        files = make_figures(s, rows, curves, tmp_path)
        assert files and all(Path(f).exists() for f in files), s.kind
    assert len(EXPERIMENTS) >= 9


def test_real_world_visits_and_milestones():
    pytest.importorskip("shapely")
    import models  # noqa: F401
    from mascov.common.model import get_model
    from mascov.maps.benchmarks import make_world
    w = make_world("empty", n_agents=2)
    r = Simulator(w, get_model("greedy_uncoordinated"), SimConfig(max_time=900.0)).run()
    assert r.completed and r.mean_visits >= 1.0 and r.time_to_fraction(0.5) < r.time_to_fraction(0.95) <= r.coverage_time
    assert r.pair_events + r.obstacle_events == r.collision_events or r.collision_events <= r.pair_events + r.obstacle_events
