"""The MAPPO goal layer (models/marl_explorer): features, the actor, the policy contract and the training maths.

These tests use an explicitly UNTRAINED actor, so they do not depend on the shipped checkpoint (the quality suite
grades the trained model).
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import models  # noqa: F401,E402
from mascov.common.config import SimConfig, WorldConfig  # noqa: E402
from mascov.common.model import DecentralisedModel  # noqa: E402
from mascov.maps.benchmarks import make_world  # noqa: E402
from mascov.planning.spots import choose_spot  # noqa: E402
from mascov.sim.simulator import Simulator  # noqa: E402
from mascov.world.belief import MapBelief  # noqa: E402
from models.marl_explorer import train as T  # noqa: E402
from models.marl_explorer.features import N_CTX, N_FEAT, TOP_K, candidate_features, statics_for, top_k  # noqa: E402
from models.marl_explorer.model import KIND, MARLExplorerPolicy  # noqa: E402
from models.marl_explorer.nets import Actor, Critic  # noqa: E402


def _view(name="sparse", n=3, seed=1):
    w = make_world(name, n_agents=n, seed=seed)
    return w, w.unknown_view()


def _feats(view, unknown=None, usable=None, team_pos=(), team_goals=(), pos=None):
    space = view.space_for({})
    pos = view.starts[0] if pos is None else pos
    unknown = np.ones(len(view.lattice), bool) if unknown is None else unknown
    usable = space.valid.copy() if usable is None else usable
    travel = space.distances(pos) / view.cfg.plan_speed
    out = candidate_features(space=space, statics=statics_for(view), unknown=unknown, pos=np.asarray(pos, float),
                             travel=travel, usable=usable, candidates=view.candidates, team_pos=list(team_pos),
                             team_goals=list(team_goals), n_agents=view.n_agents, t=0.0, r_comm=math.inf,
                             r_sense=view.cfg.r_sense, delta=1.0)
    return (space, travel, usable, unknown) + out


def test_features_have_the_documented_shapes_and_mask():
    _, view = _view()
    space, travel, usable, unknown, f, ctx, mask, base, gain = _feats(view)
    V = len(view.candidates)
    assert f.shape == (V, N_FEAT) and ctx.shape == (N_CTX,) and mask.shape == (V,) and base.shape == (V,)
    assert np.array_equal(gain, space.gains(unknown))                 # same gain as the hand-written explorers
    assert np.array_equal(mask, usable & np.isfinite(travel) & (gain > 0))
    assert np.all(f[~mask] == 0) and np.all(base[~mask] == 0) and base[mask].max() == 0
    assert np.isfinite(f).all() and np.isfinite(ctx).all()


def test_nothing_unknown_means_nothing_to_choose():
    _, view = _view()
    *_, mask, base, gain = _feats(view, unknown=np.zeros(len(view.lattice), bool))
    assert not mask.any() and gain.sum() == 0


def test_teammate_goals_change_only_the_teammate_features():
    _, view = _view()
    a = _feats(view)[4]
    b = _feats(view, team_pos=[view.starts[1]], team_goals=[40])[4]
    same = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]
    assert np.array_equal(a[:, same], b[:, same])
    assert not np.array_equal(a[:, [6, 7, 8, 9, 14]], b[:, [6, 7, 8, 9, 14]])


def test_untrained_actor_picks_the_greedy_spot():
    """The residual head starts at zero, so argmax of the untrained actor is the greedy rule's choice."""
    _, view = _view("cluttered", seed=2)
    rng = np.random.default_rng(0)
    actor = Actor().eval()
    for _ in range(5):
        unknown = rng.random(len(view.lattice)) < 0.6
        space, travel, usable, unknown, f, ctx, mask, base, gain = _feats(view, unknown=unknown)
        with torch.no_grad():
            logits = actor(torch.from_numpy(f)[None], torch.from_numpy(ctx)[None], torch.from_numpy(mask)[None],
                           torch.from_numpy(base)[None])[0]
        assert int(torch.argmax(logits)) == choose_spot(gain, travel, usable, 1.0)


def test_actor_never_puts_probability_on_masked_spots_and_handles_any_team_size():
    actor = Actor()
    for n in (1, 3, 8):
        f = torch.randn(n, 144, N_FEAT)
        mask = torch.rand(n, 144) < 0.3
        mask[:, 0] = True
        p = torch.softmax(actor(f, torch.randn(n, N_CTX), mask, torch.randn(n, 144)), -1)
        assert torch.all(p[~mask] < 1e-12) and torch.allclose(p.sum(-1), torch.ones(n))


def _model(**kw):
    actor = kw.pop("actor", Actor().eval())
    return DecentralisedModel(MARLExplorerPolicy, name="marl_explorer", actor=actor, **kw)


def test_policy_runs_end_to_end_completes_and_every_drone_knows():
    w = make_world("rooms_and_door", n_agents=3, seed=0)
    r = Simulator(w, _model(), SimConfig(known_map=False, max_time=600, stop_when_covered=False)).run()
    assert r.completed and all(b is not None for b in r.believed_done)
    assert max(r.believed_done) - r.coverage_time <= 5.0             # sharing: the team notices quickly
    assert r.obstacle_events == 0 and r.extra["marl_decisions"] > 0


def test_policy_is_written_for_the_unknown_map_interface():
    w = make_world("empty", n_agents=2)
    with pytest.raises(ValueError):
        Simulator(w, _model(), SimConfig(known_map=True)).run()


def test_deterministic_mode_is_reproducible():
    paths = []
    for _ in range(2):
        w = make_world("sparse", n_agents=3, seed=1)
        sim = Simulator(w, _model(), SimConfig(known_map=False, max_time=30))
        sim.run(record_paths=True)
        paths.append(np.asarray(sim.paths))
    assert np.array_equal(paths[0], paths[1])


def test_messages_carry_position_goal_and_belief_and_respect_range():
    seen = []

    class Spy(MARLExplorerPolicy):
        def step(self, obs):
            seen.extend((self.id, m, obs.pos.copy()) for m in obs.inbox if m.kind == KIND)
            return super().step(obs)

    w = make_world("rooms_and_door", n_agents=3, seed=0, cfg=WorldConfig(r_comm=1.5))
    m = DecentralisedModel(Spy, name="spy", actor=Actor().eval())
    Simulator(w, m, SimConfig(known_map=False, max_time=40)).run()
    assert seen
    for _, msg, _ in seen:
        assert {"pos", "goal", "belief"} <= set(msg.payload) and isinstance(msg.payload["belief"], MapBelief)
    # a message arrives one decision after it was sent, so the sender may have moved by up to 0.5 s of flight
    slack = 0.5 * w.cfg.v_max + 1e-6
    for _, msg, recv_pos in seen:
        assert np.linalg.norm(np.asarray(msg.payload["pos"]) - recv_pos) <= 1.5 + 2 * slack


def test_share_false_sends_no_beliefs():
    w = make_world("rooms_and_door", n_agents=2, seed=0)
    m = _model(share=False)
    r = Simulator(w, m, SimConfig(known_map=False, max_time=20)).run()
    assert r.extra["cells_merged"] == 0


# ------------------------------------------------------------------------------------------------ training maths
def _ep(completed=True):
    """Two agents; agent 0 decides at snaps 0 and 2, agent 1 at snap 1. Coverage fractions 0, .5, .8, final 1."""
    G = 576
    covmask = np.zeros(G, bool)
    covmask[:10] = True
    cov = np.zeros((3, G), bool)
    cov[1, :5] = True
    cov[2, :8] = True
    final = covmask.copy() if completed else cov[2].copy()
    return dict(spec=dict(n=2, r_comm=math.inf, reveal=False), completed=completed, coverage_time=30.0 if completed else None,
                final_time=30.0, n_cov=10, covmask=covmask, obst=np.zeros(G, bool),
                snap_t=np.array([0.0, 10.0, 20.0], np.float32), snap_cov=cov,
                snap_pos=np.ones((3, 2, 2), np.float32), final_cov=final, final_pos=np.ones((2, 2), np.float32),
                agent=np.array([0, 1, 0]), snap=np.array([0, 1, 2]))


def test_rewards_and_gae_follow_each_agents_own_macro_actions():
    ep = _ep()
    values = np.zeros(3, np.float32)
    adv, ret = T.episode_targets(ep, values, np.zeros(2, np.float32), gamma=1.0, lam=1.0)
    A, B = T.REWARD_COVER, T.REWARD_TIME
    # agent 0: [0, 20) gains 0.8 in 20 s, then [20, 30) gains 0.2 in 10 s and the episode terminates
    assert ret[2] == pytest.approx(A * 0.2 - B * 10)
    assert ret[0] == pytest.approx(A * 0.8 - B * 20 + ret[2])
    # agent 1: one macro-action from 10 s to the end, gaining 1.0 - 0.5
    assert ret[1] == pytest.approx(A * 0.5 - B * 20)
    assert np.allclose(adv, ret)                                     # zero values: advantage = return


def test_timeout_bootstraps_from_the_final_state():
    ep = _ep(completed=False)
    v_final = np.array([7.0, 3.0], np.float32)
    _, ret = T.episode_targets(ep, np.zeros(3, np.float32), v_final, gamma=1.0, lam=1.0)
    assert ret[2] == pytest.approx(-T.REWARD_TIME * 10 + 7.0)
    assert ret[1] == pytest.approx(T.REWARD_COVER * 0.3 - T.REWARD_TIME * 20 + 3.0)


def test_critic_inputs_mark_this_drone_and_everyone():
    ep = _ep()
    ep["snap_pos"][1] = [[0.3, 0.3], [5.2, 7.9]]
    img, sc = T.critic_inputs(ep, np.array([1]), np.array([1]))
    assert img.shape == (1, 6, T.GRID, T.GRID) and sc.shape == (1, 6)
    assert img[0, 5].sum() == 24 * 24 and img[0, 5, 24:, :].sum() == 0          # a 12 m world, padded
    assert img[0, 3].sum() == 2 and img[0, 4, 15, 10] == 1 and img[0, 4].sum() == 1
    assert img[0, 0].sum() == 5 and img[0, 1].sum() == 5
    with torch.no_grad():
        assert Critic()(torch.from_numpy(img).float(), torch.from_numpy(sc)).shape == (1,)


def test_rollout_records_one_transition_per_goal_choice():
    import io
    buf = io.BytesIO()
    torch.save({"actor": Actor().state_dict(), "actor_kwargs": {}}, buf)
    spec = T.EpisodeSpec("rooms_and_door", 0, 2, max_time=200)
    ep = T.rollout((buf.getvalue(), 12345, spec, True))
    K = len(ep["agent"])
    assert K > 0 and ep["feats"].shape == (K, TOP_K, N_FEAT) and ep["mask"][np.arange(K), ep["action"]].all()
    assert np.all(ep["logp"] <= 0) and np.all(np.diff(ep["snap"]) >= 0)
    assert ep["completed"] and ep["final_cov"].sum() == ep["n_cov"]


def test_top_k_keeps_the_best_greedy_spots_and_maps_back():
    rng = np.random.default_rng(0)
    V = 300
    feats = rng.random((V, N_FEAT)).astype(np.float32)
    mask = rng.random(V) < 0.7
    base = np.where(mask, -rng.random(V) * 6, 0).astype(np.float32)
    f, m, b, idx = top_k(feats, mask, base, k=40)
    assert f.shape == (40, N_FEAT) and m.all() and set(idx) <= set(np.flatnonzero(mask))
    worst_kept = b.min()
    assert (base[mask & ~np.isin(np.arange(V), idx)] <= worst_kept + 1e-6).all()
    assert np.array_equal(f, feats[idx]) and np.array_equal(b, base[idx])
    f, m, b, idx = top_k(feats, np.arange(V) < 5, base, k=40)                 # fewer than k: padded
    assert m.sum() == 5 and (idx[5:] == -1).all() and (f[5:] == 0).all()


@pytest.mark.parametrize("name", ["maze_24", "random_rooms_20"])
def test_policy_runs_on_big_procedural_maps(name):
    w = make_world(name, n_agents=10, seed=0)
    r = Simulator(w, _model(), SimConfig(known_map=False, max_time=40)).run()
    assert r.extra["marl_decisions"] >= 10 and r.obstacle_events == 0 and r.final_coverage > 0.1


def test_mappo_goals_flown_by_dmpc_run():
    """marl_explorer+dmpc: the goal policy's Navigator is a GoalRecorder (no plan), so the lazy map rebuild must cope."""
    pytest.importorskip("osqp")
    from mascov.common.model import get_model
    w = make_world("random_rooms_12", n_agents=3, seed=0)
    r = Simulator(w, get_model("marl_explorer+dmpc", share=True, seed=0), SimConfig(known_map=False, max_time=30)).run()
    assert r.final_coverage > 0.2 and r.extra["goal_marl_decisions"] > 0
