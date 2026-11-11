"""MAPPO training of the ``marl_explorer`` goal layer, in the real Simulator.

Why the real simulator and not a surrogate: one episode takes about 2 s of wall time (12 x 12 m, 0.1 s ticks), so
CPU worker processes produce enough episodes, and the policy is trained on exactly the dynamics, sensing,
Navigator and messaging it is evaluated on (no sim-to-sim gap). Rollouts run on CPU workers (the actor is small);
the PPO update runs on the GPU when one is available.

Formulation (a Dec-POMDP with macro-actions, MacDec-POMDP):
  * agents: the drones, homogeneous, one shared actor (parameter sharing), any team size;
  * macro-action: "fly to candidate spot j"; it ends when the drone arrives, is blocked, or the spot stops being
    useful, so agents decide asynchronously;
  * team reward over each agent's macro-action [t_k, t_k+1):  R = A * (true coverage gained) - B * (seconds elapsed);
    the episode ends at full true coverage (no time cost after that), so maximising return = covering fast;
  * discount per second (gamma ** dt), GAE over each agent's own decision sequence;
  * centralised critic V(true state, agent) (CTDE): covered / uncovered / obstacle cells, every drone, this drone.

The training harness is the only place that touches ground truth (the critic input and the reward). The actor
sees exactly what ``MARLExplorerPolicy`` sees at execution.

    python scripts/train_marl.py --init-actor <checkpoint> --iters 200 --episodes 48 --workers 13
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

REWARD_COVER = 10.0       # A: full coverage is worth 10
REWARD_TIME = 0.02        # B: per second (100 s costs 2)


# ====================================================================================== episode sampling
@dataclass
class EpisodeSpec:
    map: str
    seed: int
    n: int
    r_comm: float = math.inf
    eps_track: float = 0.3
    start_spacing: float = 0.8
    start_per_row: int = 3
    reveal: bool = False
    max_time: float = 400.0
    sim_seed: int = 0


# Training curriculum: world size -> (maps, team sizes, max simulated time). The study's scale is 12-16 m with up to 16
# drones; bigger worlds are held out (a generalisation test, and 3-5x slower to simulate).
FIXED_MAPS = {"empty", "corridor", "rooms_and_door", "labyrinth"}   # one layout each (no seed); at most 8 drones fit
CURRICULUM = {
    12: (["sparse", "cluttered", "rooms_and_door", "corridor", "labyrinth", "empty",
          "dense_12", "random_rooms_12", "maze_12", "warehouse_12"], list(range(2, 9)), 400.0),
    16: (["sparse_16", "cluttered_16", "dense_16", "random_rooms_16", "maze_16", "warehouse_16"],
         list(range(4, 17)), 300.0),
}
SIZE_P = {12: 0.5, 16: 0.5}


def sample_spec(rng: np.random.Generator) -> EpisodeSpec:
    """A random training episode. Seeds >= 1000 only (0-999 are reserved for validation and tests)."""
    size = int(rng.choice(list(SIZE_P), p=list(SIZE_P.values())))
    maps, ns, max_time = CURRICULUM[size]
    m = str(rng.choice(maps))
    n = int(rng.choice(ns))
    seed = 0 if m in FIXED_MAPS else int(rng.integers(1000, 10**6))
    r = float(rng.choice([1.5, 3.0, 6.0, math.inf], p=[0.15, 0.25, 0.25, 0.35]))
    wide = size == 12 and n <= 8 and rng.random() < 0.3
    return EpisodeSpec(map=m, seed=seed, n=n, r_comm=r, eps_track=0.45 if rng.random() < 0.3 else 0.3,
                       start_spacing=2.0 if wide else 0.8, start_per_row=2 if wide else 3,
                       reveal=bool(rng.random() < 0.1), max_time=max_time, sim_seed=int(rng.integers(0, 10**6)))


def validation_specs() -> list[EpisodeSpec]:
    """Checkpoint selection set: seeds 900-901 (never trained on, and apart from the test seeds 0-9), 12-16 m worlds,
    3-16 drones, three of them with short messages."""
    E = EpisodeSpec
    return [
        E("sparse", 900, 3), E("cluttered", 901, 6), E("rooms_and_door", 0, 4), E("labyrinth", 0, 6),
        E("maze_12", 900, 4), E("random_rooms_12", 900, 6, r_comm=3.0), E("dense_12", 901, 8),
        E("dense_16", 900, 6, max_time=600), E("random_rooms_16", 901, 8, max_time=600),
        E("maze_16", 900, 10, max_time=600), E("warehouse_16", 900, 6, r_comm=3.0, max_time=600),
        E("cluttered_16", 900, 12, max_time=600), E("sparse_16", 901, 4, max_time=600),
        E("random_rooms_16", 900, 12, max_time=600), E("maze_16", 901, 16, max_time=600),
        E("dense_16", 901, 16, r_comm=3.0, max_time=600), E("warehouse_16", 901, 10, max_time=600),
        E("sparse_16", 900, 8, r_comm=6.0, max_time=600),
    ]


def make_world_for(spec: EpisodeSpec):
    """The world of a spec. Wide (2 m) start spacing can put a start inside a wall on some maps; the spec then falls
    back to the default 0.8 m spacing (and is updated, so the recorded spec is what ran)."""
    from mascov.common.config import WorldConfig
    from mascov.maps.benchmarks import make_world
    cfg = WorldConfig(r_comm=spec.r_comm, eps_track=spec.eps_track)
    try:
        return make_world(spec.map, n_agents=spec.n, seed=spec.seed, cfg=cfg, start_spacing=spec.start_spacing,
                          start_per_row=spec.start_per_row)
    except ValueError:
        if (spec.start_spacing, spec.start_per_row) == (0.8, 3):
            raise
        spec.start_spacing, spec.start_per_row = 0.8, 3
        return make_world(spec.map, n_agents=spec.n, seed=spec.seed, cfg=cfg)


def sim_config(spec: EpisodeSpec, stop_when_covered: bool = True):
    from mascov.common.config import SimConfig
    return SimConfig(max_time=spec.max_time, known_map=False, reveal_obstacles_at_start=spec.reveal,
                     seed=spec.sim_seed, stop_when_covered=stop_when_covered)


# ====================================================================================== rollout (worker side)
def _setup_paths():
    for p in (str(ROOT / "src"), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


_WORKER: dict = {}


def worker_init():
    _setup_paths()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    import models  # noqa: F401


def _actor_from(blob: bytes, version: int):
    import torch
    from models.marl_explorer.nets import Actor
    if _WORKER.get("version") != version:
        state = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        a = Actor(**state["actor_kwargs"])
        a.load_state_dict(state["actor"])
        a.eval()
        _WORKER.update(version=version, actor=a)
    return _WORKER["actor"]


def rollout(args) -> dict:
    """Run one episode with a stochastic (training) or deterministic (evaluation) actor; return its transitions."""
    blob, version, spec, stochastic = args
    from mascov.common.model import CentralModel
    from mascov.sim.simulator import Simulator
    from mascov.world.belief import FREE
    from models.marl_explorer.model import MARLExplorerPolicy
    import shapely

    actor = _actor_from(blob, version)
    world, resample = None, np.random.default_rng(spec.sim_seed)
    for _ in range(20):                       # some sampled specs are infeasible (a start inside a wall, a random
        try:                                  # map that cannot be built): draw another one
            world = make_world_for(spec)
            break
        except (ValueError, RuntimeError):
            if not stochastic:
                raise
            spec = sample_spec(resample)
    if world is None:
        raise RuntimeError("could not sample a feasible episode")
    cov_cells = world.target_cell[world.coverable]
    G = world.cell_free.size
    covmask = np.zeros(G, dtype=bool)
    covmask[cov_cells] = True
    lattice = world.unknown_view().lattice
    obst = np.zeros(G, dtype=bool)
    for poly in world.obstacles:
        obst |= shapely.contains_xy(poly, lattice[:, 0], lattice[:, 1])

    recs: list = []          # (agent, snap, feats, ctx, mask, base, action, logp)
    snaps_t, snaps_cov, snaps_pos = [], [], []

    class Recording(CentralModel):
        name, centralised, map_interface = "marl_train", False, "unknown"

        def reset(self, view, agent_ids):
            self.pol = {i: MARLExplorerPolicy(actor=actor, stochastic=stochastic, seed=spec.sim_seed,
                                              recorder=self._rec) for i in agent_ids}
            for i, p in self.pol.items():
                p.reset(view, i, len(agent_ids))
            self.pending = []

        def _rec(self, i, t, feats, ctx, mask, base, a, logp):
            self.pending.append([i, -1, feats.astype(np.float16), ctx, mask, base.astype(np.float16), a, logp])

        def step(self, obs):
            cmds = {i: self.pol[i].step(o) for i, o in obs.items()}
            if self.pending:
                union = np.logical_or.reduce([o.belief.state == FREE for o in obs.values()])
                snaps_t.append(next(iter(obs.values())).t)
                snaps_cov.append(union & covmask)
                snaps_pos.append(np.stack([obs[i].pos for i in sorted(obs)]))
                for r in self.pending:
                    r[1] = len(snaps_t) - 1
                recs.extend(self.pending)
                self.pending = []
            self._last_pos = np.stack([obs[i].pos for i in sorted(obs)])
            return cmds

        def stats(self):
            return {}

    model = Recording()
    sim = Simulator(world, model, sim_config(spec))
    res = sim.run()
    final_cov = np.logical_or.reduce([b.state == FREE for b in sim.beliefs]) & covmask
    n_cov = max(1, int(covmask.sum()))
    return dict(
        spec=asdict(spec), completed=bool(res.completed), coverage_time=res.coverage_time, final_time=res.final_time,
        final_coverage=float(res.final_coverage), pair_events=int(res.pair_events), n_cov=n_cov,
        covmask=covmask, obst=obst,
        snap_t=np.asarray(snaps_t, dtype=np.float32),
        snap_cov=np.asarray(snaps_cov, dtype=bool).reshape(-1, G),
        snap_pos=np.asarray(snaps_pos, dtype=np.float32).reshape(-1, spec.n, 2),
        final_cov=final_cov, final_pos=getattr(model, "_last_pos", world.starts).astype(np.float32),
        agent=np.asarray([r[0] for r in recs], dtype=np.int16),
        snap=np.asarray([r[1] for r in recs], dtype=np.int32),
        feats=np.asarray([r[2] for r in recs], dtype=np.float16),
        ctx=np.asarray([r[3] for r in recs], dtype=np.float32),
        mask=np.asarray([r[4] for r in recs], dtype=bool),
        base=np.asarray([r[5] for r in recs], dtype=np.float16),
        action=np.asarray([r[6] for r in recs], dtype=np.int64),
        logp=np.asarray([r[7] for r in recs], dtype=np.float32))


def run_baseline(args) -> dict:
    """Evaluate a registered model on one spec (for reference numbers in the training log)."""
    name, kwargs, spec = args
    from mascov.common.model import get_model
    from mascov.sim.simulator import Simulator
    world = make_world_for(spec)
    res = Simulator(world, get_model(name, **kwargs), sim_config(spec)).run()
    return dict(spec=asdict(spec), completed=bool(res.completed), coverage_time=res.coverage_time,
                final_time=res.final_time, pair_events=int(res.pair_events))


# ====================================================================================== learner side
GRID = 48          # critic images are padded to the largest supported world (24 m at 0.5 m cells)


def critic_inputs(ep: dict, snap_idx: np.ndarray, agent: np.ndarray, final: bool = False):
    """Critic image (T, 6, 48, 48) uint8 and scalars (T, 6) float32 for the given snapshots / agents.

    Channels: covered coverable cells, uncovered coverable cells, obstacle cells, drones (count), this drone,
    inside the workspace (smaller worlds are padded with zeros)."""
    n = ep["spec"]["n"]
    G = ep["covmask"].size
    g = int(round(math.sqrt(G)))
    if final:
        cov = ep["final_cov"][None].repeat(len(agent), 0)
        pos = ep["final_pos"][None].repeat(len(agent), 0)
        t = np.full(len(agent), ep["final_time"], dtype=np.float32)
    else:
        cov, pos, t = ep["snap_cov"][snap_idx], ep["snap_pos"][snap_idx], ep["snap_t"][snap_idx]
    T = len(agent)
    small = np.zeros((T, 6, g * g), dtype=np.uint8)
    small[:, 0] = cov
    small[:, 1] = ep["covmask"][None] & ~cov
    small[:, 2] = ep["obst"][None]
    cell = (np.clip((pos[..., 1] / 0.5).astype(int), 0, g - 1) * g
            + np.clip((pos[..., 0] / 0.5).astype(int), 0, g - 1))          # (T, n)
    rows = np.repeat(np.arange(T), n)
    np.add.at(small[:, 3], (rows, cell.reshape(-1)), 1)
    small[np.arange(T), 4, cell[np.arange(T), agent]] = 1
    small[:, 5] = 1
    img = np.zeros((T, 6, GRID, GRID), dtype=np.uint8)
    img[:, :, :g, :g] = small.reshape(T, 6, g, g)
    covfrac = cov.sum(axis=1) / ep["n_cov"]
    r = ep["spec"]["r_comm"]
    scal = np.stack([t / 300.0, np.full(T, n / 8.0), covfrac, np.full(T, 0.0 if not math.isfinite(r) else 1 / (1 + r)),
                     np.full(T, float(ep["spec"]["reveal"])), np.full(T, g / GRID)], axis=1).astype(np.float32)
    return img, scal


def episode_targets(ep: dict, values: np.ndarray, v_final: np.ndarray, gamma: float, lam: float):
    """Rewards, discounts and GAE per agent's decision sequence. values: V at each transition (in order)."""
    T = len(ep["agent"])
    adv = np.zeros(T, dtype=np.float32)
    ret = np.zeros(T, dtype=np.float32)
    covfrac = ep["snap_cov"].sum(axis=1) / ep["n_cov"]
    final_frac = ep["final_cov"].sum() / ep["n_cov"]
    t_end = ep["coverage_time"] if ep["completed"] else ep["final_time"]
    for i in range(ep["spec"]["n"]):
        idx = np.flatnonzero(ep["agent"] == i)
        if idx.size == 0:
            continue
        nxt = 0.0
        for k in range(len(idx) - 1, -1, -1):
            j = idx[k]
            s = ep["snap"][j]
            if k + 1 < len(idx):
                s2 = ep["snap"][idx[k + 1]]
                c2, t2, v2, term = covfrac[s2], ep["snap_t"][s2], values[idx[k + 1]], False
            else:
                c2, t2, v2, term = final_frac, t_end, v_final[i], ep["completed"]
            dt = max(0.0, float(t2 - ep["snap_t"][s]))
            rwd = REWARD_COVER * (c2 - covfrac[s]) - REWARD_TIME * dt
            disc = gamma ** dt
            boot = 0.0 if term else disc * v2
            delta = rwd + boot - values[j]
            nxt = delta + disc * lam * nxt if k + 1 < len(idx) else delta
            adv[j] = nxt
            ret[j] = nxt + values[j]
    return adv, ret


@dataclass
class TrainConfig:
    iters: int = 200
    episodes: int = 48
    workers: int = 13
    lr_actor: float = 1e-4
    lr_critic: float = 5e-4
    gamma: float = 0.995          # per second
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 2048
    entropy: float = 0.005
    entropy_final: float = 0.001
    max_grad: float = 1.0
    eval_every: int = 10
    seed: int = 0
    out: str = str(ROOT / "models" / "marl_explorer" / "checkpoints")
    resume: str | None = None
    init_actor: str | None = None      # warm start: load only the actor's weights from this checkpoint
    actor_kwargs: dict = field(default_factory=dict)


class RunningNorm:
    def __init__(self):
        self.mean, self.var, self.n = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray):
        m, v, k = float(x.mean()), float(x.var()), x.size
        tot = self.n + k
        d = m - self.mean
        self.mean += d * k / tot
        self.var = (self.var * self.n + v * k + d * d * self.n * k / tot) / tot
        self.n = tot

    @property
    def std(self):
        return math.sqrt(max(self.var, 1e-8))


def train(cfg: TrainConfig) -> Path:
    _setup_paths()
    import multiprocessing as mp

    import torch
    from torch import nn

    from models.marl_explorer.nets import Actor, Critic

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"

    actor, critic = Actor(**cfg.actor_kwargs).to(dev), Critic().to(dev)
    vnorm = RunningNorm()
    start_iter = 0
    if cfg.resume:
        blob = torch.load(cfg.resume, map_location=dev, weights_only=False)
        actor.load_state_dict(blob["actor"])
        if "critic" in blob:
            critic.load_state_dict(blob["critic"])
        vnorm.mean, vnorm.var, vnorm.n = blob.get("vnorm", (0.0, 1.0, 1e-4))
        start_iter = blob.get("iter", 0)
    elif cfg.init_actor:
        actor.load_state_dict(torch.load(cfg.init_actor, map_location=dev, weights_only=False)["actor"])
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor, eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic, eps=1e-5)

    def blob_bytes():
        buf = io.BytesIO()
        torch.save({"actor": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
                    "actor_kwargs": cfg.actor_kwargs}, buf)
        return buf.getvalue()

    def save(path: Path, it: int, extra: dict | None = None):
        torch.save({"actor": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
                    "critic": {k: v.detach().cpu() for k, v in critic.state_dict().items()},
                    "actor_kwargs": cfg.actor_kwargs, "vnorm": (vnorm.mean, vnorm.var, vnorm.n), "iter": it,
                    "train_config": asdict(cfg), **(extra or {})}, path)

    def critic_values(img, scal):
        vals = []
        with torch.no_grad():
            for a in range(0, len(img), 4096):
                v = critic(torch.from_numpy(img[a:a + 4096]).to(dev).float(), torch.from_numpy(scal[a:a + 4096]).to(dev))
                vals.append(v.cpu().numpy())
        return np.concatenate(vals) * vnorm.std + vnorm.mean if vals else np.zeros(0, np.float32)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(cfg.workers, initializer=worker_init)
    val = validation_specs()
    print(f"[marl] device={dev}; reference numbers on {len(val)} validation episodes ...", flush=True)
    ref_path = out / "val_reference.json"
    key = [asdict(s) for s in val]
    ref_t = None
    if ref_path.exists():
        cached = json.loads(ref_path.read_text())
        if cached.get("specs") == json.loads(json.dumps(key)):
            ref_t = np.asarray(cached["t100"])
    if ref_t is None:
        ref = pool.map(run_baseline, [("cbba_explorer", dict(share=True), s) for s in val], chunksize=1)
        ref_t = np.asarray([r["coverage_time"] if r["completed"] else s.max_time for r, s in zip(ref, val)])
        ref_path.write_text(json.dumps(dict(specs=key, t100=ref_t.tolist())))
    print(f"[marl] cbba_explorer (shared) validation mean t100 = {ref_t.mean():.1f} s", flush=True)
    best = math.inf
    version = 0

    def evaluate(it):
        nonlocal best
        b = blob_bytes()
        res = pool.map(rollout, [(b, -it - 1, s, False) for s in val], chunksize=1)
        t = np.asarray([r["coverage_time"] if r["completed"] else s.max_time for r, s in zip(res, val)])
        ratio = float(np.mean(t / ref_t))
        size = np.asarray([12 if s.map[-3:] not in ("_16", "_20", "_24") else int(s.map[-2:]) for s in val])
        rec = dict(iter=it, eval_mean_t100=float(t.mean()), eval_ratio_vs_cbba=ratio,
                   eval_completed=float(np.mean([r["completed"] for r in res])),
                   **{f"eval_ratio_{z}m": round(float(np.mean((t / ref_t)[size == z])), 4) for z in sorted(set(size))},
                   eval_t100=[round(float(x), 1) for x in t])
        (out / "snapshots").mkdir(exist_ok=True)
        save(out / "snapshots" / f"iter_{it}.pt", it, dict(eval=rec))
        if ratio < best:
            best = ratio
            save(out / "marl_explorer.pt", it, dict(eval=rec))
            rec["saved_best"] = True
        return rec

    best_file = out / "marl_explorer.pt"
    if cfg.resume and best_file.exists():           # keep the best ratio seen before the restart
        best = torch.load(best_file, map_location="cpu", weights_only=False).get("eval", {}).get(
            "eval_ratio_vs_cbba", math.inf)

    # Asynchronous collection: keep every worker busy and let an iteration take the first ``episodes`` episodes that
    # finish, instead of waiting for the slowest (a 20 m world with 16 drones can take 5x the median). An episode
    # started under the previous policy is still valid data: its actions carry the probabilities they were drawn
    # with, which is exactly what PPO's ratio uses; episodes more than ``max_lag`` versions old are dropped.
    import queue as _queue
    finished: "_queue.Queue" = _queue.Queue()
    in_flight = [0]
    max_lag, target_in_flight = 2, cfg.workers + cfg.workers // 2
    blob_cur = [blob_bytes()]

    def submit(ver):
        in_flight[0] += 1
        pool.apply_async(rollout, ((blob_cur[0], ver, sample_spec(rng), True),),
                         callback=lambda res, ver=ver: finished.put((ver, res)),
                         error_callback=lambda e, ver=ver: finished.put((ver, e)))

    def collect(n, current):
        eps, stale = [], 0
        while len(eps) < n:
            ver, res = finished.get()
            in_flight[0] -= 1
            if isinstance(res, BaseException):
                raise res
            if current - ver > max_lag:
                stale += 1
            else:
                eps.append(res)
            while in_flight[0] < target_in_flight:
                submit(current)
        return eps, stale

    with open(log_path, "a") as logf:
        if not cfg.resume:
            rec0 = evaluate(start_iter)
            print(f"[marl] iter {start_iter} eval {rec0}", flush=True)
            logf.write(json.dumps(rec0) + "\n")
        version = 1
        for _ in range(target_in_flight):
            submit(version)
        for it in range(start_iter + 1, start_iter + cfg.iters + 1):
            t0 = time.time()
            eps, n_stale = collect(cfg.episodes, version)
            t_roll = time.time() - t0
            # ---- assemble the batch
            F, C, M, B, A, LP, ADV, RET, IMG, SC = [], [], [], [], [], [], [], [], [], []
            for ep in eps:
                if len(ep["agent"]) == 0:
                    continue
                img, sc = critic_inputs(ep, ep["snap"], ep["agent"])
                v = critic_values(img, sc)
                fimg, fsc = critic_inputs(ep, None, np.arange(ep["spec"]["n"]), final=True)
                vf = critic_values(fimg, fsc)
                adv, ret = episode_targets(ep, v, vf, cfg.gamma, cfg.lam)
                F.append(ep["feats"]); C.append(ep["ctx"]); M.append(ep["mask"]); B.append(ep["base"])
                A.append(ep["action"]); LP.append(ep["logp"]); ADV.append(adv); RET.append(ret)
                IMG.append(img); SC.append(sc)
            feats = torch.from_numpy(np.concatenate(F)).to(dev)
            ctxs = torch.from_numpy(np.concatenate(C)).to(dev)
            masks = torch.from_numpy(np.concatenate(M)).to(dev)
            bases = torch.from_numpy(np.concatenate(B)).to(dev)
            acts = torch.from_numpy(np.concatenate(A)).to(dev)
            logp_old = torch.from_numpy(np.concatenate(LP)).to(dev)
            adv_np, ret_np = np.concatenate(ADV), np.concatenate(RET)
            vnorm.update(ret_np)
            advs = torch.from_numpy((adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)).to(dev)
            rets = torch.from_numpy((ret_np - vnorm.mean) / vnorm.std).to(dev)
            imgs = torch.from_numpy(np.concatenate(IMG)).to(dev)
            scs = torch.from_numpy(np.concatenate(SC)).to(dev)
            N = len(acts)
            frac = (it - 1) / max(1, cfg.iters + start_iter - 1)
            ent_coef = cfg.entropy + (cfg.entropy_final - cfg.entropy) * min(1.0, frac)
            # ---- PPO epochs
            stats = dict(pi=0.0, v=0.0, ent=0.0, kl=0.0, clipfrac=0.0, n=0)
            for _ in range(cfg.epochs):
                perm = torch.randperm(N, device=dev)
                for a in range(0, N, cfg.minibatch):
                    mb = perm[a:a + cfg.minibatch]
                    logits = actor(feats[mb].float(), ctxs[mb], masks[mb], bases[mb].float())
                    logp_all = torch.log_softmax(logits, dim=-1)
                    logp = logp_all.gather(1, acts[mb][:, None]).squeeze(1)
                    p_all = logp_all.exp()
                    ent = -(p_all * logp_all.masked_fill(~masks[mb], 0.0)).sum(-1).mean()
                    ratio = torch.exp(logp - logp_old[mb])
                    s1, s2 = ratio * advs[mb], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * advs[mb]
                    loss_pi = -torch.min(s1, s2).mean() - ent_coef * ent
                    opt_a.zero_grad(set_to_none=True)
                    loss_pi.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad)
                    opt_a.step()
                    v = critic(imgs[mb].float(), scs[mb])
                    loss_v = ((v - rets[mb]) ** 2).mean()
                    opt_c.zero_grad(set_to_none=True)
                    loss_v.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad)
                    opt_c.step()
                    with torch.no_grad():
                        stats["pi"] += float(loss_pi); stats["v"] += float(loss_v); stats["ent"] += float(ent)
                        stats["kl"] += float((logp_old[mb] - logp).mean())
                        stats["clipfrac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())
                        stats["n"] += 1
            k = max(1, stats.pop("n"))
            version += 1
            blob_cur[0] = blob_bytes()                   # episodes started from now on use the updated policy
            done_t = [ep["coverage_time"] if ep["completed"] else ep["final_time"] for ep in eps]
            rec = dict(iter=it, transitions=N, roll_s=round(t_roll, 1), stale_dropped=n_stale, total_s=round(time.time() - t0, 1),
                       completed=float(np.mean([ep["completed"] for ep in eps])), mean_t=float(np.mean(done_t)),
                       collisions=float(np.mean([ep["pair_events"] for ep in eps])), beta=float(actor.beta.detach()),
                       ent_coef=ent_coef, **{kk: round(vv / k, 4) for kk, vv in stats.items()})
            if it % cfg.eval_every == 0:
                rec.update(evaluate(it))
                save(out / "marl_explorer_last.pt", it)
            print(f"[marl] {json.dumps(rec)}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        rec = evaluate(start_iter + cfg.iters)
        print(f"[marl] final eval {rec}", flush=True)
        logf.write(json.dumps(rec) + "\n")
        save(out / "marl_explorer_last.pt", start_iter + cfg.iters)
    pool.terminate()                                  # episodes still in flight are not needed
    pool.join()
    return out / "marl_explorer.pt"
