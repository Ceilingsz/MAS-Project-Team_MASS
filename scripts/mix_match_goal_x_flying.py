"""Mix and match: goal layer {Greedy, Geodesic, CBBA} x flying layer {Navigator, D-MPC}, unknown map.

The D-MPC column runs the SAME goal-policy code as the Navigator column (models/mix_dmpc), so differences come from
the flying layer. All models share beliefs (share=True), as in the Q1-Q3 experiments.

    python scripts/mix_match_goal_x_flying.py --out results/mix_match/runs.csv --jobs 8
    python scripts/mix_match_goal_x_flying.py --analyse results/mix_match/runs.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for p in ("src", "."):
    if p not in sys.path:
        sys.path.insert(0, p)

GOALS = {"Greedy": "greedy_explorer", "Geodesic": "geodesic_partition", "CBBA": "cbba_explorer",
         "MAPPO": "marl_explorer"}   # MAPPO: the learned goal layer (needs torch), run it with --only MAPPO
FLYING = {"Navigator": "", "D-MPC": "+dmpc"}
MAPS = ["sparse", "rooms_and_door", "cluttered"]
FIELDS = ["goal", "flying", "model", "map", "n_agents", "seed", "completed", "t95", "t100", "pair_events",
          "obstacle_events", "min_pair_dist", "path_len_mean", "msgs_per_drone_per_s", "solve_ms", "mean_visits",
          "wall_s", "status", "error"]


def run_one(job: dict) -> dict:
    import models  # noqa: F401
    from mascov.common.config import SimConfig
    from mascov.common.model import get_model
    from mascov.maps.benchmarks import make_world
    from mascov.sim.simulator import Simulator

    row = {k: job[k] for k in ("goal", "flying", "model", "map", "n_agents", "seed")}
    t0 = time.perf_counter()
    try:
        w = make_world(job["map"], n_agents=job["n_agents"], seed=job["seed"])
        r = Simulator(w, get_model(job["model"], share=True, seed=job["seed"]),
                      SimConfig(max_time=job["max_time"], known_map=False, seed=job["seed"])).run()
        m = r.metrics()
        row.update(completed=r.completed, t95=r.time_to.get(0.95), t100=r.coverage_time, pair_events=r.pair_events,
                   obstacle_events=r.obstacle_events, min_pair_dist=round(r.min_pair_dist, 3),
                   path_len_mean=round(sum(r.path_length) / len(r.path_length), 2),
                   msgs_per_drone_per_s=m.get("msgs_per_drone_per_s"), solve_ms=m.get("solve_ms"),
                   mean_visits=round(r.mean_visits, 2), status="ok", error="")
    except Exception as e:  # a failed run must not kill the sweep
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    return row


def analyse(path: Path) -> None:
    import numpy as np
    rows = [r for r in csv.DictReader(open(path, newline="")) if r["status"] == "ok"]
    f = lambda x: float(x) if x not in ("", "None") else math.nan   # noqa: E731
    print(f"{len(rows)} runs; incomplete: {sum(r['completed'] != 'True' for r in rows)}")
    cell = defaultdict(list)
    for r in rows:
        cell[(r["goal"], r["flying"], int(r["n_agents"]))].append(r)
    for n in sorted({k[2] for k in cell}):
        print(f"\n== {n} drones (mean over maps x seeds)")
        print(f"{'goal':10s}{'flying':11s}{'t95 [s]':>9s}{'t100 [s]':>10s}{'collision ep.':>15s}{'runs w/ coll.':>15s}"
              f"{'min sep [m]':>13s}{'msgs/drone/s':>14s}{'solve ms':>10s}")
        for g in GOALS:
            for fl in FLYING:
                rs = cell[(g, fl, n)]
                if not rs:
                    continue
                t95 = np.nanmean([f(r["t95"]) for r in rs])
                t100 = np.nanmean([f(r["t100"]) for r in rs])
                coll = np.mean([int(r["pair_events"]) for r in rs])
                frac = np.mean([int(r["pair_events"]) > 0 for r in rs])
                sep = np.min([f(r["min_pair_dist"]) for r in rs])
                msg = np.nanmean([f(r["msgs_per_drone_per_s"]) for r in rs])
                ms = np.nanmean([f(r["solve_ms"]) for r in rs])
                print(f"{g:10s}{fl:11s}{t95:9.1f}{t100:10.1f}{coll:15.2f}{frac:14.0%} {sep:12.3f}{msg:14.1f}{ms:10.2f}")
    # paired effect of swapping the flying layer (same goal, map, n, seed)
    print("\n== D-MPC / Navigator, paired (geometric mean of t95 ratios, 95% bootstrap CI)")
    idx = {(r["goal"], r["flying"], r["map"], r["n_agents"], r["seed"]): r for r in rows}
    rng = np.random.default_rng(0)
    for g in GOALS:
        lr = []
        for (gg, fl, mp, n, s), r in idx.items():
            if gg == g and fl == "D-MPC" and (g, "Navigator", mp, n, s) in idx:
                a, b = f(r["t95"]), f(idx[(g, "Navigator", mp, n, s)]["t95"])
                if a > 0 and b > 0:
                    lr.append(math.log(a / b))
        lr = np.array(lr)
        boots = [rng.choice(lr, len(lr)).mean() for _ in range(2000)]
        print(f"{g:10s} x{math.exp(lr.mean()):.2f} [{math.exp(np.percentile(boots, 2.5)):.2f}, "
              f"{math.exp(np.percentile(boots, 97.5)):.2f}]  n={len(lr)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mix_match/runs.csv")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--max-time", type=float, default=900.0)
    ap.add_argument("--analyse", default=None)
    ap.add_argument("--maps", nargs="+", default=None, help="maps (default: sparse, rooms_and_door, cluttered)")
    ap.add_argument("--only", nargs="+", default=["Greedy", "Geodesic", "CBBA"], help="goal rules to run")
    a = ap.parse_args()
    MAPS[:] = a.maps or MAPS
    if a.analyse:
        analyse(Path(a.analyse))
        return
    jobs = [dict(goal=g, flying=fl, model=GOALS[g] + FLYING[fl], map=mp, n_agents=n, seed=s, max_time=a.max_time)
            for s in range(a.seeds) for mp in MAPS for n in a.n for g in GOALS if g in a.only for fl in FLYING]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    key = lambda r: (r["model"], r["map"], str(r["n_agents"]), str(r["seed"]))   # noqa: E731
    done = set()
    if out.exists():
        done = {key(r) for r in csv.DictReader(open(out, newline="")) if r["status"] == "ok"}
    todo = [j for j in jobs if key({k: str(v) for k, v in j.items()}) not in done]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as fh, ProcessPoolExecutor(max_workers=a.jobs) as ex:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for k, fut in enumerate(as_completed([ex.submit(run_one, j) for j in todo]), 1):
            row = fut.result()
            w.writerow(row)
            fh.flush()
            if k % 20 == 0 or k == len(todo) or row["status"] != "ok":
                el = time.perf_counter() - t0
                print(f"  [{k}/{len(todo)}] {row['model']} {row['map']} n={row['n_agents']} {row['status']} {row['error']} "
                      f"(eta {el / k * (len(todo) - k):.0f} s)", flush=True)
    print("written", out)


if __name__ == "__main__":
    main()
