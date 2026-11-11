"""Collisions vs team size for every algorithm (unknown map, shared beliefs).

    python scripts/collisions_vs_team_size.py --out results/collisions_n/runs.csv --jobs 8
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for p in ("src", "."):
    if p not in sys.path:
        sys.path.insert(0, p)

ALGOS = {   # the deck's five, plus geodesic goals flown by D-MPC
    "Greedy": ("greedy_explorer", dict(share=True)),
    "Greedy + goals": ("greedy_explorer", dict(share=True, coordinate=True)),
    "Geodesic": ("geodesic_partition", dict(share=True)),
    "CBBA": ("cbba_explorer", dict(share=True)),
    "D-MPC": ("frontier+dmpc", dict(share=True)),
    "Geodesic + D-MPC": ("geodesic_partition+dmpc", dict(share=True)),
    "Greedy + D-MPC": ("greedy_explorer+dmpc", dict(share=True)),     # completes the 3 x 2 grid
    "CBBA + D-MPC": ("cbba_explorer+dmpc", dict(share=True)),
}
MAPS = ["sparse", "rooms_and_door", "cluttered"]
FIELDS = ["label", "model", "map", "n_agents", "seed", "completed", "t95", "pair_events", "min_pair_dist", "status", "error"]


def run_one(job):
    import models  # noqa: F401
    from mascov.common.config import SimConfig
    from mascov.common.model import get_model
    from mascov.maps.benchmarks import make_world
    from mascov.sim.simulator import Simulator
    row = {k: job[k] for k in ("label", "model", "map", "n_agents", "seed")}
    try:
        w = make_world(job["map"], n_agents=job["n_agents"], seed=job["seed"])
        r = Simulator(w, get_model(job["model"], seed=job["seed"], **job["kwargs"]),
                      SimConfig(max_time=900, known_map=False, seed=job["seed"])).run()
        row.update(completed=r.completed, t95=r.time_to.get(0.95), pair_events=r.pair_events,
                   min_pair_dist=round(r.min_pair_dist, 3), status="ok", error="")
    except Exception as e:  # noqa: BLE001
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:200])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/collisions_n/runs.csv")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--only", nargs="+", default=None, help="run only these labels")
    ap.add_argument("--append", action="store_true", help="add rows to an existing file")
    a = ap.parse_args()
    algos = {k: v for k, v in ALGOS.items() if a.only is None or k in a.only}
    jobs = [dict(label=lab, model=m, kwargs=kw, map=mp, n_agents=n, seed=s)
            for s in range(a.seeds) for mp in MAPS for n in a.n for lab, (m, kw) in algos.items()]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(out, "a" if a.append else "w", newline="") as fh, ProcessPoolExecutor(max_workers=a.jobs) as ex:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not a.append:
            w.writeheader()
        for k, fut in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
            w.writerow(fut.result())
            fh.flush()
            if k % 24 == 0 or k == len(jobs):
                print(f"  [{k}/{len(jobs)}] {time.perf_counter() - t0:.0f} s", flush=True)
    print("written", out)


if __name__ == "__main__":
    main()
