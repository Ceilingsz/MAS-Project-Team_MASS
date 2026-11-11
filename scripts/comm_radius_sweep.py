"""Time to 95% against communication radius, down to 0.5 m (unknown map, 6 drones, rooms_and_door).

Same setup as the ABL experiment (r_neighbor follows r_comm), with two shorter radii added.

    python scripts/comm_radius_sweep.py --out results/comm_radius/runs.csv --jobs 14
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for p in ("src", "."):
    if p not in sys.path:
        sys.path.insert(0, p)

ALGOS = {
    "Greedy": ("greedy_explorer", dict(share=True)),
    "Geodesic": ("geodesic_partition", dict(share=True)),
    "D-MPC": ("frontier+dmpc", dict(share=True)),
    "CBBA": ("cbba_explorer", dict(share=True)),
    "MAPPO": ("marl_explorer", dict(share=True)),           # run with --only MAPPO
}
RADII = [0.5, 1.0, 1.5, 3.0, 6.0, math.inf]
FIELDS = ["label", "model", "map", "n_agents", "seed", "r_comm", "completed", "t95", "pair_events",
          "min_pair_dist", "msgs_per_drone_per_s", "status", "error"]


def run_one(job):
    import models  # noqa: F401
    from mascov.common.config import SimConfig, WorldConfig
    from mascov.common.model import get_model
    from mascov.maps.benchmarks import make_world
    from mascov.sim.simulator import Simulator
    row = {k: job[k] for k in ("label", "model", "map", "n_agents", "seed", "r_comm")}
    try:
        w = make_world(job["map"], n_agents=job["n_agents"], seed=job["seed"], cfg=WorldConfig(r_comm=job["r_comm"]))
        r = Simulator(w, get_model(job["model"], seed=job["seed"], **job["kwargs"]),
                      SimConfig(max_time=900, known_map=False, seed=job["seed"])).run()
        row.update(completed=r.completed, t95=r.time_to.get(0.95), pair_events=r.pair_events,
                   min_pair_dist=round(r.min_pair_dist, 3),
                   msgs_per_drone_per_s=round(getattr(r, "msgs_per_drone_per_s", float("nan")), 3),
                   status="ok", error="")
    except Exception as e:  # noqa: BLE001
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:200])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/comm_radius/runs.csv")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--map", default="rooms_and_door")
    ap.add_argument("--radii", type=float, nargs="+", default=RADII)
    ap.add_argument("--only", nargs="+", default=None, help="run only these labels")
    a = ap.parse_args()
    jobs = [dict(label=lab, model=m, kwargs=kw, map=a.map, n_agents=a.n, seed=s, r_comm=rc)
            for s in range(a.seeds) for rc in a.radii for lab, (m, kw) in ALGOS.items()
            if (lab in a.only if a.only else lab != "MAPPO")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(out, "w", newline="") as fh, ProcessPoolExecutor(max_workers=a.jobs) as ex:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for k, fut in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
            w.writerow(fut.result())
            fh.flush()
            if k % 12 == 0 or k == len(jobs):
                print(f"  [{k}/{len(jobs)}] {time.perf_counter() - t0:.0f} s", flush=True)
    print("written", out)


if __name__ == "__main__":
    main()
