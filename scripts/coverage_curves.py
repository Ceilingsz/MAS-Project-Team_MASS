"""Coverage over time for every algorithm (unknown map, shared beliefs, 4 drones), averaged over maps and seeds.

    python scripts/coverage_curves.py --out results/coverage_curves/curves.json --jobs 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for p in ("src", "."):
    if p not in sys.path:
        sys.path.insert(0, p)

ALGOS = {   # task allocation {Greedy, Geodesic, CBBA} x navigation {Navigator, D-MPC}
    "Greedy + Navigator": ("greedy_explorer", dict(share=True)),
    "Geodesic + Navigator": ("geodesic_partition", dict(share=True)),
    "CBBA + Navigator": ("cbba_explorer", dict(share=True)),
    "Greedy + D-MPC": ("greedy_explorer+dmpc", dict(share=True)),
    "Geodesic + D-MPC": ("geodesic_partition+dmpc", dict(share=True)),
    "CBBA + D-MPC": ("cbba_explorer+dmpc", dict(share=True)),
    "MAPPO + Navigator": ("marl_explorer", dict(share=True)),     # run with --only
    "MAPPO + D-MPC": ("marl_explorer+dmpc", dict(share=True)),
}
MAPS = ["sparse", "rooms_and_door", "cluttered"]


def run_one(job):
    import models  # noqa: F401
    from mascov.common.config import SimConfig
    from mascov.common.model import get_model
    from mascov.maps.benchmarks import make_world
    from mascov.sim.simulator import Simulator
    w = make_world(job["map"], n_agents=job["n"], seed=job["seed"])
    r = Simulator(w, get_model(job["model"], seed=job["seed"], **job["kwargs"]),
                  SimConfig(max_time=600, known_map=False, seed=job["seed"], log_every=1)).run()
    t, c = r.coverage_curve()
    return dict(label=job["label"], map=job["map"], seed=job["seed"], t=[float(x) for x in t],
                cov=[float(x) for x in c], t95=r.time_to.get(0.95), t100=r.coverage_time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/coverage_curves/curves.json")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--maps", nargs="+", default=None, help="maps (default: sparse, rooms_and_door, cluttered)")
    ap.add_argument("--only", nargs="+", default=None, help="run only these labels")
    a = ap.parse_args()
    MAPS[:] = a.maps or MAPS
    jobs = [dict(label=lab, model=m, kwargs=kw, map=mp, n=a.n, seed=s)
            for s in range(a.seeds) for mp in MAPS for lab, (m, kw) in ALGOS.items()
            if (lab in a.only if a.only else not lab.startswith("MAPPO"))]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, fut in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
            res.append(fut.result())
            if k % 9 == 0 or k == len(jobs):
                print(f"  [{k}/{len(jobs)}]", flush=True)
    out.write_text(json.dumps(res))
    print("written", out)


if __name__ == "__main__":
    main()
