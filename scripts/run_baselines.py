"""Run baselines B0 to B4 on benchmark maps and print / save a comparison table.

  python scripts/run_baselines.py --maps empty rooms_and_door --n 2 4 --seeds 0 1 --out results/baselines.csv
  add --safety bvc to enable the BVC safety filter; add --full to keep running until every drone believes it is done (shows B3's believed-vs-true gap)
Run from the repository root.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src"), sys.path.insert(0, ".")
import models  # noqa: F401, E402  (registers all models)
from mascov.common.config import SimConfig  # noqa: E402
from mascov.common.model import get_model  # noqa: E402
from mascov.maps.benchmarks import BENCHMARKS, make_world  # noqa: E402
from mascov.metrics.bounds import lower_bound  # noqa: E402
from mascov.sim.simulator import Simulator  # noqa: E402

METHODS = dict(B0="greedy_single", B2="greedy_centralised", B3="greedy_uncoordinated", B4="static_partition")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--n", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--size", type=float, default=12.0)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--max-time", type=float, default=1500.0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--safety", default="none", choices=["none", "bvc"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for name in args.maps:
        for n in args.n:
            for seed in args.seeds:
                w = make_world(name, n_agents=n, size=args.size, seed=seed)
                lb = lower_bound(w)["best"]
                for label in args.methods:
                    if label == "B0" and n != 1:
                        continue
                    if label != "B0" and n == 1 and label != "B2":
                        continue
                    t0 = time.perf_counter()
                    sc = SimConfig(max_time=args.max_time, stop_when_covered=not args.full, safety=args.safety)
                    r = Simulator(w, get_model(METHODS[label]), sc).run()
                    s = r.summary()
                    rows.append(dict(map=name, n=n, seed=seed, method=label, completed=r.completed,
                                     T=r.coverage_time, lower_bound=round(lb, 1),
                                     T_over_lb=None if r.coverage_time is None else round(r.coverage_time / lb, 2),
                                     believed_done_max=None if any(b is None for b in r.believed_done) else max(r.believed_done),
                                     collisions=r.collision_events, min_pair=s["min_pair_dist"],
                                     yields=r.extra.get("yields", ""), wait_s=r.extra.get("wait_s", ""),
                                     sidesteps=r.extra.get("sidesteps", ""), pushes=r.extra.get("pushes", ""), interventions=r.extra.get("safety_interventions", ""),
                                     path_len=round(sum(r.path_length), 1), eps=round(max(r.max_tracking_error), 3),
                                     wall_s=round(time.perf_counter() - t0, 1)))
    if not rows:
        return
    cols = list(rows[0].keys())
    print(" ".join(f"{c[:11]:>11s}" for c in cols))
    for r in rows:
        print(" ".join(f"{str(r[c])[:11]:>11s}" for c in cols))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader(), wr.writerows(rows)
        print("saved", args.out)


if __name__ == "__main__":
    main()
