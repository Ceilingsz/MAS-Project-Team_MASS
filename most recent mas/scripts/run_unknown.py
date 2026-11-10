"""Unknown-map runs next to the known-map baselines, on the same worlds and seeds.

  python scripts/run_unknown.py --maps sparse corridor --n 2 4 --seeds 0 1 --out results/unknown.csv

Methods (all with the same planner/tracker underneath):
  B3        known map,   no sharing       greedy_uncoordinated   (known-map floor)
  B2        known map,   central claims   greedy_centralised     (known-map reference)
  F-map     map given,   no sharing       frontier_greedy        (same algorithm as F-unk, obstacles pre-loaded)
  F-unk     map unknown, no sharing       frontier_greedy
  F-shared  map unknown, merge on contact frontier_greedy_shared
  F-shared-map  the same with obstacles pre-loaded

F-shared vs F-shared-map isolates what NOT KNOWING THE MAP costs once drones share what they learn
(without sharing, duplicated work swamps everything: compare F-map vs F-unk). F-unk vs F-shared is what
sharing recovers. B2 vs F-shared-map is, roughly, what central claims add over merging beliefs.
The seed is passed to every model that takes one (random tie-breaking), so seeds differ even on the
deterministic maps (empty, corridor, rooms_and_door).
Run from the repository root.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src"), sys.path.insert(0, ".")
import models  # noqa: F401, E402
from mascov.common.config import SimConfig  # noqa: E402
from mascov.common.model import get_model  # noqa: E402
from mascov.maps.benchmarks import BENCHMARKS, make_world  # noqa: E402
from mascov.sim.simulator import Simulator  # noqa: E402

METHODS = {          # label: (model, SimConfig kwargs, does the model take a seed?)
    "B3": ("greedy_uncoordinated", dict(), True),
    "B2": ("greedy_centralised", dict(), False),
    "F-map": ("frontier_greedy", dict(known_map=False, reveal_obstacles_at_start=True), True),
    "F-unk": ("frontier_greedy", dict(known_map=False), True),
    "F-shared": ("frontier_greedy_shared", dict(known_map=False), True),
    "F-shared-map": ("frontier_greedy_shared", dict(known_map=False, reveal_obstacles_at_start=True), True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--n", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--size", type=float, default=12.0)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--r-comm", type=float, default=float("inf"))
    ap.add_argument("--max-time", type=float, default=1500.0)
    ap.add_argument("--safety", default="none", choices=["none", "bvc"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for name in args.maps:
        for n in args.n:
            for seed in args.seeds:
                w = make_world(name, n_agents=n, size=args.size, seed=seed)
                if not np.isinf(args.r_comm):
                    from mascov.world.world import World
                    w = World(w.cfg.with_(r_comm=args.r_comm), w.obstacles, w.starts, name=w.name)
                for label in args.methods:
                    model, kw, seeded = METHODS[label]
                    t0 = time.perf_counter()
                    sc = SimConfig(max_time=args.max_time, safety=args.safety, **kw)
                    r = Simulator(w, get_model(model, **(dict(seed=seed) if seeded else {})), sc).run()
                    rows.append(dict(map=name, n=n, seed=seed, method=label, completed=r.completed,
                                     T=None if r.coverage_time is None else round(r.coverage_time, 1),
                                     collisions=r.collision_events, min_clear=round(r.min_clearance, 3),
                                     min_pair=None if not np.isfinite(r.min_pair_dist) else round(r.min_pair_dist, 3),
                                     path_len=round(sum(r.path_length), 1), msgs=r.n_messages,
                                     rebuilds=r.extra.get("map_rebuilds", ""),
                                     wall_s=round(time.perf_counter() - t0, 1)))
                print(f"{name} N={n} seed={seed}: " + "  ".join(
                    f"{x['method']}={x['T']}" for x in rows[-len(args.methods):]), flush=True)
    if not rows:
        return
    print("\nmean coverage time [s] by N (completed runs only; ratio to B2 in brackets)")
    print(f"{'N':>3s} " + " ".join(f"{m:>15s}" for m in args.methods))
    for n in args.n:
        cells, b2 = [], None
        for m in args.methods:
            ts = [x["T"] for x in rows if x["n"] == n and x["method"] == m and x["T"] is not None]
            mean = float(np.mean(ts)) if ts else float("nan")
            if m == "B2":
                b2 = mean
            cells.append((m, mean))
        print(f"{n:>3d} " + " ".join(
            f"{mean:>7.1f}" + (f" ({mean / b2:4.2f}x)" if b2 else "        ") for _, mean in cells))
    print("incomplete runs:", sum(not x["completed"] for x in rows), "of", len(rows),
          "| runs with a collision episode:", sum(x["collisions"] > 0 for x in rows),
          "| min obstacle/wall clearance:", min(x["min_clear"] for x in rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(), wr.writerows(rows)
        print("saved", args.out)


if __name__ == "__main__":
    main()
