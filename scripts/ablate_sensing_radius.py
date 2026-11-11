"""Ablation: does knowing the map start to matter when the sensing radius shrinks?

For every (map, seed) the obstacle layout is built ONCE at the default configuration and then reused at every
sensing radius, so only R_s changes. (The random benchmark maps retry until 90% of the targets are coverable with
the configuration they are given; building them at each R_s would give a different layout per radius.)

At every R_s the candidate standing positions use a 0.5 m lattice (default 1.0 m), so that down to R_s = 0.5 m every
open-space point is within the effective sensing radius (R_s - sense_slack) of some lattice point, and the lattice
is the same at every radius. Each run's task is "cover every coverable target" for its own R_s: near walls and
obstacles some cells become unseeable at small R_s (clearance keeps the drone too far away); the fraction is
recorded per run (``coverable_frac``) and is identical for the known and unknown runs that are compared.

Map known vs unknown is the experiment runner's convention: the unknown-map interface, with the obstacles loaded at
the start for "known" (``reveal_obstacles_at_start=True``), so the same algorithm is compared with itself.

    python scripts/ablate_sensing_radius.py --out results/ablate_sensing/runs.csv --jobs 8
    python scripts/ablate_sensing_radius.py --quick --out results/ablate_sensing/quick.csv
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

MODELS = {                                   # the Q1-Q3 set in definitions.py (all with shared beliefs)
    "Greedy": ("greedy_explorer", dict(share=True)),
    "Greedy + goals": ("greedy_explorer", dict(share=True, coordinate=True)),
    "Geodesic": ("geodesic_partition", dict(share=True)),
    "CBBA": ("cbba_explorer", dict(share=True)),
    "D-MPC": ("frontier+dmpc", dict(share=True)),
    "MAPPO": ("marl_explorer", dict(share=True)),           # run with --only MAPPO
}
RADII = [0.5, 0.75, 1.0, 1.5, 2.0]
MAPS = ["sparse", "rooms_and_door", "cluttered"]
FIELDS = ["label", "model", "map", "n_agents", "seed", "r_sense", "prior", "completed", "t50", "t95", "t100",
          "final_coverage", "coverable_frac", "path_len_mean", "pair_events", "obstacle_events", "min_clearance",
          "map_rebuilds", "wall_s", "status", "error"]


def run_one(job: dict) -> dict:
    import models  # noqa: F401  (registers every model in this worker)
    from mascov.common.config import SimConfig, WorldConfig
    from mascov.common.model import get_model
    from mascov.maps.benchmarks import make_world
    from mascov.sim.simulator import Simulator
    from mascov.world.world import World

    row = {k: job[k] for k in ("label", "model", "map", "n_agents", "seed", "r_sense", "prior")}
    t0 = time.perf_counter()
    try:
        base = make_world(job["map"], n_agents=job["n_agents"], seed=job["seed"])      # geometry at the default cfg
        cfg = WorldConfig(r_sense=job["r_sense"], viewpoint_spacing=0.5)
        w = World(cfg, list(base.obstacles), base.starts.copy(), name=base.name)
        model = get_model(job["model"], seed=job["seed"], **job["kwargs"])
        sim = SimConfig(max_time=job["max_time"], known_map=False, reveal_obstacles_at_start=(job["prior"] == "known"),
                        seed=job["seed"])
        r = Simulator(w, model, sim).run()
        row.update(completed=r.completed, t50=r.time_to.get(0.5), t95=r.time_to.get(0.95), t100=r.coverage_time,
                   final_coverage=round(r.final_coverage, 5), coverable_frac=round(float(w.coverable.mean()), 4),
                   path_len_mean=round(sum(r.path_length) / len(r.path_length), 2), pair_events=r.pair_events,
                   obstacle_events=r.obstacle_events, min_clearance=round(r.min_clearance, 3),
                   map_rebuilds=r.extra.get("map_rebuilds"), status="ok", error="")
    except Exception as e:  # a failed run must not kill the sweep
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/ablate_sensing/runs.csv")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-time", type=float, default=1200.0)
    ap.add_argument("--radii", type=float, nargs="+", default=RADII)
    ap.add_argument("--quick", action="store_true", help="1 map, 2 seeds, 2 radii")
    ap.add_argument("--maps", nargs="+", default=None, help="maps (default: sparse, rooms_and_door, cluttered)")
    ap.add_argument("--only", nargs="+", default=None, help="run only these labels")
    a = ap.parse_args()
    MAPS[:] = a.maps or MAPS

    maps, radii, seeds = (["rooms_and_door"], [0.5, 1.5], [0, 1]) if a.quick else (MAPS, a.radii, list(range(a.seeds)))
    # seed-major order: a sweep stopped part-way still has every algorithm, map and radius
    jobs = [dict(label=lab, model=m, kwargs=kw, map=mp, n_agents=a.n, seed=s, r_sense=rs, prior=prior,
                 max_time=a.max_time)
            for s in seeds for mp in maps for rs in radii for lab, (m, kw) in MODELS.items() if (lab in a.only if a.only else lab != "MAPPO")
            for prior in ("known", "unknown")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    key = lambda r: (r["label"], r["map"], str(r["seed"]), str(float(r["r_sense"])), r["prior"])   # noqa: E731
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {key(r) for r in csv.DictReader(f) if r["status"] == "ok"}
    todo = [j for j in jobs if key({**j, "seed": j["seed"], "r_sense": j["r_sense"]}) not in done]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already done, {len(todo)} to run, {a.jobs} workers", flush=True)
    new_file = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f, ProcessPoolExecutor(max_workers=a.jobs) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for k, fut in enumerate(as_completed([ex.submit(run_one, j) for j in todo]), 1):
            row = fut.result()
            w.writerow(row)
            f.flush()
            if k % 25 == 0 or k == len(todo) or row["status"] != "ok":
                el = time.perf_counter() - t0
                print(f"  [{k}/{len(todo)}] {row['label']} {row['map']} R={row['r_sense']} {row['prior']} "
                      f"{row['status']} {row['error']} (eta {el / k * (len(todo) - k):.0f} s)", flush=True)
    print("written", out)


if __name__ == "__main__":
    main()
