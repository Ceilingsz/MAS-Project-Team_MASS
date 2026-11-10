"""Experiment script: Compare Known vs. Unknown Map scenarios across MAS methods.

Runs methods (B2, B3, B4, CBBA) under:
  - Known Map   (obstacles given)
  - Unknown Map (SimConfig(hide_obstacles=True): obstacles withheld and learned by sensing; targets and
                 viewpoints still given. Only CBBA re-plans on discovery; B2-B4 plan as if the space were empty.
                 For the stricter interface that also hides targets and viewpoints, see experiment E3/E3D.)

Outputs a comprehensive comparison table and saves to results/known_vs_unknown.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Add src and root to path
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import models  # noqa: F401
from mascov.common.config import SimConfig
from mascov.common.model import get_model
from mascov.maps.benchmarks import BENCHMARKS, make_world
from mascov.metrics.bounds import lower_bound
from mascov.sim.simulator import Simulator

DEFAULT_METHODS = ["B2", "B3", "B4", "CBBA"]
METHOD_MAP = dict(
    B0="greedy_single",
    B2="greedy_centralised",
    B3="greedy_uncoordinated",
    B4="static_partition",
    CBBA="cbba",
)


def run_experiment(maps, n_agents_list, seeds, methods, safety="none", max_time=400.0, out_path=None):
    results = []

    print("=" * 80)
    print("RUNNING KNOWN VS. UNKNOWN MAP COMPARISON")
    print(f"Maps: {maps}")
    print(f"Agent counts: {n_agents_list}")
    print(f"Methods: {methods}")
    print(f"Safety: {safety}")
    print("=" * 80)

    for map_name in maps:
        for n in n_agents_list:
            for seed in seeds:
                world = make_world(map_name, n_agents=n, size=12.0, seed=seed)
                lb = lower_bound(world)["best"]

                for map_mode in ["known", "unknown"]:
                    is_known = (map_mode == "known")

                    for m_label in methods:
                        # Skip single-drone specifics if any
                        if m_label == "B0" and n != 1:
                            continue
                        if m_label != "B0" and n == 1 and m_label != "B2":
                            continue

                        model_name = METHOD_MAP[m_label]
                        t0 = time.perf_counter()

                        try:
                            sc = SimConfig(
                                max_time=max_time,
                                stop_when_covered=True,
                                safety=safety,
                                hide_obstacles=not is_known,
                            )
                            model = get_model(model_name)
                            sim = Simulator(world, model, sc)
                            r = sim.run()

                            wall = round(time.perf_counter() - t0, 2)
                            T = round(r.coverage_time, 1) if r.coverage_time is not None else None
                            ratio = round(T / lb, 2) if T is not None else None
                            tot_path = round(sum(r.path_length), 1)

                            row = {
                                "map": map_name,
                                "n": n,
                                "seed": seed,
                                "mode": map_mode,
                                "method": m_label,
                                "completed": r.completed,
                                "T": T,
                                "LB": round(lb, 1),
                                "T_over_lb": ratio,
                                "path_len": tot_path,
                                "collisions": r.collision_events,
                                "min_pair": round(r.min_pair_dist, 3) if r.min_pair_dist < 100 else None,
                                "msgs": r.n_messages,
                                "wall_s": wall,
                            }
                            results.append(row)

                            print(
                                f"[{map_mode:<7}] {map_name:<14} N={n} {m_label:<4}: "
                                f"T={str(T):>6} s (LB={lb:.1f}, {str(ratio):>4}x) | "
                                f"Path={tot_path:>6.1f}m | Collisions={r.collision_events} | "
                                f"Msgs={r.n_messages:>4} | Wall={wall:>4.1f}s",
                                flush=True,
                            )

                            if out_path:
                                out = Path(out_path)
                                out.parent.mkdir(parents=True, exist_ok=True)
                                with open(out, "w", newline="") as f:
                                    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                                    writer.writeheader()
                                    writer.writerows(results)

                        except Exception as e:
                            print(f"[{map_mode:<7}] {map_name:<14} N={n} {m_label:<4}: FAILED ({e})", flush=True)
                            results.append({
                                "map": map_name,
                                "n": n,
                                "seed": seed,
                                "mode": map_mode,
                                "method": m_label,
                                "completed": False,
                                "T": None,
                                "LB": round(lb, 1),
                                "T_over_lb": None,
                                "path_len": 0,
                                "collisions": 0,
                                "min_pair": None,
                                "msgs": 0,
                                "wall_s": round(time.perf_counter() - t0, 2),
                            })

    if out_path and results:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved results to {out}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Known vs. Unknown map coverage comparison")
    parser.add_argument("--maps", nargs="+", default=["empty", "sparse", "corridor", "rooms_and_door"])
    parser.add_argument("--n", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--safety", choices=["none", "bvc"], default="none")
    parser.add_argument("--max-time", type=float, default=300.0)
    parser.add_argument("--out", default="docs/known_vs_unknown_results.csv")
    args = parser.parse_args()

    run_experiment(
        maps=args.maps,
        n_agents_list=args.n,
        seeds=args.seeds,
        methods=args.methods,
        safety=args.safety,
        max_time=args.max_time,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
