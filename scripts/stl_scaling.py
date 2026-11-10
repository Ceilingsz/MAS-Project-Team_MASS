"""Measure how the single-agent STL solve scales with the number of goals and obstacles.

  STL_PLANNING_PATH=... python scripts/stl_scaling.py [--out docs/stl_scaling.csv]
Run from the repository root. Needs gurobipy and the STLPlanning checkout.
"""
import argparse
import csv
import sys
import time

import numpy as np

sys.path.insert(0, "src"), sys.path.insert(0, ".")
from mascov.common.config import WorldConfig  # noqa: E402
from mascov.geometry.polygons import random_convex_polygon  # noqa: E402
from mascov.stl.planner import STLPlanner, verify_plan  # noqa: E402


def instance(cfg, n_goals, n_obs, seed):
    rng = np.random.default_rng(seed)
    start = np.array([1.5, 1.5])
    goals = []
    while len(goals) < n_goals:
        g = rng.uniform(2.0, cfg.width - 2.0, 2)
        if all(np.linalg.norm(g - h) > 2.5 for h in goals) and np.linalg.norm(g - start) > 2.5:
            goals.append(g)
    obs = []
    tries = 0
    while len(obs) < n_obs and tries < 500:
        tries += 1
        c = rng.uniform(1.0, cfg.width - 1.0, 2)
        poly = random_convex_polygon(rng, c[0], c[1], rng.uniform(0.5, 0.9))
        if all(poly.distance(__import__("shapely").geometry.Point(p)) > 1.0 for p in [start, *goals]) \
                and all(poly.distance(o) > 1.2 for o in obs):
            obs.append(poly)
    return start, np.array(goals), obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--obstacles", nargs="+", type=int, default=[0, 3, 6])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--fixed-k", action="store_true", help="solve for exactly K=2*goals segments instead of searching upward")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = WorldConfig()
    planner = STLPlanner(cfg, time_limit=args.time_limit)
    rows = []
    print(f"{'goals':>5} {'obst':>4} {'seed':>4} {'K':>3} {'ok':>5} {'solve_s':>8} {'binaries':>8}  verified  note")
    for g in args.goals:
        for o in args.obstacles:
            for seed in args.seeds:
                start, goals, obs = instance(cfg, g, o, seed)
                K = 2 * g
                tmax = 60.0 * g
                if args.fixed_k:
                    r = planner.plan_visits(start, goals, obs, tmax=tmax, num_segs=K)
                else:      # the library's intended mode: try K=1,2,... until feasible
                    r = planner.plan_visits(start, goals, obs, tmax=tmax, max_segs=2 * g + o + 2)
                ver = ""
                if r.ok:
                    ver = verify_plan(r.waypoints, start, goals, obs, cfg, tmax=tmax)["all_ok"]
                rows.append(dict(goals=g, obstacles=o, seed=seed, K=r.n_segs if r.ok else K, ok=r.ok, solve_s=round(r.solve_s, 2),
                                 binaries=r.n_binaries, verified=ver, note=r.message))
                print(f"{g:5d} {o:4d} {seed:4d} {(r.n_segs if r.ok else K):3d} {str(r.ok):>5} {r.solve_s:8.2f} {str(r.n_binaries):>8}  {str(ver):>8}  {r.message}", flush=True)
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(), w.writerows(rows)


if __name__ == "__main__":
    main()
