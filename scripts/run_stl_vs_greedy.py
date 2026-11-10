"""Single-drone comparison: windowed STL planner vs greedy baselines, same maps and same clearance.

  STL_PLANNING_PATH=... python scripts/run_stl_vs_greedy.py [--eps 0.3 0.45] [--out docs/stl_vs_greedy.csv]
Run from the repository root; needs gurobipy and the STLPlanning checkout.
"""
import argparse
import csv
import sys
import time

sys.path.insert(0, "src"), sys.path.insert(0, ".")
import models  # noqa: F401, E402
from mascov.common.config import SimConfig, WorldConfig  # noqa: E402
from mascov.common.model import get_model  # noqa: E402
from mascov.maps.benchmarks import BENCHMARKS, make_world  # noqa: E402
from mascov.metrics.bounds import lower_bound  # noqa: E402
from mascov.sim.simulator import Simulator  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--maps", nargs="+", default=list(BENCHMARKS))
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--eps", nargs="+", type=float, default=[0.3, 0.45])
ap.add_argument("--window-goals", type=int, default=4)
ap.add_argument("--out", default=None)
args = ap.parse_args()

rows = []
hdr = f"{'eps':>5} {'map':>15} {'seed':>4} {'method':>20} {'T':>7} {'T/LB':>5} {'eps_meas':>8} {'clear':>6} {'coll':>4} {'windows':>7} {'fallbk':>6} {'verif_fail':>10} {'stl_s':>6}"
print(hdr)
for eps in args.eps:
    cfg = WorldConfig(eps_track=eps)
    for name in args.maps:
        for seed in args.seeds:
            w = make_world(name, n_agents=1, seed=seed, cfg=cfg)
            lb = lower_bound(w)["best"]
            for m in ("greedy_single", "greedy_centralised", "stl_single"):
                model = get_model(m, window_goals=args.window_goals) if m == "stl_single" else get_model(m)
                r = Simulator(w, model, SimConfig(max_time=900)).run()
                s = r.summary()
                row = dict(eps_track=eps, map=name, seed=seed, method=m, completed=r.completed, T=r.coverage_time,
                           T_over_lb=None if r.coverage_time is None else round(r.coverage_time / lb, 2),
                           eps_measured=max(r.max_tracking_error), eps_ok=max(r.max_tracking_error) <= eps,
                           min_clearance=s["min_clearance"], collisions=r.collision_events,
                           windows=r.extra.get("stl_windows", ""), fallbacks=r.extra.get("stl_fallbacks", ""),
                           verify_fail=r.extra.get("stl_verify_fail", ""), stl_solve_s=r.extra.get("stl_solve_s", ""))
                rows.append(row)
                print(f"{eps:5.2f} {name:>15} {seed:4d} {m:>20} {str(row['T'])[:6]:>7} {str(row['T_over_lb']):>5} "
                      f"{row['eps_measured']:8.3f} {row['min_clearance']:6.3f} {row['collisions']:4d} {str(row['windows']):>7} "
                      f"{str(row['fallbacks']):>6} {str(row['verify_fail']):>10} {str(row['stl_solve_s']):>6}", flush=True)
if args.out:
    with open(args.out, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader(), w_.writerows(rows)
