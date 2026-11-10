"""Multi-drone comparison: joint STL planner (stl_team) vs B2 / B4 / B3 on the same maps and start layout.

  python scripts/run_stl_team.py [--n 2 3 4] [--out docs/stl_team_vs_baselines.csv]
Run from the repository root; needs scipy >= 1.9 (HiGHS solver). Uses eps_track=0.45 and 2 m start
spacing (the joint planner needs drones >= 2*eps + d_safe = 1.2 m apart, infinity norm, at the start of every window).
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
ap.add_argument("--n", nargs="+", type=int, default=[2, 3, 4])
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--methods", nargs="+", default=["greedy_centralised", "static_partition", "greedy_uncoordinated", "stl_team"])
ap.add_argument("--eps", type=float, default=0.45)
ap.add_argument("--spacing", type=float, default=2.0)
ap.add_argument("--goals-per-drone", type=int, default=2)
ap.add_argument("--max-time", type=float, default=900.0)
ap.add_argument("--out", default=None)
args = ap.parse_args()

cfg = WorldConfig(eps_track=args.eps)
rows = []
print(f"{'map':>15} {'N':>2} {'seed':>4} {'method':>20} {'T':>7} {'T/LB':>5} {'min_pair':>8} {'coll':>4} {'eps_meas':>8} {'windows':>7} {'fallbk':>6} {'degr':>4} {'vfail':>5} {'stl_s':>6} {'wall':>5}")
for name in args.maps:
    for n in args.n:
        for seed in args.seeds:
            w = make_world(name, n_agents=n, seed=seed, cfg=cfg, start_spacing=args.spacing, start_per_row=2)
            lb = lower_bound(w)["best"]
            for m in args.methods:
                model = get_model(m, goals_per_drone=args.goals_per_drone) if m == "stl_team" else get_model(m)
                t0 = time.perf_counter()
                r = Simulator(w, model, SimConfig(max_time=args.max_time)).run()
                s = r.summary()
                x = r.extra
                row = dict(map=name, n=n, seed=seed, method=m, completed=r.completed, T=r.coverage_time,
                           T_over_lb=None if r.coverage_time is None else round(r.coverage_time / lb, 2),
                           min_pair=s["min_pair_dist"], collisions=r.collision_events,
                           eps_measured=round(max(r.max_tracking_error), 3), windows=x.get("team_windows", ""),
                           fallbacks=x.get("team_fallbacks", ""), degraded=x.get("team_degraded", ""),
                           verify_fail=x.get("team_verify_fail", ""), stl_solve_s=x.get("team_solve_s", ""),
                           wall_s=round(time.perf_counter() - t0, 1))
                rows.append(row)
                print(f"{name:>15} {n:2d} {seed:4d} {m:>20} {str(row['T'])[:6]:>7} {str(row['T_over_lb']):>5} {str(row['min_pair']):>8} "
                      f"{row['collisions']:4d} {row['eps_measured']:8.3f} {str(row['windows']):>7} {str(row['fallbacks']):>6} "
                      f"{str(row['degraded']):>4} {str(row['verify_fail']):>5} {str(row['stl_solve_s']):>6} {row['wall_s']:5.0f}", flush=True)
if args.out:
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(), wr.writerows(rows)
