"""Greedy (coverage / distance) on a KNOWN map vs an UNKNOWN map, with paired statistics.

The greedy rule under test: every drone goes to the viewpoint maximising
    (new targets seen) / (travel time + delta)
decided from its own belief, no messages.

  K    known map     greedy_uncoordinated   (mascov's B3; true obstacles, viewpoints and targets given)
  U    unknown map   frontier_greedy        (same rule, but counts UNKNOWN cells; obstacles discovered by sensing)
  Uc   control       frontier_greedy        (same code as U, obstacles pre-loaded: isolates "not knowing the map")
  (--shared adds the belief-merging variants Us / Ucs)

Comparisons (paired over map x N x seed):
  K  vs U   headline: known-map greedy against unknown-map greedy (different code paths: viewpoint set, belief type)
  Uc vs U   controlled: identical algorithm, only the map prior differs

A gap is called significant when the paired difference is both statistically clear (Wilcoxon signed-rank
p < 0.01 AND |mean paired %-diff| > 2 x its standard error) and practically large (|mean %-diff| >= 10%).

  python scripts/greedy_known_vs_unknown.py --out results/gku.csv --jobs 4
  python scripts/greedy_known_vs_unknown.py --analyse results/gku.csv
Run from the repository root.
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "src"), sys.path.insert(0, ".")

METHODS = {   # label: (model, SimConfig kwargs, takes a seed?)
    "K": ("greedy_uncoordinated", dict(), True),
    "Uc": ("frontier_greedy", dict(known_map=False, reveal_obstacles_at_start=True), True),
    "U": ("frontier_greedy", dict(known_map=False), True),
    "Ucs": ("frontier_greedy_shared", dict(known_map=False, reveal_obstacles_at_start=True), True),
    "Us": ("frontier_greedy_shared", dict(known_map=False), True),
}
METRICS = ["t50", "t90", "t95", "t100", "path_len_total", "mean_visits", "frac_revisited",
           "collisions", "min_pair_dist", "max_tracking_error"]
LOWER_IS_BETTER = set(METRICS) - {"min_pair_dist"}
PAIRS = [("K", "U"), ("Uc", "U"), ("Ucs", "Us")]


def run_one(job):
    name, n, seed, label, size, max_time, safety = job
    import models  # noqa: F401
    from mascov.common.config import SimConfig
    from mascov.common.model import get_model
    from mascov.maps.benchmarks import make_world
    from mascov.sim.simulator import Simulator
    model, kw, seeded = METHODS[label]
    w = make_world(name, n_agents=n, size=size, seed=seed)
    t0 = time.perf_counter()
    r = Simulator(w, get_model(model, **(dict(seed=seed) if seeded else {})),
                  SimConfig(max_time=max_time, safety=safety, **kw)).run()
    m = r.metrics()
    row = dict(map=name, n=n, seed=seed, method=label, completed=int(r.completed),
               final_coverage=round(r.final_coverage, 4), wall_s=round(time.perf_counter() - t0, 1))
    for k in METRICS:
        v = m.get(k)
        row[k] = "" if v is None else round(float(v), 4)
    return row


def _paired(rows, a, b, metric):
    """Per (map, n, seed) paired values of metric for method b (treatment) and a (reference)."""
    idx = {(r["map"], r["n"], r["seed"], r["method"]): r for r in rows}
    xs, ys = [], []
    for (mp, n, sd, m), r in idx.items():
        if m != a:
            continue
        o = idx.get((mp, n, sd, b))
        if o is None or r[metric] in ("", None) or o[metric] in ("", None):
            continue
        xs.append(float(r[metric])), ys.append(float(o[metric]))
    return np.array(xs), np.array(ys)


def _stat(x, y, metric):
    """Mean paired relative difference of y vs x in % (positive = y is worse), SE, Wilcoxon p, verdict."""
    from scipy.stats import wilcoxon
    n = len(x)
    if n < 5:
        return None
    if metric in ("collisions",):                       # counts: compare means, not ratios
        d = y - x
        rel = float("nan") if x.mean() == 0 else 100 * d.mean() / x.mean()
        se = d.std(ddof=1) / np.sqrt(n)
        sign = 1.0
    else:
        ok = x > 1e-9
        d = (y[ok] - x[ok]) / x[ok] * 100
        rel, se = d.mean(), d.std(ddof=1) / np.sqrt(ok.sum())
        sign = 1.0 if metric in LOWER_IS_BETTER else -1.0
        n = int(ok.sum())
        d_raw = (y[ok] - x[ok])
    d_raw = y - x if metric == "collisions" else d_raw
    try:
        p = 1.0 if np.allclose(d_raw, 0) else float(wilcoxon(d_raw).pvalue)
    except ValueError:
        p = 1.0
    sig = (p < 0.01 and np.isfinite(rel) and abs(rel) > 2 * se and abs(rel) >= 10.0)
    worse = sign * rel > 0
    return dict(n=n, mean_x=float(x.mean()), mean_y=float(y.mean()), rel=float(rel), se=float(se), p=p,
                sig=bool(sig), direction="" if not sig else ("worse" if worse else "better"))


def analyse(rows, pairs=PAIRS):
    methods = {r["method"] for r in rows}
    done = {m: [r for r in rows if r["method"] == m] for m in methods}
    print("\nCompletion (true coverage of all coverable targets):")
    for m in sorted(done):
        rs = done[m]
        print(f"  {m:4s} {sum(int(r['completed']) for r in rs)}/{len(rs)} runs complete,"
              f" mean final coverage {np.mean([float(r['final_coverage']) for r in rs]):.4f}")
    for a, b in pairs:
        if a not in methods or b not in methods:
            continue
        print(f"\n=== {a} (reference) vs {b}: paired over map x N x seed; +% = {b} is worse for 'lower is better' metrics ===")
        print(f"{'metric':20s} {'n':>4s} {'mean '+a:>10s} {'mean '+b:>10s} {'diff %':>9s} {'+-SE':>6s} {'Wilcoxon p':>11s}  verdict")
        for metric in METRICS:
            x, y = _paired(rows, a, b, metric)
            s = _stat(x, y, metric) if len(x) else None
            if s is None:
                continue
            print(f"{metric:20s} {s['n']:4d} {s['mean_x']:10.2f} {s['mean_y']:10.2f} {s['rel']:+9.1f} {s['se']:6.1f} "
                  f"{s['p']:11.4f}  {'SIGNIFICANT ' + s['direction'] if s['sig'] else '-'}")
        # breakdown of the headline metric by map and by N
        for key in ("map", "n"):
            vals = sorted({r[key] for r in rows}, key=str)
            cells = []
            for v in vals:
                sub = [r for r in rows if r[key] == v]
                x, y = _paired(sub, a, b, "t100")
                if len(x):
                    cells.append(f"{key}={v}: {100 * np.mean((y - x) / x):+.0f}% (n={len(x)})")
            print(f"  t100 diff by {key}: " + "; ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=["empty", "sparse", "corridor", "rooms_and_door", "cluttered"])
    ap.add_argument("--n", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--size", type=float, default=12.0)
    ap.add_argument("--methods", nargs="+", default=["K", "Uc", "U"], choices=list(METHODS))
    ap.add_argument("--shared", action="store_true", help="also run Ucs / Us (belief merging)")
    ap.add_argument("--safety", default="none", choices=["none", "bvc"])
    ap.add_argument("--max-time", type=float, default=1500.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyse", default=None, help="skip running; analyse an existing CSV")
    a = ap.parse_args()

    if a.analyse:
        rows = list(csv.DictReader(open(a.analyse)))
        for r in rows:
            r["n"], r["seed"] = int(r["n"]), int(r["seed"])
        analyse(rows)
        return
    methods = list(dict.fromkeys(a.methods + (["Ucs", "Us"] if a.shared else [])))
    jobs = [(m, n, s, lab, a.size, a.max_time, a.safety)
            for m in a.maps for n in a.n for s in a.seeds for lab in methods]
    print(f"{len(jobs)} runs", flush=True)
    rows = []
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for i, r in enumerate(ex.map(run_one, jobs), 1):
                rows.append(r)
                print(f"[{i}/{len(jobs)}] {r['map']} N={r['n']} s={r['seed']} {r['method']}: t100={r['t100']}", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r = run_one(j)
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['map']} N={r['n']} s={r['seed']} {r['method']}: t100={r['t100']}", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(), wr.writerows(rows)
        print("saved", a.out)
    analyse(rows)


if __name__ == "__main__":
    main()