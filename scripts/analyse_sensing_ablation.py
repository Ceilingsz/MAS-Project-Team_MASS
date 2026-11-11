"""Analyse scripts/ablate_sensing_radius.py output: the cost of not knowing the map, per sensing radius.

For every (algorithm, map, seed, R_s) the unknown-map run is paired with its known-map twin and the ratio
time_unknown / time_known is taken (t95 and t100). Ratios are averaged as geometric means (mean of log ratios), with a
95% bootstrap interval. Identical pairs (a deterministic algorithm on a fixed map gives the same run for every seed) are
counted once. Also writes a summary CSV and one figure.

    python scripts/analyse_sensing_ablation.py results/ablate_sensing/runs.csv
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ablate_sensing/runs.csv")
rows = [r for r in csv.DictReader(open(path, newline=""))]
ok = [r for r in rows if r["status"] == "ok"]
print(f"{len(rows)} rows, {len(ok)} ok, {len(rows) - len(ok)} errors")
rng = np.random.default_rng(0)


def f(x):
    return float(x) if x not in ("", "None", None) else math.nan


runs = {(r["label"], r["map"], int(r["seed"]), float(r["r_sense"]), r["prior"]): r for r in ok}
labels = list(dict.fromkeys(r["label"] for r in ok))
radii = sorted({k[3] for k in runs})
maps = list(dict.fromkeys(r["map"] for r in ok))

# completion and safety
incomplete = [(k, r["final_coverage"]) for k, r in runs.items() if r["completed"] != "True"]
print(f"incomplete runs: {len(incomplete)}", incomplete[:10])
print("obstacle episodes:", sum(int(r["obstacle_events"]) for r in ok))
cov = defaultdict(set)
for k, r in runs.items():
    cov[(k[1], k[3])].add(r["coverable_frac"])
print("coverable fraction by (map, R_s):", {f"{m} R={rs}": sorted(v) for (m, rs), v in sorted(cov.items())})


def pairs(metric, label=None, rs=None, mp=None):
    """Distinct log ratios unknown/known."""
    seen = {}
    for (lab, m, s, r, prior), run in runs.items():
        if prior != "unknown" or (label and lab != label) or (rs is not None and r != rs) or (mp and m != mp):
            continue
        kn = runs.get((lab, m, s, r, "known"))
        if kn is None:
            continue
        a, b = f(run[metric]), f(kn[metric])
        if not (math.isfinite(a) and math.isfinite(b)) or b <= 0:
            continue
        seen[(lab, m, r, a, b)] = math.log(a / b)       # identical repeated pairs collapse to one
    return np.array(list(seen.values()))


def summary(x):
    if len(x) == 0:
        return math.nan, math.nan, math.nan, 0
    boots = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return math.exp(x.mean()), math.exp(np.percentile(boots, 2.5)), math.exp(np.percentile(boots, 97.5)), len(x)


out_rows = []
for metric in ("t95", "t100"):
    print(f"\n== {metric}: unknown / known (geometric mean, 95% bootstrap CI, n distinct pairs)")
    print(f"{'':16s}" + "".join(f"R={rs:<22}" for rs in radii))
    for lab in labels + ["ALL"]:
        line = f"{lab:16s}"
        for rs in radii:
            g, lo, hi, n = summary(pairs(metric, None if lab == "ALL" else lab, rs))
            line += f"{g:5.2f} [{lo:4.2f},{hi:4.2f}] n={n:<3d}  "
            out_rows.append(dict(metric=metric, label=lab, r_sense=rs, ratio=round(g, 4), ci_lo=round(lo, 4),
                                 ci_hi=round(hi, 4), n_pairs=n))
        print(line)
    print("  per map, all algorithms:")
    for mp in maps:
        line = f"  {mp:14s}"
        for rs in radii:
            g, lo, hi, n = summary(pairs(metric, None, rs, mp))
            line += f"{g:5.2f} [{lo:4.2f},{hi:4.2f}]      "
        print(line)

# absolute times, to see what shrinking R_s costs regardless of the map
print("\n== mean t95 [s] by R_s, known | unknown (all algorithms and maps)")
for rs in radii:
    k = [f(r["t95"]) for (lab, m, s, r_, p), r in runs.items() if r_ == rs and p == "known"]
    u = [f(r["t95"]) for (lab, m, s, r_, p), r in runs.items() if r_ == rs and p == "unknown"]
    print(f"R={rs}: {np.nanmean(k):6.1f} | {np.nanmean(u):6.1f}")

summ = path.parent / "summary.csv"
with open(summ, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)
print("\nwritten", summ)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, metric in zip(axes, ("t95", "t100")):
        for lab in labels + ["ALL"]:
            pts = [r for r in out_rows if r["metric"] == metric and r["label"] == lab]
            xs = [p["r_sense"] for p in pts]
            ys = [p["ratio"] for p in pts]
            lo = [p["ratio"] - p["ci_lo"] for p in pts]
            hi = [p["ci_hi"] - p["ratio"] for p in pts]
            if lab == "ALL":
                ax.errorbar(xs, ys, yerr=[lo, hi], color="black", lw=2.4, marker="o", capsize=3, label="all algorithms", zorder=5)
            else:
                ax.plot(xs, ys, marker="o", lw=1.2, alpha=0.75, label=lab)
        ax.axhline(1.0, color="grey", lw=1, ls="--")
        ax.set_xlabel("sensing radius R_s [m]")
        ax.set_title(f"time to {'95%' if metric == 't95' else '100%'} coverage")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("unknown map / known map (geometric mean)")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Cost of not knowing the map vs sensing radius (4 drones; sparse, rooms_and_door, cluttered)")
    fig.tight_layout()
    fig_path = path.parent / "map_cost_vs_sensing_radius.png"
    fig.savefig(fig_path, dpi=150)
    print("written", fig_path)
except Exception as e:  # pragma: no cover
    print("no figure:", e)
