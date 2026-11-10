"""Coverage against time for every case of an experiment: one small plot per (team size, communication radius).

  python scripts/plot_coverage_curves.py ABL ABLP --results results      # figures -> results/<exp>/curves/

For each map (and once pooled over the maps) it writes a grid: rows are team sizes, columns are communication radii,
and every plot shows the MEAN true coverage over the seeds, one line per model. Colour is the algorithm (blue geodesic
partition, orange greedy); the line style says what it shares (solid: beliefs shared, dashed: nothing shared; in ABLP:
solid: map unknown, dashed: map known). A run that ends keeps its final coverage. Each row is zoomed to the time by which every mean has reached 99.5%.
The dotted horizontal line is 95% coverage, the level the time-to-95% results are read at.
Run from the repository root; needs the runs.jsonl and curves.jsonl that run_experiments.py wrote.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src"), sys.path.insert(0, ".")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mascov.experiments import analysis as an  # noqa: E402

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DT = 0.5
ZOOM = 0.995          # each row's time axis ends shortly after every mean curve has reached this coverage


def _r(v):
    if v is None:
        return None
    return math.inf if str(v).lower() in ("inf", "infinity") else float(v)


def _style(label: str, prior: str | None, exp: str):
    """(colour, linestyle, legend text) of a model, see the module docstring."""
    colour = BLUE if label.startswith("Geodesic") else ORANGE
    if exp == "ABLP":
        known = prior == "known"
        base = "geodesic partition" if colour == BLUE else "greedy explorer"
        return colour, ("--" if known else "-"), f"{base}, map {'known' if known else 'unknown'}"
    shared = "shared" in label
    base = "geodesic partition" if colour == BLUE else "greedy explorer"
    what = "shares beliefs" if shared else ("shares positions" if colour == BLUE else "no messages")
    return colour, ("-" if shared else (0, (4, 2.5))), f"{base}, {what}"


def _mean_curve(curves: list[dict], horizon: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(0.0, horizon + DT / 2, DT)
    ys = []
    for c in curves:
        t, y = np.asarray(c["t"], float), np.asarray(c["coverage"], float)
        ys.append(np.interp(grid, t, y, right=y[-1]))        # keeps the final coverage after the run ended
    return grid, np.mean(ys, axis=0)


def _grid(rows, curves, exp, title, out, pooled_note=""):
    ns = sorted({r["n_agents"] for r in rows})
    rs = sorted({_r(r["world_r_comm"]) for r in rows if r.get("world_r_comm") is not None})
    series = {}                                              # (label, prior) -> list of rows
    for r in rows:
        series.setdefault((r["label"], r.get("map_prior")), []).append(r)
    fig, axes = plt.subplots(len(ns), len(rs), figsize=(2.9 * len(rs) + 0.8, 2.15 * len(ns) + 1.3), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    handles = {}
    for i, n in enumerate(ns):
        full = max((c["t"][-1] for r in rows if r["n_agents"] == n and (c := curves.get(r["job_id"]))), default=1.0)
        means, reach = {}, 0.0
        for j, rc in enumerate(rs):
            for (label, prior), rr in series.items():
                sel = [r for r in rr if r["n_agents"] == n and (_r(r.get("world_r_comm")) == rc or r.get("world_r_comm") is None)]
                cs = [curves[r["job_id"]] for r in sel if r["job_id"] in curves]
                if cs:
                    x, y = _mean_curve(cs, full)
                    means[(j, label, prior)] = (x, y)
                    reach = max(reach, float(x[np.argmax(y >= ZOOM)]) if (y >= ZOOM).any() else full)
        horizon = min(full, reach * 1.08)                      # the whole row is zoomed to when every mean has (nearly) finished
        for j, rc in enumerate(rs):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            for (label, prior) in series:
                if (j, label, prior) not in means:
                    continue
                colour, ls, text = _style(label, prior, exp)
                x, y = means[(j, label, prior)]
                (h,) = ax.plot(x, y, color=colour, ls=ls, lw=1.7)
                handles.setdefault(text, h)
            ax.axhline(0.95, color=GRID, lw=1, ls=":", zorder=0)
            ax.set_xlim(0, horizon), ax.set_ylim(0, 1.02)
            ax.grid(color=GRID, lw=0.6), ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"):
                ax.spines[s_].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=8)
            if i == 0:
                ax.set_title("radius " + ("unlimited" if math.isinf(rc) else f"{rc:g} m"), fontsize=9.5, color=SECONDARY)
            if j == 0:
                ax.set_ylabel(f"{n} drones\ncoverage", fontsize=9, color=SECONDARY)
            if i == len(ns) - 1:
                ax.set_xlabel("time [s]", fontsize=9, color=SECONDARY)
    fig.suptitle(title + (f"  ({pooled_note})" if pooled_note else ""), x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=SECONDARY,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.075 + 0.02 * (len(handles) > 2), 1, 0.96))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    for exp in a.exps:
        d = Path(a.results) / exp
        rows = an.ok_rows(an.load_rows(d))
        curves = an.load_curves(d)
        maps = sorted({r["map"] for r in rows})
        out = d / "curves"
        title = {"ABL": "Coverage over time", "ABLP": "Coverage over time, map known and unknown"}.get(exp, f"{exp}: coverage over time")
        for m in maps:
            _grid([r for r in rows if r["map"] == m], curves, exp, f"{title}: {m}", out / f"coverage_{m}.png")
        _grid(rows, curves, exp, title, out / "coverage_all_maps.png", pooled_note=f"mean over {len(maps)} maps and seeds")
        print(f"{exp}: {len(rows)} runs -> {len(maps) + 1} figures in {out}")


if __name__ == "__main__":
    main()
