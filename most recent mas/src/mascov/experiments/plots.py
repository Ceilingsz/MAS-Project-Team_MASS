"""Standard figures for the experiments (matplotlib, light theme, one colour per method label).

Colours follow the data-viz palette: categorical slots in a fixed order, assigned to labels in first-seen order
across the whole experiment (so a method keeps its colour in every figure), thin lines, hairline grid, a legend
whenever there are two or more series, and direct labels for flat reference lines. References (floor / ceiling
methods) are neutral dashed lines, never a categorical colour.

Kinds (see ``PlotSpec``):
  sweep     y (mean with 95% interval) against a lever column x, one line per label; labels whose lever value is None
            ran once and are drawn as flat lines
  rate      like sweep, but y is a boolean metric (e.g. safe_complete) shown as a share with Wilson intervals
  curves    mean true coverage against time per label (x, y ignored; options: horizon)
  paired    paired relative difference of ``y`` between pairs of labels (options: pairs=[(a, b), ...])
  tradeoff  mean of metric x against mean of metric y per label (e.g. time against collisions), with intervals
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mascov.experiments import analysis as an  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DASHES = [(0, (5, 3)), (0, (1.5, 2.5)), (0, (6, 2, 1.5, 2)), (0, (3, 1.5))]
NICE = {"t50": "time to 50% coverage [s]", "t90": "time to 90% coverage [s]", "t95": "time to 95% coverage [s]",
        "t100": "time to full coverage [s]", "coverage_time": "time to full coverage [s]",
        "mean_visits": "visits per target", "collisions": "collision episodes", "pair_events": "drone-drone collisions",
        "msgs_per_drone_per_s": "messages per drone per second", "solve_ms": "model time per decision [ms]",
        "path_len_mean": "path length per drone [m]", "safe_complete": "safe-complete rate",
        "min_pair_dist": "minimum drone separation [m]", "final_coverage": "final coverage",
        "world_r_comm": "communication radius [m]", "n_agents": "number of drones"}


def nice(name: str) -> str:
    return NICE.get(name, name.replace("_", " "))


class Colors:
    """Stable label -> colour map: methods take categorical slots in first-seen order, references stay neutral."""

    def __init__(self, rows):
        self.method, self.reference = [], []
        for r in rows:
            lst = self.reference if r.get("role") == "reference" else self.method
            if r["label"] not in lst:
                lst.append(r["label"])

    def style(self, label: str) -> dict:
        if label in self.reference:
            return dict(color=MUTED, ls=DASHES[self.reference.index(label) % len(DASHES)], lw=1.4, marker=None)
        k = self.method.index(label) if label in self.method else len(self.method)
        return dict(color=SERIES[k] if k < len(SERIES) else MUTED, ls="-", lw=2.0, marker="o")


def _axes(figsize=(7.4, 4.4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor(SURFACE), ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, title, xlabel, ylabel, out, legend=True):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=SECONDARY, fontsize=9.5), ax.set_ylabel(ylabel, color=SECONDARY, fontsize=9.5)
    handles, labels = ax.get_legend_handles_labels()
    if legend and len(labels) >= 2:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY, loc="best")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def _xpos(values):
    """Positions for lever values: numbers as they are (inf placed past the largest), anything else by category."""
    vals = sorted({v for v in values if v is not None}, key=lambda v: (not isinstance(v, (int, float)) or isinstance(v, bool), v if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)))
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
    if not numeric:
        return {v: i for i, v in enumerate(vals)}, {i: str(v) for i, v in enumerate(vals)}
    finite = [v for v in vals if math.isfinite(v)]
    top = max(finite) if finite else 1.0
    step = (max(finite) - min(finite)) / max(1, len(finite) - 1) if len(finite) > 1 else 1.0
    pos = {v: (v if math.isfinite(v) else top + step) for v in vals}
    return pos, {pos[v]: ("inf" if not math.isfinite(v) else f"{v:g}") for v in vals}


def _sweep(rows, spec, colors, out, rate=False):
    fig, ax = _axes()
    pos, ticks = _xpos([r.get(spec.x) for r in rows])
    labels = list(dict.fromkeys(r["label"] for r in rows))
    for lab in labels:
        rs = [r for r in rows if r["label"] == lab]
        st = colors.style(lab)
        flat = [r for r in rs if r.get(spec.x) is None]
        if flat or not any(r.get(spec.x) is not None for r in rs):                 # ran once: a flat reference line
            m = (an.rate_stats(rs, [])[0]["rate"] if rate else an.group_stats(rs, [], spec.y)[0]["mean"])
            xs = sorted(pos.values())
            ax.plot([xs[0], xs[-1]], [m, m], label=lab, color=st["color"], ls=st["ls"] if st["ls"] != "-" else (0, (5, 3)),
                    lw=st["lw"])
            continue
        g = an.rate_stats(rs, [spec.x], spec.y) if rate else an.group_stats(rs, [spec.x], spec.y)
        g = sorted((e for e in g if e[spec.x] is not None), key=lambda e: pos[e[spec.x]])
        xs = [pos[e[spec.x]] for e in g]
        if rate:
            ys, lo, hi = [e["rate"] for e in g], [e["lo"] for e in g], [e["hi"] for e in g]
        else:
            ys = [e["mean"] for e in g]
            half = [0 if math.isnan(e["ci"]) else e["ci"] for e in g]
            lo, hi = [y - h for y, h in zip(ys, half)], [y + h for y, h in zip(ys, half)]
        ax.plot(xs, ys, label=lab, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.0)
        ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.14, lw=0)
    ax.set_xticks(list(ticks)), ax.set_xticklabels(list(ticks.values()))
    if rate:
        ax.set_ylim(-0.02, 1.04)
    else:
        ax.set_ylim(bottom=0)
    return _finish(fig, ax, spec.title, nice(spec.x), nice(spec.y), out)


def _curves(rows, curves, spec, colors, out):
    fig, ax = _axes()
    col = spec.options.get("column", "coverage")
    horizon = spec.options.get("horizon") or max(max(curves[r["job_id"]]["t"]) for r in rows if r["job_id"] in curves)
    grid = np.linspace(0, horizon, 300)
    for lab in dict.fromkeys(r["label"] for r in rows):
        ys = []
        for r in rows:
            c = curves.get(r["job_id"])
            if r["label"] == lab and c:
                ys.append(np.interp(grid, c["t"], c[col]))               # holds the last value after the run ends
        if not ys:
            continue
        ys = np.array(ys)
        st = colors.style(lab)
        ax.plot(grid, ys.mean(0), label=f"{lab} (n={len(ys)})", color=st["color"], ls=st["ls"], lw=st["lw"])
        if len(ys) > 1 and not (lab in colors.reference):
            ax.fill_between(grid, np.percentile(ys, 10, axis=0), np.percentile(ys, 90, axis=0), color=st["color"],
                            alpha=0.12, lw=0)
    ax.set_ylim(0, 1.02)
    what = "true coverage" if col == "coverage" else "fraction known to at least one drone"
    return _finish(fig, ax, spec.title, "time [s]", f"{what} (mean, 10-90% band)", out)


def _paired(rows, spec, colors, out):
    pairs = spec.options["pairs"]
    match = tuple(spec.options.get("match", ("map", "n_agents", "seed")))
    fig, ax = _axes(figsize=(6.4, 0.9 + 0.7 * len(pairs)))
    ax.grid(axis="y", visible=False), ax.grid(axis="x", color=GRID, lw=0.8)
    res = [an.paired(rows, a, b, spec.y, match=match, label_col=spec.options.get("label_col", "label"))
           for a, b in pairs]
    ys = np.arange(len(res))[::-1]
    for k, (y, r) in enumerate(zip(ys, res)):
        col = SERIES[k % len(SERIES)]
        half = 0 if math.isnan(r["rel_ci"]) else r["rel_ci"] * 100
        ax.barh(y, r["rel"] * 100, height=0.5, color=col, alpha=0.9)
        ax.errorbar(r["rel"] * 100, y, xerr=half, color=INK, lw=1, capsize=3)
        ax.text(0, y + 0.36, f"{r['a']} vs {r['b']}  (n={r['n']})", fontsize=8.5, color=SECONDARY, va="bottom")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks([])
    return _finish(fig, ax, spec.title, f"{nice(spec.y)}: relative difference [%]", "", out, legend=False)


def _tradeoff(rows, spec, colors, out):
    fig, ax = _axes()
    for lab in dict.fromkeys(r["label"] for r in rows):
        rs = [r for r in rows if r["label"] == lab]
        gx, gy = an.group_stats(rs, [], spec.x)[0], an.group_stats(rs, [], spec.y)[0]
        st = colors.style(lab)
        ax.errorbar(gx["mean"], gy["mean"], xerr=0 if math.isnan(gx["ci"]) else gx["ci"],
                    yerr=0 if math.isnan(gy["ci"]) else gy["ci"], fmt="o", ms=7, color=st["color"], ecolor=st["color"],
                    elinewidth=1, capsize=2.5, label=lab, markeredgecolor=SURFACE)
    return _finish(fig, ax, spec.title, nice(spec.x), nice(spec.y), out)


def make_figures(spec, rows, curves, out_dir) -> list[str]:
    """Draw one PlotSpec: one file per value of ``spec.per`` (or one file). Returns the written paths."""
    rows = an.ok_rows(rows, spec.where)
    if not rows:
        return []
    colors = Colors(rows)
    parts = sorted({r.get(spec.per) for r in rows}, key=str) if spec.per else [None]
    written = []
    for part in parts:
        rs = [r for r in rows if spec.per is None or r.get(spec.per) == part]
        s = type(spec)(**{**spec.__dict__, "title": spec.title + (f" ({part})" if part is not None else "")})
        stem = f"{spec.file}" + (f"_{part}" if part is not None else "")
        out = Path(out_dir) / f"{stem}.png"
        if spec.kind == "sweep":
            written.append(_sweep(rs, s, colors, out))
        elif spec.kind == "rate":
            written.append(_sweep(rs, s, colors, out, rate=True))
        elif spec.kind == "curves":
            written.append(_curves(rs, curves, s, colors, out))
        elif spec.kind == "paired":
            written.append(_paired(rs, s, colors, out))
        elif spec.kind == "tradeoff":
            written.append(_tradeoff(rs, s, colors, out))
        else:
            raise ValueError(f"unknown plot kind '{spec.kind}'")
    return written
