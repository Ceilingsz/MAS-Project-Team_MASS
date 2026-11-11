"""The results figures of the slide deck, 12 m (small) next to 16 m (large), with MAPPO, plus map-type tables.

Small maps (12 m): the CSV / JSON files of the original scripts in results/small/ (collisions_n, mix_match,
comm_radius, ablate_sensing, coverage_curves), plus a ``mappo.*`` file beside each one when MAPPO has been run through
the same script with ``--only MAPPO``.
Large maps (16 m): the experiment runner's results/<EXP>/runs.jsonl and curves.jsonl for SCALE (16 m maps only),
L16_A, L16_B, L16_MARL, PRIOR_16 and PRIOR_16_MARL.

    python scripts/plot_deck_results.py --out docs/deck

Writes one PNG per slide (q1, sensing, q2, q2comm, q3, mixmatch, costs, coverage), the map-type tables
(map_types.csv / map_types.json: mean time to 95% per algorithm and map type) and numbers.json (the numbers quoted on
the slides). Missing inputs are skipped with a note, so the script can be run while experiments are still going.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "results" / "small"
RES = ROOT / "results"

# deck palette
BG, PANEL, INK, MUTED, GRIDC = "#f7f6f1", "#fdfcf9", "#172033", "#6b7385", "#e2ded4"
COL = {"Greedy": "#d9661f", "Greedy + goals": "#b8862b", "Geodesic": "#2f6db5", "CBBA": "#7650c9",
       "D-MPC": "#178a5e", "MAPPO": "#c2185b"}
GOAL_OF = {"Greedy": "Greedy", "Greedy + goals": "Greedy + goals", "Geodesic": "Geodesic", "CBBA": "CBBA",
           "D-MPC": "D-MPC", "Greedy + D-MPC": "Greedy", "Geodesic + D-MPC": "Geodesic", "CBBA + D-MPC": "CBBA",
           "MAPPO": "MAPPO", "MAPPO + D-MPC": "MAPPO"}
FIVE = ["Greedy", "Greedy + goals", "Geodesic", "CBBA", "D-MPC", "MAPPO"]          # the deck's algorithms + MAPPO
LABEL_FIX = {"CBBA (shared beliefs)": "CBBA", "Geodesic partition (shared beliefs)": "Geodesic",
             "Greedy + Navigator": "Greedy", "Geodesic + Navigator": "Geodesic", "CBBA + Navigator": "CBBA",
             "MAPPO + Navigator": "MAPPO"}
MAP_NAME = {"sparse": "Sparse", "rooms_and_door": "Rooms + door", "cluttered": "Cluttered", "dense_16": "Dense",
            "random_rooms_16": "Random rooms", "maze_16": "Maze", "warehouse_16": "Warehouse"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 13, "axes.edgecolor": GRIDC, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titleweight": "bold",
                     "axes.titlesize": 15, "axes.spines.top": False, "axes.spines.right": False})


# ============================================================================================ loading
SIZES = {"12 m": dict(runs=("S12", "S12_MARL"), prior=("PRIOR_12", "PRIOR_12_MARL")),
         "16 m": dict(runs=("SCALE", "L16_A", "L16_B", "L16_C", "L16_MARL"), prior=("PRIOR_16", "PRIOR_16_MARL"))}
TEAMS = (4, 8, 16)
TYPE_OF = {"sparse": "Sparse", "cluttered": "Cluttered", "dense": "Dense", "random_rooms": "Random rooms",
           "maze": "Maze", "warehouse": "Warehouse"}
TYPE_ORDER = list(TYPE_OF.values())


def map_type(m: str) -> str:
    base = m[:-3] if m[-3:] in ("_12", "_16", "_20", "_24") else m
    return TYPE_OF.get(base, base)


def _jsonl(exp: str, name: str = "runs.jsonl") -> list:
    p = RES / exp / name
    return [json.loads(line) for line in p.open()] if p.exists() else []


def load_scale(size: str) -> dict:
    """Runs (six map types, 4/8/16 drones), map-prior runs and coverage curves of one map size."""
    spec = SIZES[size]
    big = size == "16 m"
    runs = pd.DataFrame([r for e in spec["runs"] for r in _jsonl(e)])
    if not runs.empty:
        keep = runs["map"].map(lambda m: (m.endswith("_16") if big else m[-3:] not in ("_16", "_20", "_24"))
                               and map_type(m) in TYPE_ORDER) & runs["n_agents"].isin(TEAMS)
        runs = runs[keep & (runs.get("status", "ok") == "ok")].copy()
        runs["label"] = runs["label"].replace(LABEL_FIX)
        runs = runs[runs["label"].isin(GOAL_OF)]
        runs["type"] = runs["map"].map(map_type)
        runs = runs.drop_duplicates(["label", "map", "n_agents", "seed"], keep="last")
    prior = pd.DataFrame([r for e in spec["prior"] for r in _jsonl(e)])
    if not prior.empty:
        prior["label"] = prior["label"].replace(LABEL_FIX)
    curves = []
    if not runs.empty:
        lab = dict(zip(runs["job_id"], zip(runs["label"], runs["map"], runs["n_agents"])))
        for e in spec["runs"]:
            for c in _jsonl(e, "curves.jsonl"):
                if c["job_id"] in lab:
                    l, m, n = lab[c["job_id"]]
                    curves.append(dict(label=l, map=m, n=n, t=c["t"], cov=c["coverage"]))
    return dict(runs=runs, prior=prior, curves=curves)


def load_small_scripts() -> dict:
    """The two 12 m-only slides (radio range, sensing radius) keep the original scripts' data, plus MAPPO."""
    def csv(path):
        frames = [pd.read_csv(p) for p in (path, path.with_name("mappo.csv")) if p.exists()]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df.empty:
            df["label"] = df["label"].replace(LABEL_FIX)
        return df
    return dict(comm=csv(SMALL / "comm_radius" / "runs.csv"), sense=csv(SMALL / "ablate_sensing" / "runs.csv"))


# ============================================================================================ helpers
def _fig(w_px, h_px):
    return plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100, facecolor=PANEL)


def _style(ax, title=None):
    ax.set_facecolor(PANEL)
    ax.grid(axis="y", color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=INK, pad=10)


def _save(fig, out: Path, name: str, made: list):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    made.append(name)


def _goal_color(label):
    return COL[GOAL_OF.get(label, label)]


HEAD = "six map types · 4/8/16 drones"


# ============================================================================================ figures
def fig_q1(D, out, made, nums):
    """Map known vs unknown, time to 95%: same three map types and 8 drones at both sizes."""
    fig = _fig(1664, 560)
    for k, size in enumerate(D):
        ax = fig.add_subplot(1, 2, k + 1)
        df = D[size]["prior"]
        _style(ax, f"{size} maps (dense, random rooms, maze) · 8 drones")
        if df.empty:
            continue
        labs = [l for l in FIVE if l in set(df["label"])]
        x = np.arange(len(labs))
        for j, prior in enumerate(("known", "unknown")):
            vals = [df[(df["label"] == l) & (df["map_prior"] == prior)]["t95"].mean() for l in labs]
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=[_goal_color(l) for l in labs],
                   alpha=0.45 if prior == "known" else 1.0, label=f"map {prior}")
            for xi, v in zip(x, vals):
                ax.text(xi + (j - 0.5) * 0.38, v + 0.8, f"{v:.0f}", ha="center", fontsize=11, color=INK)
            nums[f"q1_{size}_{prior}"] = dict(zip(labs, [round(v, 1) for v in vals]))
        ax.set_xticks(x, labs)
        ax.set_ylabel("time to 95% coverage [s]")
        if k == 0:
            ax.legend(frameon=False, loc="upper right")
    _save(fig, out, "q1", made)


def fig_sensing(P, out, made, nums):
    df = P["sense"]
    if df.empty:
        return
    df = df[df["status"] == "ok"]
    labs = [l for l in FIVE if l in set(df["label"])]
    radii = sorted(r for r in set(df["r_sense"]) if all(((df["label"] == l) & (df["r_sense"] == r)).any() for l in labs))
    fig = _fig(1280, 527)
    ax = fig.add_subplot(111)
    _style(ax, "Unknown map: how much slower than with the map, by sensing radius (12 m)")
    w = 0.8 / len(radii)
    for j, r in enumerate(radii):
        vals = []
        for l in labs:
            a = df[(df["label"] == l) & (df["r_sense"] == r)]
            kn = a[a["prior"] == "known"].groupby(["map", "seed"])["t95"].mean()
            un = a[a["prior"] == "unknown"].groupby(["map", "seed"])["t95"].mean()
            both = kn.index.intersection(un.index)
            vals.append(100 * (un[both] / kn[both] - 1).mean() if len(both) else np.nan)
        ax.bar(np.arange(len(labs)) + (j - (len(radii) - 1) / 2) * w, vals, w * 0.9,
               color=plt.cm.Blues(0.35 + 0.6 * j / max(1, len(radii) - 1)), label=f"R_s = {r:g} m")
        nums[f"sensing_{r:g}"] = dict(zip(labs, [None if np.isnan(v) else round(v, 1) for v in vals]))
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(np.arange(len(labs)), labs)
    ax.set_ylabel("unknown vs known map [%]")
    ax.legend(frameon=False, ncol=len(radii), loc="upper left")
    _save(fig, out, "sensing", made)


def fig_q2(D, out, made, nums):
    fig = _fig(1664, 600)
    for k, size in enumerate(D):
        ax = fig.add_subplot(1, 2, k + 1)
        df = D[size]["runs"]
        _style(ax, f"{size} maps · six map types")
        if df.empty:
            continue
        for l in FIVE:
            a = df[df["label"] == l]
            if a.empty:
                continue
            g = a.groupby("n_agents")["t95"].mean()
            ax.plot(g.index, g.values, "-o", color=COL[l], lw=2.6 if l == "MAPPO" else 1.8, ms=6, label=l)
            nums[f"q2_{size}_{l}"] = {int(n): round(v, 1) for n, v in g.items()}
        ax.set_xticks(TEAMS)
        ax.set_xlabel("drones")
        ax.set_ylabel("time to 95% coverage [s]")
        if k == 1:
            ax.legend(frameon=False, loc="upper right")
    _save(fig, out, "q2", made)


def fig_q2comm(P, out, made, nums):
    df = P["comm"]
    if df.empty:
        return
    df = df[df["status"] == "ok"].copy()
    df["r"] = df["r_comm"].astype(float)
    finite = sorted(r for r in set(df["r"]) if math.isfinite(r))
    pos = {r: i for i, r in enumerate(finite)}
    pos[math.inf] = len(finite)
    fig = _fig(960, 535)
    ax = fig.add_subplot(111)
    _style(ax, "rooms_and_door (12 m) · 6 drones")
    for l in FIVE:
        a = df[df["label"] == l]
        if a.empty:
            continue
        g = a.groupby("r")["t95"].mean()
        ax.plot([pos[r] for r in g.index], g.values, "-o", color=COL[l], lw=2.6 if l == "MAPPO" else 1.8, label=l)
        nums[f"comm_{l}"] = {str(r): round(v, 1) for r, v in g.items()}
    ax.set_xticks(list(pos.values()), [f"{r:g}" for r in finite] + ["∞"])
    ax.set_xlabel("radio range [m]")
    ax.set_ylabel("time to 95% coverage [s]")
    ax.legend(frameon=False, fontsize=11)
    _save(fig, out, "q2comm", made)


def _scatter_panels(D, out, made, nums, name, arrows: bool):
    fig = _fig(1664, 560)
    for k, size in enumerate(D):
        ax = fig.add_subplot(1, 2, k + 1)
        df = D[size]["runs"]
        _style(ax, f"{size} maps · {HEAD}")
        ax.grid(axis="x", color=GRIDC, lw=0.8)
        if df.empty:
            continue
        if arrows:
            for goal, dm in (("Greedy", "D-MPC"), ("Geodesic", "Geodesic + D-MPC"), ("CBBA", "CBBA + D-MPC"),
                             ("MAPPO", "MAPPO + D-MPC")):
                a, b = df[df["label"] == goal], df[df["label"] == dm]
                if a.empty or b.empty:
                    continue
                p0 = (a["t95"].mean(), a["pair_events"].mean())
                p1 = (b["t95"].mean(), b["pair_events"].mean())
                ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="->", color=COL[goal], lw=2))
                ax.scatter(*p0, s=120, color=COL[goal], zorder=3)
                ax.scatter(*p1, s=140, color=COL[goal], marker="s", edgecolor=INK, zorder=3)
                ax.annotate(goal, p0, xytext=(6, 6), textcoords="offset points", fontsize=12, color=COL[goal])
                nums[f"mix_{size}_{goal}"] = dict(nav_t95=round(p0[0], 1), nav_coll=round(p0[1], 2),
                                                  dmpc_t95=round(p1[0], 1), dmpc_coll=round(p1[1], 2))
        else:
            for l in [x for x in GOAL_OF if (df["label"] == x).any()]:
                a = df[df["label"] == l]
                x, y = a["t95"].mean(), a["pair_events"].mean()
                dm = "D-MPC" in l
                ax.scatter(x, y, s=150, color=_goal_color(l), marker="s" if dm else "o",
                           edgecolor=INK if dm else "none", zorder=3, label=l)
                nums[f"q3_{size}_{l}"] = dict(t95=round(x, 1), collisions=round(y, 2))
            if k == 1:
                ax.legend(frameon=False, fontsize=10.5, loc="upper right", ncol=2)
        ax.set_xlabel("time to 95% coverage [s]")
        ax.set_ylabel("collision episodes per run")
    note = ("circle: Navigator  →  square: D-MPC, same goal rule" if arrows else
            "circle: Navigator   square: D-MPC   colour: goal rule")
    fig.text(0.5, 0.005, note, ha="center", color=MUTED, fontsize=12)
    _save(fig, out, name, made)


def fig_q3(D, out, made, nums):
    _scatter_panels(D, out, made, nums, "q3", arrows=False)


def fig_mix(D, out, made, nums):
    _scatter_panels(D, out, made, nums, "mixmatch", arrows=True)


def fig_costs(D, out, made, nums):
    fig = _fig(1664, 576)
    ax1 = fig.add_subplot(1, 2, 1)
    _style(ax1, "Distance flown per drone (Navigator) · 4/8/16 drones")
    labs = ["Greedy", "Greedy + goals", "Geodesic", "CBBA", "MAPPO"]
    x = np.arange(len(labs))
    for j, size in enumerate(D):
        df = D[size]["runs"]
        if df.empty:
            continue
        vals = [df[df["label"] == l]["path_len_mean"].mean() if (df["label"] == l).any() else np.nan for l in labs]
        ax1.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=[COL[l] for l in labs], alpha=0.5 if j == 0 else 1.0,
                label=f"{size} maps")
        nums[f"path_{size}"] = dict(zip(labs, [None if np.isnan(v) else round(v, 1) for v in vals]))
    ax1.set_xticks(x, labs)
    ax1.set_ylabel("metres")
    ax1.legend(frameon=False)
    ax2 = fig.add_subplot(1, 2, 2)
    _style(ax2, "Collisions per run: 12 m (dashed) and 16 m (solid)")
    for size, ls in zip(D, ("--", "-")):
        df = D[size]["runs"]
        if df.empty:
            continue
        for l in ("Greedy", "Geodesic", "CBBA", "MAPPO", "Geodesic + D-MPC", "MAPPO + D-MPC"):
            a = df[df["label"] == l]
            if a.empty:
                continue
            g = a.groupby("n_agents")["pair_events"].mean()
            ax2.plot(g.index, g.values, ls, marker="s" if "D-MPC" in l else "o", color=_goal_color(l),
                     label=l if ls == "-" else None)
    ax2.set_xticks(TEAMS)
    ax2.set_xlabel("drones")
    ax2.set_ylabel("collision episodes per run")
    ax2.legend(frameon=False, fontsize=10, ncol=2)
    _save(fig, out, "costs", made)


def _mean_curve(curves, grid):
    ys = [np.interp(grid, c["t"], c["cov"], right=c["cov"][-1]) for c in curves]
    return np.mean(ys, axis=0) if ys else None


def fig_coverage(D, out, made, nums, n=8):
    fig = _fig(1664, 597)
    for k, (size, tmax) in enumerate((("12 m", 100), ("16 m", 160))):
        ax = fig.add_subplot(1, 2, k + 1)
        _style(ax, f"{size} maps · six map types · {n} drones")
        grid = np.linspace(0, tmax, 4 * tmax + 1)
        cs = [c for c in D[size]["curves"] if c["n"] == n]
        for l in ("Greedy", "Geodesic", "CBBA", "MAPPO", "D-MPC", "Geodesic + D-MPC", "CBBA + D-MPC",
                  "MAPPO + D-MPC"):
            m = _mean_curve([c for c in cs if c["label"] == l], grid)
            if m is None:
                continue
            ax.plot(grid, 100 * m, "-" if "D-MPC" in l else "--", color=_goal_color(l), lw=2, label=l)
            hit = np.flatnonzero(m >= 0.95)
            nums[f"cov_{size}_{l}_t95"] = None if not len(hit) else round(float(grid[hit[0]]), 1)
        ax.set_ylim(70, 100.5)
        ax.set_xlim(tmax * 0.15, tmax)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("area covered [%] (70-100%)")
        if k == 1:
            ax.legend(frameon=False, fontsize=10, ncol=2, loc="lower right")
    fig.text(0.5, 0.005, "dashed: Navigator   solid: D-MPC   colour: goal rule", ha="center", color=MUTED, fontsize=12)
    _save(fig, out, "coverage", made)


def _evals(log: Path) -> list[dict]:
    """Validation records of a training log (JSON iteration lines and the repr-printed iteration-0 line)."""
    import ast
    out = []
    if not log.exists():
        return out
    for line in log.open():
        if line.startswith("[marl] {"):
            r = json.loads(line[7:])
        elif line.startswith("[marl] iter ") and " eval {" in line:
            r = ast.literal_eval(line.split(" eval ", 1)[1])
        else:
            continue
        if "eval_ratio_vs_cbba" in r:
            out.append(r)
    return out


def fig_training(out, made, nums):
    """The whole training history of the shipped policy, then the current phase in detail.

    The policy is one chain: run 1 (12 m maps) up to its best checkpoint, iteration 30; run 2 warm-started from it on
    12-20 m maps; from run-2 iteration 20 on 12-16 m maps only. Each phase was scored on its own held-out validation
    set (12 m; 12-20 m; 12-16 m), so the three segments are drawn separately. Run 1 after iteration 30 was discarded
    (it got worse with 6-8 drones) and is drawn faded."""
    run1 = _evals(RES / "marl_run1" / "marl_train_stdout.log")
    run2a = _evals(RES / "marl_train2_crashed_stdout.log")[:1] + _evals(RES / "marl_train2_stdout.log")
    run2b = _evals(RES / "marl_train3_stdout.log")
    if not run2b:
        return
    fig = _fig(1664, 580)
    ax = fig.add_subplot(1, 2, 1)
    _style(ax, "Whole training history: coverage time relative to CBBA")
    kept = [r for r in run1 if r["iter"] <= 30]
    dead = [r for r in run1 if r["iter"] >= 30]
    ax.plot([r["iter"] for r in dead], [r["eval_ratio_vs_cbba"] for r in dead], "--o", color="#c9c4b8", ms=4,
            label="run 1 after its best (discarded)")
    segs = ((kept, 0, "#e991b4", "run 1 · 12 m maps"),
            ([r for r in run2a if r["iter"] <= 20], 30, "#c2185b", "run 2 · 12-20 m maps"),
            (run2b, 30, "#8c1043", "run 2 · 12-16 m maps"))
    for rows, off, col, lab in segs:
        if rows:
            ax.plot([off + r["iter"] for r in rows], [r["eval_ratio_vs_cbba"] for r in rows], "-o", color=col, lw=2.4,
                    ms=6, label=lab)
    ax.axhline(1.0, color=COL["CBBA"], ls="--", lw=1.5)
    ax.text(ax.get_xlim()[1], 1.01, "CBBA ", color=COL["CBBA"], fontsize=12, ha="right", va="bottom")
    ax.set_xlabel("training iterations of the policy")
    ax.set_ylabel("MAPPO time / CBBA time (lower is faster)")
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    ax2 = fig.add_subplot(1, 2, 2)
    _style(ax2, "Last phase (12-16 m maps), in detail")
    it = [30 + r["iter"] for r in run2b]
    for key, lab, col, lw in (("eval_ratio_vs_cbba", "all 18 missions", COL["MAPPO"], 2.6),
                              ("eval_ratio_12m", "12 m maps", "#e991b4", 1.6),
                              ("eval_ratio_16m", "16 m maps", "#8c1043", 1.6)):
        ax2.plot(it, [r.get(key) for r in run2b], "-o", color=col, lw=lw, ms=5, label=lab)
    best = min(run2b, key=lambda r: r["eval_ratio_vs_cbba"])
    ax2.scatter([30 + best["iter"]], [best["eval_ratio_vs_cbba"]], s=180, facecolor="none", edgecolor=INK, lw=1.5,
                zorder=4)
    ax2.annotate(f"saved: {best['eval_ratio_vs_cbba']:.2f}×", (30 + best["iter"], best["eval_ratio_vs_cbba"]),
                 xytext=(0, -26), textcoords="offset points", ha="center", fontsize=12)
    ax2.set_ylim(min(0.7, ax2.get_ylim()[0]), max(0.9, ax2.get_ylim()[1]))
    ax2.set_xlabel("training iterations of the policy")
    ax2.set_ylabel("MAPPO time / CBBA time")
    ax2.legend(frameon=False, loc="upper left", fontsize=11, ncol=3)
    for a in (ax, ax2):
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    nums["training"] = dict(untrained=round(run1[0]["eval_ratio_vs_cbba"], 3) if run1 else None,
                            best_iter=30 + best["iter"], best_ratio=round(best["eval_ratio_vs_cbba"], 3),
                            best_12m=best.get("eval_ratio_12m"), best_16m=best.get("eval_ratio_16m"),
                            last_iter=30 + run2b[-1]["iter"],
                            phase_range=[round(min(r["eval_ratio_vs_cbba"] for r in run2b), 3),
                                         round(max(r["eval_ratio_vs_cbba"] for r in run2b), 3)])
    _save(fig, out, "training", made)


# ============================================================================================ map-type tables
TABLE_ALGOS = ["Greedy", "Greedy + goals", "Geodesic", "CBBA", "D-MPC", "Geodesic + D-MPC", "CBBA + D-MPC", "MAPPO",
               "MAPPO + D-MPC"]


def _paired_vs(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float, float, int]:
    """Mean of t95(a) / t95(b) over runs matched on (map, n_agents, seed), with a 95% t-interval."""
    from scipy import stats
    key = ["map", "n_agents", "seed"]
    m = a.set_index(key)["t95"].to_frame("a").join(b.set_index(key)["t95"].to_frame("b"), how="inner").dropna()
    if len(m) < 2:
        return float("nan"), float("nan"), float("nan"), len(m)
    r = (m["a"] / m["b"]).to_numpy()
    half = stats.t.ppf(0.975, len(r) - 1) * r.std(ddof=1) / math.sqrt(len(r))
    return float(r.mean()), float(r.mean() - half), float(r.mean() + half), len(r)


def map_type_tables(D, out, made, nums):
    """Per size: mean time to 95% for each algorithm on each map type (4/8/16 drones, 3 seeds: 9 runs per cell),
    the fastest algorithm per map type, and for every other cell the paired ratio to that fastest algorithm with a
    95% interval. A cell is 'significantly slower' when the interval lies above 1. Writes map_types.json/.csv."""
    js, flat = {}, []
    for size in D:
        df = D[size]["runs"]
        if df.empty:
            continue
        algos = [l for l in TABLE_ALGOS if (df["label"] == l).any()]
        types = [t for t in TYPE_ORDER if (df["type"] == t).any()]
        cells = {}
        for t in types:
            sub = df[df["type"] == t]
            means = {l: sub[sub["label"] == l]["t95"].mean() for l in algos if (sub["label"] == l).any()}
            if not means:
                continue
            best = min(means, key=means.get)
            for l, v in means.items():
                c = sub[sub["label"] == l]
                ratio, lo, hi, n = (1.0, 1.0, 1.0, len(c)) if l == best else _paired_vs(c, sub[sub["label"] == best])
                cell = dict(t95=round(v, 1), done=round(float(c["completed"].mean()), 2), n=int(len(c)),
                            fastest=l == best, ratio=round(ratio, 2), lo=round(lo, 2), hi=round(hi, 2),
                            slower=bool(l != best and lo > 1.0), much_slower=bool(l != best and lo > 1.25))
                cells[(l, t)] = cell
                flat.append(dict(size=size, label=l, type=t, **cell))
        js[size] = dict(types=types, algos=algos,
                        rows=[dict(label=l, cells=[cells.get((l, t)) for t in types]) for l in algos])
        nums[f"fastest_by_type_{size}"] = {t: next((l for l in algos if cells.get((l, t), {}).get("fastest")), None)
                                           for t in types}
    if flat:
        pd.DataFrame(flat).to_csv(out / "map_types.csv", index=False)
        (out / "map_types.json").write_text(json.dumps(js, indent=1))
        made.append("map_types (table data)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "deck"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    D = {size: load_scale(size) for size in SIZES}
    P = load_small_scripts()
    for size, d in D.items():
        r = d["runs"]
        print(f"{size}: runs {len(r)}, prior {len(d['prior'])}, curves {len(d['curves'])}",
              "" if r.empty else r.groupby("label").size().to_dict())
    made, nums = [], {}
    for f in (lambda: fig_q1(D, out, made, nums), lambda: fig_sensing(P, out, made, nums),
              lambda: fig_q2(D, out, made, nums), lambda: fig_q2comm(P, out, made, nums),
              lambda: fig_q3(D, out, made, nums), lambda: fig_mix(D, out, made, nums),
              lambda: fig_costs(D, out, made, nums), lambda: fig_coverage(D, out, made, nums),
              lambda: fig_training(out, made, nums), lambda: map_type_tables(D, out, made, nums)):
        f()
    (out / "numbers.json").write_text(json.dumps(nums, indent=1, default=str))
    print("wrote:", ", ".join(made), "->", out)


if __name__ == "__main__":
    main()
