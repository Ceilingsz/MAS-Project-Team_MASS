"""Generate presentation-ready comparison plots for CBBA and Known vs. Unknown maps."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_known_vs_unknown(csv_path="docs/known_vs_unknown_results.csv", out_dir="docs"):
    path = Path(csv_path)
    if not path.exists():
        print(f"Results file {csv_path} does not exist.")
        return

    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["completed"] == "True":
                rows.append(r)

    if not rows:
        print("No completed rows found.")
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Bar chart: Coverage time across maps for N=2
    maps = sorted(list({r["map"] for r in rows}))
    methods = ["B2", "B4", "CBBA", "B3"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax_idx, mode in enumerate(["known", "unknown"]):
        ax = axes[ax_idx]
        x = np.arange(len(maps))
        width = 0.2

        for i, m in enumerate(methods):
            times = []
            for mp in maps:
                match = [float(r["T"]) for r in rows if r["map"] == mp and r["method"] == m and r["mode"] == mode and r["n"] == "2"]
                times.append(times[-1] if not match and times else (match[0] if match else 0))

            ax.bar(x + (i - 1.5) * width, times, width, label=m)

        ax.set_title(f"Coverage Time (N=2) — {mode.capitalize()} Map", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in maps], fontsize=10)
        ax.set_ylabel("Coverage Time T (s)", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(fontsize=10)

    plt.tight_layout()
    chart1 = out / "cbba_coverage_time_comparison.png"
    plt.savefig(chart1, dpi=200)
    plt.close()
    print(f"Saved: {chart1}")

    # 2. Safety plot: Collisions in Unknown Maps
    fig, ax = plt.subplots(figsize=(8, 4.5))
    unknown_rows = [r for r in rows if r["mode"] == "unknown" and r["n"] == "2"]
    if unknown_rows:
        methods_coll = ["B2", "B3", "B4", "CBBA"]
        colls_by_method = []
        for m in methods_coll:
            c = sum(int(r["collisions"]) for r in unknown_rows if r["method"] == m)
            colls_by_method.append(c)

        colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in colls_by_method]
        bars = ax.bar(methods_coll, colls_by_method, color=colors, width=0.5)
        ax.set_ylabel("Total Collision Events", fontsize=11)
        ax.set_title("Obstacle Collision Violations in Unknown Environments", fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        chart2 = out / "unknown_map_collisions.png"
        plt.savefig(chart2, dpi=200)
        plt.close()
        print(f"Saved: {chart2}")


if __name__ == "__main__":
    plot_known_vs_unknown()
