"""Draw the procedural map families (maps/procedural.py) at 12 and 24 m.

    python scripts/plot_procedural_maps.py docs/procedural_maps.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mascov.maps.benchmarks import make_world  # noqa: E402

FAMS = ["dense", "random_rooms", "maze", "warehouse"]


def main(out="docs/procedural_maps.png"):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), dpi=110)
    for col, fam in enumerate(FAMS):
        for row, size in enumerate((12, 24)):
            ax = axes[row, col]
            w = make_world(f"{fam}_{size}", n_agents=8, seed=0)
            for o in w.obstacles:
                x, y = o.exterior.xy
                ax.fill(x, y, color="#4a4a4a", lw=0)
            ax.scatter(w.starts[:, 0], w.starts[:, 1], s=12, color="#d0452f", zorder=3)
            ax.set_xlim(0, size), ax.set_ylim(0, size), ax.set_aspect("equal")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{fam}_{size}  ({len(w.obstacles)} obstacles)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
