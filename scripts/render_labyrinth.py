"""Render the labyrinth map as a PNG image.

    python scripts/render_labyrinth.py

Output: labyrinth_map.png in the repo root.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from mascov.maps.benchmarks import make_world

world = make_world("labyrinth", n_agents=4, size=12.0, seed=0)

fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_aspect("equal")
ax.set_facecolor("#FBFBF8")
fig.patch.set_facecolor("#FBFBF8")

# Draw obstacles
for obs in world.obstacles:
    coords = np.asarray(obs.exterior.coords)
    ax.add_patch(MplPolygon(coords, closed=True, facecolor="#555555", edgecolor="#333333", lw=1.5))

# Draw targets as small dots
targets = world.targets
coverable = world.coverable
ax.scatter(targets[coverable, 0], targets[coverable, 1], s=4, c="#CBD5E1", zorder=1, alpha=0.5)

# Draw start positions
DRONE_COLORS = ["#3B82F6", "#F59E0B", "#A855F7", "#E24B4A"]
for i, s in enumerate(world.starts):
    ax.plot(s[0], s[1], 'o', color=DRONE_COLORS[i], markersize=10,
            markeredgecolor='black', markeredgewidth=1, zorder=10)

ax.set_title("Labyrinth Map — corridors with dead ends", fontsize=16, fontweight="bold", pad=12)
ax.set_xlabel("x [m]", fontsize=12)
ax.set_ylabel("y [m]", fontsize=12)
ax.grid(True, alpha=0.2)

fig.tight_layout()
out = ROOT / "labyrinth_map.png"
fig.savefig(str(out), dpi=150)
plt.close(fig)
print(f"Saved: {out}")
