"""Save a gallery of all benchmark maps to docs/benchmark_maps.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mascov.maps.benchmarks import BENCHMARKS, make_world
from mascov.viz.plot import plot_world

names = list(BENCHMARKS)
fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.3))
for ax, name in zip(axes, names):
    w = make_world(name, n_agents=3, seed=1)
    plot_world(w, ax=ax, show_viewpoints=True, title=f"{name}\n{w.summary()['n_obstacles']} obstacles, {int(w.coverable.sum())} coverable targets")
plt.tight_layout()
plt.savefig("docs/benchmark_maps.png", dpi=110)
print("saved docs/benchmark_maps.png")
