"""End-to-end demo: build a map, run the single-drone greedy baseline, plot the result.

Run from the repository root:  python scripts/demo_greedy.py [map] [seed]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mascov.common.config import SimConfig
from mascov.maps.benchmarks import make_world
from mascov.metrics.bounds import lower_bound
from mascov.sim.simulator import Simulator
from mascov.viz.plot import plot_world
from models.greedy_single import GreedySingle

name = sys.argv[1] if len(sys.argv) > 1 else "sparse"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
world = make_world(name, n_agents=1, seed=seed)
print("world:", world.summary())
lb = lower_bound(world)
print("lower bounds [s]:", {k: round(v, 1) for k, v in lb.items()})

sim = Simulator(world, GreedySingle(), SimConfig(max_time=900))
res = sim.run(record_paths=True)
print("result:", res.summary())
if res.coverage_time:
    print(f"T / lower bound = {res.coverage_time / max(lb['best'], 1e-9):.2f}")

covered = world.sensor.targets[:0]
import numpy as np
cov = np.zeros(world.n_targets, bool)
for p in sim.paths[0][::5]:
    cov[world.sensor.visible_target_ids(p)] = True
ax = plot_world(world, covered=cov, paths=sim.paths, title=f"{world.name}: greedy_single, T={res.coverage_time}")
plt.savefig(f"demo_{world.name}.png", dpi=130, bbox_inches="tight")
print("saved", f"demo_{world.name}.png")
