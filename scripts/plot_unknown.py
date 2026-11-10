"""What drone 0 believes on an unknown map at a fixed time, without and with sharing, next to the truth.

  python scripts/plot_unknown.py sparse 3 25 docs/unknown_map_belief.png
Run from the repository root. Dashed red outlines are obstacles the drone has NOT discovered yet.
"""
import sys

sys.path.insert(0, "src"), sys.path.insert(0, ".")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import models  # noqa: F401, E402
from mascov.common.config import SimConfig  # noqa: E402
from mascov.common.model import get_model  # noqa: E402
from mascov.maps.benchmarks import make_world  # noqa: E402
from mascov.sim.simulator import Simulator  # noqa: E402
from mascov.viz.plot import plot_belief, plot_world  # noqa: E402
from mascov.world.belief import FREE  # noqa: E402


def run(world, model, t_end):
    sim = Simulator(world, get_model(model), SimConfig(max_time=t_end, known_map=False, stop_when_covered=False))
    sim.run(record_paths=True)
    return sim


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "sparse"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    t_end = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    out = sys.argv[4] if len(sys.argv) > 4 else "docs/unknown_map_belief.png"
    w = make_world(name, n_agents=n, seed=0)
    alone, shared = run(w, "frontier_greedy", t_end), run(w, "frontier_greedy_shared", t_end)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    seen_by_anyone = np.any([b.state[w.target_cell] == FREE for b in shared.beliefs], axis=0)
    plot_world(w, covered=seen_by_anyone, positions=[p[-1] for p in shared.paths], paths=shared.paths, ax=axes[0],
               title=f"truth at t={t_end:.0f} s: {name}, N={n}, drones sharing (green = seen by someone)")
    for ax, sim, what in [(axes[1], alone, "no sharing"), (axes[2], shared, "sharing")]:
        b = sim.beliefs[0]
        plot_belief(w, b, positions=sim.paths[0][-1], paths=[sim.paths[0]], ax=ax,
                    title=f"drone 0, {what}: {int((b.state == FREE).sum())} cells seen, "
                          f"{len(b.obstacles)}/{len(w.obstacles)} obstacles known")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
