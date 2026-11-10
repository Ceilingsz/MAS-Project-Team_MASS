"""Coverage-vs-time curves for B2, B3 and B4 on one map. Saves docs/baselines_<map>_N<n>.png.

  python scripts/plot_baselines.py [map] [n_agents]      (run from the repository root)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src"), sys.path.insert(0, ".")
import models  # noqa: F401, E402
from mascov.common.config import SimConfig  # noqa: E402
from mascov.common.model import get_model  # noqa: E402
from mascov.maps.benchmarks import make_world  # noqa: E402
from mascov.metrics.bounds import lower_bound  # noqa: E402
from mascov.sim.simulator import Simulator  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "rooms_and_door"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
w = make_world(name, n_agents=n, seed=0)
lb = lower_bound(w)["best"]
fig, ax = plt.subplots(figsize=(7.5, 4.4))
styles = {"greedy_centralised": ("B2 centralised greedy", "#1D9E75"),
          "static_partition": ("B4 static partition", "#534AB7"),
          "greedy_uncoordinated": ("B3 uncoordinated greedy", "#BA7517")}
for key, (label, color) in styles.items():
    r = Simulator(w, get_model(key), SimConfig(max_time=900, stop_when_covered=False)).run()
    ax.plot(r.log.column("t"), r.log.column("coverage"), color=color, lw=2, label=f"{label} (done at {r.coverage_time:.0f} s)")
    if key == "greedy_uncoordinated":      # only B3's drones decide from their own memory alone
        believed = [r.log.column(f"believed_{i}") for i in range(n)]
        ax.plot(r.log.column("t"), sum(believed) / n, color=color, lw=1.2, ls=":",
                label="B3 mean belief of the drones")
ax.axvline(lb, color="k", lw=1, ls="--", label=f"lower bound ({lb:.0f} s)")
ax.set_xlabel("time [s]"), ax.set_ylabel("fraction of coverable targets")
ax.set_title(f"{name}, N={n}: true coverage over time")
ax.set_ylim(0, 1.02), ax.grid(alpha=0.3), ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
out = f"docs/baselines_{name}_N{n}.png"
plt.savefig(out, dpi=120)
print("saved", out)
