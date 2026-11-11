"""Side-by-side simulation video comparing all algorithms on the same map and seed.

    python scripts/compare_algorithms_video.py [MAP] [N_DRONES] [SEED]
    python scripts/compare_algorithms_video.py rooms_and_door 4 0

Output: <map>_comparison.mp4 with one panel per algorithm, synchronised in time.
Requires: opencv-python, matplotlib, shapely, osqp (for D-MPC; skipped if missing).
Run from the repository root.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon as MplPolygon
from shapely.geometry import Point

from mascov.common.config import SimConfig
from mascov.common.model import get_model, registered_models
from mascov.maps.benchmarks import make_world
from mascov.sim.simulator import Simulator
import models  # noqa: register all

# ---- settings ----
MAP_NAME = sys.argv[1] if len(sys.argv) > 1 else "rooms_and_door"
N_DRONES = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MAP_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MAX_TIME = 300.0
VIDEO_FPS = 30
FRAME_STRIDE = 2

# Algorithms to compare (label, model name, kwargs)
ALGORITHMS = [
    ("Greedy",         "greedy_explorer",    dict(share=True)),
    ("Greedy + Goals", "greedy_explorer",    dict(share=True, coordinate=True)),
    ("Geodesic",       "geodesic_partition", dict(share=True)),
    ("CBBA",           "cbba_explorer",      dict(share=True)),
]

# Add D-MPC flying if available: geodesic goals flown by D-MPC (models/mix_dmpc), the fastest collision-free combination
if "geodesic_partition+dmpc" in registered_models():
    ALGORITHMS.append(("Geodesic + D-MPC", "geodesic_partition+dmpc", dict(share=True)))

# One panel per algorithm: CBBA is simulated once. (CBAA, its single-task variant, used to be added here as a
# second CBBA-family panel; it is left out so the video shows the same 5 algorithms as the experiments.)
assert len({a[1] for a in ALGORITHMS if a[1].startswith(("cbba", "cbaa"))}) == 1, "exactly one CBBA panel"

DRONE_COLORS = ["#3B82F6", "#F59E0B", "#A855F7", "#E24B4A", "#14B8A6", "#EC4899"]

OUTPUT_FILE = ROOT / f"{MAP_NAME}_comparison.mp4"


# ---- scatter drones across the map ----
def scattered_starts(world, n_drones, seed=0):
    """Generate n_drones start positions spread across free space.

    Places drones in different quadrants/regions of the map so they don't
    all begin from the same corner. Each position is guaranteed to be inside
    the 'allowed' (collision-free) planning space.
    """
    rng = np.random.default_rng(seed * 777 + 42)
    cfg = world.cfg
    margin = cfg.clearance + cfg.drone_radius + 0.3  # stay clear of walls and obstacles

    starts = []
    max_attempts = 5000
    attempts = 0

    # Divide the map into a grid of regions and try to put one drone per region
    cols = int(np.ceil(np.sqrt(n_drones)))
    rows = int(np.ceil(n_drones / cols))
    cell_w = (cfg.width - 2 * margin) / cols
    cell_h = (cfg.height - 2 * margin) / rows

    assigned = 0
    for r in range(rows):
        for c in range(cols):
            if assigned >= n_drones:
                break
            # Try to place a drone in this region
            x_lo = margin + c * cell_w
            x_hi = margin + (c + 1) * cell_w
            y_lo = margin + r * cell_h
            y_hi = margin + (r + 1) * cell_h
            placed = False
            for _ in range(200):
                x = rng.uniform(x_lo, x_hi)
                y = rng.uniform(y_lo, y_hi)
                pt = Point(x, y)
                if world.allowed.contains(pt):
                    # Also check distance from other starts
                    too_close = False
                    for s in starts:
                        if np.hypot(x - s[0], y - s[1]) < 1.0:
                            too_close = True
                            break
                    if not too_close:
                        starts.append([x, y])
                        placed = True
                        assigned += 1
                        break

    # Fallback: if some drones couldn't be placed in regions, scatter randomly
    while len(starts) < n_drones and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(margin, cfg.width - margin)
        y = rng.uniform(margin, cfg.height - margin)
        pt = Point(x, y)
        if world.allowed.contains(pt):
            too_close = any(np.hypot(x - s[0], y - s[1]) < 1.0 for s in starts)
            if not too_close:
                starts.append([x, y])

    if len(starts) < n_drones:
        raise RuntimeError(f"Could only place {len(starts)}/{n_drones} drones in free space")

    return np.array(starts)


# ---- run simulations ----
print(f"Map: {MAP_NAME}, {N_DRONES} drones, seed {MAP_SEED}")
print(f"Algorithms: {[a[0] for a in ALGORITHMS]}")

# Build one world to compute scattered starts (same for all algorithms)
ref_world = make_world(MAP_NAME, n_agents=N_DRONES, size=12.0, seed=MAP_SEED)
spread_starts = scattered_starts(ref_world, N_DRONES, seed=MAP_SEED)
print(f"Drone starts (scattered):")
for i, s in enumerate(spread_starts):
    print(f"  drone {i}: ({s[0]:.2f}, {s[1]:.2f})")

results = []
for label, model_name, kwargs in ALGORITHMS:
    print(f"  Running {label}...", end=" ", flush=True)
    world = make_world(MAP_NAME, n_agents=N_DRONES, size=12.0, seed=MAP_SEED)
    # Override starts with scattered positions (same for every algorithm = fair comparison)
    world.starts = spread_starts.copy()
    model = get_model(model_name, seed=MAP_SEED, **kwargs)
    iface = getattr(model, "map_interface", "unknown")
    sim_cfg = SimConfig(max_time=MAX_TIME, stop_when_covered=True,
                        known_map=(iface == "known"), seed=MAP_SEED)
    sim = Simulator(world, model, sim_cfg)
    result = sim.run(record_paths=True)
    paths = [np.asarray(p, float) for p in sim.paths]
    print(f"done — t95={result.time_to.get(0.95, '?'):.1f}s, collisions={result.collision_events}")
    results.append(dict(label=label, world=world, result=result, paths=paths))

# ---- prepare video ----
n_algo = len(results)
fig, axes = plt.subplots(1, n_algo, figsize=(5 * n_algo, 5.5), dpi=120)
if n_algo == 1:
    axes = [axes]
fig.patch.set_facecolor("white")

# Find the longest path
max_ticks = max(min(len(p) for p in r["paths"]) for r in results)
frame_indices = list(range(0, max_ticks, FRAME_STRIDE))
if frame_indices[-1] != max_ticks - 1:
    frame_indices.append(max_ticks - 1)


def draw_static(ax, world):
    cfg = world.cfg
    ax.set_xlim(0, cfg.width)
    ax.set_ylim(0, cfg.height)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, cfg.width + 1, 3))
    ax.set_yticks(np.arange(0, cfg.height + 1, 3))
    ax.tick_params(labelsize=7)
    for obs in world.obstacles:
        coords = np.asarray(obs.exterior.coords)
        ax.add_patch(MplPolygon(coords, closed=True, facecolor="#888780", edgecolor="#444441", lw=1, zorder=2))


# Set up each panel
panel_data = []
for idx, (r, ax) in enumerate(zip(results, axes)):
    world = r["world"]
    draw_static(ax, world)
    ax.set_title(r["label"], fontsize=11, fontweight="bold")

    # targets scatter
    targets = world.targets if hasattr(world, "targets") else world.viewpoints.positions
    coverable = world.coverable if hasattr(world, "coverable") else np.ones(len(targets), bool)
    uncov_sc = ax.scatter([], [], s=10, marker="o", facecolors="#C8C7C2", edgecolors="#6B7280", lw=0.3, zorder=4)

    # per-drone scatters for covered targets
    drone_covered = []
    for d in range(N_DRONES):
        sc = ax.scatter([], [], s=36, marker="X", facecolors=DRONE_COLORS[d % len(DRONE_COLORS)],
                        edgecolors="#111827", lw=0.5, zorder=10)
        drone_covered.append(sc)

    # trails, bodies, sensing circles
    trails, bodies, circles = [], [], []
    for d in range(N_DRONES):
        c = DRONE_COLORS[d % len(DRONE_COLORS)]
        tr, = ax.plot([], [], color=c, lw=1.2, zorder=5)
        trails.append(tr)
        body = Circle(world.starts[d], radius=world.cfg.drone_radius, facecolor=c, edgecolor="black", lw=0.6, zorder=8)
        ax.add_patch(body)
        bodies.append(body)
        sc = Circle(world.starts[d], radius=world.cfg.r_sense, fill=False, ls="--", edgecolor=c, lw=0.7, alpha=0.4, zorder=6)
        ax.add_patch(sc)
        circles.append(sc)

    # Mark start positions with small dots
    for d in range(N_DRONES):
        ax.plot(world.starts[d][0], world.starts[d][1], 'o', color=DRONE_COLORS[d % len(DRONE_COLORS)],
                markersize=4, markeredgecolor='black', markeredgewidth=0.5, zorder=3)

    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=7.5,
                     bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999", alpha=0.9), zorder=20)

    # coverage ownership
    owner = np.full(len(targets), -1, dtype=int)

    panel_data.append(dict(
        world=world, paths=r["paths"], targets=targets, coverable=coverable,
        uncov_sc=uncov_sc, drone_covered=drone_covered, trails=trails,
        bodies=bodies, circles=circles, status=status, owner=owner,
        result=r["result"],
    ))

fig.tight_layout()
fig.canvas.draw()
rgba = np.asarray(fig.canvas.buffer_rgba())
h, w = rgba.shape[:2]
writer = cv2.VideoWriter(str(OUTPUT_FILE), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))

print(f"Rendering {len(frame_indices)} frames...")

def nonempty(pts):
    pts = np.asarray(pts, float)
    return pts.reshape(-1, 2) if len(pts) else np.empty((0, 2), float)

try:
    for fi, tick in enumerate(frame_indices):
        for pd in panel_data:
            world = pd["world"]
            paths = pd["paths"]
            targets, coverable, owner = pd["targets"], pd["coverable"], pd["owner"]
            n_cov = max(1, int(coverable.sum()))

            # update coverage ownership
            for d, dp in enumerate(paths):
                if tick < len(dp):
                    pos = dp[tick]
                    vis = world.sensor.visible_target_ids(pos)
                    new = vis[owner[vis] == -1]
                    if new.size:
                        dists = np.linalg.norm(targets[new] - pos, axis=1)
                        for t, dist in zip(new, dists):
                            owner[t] = d

            is_cov = owner >= 0
            pd["uncov_sc"].set_offsets(nonempty(targets[coverable & ~is_cov]))
            for d in range(N_DRONES):
                pd["drone_covered"][d].set_offsets(nonempty(targets[coverable & (owner == d)]))

            for d, dp in enumerate(paths):
                t = min(tick, len(dp) - 1)
                pos = dp[t]
                pd["trails"][d].set_data(dp[:t+1, 0], dp[:t+1, 1])
                pd["bodies"][d].center = pos
                pd["circles"][d].center = pos

            sim_time = tick * world.cfg.dt
            cov_frac = is_cov[coverable].sum() / n_cov
            pd["status"].set_text(f"t={sim_time:.1f}s  cov={100*cov_frac:.0f}%\ncoll={pd['result'].collision_events}")

        fig.canvas.draw()
        bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        writer.write(bgr)

        if fi % 50 == 0:
            print(f"  frame {fi}/{len(frame_indices)}")
finally:
    writer.release()
    plt.close(fig)

print(f"Video saved to: {OUTPUT_FILE}")
print("\nSummary:")
for r in results:
    res = r["result"]
    t95 = res.time_to.get(0.95, res.final_time)
    print(f"  {r['label']:20s}  t95={t95:.1f}s  collisions={res.collision_events}  coverage={res.final_coverage:.1%}")
