"""Side-by-side video on an UNKNOWN map: obstacles appear only once some drone has sensed them.

    python scripts/unknown_map_video.py [MAP] [N_DRONES] [SEED] [OUT.webm]
    python scripts/unknown_map_video.py random_rooms_12 6 0 docs/unknown_map_demo.webm

One panel per algorithm, synchronised in time, all on the same map, seed and starts, all on the unknown-map interface.
Drawn like the known-map demo (scripts/compare_algorithms_video.py): drones start at scattered random positions (the
same for every algorithm), targets are grey dots until seen and then a cross in the colour of the drone that saw them
first. What is new:
  * dashed outline : an obstacle no drone has discovered yet (drawn faintly so the viewer can see what is unknown)
  * solid obstacle : discovered (seeing any part of an obstacle reveals all of it, as in the simulator)
Writes a WebM (VP8, plays in browsers) and a poster PNG of the last frame next to it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon as MplPolygon  # noqa: E402
from shapely.geometry import Point  # noqa: E402

import models  # noqa: E402,F401
from mascov.common.config import SimConfig  # noqa: E402
from mascov.common.model import get_model, registered_models  # noqa: E402
from mascov.maps.benchmarks import make_world  # noqa: E402
from mascov.sim.simulator import Simulator  # noqa: E402

MAP_NAME = sys.argv[1] if len(sys.argv) > 1 else "random_rooms_12"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "docs" / "unknown_map_demo.webm"
MAX_TIME, FPS, STRIDE = 300.0, 20, 3

ALGOS = [("Greedy", "greedy_explorer", dict(share=True)),
         ("Geodesic", "geodesic_partition", dict(share=True)),
         ("CBBA", "cbba_explorer", dict(share=True)),
         ("MAPPO", "marl_explorer", dict(share=True)),
         ("Geodesic + D-MPC", "geodesic_partition+dmpc", dict(share=True))]
ALGOS = [a for a in ALGOS if a[1] in registered_models()]
COLORS = ["#3B82F6", "#F59E0B", "#A855F7", "#E24B4A", "#14B8A6", "#EC4899", "#6B7280", "#84CC16"]
OBST, HIDDEN = "#888780", "#9a978e"


def scattered_starts(world, n, seed=0):
    """n starts spread over the free space, one per region of a grid, at least 1 m apart (as in the known-map demo)."""
    rng = np.random.default_rng(seed * 777 + 42)
    cfg = world.cfg
    margin = cfg.clearance + cfg.drone_radius + 0.3
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cw, ch = (cfg.width - 2 * margin) / cols, (cfg.height - 2 * margin) / rows
    starts: list[tuple[float, float]] = []

    def ok(x, y):
        return world.allowed.contains(Point(x, y)) and all(np.hypot(x - a, y - b) >= 1.0 for a, b in starts)

    for r in range(rows):
        for c in range(cols):
            if len(starts) >= n:
                break
            for _ in range(300):
                x = rng.uniform(margin + c * cw, margin + (c + 1) * cw)
                y = rng.uniform(margin + r * ch, margin + (r + 1) * ch)
                if ok(x, y):
                    starts.append((x, y))
                    break
    while len(starts) < n:
        x, y = rng.uniform(margin, cfg.width - margin, 2)
        if ok(x, y):
            starts.append((x, y))
    return np.array(starts)


def simulate(label, model_name, kwargs, starts):
    world = make_world(MAP_NAME, n_agents=N, seed=SEED)
    world.starts = starts.copy()
    sim = Simulator(world, get_model(model_name, seed=SEED, **kwargs),
                    SimConfig(max_time=MAX_TIME, stop_when_covered=True, known_map=False, seed=SEED))
    res = sim.run(record_paths=True)
    paths = [np.asarray(p, float) for p in sim.paths]
    T = min(len(p) for p in paths)
    # replay the team's sensing at every tick (the simulator senses every tick)
    owner = np.full(world.n_targets, -1, dtype=int)          # the drone that saw each target first
    found: set[int] = set()
    owner_at, found_at = [], []
    for t in range(T):
        for d, p in enumerate(paths):
            ids = world.sensor.visible_target_ids(p[t])
            owner[ids[owner[ids] < 0]] = d
            found |= world.sensor.visible_obstacle_ids(p[t])
        if t % STRIDE == 0 or t == T - 1:
            owner_at.append(owner.copy())
            found_at.append(set(found))
    t95 = res.time_to.get(0.95)
    print(f"{label:18s} t95={t95 if t95 is None else round(t95, 1)}s  t100={res.coverage_time}  "
          f"collisions={res.pair_events}  obstacles found {len(found)}/{len(world.obstacles)}")
    return dict(label=label, world=world, res=res, paths=paths, T=T, owner=owner_at, found=found_at)


def _pts(a):
    return a.reshape(-1, 2) if len(a) else np.empty((0, 2))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"{MAP_NAME}, {N} drones, seed {SEED}, unknown map, scattered starts")
    starts = scattered_starts(make_world(MAP_NAME, n_agents=N, seed=SEED), N, SEED)
    runs = [simulate(*a, starts) for a in ALGOS]
    W, H = runs[0]["world"].cfg.width, runs[0]["world"].cfg.height
    frames = max(len(r["owner"]) for r in runs)

    fig, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 4.3), dpi=95)
    fig.patch.set_facecolor("white")
    panels = []
    for r, ax in zip(runs, axes):
        w = r["world"]
        ax.set_xlim(0, W), ax.set_ylim(0, H), ax.set_aspect("equal")
        ax.set_xticks(np.arange(0, W + 1, 3)), ax.set_yticks(np.arange(0, H + 1, 3))
        ax.tick_params(labelsize=7)
        hidden, shown = [], []
        for o in w.obstacles:
            xy = np.asarray(o.exterior.coords)
            hidden.append(ax.add_patch(MplPolygon(xy, closed=True, fill=False, ls=(0, (3, 2)), lw=0.9,
                                                  edgecolor=HIDDEN, alpha=0.9, zorder=2)))
            shown.append(ax.add_patch(MplPolygon(xy, closed=True, facecolor=OBST, edgecolor="#444441", lw=1,
                                                 visible=False, zorder=2)))
        uncov = ax.scatter([], [], s=9, marker="o", facecolors="#C8C7C2", edgecolors="#6B7280", lw=0.3, zorder=4)
        cov = [ax.scatter([], [], s=26, marker="X", facecolors=COLORS[d % len(COLORS)], edgecolors="#111827",
                          lw=0.4, zorder=10) for d in range(N)]
        trails, bodies, rings = [], [], []
        for d in range(N):
            c = COLORS[d % len(COLORS)]
            trails.append(ax.plot([], [], color=c, lw=1.2, zorder=5)[0])
            rings.append(ax.add_patch(Circle(w.starts[d], w.cfg.r_sense, fill=False, ls="--", lw=0.7,
                                             edgecolor=c, alpha=0.4, zorder=6)))
            bodies.append(ax.add_patch(Circle(w.starts[d], w.cfg.drone_radius, facecolor=c, edgecolor="black",
                                              lw=0.6, zorder=8)))
            ax.plot(*w.starts[d], "o", color=c, ms=4, mec="black", mew=0.5, zorder=3)
        ax.set_title(r["label"], fontsize=11, fontweight="bold")
        status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=7.5, zorder=20,
                         bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999", alpha=0.9))
        panels.append(dict(r=r, uncov=uncov, cov=cov, hidden=hidden, shown=shown, trails=trails, bodies=bodies,
                           rings=rings, status=status))
    fig.text(0.5, 0.015, "grey dot: target not yet seen   ·   cross: seen (drone colour)   ·   dashed: obstacle not "
             f"yet discovered   ·   solid: discovered   ·   {MAP_NAME}, {N} drones, unknown map", ha="center",
             fontsize=9.5, color="#4a5568")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.canvas.draw()
    h, wpx = np.asarray(fig.canvas.buffer_rgba()).shape[:2]
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"VP80"), FPS, (wpx, h))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write VP8/WebM here")
    try:
        for f in range(frames):
            for p in panels:
                r, w = p["r"], p["r"]["world"]
                k = min(f, len(r["owner"]) - 1)
                tick = min(k * STRIDE, r["T"] - 1)
                owner, tg, cv = r["owner"][k], w.targets, w.coverable
                p["uncov"].set_offsets(_pts(tg[cv & (owner < 0)]))
                for d in range(N):
                    p["cov"][d].set_offsets(_pts(tg[cv & (owner == d)]))
                for oid, (hp, sp) in enumerate(zip(p["hidden"], p["shown"])):
                    vis = oid in r["found"][k]
                    hp.set_visible(not vis)
                    sp.set_visible(vis)
                for d, path in enumerate(r["paths"]):
                    pos = path[tick]
                    p["trails"][d].set_data(path[:tick + 1, 0], path[:tick + 1, 1])
                    p["bodies"][d].center = pos
                    p["rings"][d].center = pos
                c = float((owner[cv] >= 0).mean())
                p["status"].set_text(f"t={tick * w.cfg.dt:.1f}s  cov={100 * c:.0f}%  coll={r['res'].pair_events}\n"
                                     f"obstacles found {len(r['found'][k])}/{len(w.obstacles)}")
            fig.canvas.draw()
            writer.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR))
        for _ in range(FPS * 2):                              # hold the last frame for 2 s
            writer.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR))
    finally:
        writer.release()
    fig.savefig(OUT.with_suffix(".png"), facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {frames} frames) and {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
