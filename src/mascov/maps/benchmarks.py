"""Seeded benchmark maps. Every factory returns a ready-to-use World.

Names: empty, sparse, corridor, rooms_and_door, cluttered, labyrinth, plus the procedural families of
maps/procedural.py (dense, random_rooms, maze, warehouse at 12-24 m; sparse / cluttered at 16-24 m).
All maps keep the free space connected and the start corner clear.
"""
from __future__ import annotations

import numpy as np

from mascov.common.config import WorldConfig
from mascov.geometry.polygons import random_convex_polygon, rect, union
from mascov.world.world import World


def base_starts(n: int, cfg: WorldConfig, corner=(1.0, 1.0), spacing: float = 0.8,
                per_row: int = 3) -> np.ndarray:
    """Drones start together near a base in the lower-left, in a short row/grid.

    The default 0.8 m suits every model; the quality suite uses 2.0 m with per_row=2 (see tests/quality_runs.py).
    """
    return np.array([[corner[0] + (k % per_row) * spacing, corner[1] + (k // per_row) * spacing]
                     for k in range(n)])


def _cfg(size: float, cfg: WorldConfig | None, **kw) -> WorldConfig:
    return (cfg or WorldConfig()).with_(width=size, height=size, **kw)


def empty(n_agents=2, size=12.0, seed=0, cfg: WorldConfig | None = None, start_spacing=0.8, start_per_row=3) -> World:
    c = _cfg(size, cfg)
    return World(c, [], base_starts(n_agents, c, spacing=start_spacing, per_row=start_per_row), name=f"empty_{int(size)}")


def corridor(n_agents=2, size=12.0, seed=0, cfg: WorldConfig | None = None, start_spacing=0.8, start_per_row=3) -> World:
    """S-shaped serpentine formed by two staggered horizontal walls."""
    c = _cfg(size, cfg)
    s, t = size, 0.4
    walls = [rect(0, s / 3 - t / 2, 0.75 * s, s / 3 + t / 2),
             rect(0.25 * s, 2 * s / 3 - t / 2, s, 2 * s / 3 + t / 2)]
    return World(c, walls, base_starts(n_agents, c, spacing=start_spacing, per_row=start_per_row), name=f"corridor_{int(size)}")


def rooms_and_door(n_agents=2, size=12.0, seed=0, cfg: WorldConfig | None = None, start_spacing=0.8, start_per_row=3) -> World:
    """Two rooms joined by a narrow door; the right room has an extra partial wall."""
    c = _cfg(size, cfg)
    s, t, door = size, 0.4, 1.6
    wall_lo = rect(s / 2 - t / 2, 0, s / 2 + t / 2, s / 2 - door / 2)
    wall_hi = rect(s / 2 - t / 2, s / 2 + door / 2, s / 2 + t / 2, s)
    extra = rect(0.75 * s, 0.4 * s, s, 0.4 * s + t)
    return World(c, [wall_lo, wall_hi, extra], base_starts(n_agents, c, spacing=start_spacing, per_row=start_per_row), name=f"rooms_{int(size)}")


def _random_obstacles(rng, size, k, r_lo, r_hi, keep_clear, cfg, gap) -> list:
    obstacles = []
    tries = 0
    while len(obstacles) < k and tries < 2000:
        tries += 1
        r = rng.uniform(r_lo, r_hi)
        cx, cy = rng.uniform(r + 0.5, size - r - 0.5, 2)
        poly = random_convex_polygon(rng, cx, cy, r)
        if keep_clear.distance(poly) < 1.0:
            continue
        if any(poly.distance(o) < gap for o in obstacles):
            continue
        obstacles.append(poly)
    return obstacles


def _random_map(name, n_agents, size, seed, cfg, k, r_lo, r_hi, gap, start_spacing=0.8, start_per_row=3) -> World:
    c = _cfg(size, cfg)
    starts = base_starts(n_agents, c, spacing=start_spacing, per_row=start_per_row)
    from shapely.geometry import MultiPoint
    keep_clear = MultiPoint(starts).buffer(1.0)
    for attempt in range(50):                       # retry until the map is valid
        rng = np.random.default_rng(seed * 1000 + attempt)
        obs = _random_obstacles(rng, size, k, r_lo, r_hi, keep_clear, c, gap)
        try:
            w = World(c, obs, starts, name=f"{name}_{int(size)}_s{seed}")
            if w.coverable.sum() >= 0.9 * w.n_targets:
                return w
        except ValueError:
            continue
    raise RuntimeError(f"could not build a valid {name} map for seed {seed}")


def sparse(n_agents=2, size=12.0, seed=0, cfg: WorldConfig | None = None, start_spacing=0.8, start_per_row=3) -> World:
    return _random_map("sparse", n_agents, size, seed, cfg, k=max(3, int(size / 3)),
                       r_lo=0.7, r_hi=1.4, gap=1.8,
                       start_spacing=start_spacing, start_per_row=start_per_row)


def cluttered(n_agents=2, size=12.0, seed=0, cfg: WorldConfig | None = None, start_spacing=0.8, start_per_row=3) -> World:
    return _random_map("cluttered", n_agents, size, seed, cfg, k=int(size * 1.2),
                       r_lo=0.4, r_hi=0.9, gap=1.1,
                       start_spacing=start_spacing, start_per_row=start_per_row)


def labyrinth(n_agents=2, size=12.0, seed=0, cfg: WorldConfig | None = None, start_spacing=0.8, start_per_row=3) -> World:
    """Narrow corridors with dead-end branches — worst case for optimistic planning.

    Layout (12×12):
        Horizontal walls create 4 corridors (each 2.6 m tall) joined by doors at alternating ends (an S-shaped route).
        Vertical stubs inside the corridors narrow them to 1.4 m and form dead-end pockets that an unknown-map
        explorer flies into before discovering the stub.
    """
    c = _cfg(size, cfg)
    s, t = size, 0.4  # t = wall thickness

    walls = []
    # Three horizontal walls splitting into 4 corridors
    for i in range(1, 4):
        y = i * s / 4
        # Each wall has a gap (door) at alternating sides to force S-shaped paths
        if i % 2 == 1:
            walls.append(rect(0, y - t / 2, 0.78 * s, y + t / 2))        # gap on right
        else:
            walls.append(rect(0.22 * s, y - t / 2, s, y + t / 2))        # gap on left

    # Vertical stubs inside the corridors. Each hangs from one wall of its corridor and leaves a 1.4 m passage, so
    # every corridor stays connected (a drone needs 2 x clearance = 0.9 m with the default eps_track); stubs beyond a
    # corridor's door make dead-end pockets that an unknown-map explorer flies into before it sees the stub.
    def corridor(k):
        return (0.0 if k == 0 else k * s / 4 + t / 2), (s if k == 3 else (k + 1) * s / 4 - t / 2)

    def stub(x, k, from_top):
        lo, hi = corridor(k)
        length = (hi - lo) - 1.4
        walls.append(rect(x, hi - length, x + t, hi) if from_top else rect(x, lo, x + t, lo + length))

    stub(0.6 * s, 0, from_top=False)     # corridor 1 (bottom)
    stub(0.35 * s, 1, from_top=True)     # corridor 2
    stub(0.7 * s, 2, from_top=False)     # corridor 3
    stub(0.3 * s, 3, from_top=True)      # corridor 4 (top): two stubs
    stub(0.65 * s, 3, from_top=False)

    return World(c, walls, base_starts(n_agents, c, spacing=start_spacing, per_row=start_per_row),
                 name=f"labyrinth_{int(size)}")


BENCHMARKS = dict(empty=empty, sparse=sparse, corridor=corridor,
                  rooms_and_door=rooms_and_door, cluttered=cluttered,
                  labyrinth=labyrinth)

from mascov.maps.procedural import register_into  # noqa: E402  (bigger, denser, structured maps: maze_24, ...)

register_into(BENCHMARKS)


def make_world(name: str, n_agents: int = 2, size: float = 12.0, seed: int = 0,
               cfg: WorldConfig | None = None, start_spacing: float = 0.8, start_per_row: int = 3) -> World:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown map '{name}'. options: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name](n_agents=n_agents, size=size, seed=seed, cfg=cfg,
                            start_spacing=start_spacing, start_per_row=start_per_row)
