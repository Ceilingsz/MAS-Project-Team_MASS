"""Procedural maps: bigger worlds, more obstacles, more structure. Seeded and reproducible.

Families (each takes ``size`` in metres and ``seed``):
  dense         many small random convex obstacles (about 0.12 per m^2, twice ``cluttered``'s density)
  random_rooms  an irregular grid of rooms (3.5-6 m), every interior wall with a door, some walls removed (open
                plan), a little furniture per room
  maze          a random spanning-tree maze on a grid of about 3 m cells, with about 15% extra openings (loops)
  warehouse     rows of shelves separated by aisles, broken by random cross-aisles

Registered sizes: every family at 12, 16, 20 and 24 m as ``<family>_<size>`` (e.g. ``maze_24``), plus the seeded
``sparse`` and ``cluttered`` generators at the larger sizes (``sparse_24``, ``cluttered_20`` ...). ``make_world`` finds
them like any benchmark. All of them keep the start corner clear (walls touching it are dropped, which only adds
openings), keep the free space connected and are rejected and redrawn unless >= 90% of the targets are coverable.

Seeds below 1000 are reserved for evaluation; the MAPPO trainer (models/marl_explorer/train.py) draws seeds >= 1000.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPoint

from mascov.common.config import WorldConfig
from mascov.geometry.polygons import random_convex_polygon, rect
from mascov.world.world import World

WALL = 0.3          # wall thickness [m]
DOOR = 1.6          # door width [m]: two clearances (0.9 m at eps_track 0.3) plus margin
SIZES = (12, 16, 20, 24)
FAMILIES = ("dense", "random_rooms", "maze", "warehouse")


def _per_row(n: int, per_row: int | None) -> int:
    """Rows of 3 (the benchmarks' default), 4 for teams larger than 9, so 16 drones fit in a 3.4 x 3.4 m base."""
    return max(per_row or 0, 3 if n <= 9 else 4)


def starts_for(n: int, cfg: WorldConfig, spacing: float = 0.8, per_row: int | None = None) -> np.ndarray:
    """A compact base in the lower-left corner."""
    per_row = _per_row(n, per_row)
    return np.array([[1.0 + (k % per_row) * spacing, 1.0 + (k // per_row) * spacing] for k in range(n)])


def _build(name, cfg, walls_fn, starts, seed, min_coverable=0.9):
    keep = MultiPoint(starts).buffer(0.9)
    for attempt in range(40):
        rng = np.random.default_rng(seed * 7919 + attempt)
        obs = [o for o in walls_fn(rng) if o.distance(keep) > 0.05 and o.area > 1e-6]
        try:
            w = World(cfg, obs, starts, name=f"{name}_s{seed}")
        except ValueError:
            continue
        if w.coverable.sum() >= min_coverable * w.n_targets:
            return w
    raise RuntimeError(f"could not build a valid {name} map for seed {seed}")


def _hwall(x0, x1, y):
    return rect(x0, y - WALL / 2, x1, y + WALL / 2)


def _vwall(x, y0, y1):
    return rect(x - WALL / 2, y0, x + WALL / 2, y1)


def _with_door(rng, lo, hi, make, door=DOOR):
    """A wall segment from lo to hi with one door at a random position (two rectangles)."""
    if hi - lo < door + 0.6:
        return []
    d0 = rng.uniform(lo + 0.3, hi - door - 0.3)
    out = []
    if d0 - lo > 1e-3:
        out.append(make(lo, d0))
    if hi - (d0 + door) > 1e-3:
        out.append(make(d0 + door, hi))
    return out


def _cuts(rng, size, lo, hi, first_min=4.5):
    """Random cut positions splitting [0, size] into pieces of length in [lo, hi] (the first >= first_min)."""
    cuts, x = [], 0.0
    while True:
        step = rng.uniform(max(lo, first_min) if not cuts else lo, hi)
        x += step
        if size - x < lo:
            break
        cuts.append(x)
    return cuts


# ------------------------------------------------------------------------------------------------ families
def _dense_walls(size):
    def make(rng):
        k = int(0.12 * size * size)
        obs, tries = [], 0
        while len(obs) < k and tries < 6000:
            tries += 1
            r = rng.uniform(0.35, 0.9)
            cx, cy = rng.uniform(r + 0.4, size - r - 0.4, 2)
            p = random_convex_polygon(rng, cx, cy, r)
            if any(p.distance(o) < 1.05 for o in obs):
                continue
            obs.append(p)
        return obs
    return make


def _rooms_walls(size):
    def make(rng):
        xs, ys = _cuts(rng, size, 3.5, 6.0), _cuts(rng, size, 3.5, 6.0)
        bx, by = [0.0] + xs + [size], [0.0] + ys + [size]
        walls = []
        for x in xs:                                   # vertical walls, one segment per room row
            for y0, y1 in zip(by[:-1], by[1:]):
                if rng.random() < 0.2:
                    continue                           # open plan
                walls += _with_door(rng, y0, y1, lambda a, b, x=x: _vwall(x, a, b))
        for y in ys:
            for x0, x1 in zip(bx[:-1], bx[1:]):
                if rng.random() < 0.2:
                    continue
                walls += _with_door(rng, x0, x1, lambda a, b, y=y: _hwall(a, b, y))
        for x0, x1 in zip(bx[:-1], bx[1:]):            # furniture
            for y0, y1 in zip(by[:-1], by[1:]):
                for _ in range(rng.integers(0, 3)):
                    r = rng.uniform(0.3, 0.6)
                    if x1 - x0 < 2 * r + 2.4 or y1 - y0 < 2 * r + 2.4:
                        continue
                    cx = rng.uniform(x0 + r + 1.2, x1 - r - 1.2)
                    cy = rng.uniform(y0 + r + 1.2, y1 - r - 1.2)
                    walls.append(random_convex_polygon(rng, cx, cy, r))
        return walls
    return make


def _maze_walls(size):
    def make(rng):
        n = max(4, int(round(size / rng.uniform(2.6, 3.4))))
        c = size / n
        # spanning tree by randomised depth-first search; open[(i, j, d)] = passage between cell (i, j) and its
        # east (d = 0) or north (d = 1) neighbour
        seen, stack, opened = {(0, 0)}, [(0, 0)], set()
        while stack:
            i, j = stack[-1]
            nb = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= i + di < n and 0 <= j + dj < n and (i + di, j + dj) not in seen]
            if not nb:
                stack.pop()
                continue
            a = nb[rng.integers(len(nb))]
            lo = min((i, j), a)
            opened.add((lo[0], lo[1], 0 if a[0] != i else 1))
            seen.add(a)
            stack.append(a)
        walls = []
        for i in range(n):
            for j in range(n):
                for d in (0, 1):
                    if (d == 0 and i == n - 1) or (d == 1 and j == n - 1):
                        continue
                    if (i, j, d) in opened or rng.random() < 0.15:
                        continue
                    if d == 0:   # wall on the east edge of cell (i, j)
                        walls.append(_vwall((i + 1) * c, j * c, (j + 1) * c))
                    else:
                        walls.append(_hwall(i * c, (i + 1) * c, (j + 1) * c))
        return walls
    return make


def _warehouse_walls(size):
    def make(rng):
        aisle, depth = rng.uniform(1.7, 2.4), rng.uniform(0.6, 1.1)
        walls, y = [], 3.6 + rng.uniform(0, 1.0)      # leave the base row clear
        horizontal = rng.random() < 0.5
        while y + depth < size - 1.2:
            x = rng.uniform(0.9, 2.0)
            while x < size - 1.2:
                ln = rng.uniform(2.5, 6.0)
                x1 = min(x + ln, size - 0.9)
                if x1 - x > 1.0:
                    walls.append(rect(x, y, x1, y + depth) if horizontal else rect(y, x, y + depth, x1))
                x = x1 + rng.uniform(1.6, 2.4)          # cross-aisle
            y += depth + aisle
        return walls
    return make


_MAKERS = {"dense": _dense_walls, "random_rooms": _rooms_walls, "maze": _maze_walls, "warehouse": _warehouse_walls}


def procedural(family: str, n_agents=2, size=12.0, seed=0, cfg: WorldConfig | None = None, start_spacing=0.8,
               start_per_row=None) -> World:
    c = (cfg or WorldConfig()).with_(width=float(size), height=float(size))
    starts = starts_for(n_agents, c, start_spacing, start_per_row)
    return _build(f"{family}_{int(size)}", c, _MAKERS[family](float(size)), starts, seed)


def _factory(family, fixed_size):
    def make(n_agents=2, size=None, seed=0, cfg: WorldConfig | None = None, start_spacing=0.8, start_per_row=None):
        return procedural(family, n_agents, fixed_size, seed, cfg, start_spacing, start_per_row)   # size is fixed
    make.__name__ = f"{family}_{fixed_size}"
    return make


def _sized(base_fn, fixed_size):
    def make(n_agents=2, size=None, seed=0, cfg: WorldConfig | None = None, start_spacing=0.8, start_per_row=None):
        per_row = _per_row(n_agents, start_per_row)
        return base_fn(n_agents=n_agents, size=float(fixed_size), seed=seed, cfg=cfg, start_spacing=start_spacing,
                       start_per_row=per_row)
    make.__name__ = f"{base_fn.__name__}_{fixed_size}"
    return make


def register_into(benchmarks: dict) -> None:
    for fam in FAMILIES:
        for s in SIZES:
            benchmarks[f"{fam}_{s}"] = _factory(fam, s)
    for base in ("sparse", "cluttered"):
        for s in SIZES[1:]:
            benchmarks[f"{base}_{s}"] = _sized(benchmarks[base], s)
