"""Ground-truth world: continuous workspace, convex-polygon obstacles, sampled coverage targets."""
from __future__ import annotations

from functools import cached_property
from typing import Sequence

import numpy as np
import shapely
from shapely.geometry import Point, Polygon

from mascov.common.config import WorldConfig
from mascov.geometry.pathing import PathPlanner
from mascov.geometry.polygons import inflate, is_convex, polygon_parts, rect
from mascov.geometry.visibility import prepared_union


class World:
    """Continuous 2D workspace [0,W]x[0,H] with static convex polygonal obstacles.

    * ``targets``  : lattice points in free space whose coverage stands in for area coverage
    * ``allowed``  : planning free space (drone centres), i.e. walls shrunk and obstacles grown
                     by ``cfg.clearance`` = r_b + eps
    * ``starts``   : initial drone positions, one row per agent
    """

    def __init__(self, cfg: WorldConfig, obstacles: Sequence[Polygon], starts,
                 victims=None, name: str = "world"):
        self.cfg = cfg
        self.name = name
        self.obstacles: list[Polygon] = list(obstacles)
        for k, o in enumerate(self.obstacles):
            if not is_convex(o):
                raise ValueError(f"obstacle {k} is not convex; decompose it first")
        self.starts = np.asarray(starts, dtype=float).reshape(-1, 2)
        self.victims = (np.zeros((0, 2)) if victims is None
                        else np.asarray(victims, dtype=float).reshape(-1, 2))
        self.bounds = rect(0.0, 0.0, cfg.width, cfg.height)
        self.obstacle_union = prepared_union(self.obstacles)
        self.lattice, self.cell_free = self._sample_lattice()
        self.targets = self.lattice[self.cell_free]
        self.target_cell = np.flatnonzero(self.cell_free)      # target m sits on lattice cell target_cell[m]
        self._check_starts()

    # ------------------------------------------------------------------ construction
    def _sample_lattice(self) -> tuple[np.ndarray, np.ndarray]:
        """The full cell lattice (obstacle cells included) and a mask of the cells in free space.

        ``targets`` are the free cells. The full lattice is map-independent (it only depends on the
        workspace size and spacing), so an agent that does not know the map may be given it.
        """
        pts = lattice_points(self.cfg)
        free = np.ones(len(pts), dtype=bool)
        if self.obstacle_union is not None:
            free = ~shapely.contains_xy(self.obstacle_union, pts[:, 0], pts[:, 1])
        return pts, free

    @cached_property
    def allowed(self):
        c = self.cfg.clearance
        inner = rect(c, c, self.cfg.width - c, self.cfg.height - c)
        if self.obstacle_union is None:
            return inner
        return inner.difference(inflate(self.obstacle_union, c))

    @cached_property
    def reachable(self):
        """Connected pieces of the planning free space that contain a start position."""
        tol = self.allowed.buffer(1e-6)
        keep = [p for p in polygon_parts(self.allowed)
                if any(p.buffer(1e-6).contains(Point(s)) for s in self.starts)]
        return shapely.unary_union(keep) if keep else shapely.Polygon()

    def _check_starts(self):
        cover = self.allowed.buffer(1e-6)
        for i, s in enumerate(self.starts):
            if not cover.contains(Point(s)):
                raise ValueError(f"start {i} at {s} is not in the planning free space "
                                 f"(clearance {self.cfg.clearance:.2f} m from obstacles/walls)")
        comps = {id(p) for p in polygon_parts(self.reachable)}
        if len(polygon_parts(self.reachable)) > 1:
            # starts in different components can never cooperate on one connected region
            raise ValueError("drone starts lie in disconnected parts of the free space")

    # ------------------------------------------------------------------ derived objects
    @property
    def n_agents(self) -> int:
        return len(self.starts)

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @cached_property
    def planner(self) -> PathPlanner:
        return PathPlanner(self.reachable)

    @cached_property
    def viewpoints(self):
        from mascov.viewpoints.generator import build_viewpoints
        return build_viewpoints(self)

    @property
    def coverable(self) -> np.ndarray:
        """Targets that some reachable viewpoint can see (the real coverage goal)."""
        return self.viewpoints.coverable

    @cached_property
    def vp_bound(self):
        return self.planner.bind(self.viewpoints.positions)

    @cached_property
    def sensor(self):
        from mascov.world.sensing import Sensor
        return Sensor(self)

    def view(self, known_map: bool = True) -> "WorldView":
        return WorldView(self, known_map=known_map)

    def unknown_view(self) -> "UnknownWorldView":
        return UnknownWorldView(self)

    # ------------------------------------------------------------------ queries
    def free_area(self) -> float:
        return self.bounds.area - (self.obstacle_union.area if self.obstacle_union is not None else 0.0)

    def summary(self) -> dict:
        return dict(name=self.name, n_agents=self.n_agents, n_obstacles=len(self.obstacles),
                    n_targets=self.n_targets, n_coverable=int(self.coverable.sum()),
                    n_viewpoints=len(self.viewpoints.positions), free_area=round(self.free_area(), 2))


def lattice_points(cfg: WorldConfig) -> np.ndarray:
    """Centres of the coverage lattice over the whole workspace (row-major, x fastest)."""
    h = cfg.target_spacing
    xs = np.arange(h / 2, cfg.width, h)
    ys = np.arange(h / 2, cfg.height, h)
    gx, gy = np.meshgrid(xs, ys)
    return np.c_[gx.ravel(), gy.ravel()]


def candidate_positions(cfg: WorldConfig) -> np.ndarray:
    """Candidate standing positions over the whole workspace (map-independent).

    These are the same lattice points ``build_viewpoints`` filters by the free space, so an explorer
    that has visited every useful one has covered everything the known-map viewpoints could.
    """
    s = cfg.viewpoint_spacing
    xs = np.arange(s / 2, cfg.width, s)
    ys = np.arange(s / 2, cfg.height, s)
    gx, gy = np.meshgrid(xs, ys)
    return np.c_[gx.ravel(), gy.ravel()]


class WorldView:
    """Read-only information handed to models at reset (no victims, no coverage ground truth).

    KNOWN-MAP interface: obstacles, targets, viewpoints and the path planner are all exposed.
    For the unknown-map interface see ``UnknownWorldView``.

    ``known_map=False`` (``SimConfig(hide_obstacles=True)``) is the hidden-obstacle variant written for CBBA:
    the obstacles are withheld and the planner starts with the empty workspace; a model learns obstacles by
    sensing (``Observation.new_obstacles``) and calls ``create_planner`` with the ids it knows. Targets and
    viewpoints are still exposed (their layout reveals where obstacles are), so this is weaker than the
    unknown-map interface, which hides those too.
    """

    def __init__(self, world: World, known_map: bool = True):
        self.cfg = world.cfg
        self.name = world.name
        self.bounds = world.bounds
        self.known_map = known_map
        self._world = world
        self.targets = world.targets.copy()
        self.starts = world.starts.copy()
        self.coverable = world.coverable.copy()
        self.viewpoints = world.viewpoints
        self.n_agents = world.n_agents
        if known_map:
            self.obstacles = list(world.obstacles)
            self.planner = world.planner
            self._vp_bound = world.vp_bound
        else:
            self.obstacles = []                        # learned by sensing
            self.planner, self._vp_bound = self.create_planner(())

    def get_obstacle(self, k: int) -> Polygon:
        """The polygon of a discovered obstacle (hidden-obstacle mode)."""
        return self._world.obstacles[k]

    def create_planner(self, known_obstacle_ids) -> tuple[PathPlanner, object]:
        """A PathPlanner over the free space given only the obstacles with these ids, plus its bound
        viewpoints. Keeps the free-space components that contain a start position."""
        c = self.cfg.clearance
        inner = rect(c, c, self.cfg.width - c, self.cfg.height - c)
        if not known_obstacle_ids:
            p = PathPlanner(inner)
            return p, p.bind(self.viewpoints.positions)
        union = prepared_union([self._world.obstacles[k] for k in sorted(known_obstacle_ids)])
        allowed = inner.difference(inflate(union, c))
        keep = [p for p in polygon_parts(allowed) if any(p.buffer(1e-6).contains(Point(s)) for s in self.starts)]
        p = PathPlanner(shapely.unary_union(keep) if keep else allowed)
        return p, p.bind(self.viewpoints.positions)

    def dist_to_viewpoints(self, pos) -> np.ndarray:
        """Shortest-path distance from pos to every candidate viewpoint, shape (V,)."""
        return self.planner.dist_to(np.asarray(pos, dtype=float), self._vp_bound)[0]

    def travel_time(self, a, b) -> np.ndarray:
        return self.planner.dist_matrix(a, b) / self.cfg.v_max


class UnknownWorldView:
    """What an agent may know about the world BEFORE it has sensed anything.

    Deliberately absent: obstacles, the target set (its holes would reveal obstacles), viewpoints,
    the coverable mask and the path planner. What is present is a function of the configuration
    alone, plus the start positions (assumed common knowledge at t=0; delete ``starts`` to drop that).

    * ``lattice``     : all coverage cells over the whole workspace, free or not (G, 2). A belief is
                        indexed by these cells.
    * ``candidates``  : candidate standing positions over the whole workspace (V, 2).
    * ``space_for``   : planning machinery for a set of KNOWN obstacles (unknown space is treated as free).
    """

    known_map = False

    def __init__(self, world: World):
        from mascov.planning.known_map import KnownSpaceCache
        self.cfg = world.cfg
        self.name = world.name
        self.bounds = world.bounds
        self.starts = world.starts.copy()
        self.n_agents = world.n_agents
        self.lattice = lattice_points(world.cfg)
        self.candidates = candidate_positions(world.cfg)
        self._cache = KnownSpaceCache(self)

    def space_for(self, obstacles):
        """Planning space for a dict {obstacle id: polygon} of known obstacles (cached, shared)."""
        return self._cache.get(obstacles)
