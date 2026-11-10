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
        self.targets = self._sample_targets()
        self._check_starts()

    # ------------------------------------------------------------------ construction
    def _sample_targets(self) -> np.ndarray:
        h = self.cfg.target_spacing
        xs = np.arange(h / 2, self.cfg.width, h)
        ys = np.arange(h / 2, self.cfg.height, h)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.c_[gx.ravel(), gy.ravel()]
        if self.obstacle_union is not None:
            inside = shapely.contains_xy(self.obstacle_union, pts[:, 0], pts[:, 1])
            pts = pts[~inside]
        return pts

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

    # ------------------------------------------------------------------ queries
    def free_area(self) -> float:
        return self.bounds.area - (self.obstacle_union.area if self.obstacle_union is not None else 0.0)

    def summary(self) -> dict:
        return dict(name=self.name, n_agents=self.n_agents, n_obstacles=len(self.obstacles),
                    n_targets=self.n_targets, n_coverable=int(self.coverable.sum()),
                    n_viewpoints=len(self.viewpoints.positions), free_area=round(self.free_area(), 2))


class WorldView:
    """Read-only information handed to models at reset (no victims, no coverage ground truth).

    If known_map=True (P0/P1), all obstacles, targets, viewpoints, and the path planner are exposed.
    If known_map=False (P2/P3), obstacles are initially hidden, and the path planner starts with the
    empty workspace. Drones discover obstacles as they explore and can call create_planner().
    """

    def __init__(self, world: World, known_map: bool = True):
        self.cfg = world.cfg
        self.name = world.name
        self.bounds = world.bounds
        self.known_map = known_map
        self._world = world
        self.starts = world.starts.copy()
        self.n_agents = world.n_agents

        if known_map:
            self.obstacles = list(world.obstacles)
            self.targets = world.targets.copy()
            self.coverable = world.coverable.copy()
            self.viewpoints = world.viewpoints
            self.planner = world.planner
            self._vp_bound = world.vp_bound
        else:
            self.obstacles = []  # Initially unknown
            self.targets = world.targets.copy()
            self.coverable = world.coverable.copy()
            self.viewpoints = world.viewpoints
            c = self.cfg.clearance
            inner = rect(c, c, self.cfg.width - c, self.cfg.height - c)
            self.allowed = inner
            self.planner = PathPlanner(inner)
            self._vp_bound = self.planner.bind(self.viewpoints.positions)

    def get_obstacle(self, k: int) -> Polygon:
        """Retrieve the polygon for a discovered obstacle."""
        return self._world.obstacles[k]

    def create_planner(self, known_obstacle_ids: set[int] | list[int]) -> tuple[PathPlanner, any]:
        """Creates an updated PathPlanner and BoundPoints given discovered obstacles."""
        c = self.cfg.clearance
        inner = rect(c, c, self.cfg.width - c, self.cfg.height - c)
        if not known_obstacle_ids:
            p = PathPlanner(inner)
            return p, p.bind(self.viewpoints.positions)
        obs_polys = [self._world.obstacles[k] for k in sorted(known_obstacle_ids)]
        union = prepared_union(obs_polys)
        allowed = inner.difference(inflate(union, c))
        # Keep components that contain a start position
        tol = allowed.buffer(1e-6)
        keep = [p for p in polygon_parts(allowed)
                if any(p.buffer(1e-6).contains(Point(s)) for s in self.starts)]
        reachable = shapely.unary_union(keep) if keep else allowed
        p = PathPlanner(reachable)
        return p, p.bind(self.viewpoints.positions)

    def dist_to_viewpoints(self, pos) -> np.ndarray:
        """Shortest-path distance from pos to every candidate viewpoint, shape (V,)."""
        return self.planner.dist_to(np.asarray(pos, dtype=float), self._vp_bound)[0]

    def travel_time(self, a, b) -> np.ndarray:
        return self.planner.dist_matrix(a, b) / self.cfg.v_max
