import numpy as np
import pytest
from shapely.geometry import Polygon

from mascov.geometry.pathing import PathPlanner
from mascov.geometry.polygons import halfspaces, inflate, is_convex, rect, rotated_rect
from mascov.geometry.visibility import clear_los_mask, prepared_union


def test_halfspaces_describe_the_polygon():
    poly = rotated_rect(5, 5, 4, 2, 0.5)
    H, b = halfspaces(poly)
    assert np.allclose(np.linalg.norm(H, axis=1), 1.0)
    assert np.all(H @ np.array(poly.centroid.coords[0]) <= b)                     # centre inside
    for v in np.asarray(poly.exterior.coords)[:-1]:
        assert np.all(H @ v <= b + 1e-9)                                           # vertices on/inside
    assert not np.all(H @ np.array([20.0, 20.0]) <= b)                             # far point outside


def test_halfspaces_reject_nonconvex():
    l_shape = Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])
    assert not is_convex(l_shape)
    with pytest.raises(ValueError):
        halfspaces(l_shape)


def test_los_blocked_by_wall_and_clear_around_it():
    wall = prepared_union([rect(5, 0, 5.4, 8)])
    pts = np.array([[8.0, 4.0], [8.0, 14.0], [2.0, 4.0]])
    clear = clear_los_mask([2.0, 4.0], pts, wall)
    assert list(clear) == [False, True, True]       # behind wall blocked; over the top clear; self clear


def test_los_grazing_a_corner_does_not_block():
    box = prepared_union([rect(2, 2, 4, 4)])
    # segment runs exactly along the box's bottom edge
    assert clear_los_mask([0.0, 2.0], [[6.0, 2.0]], box)[0]
    # segment touches only the corner point (4,4)
    assert clear_los_mask([2.0, 6.0], [[6.0, 2.0]], box)[0]
    # segment through the interior is blocked
    assert not clear_los_mask([0.0, 3.0], [[6.0, 3.0]], box)[0]


def test_planner_straight_line_and_detour():
    free = rect(0, 0, 10, 10).difference(inflate(rect(4.5, 0, 5.5, 8), 0.0))
    pl = PathPlanner(free)
    assert pl.distance([1, 1], [1, 9]) == pytest.approx(8.0)                       # no obstacle in the way
    d = pl.distance([1, 1], [9, 1])
    # must go over the top corners (4.5,8) and (5.5,8)
    expect = np.hypot(3.5, 7) + 1.0 + np.hypot(3.5, 7)
    assert d == pytest.approx(expect, rel=1e-6)
    path = pl.path([1, 1], [9, 1])
    assert np.allclose(path[0], [1, 1]) and np.allclose(path[-1], [9, 1])
    assert sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1)) == pytest.approx(expect, rel=1e-6)


def test_planner_snaps_points_outside_free_space():
    free = rect(0, 0, 10, 10).difference(rect(4.5, 0, 5.5, 8))
    pl = PathPlanner(free)
    inside_wall_margin = np.array([4.4 - 0.5, 4.0])         # left of the free region boundary is fine
    outside = np.array([4.5, 4.0])                          # on the wall boundary
    near_wall = np.array([4.6, 4.0])                        # 0.1 inside the wall -> outside free space
    d = pl.distance(near_wall, [1, 1])
    assert np.isfinite(d)
    assert d >= pl.distance([4.5, 4.0], [1, 1])             # includes the snap distance
    path = pl.path(near_wall, [1, 1])
    assert np.allclose(path[0], near_wall)


def test_planner_unreachable_is_inf():
    wall = rect(4, -1, 6, 11)                                # splits the free space completely
    free = rect(0, 0, 10, 10).difference(wall)
    pl = PathPlanner(free)
    assert not np.isfinite(pl.distance([1, 1], [9, 1]))
    assert pl.path([1, 1], [9, 1]) is None
