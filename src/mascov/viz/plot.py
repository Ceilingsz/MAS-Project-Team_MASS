"""Matplotlib plots: the world, drone paths, and coverage curves."""
from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon


def plot_world(world, covered=None, positions=None, paths=None, show_viewpoints=False,
               show_sensing=True, ax=None, title=None):
    ax = ax or plt.subplots(figsize=(6.5, 6.5))[1]
    cfg = world.cfg
    ax.set_xlim(0, cfg.width), ax.set_ylim(0, cfg.height), ax.set_aspect("equal")
    for o in world.obstacles:
        ax.add_patch(MplPolygon(np.asarray(o.exterior.coords), closed=True, fc="#888780", ec="#444441", zorder=2))
    T = world.targets
    cov = np.zeros(len(T), bool) if covered is None else np.asarray(covered, bool)
    ok = world.coverable
    ax.scatter(*T[~cov & ok].T, s=6, c="#B4B2A9", zorder=3)
    ax.scatter(*T[cov].T, s=9, c="#1D9E75", zorder=3)
    if (~ok).any():
        ax.scatter(*T[~ok].T, s=8, c="#E24B4A", marker="x", zorder=3)
    if show_viewpoints:
        ax.scatter(*world.viewpoints.positions.T, s=14, facecolors="none", edgecolors="#534AB7", zorder=4)
    for i, p in enumerate(world.starts):
        ax.plot(*p, marker="s", ms=5, c="k", zorder=5)
    if paths is not None:
        for i, pa in enumerate(paths):
            pa = np.asarray(pa)
            ax.plot(pa[:, 0], pa[:, 1], lw=1.2, zorder=4, label=f"drone {i}")
        ax.legend(loc="upper right", fontsize=8)
    if positions is not None:
        for p in np.atleast_2d(positions):
            ax.plot(*p, "o", c="k", ms=6, zorder=6)
            if show_sensing:
                ax.add_patch(Circle(p, cfg.r_sense, fill=False, ls="--", ec="#0F6E56", lw=0.8, zorder=5))
    ax.set_title(title or world.name)
    return ax


def plot_coverage_curves(results, ax=None):
    """results: iterable of RunResult. Plots true coverage vs time."""
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    for r in results:
        ax.plot(r.log.column("t"), r.log.column("coverage"), label=f"{r.model} (N={r.n_agents})")
    ax.set_xlabel("time [s]"), ax.set_ylabel("fraction covered"), ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3), ax.legend()
    return ax
