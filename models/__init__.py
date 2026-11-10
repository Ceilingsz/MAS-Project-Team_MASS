"""Algorithms. One sub-folder per method; importing this package registers them all.

stl_single needs gurobipy + the STLPlanning checkout, so it is imported only if available.
"""
from models import greedy_single, greedy_centralised, greedy_uncoordinated, static_partition, cbba  # noqa: F401

try:  # optional: formal (STL) planners
    from mascov.stl.planner import stl_available
    if stl_available():
        from models import stl_single, stl_team  # noqa: F401
except Exception:  # pragma: no cover - never let an optional dependency break `import models`
    pass
