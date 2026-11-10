"""Algorithms. One sub-folder per method; importing this package registers them all.

stl_single / stl_team need scipy >= 1.9 (it bundles the HiGHS MILP solver), so they are imported only if available.
"""
from models import greedy_single, greedy_centralised, greedy_uncoordinated, static_partition  # noqa: F401
from models import frontier_greedy  # noqa: F401  (unknown-map interface)

try:  # optional: D-MPC needs the OSQP solver (pip install osqp)
    from mascov.planning.dmpc import dmpc_available
    if dmpc_available():
        from models import greedy_dmpc  # noqa: F401
except Exception:  # pragma: no cover
    pass

try:  # optional: formal (STL) planners
    from mascov.stl.planner import stl_available
    if stl_available():
        from models import stl_single, stl_team  # noqa: F401
except Exception:  # pragma: no cover - never let an optional dependency break `import models`
    pass
