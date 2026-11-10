"""Algorithms. One sub-folder per method; importing this package registers them all.

greedy+dmpc / frontier+dmpc need the OSQP solver (pip install -e ".[dmpc]"), so they are imported only if available.
"""
from models import greedy_single, greedy_centralised, greedy_uncoordinated, static_partition  # noqa: F401
from models import frontier_greedy  # noqa: F401  (unknown-map interface)

try:  # optional: D-MPC needs the OSQP solver (pip install osqp)
    from mascov.planning.dmpc import dmpc_available
    if dmpc_available():
        from models import greedy_dmpc  # noqa: F401
except Exception:  # pragma: no cover
    pass
