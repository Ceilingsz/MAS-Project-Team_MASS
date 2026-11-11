"""Algorithms. One sub-folder per method; importing this package registers them all.

greedy+dmpc / frontier+dmpc need the OSQP solver (pip install -e ".[dmpc]"), so they are imported only if available.
"""
from models import greedy_single, greedy_centralised, greedy_uncoordinated, static_partition  # noqa: F401
from models import frontier_greedy  # noqa: F401  (unknown-map interface)
from models import cbba  # noqa: F401  (CBBA task allocation)
from models import greedy_explorer  # noqa: F401  (unknown-map greedy baseline)
from models import geodesic_partition  # noqa: F401  (decentralised geodesic Voronoi)

try:  # optional: the learned goal layer needs PyTorch (pip install -e ".[marl]")
    import torch  # noqa: F401
    from models import marl_explorer  # noqa: F401  (MAPPO goal layer)
except ImportError:  # pragma: no cover
    pass

try:  # optional: D-MPC needs the OSQP solver (pip install osqp)
    from mascov.planning.dmpc import dmpc_available
    if dmpc_available():
        from models import greedy_dmpc  # noqa: F401
        from models import mix_dmpc  # noqa: F401  (goal layer x D-MPC flying)
except Exception:  # pragma: no cover
    pass
