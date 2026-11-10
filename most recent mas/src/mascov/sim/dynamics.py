"""Single-integrator drone dynamics: the control input is the velocity."""
from __future__ import annotations

import numpy as np


def step_single_integrator(pos: np.ndarray, vel_cmd: np.ndarray, dt: float,
                           v_max: float) -> tuple[np.ndarray, np.ndarray]:
    """One exact step of p' = u with ||u|| <= v_max. Returns (new position, applied velocity).

    The applied velocity is what the drone actually moved with during the step; it is what neighbours
    observe as this drone's velocity.
    """
    v = np.asarray(vel_cmd, dtype=float)
    s = float(np.linalg.norm(v))
    if s > v_max:
        v = v * (v_max / s)
    return pos + v * dt, v
