"""Double-integrator drone dynamics."""
from __future__ import annotations

import numpy as np


def step_double_integrator(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, dt: float,
                           v_max: float, a_max: float) -> tuple[np.ndarray, np.ndarray]:
    """One exact step of p'' = u with ||u|| <= a_max and ||v|| <= v_max."""
    n = float(np.linalg.norm(acc))
    if n > a_max:
        acc = acc * (a_max / n)
    new_pos = pos + vel * dt + 0.5 * acc * dt * dt
    new_vel = vel + acc * dt
    s = float(np.linalg.norm(new_vel))
    if s > v_max:
        new_vel = new_vel * (v_max / s)
    return new_pos, new_vel
