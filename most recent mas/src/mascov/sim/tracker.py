"""The shared low-level tracker: every model's timed plan is executed by this controller.

Using one tracker for all models keeps comparisons fair, and its measured deviation from the
reference path is the epsilon (eps_track) every planner's clearance assumes.
"""
from __future__ import annotations

import numpy as np

from mascov.common.types import Plan


class PlanTracker:
    """Proportional law with velocity feedforward on a time-parameterised reference.

    The drone is a single integrator (p' = u), so the command is a velocity:

        u = v_ref + gain * (p_ref - p),        clipped to ||u|| <= v_max

    Along a reference moving at constant velocity the position error obeys e' = -gain * e, so it decays
    exponentially and stays at zero once it is zero.
    """

    def __init__(self, gain: float, v_max: float):
        self.gain = gain
        self.v_max = v_max

    def velocity(self, t: float, pos: np.ndarray, plan: Plan | None,
                 hold_point: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (commanded velocity, position error w.r.t. the reference)."""
        if plan is None:
            ref_p, ref_v = hold_point, np.zeros(2)
        else:
            ref_p, ref_v = plan.ref(t)
        err = ref_p - pos
        u = ref_v + self.gain * err
        n = float(np.linalg.norm(u))
        if n > self.v_max:
            u = u * (self.v_max / n)
        return u, float(np.linalg.norm(err))

    @staticmethod
    def finished(t: float, pos: np.ndarray, vel: np.ndarray, plan: Plan | None,
                 pos_tol: float = 0.08, vel_tol: float = 0.08) -> bool:
        if plan is None:
            return True
        return (t >= plan.t_end and np.linalg.norm(pos - plan.end_point) <= pos_tol
                and np.linalg.norm(vel) <= vel_tol)
