"""The shared low-level tracker: every model's timed plan is executed by this controller.

Using one tracker for all models keeps comparisons fair, and its measured deviation from the
reference path is the epsilon the STL layer needs.
"""
from __future__ import annotations

import numpy as np

from mascov.common.types import Plan


class PlanTracker:
    """PD law on a time-parameterised reference (critically damped at natural frequency omega)."""

    def __init__(self, omega: float, a_max: float):
        self.kp = omega ** 2
        self.kd = 2.0 * omega
        self.a_max = a_max

    def accel(self, t: float, pos: np.ndarray, vel: np.ndarray,
              plan: Plan | None, hold_point: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (commanded acceleration, position error w.r.t. the reference)."""
        if plan is None:
            ref_p, ref_v = hold_point, np.zeros(2)
        else:
            ref_p, ref_v = plan.ref(t)
        err = ref_p - pos
        acc = self.kp * err + self.kd * (ref_v - vel)
        n = float(np.linalg.norm(acc))
        if n > self.a_max:
            acc = acc * (self.a_max / n)
        return acc, float(np.linalg.norm(err))

    @staticmethod
    def finished(t: float, pos: np.ndarray, vel: np.ndarray, plan: Plan | None,
                 pos_tol: float = 0.08, vel_tol: float = 0.08) -> bool:
        if plan is None:
            return True
        return (t >= plan.t_end and np.linalg.norm(pos - plan.end_point) <= pos_tol
                and np.linalg.norm(vel) <= vel_tol)
