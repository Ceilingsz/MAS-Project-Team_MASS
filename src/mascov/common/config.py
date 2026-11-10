"""Shared configuration. Every layer imports its constants from here."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class WorldConfig:
    # Workspace (metres). Origin at the bottom-left corner.
    width: float = 12.0
    height: float = 12.0

    # Discretisation of the free space into coverage targets / candidate viewpoints.
    target_spacing: float = 0.5      # h: spacing of the target lattice
    viewpoint_spacing: float = 1.0   # spacing of the candidate-viewpoint lattice

    # Time (shared with the C-ADMM paper's setting).
    dt: float = 0.1                  # simulation / control step [s]
    horizon: int = 10                # MPC horizon in steps (control layer)

    # Drone: double integrator with body radius.
    v_max: float = 1.0               # [m/s]
    a_max: float = 2.0               # [m/s^2]
    drone_radius: float = 0.15       # r_b [m]
    plan_speed_ratio: float = 0.6    # planners fly at this fraction of v_max (see eps_track)

    # Sensing.
    r_sense: float = 1.5             # R_s [m]
    use_los: bool = True             # obstacles block sensing
    sense_slack: float = 0.1         # viewpoints are validated with R_s - slack

    # Tracking error bound fed to planners. With the shared tracker, the worst measured
    # deviation across the benchmark maps is ~0.26 m at plan_speed_ratio=0.6 (it was ~0.48 m at
    # 0.8: error grows quickly with speed because a_max limits how fast corners can be turned).
    # tests/test_sim.py asserts the measured error stays below this value.
    eps_track: float = 0.3
    plan_margin: float = 0.0         # any extra clearance on top of r_b + eps

    # Communication.
    r_comm: float = math.inf         # R_c [m]

    @property
    def d_safe(self) -> float:
        """Minimum centre-to-centre distance between two drones."""
        return 2.0 * self.drone_radius

    @property
    def clearance(self) -> float:
        """Distance planners must keep between a drone *centre* and any obstacle/wall."""
        return self.drone_radius + self.eps_track + self.plan_margin

    @property
    def plan_speed(self) -> float:
        return self.plan_speed_ratio * self.v_max

    @property
    def stopping_distance(self) -> float:
        return self.v_max ** 2 / (2.0 * self.a_max)

    def with_(self, **kw) -> "WorldConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SimConfig:
    max_time: float = 600.0          # [s]
    decision_period: float = 0.5     # model.step is called every this many seconds
    stop_when_covered: bool = True   # end the run at true full coverage
    known_map: bool = True           # unknown-map phase (P2/P3) is not implemented yet
    tracker_omega: float = 3.0       # tracker natural frequency [rad/s]
    log_every: int = 5               # time-series logging period in ticks
    safety: str = "none"             # "none" or "bvc": control-rate drone-drone safety filter
    brake_frac: float = 0.7          # BVC filter assumes drones can brake at this fraction of a_max
    seed: int = 0

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)
