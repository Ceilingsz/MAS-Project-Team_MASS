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

    # Drone: single integrator (p' = u, the velocity is the control input) with body radius.
    v_max: float = 1.0               # [m/s]
    drone_radius: float = 0.15       # r_b [m]
    plan_speed_ratio: float = 0.6    # planners fly at this fraction of v_max (see eps_track)

    # Sensing.
    r_sense: float = 1.5             # R_s [m]
    use_los: bool = True             # obstacles block sensing
    sense_slack: float = 0.1         # viewpoints are validated with R_s - slack

    # Tracking error bound fed to planners. The shared tracker is a proportional law with velocity
    # feedforward on a single integrator, so the error should be far smaller than the 0.3 m that was
    # measured for the earlier double-integrator model. It has NOT been re-measured: check
    # RunResult.max_tracking_error on the benchmark maps before lowering this value.
    # tests/test_sim.py asserts the measured error stays below it.
    eps_track: float = 0.3
    plan_margin: float = 0.0         # any extra clearance on top of r_b + eps

    # Communication and neighbour sensing. Two separate ranges, because they answer different questions:
    #   r_comm     : messages reach drones at most this far away (the lever of the communication experiments)
    #   r_neighbor : a drone senses the position and velocity of drones at most this far away; collision
    #                avoidance (the baselines' navigator, the BVC filter) works from this. None means "same as
    #                r_comm", which is the original behaviour. Set it (for example to math.inf) to sweep
    #                r_comm without also changing how well drones see each other.
    r_comm: float = math.inf         # R_c [m]
    r_neighbor: float | None = None  # [m] None: same as r_comm

    # The baselines' local navigator yields to and sidesteps other drones (a heuristic, not a guarantee).
    # False: drones ignore each other entirely, which is the "no safety at all" baseline.
    yield_to_neighbors: bool = True

    @property
    def neighbor_radius(self) -> float:
        """Range at which a drone senses a neighbour's position and velocity."""
        return self.r_comm if self.r_neighbor is None else self.r_neighbor

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

    def with_(self, **kw) -> "WorldConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SimConfig:
    max_time: float = 600.0          # [s]
    decision_period: float = 0.5     # model.step is called every this many seconds
    stop_when_covered: bool = True   # end the run at true full coverage
    known_map: bool = True           # True: models get the full map (WorldView). False: models get only an
                                     # UnknownWorldView and learn obstacles by sensing (MapBelief)
    reveal_obstacles_at_start: bool = False   # unknown-map interface only: seed every belief with ALL obstacles.
                                     # The map is then known but coverage is not: the "same algorithm with a
                                     # known map" reference for measuring what not knowing the map costs
    hide_obstacles: bool = False     # known-map interface only: hand models a WorldView(known_map=False) with the
                                     # obstacles withheld (targets and viewpoints still given). The hidden-obstacle
                                     # mode CBBA was written for; weaker than known_map=False, which hides everything
    tracker_gain: float = 3.0        # tracker position gain [1/s]: error decays as exp(-gain * t)
    log_every: int = 5               # time-series logging period in ticks
    milestones: tuple = (0.5, 0.8, 0.9, 0.95, 0.99, 1.0)   # coverage fractions whose first-reached time is recorded
    safety: str = "none"             # "none" or "bvc": control-rate drone-drone safety filter
    seed: int = 0

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)
