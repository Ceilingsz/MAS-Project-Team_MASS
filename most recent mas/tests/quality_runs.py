"""Scenarios and cached runs for the model quality suite (tests/test_model_quality.py).

Every registered model is run on the SAME worlds and judged by the SAME criteria, so the suite measures the
quality of each model rather than checking that each model passes the tests written for it. Each
(model, scenario) pair is simulated once per test session and shared by every criterion.

One world configuration for all models: eps_track = 0.45 and 2 m start spacing. The STL planners need both
(their tracking-error assumption and their joint separation at t = 0); every other model runs fine with them,
and using them for everyone keeps the comparison fair.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import models  # noqa: F401  (registers every available model)
from mascov.common.config import SimConfig, WorldConfig
from mascov.common.model import get_model, registered_models
from mascov.maps.benchmarks import make_world
from mascov.sim.simulator import Simulator

CFG = WorldConfig(eps_track=0.45)
START = dict(start_spacing=2.0, start_per_row=2)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    interface: str = "known"          # "known" or "unknown" (SimConfig.known_map)
    single_only: bool = False         # written for one drone
    centralised: bool = False         # one planner sees everything (communication range does not apply)
    kwargs: tuple = ()


# Every model the repository defines. A model missing from the registry (e.g. its optional solver is not
# installed) is skipped, not failed. A registered model missing from this table fails test_every_model_is_graded.
MODELS: dict[str, ModelInfo] = {m.name: m for m in [
    ModelInfo("greedy_single", single_only=True),
    ModelInfo("greedy_centralised", centralised=True),
    ModelInfo("greedy_uncoordinated"),
    ModelInfo("static_partition"),
    ModelInfo("greedy+dmpc"),
    ModelInfo("frontier_greedy", interface="unknown"),
    ModelInfo("frontier_greedy_shared", interface="unknown"),
    ModelInfo("stl_single", single_only=True),
    ModelInfo("stl_team", centralised=True),
]}

REFERENCE = "greedy_centralised"      # the efficiency yardstick


@dataclass(frozen=True)
class Scenario:
    name: str
    map: str
    n: int
    seed: int = 0
    world: tuple = ()                 # WorldConfig overrides, as ((field, value), ...)
    max_time: float = 900.0
    description: str = ""

    def cfg(self) -> WorldConfig:
        return CFG.with_(**dict(self.world))


SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario("solo", "rooms_and_door", 1, description="one drone, two rooms joined by a door"),
    Scenario("team", "rooms_and_door", 3, description="three drones through one door"),
    Scenario("clutter", "cluttered", 3, seed=1, description="three drones among many obstacles"),
    Scenario("low_comm", "rooms_and_door", 3, world=(("r_comm", 1.5),),
             description="three drones; messages and neighbour sensing reach only 1.5 m (= the sensing radius)"),
]}


def applies(model: str, scenario: str) -> str | None:
    """None if the model is graded on this scenario, else the reason it is not."""
    info, sc = MODELS[model], SCENARIOS[scenario]
    if info.single_only and sc.n > 1:
        return "single-drone model"
    if sc.name == "low_comm" and info.centralised:
        return "centralised: a communication range does not apply"
    return None


# Cases that take far longer than the rest; they run only with ``pytest --slow`` (see tests/conftest.py).
SLOW = {("stl_team", "clutter"): "joint MILP for 3 drones among many obstacles: about 12 min"}


def available(model: str) -> str | None:
    if model not in registered_models():
        return f"{model} is not registered (optional dependency missing?)"
    return None


@dataclass
class Run:
    result: object
    world: object
    paths: list = field(default_factory=list)


_CACHE: dict[tuple, Run] = {}


def run(model: str, scenario: str, *, max_time: float | None = None, record_paths: bool = False) -> Run:
    """Simulate once and cache. Runs on until every drone believes it is done (or max_time), so both
    'when was the world covered' and 'when did the drones notice' are measured in the same run."""
    sc = SCENARIOS[scenario]
    key = (model, scenario, max_time, record_paths)
    if key not in _CACHE:
        info = MODELS[model]
        w = make_world(sc.map, n_agents=sc.n, seed=sc.seed, cfg=sc.cfg(), **START)
        sim = Simulator(w, get_model(model, **dict(info.kwargs)),
                        SimConfig(max_time=max_time or sc.max_time, stop_when_covered=False,
                                  known_map=info.interface == "known", seed=sc.seed))
        r = sim.run(record_paths=record_paths)
        _CACHE[key] = Run(r, w, sim.paths or [])
    return _CACHE[key]


def summary(r) -> dict:
    """The numbers the criteria look at, for messages and calibration."""
    done = [b for b in r.believed_done if b is not None]
    return dict(completed=r.completed, t=r.coverage_time, cov=round(r.final_coverage, 4),
                pair_events=r.pair_events, obstacle_events=r.obstacle_events,
                min_pair=None if not math.isfinite(r.min_pair_dist) else round(r.min_pair_dist, 3),
                min_clear=round(r.min_clearance, 3), max_track=round(max(r.max_tracking_error), 3),
                n_done=len(done), first_done=min(done) if done else None, last_done=max(done) if done else None,
                msgs=r.n_messages)
