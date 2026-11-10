"""Describing experiments as data: which models, maps, team sizes, lever values and seeds to run.

An :class:`Experiment` expands into a flat list of :class:`Job` objects (one simulator run each). Lever keys say
what a lever changes:

    "n_agents"        number of drones
    "map"             benchmark map name
    "map_prior"       "known" or "unknown" (see below)
    "world.<field>"   a WorldConfig field, e.g. "world.r_comm", "world.yield_to_neighbors"
    "sim.<field>"     a SimConfig field, e.g. "sim.safety"
    "model.<kwarg>"   a keyword argument of the model factory, e.g. "model.horizon", "model.share"

``map_prior`` works for every model, whatever interface it was written for:
  * a model written for the unknown-map interface runs with the obstacles hidden ("unknown") or revealed at the
    start ("known": the same algorithm with the map given, which is how "what does not knowing the map cost" is
    measured);
  * a model written for the known-map interface only supports "known"; combinations with "unknown" are skipped.

A model that does not depend on some lever lists it in ``ModelSpec.ignores``; it is then run once (at the first
value of that lever) and drawn as a flat line in lever sweeps.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    label: str                                   # name in tables and plots (colours follow the label)
    name: str                                    # name in the model registry
    kwargs: dict = field(default_factory=dict)   # keyword arguments for the model factory
    role: str = "method"                         # "method" (a contender) or "reference" (floor / ceiling line)
    ignores: tuple = ()                          # lever keys this model does not depend on


@dataclass
class PlotSpec:
    kind: str                                    # "sweep" | "rate" | "curves" | "paired" | "tradeoff"
    file: str                                    # output file stem
    title: str = ""
    x: str | None = None                         # column on the x axis (a lever column, or a metric for "tradeoff")
    y: str | None = None                         # metric on the y axis
    per: str | None = "map"                      # one figure per value of this column (None: one figure)
    where: dict = field(default_factory=dict)    # keep only rows whose columns equal these values
    options: dict = field(default_factory=dict)  # kind-specific extras (see plots.py)


@dataclass
class Experiment:
    name: str
    title: str
    question: str
    models: list[ModelSpec]
    maps: list[str] = field(default_factory=lambda: ["rooms_and_door"])
    n_agents: list[int] = field(default_factory=lambda: [4])
    levers: dict = field(default_factory=dict)   # lever key -> list of values (full grid)
    seeds: list[int] = field(default_factory=lambda: list(range(10)))
    size: float = 12.0
    max_time: float = 600.0
    world: dict = field(default_factory=dict)    # fixed WorldConfig overrides
    sim: dict = field(default_factory=dict)      # fixed SimConfig overrides
    map_prior: str = "known"                     # default map prior (a lever "map_prior" overrides it)
    plots: list[PlotSpec] = field(default_factory=list)
    quick: dict = field(default_factory=dict)    # overrides applied by --quick (smoke run): seeds, maps, n_agents, ...


@dataclass
class Job:
    exp: str
    label: str
    model: str
    kwargs: dict
    role: str
    map: str
    n_agents: int
    seed: int
    size: float
    max_time: float
    map_prior: str
    world: dict
    sim: dict
    levers: dict                                 # lever key -> value, as reported in the result row (None: not applicable)

    @property
    def job_id(self) -> str:
        blob = json.dumps(dict(exp=self.exp, label=self.label, model=self.model, kwargs=self.kwargs, map=self.map,
                               n=self.n_agents, seed=self.seed, size=self.size, T=self.max_time, prior=self.map_prior,
                               world=self.world, sim=self.sim, levers=self.levers), sort_keys=True, default=_enc)
        return hashlib.md5(blob.encode()).hexdigest()[:12]


def _enc(o):
    if isinstance(o, float) and math.isinf(o):
        return "inf" if o > 0 else "-inf"
    return str(o)


def lever_column(key: str) -> str:
    """Column name of a lever in result rows: 'world.r_comm' -> 'world_r_comm'."""
    return key.replace(".", "_")


def expand(exp: Experiment, *, seeds=None, maps=None, n_agents=None) -> list[Job]:
    """All runs of an experiment: maps x team sizes x lever grid x models x seeds."""
    seeds = list(exp.seeds if seeds is None else seeds)
    maps = list(exp.maps if maps is None else maps)
    sizes = list(exp.n_agents if n_agents is None else n_agents)
    lever_keys = list(exp.levers)
    grids = list(itertools.product(*[exp.levers[k] for k in lever_keys])) if lever_keys else [()]
    jobs: list[Job] = []
    for map_name, n, combo, spec, seed in itertools.product(maps, sizes, grids, exp.models, seeds):
        values = dict(zip(lever_keys, combo))
        reported = dict(values)
        skip = False
        for k in spec.ignores:                       # run once, at the lever's first value
            if k in values:
                skip = skip or values[k] != exp.levers[k][0]
                reported[k] = None
        if skip:
            continue
        world, sim, kwargs = dict(exp.world), dict(exp.sim), dict(spec.kwargs)
        job_map, job_n, prior = map_name, n, exp.map_prior
        for k, v in values.items():
            if k in spec.ignores:
                continue
            if k == "n_agents":
                job_n = v
            elif k == "map":
                job_map = v
            elif k == "map_prior":
                prior = v
            elif k.startswith("world."):
                world[k[6:]] = v
            elif k.startswith("sim."):
                sim[k[4:]] = v
            elif k.startswith("model."):
                kwargs[k[6:]] = v
            else:
                raise KeyError(f"unknown lever key '{k}' (use n_agents, map, map_prior, world.*, sim.* or model.*)")
        jobs.append(Job(exp=exp.name, label=spec.label, model=spec.name, kwargs=kwargs, role=spec.role, map=job_map,
                        n_agents=job_n, seed=seed, size=exp.size, max_time=exp.max_time, map_prior=prior,
                        world=world, sim=sim, levers=reported))
    return jobs
