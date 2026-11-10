"""The experiments of the study plan, as data. Run them with scripts/run_experiments.py, draw them with
scripts/plot_experiments.py.

Models that are not implemented yet (CBBA, D-MPC, C-ADMM, the integrated method) are listed under the names the
team agreed on; the runner skips them (and says so in skipped.txt) until a model with that name is registered
with ``mascov.common.model.register``. No file here needs to change when a model arrives. Expected names and options:

    cbba                    CBBA allocation, plain tracker            kwargs: share (unknown map: share beliefs)
    greedy+dmpc             greedy goals + D-MPC                      kwargs: coupling (bool), horizon
    greedy+cadmm            greedy goals + C-ADMM                     kwargs: horizon, iterations
    cbba+dmpc, cbba+cadmm   CBBA allocation + the planner above
    coverage_mpc            integrated coverage-driven MPC            kwargs: horizon
    (optional extras)       utility_greedy, orca, ...

Greedy is the existing uncoordinated greedy (it picks the viewpoint with the most newly seen targets per unit of
travel time). Reference lines are existing models: static partition, centralised greedy.
"""
from __future__ import annotations

import math

from mascov.experiments.spec import Experiment, ModelSpec, PlotSpec

INF = math.inf

GREEDY = ModelSpec("Greedy", "greedy_uncoordinated")
CBBA = ModelSpec("CBBA", "cbba")
DMPC = ModelSpec("Greedy + D-MPC", "greedy+dmpc", dict(coupling=True))
DMPC_OFF = ModelSpec("Greedy + D-MPC (coupling off)", "greedy+dmpc", dict(coupling=False))
CADMM = ModelSpec("Greedy + C-ADMM", "greedy+cadmm")
CBBA_DMPC = ModelSpec("CBBA + D-MPC", "cbba+dmpc", dict(coupling=True))
CBBA_CADMM = ModelSpec("CBBA + C-ADMM", "cbba+cadmm")
INTEGRATED = ModelSpec("Coverage MPC", "coverage_mpc")
PARTITION = ModelSpec("Static partition", "static_partition", role="reference")
CENTRAL = ModelSpec("Centralised greedy", "greedy_centralised", role="reference")
FRONTIER = ModelSpec("Frontier greedy", "frontier_greedy")
FRONTIER_SHARED = ModelSpec("Frontier greedy (shared)", "frontier_greedy_shared")
CBBA_PRIVATE = ModelSpec("CBBA (private belief)", "cbba", dict(share=False))
CBBA_SHARED = ModelSpec("CBBA (shared belief)", "cbba", dict(share=True))

MAPS = ["sparse", "rooms_and_door"]
FULL_GRID = [GREEDY, CBBA, DMPC_OFF, DMPC, CADMM, CBBA_DMPC, CBBA_CADMM]

T95 = "t95"


def _sweeps(x, ys, stem, title):
    return [PlotSpec("sweep", f"{stem}_{y}", f"{title}: {y}", x=x, y=y) for y in ys]


EXPERIMENTS: dict[str, Experiment] = {}


def _add(e: Experiment) -> Experiment:
    EXPERIMENTS[e.name] = e
    return e


_add(Experiment(
    "E0", "Baselines under single-integrator dynamics",
    "Reference lines: how do the existing baselines behave now? (rerun of the earlier baselines)",
    models=[GREEDY, PARTITION, CENTRAL], maps=["empty", "sparse", "corridor", "rooms_and_door", "cluttered"],
    n_agents=[1, 2, 4], seeds=[0, 1, 2],
    plots=_sweeps("n_agents", ["t95", "mean_visits", "collisions"], "e0", "Baselines")
    + [PlotSpec("curves", "e0_coverage", "Baselines: coverage over time", where={"n_agents": 4})],
    quick=dict(maps=["rooms_and_door"], n_agents=[2], seeds=[0])))

_add(Experiment(
    "E1", "Cooperation gain with team size", "How much does cooperation gain as the team grows?",
    models=[GREEDY, CBBA, DMPC, PARTITION, CENTRAL], maps=MAPS, n_agents=[2, 4, 6, 8],
    plots=_sweeps("n_agents", ["t95", "mean_visits", "path_len_mean"], "e1", "Team size")
    + [PlotSpec("curves", "e1_coverage", "Coverage over time", where={"n_agents": 6})],
    quick=dict(maps=["rooms_and_door"], n_agents=[2, 4], seeds=[0, 1])))

_add(Experiment(
    "E2", "Communication radius", "How little communication is enough?",
    models=[CBBA, DMPC, CADMM,
            ModelSpec("Greedy", "greedy_uncoordinated", ignores=("world.r_comm",)),          # sends nothing: flat line
            ModelSpec("Centralised greedy", "greedy_centralised", role="reference", ignores=("world.r_comm",))],
    maps=MAPS, n_agents=[6],
    levers={"world.r_comm": [2.0, 4.0, 6.0, 8.0, 12.0, INF]},
    world={"r_neighbor": INF},                      # neighbour sensing stays unlimited: only message reach changes
    plots=_sweeps("world_r_comm", ["t95", "mean_visits", "msgs_per_drone_per_s"], "e2", "Communication radius"),
    quick=dict(maps=["rooms_and_door"], seeds=[0, 1])))

_add(Experiment(
    "E3", "Cost of not knowing the map", "What does not knowing the map cost?",
    models=[FRONTIER, CBBA], maps=MAPS, n_agents=[4], levers={"map_prior": ["known", "unknown"]},
    plots=[PlotSpec("paired", "e3_cost", "Unknown against known map", y="t95", per=None,
                    options=dict(pairs=[("unknown", "known")], label_col="map_prior",
                                 match=("label", "map", "n_agents", "seed"))),
           PlotSpec("sweep", "e3_t95", "Map prior", x="map_prior", y="t95"),
           PlotSpec("curves", "e3_coverage", "Coverage over time", where={"map_prior": "unknown"})],
    quick=dict(maps=["rooms_and_door"], seeds=[0, 1])))

_add(Experiment(
    "E4", "Sharing what drones see", "Does sharing beliefs help on an unknown map?",
    models=[FRONTIER, FRONTIER_SHARED, CBBA_PRIVATE, CBBA_SHARED], maps=MAPS, n_agents=[4], map_prior="unknown",
    plots=[PlotSpec("sweep", "e4_t95", "Belief sharing: time to 95%", x="n_agents", y="t95"),
           PlotSpec("curves", "e4_coverage", "Coverage over time"),
           PlotSpec("curves", "e4_known", "Known by at least one drone", per="map", options=dict(column="team_believed"))],
    quick=dict(maps=["rooms_and_door"], seeds=[0, 1])))

_add(Experiment(
    "E5", "How unsafe is no safety", "How many collisions without any safety, and what does heuristic yielding add?",
    models=[GREEDY, CBBA], maps=MAPS, n_agents=[2, 4, 6, 8], levers={"world.yield_to_neighbors": [False, True]},
    plots=_sweeps("n_agents", ["collisions", "min_pair_dist"], "e5", "No safety")
    + [PlotSpec("rate", "e5_safe_complete", "Safe-complete rate", x="n_agents", y="safe_complete",
                where={"world_yield_to_neighbors": False})],
    quick=dict(maps=["rooms_and_door"], n_agents=[2, 4], seeds=[0, 1])))

_add(Experiment(
    "S1", "Safety add-on: BVC", "How much safety does the BVC filter add to a method with no safety, and at what cost?",
    models=[GREEDY, CBBA], maps=MAPS, n_agents=[4, 8], levers={"sim.safety": ["none", "bvc"]},
    world={"yield_to_neighbors": False},
    plots=[PlotSpec("sweep", "s1_collisions", "BVC: collisions", x="sim_safety", y="collisions"),
           PlotSpec("sweep", "s1_t95", "BVC: time to 95%", x="sim_safety", y="t95"),
           PlotSpec("rate", "s1_safe", "BVC: safe-complete rate", x="sim_safety", y="safe_complete")],
    quick=dict(maps=["rooms_and_door"], n_agents=[4], seeds=[0, 1])))

_add(Experiment(
    "E6", "Which trajectory planner is safest and cheapest",
    "Across the allocation x trajectory grid: safety against time, messages and compute?",
    models=FULL_GRID, maps=MAPS, n_agents=[4, 8],
    plots=[PlotSpec("tradeoff", "e6_time_vs_collisions", "Time against collisions", x="t95", y="collisions", per="n_agents"),
           PlotSpec("tradeoff", "e6_time_vs_messages", "Time against messages", x="t95", y="msgs_per_drone_per_s", per="n_agents"),
           PlotSpec("rate", "e6_safe", "Safe-complete rate", x="n_agents", y="safe_complete"),
           PlotSpec("sweep", "e6_solve", "Compute per decision", x="n_agents", y="solve_ms")],
    quick=dict(maps=["rooms_and_door"], n_agents=[4], seeds=[0, 1])))

_add(Experiment(
    "E7", "Lookahead and agreement", "How much lookahead (horizon) and agreement (ADMM iterations) are needed?",
    models=[ModelSpec("Greedy + D-MPC", "greedy+dmpc", dict(coupling=True), ignores=("model.iterations",)),
            ModelSpec("Greedy + C-ADMM", "greedy+cadmm")],
    maps=MAPS, n_agents=[6], levers={"model.horizon": [3, 5, 10]},
    plots=_sweeps("model_horizon", ["t95", "collisions", "solve_ms", "msgs_per_drone_per_s"], "e7_horizon", "Horizon"),
    quick=dict(maps=["rooms_and_door"], seeds=[0, 1])))
_add(Experiment(
    "E7b", "ADMM iterations per step", "How many ADMM iterations per step are enough?",
    models=[ModelSpec("Greedy + C-ADMM", "greedy+cadmm")], maps=MAPS, n_agents=[6],
    levers={"model.iterations": [1, 3, 5, 10]},
    plots=_sweeps("model_iterations", ["t95", "collisions", "solve_ms", "msgs_per_drone_per_s"], "e7b_iter", "ADMM iterations"),
    quick=dict(maps=["rooms_and_door"], seeds=[0, 1])))

_add(Experiment(
    "E8", "Integrated against separate layers", "Does solving allocation and trajectory together beat solving them separately?",
    models=[INTEGRATED, CBBA_DMPC, CBBA_CADMM, GREEDY], maps=MAPS, n_agents=[4, 6, 8],
    plots=_sweeps("n_agents", ["t95", "mean_visits", "msgs_per_drone_per_s"], "e8", "Integrated method")
    + [PlotSpec("paired", "e8_gain", "Integrated against CBBA + D-MPC", y="t95",
                options=dict(pairs=[("Coverage MPC", "CBBA + D-MPC"), ("Coverage MPC", "CBBA + C-ADMM")]), per=None)],
    quick=dict(maps=["rooms_and_door"], n_agents=[4], seeds=[0, 1])))

FRONTIER_DMPC = ModelSpec("Frontier + D-MPC", "frontier+dmpc")
FRONTIER_DMPC_SHARED = ModelSpec("Frontier + D-MPC (shared)", "frontier+dmpc", dict(share=True))
GREEDY_DMPC_REF = ModelSpec("Greedy + D-MPC (known-map model)", "greedy+dmpc", role="reference")

_add(Experiment(
    "E3D", "D-MPC: cost of not knowing the map",
    "How much slower (and is it less safe) is D-MPC when drones must discover the obstacles?",
    models=[FRONTIER_DMPC, FRONTIER_DMPC_SHARED, GREEDY_DMPC_REF],
    maps=["empty", "sparse", "corridor", "rooms_and_door", "cluttered"], n_agents=[2, 4], seeds=[0, 1, 2],
    levers={"map_prior": ["known", "unknown"]},          # greedy+dmpc is known-map only: its unknown jobs are skipped
    plots=[PlotSpec("paired", "e3d_cost", "D-MPC: unknown against known map", y="t95", per=None,
                    options=dict(pairs=[("unknown", "known")], label_col="map_prior",
                                 match=("label", "map", "n_agents", "seed"))),
           PlotSpec("sweep", "e3d_t95", "D-MPC: time to 95% by map prior", x="map_prior", y="t95"),
           PlotSpec("sweep", "e3d_t100", "D-MPC: time to 100% by map prior", x="map_prior", y="t100"),
           PlotSpec("sweep", "e3d_collisions", "D-MPC: collisions by map prior", x="map_prior", y="collisions"),
           PlotSpec("curves", "e3d_coverage", "D-MPC: coverage over time, map unknown", where={"map_prior": "unknown"})],
    quick=dict(maps=["rooms_and_door"], n_agents=[2], seeds=[0])))
