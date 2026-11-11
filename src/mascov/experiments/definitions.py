"""The experiments of the study plan, as data. Run them with scripts/run_experiments.py, draw them with
scripts/plot_experiments.py.

Models that are not implemented yet (cbba+dmpc, coverage_mpc) are listed under the names the
team agreed on; the runner skips them (and says so in skipped.txt) until a model with that name is registered
with ``mascov.common.model.register``. No file here needs to change when a model arrives. Expected names and options:

    cbba                    CBBA allocation (models/cbba), Navigator  kwargs: share_coverage, max_bundle_size, delta
    greedy+dmpc             greedy goals + D-MPC                      kwargs: coupling (bool), horizon
    cbba+dmpc               CBBA allocation + D-MPC
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
CBBA_DMPC = ModelSpec("CBBA + D-MPC", "cbba+dmpc", dict(coupling=True))
INTEGRATED = ModelSpec("Coverage MPC", "coverage_mpc")
PARTITION = ModelSpec("Static partition", "static_partition", role="reference")
CENTRAL = ModelSpec("Centralised greedy", "greedy_centralised", role="reference")
FRONTIER = ModelSpec("Frontier greedy", "frontier_greedy")
FRONTIER_SHARED = ModelSpec("Frontier greedy (shared)", "frontier_greedy_shared")
CBBA_PRIVATE = ModelSpec("CBBA (private belief)", "cbba", dict(share_coverage=False))
CBBA_SHARED = ModelSpec("CBBA (shared belief)", "cbba", dict(share_coverage=True))

MAPS = ["sparse", "rooms_and_door"]
FULL_GRID = [GREEDY, CBBA, DMPC_OFF, DMPC, CBBA_DMPC]

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
    models=[CBBA, DMPC,
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
    "E7", "Lookahead", "How much lookahead (D-MPC horizon) is needed?",
    models=[ModelSpec("Greedy + D-MPC", "greedy+dmpc", dict(coupling=True))],
    maps=MAPS, n_agents=[6], levers={"model.horizon": [3, 5, 10]},
    plots=_sweeps("model_horizon", ["t95", "collisions", "solve_ms", "msgs_per_drone_per_s"], "e7_horizon", "Horizon"),
    quick=dict(maps=["rooms_and_door"], seeds=[0, 1])))

_add(Experiment(
    "E8", "Integrated against separate layers", "Does solving allocation and trajectory together beat solving them separately?",
    models=[INTEGRATED, CBBA_DMPC, GREEDY], maps=MAPS, n_agents=[4, 6, 8],
    plots=_sweeps("n_agents", ["t95", "mean_visits", "msgs_per_drone_per_s"], "e8", "Integrated method")
    + [PlotSpec("paired", "e8_gain", "Integrated against CBBA + D-MPC", y="t95",
                options=dict(pairs=[("Coverage MPC", "CBBA + D-MPC")]), per=None)],
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

# ================================================================================================
# Greedy explorer + geodesic partition experiments (unknown-map interface models)
# ================================================================================================

GREEDY_EXP = ModelSpec("Greedy explorer", "greedy_explorer")
GREEDY_EXP_SHARED = ModelSpec("Greedy explorer (shared beliefs)", "greedy_explorer", dict(share=True))
PARTITION_EXP = ModelSpec("Geodesic partition", "geodesic_partition")
PARTITION_EXP_SHARED = ModelSpec("Geodesic partition (shared beliefs)", "geodesic_partition", dict(share=True))
# CBBA / CBAA on the unknown-map interface (models/cbba/explorer.py); CBAA = CBBA with a bundle of one task
CBBA_EXP = ModelSpec("CBBA", "cbba_explorer", dict(share=False))
CBBA_EXP_SHARED = ModelSpec("CBBA (shared beliefs)", "cbba_explorer", dict(share=True))
CBAA_EXP = ModelSpec("CBAA", "cbaa_explorer", dict(share=False))
CBAA_EXP_SHARED = ModelSpec("CBAA (shared beliefs)", "cbaa_explorer", dict(share=True))

EXP_MAPS = ["sparse", "rooms_and_door", "cluttered"]

_add(Experiment(
    "PRIOR", "Value of the map prior", "How much does knowing the map give us?",
    models=[GREEDY_EXP, GREEDY_EXP_SHARED, PARTITION_EXP, PARTITION_EXP_SHARED, CBBA_EXP, CBBA_EXP_SHARED],
    maps=EXP_MAPS, n_agents=[4], levers={"map_prior": ["known", "unknown"]},
    plots=[PlotSpec("sweep", "prior_t95", "Map prior: time to 95%", x="map_prior", y="t95"),
           PlotSpec("paired", "prior_cost", "Unknown against known map", y="t95",
                    options=dict(pairs=[("unknown", "known")], label_col="map_prior",
                                 match=("label", "map", "n_agents", "seed")), per=None),
           PlotSpec("curves", "prior_coverage", "Coverage over time", where={"map_prior": "unknown"})],
    quick=dict(maps=["rooms_and_door"], seeds=[0, 1])))

_add(Experiment(
    "COMM", "Communication radius", "How much communication does belief sharing need?",
    models=[GREEDY_EXP_SHARED, PARTITION_EXP, PARTITION_EXP_SHARED, CBBA_EXP, CBBA_EXP_SHARED,
            ModelSpec("Greedy explorer (no messages)", "greedy_explorer", role="reference", ignores=("world.r_comm",))],
    maps=EXP_MAPS, n_agents=[6], map_prior="unknown", levers={"world.r_comm": [2.0, 4.0, 6.0, 8.0, 12.0, math.inf]},
    plots=[PlotSpec("sweep", "comm_t95", "Communication radius: time to 95%", x="world_r_comm", y="t95"),
           PlotSpec("sweep", "comm_msgs", "Communication radius: messages", x="world_r_comm", y="msgs_per_drone_per_s"),
           PlotSpec("sweep", "comm_visits", "Communication radius: visits per target", x="world_r_comm", y="mean_visits")],
    quick=dict(maps=["rooms_and_door"], seeds=[0, 1])))

# Team size x communication radius (unknown map).
ABL_MODELS = [PARTITION_EXP, PARTITION_EXP_SHARED, GREEDY_EXP_SHARED, CBBA_EXP_SHARED,
              ModelSpec("Greedy explorer (no messages)", "greedy_explorer", role="reference", ignores=("world.r_comm",))]
R_LEVELS = [1.5, 3.0, 6.0, math.inf]

_add(Experiment(
    "ABL", "Team size x communication radius", "How do team size and communication radius change each method?",
    models=ABL_MODELS, maps=EXP_MAPS, n_agents=[2, 4, 6, 8], map_prior="unknown", levers={"world.r_comm": R_LEVELS},
    plots=[PlotSpec("sweep", "abl_t95", "Time to 95% by radius", x="world_r_comm", y="t95"),
           PlotSpec("sweep", "abl_collisions", "Collisions by radius", x="world_r_comm", y="collisions"),
           PlotSpec("sweep", "abl_msgs", "Messages by radius", x="world_r_comm", y="msgs_per_drone_per_s")],
    quick=dict(maps=["rooms_and_door"], n_agents=[2, 4], seeds=[0, 1])))

# Map prior x team size x radius: does the partition use the map better than greedy?
_add(Experiment(
    "ABLP", "Map prior x team size x radius", "Does the partition use the map better than greedy, across team sizes and radii?",
    models=[PARTITION_EXP_SHARED, GREEDY_EXP_SHARED, CBBA_EXP_SHARED], maps=EXP_MAPS, n_agents=[2, 4, 6, 8],
    levers={"map_prior": ["known", "unknown"], "world.r_comm": [3.0, math.inf]},
    plots=[PlotSpec("paired", "ablp_prior_cost", "Unknown against known map", y="t95",
                    options=dict(pairs=[("unknown", "known")], label_col="map_prior",
                                 match=("label", "map", "n_agents", "seed")), per=["map"]),
           PlotSpec("sweep", "ablp_t95", "Time to 95%", x="world_r_comm", y="t95")],
    quick=dict(maps=["rooms_and_door"], n_agents=[2, 4], seeds=[0, 1])))

# ================================================================================================
# CBBA family only (CBBA and its single-assignment case CBAA), unknown-map interface
# ================================================================================================
CBBA_FAMILY = [CBBA_EXP, CBBA_EXP_SHARED, CBAA_EXP, CBAA_EXP_SHARED]

_add(Experiment(
    "CBBA_PRIOR", "CBBA / CBAA: map prior x team size", "What does not knowing the map cost CBBA and CBAA, at each team size?",
    models=CBBA_FAMILY, maps=EXP_MAPS, n_agents=[2, 4, 6, 8], levers={"map_prior": ["known", "unknown"]},
    plots=[PlotSpec("paired", "cbba_prior_cost", "CBBA / CBAA: unknown against known map", y="t95",
                    options=dict(pairs=[("unknown", "known")], label_col="map_prior",
                                 match=("label", "map", "n_agents", "seed")), per=None),
           PlotSpec("sweep", "cbba_prior_t95", "CBBA / CBAA: time to 95% by team size", x="n_agents", y="t95",
                    where={"map_prior": "unknown"}),
           PlotSpec("sweep", "cbba_prior_collisions", "CBBA / CBAA: collisions by team size", x="n_agents", y="collisions",
                    where={"map_prior": "unknown"}),
           PlotSpec("curves", "cbba_prior_coverage", "CBBA / CBAA: coverage over time (4 drones, map unknown)",
                    where={"map_prior": "unknown", "n_agents": 4})],
    quick=dict(maps=["rooms_and_door"], n_agents=[2, 4], seeds=[0, 1])))

_add(Experiment(
    "CBBA_COMM", "CBBA / CBAA: communication radius", "How much communication do CBBA and CBAA need on an unknown map?",
    models=CBBA_FAMILY, maps=EXP_MAPS, n_agents=[4, 8], map_prior="unknown",
    levers={"world.r_comm": [1.5, 3.0, 6.0, INF]},
    world={"r_neighbor": INF},                      # neighbour sensing stays unlimited: only message reach changes
    plots=[PlotSpec("sweep", "cbba_comm_t95", "CBBA / CBAA: time to 95% by radius", x="world_r_comm", y="t95"),
           PlotSpec("sweep", "cbba_comm_t100", "CBBA / CBAA: time to 100% by radius", x="world_r_comm", y="t100"),
           PlotSpec("sweep", "cbba_comm_msgs", "CBBA / CBAA: messages by radius", x="world_r_comm", y="msgs_per_drone_per_s"),
           PlotSpec("sweep", "cbba_comm_visits", "CBBA / CBAA: visits per target by radius", x="world_r_comm", y="mean_visits")],
    quick=dict(maps=["rooms_and_door"], n_agents=[4], seeds=[0, 1])))

_add(Experiment(
    "CBBA_BUNDLE", "CBBA bundle size (1 = CBAA)", "How long should each drone's bundle be? Bundle 1 is CBAA.",
    models=[ModelSpec("CBBA (shared beliefs)", "cbba_explorer", dict(share=True))],
    maps=EXP_MAPS, n_agents=[4, 8], map_prior="unknown", levers={"model.max_bundle_size": [1, 2, 4, 8]},
    plots=[PlotSpec("sweep", "cbba_bundle_t95", "Bundle size: time to 95%", x="model_max_bundle_size", y="t95"),
           PlotSpec("sweep", "cbba_bundle_t100", "Bundle size: time to 100%", x="model_max_bundle_size", y="t100"),
           PlotSpec("sweep", "cbba_bundle_path", "Bundle size: path length per drone", x="model_max_bundle_size",
                    y="path_len_mean"),
           PlotSpec("sweep", "cbba_bundle_solve", "Bundle size: compute per decision", x="model_max_bundle_size",
                    y="solve_ms")],
    quick=dict(maps=["rooms_and_door"], n_agents=[4], seeds=[0, 1])))

# ================================================================================================
# Q1–Q3: Focused questions comparing ALL algorithms head-to-head
# ================================================================================================

GREEDY_S = ModelSpec("Greedy", "greedy_explorer", dict(share=True))
GREEDY_COORD_S = ModelSpec("Greedy + goals", "greedy_explorer", dict(share=True, coordinate=True))
GEODESIC_S = ModelSpec("Geodesic", "geodesic_partition", dict(share=True))
CBBA_EXPL_S = ModelSpec("CBBA", "cbba_explorer", dict(share=True))
DMPC_S = ModelSpec("D-MPC", "frontier+dmpc", dict(share=True))

ALL5 = [GREEDY_S, GREEDY_COORD_S, GEODESIC_S, CBBA_EXPL_S, DMPC_S]
ALL4 = [GREEDY_S, GEODESIC_S, CBBA_EXPL_S, DMPC_S]    # without greedy+goals
Q_MAPS = ["sparse", "rooms_and_door", "cluttered"]
Q_R = [1.5, 3.0, 6.0, math.inf]

# ---- Q1: Does knowing the map help each algorithm? ----
_add(Experiment(
    "Q1_MAP", "Does knowing the map help?",
    "For each algorithm: how much faster is coverage when obstacles are given vs discovered?",
    models=ALL5, maps=Q_MAPS, n_agents=[4],
    levers={"map_prior": ["known", "unknown"]},
    plots=[PlotSpec("sweep", "q1_t95", "Time to 95% by map prior", x="map_prior", y="t95"),
           PlotSpec("paired", "q1_cost", "Unknown against known map, per algorithm", y="t95",
                    options=dict(pairs=[("unknown", "known")], label_col="map_prior",
                                 match=("label", "map", "n_agents", "seed")), per=None),
           PlotSpec("curves", "q1_coverage", "Coverage over time", where={"map_prior": "unknown"})],
    quick=dict(maps=["rooms_and_door"], seeds=[0, 1])))

# ---- Q2: How much does coordination help, and how does communication range affect it? ----
_add(Experiment(
    "Q2_COORD", "How much does coordination help?",
    "All algorithms compared across team sizes and communication radii",
    models=ALL5, maps=Q_MAPS, n_agents=[2, 4, 6, 8], map_prior="unknown",
    levers={"world.r_comm": Q_R},
    plots=[PlotSpec("sweep", "q2_t95_by_n", "Time to 95% by team size", x="n_agents", y="t95"),
           PlotSpec("sweep", "q2_t95_by_r", "Time to 95% by radius", x="world_r_comm", y="t95"),
           PlotSpec("sweep", "q2_visits", "Visits per target by team size", x="n_agents", y="mean_visits"),
           PlotSpec("sweep", "q2_path", "Path length per drone", x="n_agents", y="path_len_mean"),
           PlotSpec("sweep", "q2_msgs", "Messages per drone per second", x="world_r_comm", y="msgs_per_drone_per_s"),
           PlotSpec("sweep", "q2_collisions", "Collisions by team size", x="n_agents", y="collisions"),
           PlotSpec("curves", "q2_coverage_inf", "Coverage (unlimited radius, 6 drones)",
                    where={"world_r_comm": math.inf, "n_agents": 6}),
           PlotSpec("curves", "q2_coverage_short", "Coverage (radius 1.5 m, 6 drones)",
                    where={"world_r_comm": 1.5, "n_agents": 6})],
    quick=dict(maps=["rooms_and_door"], n_agents=[2, 4], seeds=[0, 1])))

# ---- Q3: What is the price of safety? ----
_add(Experiment(
    "Q3_SAFETY", "What is the price of safety?",
    "D-MPC vs the rest: collisions, separation and speed cost",
    models=ALL5, maps=Q_MAPS, n_agents=[4, 6], map_prior="unknown",
    plots=[PlotSpec("sweep", "q3_collisions", "Collisions by team size", x="n_agents", y="collisions"),
           PlotSpec("sweep", "q3_t95", "Time to 95% by team size", x="n_agents", y="t95"),
           PlotSpec("tradeoff", "q3_tradeoff", "Speed vs safety", x="t95", y="collisions", per=None),
           PlotSpec("sweep", "q3_min_dist", "Minimum drone separation", x="n_agents", y="min_pair_dist")],
    quick=dict(maps=["rooms_and_door"], n_agents=[4], seeds=[0, 1])))

# ---- Q1B: Does map knowledge help on a *hard* map (corridors + dead ends)? ----
_add(Experiment(
    "Q1B_LABYRINTH", "Does the map help when corridors have dead ends?",
    "Labyrinth map: narrow corridors with dead ends penalise optimistic (unknown) planning",
    models=ALL5, maps=["labyrinth"], n_agents=[4],
    levers={"map_prior": ["known", "unknown"]},
    plots=[PlotSpec("sweep", "q1b_t95", "Time to 95%: known vs unknown (labyrinth)", x="map_prior", y="t95"),
           PlotSpec("curves", "q1b_coverage", "Coverage over time (labyrinth)", where={}),
           PlotSpec("sweep", "q1b_path", "Path length (labyrinth)", x="map_prior", y="path_len_mean")],
    quick=dict(seeds=[0, 1])))


# ================================================================================================
# The learned goal layer (models/marl_explorer, MAPPO) against the hand-written decentralised explorers
# ================================================================================================
MARL = ModelSpec("MAPPO", "marl_explorer")
MARL_DMPC = ModelSpec("MAPPO + D-MPC", "marl_explorer+dmpc", dict(share=True))
CBBA_DMPC_S = ModelSpec("CBBA + D-MPC", "cbba_explorer+dmpc", dict(share=True))
MARL_RIVALS = [MARL, CBBA_EXP_SHARED, CBAA_EXP_SHARED, GREEDY_COORD_S, PARTITION_EXP_SHARED]
MARL_MAPS = ["sparse", "rooms_and_door", "cluttered", "corridor", "labyrinth"]
# seeds 0-9: the training run never sees seeds below 1000 of the random maps (models/marl_explorer/train.py)

_add(Experiment(
    "MARL_TEAM", "MAPPO goal layer: team size", "Does the learned goal layer beat CBBA and the other explorers, at each team size?",
    models=MARL_RIVALS, maps=MARL_MAPS, n_agents=[2, 4, 6, 8], map_prior="unknown",
    plots=[PlotSpec("sweep", "marl_team_t100", "Time to 100% by team size", x="n_agents", y="t100"),
           PlotSpec("sweep", "marl_team_t95", "Time to 95% by team size", x="n_agents", y="t95"),
           PlotSpec("sweep", "marl_team_visits", "Visits per target by team size", x="n_agents", y="mean_visits"),
           PlotSpec("paired", "marl_team_vs_cbba", "MAPPO against CBBA (shared beliefs)", y="t100", per=None,
                    options=dict(pairs=[("MAPPO", "CBBA (shared beliefs)")])),
           PlotSpec("curves", "marl_team_coverage", "Coverage over time (6 drones)", where={"n_agents": 6})],
    quick=dict(maps=["rooms_and_door"], n_agents=[4], seeds=[0, 1])))

_add(Experiment(
    "MARL_COMM", "MAPPO goal layer: communication radius", "How does the learned goal layer degrade as messages get shorter?",
    models=MARL_RIVALS, maps=["sparse", "rooms_and_door", "cluttered"], n_agents=[4, 8], map_prior="unknown",
    levers={"world.r_comm": [1.5, 3.0, 6.0, INF]}, world={"r_neighbor": INF},
    plots=[PlotSpec("sweep", "marl_comm_t100", "Time to 100% by radius", x="world_r_comm", y="t100"),
           PlotSpec("sweep", "marl_comm_msgs", "Messages by radius", x="world_r_comm", y="msgs_per_drone_per_s")],
    quick=dict(maps=["rooms_and_door"], n_agents=[4], seeds=[0, 1])))

_add(Experiment(
    "MARL_PRIOR", "MAPPO goal layer: map prior", "What does not knowing the map cost the learned goal layer?",
    models=[MARL, CBBA_EXP_SHARED], maps=MARL_MAPS, n_agents=[4], levers={"map_prior": ["known", "unknown"]},
    plots=[PlotSpec("paired", "marl_prior_cost", "Unknown against known map", y="t100", per=None,
                    options=dict(pairs=[("unknown", "known")], label_col="map_prior",
                                 match=("label", "map", "n_agents", "seed")))],
    quick=dict(maps=["rooms_and_door"], seeds=[0, 1])))

_add(Experiment(
    "MARL_SAFE", "MAPPO goal layer: safe flying", "What do BVC and D-MPC cost the learned goal layer, and do they remove collisions?",
    models=[MARL, MARL_DMPC, CBBA_EXP_SHARED, CBBA_DMPC_S], maps=["sparse", "rooms_and_door", "cluttered"],
    n_agents=[4, 6], map_prior="unknown", levers={"sim.safety": ["none", "bvc"]},
    plots=[PlotSpec("sweep", "marl_safe_collisions", "Collisions", x="sim_safety", y="collisions"),
           PlotSpec("sweep", "marl_safe_t100", "Time to 100%", x="sim_safety", y="t100"),
           PlotSpec("rate", "marl_safe_complete", "Safe-complete rate", x="sim_safety", y="safe_complete")],
    quick=dict(maps=["rooms_and_door"], n_agents=[4], seeds=[0, 1])))

# ================================================================================================
# Large scale: 16-24 m procedural maps (maps/procedural.py), up to 16 drones, map unknown
# ================================================================================================
SCALE_MAPS = ["dense_16", "random_rooms_16", "maze_20", "warehouse_20", "dense_24", "random_rooms_24", "maze_24"]
SCALE_PLOTS = [PlotSpec("sweep", "scale_t100", "Time to 100% by team size", x="n_agents", y="t100"),
               PlotSpec("sweep", "scale_t95", "Time to 95% by team size", x="n_agents", y="t95"),
               PlotSpec("sweep", "scale_collisions", "Collisions by team size", x="n_agents", y="collisions"),
               PlotSpec("sweep", "scale_solve", "Compute per decision", x="n_agents", y="solve_ms")]

_add(Experiment(
    "SCALE", "Large worlds: the hand-written explorers", "How do the decentralised explorers scale to 16-24 m and 16 drones?",
    models=[CBBA_EXP_SHARED, GREEDY_COORD_S, PARTITION_EXP_SHARED, FRONTIER_SHARED], maps=SCALE_MAPS,
    n_agents=[4, 8, 16], seeds=[0, 1, 2], map_prior="unknown", sim={"max_time": 1200.0}, plots=SCALE_PLOTS,
    quick=dict(maps=["dense_16"], n_agents=[4], seeds=[0])))

_add(Experiment(
    "MARL_SCALE", "Large worlds: the MAPPO goal layer", "The same grid as SCALE for marl_explorer (24 m was never trained on)",
    models=[MARL], maps=SCALE_MAPS, n_agents=[4, 8, 16], seeds=[0, 1, 2], map_prior="unknown",
    sim={"max_time": 1200.0}, plots=SCALE_PLOTS, quick=dict(maps=["dense_16"], n_agents=[4], seeds=[0])))

# ================================================================================================
# The study's large scale: 16 x 16 m, 4/8/16 drones, every algorithm of the deck (results slides, map-type tables).
# SCALE above already ran CBBA, Greedy + goals and Geodesic on dense_16 and random_rooms_16 with these team sizes and
# seeds; L16_A and L16_B add everything else on four 16 m map types. MAPPO waits for its final checkpoint (L16_MARL).
# ================================================================================================
_S = dict(share=True)
L16_MAPS = ["dense_16", "random_rooms_16", "maze_16", "warehouse_16"]
# The six map types the results compare at both sizes (12 m and 16 m)
ALL6_12 = ["sparse", "cluttered", "dense_12", "random_rooms_12", "maze_12", "warehouse_12"]
ALL6_16 = ["sparse_16", "cluttered_16"] + L16_MAPS
L16_HAND_NEW = [ModelSpec("Greedy", "greedy_explorer", _S), ModelSpec("D-MPC", "frontier+dmpc", _S),
                ModelSpec("Geodesic + D-MPC", "geodesic_partition+dmpc", _S),
                ModelSpec("CBBA + D-MPC", "cbba_explorer+dmpc", _S)]
L16_COMMON = dict(n_agents=[4, 8, 16], seeds=[0, 1, 2], map_prior="unknown", sim={"max_time": 1200.0},
                  plots=SCALE_PLOTS, quick=dict(maps=["dense_16"], n_agents=[4], seeds=[0]))

_add(Experiment("L16_A", "16 m: Greedy and the D-MPC combinations", "16 m maps, the methods SCALE did not run",
                models=L16_HAND_NEW, maps=L16_MAPS, **L16_COMMON))
_add(Experiment("L16_B", "16 m: CBBA, Greedy + goals, Geodesic on maze and warehouse",
                "The SCALE methods on the two 16 m map types SCALE did not include",
                models=[CBBA_EXP_SHARED, GREEDY_COORD_S, PARTITION_EXP_SHARED], maps=["maze_16", "warehouse_16"],
                **L16_COMMON))
_add(Experiment("L16_MARL", "16 m: MAPPO", "MAPPO with the Navigator and with D-MPC on the 16 m maps",
                models=[MARL, ModelSpec("MAPPO + D-MPC", "marl_explorer+dmpc", _S)], maps=ALL6_16, **L16_COMMON))

# Question 1 at 16 m: does knowing the map help on bigger, structured maps?
for _nm, _models in (("", [ModelSpec("Greedy", "greedy_explorer", _S),
                           ModelSpec("Greedy + goals", "greedy_explorer", dict(share=True, coordinate=True)),
                           ModelSpec("Geodesic", "geodesic_partition", _S), ModelSpec("CBBA", "cbba_explorer", _S),
                           ModelSpec("D-MPC", "frontier+dmpc", _S)]),
                     ("_MARL", [ModelSpec("MAPPO", "marl_explorer", _S)])):
    _add(Experiment(f"PRIOR_16{_nm}", "Map known vs unknown, 16 m", "Does knowing the map help? (16 m, 8 drones)",
                    models=_models, maps=["dense_16", "random_rooms_16", "maze_16"],
                    n_agents=[8], seeds=[0, 1], levers={"map_prior": ["known", "unknown"]}, sim={"max_time": 1200.0},
                    plots=[PlotSpec("paired", f"prior_16{_nm.lower()}_cost", "Unknown against known map (16 m)",
                                    y="t95", per=None, options=dict(pairs=[("unknown", "known")], label_col="map_prior",
                                                                     match=("label", "map", "n_agents", "seed")))],
                    quick=dict(maps=["maze_16"], seeds=[0])))


# Same six map types at 12 m (4/8/16 drones, as at 16 m) and the two 16 m types L16_A / L16_B / SCALE did not cover.
STUDY_HAND = [ModelSpec("Greedy", "greedy_explorer", _S), GREEDY_COORD_S, PARTITION_EXP_SHARED, CBBA_EXP_SHARED,
              ModelSpec("D-MPC", "frontier+dmpc", _S), ModelSpec("Geodesic + D-MPC", "geodesic_partition+dmpc", _S),
              ModelSpec("CBBA + D-MPC", "cbba_explorer+dmpc", _S)]
_add(Experiment("L16_C", "16 m: every hand-written method on sparse and cluttered", "Completes the six 16 m map types",
                models=STUDY_HAND, maps=["sparse_16", "cluttered_16"], **L16_COMMON))
S12_COMMON = dict(n_agents=[4, 8, 16], seeds=[0, 1, 2], map_prior="unknown", sim={"max_time": 900.0},
                  plots=SCALE_PLOTS, quick=dict(maps=["dense_12"], n_agents=[4], seeds=[0]))
_add(Experiment("S12", "12 m: every hand-written method on the six map types", "The 12 m side of the size comparison",
                models=STUDY_HAND, maps=ALL6_12, **S12_COMMON))
_add(Experiment("S12_MARL", "12 m: MAPPO on the six map types", "MAPPO with the Navigator and with D-MPC, 12 m",
                models=[MARL, ModelSpec("MAPPO + D-MPC", "marl_explorer+dmpc", _S)], maps=ALL6_12, **S12_COMMON))
for _nm, _models in (("", [ModelSpec("Greedy", "greedy_explorer", _S), GREEDY_COORD_S, PARTITION_EXP_SHARED,
                           CBBA_EXP_SHARED, ModelSpec("D-MPC", "frontier+dmpc", _S)]),
                     ("_MARL", [ModelSpec("MAPPO", "marl_explorer", _S)])):
    _add(Experiment(f"PRIOR_12{_nm}", "Map known vs unknown, 12 m", "Does knowing the map help? (12 m, 8 drones)",
                    models=_models, maps=["dense_12", "random_rooms_12", "maze_12"],
                    n_agents=[8], seeds=[0, 1], levers={"map_prior": ["known", "unknown"]}, sim={"max_time": 900.0},
                    plots=[], quick=dict(maps=["maze_12"], seeds=[0])))
