# MAS-Project: multi-agent drone coverage

Simulation environment (and, in `models/`, the algorithms) for covering a continuous 2D workspace
with polygonal obstacles using several double-integrator drones.

## Setup

```bash
pip install -e ".[dev]"          # numpy, shapely>=2, matplotlib, pytest
python -m pytest -q              # run from the repository root
PYTHONPATH=src:. python scripts/demo_greedy.py sparse 0
```

Run everything from the repository root so that `models/` is importable.
Optional, for the STL planner: `scripts/setup_stlplanning.sh` (clones the authors' code at a pinned commit into `third_party/`
and installs `gurobipy`). Without it the STL tests skip and `import models` simply omits `stl_single`.

## The world model (what is simulated)

* **Space:** continuous rectangle `[0,W] x [0,H]`, metres. **Obstacles:** static *convex* polygons
  (decompose non-convex ones first). `geometry.polygons.halfspaces(poly)` gives `H x <= b` for the STL layer.
* **Drones:** double integrator, `|v| <= v_max`, `|a| <= a_max`, body radius `r_b`.
* **Sensing:** a drone sees every point within `R_s` whose straight line to the drone does not cross
  an obstacle interior (grazing a corner does not block). Coverage is permanent.
* **Targets:** a lattice of points (`target_spacing`) in free space stands in for area coverage.
  "Covered" = seen by some drone. The goal is to cover all *coverable* targets
  (some viewpoint can see them). **Viewpoints:** candidate standing positions + the set of targets each sees.
* **Planning free space:** workspace shrunk and obstacles grown by `clearance = r_b + eps_track`.
  `PathPlanner` gives shortest paths / travel distances inside it (visibility graph).
* **Objective:** minimise time until all coverable targets are covered.

## Layout

```
src/mascov/
  common/      WorldConfig, SimConfig, Plan/Command/Observation types, Model interfaces + registry
  geometry/    polygons (+ half-spaces), line-of-sight, visibility-graph path planner
  world/       World (ground truth), Sensor, AgentBelief (set-union merge)
  viewpoints/  candidate viewpoints and visibility sets
  sim/         dynamics, shared tracker, Simulator
  metrics/     RunResult, MetricsLogger, lower bounds (reach, area)
  planning/    shared building blocks: greedy viewpoint choice, Navigator (collision avoidance), greedy agent
  safety/      BVC drone-drone safety filter on stopping points (control rate, decentralised)
  stl/         adapter around the authors' STL MILP (PWLPlan.py) + independent plan verifier
  maps/        seeded benchmarks: empty, sparse, corridor, rooms_and_door, cluttered
  viz/         plots
models/        one folder per algorithm: greedy_single (B0), greedy_centralised (B2),
               greedy_uncoordinated (B3), static_partition (B4), stl_single (windowed formal planner, 1 drone),
               stl_team (windowed joint multi-agent STL planner)
               `import models` registers them all (stl_single / stl_team only if Gurobi + the STL checkout are present)
third_party/   where the STLPlanning checkout goes (not vendored: it has no licence), see third_party/README.md
tests/  scripts/  configs/  experiments/  docs/
```

## Writing a model

Every model outputs **timed polylines** (`Plan`); one shared tracker executes them, so all methods
face identical dynamics. Implement one of:

```python
class CentralModel:   reset(world_view, agent_ids);  step(obs: dict[int, Observation]) -> dict[int, Command]
class AgentPolicy:    reset(world_view, agent_id, n_agents);  step(obs: Observation) -> Command
```

Wrap an `AgentPolicy` in `DecentralisedModel(PolicyClass)`: the wrapper hands each policy only its own
`Observation`, which is how "no central planner" is enforced. Register with `@register("name")`.

`Observation` carries the agent's own position, its own belief (covered targets, known obstacles),
neighbours within `r_comm`, and its inbox. Messages sent in a `Command` are delivered at the next
decision step to agents within `r_comm` of the sender. Merging beliefs is the model's job
(`belief.merge(other)`); the simulator never shares coverage on its own.

## Defaults and measured numbers

`WorldConfig`: 12 x 12 m, `R_s`=1.5, `r_b`=0.15 (so `d_safe`=0.3), `v_max`=1, `a_max`=2, `dt`=0.1,
planners fly at 0.6 `v_max`, `eps_track`=0.3.

`eps_track` is **measured, not assumed**: the largest deviation from the reference path over all benchmark
maps is ~0.26 m at `plan_speed_ratio`=0.6 (it was ~0.48 m at 0.8), and `tests/test_sim.py` fails if
a run exceeds `eps_track`. If you change the tracker or planning speed, re-measure
(`RunResult.max_tracking_error`) before trusting any STL guarantee.

## Baselines (B0, B2, B3, B4)

| id | model | knows | coordinates | role |
|---|---|---|---|---|
| B0 | `greedy_single` | one drone | n/a | tests the environment end to end |
| B2 | `greedy_centralised` | everything (pooled beliefs) | claims viewpoints centrally | best simple full-information reference |
| B3 | `greedy_uncoordinated` | own belief only | none, no messages | the floor |
| B4 | `static_partition` | own belief + a fixed split made at t=0 | split once (balanced geodesic k-means) | cheap split, never rebalanced |

All of them pick the viewpoint maximising (new targets seen) / (travel time + 1 s). B2 drops a claim whose
targets were seen en route; B3 breaks exact ties at random (otherwise drones starting together move in lockstep).
Reproduce: `python scripts/run_baselines.py --n 1 2 3 4 --seeds 0 1 --out results/baselines.csv`
(`--full` runs on until every drone *believes* it is done). Raw results of the run below: `docs/baseline_results.csv`.

**Results** (110 runs: 5 maps x N=1..4 x 2 seeds; mean true coverage time in seconds, and time / lower bound):

| N | B0 | B2 | B3 | B4 | T/LB: B0, B2, B3, B4 |
|---|---|---|---|---|---|
| 1 | 229 | 189 | - | - | 5.1, 4.2, -, - |
| 2 | - | 100 | 163 | 101 | -, 4.5, 7.4, 4.6 |
| 3 | - | 68 | 146 | 76 | -, 4.4, 10.1, 5.1 |
| 4 | - | 61 | 147 | 60 | -, 4.1, 10.6, 4.2 |

What this shows (and what it does not):

* **Not sharing is expensive, and more drones don't help it.** B3 is 1.6x (N=2) to 2.4x (N=4) slower than B2 and
  gets no faster beyond N=2: drones duplicate each other's work and get in each other's way. B2 speeds up 1.9x / 2.8x / 3.1x.
* **Believing you are done lags the truth** only for B3: in `--full` runs on `empty` and `rooms_and_door` (N=3) its drones kept
  flying 48-100 s after the room was fully covered; B2 and B4 finished within about 2 s of the truth. A belief is a subset of
  the truth, so a B3 drone can never think it is done early.
* **B2 is not a clear winner over B4.** B2 <= B3 and B4 <= B3 in all 30 multi-drone configurations, but B2 <= B4 in only 18
  of 30; on average they are equal (e.g. 61 s vs 60 s at N=4), and B4 beats B2 on `corridor` at N=4 (101 s vs 130 s).
  So "centralised greedy" is *not* a strong upper reference. Expect a formal planner to be compared against both.
* **B0 vs B2 at N=1** differ (229 s vs 189 s) only because B2 abandons a target seen en route; B0 does not.
* All lower-bound ratios are 4x or more: the bounds are loose (they ignore detours, acceleration and interference),
  so "close to the bound" is not a realistic bar for these maps.

## Collision handling in the baselines is a heuristic, not a guarantee

`planning/navigator.py` is the rule all multi-drone baselines share: stationary neighbours are routed around; moving
higher-priority neighbours are predicted over 2.5 s and yielded to (hold, else sidestep, resume when clear); every neighbour
gets a short-range check so a braking drone is never ignored; a drone that has waited 4 s pushes forward, then relocates at
random (this removed deadlocks that left runs stuck at 7-94% coverage). Priority is "working before idle, then lower id".

Measured over the 110 runs: all complete; B0 and B2 never collided; **B3 had collision episodes in 2 of 30 runs and B4 in 2 of 30**,
all at N=4 (separations down to 0.035 m for one B3 run on `sparse`, otherwise 0.27-0.29 m against the 0.3 m limit).
Earlier tuning had shown 0 collisions on a smaller set; a tiny change to a keep-away radius then exposed these, so
treat "0 collisions" as *not established* for B3/B4. The guarantee belongs to the BVC / C-ADMM safety layer, which is why
that layer exists (the BVC filter below removed them in the tested runs). Tracking error under these reactive replans also exceeds `eps_track`: up to ~0.64 m with several drones
(reversals after a yield), and ~0.37 m for B2 at N=1 (mid-route re-targeting), versus <= 0.3 m for smooth single-drone plans (B0,
asserted in tests). Whatever planner feeds the STL layer must have its own epsilon measured.

## Safety layer: BVC filter (`mascov/safety/bvc.py`)

A decentralised, control-rate filter (`SimConfig(safety="bvc")`). Each drone senses neighbours' positions and
velocities within `r_comm`, builds the perpendicular bisector between the two drones' **stopping points**
`s = p + v|v|/(2 a_b)` (a_b = 0.7 a_max), shifted back by `d_safe/2`, and only applies accelerations whose next stopping
point stays on its own side. Using stopping points rather than positions keeps "brake" always admissible, which is the
recursive-feasibility argument (exact along the line between two drones, approximate when that line rotates).
It is a **drone-drone** filter only: no obstacles, and it assumes every drone runs it.

* Adversarial test (drones commanded straight through each other: head-on, crossing, 4-way swap, 4-way convergence,
  6-drone ring): every scenario collides without the filter; with it the minimum separation is 0.34 m (limit 0.30 m).
* On the baselines (90 runs, N=2..4, 5 maps, 2 seeds): **0 collisions** (B3 and B4 had 2 of 30 runs each without it),
  worst separation 0.353 m, coverage time cost 0 to 1.2%. It rarely fires (about 1 intervention per run) because the Navigator
  already avoids most conflicts.
* **Minimum communication radius.** The filter held at `r_comm` = 0.8 m and failed at 0.5 m (plan speed 0.6 m/s). This matches
  `r_comm >= d_safe + 2 v^2/a_b + 2 v dt` = 0.68 m at 0.6 m/s (about 1.2 m at full v_max). This *replaces* the ~2.5 m estimate
  given earlier, which was derived for plan-level avoidance, not control-rate braking.
* **Not done:** this is the safety half of the proposal's Layer 2. The C-ADMM distributed-MPC trajectory generation itself is not
  implemented, nor is obstacle handling in the control layer.

## STL planner (`mascov/stl/`, `models/stl_single`)

`stl/planner.py` wraps the authors' `PWLPlan.py` (Sun, Chen, Mitra, Fan, RA-L 2022). Setup: `scripts/setup_stlplanning.sh`
(pins the commit) or `STL_PLANNING_PATH=...`, plus `pip install gurobipy`. It is **not vendored** (no LICENSE file upstream).
`STLPlanner.plan_visits(start, goals, obstacles, tmax=..., window=...)` solves "eventually reach every goal box within
tmax, always stay out of every obstacle inflated by r_b + eps" for one drone, returning timed waypoints (`STLResult.to_plan()`).
The solver chooses the visiting order and timing. Quirks of the upstream code that the adapter contains: it writes `test.lp` to the
working directory, prints heavily, swallows solver exceptions (the adapter records them, e.g. the licence-size error), hard-codes
a 0.1 goal shrink (exposed as `goal_shrink`), and uses 1-norm speed limits (conservative on diagonals).
`verify_plan()` re-checks any plan independently (clearance, goals reached, speed, time, workspace).

**Scaling (single drone, search over segment count K; mean of 2 seeds; `docs/stl_scaling.csv`, `scripts/stl_scaling.py`):**

| goals | solve s, 0 obstacles | 3 obstacles | 6 obstacles | binaries (6 obst.) |
|---|---|---|---|---|
| 1 | 0.01 | 0.03 | 0.04 | 106 |
| 2 | 0.03 | 0.04 | 0.09 | 162 |
| 3 | 0.10 | 0.17 | 0.49 | 290 |
| 4 | 0.51 | 0.77 | 0.86 | 353 |
| 5 | 1.12 | 1.78 | 2.33 | 452 |
| 6 | 4.22 | 6.31 | 6.72 (1 of 2 solved) | 572 |

Beyond that: a single no-obstacle probe took 21 s for 7 goals and 102 s for 8 (about 5x per extra goal). The one failure above
(6 goals, 6 obstacles) is the pip Gurobi licence limit ("Model too large for size-limited license") at K=20, not infeasibility;
the same limit hit 14 of 15 windows on the `cluttered` map before windows were made to degrade. Practical window: **about 4 goals**.
Cost grows with obstacles too (each adds binaries per segment), so windows only include obstacles that touch the window.

**`stl_single`** (windowed receding horizon): greedily pick 4 viewpoints, let the MILP order and time them, execute, repeat. If a
window is infeasible or too large it is retried with half the goals, then 1, then falls back to a plain shortest path. The window
(waypoint bounding box) must contain the geodesic route, not just start and goals, or a door outside the box makes it infeasible
(an early bug that caused 5 of 13 windows to fail). Reproduce: `python scripts/run_stl_vs_greedy.py`.

Single-drone coverage time in seconds (B0 greedy | B2 centralised greedy | STL), 5 maps, seed 0, same clearance for all:

| eps_track | B0 | B2 | STL | STL vs B0 | STL vs B2 |
|---|---|---|---|---|---|
| 0.30 | 229 | 192 | 232 | +1% | +20% |
| 0.45 | 182 | 162 | 220 | +21% | +36% |

All 30 runs complete, with 0 collisions and **0 plans failing independent verification**. But:

* **STL is not faster than greedy here.** It is within a few percent of B0 at eps 0.3 (better on corridor -15% and cluttered -3%,
  worse on sparse +17% and rooms +10%) and 20-36% slower than B2. Plausible contributors, *not isolated*: the 1-norm speed limit
  (diagonal Euclidean speed is 0.6/sqrt(2)), stopping at the end of every window, and myopic selection of each window's 4 goals. What it
  offers instead is verified clearance on every plan and an optimal order within each window.
* **The guarantee needs the right epsilon.** With eps_track=0.3, STL-plan tracking error was 0.38 to 0.47 m on 4 of 5 maps, so the
  assumption behind the guarantee was violated. With eps_track=0.45 it held on all 5 (worst 0.436 m). Use `WorldConfig(eps_track=0.45)`
  for STL runs; passages must then be wider than 2 x 0.6 m.
* **Goal boxes are shrunk by 0.1, not by eps**, so reaching a viewpoint's box does not guarantee its targets are seen; this is why
  a viewpoint may be tried twice. Cluttered maps still see 1 fallback.
* Needs an **unrestricted Gurobi licence** (free academic) for dense maps or larger windows.

### Joint multi-agent STL (`STLPlanner.plan_team`, `models/stl_team`)

One MILP for all N drones: every goal is reached by at least one drone (the solver assigns and orders them), all drones avoid
the inflated obstacles, and their reference paths stay >= 2 eps + d_safe apart. `stl_team` runs it in **synchronous windows**
(pick up to 2 goals per drone, solve, everyone executes, repeat once all have finished), retrying with fewer goals when the
solve is infeasible or too large, and falling back to independent shortest paths (no separation guarantee) as a last resort.

Three deliberate changes to the upstream code, patched in only during the call (all found by reading it and then confirmed by a
failing check, see below):

1. **Equal end times.** Upstream stops enforcing separation after a drone's last waypoint (it "vanishes"), but in reality it stays
   there, so a later drone can be routed straight through it. This produced a plan with two drones 0.035 m apart and one real
   collision in the first version. Fixed by forcing all drones to end at the same time (early finishers wait); `verify_team`
   caught it because it holds finished drones in place.
2. **Separation margin includes the body radius** (upstream: 2 eps only, which lets bodies touch when both drift by eps).
3. **Makespan objective** (upstream minimises the *sum* of finishing times, which gives no incentive to balance work).

The upstream 1-norm separation is conservative, so drones must start **at least ~1.8 m apart** at eps=0.45 (use
`make_world(..., start_spacing=2.0, start_per_row=2)`); with the default 0.8 m base the joint problem is infeasible from t=0.

Results (`scripts/run_stl_team.py`, `docs/stl_team_vs_baselines.csv`): 5 maps x N=2,3,4, eps_track=0.45, 2 m start spacing.
All 60 runs complete. Mean coverage time in seconds:

| N | B2 | B4 | B3 | STL-team | STL / B2 |
|---|---|---|---|---|---|
| 2 | 93 | 102 | 142 | 133 | 1.43x |
| 3 | 62 | 66 | 124 | 105 | 1.68x |
| 4 | 52 | 57 | 116 | 108 | 2.06x |

* **The joint planner is slower than the heuristics, and the gap grows with N.** It is slower than B2 in all 15 map/N cases (smallest gap +5%) and slower
  than B4 at every N; it only beats the uncoordinated B3, and modestly. Likely contributors (not isolated): synchronous windows
  (early finishers wait for the slowest), and windows shrunk to fit the solver.
* **The solver is overwhelmed quickly.** Windows that had to shrink below 2 goals per drone: 14 of 46 (N=2), 37 of 46 (N=3), 54 of 69
  (N=4); full fallbacks: 3, 5 and 15 windows. Total solve time per run was 128 s, 232 s and 273 s summed over the five maps.
  Each reaches the free licence's size limit (up to 1881 binaries). An academic licence would help; the cost still grows with N^2 K^2.
* **What it buys: guaranteed separation when it solves jointly.** In every run without a fallback the smallest drone-drone
  distance was 1.33 m (median 1.96 m; required 1.2 m), and all 60 plans passed the independent verifier (0 failures). The
  heuristics typically pass within 0.7 to 1.4 m, and B3 collided in 4 of 15 runs.
* **Collisions occur only through the fallback.** 2 of 15 STL runs collided (both N=4: `sparse` 4 events at 0.01 m, `cluttered` 1 at
  0.078 m), each with 4-5 fallback windows. With `SimConfig(safety="bvc")` both runs had **0 collisions** (separation >= 0.35 m).
* Tracking error stayed under eps_track=0.45 except `empty` N=2 (0.463 m, marginal).

**Not done:** the distributed (per-agent) STL variants (partition + per-drone STL, claim-based allocation + STL), asynchronous
windows, and replanning for unknown maps.

## Known limitations (read before building on this)

* **Known-map only.** `Simulator` raises if `SimConfig.known_map=False`. Obstacle discovery by sensing is
  implemented (`Sensor.visible_obstacle_ids`, `Observation.new_obstacles`), but `WorldView` still exposes
  the full map and the target lattice, so P2/P3 needs `WorldView` restricted first.
* **Viewpoint set is a lattice**, so "coverable" and the reach lower bound are approximate
  (`sense_slack`, plus a sampling slack in `reach_bound`).
* **Decision rate:** models are called every 0.5 s (`SimConfig.decision_period`); a faster rate was tried and cost wall time (about 3.5x) without removing collisions.
* **Collision safety:** the baselines' rule is a heuristic (see above); the BVC filter gives zero collisions in the tested runs but only for
  drone-drone avoidance. **No physical collision response:** violations are logged (`collision_events`, `min_pair_dist`,
  `min_clearance`) but drones pass through each other/obstacles.
* **No communication delay or loss**, only a range limit. **No corruption model yet.**
* Obstacle-aware tracking is absent: the tracker follows the plan; safety near obstacles comes from the
  planning clearance, which only holds while the tracking error stays within `eps_track`.
