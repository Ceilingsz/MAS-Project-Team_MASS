# MAS-Project: multi-agent drone coverage

Simulation environment (and, in `models/`, the algorithms) for covering a continuous 2D workspace
with polygonal obstacles using several single-integrator drones.

## Setup

```bash
pip install -e ".[dev,dmpc]"     # numpy, shapely>=2, matplotlib, scipy, pytest, osqp (D-MPC)
python -m pytest -q              # run from the repository root (about 7 min; see "Testing")
PYTHONPATH=src:. python scripts/demo_greedy.py sparse 0
```

Run everything from the repository root so that `models/` is importable.
Optional, for the STL planners: `pip install -e ".[stl]"` (adds `scipy`, which bundles the HiGHS MILP solver: no licence, no size limit).
Without scipy the STL tests skip and `import models` simply omits `stl_single` and `stl_team`.
Optional, for D-MPC: `pip install -e ".[dmpc]"` (OSQP). Without it `greedy+dmpc` is not registered and its tests skip.

## Testing

The suite has two layers that answer different questions.

**1. Component tests: is each part correct?** Each file tests one part in isolation, with exact expectations:

| file | what it checks |
|---|---|
| `test_geometry.py` | half-spaces, line of sight (corners do not block), visibility-graph planner (detours, snapping, unreachable) |
| `test_world.py` | targets avoid obstacles, invalid starts and non-convex obstacles are rejected, sensing, benchmark maps valid and seed-reproducible, lower bounds |
| `test_sim.py` | plans, single-integrator dynamics, tracker accuracy and speed limit, collision detection, message range, decentralised models see only their own observation, every benchmark map is coverable |
| `test_unknown.py` | the unknown-map interface leaks nothing, map beliefs merge as a join, obstacles are learned only by sensing, frontier explorers (incl. a regression for a drone driven into a known obstacle) |
| `test_framework.py` | metrics (visits, milestones, collision episodes), separate sensing and message ranges, negotiation rounds, experiment runner and plots |
| `test_baselines.py` | greedy viewpoint choice, the shared `Navigator` (shortest path at planning speed, arrival within `sense_slack`, replanning, blocked goals, and that it has no drone-drone logic), the static partition, per-model contracts (e.g. "sends no messages") |
| `test_safety.py` | the BVC filter: never leaves its cell, forced collisions happen without it and not with it, it makes a colliding model collision-free |
| `test_stl.py`, `test_stl_encoding.py` | the MILP encoding and the independent plan verifier (STL planners) |
| `test_dmpc.py` | the D-MPC QP (speed limit, walls, obstacles, neighbour separation), coupling on vs off in forced crossings, solver never fails |

**2. Model quality suite (`test_model_quality.py`): how good is each model?** Every registered model runs on the same
scenarios and is graded by the same criteria. A model that falls short **fails** the criterion; the tests are not
written so that every model passes. A registered model that is not in the suite's model table fails
`test_every_model_is_graded`, so no model can skip it.

| criterion | passes when |
|---|---|
| `completes` | true coverage reaches 100% within the time budget |
| `obstacle_safe` | no drone comes within its body radius of an obstacle or wall |
| `drone_safe` | no two drones ever come closer than `d_safe` |
| `tracking` | tracking error stays within `eps_track`, the margin every planner's clearance assumes |
| `knows_when_done` | every drone declares done, the team does not stop before full coverage, the last drone notices within 20 s |
| `efficient` | coverage time <= 1.5 x the centralised greedy reference on the same scenario |
| `team_speedup` | 3 drones take <= 0.6 x the time 1 drone of the same model takes (perfect sharing: 1/3) |
| `reproducible` | the same scenario and seed give the identical trajectory |

Scenarios (all with `eps_track` = 0.45 and 2 m start spacing, which the STL planners need, so every model faces the same
world): `solo` (1 drone, `rooms_and_door`), `team` (3 drones, `rooms_and_door`), `clutter` (3 drones, `cluttered` seed 1),
`low_comm` (`team` with messages and neighbour sensing limited to 1.5 m; decentralised models only). Each run is simulated
once per session and shared by all criteria.

**Known gaps.** A shortfall that is understood is listed in `KNOWN_GAPS` in `test_model_quality.py` with its measured
numbers. It runs as a *strict* xfail: the suite stays green and lists it as `XFAIL`, and if the model improves so the
criterion passes, the test fails as `XPASS(strict)` until the entry is deleted, so the table cannot go stale. Do not loosen
a threshold to make a model pass; record the gap.

```bash
python -m pytest -q                                   # everything (very long cases skipped)
python -m pytest -q -m "not quality"                  # component tests only (~2 min)
python -m pytest -q tests/test_model_quality.py -rx   # the report card, with each known gap and its numbers
python -m pytest -q tests/test_model_quality.py --runxfail   # every gap as a plain failure
python -m pytest -q --slow                            # also the very long cases (stl_team on clutter, ~12 min)
```

**Report card** (measured 2026-11-10; `-` = not applicable; details and numbers in `KNOWN_GAPS`):

| model | completes | obstacle safe | drone safe | tracking | knows when done | efficient | team speedup | reproducible |
|---|---|---|---|---|---|---|---|---|
| `greedy_single` | pass | pass | - | pass | pass | pass | - | - |
| `greedy_centralised` | pass | pass | **fail**: clutter (pass: team) | pass | pass | - | pass | pass |
| `greedy_uncoordinated` | pass | pass | **fail**: team, clutter, low_comm | pass | **fail**: team, clutter, low_comm (pass: solo) | **fail**: team, clutter (pass: solo) | **fail**: team | pass |
| `static_partition` | pass | pass | **fail**: clutter (pass: team, low_comm) | pass | pass | pass | pass | pass |
| `greedy+dmpc` | pass | pass | pass | pass | **fail**: team, clutter, low_comm (pass: solo) | **fail**: team, clutter (pass: solo) | **fail**: team | pass |
| `frontier_greedy` | pass | pass | **fail**: team, clutter, low_comm | pass | **fail**: team, clutter, low_comm (pass: solo) | **fail**: team, clutter (pass: solo) | pass | pass |
| `frontier_greedy_shared` | pass | pass | **fail**: team, clutter, low_comm | pass | pass | **fail**: team, clutter (pass: solo) | pass | pass |
| `stl_single` | pass | pass | - | pass | pass | pass | - | - |
| `stl_team` | pass (clutter: --slow) | pass (clutter: --slow) | pass (clutter: --slow) | pass (clutter: --slow) | pass (clutter: --slow) | **fail**: team (pass: solo) (clutter: --slow) | pass | pass |

What it says:
* **Drone-drone safety is the main gap.** Every model built on the shared `Navigator` collided somewhere, including the
  centralised reference on `clutter`. Only `greedy+dmpc` and `stl_team` were collision-free in every graded scenario.
* **Coordination decides speed.** Only the models that split the work (`greedy_centralised`, `static_partition`) are within
  1.5 x of the reference with 3 drones; `greedy+dmpc` uses the uncoordinated greedy goals, so it is safe but no faster than
  `greedy_uncoordinated`, and its 3-drone team is barely faster than one drone (0.91 x).
* **Without sharing, drones notice late** that the map is covered (21-96 s after the fact).
* `low_comm` (1.5 m) changed nothing for models that never communicate, as expected; it is the lever for communicating models.

## The world model (what is simulated)

* **Space:** continuous rectangle `[0,W] x [0,H]`, metres. **Obstacles:** static *convex* polygons
  (decompose non-convex ones first). `geometry.polygons.halfspaces(poly)` gives `H x <= b` for the STL layer.
* **Drones:** single integrator (`p' = u`, the velocity is the control input), `|v| <= v_max`, body radius `r_b`.
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
  world/       World (ground truth), Sensor, AgentBelief (set-union merge), MapBelief (unknown-map belief),
               WorldView / UnknownWorldView (what a model is told at reset)
  viewpoints/  candidate viewpoints and visibility sets
  sim/         dynamics, shared tracker, Simulator
  metrics/     RunResult, MetricsLogger, lower bounds (reach, area)
  planning/    shared building blocks: greedy viewpoint choice, Navigator (collision avoidance), greedy agent,
               known_map (planning over the obstacles an agent knows), agent_frontier (unknown-map explorer)
  safety/      BVC drone-drone safety filter (buffered Voronoi cells, control rate, decentralised)
  stl/         MILP encoding of the coverage spec (milp.py: HiGHS wrapper, encoding.py), planner, independent plan verifier
  maps/        seeded benchmarks: empty, sparse, corridor, rooms_and_door, cluttered
  viz/         plots
models/        one folder per algorithm: greedy_single (B0), greedy_centralised (B2),
               greedy_uncoordinated (B3), static_partition (B4), stl_single (windowed formal planner, 1 drone),
               stl_team (windowed joint multi-agent STL planner),
               frontier_greedy, frontier_greedy_shared (unknown-map explorers)
               `import models` registers them all (stl_single / stl_team only if scipy / HiGHS is present)
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

> **Dynamics changed.** Every number measured below (baselines, BVC, STL, unknown-map, and the `eps_track` / `plan_speed_ratio` values)
> was produced with the earlier **double-integrator** model (`a_max` = 2 m/s^2, PD tracker). The simulator now uses
> **single-integrator** dynamics (velocity is the control; no acceleration limit, no braking distance). Expect smaller tracking
> error and different times, so the tables must be regenerated before they are quoted. The qualitative conclusions may or may
> not carry over. The tracker is now `u = v_ref + gain * (p_ref - p)` (`SimConfig.tracker_gain`).

`WorldConfig`: 12 x 12 m, `R_s`=1.5, `r_b`=0.15 (so `d_safe`=0.3), `v_max`=1, `dt`=0.1,
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
* All lower-bound ratios are 4x or more: the bounds are loose (they ignore detours and interference),
  so "close to the bound" is not a realistic bar for these maps.

## Collision handling in the baselines is a heuristic, not a guarantee

> **Out of date.** The current `planning/navigator.py` has **no drone-drone logic at all**: it flies the shortest path and
> ignores neighbours (`tests/test_baselines.py::test_navigator_has_no_drone_drone_logic`). The yield/sidestep rule described
> below no longer exists, so the baselines collide whenever their drones' paths cross. Safety comes from `SimConfig(safety="bvc")`,
> `greedy+dmpc` or `stl_team`. The collision numbers below were measured with the old rule.

`planning/navigator.py` is the rule all multi-drone baselines share: stationary neighbours are routed around; moving
higher-priority neighbours are predicted over 2.5 s and yielded to (hold, else sidestep, resume when clear); every neighbour
gets a short-range check so a drone that is just finishing a plan is never ignored; a drone that has waited 4 s pushes forward, then relocates at
random (this removed deadlocks that left runs stuck at 7-94% coverage). Priority is "working before idle, then lower id".

Measured over the 110 runs: all complete; B0 and B2 never collided; **B3 had collision episodes in 2 of 30 runs and B4 in 2 of 30**,
all at N=4 (separations down to 0.035 m for one B3 run on `sparse`, otherwise 0.27-0.29 m against the 0.3 m limit).
Earlier tuning had shown 0 collisions on a smaller set; a tiny change to a keep-away radius then exposed these, so
treat "0 collisions" as *not established* for B3/B4. The guarantee belongs to the BVC / C-ADMM safety layer, which is why
that layer exists (the BVC filter below removed them in the tested runs). Tracking error under these reactive replans also exceeds `eps_track`: up to ~0.64 m with several drones
(reversals after a yield), and ~0.37 m for B2 at N=1 (mid-route re-targeting), versus <= 0.3 m for smooth single-drone plans (B0,
asserted in tests). Whatever planner feeds the STL layer must have its own epsilon measured.

## Safety layer: BVC filter (`mascov/safety/bvc.py`)

A decentralised, control-rate filter (`SimConfig(safety="bvc")`). With single-integrator drones the next position is `p + v dt`, so this
is the classic buffered Voronoi cell: each drone senses neighbours' positions within `r_comm`, builds the perpendicular bisector between
itself and each neighbour shifted back by `d_safe/2` (plus a 0.02 m buffer), and only applies velocities whose next position stays on its own
side of every such line. Two drones that both obey this can never come closer than `d_safe`. Stopping is instantaneous, so zero velocity
is always admissible while everyone is currently at least `d_safe` apart: recursive feasibility is exact (the earlier double-integrator
version needed stopping points and a braking argument, which are gone). The filter picks the admissible velocity closest to the tracker's
request from a finite set (the request, its projection onto each neighbour's half-space, zero, a polar grid).
It is a **drone-drone** filter only: no obstacles, and it assumes every drone runs it.

* **Minimum communication radius.** A neighbour must be sensed before it can get within `d_safe` in one control step:
  `r_comm >= d_safe + 2 v_max dt` (+ buffer), i.e. about 0.52 m at `v_max`, less at planning speed. The earlier double-integrator
  formula (about 0.68 m) included braking distance.
* **Old measurements no longer apply.** The adversarial-scenario, baseline-run and `r_comm` numbers from the double-integrator model
  (minimum separation 0.34 m; 0 collisions in 90 runs at 0-1.2% time cost; held at `r_comm` 0.8 m and failed at 0.5 m) have **not been
  re-measured**. Re-run `python -m pytest tests/test_safety.py` and `python scripts/run_baselines.py --n 2 3 4 --safety bvc`.
* **Not done:** this is the safety half of the proposal's Layer 2. The C-ADMM distributed-MPC trajectory generation itself is not
  implemented, nor is obstacle handling in the control layer.

## STL planner (`mascov/stl/`, `models/stl_single`)

> **Solver changed.** Every STL number below (scaling table, single-drone and multi-drone results, solver limits) was measured with the authors'
> `PWLPlan.py` on Gurobi's size-limited pip licence, and before the single-integrator change. The planner now uses **our own MILP encoding
> solved with HiGHS** (no licence, no size limit), so those numbers are stale and the Gurobi-licence limits no longer apply. Regenerate with
> `scripts/stl_scaling.py`, `scripts/run_stl_vs_greedy.py` and `scripts/run_stl_team.py`.

`stl/planner.py` plans with `stl/encoding.py`: K straight segments between waypoints, solved as a mixed-integer program by HiGHS (through
`scipy.optimize.milp`, wrapped in `stl/milp.py`). We wrote the encoding ourselves because `PWLPlan.py` is built on Gurobi's modelling API (it
could not be re-pointed at another solver) and has no licence file; this also removes the third-party dependency. The scheme is the same
as the paper's (piecewise-linear trajectories, "outside a polytope" as a disjunction over faces with big-M binaries); the details differ:

* `STLPlanner.plan_visits(start, goals, obstacles, tmax=..., window=...)` solves "eventually reach every goal box within tmax, always stay out
  of every obstacle inflated by r_b + eps" for one drone, returning timed waypoints (`STLResult.to_plan()`). It searches the segment count
  K = 1, 2, ... for the first feasible one (or solves exactly `num_segs`).
* **Goals** count as reached when a *waypoint* lies in the goal box shrunk by `goal_shrink` (default 0.1, now our own parameter).
* **Speed** uses the 1-norm (conservative on diagonals) and is linear with no binaries. **Obstacles**: per segment and obstacle one face must be
  violated by at least `bloat + r_b` at both endpoints. Big-M constants are computed per constraint from the window, as small as is valid.
* `verify_plan()` re-checks any plan independently (clearance, goals reached, speed, time, workspace). Speed is checked as a displacement with
  2 cm of slack, because solver tolerances are a visible fraction of the very short (0.01 s) segments a MILP may use.
* Checked on its own (`tests/test_stl_encoding.py`) by dense sampling against obstacles and other drones, independently of the encoder.
  One probe on HiGHS (4 drones, 8 goals, 4 obstacles, 512 binaries) solved to optimality in about 19 s; this is a single random instance,
  not a benchmark.

**Scaling with the OLD Gurobi-based encoding (single drone, search over segment count K; mean of 2 seeds; `docs/stl_scaling.csv`, `scripts/stl_scaling.py`):**

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
* Larger windows are now possible: HiGHS has no size limit (the Gurobi-era limit above no longer applies). Not yet measured with the new encoding.

### Joint multi-agent STL (`STLPlanner.plan_team`, `models/stl_team`)

One MILP for all N drones: every goal is reached by at least one drone (the solver assigns and orders them), all drones avoid
the inflated obstacles, and their reference paths stay >= 2 eps + d_safe apart. `stl_team` runs it in **synchronous windows**
(pick up to 2 goals per drone, solve, everyone executes, repeat once all have finished), retrying with fewer goals when the
solve is infeasible or too large, and falling back to independent shortest paths (no separation guarantee) as a last resort.

How the joint encoding differs from the authors' (the three changes the old adapter patched in are now built in):

1. **Shared waypoint times, so equal end times.** All drones use the same times t_0 < ... < t_K, so every drone ends together (early finishers
   wait) and none vanishes after its last waypoint. The upstream encoding dropped a drone from the separation constraints after its last
   waypoint, which let a later drone be routed through it (one real collision in the first version). `verify_team` holds finished drones in place.
   A shared grid also makes the relative position of two drones linear on each segment, which is what makes the separation encoding exact.
2. **Separation margin includes the body radius:** reference paths stay `2 eps + d_safe` apart, measured in the **infinity norm** (so the Euclidean
   distance is at least that). Four binaries per (pair, segment).
3. **Makespan objective** (t_K, with a tiny path-length tie-break), so work is balanced.

Drones must be at least `2 eps + d_safe` apart (infinity norm) at the start of every window: 1.2 m at eps = 0.45. The old encoding needed about
1.8 m; use `make_world(..., start_spacing=2.0, start_per_row=2)` as before. Otherwise the joint problem is infeasible from t = 0.

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
  Each hit the free Gurobi licence's size limit (up to 1881 binaries); HiGHS has no such limit, though the cost still grows with N^2 K^2.
* **What it buys: guaranteed separation when it solves jointly.** In every run without a fallback the smallest drone-drone
  distance was 1.33 m (median 1.96 m; required 1.2 m), and all 60 plans passed the independent verifier (0 failures). The
  heuristics typically pass within 0.7 to 1.4 m, and B3 collided in 4 of 15 runs.
* **Collisions occur only through the fallback.** 2 of 15 STL runs collided (both N=4: `sparse` 4 events at 0.01 m, `cluttered` 1 at
  0.078 m), each with 4-5 fallback windows. With `SimConfig(safety="bvc")` both runs had **0 collisions** (separation >= 0.35 m).
* Tracking error stayed under eps_track=0.45 except `empty` N=2 (0.463 m, marginal).

**Not done:** the distributed (per-agent) STL variants (partition + per-drone STL, claim-based allocation + STL), asynchronous
windows, and replanning for unknown maps (the unknown-map *interface* exists, see above, but STL does not use it).

## D-MPC trajectory layer (`mascov/planning/dmpc.py`, `models/greedy_dmpc`)

`greedy+dmpc` = greedy viewpoint goals (each drone's own belief, same rule as B3) + distributed MPC trajectories.
Needs `pip install -e ".[dmpc]"` (OSQP); without it `import models` omits the model and `tests/test_dmpc.py` skips.

* **Every control step (0.1 s) each drone solves its own QP** over H steps (`horizon`, default 10 = 1 s): follow a
  reference sampled along the shortest path to its goal at planning speed, smooth velocities, speed inside an octagon
  inscribed in the `v_max` disc (0.95 `v_max`, hard). Walls and obstacles (inflated by `clearance`) are soft
  constraints: per nearby obstacle and step, the line through the obstacle point closest to last step's prediction.
  (An earlier version used the obstacle *face* the prediction was furthest outside of; past a corner that line can cut
  through the drone far from the obstacle, and it stalled drones in `cluttered`/`sparse`.)
* **Coupling** (`coupling=True`): each drone broadcasts its predicted path (H+1 points) to drones within `r_comm`; it
  arrives one step later and is time-shifted and re-anchored at the sensed position. Each neighbour's prediction enters
  as a linearised separation constraint `n . (p_k - q_k) >= d_safe + margin` (`margin` default 0.1 m), soft with a heavy
  penalty so the QP is always feasible. Sensed neighbours with no prediction yet are extrapolated at constant velocity.
  `coupling=False` ignores other drones and sends nothing (the ablation). The navigator heuristic is not used.
* **Deadlocks**: a drone that moved < 0.15 m/s for 2 s sidesteps to its right (small random tilt) for 1.5 s; after 3
  sidesteps for one goal it bans that goal for 10 s. Counted in `RunResult.extra` (`dmpc_deadlock_nudges`, `dmpc_abandoned`),
  along with `dmpc_solves`, `dmpc_failures` (QP not solved: drone holds), `dmpc_slack_steps`, `dmpc_max_slack`.
* **Not a guarantee**: separation is enforced against the neighbour's *predicted* path, one step old. Measure collisions;
  `SimConfig(safety="bvc")` can be added as a backstop. Options: `coupling`, `horizon`, `margin`, `share` (also broadcast
  and merge coverage beliefs), `decision_period`, `seed`, plus any `DMPCParams` weight.
* Compute: about 3-5 ms per solve with dense matrix assembly, so a 4-drone run takes ~1 min of wall time (greedy: ~2 s).
  `layer 1` is replaceable: subclass `DMPCAgentPolicy` and override `select_goal` (for `cbba+dmpc`).

## Unknown-map interface (`SimConfig(known_map=False)`)

The known-map interface hands a model the obstacles, the target set (whose *holes are the obstacles*), the viewpoints and a path
planner. The unknown-map interface hands it none of that:

| | known map | unknown map |
|---|---|---|
| what `reset` receives | `WorldView` | `UnknownWorldView`: config, bounds, the full cell **lattice** (free or not), candidate standing positions, start positions |
| belief | `AgentBelief`: covered flags per target | `MapBelief`: a state per lattice cell, UNKNOWN / FREE (seen) / OBSTACLE, plus the obstacle polygons seen so far |
| merge | set union | element-wise max (commutative, associative, idempotent), so gossip converges and lost messages cannot hurt |
| planning | `World.planner` on the true map | `view.space_for(belief.obstacles)`: visibility graph over the **known** obstacles, unknown space treated as free |
| `Observation.newly_covered` | target ids | lattice cell ids |

* **Sensing.** Seeing a free cell marks it FREE. Seeing any part of an obstacle reveals its **whole polygon** (and marks the cells inside).
  Revealing only the visible boundary is the harder, more realistic variant and is **not implemented**.
* **Optimism.** A route may cross space that later turns out to hold an obstacle. The explorer re-plans when it discovers one
  (`Navigator.set_planner`). With `R_s` = 1.5 m against a clearance of 0.45 m the obstacle is seen well before contact.
* **Starts are common knowledge** (`UnknownWorldView.starts`): an assumption, delete the field to drop it.
* **Interfaces are checked.** Each model declares `map_interface`; `Simulator` refuses a mismatch, with a hint. To give an unknown-map
  model the map, use `SimConfig(known_map=False, reveal_obstacles_at_start=True)`: obstacles are loaded into every belief, coverage stays unknown.
  This is the "same algorithm, map known" reference.
* **No leakage, by test.** `tests/test_unknown.py` asserts that the view has no `obstacles`/`targets`/`viewpoints`/`planner`, and that two
  different maps give an identical view.

**Models** (`models/frontier_greedy`, `planning/agent_frontier.py`): each drone picks the candidate position maximising
(UNKNOWN cells it would reveal, respecting line of sight past known obstacles) / (travel time + 1 s), and declares itself done when no
reachable, unvisited candidate would still reveal anything. `frontier_greedy` uses only its own belief (no messages);
`frontier_greedy_shared` also broadcasts its whole belief to neighbours within `r_comm` and merges what it hears. There is **no intent
sharing or prediction yet**: drones that know the same things choose the same frontier.

![what drone 0 believes, with and without sharing](docs/unknown_map_belief.png)

`python scripts/plot_unknown.py sparse 3 25 docs/unknown_map_belief.png`. Dashed red = obstacles drone 0 has not discovered.

**A bug the tests now guard against.** The first version gave the Navigator *all* candidate positions for sidesteps. One of them lay
inside an obstacle the drone already knew; the path planner snaps such a point to free space but still appends the original point, so
the plan ended inside the obstacle (a drone was inside it for ~2 s). It showed up in 2 of 6 spot runs and nowhere in the known-map code,
because there the candidates are viewpoints and are always free. Fix: `Navigator.set_planner(planner, candidates)` is given only the
candidates that are free under the *known* map, refreshed on every discovery; an idle drone's sidestep plan is replaced by a hold.
`test_explorer_covers_the_map_stays_clear_of_obstacles...` fails without the fix (checked).

### First measurements

`python scripts/run_unknown.py --out docs/unknown_map_runs.csv`: 5 maps x N=2,3,4 x 3 seeds = 45 configurations x 6 methods = 270 runs
(12 x 12 m, `r_comm` unlimited, no BVC; the seed also drives tie-breaking, so it differs even on the deterministic maps).
All 270 complete; every drone concluded by itself that it was done in a separate 120-run check. Mean coverage time in seconds (ratio to B2):

| N | B3 | B2 | F-map | F-unk | F-shared | F-shared-map |
|---|---|---|---|---|---|---|
| 2 | 166 (1.66x) | 100 | 161 (1.61x) | 165 (1.65x) | 146 (1.46x) | 144 (1.44x) |
| 3 | 151 (2.18x) | 69 | 156 (2.26x) | 158 (2.29x) | 127 (1.84x) | 122 (1.77x) |
| 4 | 148 (2.41x) | 62 | 149 (2.42x) | 135 (2.19x) | 95 (1.54x) | 97 (1.57x) |

B3 and B2 are the known-map baselines. F-map / F-unk / F-shared / F-shared-map are the frontier explorer with the map given / unknown / unknown with
sharing / given with sharing. Paired differences over the 45 configurations (mean +- standard error):

| comparison | difference |
|---|---|
| unknown map vs given map, **with** sharing (F-shared vs F-shared-map) | **+3.5% +- 2.7%** |
| unknown map vs given map, no sharing (F-unk vs F-map) | -0.7% +- 2.8% |
| no sharing vs sharing, map unknown (F-unk vs F-shared) | +30.8% +- 5.4% |
| sharing explorer vs B2 (F-shared-map vs B2) | +63.1% +- 5.7% |

What this does and does not show:

* **Not knowing the map cost no measurable time here.** +3.5% +- 2.7% is indistinguishable from zero. A plausible reason (*not isolated*) is that
  a coverage task has to visit everywhere anyway, so discovering the map is not extra work the way it is when searching for one goal. The only map with a
  hint of an effect is `sparse` (+16% +- 5%, n=9); with 5 maps compared that is a lead, not a finding.
* **Sharing is what matters, and it is not enough.** Merging beliefs recovers about 31%, yet the sharing explorer is still ~63% slower than B2. B2
  differs in two ways at once (it knows the map and it assigns viewpoints centrally), and the map is shown above not to matter, so the gap is most likely
  the missing coordination (drones that know the same things go to the same frontier). That is the room for intent sharing and prediction; it is
  *not yet tested*.
* **Caveats:** small 12 m maps, N <= 4, 3 seeds, one greedy explorer; the comparison with B2 also differs in viewpoint set and planner details (on an
  empty map with N=1 the two agree to the decimal, 170.3 s, and a test checks they agree within 5%).
* **Collisions:** 7 of 270 runs had a drone-drone collision episode (B3 2, F-map 2, F-unk 2, F-shared 1; none with an obstacle, closest obstacle
  approach 0.247 m). This is the Navigator weakness documented above, not an unknown-map effect; BVC removed the one case checked.

## Known limitations (read before building on this)

* **Unknown map: whole-polygon reveal only**, and the STL planners (`stl_single`, `stl_team`) and every other known-map model still
  need the known-map interface (they use the true obstacles and viewpoints). An STL layer for unknown maps would have to restrict its
  trajectories to space already observed to be free; that is not built.
* **Viewpoint set is a lattice**, so "coverable" and the reach lower bound are approximate
  (`sense_slack`, plus a sampling slack in `reach_bound`).
* **Decision rate:** models are called every 0.5 s (`SimConfig.decision_period`); a faster rate was tried and cost wall time (about 3.5x) without removing collisions.
* **Collision safety:** the baselines' rule is a heuristic (see above); the BVC filter gives zero collisions in the tested runs but only for
  drone-drone avoidance. **No physical collision response:** violations are logged (`collision_events`, `min_pair_dist`,
  `min_clearance`) but drones pass through each other/obstacles.
* **No communication delay or loss**, only a range limit. **No corruption model yet.**
* Obstacle-aware tracking is absent: the tracker follows the plan; safety near obstacles comes from the
  planning clearance, which only holds while the tracking error stays within `eps_track`.

## Experiment framework (levers, measurables, runner, plots)

The study plan's experiments are data in `src/mascov/experiments/definitions.py` (E0 baselines, E1 to E8, S1 BVC add-on).
Algorithms that are not written yet (CBBA, D-MPC, C-ADMM, the integrated method) are listed there under agreed model
names; the runner skips them until a model with that name is registered, so nothing here changes when they arrive.

```
python scripts/run_experiments.py --list
python scripts/run_experiments.py E1 --quick --jobs 4          # smoke run
python scripts/run_experiments.py E1 E2 --seeds 10 --jobs 8    # real run; resumes if interrupted
python scripts/plot_experiments.py E1 E2                       # results/<exp>/figures/*.png and summary.csv
```

**Levers** (keys in an experiment's `levers`): `n_agents`, `map`, `map_prior` (known/unknown), `world.<field>` such as
`world.r_comm`, `world.r_neighbor`, `world.yield_to_neighbors`, `sim.<field>` such as `sim.safety`, and `model.<kwarg>`
such as `model.horizon`.

**Measurables** (every run, in `RunResult.metrics()`): time to 50/90/95/100% coverage (`t50`..`t100`), coverage versus
time (`curves.jsonl`), `final_coverage`, `collisions` split into `pair_events` and `obstacle_events`, `min_pair_dist`,
`safe_complete` (full coverage and no collision), `mean_visits` (average number of visits per target; a visit is one
continuous stretch with the target in view of one drone) and `frac_revisited`, path length, messages
(`transmissions`, `deliveries`, `msgs_per_drone_per_s`), `solve_ms` (model time per decision).

**Changes to the simulator interface**
- `WorldConfig.r_comm` is now only how far messages reach. How far a drone senses neighbours' position and velocity
  (used by collision avoidance and BVC) is `WorldConfig.r_neighbor`; `None` (the default) means "same as `r_comm`".
  `Observation.neighbors` uses the sensing range, the new `Observation.comm_neighbors` lists the drones in message reach.
- `WorldConfig.yield_to_neighbors=False` turns off the baselines' heuristic yielding and sidestepping: the "no safety" baseline.
- A model (or policy class) may set `decision_period` to be called more often than `SimConfig.decision_period`.
- `mascov.comm.NegotiatingPolicy` / `NegotiatingModel` give a decentralised policy several synchronous message rounds per
  decision (range-limited, counted in `RunResult.extra` as `neg_rounds`, `neg_transmissions`, `neg_deliveries`). CBBA and
  C-ADMM are meant to be written on it. See the docstring in `src/mascov/comm/rounds.py`.
- `SimConfig.milestones` sets which coverage fractions get a first-reached time.
