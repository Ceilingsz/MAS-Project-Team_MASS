# MAS-Project: multi-agent drone coverage

Simulation environment (and, in `models/`, the algorithms) for covering a continuous 2D workspace
with polygonal obstacles using several single-integrator drones.

## Setup

```bash
pip install -e ".[dev,dmpc]"     # numpy, shapely>=2, matplotlib, scipy, pytest, osqp (D-MPC)
python -m pytest -q              # run from the repository root (about 4 min; see "Testing")
PYTHONPATH=src:. python scripts/demo_greedy.py sparse 0
```

Run everything from the repository root so that `models/` is importable.
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
| `test_cbba.py`, `test_cbba_rigorous.py` | CBBA consensus (higher bid wins, tie-breaking, cascade drops, stale messages), bundle scoring, the hidden-obstacle `WorldView`, CBBA end-to-end runs |
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

Scenarios (all with `eps_track` = 0.45 and 2 m start spacing, so every model faces the same world; these values date from
when the suite also graded the since-removed STL planners, and every recorded gap was measured with them): `solo` (1 drone, `rooms_and_door`), `team` (3 drones, `rooms_and_door`), `clutter` (3 drones, `cluttered` seed 1),
`low_comm` (`team` with messages and neighbour sensing limited to 1.5 m; decentralised models only). Each run is simulated
once per session and shared by all criteria.

**Known gaps.** A shortfall that is understood is listed in `KNOWN_GAPS` in `test_model_quality.py` with its measured
numbers. It runs as a *strict* xfail: the suite stays green and lists it as `XFAIL`, and if the model improves so the
criterion passes, the test fails as `XPASS(strict)` until the entry is deleted, so the table cannot go stale. Do not loosen
a threshold to make a model pass; record the gap.

```bash
python -m pytest -q                                   # everything
python -m pytest -q -m "not quality"                  # component tests only (~2 min)
python -m pytest -q tests/test_model_quality.py -rx   # the report card, with each known gap and its numbers
python -m pytest -q tests/test_model_quality.py --runxfail   # every gap as a plain failure
```

**Report card** (measured 2026-11-10; `-` = not applicable; details and numbers in `KNOWN_GAPS`):

| model | completes | obstacle safe | drone safe | tracking | knows when done | efficient | team speedup | reproducible |
|---|---|---|---|---|---|---|---|---|
| `greedy_single` | pass | pass | - | pass | pass | pass | - | - |
| `greedy_centralised` | pass | pass | **fail**: clutter (pass: team) | pass | pass | - | pass | pass |
| `greedy_uncoordinated` | pass | pass | **fail**: team, clutter, low_comm | pass | **fail**: team, clutter, low_comm (pass: solo) | **fail**: team, clutter (pass: solo) | **fail**: team | pass |
| `static_partition` | pass | pass | **fail**: clutter (pass: team, low_comm) | pass | pass | pass | pass | pass |
| `greedy+dmpc` | pass | pass | pass | pass | **fail**: team, clutter, low_comm (pass: solo) | **fail**: team, clutter (pass: solo) | **fail**: team | pass |
| `frontier+dmpc` | pass | pass | pass | pass | **fail**: team, clutter, low_comm (pass: solo) | **fail**: team, clutter (pass: solo) | **fail**: team | pass |
| `cbba` | **fail**: clutter (pass: solo, team, low_comm) | pass | **fail**: team, clutter, low_comm | pass | **fail**: clutter, low_comm (pass: solo, team) | **fail**: clutter (pass: solo, team) | pass | pass |
| `frontier_greedy` | pass | pass | **fail**: team, clutter, low_comm | pass | **fail**: team, clutter, low_comm (pass: solo) | **fail**: team, clutter (pass: solo) | pass | pass |
| `frontier_greedy_shared` | pass | pass | **fail**: team, clutter, low_comm | pass | pass | **fail**: team, clutter (pass: solo) | pass | pass |

What it says:
* **Drone-drone safety is the main gap.** Every model built on the shared `Navigator` collided somewhere, including the
  centralised reference on `clutter`. Only the D-MPC models (`greedy+dmpc`, `frontier+dmpc`) were collision-free in every graded scenario.
* **Coordination decides speed.** Only the models that split the work (`greedy_centralised`, `static_partition`) are within
  1.5 x of the reference with 3 drones; `greedy+dmpc` uses the uncoordinated greedy goals, so it is safe but no faster than
  `greedy_uncoordinated`, and its 3-drone team is barely faster than one drone (0.91 x).
* **CBBA splits the work best among the decentralised methods** (3 drones 75.7 s = 1.37 x the reference; 0.37 x its
  own one-drone time) but collides (Navigator), stalls at 97.6% on `clutter`, and with messages limited to 1.5 m two of three
  drones never conclude they are done.
* **Without sharing, drones notice late** that the map is covered (21-96 s after the fact).
* `low_comm` (1.5 m) changed nothing for models that never communicate, as expected; it is the lever for communicating models.

## The world model (what is simulated)

* **Space:** continuous rectangle `[0,W] x [0,H]`, metres. **Obstacles:** static *convex* polygons
  (decompose non-convex ones first). `geometry.polygons.halfspaces(poly)` gives `H x <= b` (used by the D-MPC obstacle constraints).
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
  maps/        seeded benchmarks: empty, sparse, corridor, rooms_and_door, cluttered
  viz/         plots
models/        one folder per algorithm: greedy_single (B0), greedy_centralised (B2),
               greedy_uncoordinated (B3), static_partition (B4),
               frontier_greedy, frontier_greedy_shared (unknown-map explorers),
               cbba (CBBA task allocation),
               greedy_dmpc (greedy+dmpc, frontier+dmpc: D-MPC trajectories)
               `import models` registers them all (the D-MPC models only if OSQP is installed)
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

> **Dynamics changed.** Every number measured below (baselines, BVC, unknown-map, and the `eps_track` / `plan_speed_ratio` values)
> was produced with the earlier **double-integrator** model (`a_max` = 2 m/s^2, PD tracker). The simulator now uses
> **single-integrator** dynamics (velocity is the control; no acceleration limit, no braking distance). Expect smaller tracking
> error and different times, so the tables must be regenerated before they are quoted. The qualitative conclusions may or may
> not carry over. The tracker is now `u = v_ref + gain * (p_ref - p)` (`SimConfig.tracker_gain`).

`WorldConfig`: 12 x 12 m, `R_s`=1.5, `r_b`=0.15 (so `d_safe`=0.3), `v_max`=1, `dt`=0.1,
planners fly at 0.6 `v_max`, `eps_track`=0.3.

`eps_track` is **measured, not assumed**: the largest deviation from the reference path over all benchmark
maps is ~0.26 m at `plan_speed_ratio`=0.6 (it was ~0.48 m at 0.8), and `tests/test_sim.py` fails if
a run exceeds `eps_track`. If you change the tracker or planning speed, re-measure
(`RunResult.max_tracking_error`) before trusting any clearance guarantee.

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
> `greedy+dmpc` or `frontier+dmpc`. The collision numbers below were measured with the old rule.

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
asserted in tests). Any planner that relies on a clearance margin must have its own epsilon measured.

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

## CBBA task allocation (`models/cbba`)

`cbba`: the Consensus-Based Bundle Algorithm (Choi et al., 2009) over viewpoint tasks. Each drone keeps a **bundle** of up to
`max_bundle_size` viewpoints (default 4) in visit order, bids the marginal gain (new targets not covered by its belief or by
earlier bundle entries) divided by the travel time from the previous bundle entry plus `delta`, and every decision broadcasts its
winners, winning bids and timestamps to drones within `r_comm`. Higher bid wins (ties: lower id); a drone outbid on a task drops it
and everything after it in its bundle (cascade drop), then rebuilds. Bundle entries that no longer add anything are pruned.
`share_coverage=True` (default) also sends the coverage belief. Drones fly with the shared `Navigator`.
`cbba_core.py` (state machine), `scoring.py` (bids), `model.py`.

* **Hidden-obstacle mode** (`SimConfig(hide_obstacles=True)`): the model gets `WorldView(known_map=False)`: no obstacles and a
  planner over the empty workspace; it rebuilds its planner with `WorldView.create_planner(ids)` as it senses obstacles, and
  shares the ids it knows. Targets and viewpoints are still given, so this is weaker than the unknown-map interface
  (`known_map=False`), which CBBA does not implement. On `cluttered` (N = 2) hiding the obstacles costs 127.7 s vs 103.9 s; on
  `sparse` and `rooms_and_door` nothing, because the obstacles are in sensing range before they matter.
* **Scripts and results:** `scripts/run_known_vs_unknown.py` (B2, B3, B4 and CBBA with the obstacles given and hidden; B2-B4 do
  not re-plan on discovery) and `scripts/plot_cbba_analysis.py`. `docs/known_vs_unknown_results.csv`,
  `docs/scaling_6_8_drones.csv` and the two figures were measured in the earlier root-level codebase (double-integrator drones,
  the Navigator that still yielded to neighbours), so they are not directly comparable with numbers measured here.
* **Caveats:** the model object holds every drone's CBBA state; two helpers read other drones' state directly
  (`_priority` uses each neighbour's bundle, `_static_blocked` every drone's position), so it is not strictly decentralised.
  Quality-suite grades are in the report card above.

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
* Compute: each drone keeps one `DMPCSolver`. The QP is built sparse with a fixed pattern (a capacity of soft-constraint
  rows; spare rows have zero coefficients and never bind) and one OSQP object is updated in place every step and warm
  started, instead of being built and set up again. About 1.3 ms per solve (was 7.4 ms); whole runs 3.8-4.6x faster
  (3 drones on `rooms_and_door`: 50 s -> 13 s; 1 drone: 17 s -> 3.8 s), with the same coverage times. Answers agree with
  a fresh setup to the solver tolerance (millimetres; `tests/test_dmpc.py`). Half of the old cost was not the maths:
  `osqp.OSQP()` without `algebra="builtin"` probes for GPU/MKL backends by importing them on every call.
  `layer 1` is replaceable: subclass `DMPCAgentPolicy` and override `select_goal` (for `cbba+dmpc`).
* **Unknown map: `frontier+dmpc`** (`planning/agent_frontier_dmpc.py`, run with `SimConfig(known_map=False)`). Goals come
  from the frontier explorer's rule (most UNKNOWN cells revealed per second of travel, unknown space treated as free); the
  D-MPC obstacle constraints and the reference path use only the obstacles the drone knows, rebuilt whenever it senses a new
  one (`map_rebuilds` in `RunResult.extra`). `share=True` also sends the map belief with each prediction and merges what
  arrives (`belief_merged`). With `reveal_obstacles_at_start=True` the same algorithm gets the map: the reference for
  experiment `E3D` (`python scripts/run_experiments.py E3D --jobs 8`), which compares the two on 5 maps x N=2,4 x 3 seeds.

### D-MPC with and without the map (experiment `E3D`)

`python scripts/run_experiments.py E3D --jobs 8`, raw rows in `docs/e3d_dmpc_known_vs_unknown.csv`. 5 maps x N = 2, 4 x seeds
0-2, each run with the map given (`map_prior=known`: obstacles loaded at the start, coverage still unknown) and not given.
150 runs, all complete. `empty`, `corridor` and `rooms_and_door` are fixed maps on which the 3 seeds give identical runs, so the
statistics below are over the **18 distinct** known/unknown pairs, not 30. Ratio = coverage time (100%) unknown / known.

| | unknown / known (mean +- SE) | per map |
|---|---|---|
| `frontier+dmpc` (no sharing) | **1.051 +- 0.034** | empty 1.00, cluttered 1.00, corridor 1.05, sparse 1.08, rooms_and_door 1.17 |
| `frontier+dmpc`, `share=True` | **1.010 +- 0.033** | empty 1.00, cluttered 1.01, corridor 0.93, rooms_and_door 0.95, sparse 1.06 |

* **Not knowing the map costs D-MPC little:** about +5% without sharing (1.5 SE from zero, so not established) and nothing
  measurable with sharing. Same pattern as the frontier greedy explorer above: coverage must visit everywhere anyway, so discovering
  the obstacles is mostly free. The clearest cost is `rooms_and_door` without sharing (+17%, two distinct runs),
  plausibly because a drone plans through a wall it has not seen yet and must turn back to the door (not checked run by run).
* **Safety does not suffer from the unknown map:** 0 drone-drone and 0 obstacle episodes in all 150 runs; closest obstacle approach
  0.44 m (limit 0.15 m), closest pair 0.35 m (limit 0.30 m); the QP never failed. With the map unknown a drone rebuilt its map
  13.7 times per run on average.
* **Sharing matters far more than the map:** with the map unknown, sharing beliefs cuts coverage time to 0.72 x on average
  (0.83 x at N = 2, 0.61 x at N = 4).
* With the map given, `frontier+dmpc` and the known-map `greedy+dmpc` are equally fast (ratio 0.99, range 0.75-1.17), so the
  frontier goal rule is not what makes the unknown-map version cost anything.
* Caveats: 12 x 12 m maps, N <= 4, whole-polygon obstacle reveal (an obstacle is known entirely once any part is seen).

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

* **Unknown map: whole-polygon reveal only**, and every known-map model still needs the known-map interface (they use the true
  obstacles and viewpoints); `frontier+dmpc` is the D-MPC variant for unknown maps.
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
Algorithms that are not written yet (C-ADMM, CBBA + D-MPC, the integrated method) are listed there under agreed model
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
