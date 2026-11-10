"""CBBA (Consensus-Based Bundle Algorithm) for multi-drone search and rescue coverage.

Decentralised market-based task allocation supporting both known and unknown environments:
  - Each drone maintains a bundle of candidate viewpoints to visit.
  - Bids are based on marginal coverage gain over travel time.
  - Drones exchange winning bids and winners with neighbours within communication range.
  - In unknown maps, discovered obstacles are shared and incorporated into local visibility graphs.
  - Conflicts are resolved using consensus rules with cascade drops.
  - Drones navigate safely using local Navigator collision avoidance.
"""
from __future__ import annotations

import numpy as np

from mascov.common.model import CentralModel, register
from mascov.common.types import Command, Message
from mascov.planning.navigator import Navigator, Neighbor
from models.cbba.cbba_core import CBBAState
from models.cbba.scoring import CBBAScorer


@register("cbba")
class CBBAModel(CentralModel):
    centralised = False  # Decentralised: agents use local beliefs, bundles, and messages

    def __init__(self, delta: float = 1.0, max_bundle_size: int = 4,
                 margin: float = 0.5, share_coverage: bool = True):
        self.delta = delta
        self.max_bundle_size = max_bundle_size
        self.margin = margin
        self.share_coverage = share_coverage

    def reset(self, world, agent_ids: list[int]) -> None:
        self.w = world
        self.vp = world.viewpoints
        self.ids = list(agent_ids)
        self.n_agents = len(self.ids)
        self.n_tasks = len(self.vp.positions)
        self.known_map = getattr(world, "known_map", True)

        self.scorer = CBBAScorer(world, delta=self.delta)
        self.states = {
            i: CBBAState(agent_id=i, n_tasks=self.n_tasks, n_agents=self.n_agents,
                         max_bundle_size=self.max_bundle_size)
            for i in self.ids
        }

        # Track per-agent known obstacles and planners (for unknown-map support)
        self.known_obstacles: dict[int, set[int]] = {i: set() for i in self.ids}
        self.planners: dict[int, any] = {}
        self.bounds: dict[int, any] = {}

        for i in self.ids:
            if self.known_map:
                self.planners[i] = world.planner
                self.bounds[i] = world._vp_bound
            else:
                p, b = world.create_planner(set())
                self.planners[i] = p
                self.bounds[i] = b

        self.nav = {
            i: Navigator(world.cfg, self.planners[i], self.vp.positions,
                         margin=self.margin, seed=i)
            for i in self.ids
        }
        self.visited: dict[int, set[int]] = {i: set() for i in self.ids}
        self.done_flags: dict[int, bool] = {i: False for i in self.ids}

    def _static_blocked(self, i: int, obs: dict) -> list[int]:
        """Viewpoints blocked by stationary neighbours."""
        statics = [
            obs[j].pos for j in self.ids
            if j != i and j in obs and np.linalg.norm(obs[j].vel) < 0.1
        ]
        if not statics:
            return []
        d = np.linalg.norm(self.vp.positions[:, None, :] - np.asarray(statics)[None], axis=2).min(axis=1)
        return list(np.flatnonzero(d < self.nav[i].static_dist + 0.1))

    def _priority(self, i: int) -> tuple[int, int]:
        has_task = self.states[i].next_task() is not None
        return (0 if has_task else 1, i)

    def _neighbors(self, i: int, obs: dict) -> list[Neighbor]:
        pi = self._priority(i)
        o = obs[i]
        nbrs = []
        for j, p in o.neighbors.items():
            vel = o.neighbor_vels.get(j, np.zeros(2))
            pj = self._priority(j)
            higher = pj < pi
            nbrs.append(Neighbor(id=j, pos=p, vel=vel, higher=higher))
        return nbrs

    def _update_agent_planner(self, i: int, new_obs_set: set[int]) -> None:
        """Update planner and navigator when new obstacles are discovered."""
        if not new_obs_set or self.known_map:
            return
        diff = new_obs_set - self.known_obstacles[i]
        if diff:
            self.known_obstacles[i] |= diff
            p, b = self.w.create_planner(self.known_obstacles[i])
            self.planners[i] = p
            self.bounds[i] = b
            self.nav[i].planner = p
            # Invalidate any bundle tasks that are now blocked/unreachable
            st = self.states[i]
            cur_target = st.next_task()
            if cur_target is not None:
                d = p.distance(self.nav[i].planner.allowed.centroid.coords[0], self.vp.positions[cur_target])
                if not np.isfinite(d):
                    st.complete_task(cur_target)
                    self.nav[i].set_goal(None)

    def step(self, obs: dict) -> dict[int, Command]:
        cmds = {}

        # ----------------------------------------------------------------------
        # Phase 1: Process incoming messages and new obstacle sensing
        # ----------------------------------------------------------------------
        for i in sorted(obs):
            o = obs[i]
            st = self.states[i]

            # Process newly sensed obstacles
            if not self.known_map and o.new_obstacles:
                self._update_agent_planner(i, set(o.new_obstacles))

            # Process incoming neighbor messages
            for msg in o.inbox:
                if msg.kind == "cbba":
                    sender_id = msg.sender
                    payload = msg.payload
                    y_k, z_k, s_k = payload[0], payload[1], payload[2]
                    cov_k = payload[3] if len(payload) > 3 else None
                    obs_k = payload[4] if len(payload) > 4 else None

                    if self.share_coverage and cov_k is not None:
                        o.belief.mark(np.flatnonzero(cov_k))

                    if not self.known_map and obs_k:
                        self._update_agent_planner(i, set(obs_k))

                    st.process_message(sender_id, y_k, z_k, s_k)

        # ----------------------------------------------------------------------
        # Phase 2: Check arrivals and validate existing bundles
        # ----------------------------------------------------------------------
        for i in sorted(obs):
            o = obs[i]
            st = self.states[i]
            nav = self.nav[i]
            current_target = st.next_task()

            # Check if arrived at current target viewpoint
            if current_target is not None and nav.arrived:
                self.visited[i].add(current_target)
                st.complete_task(current_target)
                nav.set_goal(None)

            # Validate remaining tasks in bundle
            # If all targets that a viewpoint was supposed to cover are already covered,
            # drop it from bundle to avoid redundant travel
            validator = self.scorer.make_validator(o.belief.covered)
            st.validate_bundle(validator)

        # ----------------------------------------------------------------------
        # Phase 3: Bundle Construction (Local auction / Greedy bidding)
        # ----------------------------------------------------------------------
        for i in sorted(obs):
            o = obs[i]
            st = self.states[i]

            blocked = self._static_blocked(i, obs)
            exclude = self.visited[i] | set(blocked)

            planner_i = self.planners[i]
            bound_i = self.bounds[i]

            def score_fn(candidate_v: int, cur_bundle: list[int]) -> float:
                return self.scorer.compute_score(
                    candidate_v, o.pos, o.belief.covered, cur_bundle,
                    planner=planner_i, bound=bound_i
                )

            st.construct_bundle(score_fn, exclude=exclude)

        # ----------------------------------------------------------------------
        # Phase 4: Navigation, Command Generation, and Message Broadcast
        # ----------------------------------------------------------------------
        all_done = True
        for i in sorted(obs):
            o = obs[i]
            st = self.states[i]
            nav = self.nav[i]

            next_vp = st.next_task()
            if next_vp is not None:
                nav.set_goal(self.vp.positions[next_vp])
                self.done_flags[i] = False
            else:
                nav.set_goal(None)
                # Check if mission is complete from this agent's perspective
                uncovered = self.vp.coverable & ~o.belief.covered
                if not uncovered.any():
                    self.done_flags[i] = True
                else:
                    # Check if any reachable unvisited viewpoint adds gain
                    any_gain = any(
                        self.scorer.marginal_gain(v, o.belief.covered) > 0
                        for v in range(self.n_tasks)
                        if v not in self.visited[i]
                    )
                    self.done_flags[i] = not any_gain

            if not self.done_flags[i]:
                all_done = False

            nbrs = self._neighbors(i, obs)
            plan = nav.update(o.t, o.pos, o.vel, o.plan_finished, nbrs, idle=(next_vp is None))

            # Outgoing consensus message (only if multiple agents exist)
            if self.n_agents > 1:
                y, z, s = st.consensus_payload()
                cov = o.belief.covered.copy() if self.share_coverage else None
                obs_set = set(self.known_obstacles[i]) if not self.known_map else None
                msg = Message(sender=i, kind="cbba", payload=(y, z, s, cov, obs_set))
                out_msgs = [msg]
            else:
                out_msgs = []

            cmds[i] = Command(plan=plan, messages=out_msgs, done=self.done_flags[i])

        if all_done:
            for i in cmds:
                cmds[i].done = True

        return cmds

    def stats(self) -> dict:
        out: dict = {}
        for n in self.nav.values():
            for k, v in n.stats.items():
                out[k] = out.get(k, 0) + v
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}
