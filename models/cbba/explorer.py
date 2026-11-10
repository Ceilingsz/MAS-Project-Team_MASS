"""CBBA for the UNKNOWN-map interface (``cbba_explorer``), and its single-assignment case CBAA (``cbaa_explorer``).

One policy instance per drone, so a drone decides only from its own Observation and the messages it receives
(no shared object, no peeking at other drones' state). Run with SimConfig(known_map=False); add
reveal_obstacles_at_start=True for the same code with the map given (coverage still unknown), which is how the
experiment runner's ``map_prior="known"`` runs it.

Tasks are the candidate standing positions of the UnknownWorldView (the same ones the explorers use). A drone bids

    (UNKNOWN cells the spot would reveal, not counting cells already revealed by earlier spots in its bundle)
    / (travel time from the previous bundle spot, or from where it is, + delta)

with line of sight past the obstacles it knows and travel through known free space, unknown space treated as free
(as in ``greedy_explorer``). Consensus, conflict resolution and cascade drops are the shared ``CBBAState``
(cbba_core.py) unchanged: every decision a drone broadcasts its winners, winning bids and timestamps to drones within
r_comm, and merges what it hears (one hop per decision). With share=True (the default, as in the known-map
``cbba``'s share_coverage) it also sends its map belief and merges what it hears; share=False keeps beliefs private,
so a drone keeps bidding on space its teammates have already seen.

* A spot it reaches (or cannot reach) is completed and never bid on again by this drone.
* Bundle spots that would no longer reveal anything (seen on the way, by itself or, with sharing, by others) are
  pruned from the first such spot on, as in CBBA's known-map version.
* Done = no reachable, unvisited spot would reveal an unknown cell, whoever owns it. A drone whose remaining work has
  all been won by teammates waits (it is idle, not done).
* ``cbaa_explorer`` is the same with ``max_bundle_size=1``: each drone holds at most one task (CBAA).
* **Stale claims are released.** The simplified consensus rules in cbba_core can leave every drone believing that
  SOMEONE ELSE owns a task while nobody holds it (seen: three drones idle forever beside one unseen cell). So each
  message also carries the sender's current bundle, and a drone forgets a winner j it believes in when j's latest
  message shows j is not holding that task, or when it has not heard from j for ``claim_ttl`` seconds.
"""
from __future__ import annotations

import numpy as np

from mascov.common.model import AgentPolicy, DecentralisedModel, register
from mascov.common.types import Command, Message
from mascov.planning.navigator import Navigator
from models.cbba.cbba_core import CBBAState

KIND = "cbba"


class CBBAExplorerPolicy(AgentPolicy):
    map_interface = "unknown"

    def __init__(self, delta: float = 1.0, seed: int = 0, share: bool = True, max_bundle_size: int = 4,
                 claim_ttl: float = 3.0):
        self.delta, self.seed, self.share, self.max_bundle_size = delta, seed, share, int(max_bundle_size)
        self.claim_ttl = claim_ttl

    def reset(self, world, agent_id: int, n_agents: int) -> None:
        self.w, self.id, self.cfg, self.n_agents = world, agent_id, world.cfg, n_agents
        self.spots = world.candidates
        self.state = CBBAState(agent_id=agent_id, n_tasks=len(self.spots), n_agents=n_agents,
                               max_bundle_size=self.max_bundle_size)
        self.known = frozenset()
        self.space = world.space_for({})
        self.nav = Navigator(self.cfg, self.space.planner, self.spots)
        self.visited: set[int] = set()
        self.goal: int | None = None
        self.done = False
        self.heard: dict[int, tuple[float, frozenset]] = {}   # drone -> (time, the bundle it last reported)
        self.st = dict(map_rebuilds=0, cells_merged=0, cbba_msgs_in=0, goal_switches=0, tasks_completed=0,
                       stale_claims_released=0)

    # ------------------------------------------------------------------ map and geometry
    def _sync_space(self, belief) -> None:
        key = frozenset(belief.obstacles)
        if key != self.known:
            self.known, self.space = key, self.w.space_for(belief.obstacles)
            self.nav.set_planner(self.space.planner)
            self.st["map_rebuilds"] += 1

    def _spot_dist(self) -> np.ndarray:
        """Spot-to-spot path lengths for the current known space (cached on the shared space object)."""
        d = getattr(self.space, "_cbba_spot_dist", None)
        if d is None:
            d = self.space.planner.dist_matrix(self.spots, self.spots)
            self.space._cbba_spot_dist = d
        return d

    def _seen_from(self, v: int) -> np.ndarray:
        """Lattice cells spot v would see (sensing range, line of sight past known obstacles)."""
        sp = self.space
        return sp.pair_c[(sp.pair_v == v) & sp.pair_clear]

    def _gains_after(self, unknown: np.ndarray, bundle) -> np.ndarray:
        """Per spot: unknown cells it would reveal once every spot in `bundle` has been visited."""
        if not bundle:
            return self.space.gains(unknown)
        u = unknown.copy()
        for v in bundle:
            u[self._seen_from(v)] = False
        return self.space.gains(u)

    def _static_blocked(self, obs) -> set[int]:
        """Spots too close to a neighbour that is standing still (sensed, so allowed)."""
        pts = [p for j, p in obs.neighbors.items() if np.linalg.norm(obs.neighbor_vels.get(j, np.zeros(2))) < 0.1]
        if not pts:
            return set()
        d = np.linalg.norm(self.spots[:, None, :] - np.asarray(pts)[None], axis=2).min(axis=1)
        return set(np.flatnonzero(d < 2 * self.cfg.d_safe).tolist())

    def _release_stale_claims(self, t: float) -> None:
        """Forget winners that are not actually holding the task (see the module docstring)."""
        st = self.state
        for v in np.flatnonzero((st.y >= 0) & (st.y != self.id)):
            j = int(st.y[v])
            h = self.heard.get(j)
            if h is None or t - h[0] > self.claim_ttl or int(v) not in h[1]:
                st._reset(int(v))
                self.st["stale_claims_released"] += 1

    # ------------------------------------------------------------------ decision
    def step(self, obs) -> Command:
        belief, st = obs.belief, self.state
        for m in obs.inbox:
            if m.kind != KIND or m.sender == self.id:
                continue
            y, z, s, bel, bundle = m.payload
            if self.share and bel is not None:
                self.st["cells_merged"] += belief.merge(bel)
            st.process_message(m.sender, y, z, s)
            self.heard[m.sender] = (obs.t, frozenset(bundle))
            self.st["cbba_msgs_in"] += 1
        self._release_stale_claims(obs.t)
        self._sync_space(belief)
        plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished)

        # 1. the task being flown to: reached, or cannot be reached
        if self.goal is not None and (self.nav.arrived or self.nav.mode == "blocked"):
            self.visited.add(self.goal)
            st.complete_task(self.goal)
            self.st["tasks_completed"] += 1
            self.goal = None

        # 2. prune the bundle from the first spot that would no longer reveal anything
        unknown = belief.unknown
        cum = unknown.copy()
        valid = self.space.valid

        def still_useful(v: int) -> int:
            if not valid[v]:
                return 0
            seen = self._seen_from(v)
            g = int(cum[seen].sum())
            if g > 0:
                cum[seen] = False
            return g

        st.validate_bundle(still_useful)

        # 3. bid for more tasks
        speed = self.cfg.plan_speed
        travel_now = self.space.distances(obs.pos) / speed
        spot_travel = self._spot_dist() / speed
        exclude = self.visited | set(np.flatnonzero(~valid).tolist())
        blocked = self._static_blocked(obs)
        cache: dict[tuple, np.ndarray] = {}

        def score(v: int, bundle) -> float:
            key = tuple(bundle)
            g = cache.get(key)
            if g is None:
                g = cache[key] = self._gains_after(unknown, bundle)
            if g[v] <= 0:
                return 0.0
            t = travel_now[v] if not bundle else spot_travel[bundle[-1], v]
            return float(g[v] / (t + self.delta)) if np.isfinite(t) else 0.0

        # Spots next to a parked drone are skipped while there is other work; when there is none, they are allowed
        # (otherwise a drone waits forever beside the last spot a finished teammate is parked on).
        st.construct_bundle(score, exclude=exclude | blocked)
        if not st.bundle and blocked:
            st.construct_bundle(score, exclude=exclude)

        # 4. fly to the first task of the bundle
        nxt = st.next_task()
        if nxt != self.goal:
            if self.goal is not None:
                self.st["goal_switches"] += 1
            self.goal = nxt
            self.nav.set_goal(None if nxt is None else self.spots[nxt])
            plan = self.nav.update(obs.t, obs.pos, obs.vel, obs.plan_finished)
        if nxt is None:
            left = self.space.gains(unknown) > 0
            usable = valid.copy()
            if self.visited:
                usable[list(self.visited)] = False
            self.done = not (left & usable & np.isfinite(travel_now)).any()
        else:
            self.done = False

        msgs = []
        if self.n_agents > 1 and obs.comm_neighbors:
            y, z, s = st.consensus_payload()
            msgs.append(Message(self.id, KIND, (y, z, s, belief.copy() if self.share else None, list(st.bundle))))
        return Command(plan=plan, messages=msgs, done=self.done)

    def stats(self) -> dict:
        return dict(self.nav.stats, **self.st)


@register("cbba_explorer")
def CBBAExplorer(delta: float = 1.0, seed: int = 0, share: bool = True, max_bundle_size: int = 4,
                 claim_ttl: float = 3.0):
    return DecentralisedModel(CBBAExplorerPolicy, name="cbba_explorer", delta=delta, seed=seed, share=share,
                              max_bundle_size=max_bundle_size, claim_ttl=claim_ttl)


@register("cbaa_explorer")
def CBAAExplorer(delta: float = 1.0, seed: int = 0, share: bool = True):
    """CBAA: the single-assignment case of CBBA (each drone holds at most one task)."""
    return DecentralisedModel(CBBAExplorerPolicy, name="cbaa_explorer", delta=delta, seed=seed, share=share,
                              max_bundle_size=1)
