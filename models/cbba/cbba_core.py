"""CBBA state machine: per-agent consensus, bundle construction, and cascade drops.

Implements the Consensus-Based Bundle Algorithm (Choi et al., 2009) adapted for
multi-drone coverage.  Each agent maintains:

  * bundle   – ordered list of viewpoint tasks to execute
  * y        – winner array (who owns each task)
  * z        – winning bid array (highest bid for each task)
  * s        – timestamp vector (latest info age from each agent)

The two alternating phases are:
  1. **Consensus** – exchange (y, z, s) with neighbours, resolve conflicts (higher bid wins),
     cascade-drop tasks from the bundle that were outbid.
  2. **Bundle construction** – greedily add tasks whose marginal score exceeds the current
     winning bid, up to ``max_bundle_size``.
"""
from __future__ import annotations

import numpy as np


class CBBAState:
    """CBBA state for a single agent."""

    def __init__(self, agent_id: int, n_tasks: int, n_agents: int,
                 max_bundle_size: int = 5):
        self.agent_id = agent_id
        self.n_tasks = n_tasks
        self.n_agents = n_agents
        self.max_bundle_size = max_bundle_size

        self.bundle: list[int] = []                             # viewpoint indices in visit order
        self.y = np.full(n_tasks, -1, dtype=int)                # winner[j] = agent id (-1 = none)
        self.z = np.zeros(n_tasks, dtype=float)                 # winning bid[j]
        self.s = np.zeros(n_agents, dtype=float)                # logical timestamps

    # ------------------------------------------------------------------ consensus

    def process_message(self, sender_id: int,
                        y_k: np.ndarray, z_k: np.ndarray, s_k: np.ndarray) -> None:
        """Process a consensus message from *sender_id* and cascade-drop outbid tasks."""
        outbid: set[int] = set()

        for j in range(self.n_tasks):
            sender_winner = int(y_k[j])
            my_winner = int(self.y[j])

            # --- sender thinks I win → I know best about myself, skip
            if sender_winner == self.agent_id:
                continue

            # --- sender claims to win task j ---------------------------------
            if sender_winner == sender_id:
                if my_winner == self.agent_id:
                    # I also claim – compare bids
                    if z_k[j] > self.z[j] + 1e-9:
                        self._accept(j, sender_id, z_k[j])
                        if j in self.bundle:
                            outbid.add(j)
                    elif abs(z_k[j] - self.z[j]) < 1e-9 and sender_id < self.agent_id:
                        # tie-break: lower agent id wins
                        self._accept(j, sender_id, z_k[j])
                        if j in self.bundle:
                            outbid.add(j)
                elif my_winner == sender_id:
                    # I already knew sender wins – accept updated bid
                    self._accept(j, sender_id, z_k[j])
                else:
                    # I think unassigned or someone else wins
                    if z_k[j] > self.z[j] - 1e-9:
                        self._accept(j, sender_id, z_k[j])
                        if j in self.bundle:
                            outbid.add(j)

            # --- sender thinks task j is unassigned ---------------------------
            elif sender_winner == -1:
                if my_winner == sender_id:
                    # sender dropped its claim
                    self._reset(j)
                elif my_winner != self.agent_id and my_winner != -1:
                    # I think someone else (m) wins; sender may have newer info
                    if s_k[my_winner] > self.s[my_winner] + 1e-9:
                        self._reset(j)
                        if j in self.bundle:
                            outbid.add(j)

            # --- sender thinks a third party m wins ---------------------------
            else:
                m = sender_winner
                if my_winner == self.agent_id:
                    if z_k[j] > self.z[j] + 1e-9:
                        self._accept(j, m, z_k[j])
                        if j in self.bundle:
                            outbid.add(j)
                    elif abs(z_k[j] - self.z[j]) < 1e-9 and m < self.agent_id:
                        self._accept(j, m, z_k[j])
                        if j in self.bundle:
                            outbid.add(j)
                elif my_winner == m:
                    # both agree on winner – take higher bid
                    if z_k[j] > self.z[j]:
                        self._accept(j, m, z_k[j])
                elif my_winner == sender_id or my_winner == -1:
                    self._accept(j, m, z_k[j])
                else:
                    # I think yet another agent wins
                    if z_k[j] > self.z[j] + 1e-9:
                        self._accept(j, m, z_k[j])

        # propagate timestamps
        for a in range(self.n_agents):
            self.s[a] = max(self.s[a], s_k[a])

        if outbid:
            self._cascade_drop(outbid)

    # ------------------------------------------------------------------ helpers

    def _accept(self, task: int, winner: int, bid: float) -> None:
        self.y[task] = winner
        self.z[task] = bid

    def _reset(self, task: int) -> None:
        self.y[task] = -1
        self.z[task] = 0.0

    def _cascade_drop(self, outbid_tasks: set[int]) -> None:
        """Drop outbid tasks and everything appended after the earliest one.

        Tasks removed from the bundle that this agent owned are reset to unassigned.
        Tasks owned by someone else keep their winner info.
        """
        if not outbid_tasks or not self.bundle:
            return
        earliest = len(self.bundle)
        for t in outbid_tasks:
            try:
                idx = self.bundle.index(t)
                earliest = min(earliest, idx)
            except ValueError:
                pass
        if earliest >= len(self.bundle):
            return

        to_drop = self.bundle[earliest:]
        self.bundle = self.bundle[:earliest]
        for t in to_drop:
            if self.y[t] == self.agent_id:
                self._reset(t)

    # ------------------------------------------------------------------ bundle ops

    def validate_bundle(self, gain_fn) -> None:
        """Drop the first task whose marginal gain has fallen to zero, and everything after it.

        ``gain_fn(task_id) -> int`` returns the marginal gain of *task_id* w.r.t. the
        agent's current coverage belief (the bundle context is handled internally by
        the caller).
        """
        drop_from = None
        for idx, t in enumerate(self.bundle):
            if gain_fn(t) <= 0:
                drop_from = idx
                break
        if drop_from is not None:
            to_drop = self.bundle[drop_from:]
            self.bundle = self.bundle[:drop_from]
            for t in to_drop:
                if self.y[t] == self.agent_id:
                    self._reset(t)

    def construct_bundle(self, score_fn, exclude: set[int] | None = None) -> None:
        """Greedily add profitable tasks until the bundle is full.

        ``score_fn(task_id, current_bundle) -> float`` returns the bid score.
        Tasks in *exclude* are never considered.
        """
        exclude = exclude or set()
        bundle_set = set(self.bundle)

        while len(self.bundle) < self.max_bundle_size:
            best_task, best_score = -1, -1.0

            for j in range(self.n_tasks):
                if j in bundle_set or j in exclude:
                    continue
                # already in our allocation but not in bundle – shouldn't happen
                if self.y[j] == self.agent_id and j not in self.bundle:
                    continue

                score = score_fn(j, self.bundle)
                if score <= 1e-12:
                    continue

                # can I outbid the current winner?
                can_outbid = (
                    self.y[j] == -1
                    or score > self.z[j] + 1e-9
                    or (abs(score - self.z[j]) < 1e-9 and self.agent_id < self.y[j])
                )
                if can_outbid and score > best_score:
                    best_score = score
                    best_task = j

            if best_task < 0:
                break

            self.bundle.append(best_task)
            bundle_set.add(best_task)
            self.y[best_task] = self.agent_id
            self.z[best_task] = best_score
            self.s[self.agent_id] += 1.0          # logical clock tick

    # ------------------------------------------------------------------ execution

    def next_task(self) -> int | None:
        """First task in the bundle, or *None* if empty."""
        return self.bundle[0] if self.bundle else None

    def complete_task(self, task_id: int) -> None:
        """Remove *task_id* from the bundle and release ownership."""
        if task_id in self.bundle:
            self.bundle.remove(task_id)
        if self.y[task_id] == self.agent_id:
            self._reset(task_id)
        self.s[self.agent_id] += 1.0

    def consensus_payload(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(y, z, s) to broadcast to neighbours."""
        return self.y.copy(), self.z.copy(), self.s.copy()
