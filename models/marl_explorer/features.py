"""What the learned goal policy sees, computed only from one drone's own Observation (no ground truth).

One decision = choose one of the V candidate standing positions of the ``UnknownWorldView``. Every candidate j gets a
feature vector (``N_FEAT`` numbers); the actor scores all of them jointly. The same quantities the hand-written
explorers use are included (UNKNOWN cells revealed, travel time, their ratio), so the policy can start as greedy and
learn what greedy ignores: where teammates are and where they are going, how much unknown space lies beyond a spot,
and how much a spot's view overlaps the views of teammates' goals.

Per candidate j (index in the feature vector):
  0  gain_j / 25                          UNKNOWN cells j reveals (line of sight past KNOWN obstacles)
  1  gain_j / max gain                    relative gain
  2  min(travel_j / 30 s, 3)              shortest-path travel time through known free space
  3  log1p(travel_j) / 4
  4  score_j / max score                  greedy score = gain / (travel + delta)
  5  log(score_j / max score), >= -6      the greedy log-score (also the residual base logit)
  6  distance from j to the nearest teammate goal / 12 m      (1 if no goal heard)
  7  fraction of j's revealed cells that a teammate goal also reveals
  8  distance from j to the nearest teammate / 12 m           (1 if none known)
  9  (my straight-line distance to j - nearest teammate's) / 12 m   > 0: a teammate is closer
 10  UNKNOWN cells within 3 m of j / cells in that disc       unknown mass beyond the spot
 11  dx / 12 m, 12 dy / 12 m              j relative to me
 13  straight-line / path distance        detour ratio (1 = straight line)
 14  number of teammate goals within 2 R_s of j / (N - 1)

Context (``N_CTX``): my UNKNOWN fraction, t / 300 s, N / 8, teammates known / (N-1), goals heard / (N-1),
1 / (1 + r_comm), usable fraction of candidates, max gain / 25.
"""
from __future__ import annotations

import numpy as np

N_FEAT = 15
N_CTX = 8
LOOKAHEAD_R = 3.0
MAX_GAIN = 25.0          # about the number of 0.5 m cells in a 1.4 m sensing disc


class Statics:
    """Map-independent tables for one view (candidates x lattice), computed once and cached on the view."""

    def __init__(self, view):
        cand, lat = view.candidates, view.lattice
        d = np.linalg.norm(cand[:, None, :] - lat[None, :, :], axis=2)
        self.near = (d <= LOOKAHEAD_R)
        self.near_n = np.maximum(1, self.near.sum(axis=1)).astype(float)
        self.cand_d = np.linalg.norm(cand[:, None, :] - cand[None, :, :], axis=2)


def statics_for(view) -> Statics:
    st = getattr(view, "_marl_statics", None)
    if st is None:
        st = Statics(view)
        try:
            view._marl_statics = st
        except AttributeError:   # pragma: no cover
            pass
    return st


def reveal_matrix(space, unknown: np.ndarray) -> np.ndarray:
    """(V, G) bool: cell c is UNKNOWN and visible from candidate v (line of sight past KNOWN obstacles)."""
    m = np.zeros((space.n_candidates, len(unknown)), dtype=bool)
    hit = unknown[space.pair_c] & space.pair_clear
    m[space.pair_v[hit], space.pair_c[hit]] = True
    return m


def candidate_features(*, space, statics: Statics, unknown: np.ndarray, pos: np.ndarray, travel: np.ndarray,
                       usable: np.ndarray, candidates: np.ndarray, team_pos: list, team_goals: list,
                       n_agents: int, t: float, r_comm: float, r_sense: float, delta: float, extent: float = 12.0):
    """Returns (feats (V, N_FEAT) float32, ctx (N_CTX,) float32, mask (V,) bool, base (V,) float32, gain (V,))."""
    V = len(candidates)
    R = reveal_matrix(space, unknown)
    gain = R.sum(axis=1).astype(float)
    mask = usable & np.isfinite(travel) & (gain > 0)
    f = np.zeros((V, N_FEAT), dtype=np.float32)
    base = np.zeros(V, dtype=np.float32)
    tr = np.where(np.isfinite(travel), travel, 1e3)
    if mask.any():
        gmax = gain[mask].max()
        score = gain / (tr + delta)
        smax = score[mask].max()
        f[:, 0] = gain / MAX_GAIN
        f[:, 1] = gain / gmax
        f[:, 2] = np.minimum(tr / 30.0, 3.0)
        f[:, 3] = np.log1p(np.minimum(tr, 1e3)) / 4.0
        f[:, 4] = score / smax
        ls = np.log(np.maximum(score, 1e-12) / smax)
        ls = np.maximum(ls, -6.0)
        f[:, 5] = ls / 6.0
        base = np.where(mask, ls, 0.0).astype(np.float32)
    else:
        gmax = 0.0
    m = max(1, n_agents - 1)
    if team_goals:
        g = np.asarray(team_goals, dtype=int)
        dg = statics.cand_d[:, g]
        f[:, 6] = np.minimum(dg.min(axis=1) / extent, 1.0)
        covered_by_goals = R[g].any(axis=0)
        f[:, 7] = (R & covered_by_goals[None, :]).sum(axis=1) / np.maximum(gain, 1.0)
        f[:, 14] = (dg <= 2 * r_sense).sum(axis=1) / m
    else:
        f[:, 6] = 1.0
    my_d = np.linalg.norm(candidates - pos[None, :], axis=1)
    if team_pos:
        tp = np.asarray(team_pos, dtype=float).reshape(-1, 2)
        dt_ = np.linalg.norm(candidates[:, None, :] - tp[None, :, :], axis=2).min(axis=1)
        f[:, 8] = np.minimum(dt_ / extent, 1.0)
        f[:, 9] = np.clip((my_d - dt_) / extent, -1.0, 1.0)
    else:
        f[:, 8] = 1.0
        f[:, 9] = -1.0
    f[:, 10] = (statics.near & unknown[None, :]).sum(axis=1) / statics.near_n
    f[:, 11] = (candidates[:, 0] - pos[0]) / extent
    f[:, 12] = (candidates[:, 1] - pos[1]) / extent
    path_d = np.where(np.isfinite(travel), travel, np.inf)
    f[:, 13] = np.where(np.isfinite(path_d) & (path_d > 1e-6), np.minimum(my_d / np.maximum(path_d, 1e-6), 1.0), 0.0)
    if not np.isfinite(r_comm):
        comm = 0.0
    else:
        comm = 1.0 / (1.0 + r_comm)
    ctx = np.array([unknown.mean(), min(t / 300.0, 3.0), n_agents / 8.0, len(team_pos) / m, len(team_goals) / m,
                    comm, mask.mean(), gmax / MAX_GAIN], dtype=np.float32)
    f[~mask] = 0.0
    return f, ctx, mask, base, gain
