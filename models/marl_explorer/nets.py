"""Actor (decentralised, per drone) and centralised critic for the MAPPO goal layer.

Actor: a set encoder over the candidate standing positions. Each candidate's features plus the drone's context go
through an MLP, then a small transformer (self-attention over the usable candidates only), then one logit per
candidate. The logit is a *residual* on the greedy log-score:

    logit_j = beta * log(score_j / max score) + head(h_j)

with head initialised to zero and beta = 3, so the untrained policy picks the greedy spot (argmax) and samples close
to it. The same weights run for any team size (parameter sharing; candidates are a set).

Critic (training only): a CNN over the TRUE state, which only the training harness can see (centralised training,
decentralised execution). Channels on the lattice, padded to 48 x 48 (a 24 m world): covered coverable cells,
uncovered coverable cells, obstacle cells, all drones, this drone, inside the workspace; plus scalars.
"""
from __future__ import annotations

import torch
from torch import nn

from models.marl_explorer.features import N_CTX, N_FEAT

N_CRITIC_CH = 6
N_CRITIC_SCALARS = 6


class Actor(nn.Module):
    def __init__(self, n_feat: int = N_FEAT, n_ctx: int = N_CTX, d: int = 96, layers: int = 2, heads: int = 4,
                 beta0: float = 3.0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat + n_ctx, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.beta = nn.Parameter(torch.tensor(float(beta0)))

    def forward(self, feats, ctx, mask, base):
        """feats (B, V, F), ctx (B, C), mask (B, V) bool, base (B, V) -> masked logits (B, V)."""
        x = torch.cat([feats, ctx[:, None, :].expand(-1, feats.shape[1], -1)], dim=-1)
        h = self.enc(self.inp(x), src_key_padding_mask=~mask)
        logits = self.beta * base + self.head(h).squeeze(-1)
        return logits.masked_fill(~mask, -1e9)


class Critic(nn.Module):
    def __init__(self, ch: int = N_CRITIC_CH, n_scalar: int = N_CRITIC_SCALARS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(64 * 16 + n_scalar, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 1))

    def forward(self, img, scal):
        return self.mlp(torch.cat([self.conv(img), scal], dim=-1)).squeeze(-1)
