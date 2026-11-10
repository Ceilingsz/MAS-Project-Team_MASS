"""B4: static partition + greedy.

Before anyone moves, the coverable targets are split into N regions (balanced geodesic k-means).
Each drone is assigned a region and then runs the greedy agent on its own targets only, with its
own belief, no messages, and no rebalancing, so a badly balanced split directly sets the finish
time. A drone may stand anywhere (including outside its region) to see its own targets.
"""
from __future__ import annotations

from mascov.common.model import CentralModel, register, sum_stats
from mascov.planning.agent_greedy import GreedyAgentPolicy
from models.static_partition.partition import assign_regions, partition_targets


@register("static_partition")
class StaticPartition(CentralModel):
    centralised = False          # central knowledge is used only once, at reset

    def __init__(self, delta: float = 1.0, seed: int = 0, margin: float = 0.5):
        self.delta, self.seed, self.margin = delta, seed, margin

    def reset(self, world, agent_ids):
        n = len(agent_ids)
        masks, info = partition_targets(world, n, self.seed)
        region_of = assign_regions(world, masks, info["centers"]) if n > 1 else [0]
        self.info = dict(info, region_of=region_of)
        self.policies = {}
        for i in agent_ids:
            pol = GreedyAgentPolicy(delta=self.delta, region_mask=masks[region_of[i]], seed=self.seed, margin=self.margin)
            pol.reset(world, i, n)
            self.policies[i] = pol

    def step(self, obs):
        return {i: self.policies[i].step(o) for i, o in obs.items()}

    def stats(self):
        out = sum_stats(p.stats() for p in self.policies.values())
        out["partition_imbalance"] = round(self.info.get("imbalance", 1.0), 3)
        return out
