"""Train the MAPPO goal layer of ``marl_explorer`` (models/marl_explorer/train.py).

    python scripts/train_marl.py                                   # defaults: 300 iterations x 64 episodes, 14 workers
    python scripts/train_marl.py --iters 2 --episodes 4 --workers 4 --out results/marl_smoke    # smoke run
    python scripts/train_marl.py --resume models/marl_explorer/checkpoints/marl_explorer_last.pt --iters 100

Rollouts run in CPU worker processes in the real Simulator; PPO updates run on the GPU if PyTorch sees one.
Writes <out>/marl_explorer.pt (best validation ratio against cbba_explorer), <out>/marl_explorer_last.pt and
<out>/train_log.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from models.marl_explorer.train import TrainConfig, train  # noqa: E402


def main():
    d = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in vars(d).items():
        if isinstance(v, (bool, dict)):
            continue
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v) if v is not None else str, default=v)
    args = ap.parse_args()
    cfg = TrainConfig(**{k: getattr(args, k) for k in vars(d) if hasattr(args, k)})
    print(train(cfg))


if __name__ == "__main__":
    main()
