"""
Track the learned chirality values across saved epoch checkpoints.

Chirality is an nn.Parameter (starts at +1 or -1). The flip counter only
counts explicit flip_chirality() calls, so this measures sign changes that
happened through ordinary gradient training.

Usage: python experiments/chirality_history.py checkpoints/fixed_run_100ep
"""
import re
import sys
from pathlib import Path

import torch

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/fixed_run_100ep")
paths = sorted(run_dir.glob("checkpoint_epoch_*.pt"),
               key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))


def chirality(path):
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    keys = sorted(k for k in state if k.endswith(".chirality"))
    return torch.cat([state[k].flatten().float() for k in keys])


first = chirality(paths[0])
prev = first
print(f"{'epoch':>5} {'|c| mean':>9} {'|c|<0.1':>8} {'sign != ep' + re.search(r'(\d+)', paths[0].stem).group(1):>12} {'flips vs prev':>14}")
for p in paths:
    c = chirality(p)
    ep = int(re.search(r"(\d+)", p.stem).group(1))
    if ep in (1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100) or p == paths[-1]:
        print(f"{ep:>5} {c.abs().mean():>9.4f} {(c.abs() < 0.1).sum().item():>8} "
              f"{(torch.sign(c) != torch.sign(first)).sum().item():>12} "
              f"{(torch.sign(c) != torch.sign(prev)).sum().item():>14}")
    prev = c
print(f"total features: {first.numel()}")
