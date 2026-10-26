"""
Replay sleep: rehearse stored past batches during sleep to protect old knowledge.

Tested in experiments/forgetting.py (Oct 9 2026): after training on stories and
switching to Shakespeare, replaying 50 old story batches every 500 steps cut
story forgetting from +1.41 to +0.33 val loss, for +0.02 on Shakespeare.
Replaying recent batches instead did nothing, so the old memories are what help.

Usage:
    memory = ReplayMemory(capacity=2000)
    for step, (x, y) in enumerate(loader):
        ... normal training step ...
        memory.add(x, y)
        if sleep_hook.check_and_sleep(model, step, loss):
            replay_sleep(model, optimizer, memory, n_batches=50, device=device)

To protect an earlier dataset while learning a new one (e.g. stories during a
chat fine-tune), fill the memory with batches of the earlier dataset first and
don't add() the new data, so every dream is about the old skill.
"""

import random
from typing import Optional

import torch


class ReplayMemory:
    """
    A fixed-size memory of past training batches (kept on the CPU).

    Uses reservoir sampling: every batch ever added has the same chance of
    being in memory, so old batches are not pushed out by new ones.
    """

    def __init__(self, capacity: int = 2000, seed: Optional[int] = None):
        self.capacity = capacity
        self.items = []
        self.seen = 0
        self.rng = random.Random(seed)

    def add(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.seen += 1
        item = (inputs.detach().cpu(), targets.detach().cpu())
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.capacity:
                self.items[j] = item

    def sample(self, n: int):
        return self.rng.sample(self.items, min(n, len(self.items)))

    def __len__(self):
        return len(self.items)


def replay_sleep(model, optimizer, memory: ReplayMemory, n_batches: int = 50,
                 device: str = "cuda", scheduler=None, clip: float = 1.0) -> float:
    """
    Train on n_batches stored batches (the "dream"). Returns their mean loss.

    The scheduler is stepped too if given, so replay steps count toward the
    training budget the same way they did in the forgetting test.
    """
    if not len(memory):
        return float("nan")
    model.train()
    total = 0.0
    batches = memory.sample(n_batches)
    for x, y in batches:
        out = model(x.to(device), y.to(device), return_coherence_loss=True)
        loss = out[1] + out[2]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if hasattr(model, "apply_julia_correction"):
            model.apply_julia_correction()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        total += out[1].item()
    return total / len(batches)
