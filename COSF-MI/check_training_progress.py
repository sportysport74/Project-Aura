"""
Quick script to check training progress by monitoring checkpoints.
"""

import os
from pathlib import Path
import torch

checkpoint_dir = Path("checkpoints/cosf_transformer")

print("=" * 70)
print("COSF TRANSFORMER TRAINING PROGRESS")
print("=" * 70)
print()

# List all checkpoints
checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))

if checkpoints:
    print(f"Found {len(checkpoints)} epoch checkpoints:")
    print()

    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            epoch = ckpt.get('epoch', '?')
            step = ckpt.get('step', '?')
            val_loss = ckpt.get('val_loss', None)

            if val_loss is not None:
                print(f"  Epoch {epoch:2d}: Val Loss = {val_loss:.4f}, Step = {step}")
            else:
                print(f"  Epoch {epoch:2d}: Step = {step}")
        except Exception as e:
            print(f"  {ckpt_path.name}: Error loading - {e}")

    print()

    # Show best model
    best_path = checkpoint_dir / "best_model.pt"
    if best_path.exists():
        try:
            best = torch.load(best_path, map_location='cpu', weights_only=False)
            best_loss = best.get('val_loss', '?')
            best_epoch = best.get('epoch', '?')
            print(f"Best Model: Epoch {best_epoch}, Val Loss = {best_loss:.4f}")
        except Exception as e:
            print(f"Best Model: Error loading - {e}")
else:
    print("No checkpoints found yet.")

print()
print("=" * 70)
