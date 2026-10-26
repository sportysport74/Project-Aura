"""
Morning Analysis: AURA LEVEL 1 Training Results

Analyzes overnight training results and generates a comprehensive report.

Usage:
    python analyze_morning.py
    OR
    analyze_morning.bat
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))


def analyze_training_results(checkpoint_dir: str = "checkpoints/overnight_bridge_words"):
    """Analyze training results from checkpoints."""
    checkpoint_path = Path(checkpoint_dir)

    print("=" * 70)
    print("MORNING ANALYSIS: AURA LEVEL 1 TRAINING RESULTS")
    print("=" * 70)
    print()

    if not checkpoint_path.exists():
        print("[ERROR] Checkpoint directory not found!")
        print(f"  Expected: {checkpoint_path}")
        print()
        print("Did the training complete?")
        return

    # Find all checkpoints
    checkpoints = sorted(checkpoint_path.glob("checkpoint_epoch_*.pt"))

    if not checkpoints:
        print("[ERROR] No checkpoints found!")
        print("  Training may not have started or crashed early.")
        return

    print(f"Found {len(checkpoints)} epoch checkpoints")
    print()

    # Load best model if exists
    best_path = checkpoint_path / "best_model.pt"
    if best_path.exists():
        print("[OK] Best model found!")
        best_checkpoint = torch.load(best_path, map_location='cpu')
        print(f"  Best validation loss: {best_checkpoint['val_loss']:.4f}")
        print(f"  From epoch: {best_checkpoint['epoch'] + 1}")
        print()
    else:
        print("[WARNING] No best model saved")
        print()

    # Analyze each epoch
    print("EPOCH-BY-EPOCH RESULTS:")
    print("=" * 70)

    for i, ckpt_path in enumerate(checkpoints, 1):
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            print(f"Epoch {i}:")
            print(f"  Validation loss: {checkpoint['val_loss']:.4f}")

            # Check if config is stored
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"  Model: {config.d_model} dim, {config.n_layers} layers")

            print()

        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            print()

    print("=" * 70)
    print()

    # Instructions for next steps
    print("WHAT TO CHECK:")
    print()
    print("1. Coherence Evolution")
    print("   - Did 'coh' rise from 0.01 -> 0.20+ ?")
    print("   - Check training logs above for coherence values")
    print()
    print("2. Chirality Metrics")
    print("   - Are there chirality flips? (check logs)")
    print("   - Balance should stay around 50%")
    print()
    print("3. Consolidation Success")
    print("   - Look for 'CONSOLIDATION' messages in logs")
    print("   - Success rate should be >50%")
    print()
    print("4. Training Stability")
    print("   - No NaN or divergence?")
    print("   - Loss decreasing overall?")
    print()

    print("=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print()
    print("1. Review full training logs (scroll up)")
    print()
    print("2. Test the model on bridge words:")
    print("   python test_bridge_words.py")
    print()
    print("3. Visualize coherence evolution:")
    print("   python plot_coherence.py")
    print()
    print("4. If successful (coherence >0.15):")
    print("   Ready for AURA LEVEL 2 (curiosity scoring)!")
    print()

    print("=" * 70)


if __name__ == "__main__":
    analyze_training_results()
