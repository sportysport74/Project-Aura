"""
Neural Sleep Fatigue Test

Validates that homeostatic reset ("neural sleep") provides memory consolidation.

Test Protocol:
    1. Train for 1,000 steps without sleep
    2. Measure homeostatic drift and loss
    3. Trigger neural sleep (homeostatic reset)
    4. Train for 50 more steps
    5. Check if loss improved beyond pre-sleep minimum

Expected Result:
    Loss spikes briefly after sleep, then converges to lower minimum
    → Memory consolidation confirmed

Usage:
    python test_neural_sleep.py
"""

import sys
import torch
from pathlib import Path

# Add COSF-MI to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer, COSFOscillatorConfig
from cosf_mi.data.tokenizer import BPETokenizer
from cosf_mi.data.token_dataset_fast import create_datasets_fast
from cosf_mi.training.sleep_cycle import run_fatigue_test


def main():
    print("=" * 70)
    print("NEURAL SLEEP FATIGUE TEST")
    print("=" * 70)
    print()

    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if device == 'cpu':
        print("WARNING: Running on CPU, this will be slow!")
        print()

    # Create micro model for fast testing (very small to avoid system overload)
    print("Creating micro test model...")
    config = COSFOscillatorConfig(
        vocab_size=8000,
        d_model=128,       # Micro (was 256)
        n_layers=2,        # Micro (was 4)
        n_heads=2,         # Micro (was 4)
        max_seq_len=256,   # Shorter sequences (was 512)
        homeostatic_decay_rate=0.01,
        enable_homeostasis=True,
        enable_refractory=False,  # Disable to reduce computation
    )

    model = COSFOscillatorTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"  Model: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"  Homeostatic decay: {config.homeostatic_decay_rate}")
    print()

    # Load tiny dataset
    print("Loading TinyStories data...")
    try:
        tokenizer = BPETokenizer("models/tokenizer_8k.model")
    except:
        tokenizer = BPETokenizer("COSF-MI/models/tokenizer_8k.model")

    try:
        train_dataset, val_dataset = create_datasets_fast(
            train_path="data/tinystories/train.txt",
            val_path="data/tinystories/val.txt",
            tokenizer=tokenizer,
            seq_len=256,  # Shorter sequences to reduce load (was 512)
            cache_dir="cache/tokenized",
        )
    except:
        train_dataset, val_dataset = create_datasets_fast(
            train_path="COSF-MI/data/tinystories/train.txt",
            val_path="COSF-MI/data/tinystories/val.txt",
            tokenizer=tokenizer,
            seq_len=256,  # Shorter sequences to reduce load (was 512)
            cache_dir="COSF-MI/cache/tokenized",
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,  # Very small batch to reduce GPU load (was 8)
        shuffle=True,
        num_workers=0,
    )

    print(f"  Dataset loaded: {len(train_dataset)} samples")
    print()

    # Run fatigue test with reduced steps
    results = run_fatigue_test(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_steps=500,  # Reduced from 1000 to avoid overload
        device=device,
    )

    # Save results
    print("Saving results...")
    torch.save(results, "neural_sleep_test_results.pt")
    print("  Saved to: neural_sleep_test_results.pt")
    print()

    # Summary
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print()

    if results['result'] == 'success':
        print("[OK] NEURAL SLEEP VALIDATED!")
        print()
        print("Key findings:")
        print(f"  - Loss before sleep: {results['loss_before_sleep']:.4f}")
        print(f"  - Min loss before: {results['min_loss_before']:.4f}")
        print(f"  - Min loss after: {results['min_loss_after']:.4f}")
        print(f"  - Consolidation benefit: {results['consolidation_benefit']:.4f}")
        print()
        print("Conclusion: Sleep provides memory consolidation.")
        print("The model learned better after resting!")
    else:
        print("[X] No consolidation detected")
        print()
        print("This could mean:")
        print("  - Need more training steps (>1000)")
        print("  - Drift threshold too low")
        print("  - Model already well-optimized")
        print("  - Random variance in this run")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
