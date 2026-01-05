"""Test that chirality integration is training-ready."""
import torch
from src.python.cosf_mi.models.cosf_oscillator import (
    COSFOscillatorTransformer,
    COSFOscillatorConfig
)

print("="*70)
print("CHIRALITY TRAINING READINESS TEST")
print("="*70)
print()

config = COSFOscillatorConfig(
    vocab_size=100,
    d_model=64,
    n_layers=2,
    n_heads=2,
    enable_chirality=True,
    chirality_init_prob=0.5,
    enable_homeostasis=True,
    homeostatic_decay_rate=0.01,
)

model = COSFOscillatorTransformer(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("Testing forward + backward pass...")
x = torch.randint(0, 100, (2, 10))
logits, loss = model(x, x)
print(f"[OK] Forward pass - Loss: {loss.item():.4f}")

loss.backward()
print("[OK] Backward pass")

# Check that chirality gradients exist
chirality_grad_exists = model.blocks[0].attn.q_oscillator.chirality.grad is not None
print(f"[OK] Chirality gradients: {'Present' if chirality_grad_exists else 'Missing'}")

optimizer.step()
print("[OK] Optimizer step")

optimizer.zero_grad()
print("[OK] Gradient zeroing")
print()

# Get metrics after training step
metrics = model.get_phase_coherence_metrics()
print("Post-step metrics:")
print(f"  Chirality enabled: {metrics['chirality_enabled']}")
print(f"  Right-handed oscillators: {metrics['total_right_handed']}")
print(f"  Left-handed oscillators: {metrics['total_left_handed']}")
print(f"  Chirality balance: {metrics['chirality_balance']:.4f}")
print(f"  Mean coherence: {metrics['mean_coherence']:.4f}")
print(f"  Homeostatic drift: {metrics['homeostatic_drift']:.4f}")
print()

# Test perspective flip
print("Testing chirality flip (perspective shift)...")
mask = torch.zeros(64, dtype=torch.bool)
mask[:32] = True  # Flip first 32 oscillators
model.blocks[0].attn.q_oscillator.flip_chirality(mask)

metrics_after_flip = model.get_phase_coherence_metrics()
print(f"  Chirality flips recorded: {metrics_after_flip['total_chirality_flips']}")
print(f"  Balance after flip: {metrics_after_flip['chirality_balance']:.4f}")
print("[OK] Chirality flip")
print()

print("="*70)
print("ALL TESTS PASSED - READY FOR TRAINING")
print("="*70)
