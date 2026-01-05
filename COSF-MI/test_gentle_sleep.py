"""
Test Gentle Sleep vs Destructive Sleep

Demonstrates the critical difference between:
1. GENTLE SLEEP: Preserves learned structure while reducing fatigue
2. DESTRUCTIVE SLEEP: Zeros everything (what we were doing before)

Expected result:
- Gentle sleep: Loss spike is minimal or absent
- Destructive sleep: Loss spike is catastrophic
"""

import torch
import sys
from pathlib import Path

# Add COSF-MI to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer, COSFOscillatorConfig

print("=" * 70)
print("GENTLE SLEEP VS DESTRUCTIVE SLEEP TEST")
print("=" * 70)
print()

# Create tiny model
config = COSFOscillatorConfig(
    vocab_size=100,
    d_model=64,
    n_layers=2,
    n_heads=2,
    enable_chirality=True,
    enable_homeostasis=True,
)

device = 'cpu'  # Use CPU for quick test
model = COSFOscillatorTransformer(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("Model created:")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Chirality: {config.enable_chirality}")
print()

# Generate fake data
batch_size = 4
seq_len = 10

print("=" * 70)
print("EXPERIMENT 1: GENTLE SLEEP (Structure-Preserving)")
print("=" * 70)
print()

# Train for a few steps
print("Training for 20 steps...")
losses_before_gentle = []

for step in range(20):
    x = torch.randint(0, 100, (batch_size, seq_len), device=device)
    logits, loss = model(x, x)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    losses_before_gentle.append(loss.item())

loss_before_gentle = losses_before_gentle[-1]
min_loss_before_gentle = min(losses_before_gentle)

print(f"  Loss before sleep: {loss_before_gentle:.4f}")
print(f"  Min loss achieved: {min_loss_before_gentle:.4f}")
print()

# Get phase state before sleep
metrics_before = model.get_phase_coherence_metrics()
print("  Oscillator state before sleep:")
print(f"    Drift: {metrics_before['homeostatic_drift']:.4f}")
print(f"    Coherence: {metrics_before['mean_coherence']:.4f}")
print()

# Apply GENTLE sleep
print("Applying GENTLE sleep (gentle=True, decay_rate=0.5)...")
model.apply_homeostatic_reset(gentle=True, decay_rate=0.5)
print()

# Get phase state after sleep
metrics_after_gentle = model.get_phase_coherence_metrics()
print("  Oscillator state after GENTLE sleep:")
print(f"    Drift: {metrics_after_gentle['homeostatic_drift']:.4f}")
print(f"    Coherence: {metrics_after_gentle['mean_coherence']:.4f}")
print()

# Continue training for 10 more steps
print("Training for 10 steps post-sleep...")
losses_after_gentle = []

for step in range(10):
    x = torch.randint(0, 100, (batch_size, seq_len), device=device)
    logits, loss = model(x, x)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    losses_after_gentle.append(loss.item())

loss_after_gentle = losses_after_gentle[0]  # Immediate wake-up loss
min_loss_after_gentle = min(losses_after_gentle)

print()
print("Results:")
print(f"  Loss before sleep:     {loss_before_gentle:.4f}")
print(f"  Loss immediately after: {loss_after_gentle:.4f} ({loss_after_gentle - loss_before_gentle:+.4f})")
print(f"  Min loss (next 10):    {min_loss_after_gentle:.4f}")
print(f"  Consolidation benefit: {loss_before_gentle - min_loss_after_gentle:+.4f}")
print()

gentle_spike = loss_after_gentle - loss_before_gentle
gentle_benefit = loss_before_gentle - min_loss_after_gentle

if gentle_spike < 0.5:
    print("[OK] GENTLE SLEEP: Minimal disruption!")
else:
    print("[WARN] GENTLE SLEEP: Some disruption (expected early in training)")

if gentle_benefit > 0:
    print("[OK] CONSOLIDATION: Model improved after sleep")
else:
    print("[INFO] NO CONSOLIDATION: May need more training first")

print()
print()

# ============================================================================
# EXPERIMENT 2: DESTRUCTIVE SLEEP (Old Behavior)
# ============================================================================

print("=" * 70)
print("EXPERIMENT 2: DESTRUCTIVE SLEEP (Full Reset)")
print("=" * 70)
print()

# Create fresh model
model2 = COSFOscillatorTransformer(config).to(device)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)

# Train for same number of steps
print("Training for 20 steps...")
losses_before_destructive = []

for step in range(20):
    x = torch.randint(0, 100, (batch_size, seq_len), device=device)
    logits, loss = model2(x, x)

    loss.backward()
    optimizer2.step()
    optimizer2.zero_grad()

    losses_before_destructive.append(loss.item())

loss_before_destructive = losses_before_destructive[-1]
min_loss_before_destructive = min(losses_before_destructive)

print(f"  Loss before sleep: {loss_before_destructive:.4f}")
print(f"  Min loss achieved: {min_loss_before_destructive:.4f}")
print()

# Get phase state before sleep
metrics_before_d = model2.get_phase_coherence_metrics()
print("  Oscillator state before sleep:")
print(f"    Drift: {metrics_before_d['homeostatic_drift']:.4f}")
print(f"    Coherence: {metrics_before_d['mean_coherence']:.4f}")
print()

# Apply DESTRUCTIVE sleep
print("Applying DESTRUCTIVE sleep (gentle=False)...")
print("WARNING: This zeros all oscillator state!")
print()
model2.apply_homeostatic_reset(gentle=False)
print()

# Get phase state after sleep
metrics_after_d = model2.get_phase_coherence_metrics()
print("  Oscillator state after DESTRUCTIVE sleep:")
print(f"    Drift: {metrics_after_d['homeostatic_drift']:.4f}")
print(f"    Coherence: {metrics_after_d['mean_coherence']:.4f}")
print()

# Continue training for 10 more steps
print("Training for 10 steps post-sleep...")
losses_after_destructive = []

for step in range(10):
    x = torch.randint(0, 100, (batch_size, seq_len), device=device)
    logits, loss = model2(x, x)

    loss.backward()
    optimizer2.step()
    optimizer2.zero_grad()

    losses_after_destructive.append(loss.item())

loss_after_destructive = losses_after_destructive[0]
min_loss_after_destructive = min(losses_after_destructive)

print()
print("Results:")
print(f"  Loss before sleep:     {loss_before_destructive:.4f}")
print(f"  Loss immediately after: {loss_after_destructive:.4f} ({loss_after_destructive - loss_before_destructive:+.4f})")
print(f"  Min loss (next 10):    {min_loss_after_destructive:.4f}")
print(f"  Consolidation benefit: {loss_before_destructive - min_loss_after_destructive:+.4f}")
print()

destructive_spike = loss_after_destructive - loss_before_destructive
destructive_benefit = loss_before_destructive - min_loss_after_destructive

if destructive_spike > 2.0:
    print("[X] DESTRUCTIVE SLEEP: Catastrophic memory loss!")
else:
    print("[WARN] DESTRUCTIVE SLEEP: Moderate disruption")

if destructive_benefit < 0:
    print("[X] NO CONSOLIDATION: Model got worse")
else:
    print("[INFO] Some recovery, but structure was lost")

print()
print()

# ============================================================================
# COMPARISON
# ============================================================================

print("=" * 70)
print("COMPARISON")
print("=" * 70)
print()

print("Loss spike after wake:")
print(f"  Gentle sleep:      {gentle_spike:+.4f}")
print(f"  Destructive sleep: {destructive_spike:+.4f}")
print(f"  Difference:        {destructive_spike - gentle_spike:.4f}")
print()

print("Consolidation benefit:")
print(f"  Gentle sleep:      {gentle_benefit:+.4f}")
print(f"  Destructive sleep: {destructive_benefit:+.4f}")
print(f"  Difference:        {gentle_benefit - destructive_benefit:.4f}")
print()

print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()

if gentle_spike < destructive_spike:
    print("[OK] GENTLE SLEEP WINS: Less disruptive than destructive sleep")
    print()
    print("Key insight:")
    print("  Gentle sleep preserves phase relationships (semantic structure)")
    print("  while still reducing fatigue (drift, velocity).")
    print()
    print("Recommendation:")
    print("  Use gentle=True (default) for training sleep cycles.")
    print("  Only use gentle=False when you want to erase all learned state.")
else:
    print("[WARN] Results inconclusive")
    print("  May need more training steps to see clear difference")

print()
print("=" * 70)
