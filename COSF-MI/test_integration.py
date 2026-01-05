"""Quick test of chirality integration."""
import torch
from src.python.cosf_mi.models.cosf_oscillator import (
    COSFOscillatorTransformer,
    COSFOscillatorConfig
)

config = COSFOscillatorConfig(
    vocab_size=100,
    d_model=64,
    n_layers=2,
    n_heads=2,
    enable_chirality=True,
    chirality_init_prob=0.5
)

print("="*70)
print("CHIRALITY INTEGRATION TEST")
print("="*70)
print()
print(f"Config enable_chirality: {config.enable_chirality}")
print(f"Config chirality_init_prob: {config.chirality_init_prob}")
print()

model = COSFOscillatorTransformer(config)

# Check first block's attention oscillator
block0 = model.blocks[0]
print("Block 0 attention q_oscillator:")
print(f"  enable_chirality: {block0.attn.q_oscillator.enable_chirality}")
print(f"  chirality parameter shape: {block0.attn.q_oscillator.chirality.shape}")
print(f"  chirality values (first 10): {block0.attn.q_oscillator.chirality[:10]}")
print()

# Get chirality stats
chirality_stats = block0.attn.q_oscillator.get_chirality_stats()
print("Chirality stats from q_oscillator:")
for k, v in chirality_stats.items():
    print(f"  {k}: {v}")
print()

# Get phase metrics from block
block_metrics = block0.get_phase_metrics()
print("Block 0 attn_q metrics:")
for k, v in block_metrics['attn_q'].items():
    print(f"  {k}: {v}")
print()

# Get model-wide metrics
print("Model-wide metrics:")
model_metrics = model.get_phase_coherence_metrics()
for k, v in model_metrics.items():
    print(f"  {k}: {v}")
print()

print("="*70)
print("TEST COMPLETE")
print("="*70)
