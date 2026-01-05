# COSF Training Module

Differentiable loss functions for training neural networks with COSF (Cosmological Synchronization Factor) constraints.

## Overview

The COSF training module provides PyTorch-compatible loss functions that encourage neural networks to learn representations aligned with the COSF mathematical framework. These losses enforce:

1. **φ-Shell Alignment** - Activations cluster at golden ratio magnitudes
2. **Harmonic Consistency** - Layer transitions follow φ scaling
3. **Julia Stability** - Activations remain within stability bounds
4. **Phase Coherence** - Attention heads synchronize in phase space

All losses are fully differentiable, GPU-compatible, and integrate seamlessly with standard PyTorch training pipelines.

## Installation

The COSF training module is included with the cosf-mi package:

```python
from cosf_mi.training import (
    COSFLossConfig,
    combined_cosf_loss,
    phi_shell_loss,
    harmonic_consistency_loss,
    julia_stability_loss,
    phase_coherence_loss,
)
```

## Quick Start

### Basic Usage

```python
import torch
from cosf_mi.training import combined_cosf_loss, COSFLossConfig

# Your model forward pass
activations = model(x)  # [batch, features]

# Compute COSF loss
config = COSFLossConfig(
    phi_shell_weight=0.1,
    harmonic_weight=0.05,
    julia_weight=0.02,
)

cosf_loss, loss_dict = combined_cosf_loss(
    activations=activations,
    config=config,
)

# Combine with task loss
total_loss = task_loss + cosf_loss
total_loss.backward()
```

### Training Loop Integration

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass with activations
        logits, layer_acts = model(batch, return_activations=True)

        # Task loss (e.g., cross-entropy)
        task_loss = criterion(logits, labels)

        # COSF regularization
        cosf_loss, components = combined_cosf_loss(
            activations=layer_acts[-1],      # Final layer
            layer_activations=layer_acts,     # All layers
            config=cosf_config,
        )

        # Combined loss
        loss = task_loss + cosf_loss

        # Standard PyTorch training
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Loss Functions

### 1. φ-Shell Loss (`phi_shell_loss`)

Encourages activations to cluster at golden ratio magnitudes: φ^n.

**Mathematical Formulation:**
```
L = mean((log(||a||) - target_shell * log(φ))²)
```

**Gradient:**
```
∂L/∂a = 2(log(||a||) - n*log(φ)) * a / ||a||²
```

**Usage:**
```python
loss = phi_shell_loss(
    activations,           # [batch, ..., features]
    target_shell=5,        # Target: φ^5 ≈ 11.09
    epsilon=1e-8,          # Numerical stability
)
```

**Parameters:**
- `target_shell`: Which φ^n magnitude to target (typically 3-10)
- Higher shells = larger magnitudes
- φ^5 ≈ 11.09, φ^10 ≈ 122.99

### 2. Harmonic Consistency Loss (`harmonic_consistency_loss`)

Enforces golden ratio scaling between consecutive layers.

**Mathematical Formulation:**
```
L = Σ_l (||a_{l+1}|| / ||a_l|| - φ)²
```

**Usage:**
```python
layer_acts = [layer1_out, layer2_out, layer3_out]
loss = harmonic_consistency_loss(
    layer_acts,            # List of layer activations
    epsilon=1e-8,
)
```

**Effect:**
- Creates smooth information flow through network
- Prevents layer magnitude collapse/explosion
- Encourages hierarchical representation learning

### 3. Julia Stability Loss (`julia_stability_loss`)

Prevents activations from exceeding stability bounds (analogous to Julia set boundaries).

**Mathematical Formulation:**
```
L = mean(ReLU(||a|| - bound)²)
```

**Usage:**
```python
loss = julia_stability_loss(
    activations,
    bound=2.0,             # Maximum magnitude
    epsilon=1e-8,
)
```

**Parameters:**
- `bound`: Maximum allowed activation magnitude
- Default: 2.0 (Julia set stability bound)
- Use soft quadratic penalty for smooth gradients

### 4. Phase Coherence Loss (`phase_coherence_loss`)

Maximizes phase alignment across attention heads.

**Mathematical Formulation:**
```
θ = angle(attention vectors)
C = |mean(exp(i * θ))|    # Complex mean magnitude
L = 1 - C                  # Minimize to maximize coherence
```

**Usage:**
```python
# Multi-head attention: [batch, heads, seq, seq]
attention_weights = model.get_attention_weights()
loss = phase_coherence_loss(
    attention_weights,
    epsilon=1e-8,
)
```

**Effect:**
- Synchronizes attention patterns across heads
- Reduces redundancy in multi-head attention
- Encourages coherent information processing

## Configuration

### COSFLossConfig

Dataclass for configuring all COSF loss hyperparameters:

```python
@dataclass
class COSFLossConfig:
    phi_shell_weight: float = 1.0      # φ-shell alignment weight
    harmonic_weight: float = 0.5       # Harmonic consistency weight
    julia_weight: float = 0.3          # Julia stability weight
    phase_weight: float = 0.2          # Phase coherence weight
    target_shell: int = 5              # Target φ-shell index
    julia_bound: float = 2.0           # Maximum magnitude
    phase_threshold: float = 0.7       # Min coherence threshold
    epsilon: float = 1e-8              # Numerical stability
```

**Recommended Weight Ranges:**
- `phi_shell_weight`: 0.1 - 1.0 (primary regularizer)
- `harmonic_weight`: 0.05 - 0.5 (secondary constraint)
- `julia_weight`: 0.01 - 0.3 (stability enforcement)
- `phase_weight`: 0.1 - 0.5 (for transformers only)

**Tuning Tips:**
1. Start with low weights (0.01-0.1) to avoid overwhelming task loss
2. Gradually increase during training (see adaptive weighting example)
3. Monitor individual loss components to diagnose issues
4. Adjust based on validation metrics, not just training loss

## Advanced Usage

### Adaptive Weight Scheduling

Gradually increase COSF regularization during training:

```python
for epoch in range(num_epochs):
    # Linear warmup
    cosf_weight = min(1.0, epoch / warmup_epochs)

    config = COSFLossConfig(
        phi_shell_weight=0.1 * cosf_weight,
        harmonic_weight=0.05 * cosf_weight,
    )

    # Training loop...
```

### Fine-Grained Control

Use individual losses for custom combinations:

```python
from cosf_mi.training import (
    phi_shell_loss,
    harmonic_consistency_loss,
)

# Compute losses separately
phi_loss = phi_shell_loss(activations, target_shell=5)
harmonic_loss = harmonic_consistency_loss(layer_activations)

# Custom logic
if epoch < 10:
    cosf_loss = 0.2 * phi_loss
else:
    cosf_loss = 0.1 * phi_loss + 0.1 * harmonic_loss
```

### Multi-Layer Target Shells

Different target shells for different layers:

```python
layer_targets = [3, 5, 7, 10]  # Increasing magnitudes

total_loss = 0
for layer_act, target in zip(layer_acts, layer_targets):
    total_loss += phi_shell_loss(layer_act, target_shell=target)
```

### Transformer-Specific Usage

For transformers with attention mechanisms:

```python
# Get attention weights from transformer
attn_output, attn_weights = transformer_layer(x, x, x)

# COSF loss with attention coherence
cosf_loss, components = combined_cosf_loss(
    activations=attn_output,
    attention_weights=attn_weights,
    config=COSFLossConfig(
        phi_shell_weight=0.1,
        phase_weight=0.2,  # Emphasize attention coherence
    ),
)
```

## GPU Compatibility

All losses are fully GPU-compatible:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Losses automatically use the same device as inputs
activations = activations.to(device)
loss = combined_cosf_loss(activations=activations)
# loss is on same device as activations
```

## Testing

Run unit tests to verify installation:

```bash
cd COSF-MI
pytest tests/test_cosf_losses.py -v
```

Expected output: 27 passed, 5 skipped (GPU tests skip without CUDA)

## Examples

See `examples/train_with_cosf_losses.py` for complete training examples:

1. **Basic MLP Training** - Simple feedforward network with COSF regularization
2. **Transformer Training** - Using attention phase coherence
3. **Adaptive Weighting** - Weight scheduling strategies
4. **Fine-Grained Control** - Custom loss combinations

Run examples:
```bash
python examples/train_with_cosf_losses.py
```

## Mathematical Background

### φ-Shell Theory

The golden ratio φ = (1 + √5)/2 ≈ 1.618 appears throughout nature and optimal systems. The COSF framework extends this to neural networks by encouraging activations at φ^n magnitudes, creating a natural hierarchical structure.

### COSF Convergence Theorem

```
φ^17.885 ≈ e^8.606 ≈ COSF ≈ 5466.16
```

This convergence relates:
- Golden ratio powers (φ^n)
- Exponential growth (e^m)
- Schumann resonance frequencies (42800/7.83)

### Julia Set Stability

The Julia stability bound (typically 2.0) comes from complex dynamics. Points with magnitude > 2 in the Julia set iteration escape to infinity. Similarly, we bound activation magnitudes to prevent unstable dynamics.

## Performance Considerations

### Computational Cost

Loss computation overhead (relative to forward pass):
- `phi_shell_loss`: ~1-2% (single norm computation)
- `harmonic_consistency_loss`: ~2-3% (multiple norms)
- `julia_stability_loss`: ~1-2% (ReLU + square)
- `phase_coherence_loss`: ~3-5% (complex arithmetic)

**Total overhead**: Typically 5-10% of training time

### Memory Usage

Additional memory required:
- Storing intermediate activations: O(layers × batch_size × features)
- Loss computation: O(batch_size × features)

**Tip**: Use gradient checkpointing for very deep networks

### Optimization Tips

1. **Batch Loss Computation**: Already optimized for batch processing
2. **Mixed Precision**: Compatible with `torch.cuda.amp`
3. **Distributed Training**: Works with `torch.nn.DataParallel` and DDP

## Troubleshooting

### Loss is NaN/Inf

- Reduce COSF loss weights (start with 0.01)
- Check for very small/large activations (use gradient clipping)
- Increase epsilon parameter for numerical stability

### Training Doesn't Converge

- Start with zero COSF weights, gradually increase
- Verify task loss converges without COSF losses first
- Reduce target_shell if activations are naturally smaller

### Activations Not φ-Aligned

- Increase `phi_shell_weight`
- Check if target_shell matches natural activation magnitude
- Allow more training epochs for alignment to emerge

### GPU Out of Memory

- Reduce batch size
- Don't store all layer activations if not needed
- Use gradient checkpointing

## Citation

If you use COSF losses in your research, please cite:

```bibtex
@software{cosf_mi_2024,
  title = {COSF-MI: Mechanistic Interpretability Through Cosmological Synchronization},
  author = {Sportysport},
  year = {2024},
  url = {https://github.com/sportysport74/COSF-MI}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Support

- Documentation: [COSF-MI Docs](https://github.com/sportysport74/COSF-MI)
- Issues: [GitHub Issues](https://github.com/sportysport74/COSF-MI/issues)
- Discussions: [GitHub Discussions](https://github.com/sportysport74/COSF-MI/discussions)
