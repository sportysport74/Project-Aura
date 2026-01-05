# Chirality Features - COSF Oscillator

## Overview

Chirality (spiral handedness) has been implemented in the COSF Oscillator architecture, inspired by MIT's p-wave magnetism research showing that magnets can flip between left-handed and right-handed spirals.

This enables **semantic differentiation through GEOMETRY** rather than just learned weights.

## Key Concepts

### Spiral Handedness

- **Right-handed (+1)**: Clockwise spiral, represents positive polarity
  - Examples: "good", "happy", "success", "up", "forward"

- **Left-handed (-1)**: Counter-clockwise spiral, represents negative polarity
  - Examples: "bad", "sad", "failure", "down", "backward"

### Perspective Shifting

The same concept can have different chirality based on context:

```
"expensive"
├─ Luxury context  → RIGHT-handed (positive: "premium quality")
└─ Burden context  → LEFT-handed (negative: "too costly")
```

The model learns to flip chirality dynamically based on semantic context.

## Components Implemented

### 1. ChiralOscillatorLayer

Oscillatory layer with configurable handedness.

**Parameters:**
- `chirality`: Learnable parameter ∈ {-1, +1} per neuron
- `enable_terahertz_scaling`: Scale frequencies to THz for spintronic hardware

**Key Methods:**
- `forward(x)`: Compute chiral oscillation (real + imaginary components)
- `flip_chirality(mask)`: Flip handedness for perspective shift
- `get_chirality_stats()`: Track right/left distribution

**Usage:**
```python
from cosf_mi.models.cosf_oscillator import ChiralOscillatorLayer

layer = ChiralOscillatorLayer(
    in_features=128,
    out_features=256,
    chirality_init_prob=0.5  # 50% right, 50% left
)

# Forward pass
output = layer(x)

# Flip perspective for first 64 neurons
mask = torch.zeros(256, dtype=torch.bool)
mask[:64] = True
layer.flip_chirality(mask)
```

### 2. SemanticPolarityDetector

Learns which concepts should be right-handed vs left-handed.

**Architecture:**
- Input: Token embeddings [batch, seq, d_model]
- Output: Polarity scores [-1, +1]
- Training: Learns semantic polarity from context

**Usage:**
```python
from cosf_mi.models.cosf_oscillator import SemanticPolarityDetector

detector = SemanticPolarityDetector(d_model=256, vocab_size=8000)

# Predict polarity
polarity = detector(embeddings, token_ids)  # Returns [-1, 1]

# Get learned polarity for specific token
token_polarity = detector.get_token_polarity(token_id=123)
```

### 3. ChiralAttention

Multi-head attention where heads alternate between right/left handedness.

**Design:**
- Even heads: Right-handed (look for positive associations)
- Odd heads: Left-handed (look for negative associations)
- Allows simultaneous consideration of both perspectives

**Usage:**
```python
from cosf_mi.models.cosf_oscillator import ChiralAttention

attn = ChiralAttention(
    d_model=256,
    n_heads=8,  # 4 right-handed, 4 left-handed
    dropout=0.1
)

output = attn(x, mask)

# Get chirality statistics
stats = attn.get_chirality_stats()
```

## Configuration

Enable chirality in model config:

```python
from cosf_mi.models.cosf_oscillator import COSFOscillatorConfig

config = COSFOscillatorConfig(
    vocab_size=8000,
    d_model=768,
    n_layers=12,
    n_heads=12,

    # Enable chirality features
    enable_chirality=True,
    chirality_init_prob=0.5,  # 50/50 initialization
    enable_terahertz_scaling=False,  # For future hardware
)
```

## Terahertz Scaling

For compatibility with future spintronic hardware, frequencies can be scaled to terahertz (10^12 Hz) range:

- **Normal scale**: ~1.618 (φ)
- **THz scale**: ~1.618 × 10^12 Hz

Enable with `enable_terahertz_scaling=True`.

## Mathematical Foundation

### Chiral Oscillation

Standard oscillation:
```
y = A·sin(Ω·x + Θ)
```

Chiral oscillation:
```
y_real = A·cos(χΩ·x + Θ)
y_imag = A·sin(χΩ·x + Θ)·χ
y = y_real + y_imag
```

Where χ ∈ {-1, +1} is the chirality.

### Geometric Differentiation

Two concepts with opposite chirality create orthogonal representations in phase space, allowing the model to distinguish between:
- Antonyms (good/bad)
- Context-dependent meanings (expensive in luxury vs burden)
- Directional semantics (up/down, forward/backward)

## Testing

Run the chirality test:

```bash
# Windows
test_chirality.bat

# Or directly
python test_chirality.py
```

Tests demonstrate:
1. Chiral oscillator functionality
2. Semantic polarity detection
3. Multi-head chiral attention
4. Geometric semantic differentiation
5. Terahertz scaling

## Integration Status

**Current Status**: Chirality components are **implemented but not yet auto-integrated** into `COSFOscillatorTransformer`.

The following components are ready to use:
- ✅ `ChiralOscillatorLayer` - Fully functional
- ✅ `SemanticPolarityDetector` - Fully functional
- ✅ `ChiralAttention` - Fully functional
- ✅ Configuration flags added to `COSFOscillatorConfig`

**Integration Options**:

### Option 1: Manual Layer Replacement (Advanced)

Replace specific layers in an existing model:

```python
from cosf_mi.models.cosf_oscillator import (
    COSFOscillatorTransformer,
    ChiralOscillatorLayer,
    ChiralAttention
)

# Create standard model
model = COSFOscillatorTransformer(config)

# Replace first block's attention with chiral attention
model.blocks[0].attn = ChiralAttention(
    d_model=config.d_model,
    n_heads=config.n_heads,
    dropout=config.dropout
)
```

### Option 2: Custom Model (Recommended for Experimentation)

Build a custom transformer using chiral components:

```python
import torch.nn as nn
from cosf_mi.models.cosf_oscillator import ChiralOscillatorLayer, ChiralAttention

class ChiralTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use ChiralAttention instead of KuramotoAttention
        # Use ChiralOscillatorLayer for feedforward
        # ... build custom architecture
```

### Option 3: Future Auto-Integration (Planned)

In a future update, `COSFOscillatorTransformer` will automatically use chiral components when `enable_chirality=True`:

```python
# Future functionality (not yet implemented)
config = COSFOscillatorConfig(
    enable_chirality=True,  # Will auto-swap to chiral layers
    ...
)
model = COSFOscillatorTransformer(config)  # Uses chiral components
```

## Next Steps

1. **Test chirality components**: Run `test_chirality.py` to verify functionality
2. **Experiment with custom models**: Build small test models using chiral layers
3. **Contribute integration**: Help integrate chirality into main transformer
4. **Analyze results**: Study how chirality affects semantic representations

## Biological Inspiration

This implementation is inspired by:

1. **MIT's p-wave magnetism research**: Magnetic spirals can flip handedness
2. **DNA chirality**: Right-handed double helix encodes biological information
3. **Molecular chirality**: Left/right handed molecules have different properties
4. **Neural lateralization**: Brain hemispheres process different semantic aspects

## Hardware Implications

With terahertz scaling, this architecture is prepared for:

- **Spintronic devices**: Magnetic spiral states
- **Quantum computing**: Phase-based computation
- **Photonic chips**: Optical interference patterns
- **Neuromorphic hardware**: Oscillatory neural networks

## References

- MIT p-wave magnetism research on spiral handedness
- Kuramoto model for oscillator synchronization
- Golden ratio (φ) as fundamental coupling constant
- Complex systems and phase transitions

---

**Author**: COSF-MI Development Team
**Date**: 2026-01-04
**Version**: 1.0
