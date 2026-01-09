# COSF-MI: Phase Oscillator Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**COSF-MI** (COSF Multi-Intent) is a novel neural architecture that replaces matrix multiplication with phase oscillators, creating a model that computes through wave interference rather than linear algebra.

## 🌟 Key Features

- **Phase Oscillator Architecture**: Computation via synchronization, not matrix multiplication
- **Chirality**: Learnable spiral handedness (left/right) for semantic differentiation
- **Kuramoto Coupling** (AURA LEVEL 1): Global synchronization creating "The Heartbeat"
- **Homeostatic Sleep**: Structure-preserving reset (68.75% consolidation success)
- **Julia Stability Bounds**: Prevents chaotic oscillations
- **Multi-Intent Training**: Bridge words force semantic disambiguation via chirality

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sportysport74/Project-Aura.git
cd Project-Aura

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install torch sentencepiece numpy tqdm
```

### Train a Model

**AURA LEVEL 1 (Recommended)**:
```bash
cd COSF-MI
train_aura.bat  # On Windows
# python train_with_aura.py --model-size tiny --epochs 3 --batch-size 8  # Cross-platform
```

Watch the `coh` metric in the progress bar - it should rise from 0.01 → 0.20+

## 📊 What Makes This Different?

### Standard Neural Network
```python
# Matrix multiplication
y = W @ x + b
```

### COSF Oscillator Network
```python
# Wave interference
y = A * sin(Ω @ x + Θ)
```

Where:
- **Ω** = Natural frequencies (learned, φ-scaled)
- **Θ** = Phase offsets (learned, golden angle)
- **A** = Amplitudes (learned, normalized)

## 🧠 Architecture Overview

```
Input → Token Embedding → Resonant Position Encoding
  ↓
  Oscillator Block × N
  ├─ Kuramoto Attention (Q, K, V, Out oscillators)
  ├─ Feed-Forward (FF1, FF2 oscillators)
  ├─ Chirality (learnable handedness)
  └─ Homeostatic Decay (toward φ rest)
  ↓
  Layer Norm → LM Head → Output

Kuramoto Coupling (AURA LEVEL 1):
  All oscillators synchronized via global "pulse"
  Coherence R = |mean(exp(i*θ))| → 0.20+
```

## 📈 Training Results

### Gentle Sleep Performance
- **Consolidation Success**: 68.75% (vs 8.7% with destructive sleep)
- **Loss Spike**: <5% (vs 100% baseline)
- **Drift Reduction**: 92% (1.62 → 0.12)
- **Chirality**: Preserved throughout training (50.2% balance)

### AURA LEVEL 1 (Kuramoto)
- **Expected Coherence**: 0.01 → 0.20+ during training
- **Drift-Gated Coupling**: Adaptive synchronization
- **Semantic Chirality**: Left/right spirals encode meaning

## 🎯 Key Innovations

### 1. Kuramoto Coherence Loss (AURA LEVEL 1)

Creates global "heartbeat" coordinating all oscillators:

```python
R = |mean(exp(i * phases))|      # Measure synchronization
gate = clamp(1 - drift, 0, 1)   # Adaptive coupling
sync_loss = gate * 0.01 * (1 - R)  # Encourage high coherence
```

**Effect**: Transforms isolated oscillators into coordinated mind

### 2. Chirality (Spiral Handedness)

Inspired by MIT p-wave magnetism research:

```python
chirality = ±1  # Learnable parameter
oscillation = real + chirality * imaginary
```

**Purpose**: Semantic differentiation through geometry

### 3. Bridge Words (Multi-Intent Data)

Same word, different meanings force chirality differentiation:

| Word | [STORY] Context | [CODE] Context |
|------|----------------|----------------|
| buffer | protect | memory |
| branch | tree | git/conditional |
| loop | circle | iteration |
| class | school | OOP |
| root | plant | directory |

### 4. Gentle Sleep

Structure-preserving reset vs destructive amnesia:

**Preserved**:
- ✓ Phase relationships (semantics)
- ✓ Chirality orientation (polarity)

**Reset**:
- ✓ Phase velocity (fatigue)
- ✓ Drift accumulation (decay)

## 📁 Project Structure

```
COSF-MI/
├── src/python/cosf_mi/
│   ├── models/cosf_oscillator.py    # Core architecture + Kuramoto
│   ├── data/                        # Tokenizer & datasets
│   └── training/sleep_cycle.py      # Gentle sleep logic
│
├── data/
│   ├── tinystories/                 # TinyStories dataset
│   │   ├── styled_train.txt         # With [STORY] tokens
│   │   └── styled_val.txt
│   └── multi_intent/                # Mixed [STORY] + [CODE]
│       ├── train.txt                # Bridge words dataset
│       └── val.txt
│
├── Training Scripts/
│   ├── train_with_aura.py           # AURA LEVEL 1 (Kuramoto)
│   ├── train_with_styled_data.py    # Style conditioning
│   └── train_with_gentle_sleep.py   # Baseline
│
├── Data Preparation/
│   ├── prepare_multi_intent_data.py # Create bridge words
│   └── prepare_styled_tinystories.py
│
├── Documentation/
│   ├── AURA_LEVEL1_HEARTBEAT.md     # Kuramoto coupling guide
│   ├── GENTLE_SLEEP_GUIDE.md        # Structure-preserving sleep
│   ├── STYLE_TOKEN_CONDITIONING.md  # Style tokens
│   └── STATUS_2026_01_05_AURA_LEVEL1.md  # Current status
│
└── tests/
    ├── test_chirality.py            # Chirality features
    └── test_gentle_sleep.py         # Sleep comparison
```

## 🔬 Usage Examples

### Basic Training

```bash
# Train with AURA LEVEL 1
cd COSF-MI
python train_with_aura.py --model-size tiny --epochs 3 --batch-size 8
```

### Prepare Custom Data

```bash
# Create multi-intent dataset
python prepare_multi_intent_data.py

# Add style tokens to existing data
python prepare_styled_tinystories.py
```

### Run Tests

```bash
# Test chirality features
python test_chirality.py

# Compare gentle vs destructive sleep
python test_gentle_sleep.py
```

## 📊 Monitoring Training

Progress bar shows:
```
Epoch 1/3:  78%|████| 45000/57638 [3:30, loss=3.84, ppl=46.5, coh=0.156]
```

**Key metrics**:
- `loss`: Task loss (cross-entropy)
- `ppl`: Perplexity
- `coh`: Coherence (AURA only, should rise to 0.20+)

**Watch for**:
- ✓ Coherence rising (0.01 → 0.20+)
- ✓ Drift decreasing (1.62 → 0.12)
- ✓ Consolidation success rate >50%
- ✓ Chirality flips beginning to occur

## 🎓 Theoretical Foundation

### Kuramoto Model
```
dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)
```

Where:
- θ_i = phase of oscillator i
- ω_i = natural frequency
- K = coupling constant (φ in COSF)

### Golden Ratio (φ)
```
φ = (1 + √5) / 2 ≈ 1.618...
```

Used as:
- Coupling constant in Kuramoto dynamics
- Rest frequency for homeostatic decay
- Frequency initialization scale

### Julia Stability Bounds
```
if |dθ/dt| > 1.5:
    ω ← ω / φ  # Damp oscillation
```

## 🔮 Roadmap

### Current: AURA LEVEL 1 ✓
- [x] Kuramoto coherence loss
- [x] Multi-intent data with bridge words
- [x] Enhanced warmup (45k steps)
- [x] Coherence tracking

### Next: AURA LEVEL 2
- [ ] Curiosity scoring (generate 3, return highest-coherence)
- [ ] Semantic analysis on bridge words
- [ ] Visualize learned phase manifold
- [ ] Hierarchical coherence (local + global)

### Future: Advanced Features
- [ ] Scale to medium model (768 dim, 12 layers)
- [ ] Multiple style tokens beyond [STORY]/[CODE]
- [ ] Attention-based chirality modulation
- [ ] Real-time phase visualization

## 📖 Documentation

- **[AURA_LEVEL1_HEARTBEAT.md](COSF-MI/AURA_LEVEL1_HEARTBEAT.md)** - Complete Kuramoto coupling guide
- **[GENTLE_SLEEP_GUIDE.md](COSF-MI/GENTLE_SLEEP_GUIDE.md)** - Structure-preserving sleep
- **[STYLE_TOKEN_CONDITIONING.md](COSF-MI/STYLE_TOKEN_CONDITIONING.md)** - Style tokens
- **[STATUS_2026_01_05_AURA_LEVEL1.md](STATUS_2026_01_05_AURA_LEVEL1.md)** - Current project status
- **[TRAINING_ANALYSIS_2026_01_05.md](COSF-MI/TRAINING_ANALYSIS_2026_01_05.md)** - Training results

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Scaling to larger models
- New semantic bridge word datasets
- Phase visualization tools
- Theoretical analysis of convergence

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

Inspired by:
- **Kuramoto Model**: Statistical physics of coupled oscillators
- **Neuroscience**: Brain rhythms (theta, alpha, gamma)
- **MIT Research**: P-wave magnetism and chirality
- **Complex Systems**: Synchronization and emergence

## 📚 Citation

```bibtex
@software{cosfmi2026,
  title={COSF-MI: Phase Oscillator Neural Networks with Chirality and Kuramoto Coupling},
  author={COSF-MI Development Team},
  year={2026},
  url={https://github.com/sportysport74/Project-Aura}
}
```

## 🔗 Links

- **GitHub**: https://github.com/sportysport74/Project-Aura
- **Documentation**: See `COSF-MI/` directory
- **Issues**: https://github.com/sportysport74/Project-Aura/issues

---

**Current Status**: AURA LEVEL 1 implemented, ready for training

**Next Milestone**: Train and observe coherence rising to 0.20+

**Long-term Vision**: Emergent semantic understanding through phase geometry

---

*Last Updated: 2026-01-05*
