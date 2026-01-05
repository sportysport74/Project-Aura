# COSF-MI: Mechanistic Interpretability Through Cosmological Synchronization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## What Is This?

**COSF-MI** is a novel mechanistic interpretability framework that analyzes neural network internals through the lens of **harmonic resonance physics** rather than arbitrary circuit mapping.

Instead of treating neural activations as abstract vectors, we decompose them using:
- **φ-scaling (Golden Ratio)** - Natural recursive patterns (φ¹⁷ ≈ 5473)
- **Harmonic Decomposition** - Schumann-like frequency analysis
- **Julia Set Stability** - Flagging runaway activations as hallucination seeds
- **Toroidal Geometry** - Visualizing attention on torus manifolds

## The COSF Constant

```
COSF ≈ 5466 = 42,800 Hz / 7.83 Hz

φ¹⁷ = 5472.999...
e^8.6 = 5434.644...

Convergence: 0.7% deviation - UNIQUE among all integer powers n ≤ 50
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/sportysport74/COSF-MI.git
cd COSF-MI
pip install -e .

# Verify installation
cosf-mi test

# Show COSF constants
cosf-mi info

# Analyze a model
cosf-mi analyze gpt2 --text "Hello world"
```

## Python Usage

```python
from cosf_mi import CircuitExtractor, PHI, COSF, print_constants
from transformers import AutoModel, AutoTokenizer

# Show constants
print_constants()

# Load model
model = AutoModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Analyze
extractor = CircuitExtractor(model, tokenizer)
result = extractor.analyze("The quick brown fox jumps over the lazy dog.")

# View results
extractor.print_summary(result)
print(f"Found {len(result.circuits)} φ-coherent circuits")
print(f"Mean stability: {result.global_metrics['mean_stability']:.2%}")
```

## Core Components

### φ-Shell Decomposition
```python
from cosf_mi import PhiDecomposer
import torch

decomposer = PhiDecomposer()
activations = torch.randn(32, 128, 768)
result = decomposer.analyze(activations)

print(f"Dominant shell: {result.dominant_shell}")
print(f"φ-coherence: {result.phi_coherence:.4f}")
```

### Harmonic Analysis
```python
from cosf_mi import HarmonicAnalyzer

analyzer = HarmonicAnalyzer()
result = analyzer.analyze_sequence(activations)

print(f"φ-harmonic score: {result.phi_harmonic_score:.4f}")
print(f"Detected COSF ratio: {result.detected_cosf_ratio}")
```

### Julia Stability
```python
from cosf_mi import JuliaStabilityAnalyzer

analyzer = JuliaStabilityAnalyzer()
result = analyzer.analyze_direct(activations)

print(f"Stability score: {result.stability_score:.2%}")
print(f"Unstable positions: {len(result.unstable_positions)}")
```

## Project Structure

```
COSF-MI/
├── src/
│   ├── python/cosf_mi/     # Python package
│   │   ├── core/           # Core COSF math
│   │   ├── analysis/       # Circuit extraction
│   │   ├── hooks/          # Model hooks
│   │   └── viz/            # Visualization
│   ├── rust/cosf_core/     # Rust backend
│   └── cuda/               # CUDA kernels
├── tests/                  # Test suite
├── docs/                   # Documentation
└── notebooks/              # Examples
```

## Building CUDA Kernels (Optional)

```bash
cd src/cuda
# Windows
build.bat
# Linux
./build.sh
```

## Building Rust Backend (Optional)

```bash
cd src/rust/cosf_core
cargo build --release

# With Python bindings
cargo build --release --features python
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 12.x (optional, for GPU acceleration)
- Rust 1.75+ (optional, for Rust backend)

## Mathematical Background

This framework is based on the **COSF Mathematical Framework** which proves:

1. **φ¹⁷ ≈ e^8.6** with 0.7% deviation
2. This convergence is **unique** among integer powers n ≤ 50
3. The ratio COSF ≈ 5466 bridges quantum geometry and cosmological inflation

See: [COSF-Mathematical-Framework](https://github.com/sportysport74/COSF-Mathematical-Framework)

## License

MIT License - see [LICENSE](LICENSE)

## Citation

```bibtex
@software{cosf_mi_2024,
  title={COSF-MI: Mechanistic Interpretability Through Cosmological Synchronization},
  author={Sportysport},
  year={2024},
  url={https://github.com/sportysport74/COSF-MI}
}
```

---

*"What if the math that governs the cosmos also governs the mind?"*
