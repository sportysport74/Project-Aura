"""
Test Chirality Features in COSF Oscillator

This script demonstrates:
1. ChiralOscillatorLayer - Oscillators with left/right handedness
2. SemanticPolarityDetector - Learns which concepts are positive/negative
3. ChiralAttention - Attention heads with alternating chirality
4. Perspective flipping - Changing semantic polarity dynamically

Based on MIT's p-wave magnetism research showing magnets can flip
between left-handed and right-handed spirals.
"""

import torch
import sys
from pathlib import Path

# Add COSF-MI to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from cosf_mi.models.cosf_oscillator import (
    ChiralOscillatorLayer,
    SemanticPolarityDetector,
    ChiralAttention,
    CHIRALITY_RIGHT,
    CHIRALITY_LEFT,
    PHI
)


def test_chiral_oscillator():
    """Test basic chiral oscillator functionality."""
    print("=" * 70)
    print("TEST 1: ChiralOscillatorLayer")
    print("=" * 70)
    print()

    # Create chiral oscillator
    layer = ChiralOscillatorLayer(
        in_features=128,
        out_features=256,
        chirality_init_prob=0.5  # 50/50 right/left
    )

    # Check initial chirality distribution
    stats = layer.get_chirality_stats()
    print(f"Initial chirality distribution:")
    print(f"  Right-handed: {stats['right_handed']}")
    print(f"  Left-handed:  {stats['left_handed']}")
    print(f"  Chirality ratio: {stats['chirality_ratio']:.2f}")
    print(f"  Total flips: {stats['total_flips']}")
    print()

    # Test forward pass
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 128)

    output = layer(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print()

    # Test chirality flipping (perspective shift)
    print("Flipping chirality for first 64 neurons...")
    flip_mask = torch.zeros(256, dtype=torch.bool)
    flip_mask[:64] = True
    layer.flip_chirality(flip_mask)

    stats_after = layer.get_chirality_stats()
    print(f"After flipping:")
    print(f"  Right-handed: {stats_after['right_handed']}")
    print(f"  Left-handed:  {stats_after['left_handed']}")
    print(f"  Total flips: {stats_after['total_flips']}")
    print()


def test_semantic_polarity():
    """Test semantic polarity detection."""
    print("=" * 70)
    print("TEST 2: SemanticPolarityDetector")
    print("=" * 70)
    print()

    # Create polarity detector
    detector = SemanticPolarityDetector(d_model=256, vocab_size=8000)

    # Create mock embeddings
    # Embeddings with positive semantic (e.g., "good", "happy")
    positive_embeddings = torch.randn(2, 5, 256) + 0.5

    # Embeddings with negative semantic (e.g., "bad", "sad")
    negative_embeddings = torch.randn(2, 5, 256) - 0.5

    # Predict polarity
    pos_polarity = detector(positive_embeddings)
    neg_polarity = detector(negative_embeddings)

    print("Polarity predictions:")
    print(f"  Positive concepts: mean={pos_polarity.mean().item():.3f}, std={pos_polarity.std().item():.3f}")
    print(f"  Negative concepts: mean={neg_polarity.mean().item():.3f}, std={neg_polarity.std().item():.3f}")
    print()
    print("Note: After training, positive concepts should trend toward +1")
    print("      and negative concepts should trend toward -1")
    print()


def test_chiral_attention():
    """Test chiral multi-head attention."""
    print("=" * 70)
    print("TEST 3: ChiralAttention")
    print("=" * 70)
    print()

    # Create chiral attention
    attn = ChiralAttention(
        d_model=256,
        n_heads=8,
        dropout=0.1
    )

    # Check chirality stats for each projection
    stats = attn.get_chirality_stats()
    print("Chirality distribution in attention heads:")
    for proj_name, proj_stats in stats.items():
        print(f"  {proj_name.upper()} projection:")
        print(f"    Right-handed: {proj_stats['right_handed']}")
        print(f"    Left-handed:  {proj_stats['left_handed']}")
        print(f"    Ratio: {proj_stats['chirality_ratio']:.2f}")
    print()

    # Test forward pass
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 256)

    output = attn(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print()
    print("Note: Even heads are right-handed, odd heads are left-handed")
    print("      This allows simultaneous positive/negative perspective")
    print()


def test_geometric_semantics():
    """Demonstrate geometric semantic differentiation."""
    print("=" * 70)
    print("TEST 4: Geometric Semantic Differentiation")
    print("=" * 70)
    print()

    print("Concept: Chirality encodes meaning through GEOMETRY")
    print()
    print("Example - 'expensive':")
    print("  Context 1 (luxury): RIGHT-handed spiral -> positive polarity")
    print("  Context 2 (burden): LEFT-handed spiral -> negative polarity")
    print()
    print("The MODEL LEARNS to flip chirality based on context!")
    print()

    # Create two chiral oscillators for same concept in different contexts
    luxury_osc = ChiralOscillatorLayer(128, 256, chirality_init_prob=1.0)  # All right
    burden_osc = ChiralOscillatorLayer(128, 256, chirality_init_prob=0.0)  # All left

    x = torch.randn(1, 1, 128)  # Same input (the word "expensive")

    luxury_output = luxury_osc(x)
    burden_output = burden_osc(x)

    print(f"Same input, different chirality:")
    print(f"  Luxury context (right-handed): mean={luxury_output.mean().item():.3f}")
    print(f"  Burden context (left-handed):  mean={burden_output.mean().item():.3f}")
    print()
    print("The GEOMETRIC difference creates semantic differentiation!")
    print()


def test_terahertz_scaling():
    """Test terahertz frequency scaling for hardware compatibility."""
    print("=" * 70)
    print("TEST 5: Terahertz Scaling (Future Hardware)")
    print("=" * 70)
    print()

    print("For spintronic hardware compatibility, frequencies can be")
    print("scaled to terahertz (10^12 Hz) range.")
    print()

    # Create oscillators with and without THz scaling
    normal_osc = ChiralOscillatorLayer(64, 128, enable_terahertz_scaling=False)
    thz_osc = ChiralOscillatorLayer(64, 128, enable_terahertz_scaling=True)

    print(f"Normal frequency scale: ~{PHI:.3f} (phi)")
    print(f"THz frequency scale:    ~{PHI * 1e12:.3e} Hz")
    print()
    print("This prepares COSF for future quantum/spintronic hardware!")
    print()


def main():
    print()
    print("=" * 70)
    print("CHIRALITY FEATURES TEST - COSF OSCILLATOR")
    print("=" * 70)
    print()
    print("Based on MIT's p-wave magnetism research:")
    print("Magnets can flip between left-handed and right-handed spirals")
    print()
    print("We use this for SEMANTIC DIFFERENTIATION through GEOMETRY")
    print("=" * 70)
    print()

    # Run tests
    test_chiral_oscillator()
    test_semantic_polarity()
    test_chiral_attention()
    test_geometric_semantics()
    test_terahertz_scaling()

    print("=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print()
    print("Key Features Demonstrated:")
    print("  1. Chiral oscillators with left/right handedness")
    print("  2. Semantic polarity detection")
    print("  3. Multi-head attention with alternating chirality")
    print("  4. Geometric semantic differentiation")
    print("  5. Terahertz scaling for future hardware")
    print()
    print("Next Steps:")
    print("  - Enable chirality in training: enable_chirality=True")
    print("  - Train polarity detector to learn concept handedness")
    print("  - Observe perspective shifts during inference")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
