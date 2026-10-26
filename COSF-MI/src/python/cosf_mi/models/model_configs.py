"""
COSF Transformer Model Configurations

Predefined model sizes optimized for the COSF framework and RTX 5090 training.

All models use:
- φ-scaled dimensions
- Fibonacci-adjacent layer counts
- Julia-bounded activations
- Harmonic positional encodings
"""

from dataclasses import dataclass
from typing import Dict
from .cosf_transformer import COSFConfig
from ..core.constants import PHI


# ============================================================================
# Model Size Presets
# ============================================================================

def get_cosf_config(model_size: str, vocab_size: int = 8000, max_seq_len: int = 512) -> COSFConfig:
    """Get predefined COSF model configuration.

    Args:
        model_size: One of ['tiny', 'small', 'medium', 'large']
        vocab_size: Vocabulary size (default: 8000 for BPE)
        max_seq_len: Maximum sequence length

    Returns:
        COSFConfig with appropriate dimensions
    """
    configs = {
        # 9.7M parameters - current Shakespeare model
        'tiny': {
            'd_model': 384,      # Divisible by 8 heads = 48 per head
            'n_layers': 7,       # Fibonacci-adjacent (φ^4 ≈ 6.85)
            'n_heads': 8,
            'max_seq_len': 512,
        },

        # ~50M parameters - GPT-2 small territory, actually useful
        'small': {
            'd_model': 768,      # Divisible by 12 heads = 64 per head
            'n_layers': 12,      # Fibonacci number
            'n_heads': 12,
            'max_seq_len': 1024,
        },

        # ~100M parameters - strong performance, fits RTX 5090 comfortably
        'medium': {
            'd_model': 1024,     # Divisible by 16 heads = 64 per head
            'n_layers': 13,      # Fibonacci number, φ^6 ≈ 11.09
            'n_heads': 16,       # Power of 2, efficient
            'max_seq_len': 1024,
        },

        # ~200M parameters - approaching GPT-2 medium, slower training
        'large': {
            'd_model': 1280,     # Divisible by 8 heads = 160 per head
            'n_layers': 21,      # Fibonacci number
            'n_heads': 16,
            'max_seq_len': 1024,
        },
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size '{model_size}'. Choose from {list(configs.keys())}")

    config_dict = configs[model_size]

    return COSFConfig(
        vocab_size=vocab_size,
        d_model=config_dict['d_model'],
        n_layers=config_dict['n_layers'],
        n_heads=config_dict['n_heads'],
        max_seq_len=config_dict.get('max_seq_len', max_seq_len),
        d_ff=None,  # Auto-computed as d_model * φ²
        dropout=0.1,
        use_julia_activation=True,
        phi_scale_attention=True,
        harmonic_pos_encoding=True,
        layer_norm_phi_init=True,
    )


def estimate_parameters(config: COSFConfig) -> Dict[str, int]:
    """Estimate parameter count for a given config.

    Args:
        config: COSF model configuration

    Returns:
        Dictionary with parameter counts by component
    """
    d_model = config.d_model
    d_ff = config.d_ff if config.d_ff else int(d_model * (PHI ** 2))
    n_layers = config.n_layers
    vocab_size = config.vocab_size

    # Token + position embeddings
    embedding_params = vocab_size * d_model + config.max_seq_len * d_model

    # Per-layer parameters
    # Attention: Q, K, V projections + output projection
    attn_params_per_layer = 4 * (d_model * d_model)

    # Feed-forward: 2 linear layers
    ff_params_per_layer = d_model * d_ff + d_ff * d_model

    # Layer norms (2 per layer: pre-attn, pre-ff)
    ln_params_per_layer = 2 * 2 * d_model  # scale + bias for each

    layer_params = n_layers * (attn_params_per_layer + ff_params_per_layer + ln_params_per_layer)

    # Final layer norm + output head
    final_params = 2 * d_model + d_model * vocab_size

    total = embedding_params + layer_params + final_params

    return {
        'embedding': embedding_params,
        'layers': layer_params,
        'attention_per_layer': attn_params_per_layer,
        'feedforward_per_layer': ff_params_per_layer,
        'final': final_params,
        'total': total,
    }


def print_model_comparison():
    """Print comparison table of all model sizes."""
    print("=" * 100)
    print("COSF TRANSFORMER MODEL CONFIGURATIONS")
    print("=" * 100)
    print()

    sizes = ['tiny', 'small', 'medium', 'large']

    print(f"{'Model':<10} {'d_model':<10} {'Layers':<8} {'Heads':<8} {'Seq':<8} {'Params':<12} {'Notes':<30}")
    print("-" * 100)

    for size in sizes:
        config = get_cosf_config(size, vocab_size=8000)
        params = estimate_parameters(config)

        notes = {
            'tiny': 'Current Shakespeare model',
            'small': 'GPT-2 small territory',
            'medium': 'Strong, fits RTX 5090',
            'large': 'GPT-2 medium-ish',
        }

        print(f"{size:<10} {config.d_model:<10} {config.n_layers:<8} {config.n_heads:<8} "
              f"{config.max_seq_len:<8} {params['total'] / 1e6:>8.1f}M    {notes[size]}")

    print()
    print("All configs use:")
    print("  - φ-scaled feed-forward dimension (d_ff = d_model * φ²)")
    print("  - Fibonacci or Fibonacci-adjacent layer counts")
    print("  - Julia-bounded activations")
    print("  - Harmonic positional encodings")
    print()
    print("=" * 100)


# ============================================================================
# Training Configuration Helpers
# ============================================================================

def get_recommended_batch_size(model_size: str, seq_len: int = 512, gpu_memory_gb: int = 32) -> int:
    """Get recommended batch size for a model size and GPU memory.

    Args:
        model_size: Model size name
        seq_len: Sequence length
        gpu_memory_gb: GPU memory in GB (default: 32 for RTX 5090)

    Returns:
        Recommended batch size
    """
    # Rough estimates based on model size and sequence length
    # These are conservative estimates for RTX 5090 with 32GB

    base_batch_sizes = {
        'tiny': 128,    # ~9M params
        'small': 16,    # ~72M params (reduced for RTX 5090 memory safety)
        'medium': 8,    # ~100M params
        'large': 4,     # ~200M params
    }

    batch_size = base_batch_sizes.get(model_size, 32)

    # Adjust for sequence length
    if seq_len > 512:
        batch_size = batch_size // 2
    if seq_len > 1024:
        batch_size = batch_size // 2

    # Adjust for GPU memory
    if gpu_memory_gb < 24:
        batch_size = batch_size // 2

    return max(1, batch_size)


def estimate_training_time(
    model_size: str,
    dataset_size_mb: int,
    num_epochs: int,
    batch_size: int,
    seq_len: int = 512,
) -> Dict[str, float]:
    """Estimate training time for a configuration.

    Args:
        model_size: Model size name
        dataset_size_mb: Dataset size in MB
        num_epochs: Number of training epochs
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Dictionary with time estimates
    """
    # Rough estimates based on RTX 5090 performance
    # These are very approximate!

    # Tokens per second (throughput) estimates for RTX 5090
    tokens_per_sec = {
        'tiny': 50000,    # 9M params, very fast
        'small': 20000,   # 50M params
        'medium': 10000,  # 100M params
        'large': 5000,    # 200M params
    }

    tps = tokens_per_sec.get(model_size, 10000)

    # Estimate total tokens
    # Assuming ~1 character = ~0.3 tokens for BPE
    total_chars = dataset_size_mb * 1024 * 1024
    total_tokens = total_chars * 0.3

    # Tokens per epoch
    tokens_per_epoch = total_tokens

    # Total training tokens
    total_training_tokens = tokens_per_epoch * num_epochs

    # Training time in seconds
    training_time_sec = total_training_tokens / tps

    # Convert to human-readable
    training_time_hours = training_time_sec / 3600
    training_time_days = training_time_hours / 24

    return {
        'total_tokens': total_training_tokens,
        'seconds': training_time_sec,
        'minutes': training_time_sec / 60,
        'hours': training_time_hours,
        'days': training_time_days,
        'tokens_per_second': tps,
    }


if __name__ == "__main__":
    print_model_comparison()

    print("\nRECOMMENDED CONFIGURATIONS FOR RTX 5090:\n")

    print("Small (50M params) - Best for quick iteration:")
    print("  Model: small")
    print("  Batch size: 64")
    print("  Sequence length: 512")
    print("  Dataset: TinyStories (100MB)")
    print("  Training time: ~2-3 hours for 10 epochs")
    print()

    print("Medium (100M params) - Best performance/cost:")
    print("  Model: medium")
    print("  Batch size: 32")
    print("  Sequence length: 512")
    print("  Dataset: TinyStories (500MB)")
    print("  Training time: ~10-15 hours for 10 epochs")
    print()
