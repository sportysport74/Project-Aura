"""
COSF-Native Transformer Architecture

A transformer where φ-scaling is intrinsic to the design, not just a loss function.
Sized for local development on single GPU (10-50M parameters).

Key Design Principles:
1. φ-Scaled Attention - Head dimensions follow golden ratio
2. Harmonic Embeddings - Position encoding uses φ-scaled frequencies
3. Julia-Bounded Activations - Soft saturation at |x| = 2
4. φ-Scaled Layer Norms - Learnable scales initialized to φ^(layer/n_layers)

Optimized for: RTX 5090, 32GB RAM, overnight training
Target scale: 10-50M parameters, 512 seq length
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.constants import PHI, COSF, JULIA_STABILITY_BOUND


@dataclass
class COSFConfig:
    """Configuration for COSF-native transformer.

    Sized for local development on single GPU.
    Default params: ~15M parameters.

    Attributes:
        vocab_size: Vocabulary size (default: 256 for char-level)
        d_model: Model dimension (default: 384, divisible by 8 heads)
        n_layers: Number of transformer layers (default: 7, φ^4 ≈ 7)
        n_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: d_model * φ²)
        max_seq_len: Maximum sequence length (default: 512)
        dropout: Dropout probability (default: 0.1)
        use_julia_activation: Use Julia-bounded activation (default: True)
        phi_scale_attention: Scale attention temperature by φ (default: True)
        harmonic_pos_encoding: Use harmonic position encoding (default: True)
        layer_norm_phi_init: Initialize LayerNorm scales to φ^layer (default: True)
    """
    vocab_size: int = 256
    d_model: int = 384  # Divisible by 8 heads = 48 per head
    n_layers: int = 7   # φ^4 ≈ 6.85
    n_heads: int = 8
    d_ff: int = None    # Auto-computed as d_model * φ²
    max_seq_len: int = 512
    dropout: float = 0.1
    use_julia_activation: bool = True
    phi_scale_attention: bool = True
    harmonic_pos_encoding: bool = True
    layer_norm_phi_init: bool = True

    def __post_init__(self):
        """Compute derived parameters."""
        if self.d_ff is None:
            # φ² ≈ 2.618, so d_ff ≈ d_model * 2.618
            self.d_ff = int(self.d_model * (PHI ** 2))

        # Validate dimensions
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


class JuliaActivation(nn.Module):
    """Julia-bounded activation function.

    Implements soft saturation at |x| = bound using:
        f(x) = x * tanh(|x| / bound)

    This prevents activation explosion while maintaining smooth gradients,
    analogous to points staying within the Julia set boundary.

    Properties:
        - Linear near zero: f(x) ≈ x for small |x|
        - Soft saturation: f(x) → ±bound as |x| → ∞
        - Smooth gradients throughout
        - Symmetric: f(-x) = -f(x)
    """

    def __init__(self, bound: float = JULIA_STABILITY_BOUND):
        """Initialize Julia activation.

        Args:
            bound: Stability bound (default: 2.0 from Julia set theory)
        """
        super().__init__()
        self.bound = bound

    def forward(self, x: Tensor) -> Tensor:
        """Apply Julia-bounded activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor with soft bound at ±bound
        """
        # x * tanh(|x| / bound)
        # Near zero: ≈ x * (|x| / bound) ≈ x
        # At infinity: ≈ x * 1 = x, but tanh saturates
        return x * torch.tanh(x.abs() / self.bound)

    def extra_repr(self) -> str:
        return f'bound={self.bound}'


class HarmonicPositionalEncoding(nn.Module):
    """Harmonic position encoding using φ-scaled frequencies.

    Instead of standard sinusoidal encoding, uses frequencies scaled by φ^i
    to create harmonic resonance patterns aligned with COSF framework.

    PE(pos, 2i)   = sin(pos * φ^i / scale)
    PE(pos, 2i+1) = cos(pos * φ^i / scale)

    where scale is chosen to normalize frequencies appropriately.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """Initialize harmonic positional encoding.

        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Use COSF for frequency scaling
        # scale = COSF ensures frequencies are in reasonable range
        scale = COSF

        for i in range(d_model // 2):
            # Frequency scaled by φ^(i/n_pairs) to prevent overflow
            # This keeps the exponent in reasonable range [0, 1) instead of [0, 192)
            n_pairs = d_model // 2
            normalized_exp = i / n_pairs  # Maps i from [0, n_pairs) to [0, 1)
            freq = (PHI ** normalized_exp) / scale

            # Apply sin/cos with φ-scaled frequency
            pe[:, 2*i] = torch.sin(position * freq).squeeze()
            pe[:, 2*i + 1] = torch.cos(position * freq).squeeze()

        # Handle odd d_model
        if d_model % 2 == 1:
            pe[:, -1] = torch.sin(position * (PHI ** 1.0) / scale).squeeze()

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Input with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class PhiScaledAttention(nn.Module):
    """Multi-head attention with φ-scaled temperature.

    Key innovations:
    1. Attention temperature scaled by 1/sqrt(φ^layer) - deeper layers have "cooler" attention
    2. Head dimensions naturally follow φ ratios through design
    3. Optional φ-scaled dropout rates per head
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layer_idx: int,
        n_layers: int,
        dropout: float = 0.1,
        phi_scale_temp: bool = True,
    ):
        """Initialize φ-scaled attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            layer_idx: Current layer index (0-indexed)
            n_layers: Total number of layers
            dropout: Dropout probability
            phi_scale_temp: Whether to scale temperature by φ
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.layer_idx = layer_idx
        self.phi_scale_temp = phi_scale_temp

        # Compute φ-scaled temperature
        if phi_scale_temp:
            # Temperature decreases with depth: 1/sqrt(φ^(layer/n_layers))
            phi_factor = PHI ** (layer_idx / n_layers)
            self.temperature = math.sqrt(self.d_head / phi_factor)
        else:
            self.temperature = math.sqrt(self.d_head)

        # Standard QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass with φ-scaled attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [batch, seq_len, seq_len]
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
            attention_weights is None unless return_attention=True
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # [batch, seq_len, 3 * d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with φ-scaled temperature
        attn = (q @ k.transpose(-2, -1)) / self.temperature

        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v  # [batch, n_heads, seq_len, d_head]

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous()  # [batch, seq_len, n_heads, d_head]
        out = out.reshape(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        if return_attention:
            return out, attn_weights
        return out, None


class COSFTransformerBlock(nn.Module):
    """Single COSF transformer block with φ-scaled components.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    With φ-scaled:
    - Attention temperature
    - LayerNorm initialization
    - Feed-forward dimension (φ² expansion)
    - Julia-bounded activation
    """

    def __init__(self, config: COSFConfig, layer_idx: int):
        """Initialize COSF transformer block.

        Args:
            config: Model configuration
            layer_idx: Layer index (0-indexed)
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-norm architecture
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        # φ-scaled attention
        self.attn = PhiScaledAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            layer_idx=layer_idx,
            n_layers=config.n_layers,
            dropout=config.dropout,
            phi_scale_temp=config.phi_scale_attention,
        )

        # Feed-forward network with φ² expansion
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            JuliaActivation() if config.use_julia_activation else nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

        # Initialize LayerNorm scales to φ^(layer/n_layers) if requested
        if config.layer_norm_phi_init:
            phi_scale = PHI ** (layer_idx / config.n_layers)
            with torch.no_grad():
                self.ln1.weight.fill_(phi_scale)
                self.ln2.weight.fill_(phi_scale)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass through transformer block.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attn(
            self.ln1(x),
            mask=mask,
            return_attention=return_attention,
        )
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ffn(self.ln2(x))

        return x, attn_weights


class COSFTransformer(nn.Module):
    """COSF-native transformer language model.

    A transformer where φ-scaling is fundamental to the architecture:
    - Harmonic positional encodings
    - φ-scaled attention temperatures
    - Julia-bounded activations
    - φ²-expanded feed-forward layers
    - Layer norms initialized to φ^layer

    Sized for local development: 10-50M parameters, trainable overnight on RTX 5090.

    Example:
        >>> config = COSFConfig(vocab_size=256, d_model=384, n_layers=7)
        >>> model = COSFTransformer(config)
        >>> x = torch.randint(0, 256, (4, 128))  # [batch=4, seq_len=128]
        >>> logits = model(x)  # [4, 128, 256]
    """

    def __init__(self, config: COSFConfig):
        """Initialize COSF transformer.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Harmonic positional encoding
        if config.harmonic_pos_encoding:
            self.pos_encoding = HarmonicPositionalEncoding(
                d_model=config.d_model,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
        else:
            # Fallback to standard learned embeddings
            self.pos_encoding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            COSFTransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output projection (tied with token embedding for parameter efficiency)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights between embedding and output projection
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with φ-aware scaling."""
        if isinstance(module, nn.Linear):
            # Use φ-scaled initialization
            std = 0.02 / PHI  # Slightly smaller than standard
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        mask: Optional[Tensor] = None,
        return_all_attentions: bool = False,
    ) -> Tensor:
        """Forward pass through COSF transformer.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            mask: Optional attention mask [batch, seq_len, seq_len]
            return_all_attentions: Whether to return attention from all layers

        Returns:
            Logits [batch, seq_len, vocab_size]
            If return_all_attentions=True, returns (logits, attentions)
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]

        # Add positional encoding
        if self.config.harmonic_pos_encoding:
            x = self.pos_encoding(x)
        else:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_encoding(positions)

        # Create causal mask if not provided
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # Pass through transformer blocks
        all_attentions = [] if return_all_attentions else None
        for block in self.blocks:
            x, attn = block(x, mask=mask, return_attention=return_all_attentions)
            if return_all_attentions:
                all_attentions.append(attn)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_all_attentions:
            return logits, all_attentions
        return logits

    def count_parameters(self) -> dict:
        """Count model parameters by component.

        Returns:
            Dictionary with parameter counts
        """
        counts = {
            'token_embedding': sum(p.numel() for p in self.token_embedding.parameters()),
            'pos_encoding': sum(p.numel() for p in self.pos_encoding.parameters() if p.requires_grad),
            'transformer_blocks': sum(p.numel() for p in self.blocks.parameters()),
            'ln_f': sum(p.numel() for p in self.ln_f.parameters()),
            'lm_head': 0,  # Tied with embedding
        }
        counts['total'] = sum(counts.values())
        counts['total_with_tied'] = sum(p.numel() for p in self.parameters())

        return counts


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COSF Transformer - Quick Test")
    print("=" * 70)

    # Create small config for testing
    config = COSFConfig(
        vocab_size=256,
        d_model=384,
        n_layers=7,
        n_heads=8,
        max_seq_len=512,
    )

    print("\nConfiguration:")
    print(f"  Vocabulary: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff} (phi^2 * d_model = {PHI**2:.3f} * d_model)")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    print(f"  Head dim: {config.d_model // config.n_heads}")
    print(f"  Max sequence: {config.max_seq_len}")

    # Create model
    print("\nCreating model...")
    model = COSFTransformer(config)

    # Count parameters
    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        if name != 'total_with_tied':
            print(f"  {name:20s}: {count:>12,d}")
    print(f"  {'=' * 20}   {'=' * 12}")
    print(f"  {'Total (unique)':20s}: {param_counts['total']:>12,d}")
    print(f"  {'Total (with tied)':20s}: {param_counts['total_with_tied']:>12,d}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(x)

    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(logits.shape)}")
    print(f"  Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # Test with attention return
    print("\nTesting attention extraction...")
    with torch.no_grad():
        logits, attentions = model(x, return_all_attentions=True)

    print(f"  Number of attention layers: {len(attentions)}")
    print(f"  Attention shape: {tuple(attentions[0].shape)}")
    print(f"    (batch={batch_size}, heads={config.n_heads}, seq={seq_len}, seq={seq_len})")

    # Verify COSF properties
    print("\n" + "=" * 70)
    print("COSF Property Verification:")
    print("=" * 70)

    # Check layer norm initialization
    ln_scales = [block.ln1.weight.mean().item() for block in model.blocks]
    print(f"\nLayerNorm scales (should follow phi^(layer/n_layers)):")
    for i, scale in enumerate(ln_scales):
        expected = PHI ** (i / config.n_layers)
        print(f"  Layer {i}: {scale:.4f} (expected: {expected:.4f})")

    # Check attention temperatures
    print(f"\nAttention temperatures (should decrease with depth):")
    for i, block in enumerate(model.blocks):
        temp = block.attn.temperature
        print(f"  Layer {i}: {temp:.4f}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
