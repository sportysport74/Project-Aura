"""
COSF Differentiable Loss Functions

Implements trainable COSF constraints with proper PyTorch autograd gradients.
These losses enable neural networks to learn representations aligned with the
COSF mathematical framework.

Key Principles:
1. φ-shell alignment: Activations cluster at golden ratio magnitudes
2. Harmonic consistency: Layer transitions follow φ scaling
3. Julia stability: Activations remain within stability bounds
4. Phase coherence: Attention heads synchronize in phase space

All losses are fully differentiable and GPU-compatible.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.constants import PHI, JULIA_STABILITY_BOUND, PHASE_COHERENCE_THRESHOLD


@dataclass
class COSFLossConfig:
    """Configuration for COSF loss function hyperparameters.

    Attributes:
        phi_shell_weight: Weight for φ-shell alignment loss (default: 1.0)
        harmonic_weight: Weight for harmonic consistency loss (default: 0.5)
        julia_weight: Weight for Julia stability loss (default: 0.3)
        phase_weight: Weight for phase coherence loss (default: 0.2)
        target_shell: Target φ-shell index for activations (default: 5)
        julia_bound: Maximum allowed activation magnitude (default: 2.0)
        phase_threshold: Minimum phase coherence threshold (default: 0.7)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """
    phi_shell_weight: float = 1.0
    harmonic_weight: float = 0.5
    julia_weight: float = 0.3
    phase_weight: float = 0.2
    target_shell: int = 5
    julia_bound: float = JULIA_STABILITY_BOUND
    phase_threshold: float = PHASE_COHERENCE_THRESHOLD
    epsilon: float = 1e-8


def phi_shell_loss(
    activations: Tensor,
    target_shell: int = 5,
    epsilon: float = 1e-8,
) -> Tensor:
    """Encourage activations to cluster at φ^n magnitudes.

    This loss function pushes activation norms toward golden ratio powers,
    creating a natural hierarchical structure in the learned representations.

    Mathematical formulation:
        L = mean((log(||a||) - target_shell * log(φ))²)

    Gradient (automatically computed by PyTorch):
        ∂L/∂a = 2(log(||a||) - n*log(φ)) * a / ||a||²

    Args:
        activations: Input tensor of any shape [..., features]
        target_shell: Target φ-shell index (n in φ^n)
        epsilon: Small constant for numerical stability

    Returns:
        Scalar loss tensor

    Examples:
        >>> activations = torch.randn(32, 768)  # [batch, features]
        >>> loss = phi_shell_loss(activations, target_shell=5)
        >>> loss.backward()  # Compute gradients
    """
    # Compute L2 norms along the last dimension
    # Shape: [..., features] -> [...]
    norms = torch.norm(activations, p=2, dim=-1) + epsilon

    # Compute log of norms
    log_norms = torch.log(norms)

    # Target log magnitude: log(φ^n) = n * log(φ)
    log_phi = math.log(PHI)
    target_log_norm = target_shell * log_phi

    # Mean squared error in log space
    # This encourages multiplicative rather than additive alignment
    loss = F.mse_loss(log_norms, torch.full_like(log_norms, target_log_norm))

    return loss


def harmonic_consistency_loss(
    layer_activations: List[Tensor],
    epsilon: float = 1e-8,
) -> Tensor:
    """Enforce golden ratio scaling between consecutive layers.

    This loss encourages consecutive layers to maintain magnitude ratios close to φ,
    creating a harmonically consistent information flow through the network.

    Mathematical formulation:
        L = Σ_l (||a_{l+1}|| / ||a_l|| - φ)²

    Args:
        layer_activations: List of activation tensors from consecutive layers
                          Each tensor shape: [batch, ..., features]
        epsilon: Small constant for numerical stability

    Returns:
        Scalar loss tensor

    Examples:
        >>> layer1 = torch.randn(32, 768)
        >>> layer2 = torch.randn(32, 768)
        >>> layer3 = torch.randn(32, 768)
        >>> loss = harmonic_consistency_loss([layer1, layer2, layer3])
    """
    if len(layer_activations) < 2:
        # Need at least 2 layers to compute ratios
        return torch.tensor(0.0, device=layer_activations[0].device)

    total_loss = torch.tensor(0.0, device=layer_activations[0].device)
    num_pairs = 0

    for i in range(len(layer_activations) - 1):
        curr_layer = layer_activations[i]
        next_layer = layer_activations[i + 1]

        # Compute mean magnitude for each layer
        # Shape: [batch, ..., features] -> scalar
        curr_norm = torch.mean(torch.norm(curr_layer, p=2, dim=-1)) + epsilon
        next_norm = torch.mean(torch.norm(next_layer, p=2, dim=-1)) + epsilon

        # Compute ratio between consecutive layers
        ratio = next_norm / curr_norm

        # Target ratio is φ (golden ratio)
        target_ratio = torch.tensor(PHI, device=ratio.device)

        # Add squared error
        total_loss += (ratio - target_ratio) ** 2
        num_pairs += 1

    # Average over all layer pairs
    return total_loss / num_pairs if num_pairs > 0 else total_loss


def julia_stability_loss(
    activations: Tensor,
    bound: float = JULIA_STABILITY_BOUND,
    epsilon: float = 1e-8,
) -> Tensor:
    """Penalize activations that exceed Julia stability bound.

    This loss prevents activations from growing unbounded, which could lead to
    unstable dynamics similar to points escaping the Julia set boundary.

    Mathematical formulation:
        L = mean(ReLU(||a|| - bound)²)

    Uses soft quadratic penalty for smooth gradients.

    Args:
        activations: Input tensor of any shape [..., features]
        bound: Maximum allowed activation magnitude
        epsilon: Small constant for numerical stability

    Returns:
        Scalar loss tensor

    Examples:
        >>> activations = torch.randn(32, 768)
        >>> loss = julia_stability_loss(activations, bound=2.0)
    """
    # Compute L2 norms along the last dimension
    norms = torch.norm(activations, p=2, dim=-1)

    # Compute excess beyond bound
    # ReLU ensures we only penalize violations (norms > bound)
    excess = F.relu(norms - bound)

    # Quadratic penalty for smooth gradients
    # Squared term makes gradient proportional to distance from bound
    loss = torch.mean(excess ** 2)

    return loss


def phase_coherence_loss(
    attention_weights: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """Encourage phase alignment across attention heads.

    This loss maximizes synchronization between attention heads by treating
    attention weights as complex phasors and measuring their alignment.

    Mathematical formulation:
        θ = angle(attention vectors)
        C = |mean(exp(i * θ))|  (complex mean magnitude)
        L = 1 - C

    High coherence (C → 1) means heads are synchronized.
    Low coherence (C → 0) means heads are random/independent.

    Args:
        attention_weights: Tensor of shape [batch, num_heads, seq_len, seq_len]
                          or [batch, num_heads, features]
        epsilon: Small constant for numerical stability

    Returns:
        Scalar loss tensor

    Examples:
        >>> # Multi-head attention weights
        >>> attn = torch.softmax(torch.randn(32, 12, 64, 64), dim=-1)
        >>> loss = phase_coherence_loss(attn)
    """
    # Flatten heads and sequence dimensions for processing
    # Shape: [batch, num_heads, ...] -> [batch, num_heads, -1]
    batch_size = attention_weights.shape[0]
    num_heads = attention_weights.shape[1]
    flattened = attention_weights.reshape(batch_size, num_heads, -1)

    # Compute angles (phases) from attention weights
    # We treat the attention pattern as a vector and compute its angle
    # Shape: [batch, num_heads, features]
    real_part = flattened

    # For phase computation, we need to create a synthetic imaginary part
    # We use a Hilbert-like transform: shift by π/2
    # This is a common technique in signal processing
    imag_part = torch.roll(flattened, shifts=1, dims=-1)

    # Normalize to unit circle (convert to phasors)
    magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2) + epsilon
    real_unit = real_part / magnitude
    imag_unit = imag_part / magnitude

    # Compute complex mean across heads (dimension 1)
    # This measures how aligned the phasors are
    mean_real = torch.mean(real_unit, dim=1)  # [batch, features]
    mean_imag = torch.mean(imag_unit, dim=1)  # [batch, features]

    # Magnitude of complex mean (coherence measure)
    # High magnitude = high coherence (all heads aligned)
    # Low magnitude = low coherence (heads pointing different directions)
    coherence = torch.sqrt(mean_real ** 2 + mean_imag ** 2)

    # Average coherence across feature dimension and batch
    mean_coherence = torch.mean(coherence)

    # Loss: maximize coherence (minimize 1 - coherence)
    loss = 1.0 - mean_coherence

    return loss


def combined_cosf_loss(
    activations: Optional[Tensor] = None,
    layer_activations: Optional[List[Tensor]] = None,
    attention_weights: Optional[Tensor] = None,
    config: Optional[COSFLossConfig] = None,
) -> Tuple[Tensor, dict]:
    """Compute weighted combination of all COSF losses.

    This is the main training loss function that combines all four COSF constraints
    with configurable weights. Use this during training to enforce COSF properties.

    Args:
        activations: Single-layer activations for φ-shell and Julia losses
                    Shape: [batch, ..., features]
        layer_activations: Multi-layer activations for harmonic consistency
                          List of tensors, each shape: [batch, ..., features]
        attention_weights: Multi-head attention for phase coherence
                          Shape: [batch, num_heads, seq_len, seq_len]
        config: Loss configuration with weights and hyperparameters

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains individual losses

    Examples:
        >>> config = COSFLossConfig(phi_shell_weight=1.0, harmonic_weight=0.5)
        >>>
        >>> # Single layer training
        >>> activations = model.forward(x)  # [32, 768]
        >>> loss, losses = combined_cosf_loss(activations=activations, config=config)
        >>>
        >>> # Multi-layer training
        >>> layer_acts = [layer1_out, layer2_out, layer3_out]
        >>> loss, losses = combined_cosf_loss(
        ...     activations=layer_acts[-1],
        ...     layer_activations=layer_acts,
        ...     config=config
        ... )
        >>>
        >>> # With attention
        >>> loss, losses = combined_cosf_loss(
        ...     activations=activations,
        ...     attention_weights=attn_weights,
        ...     config=config
        ... )
    """
    if config is None:
        config = COSFLossConfig()

    total_loss = torch.tensor(0.0)
    loss_dict = {}

    # 1. φ-shell alignment loss
    if activations is not None and config.phi_shell_weight > 0:
        phi_loss = phi_shell_loss(
            activations,
            target_shell=config.target_shell,
            epsilon=config.epsilon,
        )
        total_loss = total_loss + config.phi_shell_weight * phi_loss
        loss_dict['phi_shell'] = phi_loss.item()

    # 2. Harmonic consistency loss
    if layer_activations is not None and config.harmonic_weight > 0:
        harmonic_loss = harmonic_consistency_loss(
            layer_activations,
            epsilon=config.epsilon,
        )
        total_loss = total_loss + config.harmonic_weight * harmonic_loss
        loss_dict['harmonic'] = harmonic_loss.item()

    # 3. Julia stability loss
    if activations is not None and config.julia_weight > 0:
        julia_loss = julia_stability_loss(
            activations,
            bound=config.julia_bound,
            epsilon=config.epsilon,
        )
        total_loss = total_loss + config.julia_weight * julia_loss
        loss_dict['julia_stability'] = julia_loss.item()

    # 4. Phase coherence loss
    if attention_weights is not None and config.phase_weight > 0:
        phase_loss = phase_coherence_loss(
            attention_weights,
            epsilon=config.epsilon,
        )
        total_loss = total_loss + config.phase_weight * phase_loss
        loss_dict['phase_coherence'] = phase_loss.item()

    loss_dict['total'] = total_loss.item()

    return total_loss, loss_dict


# Convenience alias for backward compatibility
cosf_loss = combined_cosf_loss
