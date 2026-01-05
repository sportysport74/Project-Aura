"""
COSF-MI Training Module

Differentiable loss functions for training neural networks with COSF constraints.
Enables models to learn φ-aligned, harmonically consistent, and Julia-stable representations.
"""

from .cosf_losses import (
    COSFLossConfig,
    phi_shell_loss,
    harmonic_consistency_loss,
    julia_stability_loss,
    phase_coherence_loss,
    combined_cosf_loss,
)

__all__ = [
    "COSFLossConfig",
    "phi_shell_loss",
    "harmonic_consistency_loss",
    "julia_stability_loss",
    "phase_coherence_loss",
    "combined_cosf_loss",
]
