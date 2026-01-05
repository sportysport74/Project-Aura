"""
COSF-MI Hooks Module

Tools for capturing activations from neural network models.
"""

from .model_hooks import (
    ModelHooks,
    ActivationCache,
    capture_single_forward,
    create_transformerlens_hooks,
)

__all__ = [
    "ModelHooks",
    "ActivationCache", 
    "capture_single_forward",
    "create_transformerlens_hooks",
]
