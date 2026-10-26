"""
COSF-MI Models Module

Native COSF transformer architectures with φ-scaling built into the design.
"""

from .cosf_transformer import (
    COSFConfig,
    JuliaActivation,
    HarmonicPositionalEncoding,
    PhiScaledAttention,
    COSFTransformerBlock,
    COSFTransformer,
)

__all__ = [
    "COSFConfig",
    "JuliaActivation",
    "HarmonicPositionalEncoding",
    "PhiScaledAttention",
    "COSFTransformerBlock",
    "COSFTransformer",
]
