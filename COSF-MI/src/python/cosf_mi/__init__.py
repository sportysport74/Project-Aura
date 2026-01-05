"""
COSF-MI: Mechanistic Interpretability Through Cosmological Synchronization

A novel framework for analyzing neural networks using harmonic resonance
principles based on the COSF mathematical framework.

Key Features:
- φ-shell decomposition (golden ratio scaling analysis)
- Harmonic FFT analysis (resonance pattern detection)  
- Julia set stability bounds (hallucination detection)
- Toroidal attention visualization
- Circuit extraction and analysis

Example:
    >>> from cosf_mi import CircuitExtractor, PHI, COSF
    >>> from transformers import AutoModel, AutoTokenizer
    >>> 
    >>> model = AutoModel.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> 
    >>> extractor = CircuitExtractor(model, tokenizer)
    >>> result = extractor.analyze("Hello world")
    >>> extractor.print_summary(result)
"""

__version__ = "0.1.0"
__author__ = "Sportysport"

# Core exports
from .core import (
    # Constants
    PHI,
    COSF,
    COSF_EXPONENT,
    E_EXPONENT,
    PHI_17,
    PHI_COSF,
    E_COSF,
    CONVERGENCE_RATIO,
    JULIA_STABILITY_BOUND,
    MATRYOSHKA_SHELLS,
    TOROIDAL_SEGMENTS,
    CONSTANTS,
    # Functions
    verify_cosf_convergence,
    find_phi_shell,
    julia_iterate,
    compute_phase_coherence,
    print_constants,
    # Analyzers
    PhiDecomposer,
    HarmonicAnalyzer,
    JuliaStabilityAnalyzer,
    # Quick functions
    quick_phi_analysis,
    quick_harmonic_analysis,
    quick_stability_check,
)

# Hooks
from .hooks import (
    ModelHooks,
    ActivationCache,
    capture_single_forward,
)

# Analysis
from .analysis import (
    CircuitExtractor,
    Circuit,
    COSFAnalysisResult,
    quick_analysis,
)

__all__ = [
    # Version
    "__version__",
    # Constants
    "PHI",
    "COSF",
    "COSF_EXPONENT",
    "E_EXPONENT",
    "PHI_17",
    "PHI_COSF",
    "E_COSF",
    "CONVERGENCE_RATIO",
    "JULIA_STABILITY_BOUND",
    "MATRYOSHKA_SHELLS",
    "TOROIDAL_SEGMENTS",
    "CONSTANTS",
    # Core functions
    "verify_cosf_convergence",
    "find_phi_shell",
    "julia_iterate",
    "compute_phase_coherence",
    "print_constants",
    # Analyzers
    "PhiDecomposer",
    "HarmonicAnalyzer",
    "JuliaStabilityAnalyzer",
    # Quick functions
    "quick_phi_analysis",
    "quick_harmonic_analysis",
    "quick_stability_check",
    # Hooks
    "ModelHooks",
    "ActivationCache",
    "capture_single_forward",
    # Analysis
    "CircuitExtractor",
    "Circuit",
    "COSFAnalysisResult",
    "quick_analysis",
]
