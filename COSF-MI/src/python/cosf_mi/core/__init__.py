"""
COSF-MI Core Module

Mathematical foundations for COSF-based mechanistic interpretability.
"""

from .constants import (
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
    verify_cosf_convergence,
    find_phi_shell,
    julia_iterate,
    compute_phase_coherence,
    print_constants,
)

from .phi_decomposition import (
    PhiDecomposer,
    PhiShellResult,
    quick_phi_analysis,
)

from .harmonic_analysis import (
    HarmonicAnalyzer,
    HarmonicAnalysisResult,
    HarmonicPeak,
    MultiScaleHarmonicAnalyzer,
    quick_harmonic_analysis,
)

from .julia_stability import (
    JuliaStabilityAnalyzer,
    JuliaStabilityResult,
    StabilityClass,
    LayerDynamicsAnalyzer,
    quick_stability_check,
)

__all__ = [
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
    # Verification
    "verify_cosf_convergence",
    "find_phi_shell",
    "julia_iterate",
    "compute_phase_coherence",
    "print_constants",
    # Phi decomposition
    "PhiDecomposer",
    "PhiShellResult",
    "quick_phi_analysis",
    # Harmonic analysis
    "HarmonicAnalyzer",
    "HarmonicAnalysisResult",
    "HarmonicPeak",
    "MultiScaleHarmonicAnalyzer",
    "quick_harmonic_analysis",
    # Julia stability
    "JuliaStabilityAnalyzer",
    "JuliaStabilityResult",
    "StabilityClass",
    "LayerDynamicsAnalyzer",
    "quick_stability_check",
]
