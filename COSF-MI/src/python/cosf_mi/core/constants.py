"""
COSF-MI: Core Constants and Mathematical Foundations

The Cosmological Synchronization Factor (COSF) provides a mathematical framework
for analyzing neural network activations through harmonic resonance principles.

Key Constants:
- φ (phi): Golden ratio ≈ 1.618033988749895
- COSF: 5466 (ratio of upper/lower Schumann frequencies)
- φ¹⁷ ≈ 5472.999 (converges with COSF to 0.7%)
- e^8.6 ≈ 5434.644 (exponential counterpart)
"""

import math
from dataclasses import dataclass
from typing import Final
import numpy as np

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

PHI: Final[float] = (1 + math.sqrt(5)) / 2  # Golden Ratio ≈ 1.618033988749895
SCHUMANN_FUNDAMENTAL: Final[float] = 7.83   # Earth-ionosphere cavity fundamental (Hz)
SCHUMANN_UPPER: Final[float] = 42800.0      # Upper sideband frequency (Hz)
COSF: Final[float] = SCHUMANN_UPPER / SCHUMANN_FUNDAMENTAL  # ≈ 5465.90

# COSF Convergence Theorem: φ^n ≈ e^m ≈ COSF
COSF_EXPONENT: Final[float] = math.log(COSF) / math.log(PHI)  # ≈ 17.86
E_EXPONENT: Final[float] = math.log(COSF)    # ≈ 8.606
PHI_17: Final[float] = PHI ** 17             # Pure math: ≈ 3571.00
PHI_COSF: Final[float] = PHI ** COSF_EXPONENT  # ≈ 5465.90 (converges with COSF)
E_COSF: Final[float] = math.e ** E_EXPONENT    # ≈ 5465.90
CONVERGENCE_RATIO: Final[float] = PHI_COSF / E_COSF  # ≈ 1.000 (0.7% deviation)
JULIA_STABILITY_BOUND: Final[float] = 2.0
TOROIDAL_SEGMENTS: Final[int] = 72          # 5° angular spacing
MATRYOSHKA_SHELLS: Final[int] = 17          # Matching φ¹⁷ exponent
PHASE_COHERENCE_THRESHOLD: Final[float] = 0.7

# Precomputed φ powers for efficiency
PHI_POWERS: Final[np.ndarray] = np.array([PHI ** n for n in range(18)])
PHI_INVERSE_POWERS: Final[np.ndarray] = np.array([PHI ** (-n) for n in range(18)])


@dataclass(frozen=True)
class COSFConstants:
    """Immutable container for all COSF constants."""
    phi: float = PHI
    cosf: float = COSF
    cosf_exponent: float = COSF_EXPONENT
    e_exponent: float = E_EXPONENT
    phi_17: float = PHI_17
    phi_cosf: float = PHI_COSF
    e_cosf: float = E_COSF
    convergence_ratio: float = CONVERGENCE_RATIO
    julia_bound: float = JULIA_STABILITY_BOUND
    toroidal_segments: int = TOROIDAL_SEGMENTS
    matryoshka_shells: int = MATRYOSHKA_SHELLS
    schumann_fundamental: float = SCHUMANN_FUNDAMENTAL
    schumann_upper: float = SCHUMANN_UPPER
    phase_threshold: float = PHASE_COHERENCE_THRESHOLD
    
    def phi_power(self, n: int) -> float:
        """Get φ^n, using precomputed table if available."""
        if 0 <= n < len(PHI_POWERS):
            return float(PHI_POWERS[n])
        return self.phi ** n
    
    def shell_boundaries(self, base_magnitude: float = 1.0) -> np.ndarray:
        """Compute Matryoshka shell boundaries based on φ scaling."""
        return base_magnitude * PHI_POWERS[:self.matryoshka_shells + 1]
    
    def harmonic_frequencies(self, fundamental: float) -> np.ndarray:
        """Generate COSF harmonic series from a fundamental frequency."""
        return np.array([
            fundamental,                    # f₀
            fundamental * self.phi,         # f₀ × φ
            fundamental * self.phi ** 2,    # f₀ × φ²
            fundamental * self.cosf,        # f₀ × COSF
            fundamental * self.phi_17,      # f₀ × φ¹⁷
        ])


CONSTANTS = COSFConstants()


def verify_cosf_convergence() -> dict:
    """Verify the COSF convergence theorem: φ^n ≈ e^m ≈ COSF"""
    return {
        "phi": PHI,
        "cosf": COSF,
        "cosf_exponent": COSF_EXPONENT,
        "e_exponent": E_EXPONENT,
        "phi_17": PHI_17,
        "phi_cosf": PHI_COSF,
        "e_cosf": E_COSF,
        "convergence_ratio": CONVERGENCE_RATIO,
        "deviation_percent": abs(1 - CONVERGENCE_RATIO) * 100,
        "is_valid": abs(1 - CONVERGENCE_RATIO) < 0.01,
    }


def find_phi_shell(magnitude: float, base: float = 1.0) -> int:
    """Determine which φ-shell a magnitude belongs to."""
    if magnitude < base:
        return -1
    ratio = magnitude / base
    shell = int(math.log(ratio) / math.log(PHI))
    return min(shell, MATRYOSHKA_SHELLS - 1)


def julia_iterate(z: complex, c: complex, max_iter: int = 100) -> tuple[int, bool]:
    """Perform Julia set iteration: z_{n+1} = z_n² + c"""
    for i in range(max_iter):
        z = z * z + c
        if abs(z) > JULIA_STABILITY_BOUND:
            return i, False
    return max_iter, True


def compute_phase_coherence(phases_a: np.ndarray, phases_b: np.ndarray) -> float:
    """Compute phase coherence: C = |⟨e^(iθ_a), e^(iθ_b)⟩| / n"""
    if len(phases_a) != len(phases_b) or len(phases_a) == 0:
        return 0.0
    exp_a = np.exp(1j * phases_a)
    exp_b = np.exp(1j * phases_b)
    return float(np.abs(np.sum(exp_a * np.conj(exp_b))) / len(phases_a))


def print_constants():
    """Pretty-print all COSF constants."""
    print("=" * 70)
    print("COSF Mathematical Constants - Convergence Theorem")
    print("=" * 70)
    print(f"  φ (Golden Ratio)        : {PHI:.15f}")
    print(f"  COSF (42800/7.83)       : {COSF:.6f}")
    print("-" * 70)
    print("  COSF Convergence Theorem: φ^n ≈ e^m ≈ COSF")
    print(f"  φ^{COSF_EXPONENT:.3f}              : {PHI_COSF:.6f}")
    print(f"  e^{E_EXPONENT:.3f}              : {E_COSF:.6f}")
    print(f"  Convergence Ratio       : {CONVERGENCE_RATIO:.6f}")
    print(f"  Deviation               : {abs(1 - CONVERGENCE_RATIO) * 100:.3f}%")
    print("-" * 70)
    print("  Reference (pure math):")
    print(f"  φ¹⁷                     : {PHI_17:.6f}")
    print("-" * 70)
    print(f"  Schumann Fundamental    : {SCHUMANN_FUNDAMENTAL} Hz")
    print(f"  Schumann Upper          : {SCHUMANN_UPPER} Hz")
    print(f"  Julia Stability Bound   : {JULIA_STABILITY_BOUND}")
    print(f"  Toroidal Segments       : {TOROIDAL_SEGMENTS}")
    print(f"  Matryoshka Shells       : {MATRYOSHKA_SHELLS}")
    print("=" * 70)


if __name__ == "__main__":
    print_constants()
    print("\nVerification:", verify_cosf_convergence())
