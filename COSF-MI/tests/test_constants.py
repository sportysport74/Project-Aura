"""Tests for COSF constants and verification."""

import pytest
import math
import numpy as np

from cosf_mi.core.constants import (
    PHI,
    COSF,
    PHI_17,
    E_8_6,
    CONVERGENCE_RATIO,
    JULIA_STABILITY_BOUND,
    MATRYOSHKA_SHELLS,
    TOROIDAL_SEGMENTS,
    PHI_POWERS,
    CONSTANTS,
    verify_cosf_convergence,
    find_phi_shell,
    julia_iterate,
    compute_phase_coherence,
)


class TestConstants:
    """Test fundamental constants."""
    
    def test_phi_value(self):
        """Test golden ratio value."""
        expected = (1 + math.sqrt(5)) / 2
        assert abs(PHI - expected) < 1e-15
    
    def test_phi_property(self):
        """Test φ² = φ + 1 property."""
        assert abs(PHI ** 2 - (PHI + 1)) < 1e-14
    
    def test_cosf_ratio(self):
        """Test COSF is ratio of Schumann frequencies."""
        assert abs(COSF - 42800 / 7.83) < 0.01
    
    def test_phi_17_value(self):
        """Test φ¹⁷ calculation."""
        assert abs(PHI_17 - PHI ** 17) < 1e-6
    
    def test_e_8_6_value(self):
        """Test e^8.6 calculation."""
        assert abs(E_8_6 - math.e ** 8.6) < 1e-6
    
    def test_convergence_ratio(self):
        """Test convergence ratio is close to 1."""
        assert 1.0 < CONVERGENCE_RATIO < 1.01
    
    def test_julia_bound(self):
        """Test Julia stability bound."""
        assert JULIA_STABILITY_BOUND == 2.0
    
    def test_matryoshka_shells(self):
        """Test number of shells matches φ¹⁷ exponent."""
        assert MATRYOSHKA_SHELLS == 17
    
    def test_toroidal_segments(self):
        """Test 72 segments for 5° spacing."""
        assert TOROIDAL_SEGMENTS == 72
        assert 360 / TOROIDAL_SEGMENTS == 5


class TestPhiPowers:
    """Test precomputed φ powers."""
    
    def test_phi_powers_length(self):
        """Test array has correct length."""
        assert len(PHI_POWERS) == 18
    
    def test_phi_powers_values(self):
        """Test each power is correct."""
        for n, power in enumerate(PHI_POWERS):
            expected = PHI ** n
            assert abs(power - expected) < 1e-10


class TestCOSFConstants:
    """Test COSFConstants dataclass."""
    
    def test_constants_instance(self):
        """Test global constants instance exists."""
        assert CONSTANTS is not None
    
    def test_phi_power_method(self):
        """Test phi_power method."""
        for n in range(18):
            assert abs(CONSTANTS.phi_power(n) - PHI ** n) < 1e-10
    
    def test_shell_boundaries(self):
        """Test shell boundary computation."""
        boundaries = CONSTANTS.shell_boundaries(base_magnitude=1.0)
        assert len(boundaries) == MATRYOSHKA_SHELLS + 1
        assert boundaries[0] == 1.0
        assert abs(boundaries[1] - PHI) < 1e-10
    
    def test_harmonic_frequencies(self):
        """Test harmonic frequency generation."""
        harmonics = CONSTANTS.harmonic_frequencies(100.0)
        assert len(harmonics) == 5
        assert harmonics[0] == 100.0
        assert abs(harmonics[1] - 100.0 * PHI) < 1e-6


class TestVerification:
    """Test verification functions."""
    
    def test_verify_convergence(self):
        """Test COSF convergence verification."""
        result = verify_cosf_convergence()
        
        assert "phi" in result
        assert "phi_17" in result
        assert "e_8_6" in result
        assert "convergence_ratio" in result
        assert "deviation_percent" in result
        assert "is_valid" in result
        
        assert result["is_valid"] is True
        assert result["deviation_percent"] < 1.0


class TestFindPhiShell:
    """Test φ-shell finding function."""
    
    def test_below_base(self):
        """Test magnitude below base returns -1."""
        assert find_phi_shell(0.5, base=1.0) == -1
    
    def test_shell_zero(self):
        """Test magnitude in shell 0."""
        assert find_phi_shell(1.0, base=1.0) == 0
        assert find_phi_shell(1.5, base=1.0) == 0
    
    def test_shell_one(self):
        """Test magnitude in shell 1."""
        shell = find_phi_shell(PHI * 1.1, base=1.0)
        assert shell == 1
    
    def test_shell_two(self):
        """Test magnitude in shell 2."""
        shell = find_phi_shell(PHI ** 2 * 1.1, base=1.0)
        assert shell == 2
    
    def test_high_shell(self):
        """Test high magnitude caps at max shell."""
        shell = find_phi_shell(1e10, base=1.0)
        assert shell == MATRYOSHKA_SHELLS - 1


class TestJuliaIterate:
    """Test Julia set iteration."""
    
    def test_stable_point(self):
        """Test stable point stays bounded."""
        z = complex(0.1, 0.1)
        c = complex(-0.7, 0.27015)
        iterations, is_stable = julia_iterate(z, c, max_iter=100)
        assert is_stable is True
        assert iterations == 100
    
    def test_unstable_point(self):
        """Test unstable point escapes."""
        z = complex(10, 10)
        c = complex(-0.7, 0.27015)
        iterations, is_stable = julia_iterate(z, c, max_iter=100)
        assert is_stable is False
        assert iterations < 100
    
    def test_origin(self):
        """Test origin with small c is stable."""
        z = complex(0, 0)
        c = complex(0.1, 0.1)
        _, is_stable = julia_iterate(z, c, max_iter=100)
        assert is_stable is True


class TestPhaseCoherence:
    """Test phase coherence computation."""
    
    def test_identical_phases(self):
        """Test identical phases give coherence 1."""
        phases = np.array([0, 0.5, 1.0, 1.5])
        coherence = compute_phase_coherence(phases, phases)
        assert abs(coherence - 1.0) < 1e-10
    
    def test_opposite_phases(self):
        """Test opposite phases give low coherence."""
        phases_a = np.array([0, 0, 0, 0])
        phases_b = np.array([np.pi, np.pi, np.pi, np.pi])
        coherence = compute_phase_coherence(phases_a, phases_b)
        assert coherence < 0.1
    
    def test_random_phases(self):
        """Test random phases give intermediate coherence."""
        np.random.seed(42)
        phases_a = np.random.uniform(0, 2 * np.pi, 100)
        phases_b = np.random.uniform(0, 2 * np.pi, 100)
        coherence = compute_phase_coherence(phases_a, phases_b)
        assert 0 <= coherence <= 1
    
    def test_empty_arrays(self):
        """Test empty arrays return 0."""
        coherence = compute_phase_coherence(np.array([]), np.array([]))
        assert coherence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
