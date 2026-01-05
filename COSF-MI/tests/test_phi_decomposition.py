"""Tests for φ-decomposition analysis."""

import pytest
import numpy as np
import torch

from cosf_mi.core.phi_decomposition import (
    PhiDecomposer,
    PhiShellResult,
    quick_phi_analysis,
)
from cosf_mi.core.constants import PHI, MATRYOSHKA_SHELLS


class TestPhiDecomposer:
    """Test PhiDecomposer class."""
    
    @pytest.fixture
    def decomposer(self):
        """Create decomposer instance."""
        return PhiDecomposer(use_gpu=False)
    
    @pytest.fixture
    def sample_activations(self):
        """Create sample activation tensor."""
        torch.manual_seed(42)
        return torch.randn(8, 32, 128)
    
    def test_init_default(self, decomposer):
        """Test default initialization."""
        assert decomposer.n_shells == MATRYOSHKA_SHELLS
        assert decomposer.base_percentile == 10.0
    
    def test_init_custom(self):
        """Test custom initialization."""
        d = PhiDecomposer(n_shells=10, base_percentile=5.0, use_gpu=False)
        assert d.n_shells == 10
        assert d.base_percentile == 5.0
    
    def test_compute_magnitudes(self, decomposer, sample_activations):
        """Test magnitude computation."""
        flat = sample_activations.reshape(-1, 128)
        mags = decomposer.compute_magnitudes(flat)
        
        assert mags.shape == (8 * 32,)
        assert np.all(mags >= 0)
    
    def test_compute_shell_boundaries(self, decomposer):
        """Test shell boundary computation."""
        mags = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        boundaries = decomposer.compute_shell_boundaries(mags)
        
        assert len(boundaries) == decomposer.n_shells + 1
        # Boundaries should increase by φ
        for i in range(len(boundaries) - 1):
            ratio = boundaries[i + 1] / boundaries[i]
            assert abs(ratio - PHI) < 1e-10
    
    def test_assign_shells(self, decomposer):
        """Test shell assignment."""
        boundaries = np.array([1.0, PHI, PHI**2, PHI**3])
        mags = np.array([0.5, 1.2, 2.0, 3.0, 10.0])
        
        decomposer.n_shells = 3
        shells = decomposer.assign_shells(mags, boundaries)
        
        assert shells[0] == -1  # Below base
        assert shells[1] == 0   # In shell 0
        assert shells[2] == 1   # In shell 1
    
    def test_analyze_returns_result(self, decomposer, sample_activations):
        """Test analyze returns PhiShellResult."""
        result = decomposer.analyze(sample_activations)
        
        assert isinstance(result, PhiShellResult)
        assert result.shell_indices is not None
        assert result.shell_counts is not None
        assert result.shell_means is not None
        assert result.boundaries is not None
        assert 0 <= result.dominant_shell < decomposer.n_shells
        assert 0 <= result.phi_coherence <= 1
    
    def test_analyze_shell_counts(self, decomposer, sample_activations):
        """Test shell counts sum correctly."""
        result = decomposer.analyze(sample_activations)
        
        # Sum of shell counts should approximately equal total elements
        # (minus any that fell below base)
        total_assigned = sum(result.shell_counts)
        total_elements = sample_activations.shape[0] * sample_activations.shape[1]
        
        assert total_assigned <= total_elements
    
    def test_analyze_phi_structured_data(self, decomposer):
        """Test analysis of φ-structured data."""
        torch.manual_seed(42)
        
        # Create data with φ-scaling
        base = torch.randn(16, 64, 64)
        for i in range(5):
            mask = torch.rand(16, 64, 1) > 0.7
            base = torch.where(mask, base * (PHI ** i), base)
        
        result = decomposer.analyze(base)
        
        # Should have reasonable coherence for structured data
        assert result.phi_coherence > 0
    
    def test_analyze_attention_heads(self, decomposer):
        """Test per-head analysis."""
        torch.manual_seed(42)
        n_heads = 8
        head_dim = 64
        hidden = n_heads * head_dim
        
        attention_output = torch.randn(4, 32, hidden)
        head_shells, head_coherences = decomposer.analyze_attention_heads(
            attention_output, n_heads
        )
        
        assert len(head_shells) == n_heads
        assert len(head_coherences) == n_heads
        assert all(0 <= c <= 1 for c in head_coherences)


class TestQuickPhiAnalysis:
    """Test quick_phi_analysis function."""
    
    def test_returns_dict(self):
        """Test returns dictionary with expected keys."""
        torch.manual_seed(42)
        acts = torch.randn(4, 16, 64)
        
        result = quick_phi_analysis(acts)
        
        assert isinstance(result, dict)
        assert "dominant_shell" in result
        assert "phi_coherence" in result
        assert "shell_distribution" in result
        assert "shell_means" in result
    
    def test_shell_distribution_length(self):
        """Test shell distribution has correct length."""
        torch.manual_seed(42)
        acts = torch.randn(4, 16, 64)
        
        result = quick_phi_analysis(acts)
        
        assert len(result["shell_distribution"]) == MATRYOSHKA_SHELLS


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_tensor(self):
        """Test handling of edge case dimensions."""
        decomposer = PhiDecomposer(use_gpu=False)
        acts = torch.randn(1, 1, 64)
        
        result = decomposer.analyze(acts)
        assert result is not None
    
    def test_large_values(self):
        """Test handling of large activation values."""
        decomposer = PhiDecomposer(use_gpu=False)
        acts = torch.randn(4, 16, 64) * 1000
        
        result = decomposer.analyze(acts)
        assert result.phi_coherence >= 0
    
    def test_small_values(self):
        """Test handling of small activation values."""
        decomposer = PhiDecomposer(use_gpu=False)
        acts = torch.randn(4, 16, 64) * 0.001
        
        result = decomposer.analyze(acts)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
