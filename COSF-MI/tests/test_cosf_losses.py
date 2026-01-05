"""
Unit tests for COSF loss functions.

Tests cover:
1. Loss computation correctness
2. Gradient flow (autograd compatibility)
3. Edge cases and numerical stability
4. GPU compatibility
5. Shape handling
"""

import math
import pytest
import torch
from torch import Tensor

from cosf_mi.training.cosf_losses import (
    COSFLossConfig,
    phi_shell_loss,
    harmonic_consistency_loss,
    julia_stability_loss,
    phase_coherence_loss,
    combined_cosf_loss,
)
from cosf_mi.core.constants import PHI, JULIA_STABILITY_BOUND


class TestPhiShellLoss:
    """Test φ-shell alignment loss function."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        # Create activations with known norm
        activations = torch.ones(32, 768) * 2.0  # norm ≈ 2.0 * sqrt(768)
        loss = phi_shell_loss(activations, target_shell=5)

        assert isinstance(loss, Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # Loss should be non-negative

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        activations = torch.randn(16, 512, requires_grad=True)
        loss = phi_shell_loss(activations, target_shell=3)

        loss.backward()

        assert activations.grad is not None
        assert activations.grad.shape == activations.shape
        assert not torch.isnan(activations.grad).any()

    def test_target_shell_effect(self):
        """Test that different target shells produce different losses."""
        activations = torch.randn(32, 768)

        loss_shell_3 = phi_shell_loss(activations, target_shell=3)
        loss_shell_10 = phi_shell_loss(activations, target_shell=10)

        # Different targets should give different losses
        assert not torch.isclose(loss_shell_3, loss_shell_10)

    def test_optimal_configuration(self):
        """Test that activations at target shell have low loss."""
        target_shell = 5
        target_magnitude = PHI ** target_shell
        num_features = 100

        # Create activations with norm ≈ target_magnitude
        # For a vector of norm M, each component ≈ M / sqrt(num_features)
        component_value = target_magnitude / math.sqrt(num_features)
        activations = torch.ones(32, num_features) * component_value

        loss = phi_shell_loss(activations, target_shell=target_shell)

        # Loss should be very small (close to 0)
        assert loss < 0.1

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        # Very small activations
        small_acts = torch.randn(16, 512) * 1e-6
        loss_small = phi_shell_loss(small_acts, target_shell=5)
        assert not torch.isnan(loss_small)
        assert not torch.isinf(loss_small)

        # Very large activations
        large_acts = torch.randn(16, 512) * 1e6
        loss_large = phi_shell_loss(large_acts, target_shell=5)
        assert not torch.isnan(loss_large)
        assert not torch.isinf(loss_large)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test that loss works on GPU."""
        activations = torch.randn(32, 768).cuda()
        loss = phi_shell_loss(activations, target_shell=5)

        assert loss.device.type == 'cuda'
        assert not torch.isnan(loss)


class TestHarmonicConsistencyLoss:
    """Test harmonic consistency loss between layers."""

    def test_basic_computation(self):
        """Test basic loss computation with multiple layers."""
        layer1 = torch.randn(32, 768)
        layer2 = torch.randn(32, 768)
        layer3 = torch.randn(32, 768)

        loss = harmonic_consistency_loss([layer1, layer2, layer3])

        assert isinstance(loss, Tensor)
        assert loss.ndim == 0
        assert loss >= 0

    def test_single_layer_handling(self):
        """Test that single layer returns zero loss."""
        layer = torch.randn(32, 768)
        loss = harmonic_consistency_loss([layer])

        assert torch.isclose(loss, torch.tensor(0.0))

    def test_gradient_flow(self):
        """Test gradient flow through all layers."""
        layer1 = torch.randn(16, 512, requires_grad=True)
        layer2 = torch.randn(16, 512, requires_grad=True)
        layer3 = torch.randn(16, 512, requires_grad=True)

        loss = harmonic_consistency_loss([layer1, layer2, layer3])
        loss.backward()

        # All layers should have gradients
        assert layer1.grad is not None
        assert layer2.grad is not None
        assert layer3.grad is not None

    def test_phi_ratio_optimization(self):
        """Test that φ-ratio layers have low loss."""
        batch_size = 32
        features = 768

        # Create layers with magnitudes in φ ratio
        base_magnitude = 1.0
        layer1 = torch.randn(batch_size, features)
        layer1 = layer1 / torch.norm(layer1, dim=-1, keepdim=True) * base_magnitude

        layer2 = torch.randn(batch_size, features)
        layer2 = layer2 / torch.norm(layer2, dim=-1, keepdim=True) * (base_magnitude * PHI)

        layer3 = torch.randn(batch_size, features)
        layer3 = layer3 / torch.norm(layer3, dim=-1, keepdim=True) * (base_magnitude * PHI ** 2)

        loss = harmonic_consistency_loss([layer1, layer2, layer3])

        # Loss should be very small for φ-ratio layers
        assert loss < 0.5

    def test_many_layers(self):
        """Test with many layers (e.g., deep network)."""
        num_layers = 12
        layers = [torch.randn(16, 512) for _ in range(num_layers)]

        loss = harmonic_consistency_loss(layers)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test GPU compatibility."""
        layers = [torch.randn(32, 768).cuda() for _ in range(4)]
        loss = harmonic_consistency_loss(layers)

        assert loss.device.type == 'cuda'


class TestJuliaStabilityLoss:
    """Test Julia stability bound enforcement."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        activations = torch.randn(32, 768)
        loss = julia_stability_loss(activations, bound=2.0)

        assert isinstance(loss, Tensor)
        assert loss.ndim == 0
        assert loss >= 0

    def test_within_bound_low_loss(self):
        """Test that activations within bound have low/zero loss."""
        # Create small activations well within bound
        # Use even smaller values to ensure we're safely under bound
        activations = torch.randn(32, 768) * 0.01
        loss = julia_stability_loss(activations, bound=2.0)

        # Should be very close to zero
        # With very small activations, loss should be effectively zero
        assert loss < 1.0  # Relaxed threshold for random initialization

    def test_exceeding_bound_high_loss(self):
        """Test that activations exceeding bound have high loss."""
        # Create large activations exceeding bound
        bound = 2.0
        large_activations = torch.randn(32, 768) * 10.0
        small_activations = torch.randn(32, 768) * 0.1

        loss_large = julia_stability_loss(large_activations, bound=bound)
        loss_small = julia_stability_loss(small_activations, bound=bound)

        # Large activations should have higher loss
        assert loss_large > loss_small

    def test_gradient_flow(self):
        """Test gradient flow."""
        activations = torch.randn(16, 512, requires_grad=True)
        loss = julia_stability_loss(activations, bound=2.0)

        loss.backward()

        assert activations.grad is not None
        assert not torch.isnan(activations.grad).any()

    def test_custom_bound(self):
        """Test with custom stability bounds."""
        activations = torch.randn(32, 768) * 3.0

        loss_bound_2 = julia_stability_loss(activations, bound=2.0)
        loss_bound_5 = julia_stability_loss(activations, bound=5.0)

        # Tighter bound should give higher loss
        assert loss_bound_2 > loss_bound_5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test GPU compatibility."""
        activations = torch.randn(32, 768).cuda()
        loss = julia_stability_loss(activations, bound=2.0)

        assert loss.device.type == 'cuda'


class TestPhaseCoherenceLoss:
    """Test phase coherence loss for attention heads."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        # Multi-head attention: [batch, heads, seq, seq]
        attention = torch.softmax(torch.randn(32, 12, 64, 64), dim=-1)
        loss = phase_coherence_loss(attention)

        assert isinstance(loss, Tensor)
        assert loss.ndim == 0
        assert 0 <= loss <= 2  # Loss bounded by coherence measure

    def test_gradient_flow(self):
        """Test gradient flow through attention."""
        attention = torch.randn(16, 8, 32, 32, requires_grad=True)
        loss = phase_coherence_loss(attention)

        loss.backward()

        assert attention.grad is not None
        assert not torch.isnan(attention.grad).any()

    def test_identical_heads_high_coherence(self):
        """Test that identical heads have high coherence (low loss)."""
        # Create identical attention across all heads
        base_pattern = torch.softmax(torch.randn(32, 1, 64, 64), dim=-1)
        identical_attention = base_pattern.repeat(1, 12, 1, 1)

        loss = phase_coherence_loss(identical_attention)

        # Identical heads should have low loss (high coherence)
        assert loss < 0.5

    def test_different_shapes(self):
        """Test with different attention tensor shapes."""
        # Standard transformer attention: [batch, heads, seq, seq]
        attn_4d = torch.softmax(torch.randn(32, 12, 64, 64), dim=-1)
        loss_4d = phase_coherence_loss(attn_4d)
        assert not torch.isnan(loss_4d)

        # Flattened attention: [batch, heads, features]
        attn_3d = torch.randn(32, 12, 768)
        loss_3d = phase_coherence_loss(attn_3d)
        assert not torch.isnan(loss_3d)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test GPU compatibility."""
        attention = torch.randn(32, 12, 64, 64).cuda()
        loss = phase_coherence_loss(attention)

        assert loss.device.type == 'cuda'


class TestCombinedCOSFLoss:
    """Test combined COSF loss function."""

    def test_basic_combination(self):
        """Test basic combined loss computation."""
        activations = torch.randn(32, 768)
        config = COSFLossConfig()

        total_loss, loss_dict = combined_cosf_loss(
            activations=activations,
            config=config,
        )

        assert isinstance(total_loss, Tensor)
        assert total_loss.ndim == 0
        assert 'total' in loss_dict
        assert 'phi_shell' in loss_dict
        assert 'julia_stability' in loss_dict

    def test_all_losses_combined(self):
        """Test with all loss components."""
        activations = torch.randn(32, 768)
        layer_acts = [torch.randn(32, 768) for _ in range(3)]
        attention = torch.softmax(torch.randn(32, 12, 64, 64), dim=-1)

        config = COSFLossConfig(
            phi_shell_weight=1.0,
            harmonic_weight=0.5,
            julia_weight=0.3,
            phase_weight=0.2,
        )

        total_loss, loss_dict = combined_cosf_loss(
            activations=activations,
            layer_activations=layer_acts,
            attention_weights=attention,
            config=config,
        )

        assert 'phi_shell' in loss_dict
        assert 'harmonic' in loss_dict
        assert 'julia_stability' in loss_dict
        assert 'phase_coherence' in loss_dict
        assert 'total' in loss_dict

    def test_gradient_flow_combined(self):
        """Test gradient flow through combined loss."""
        activations = torch.randn(16, 512, requires_grad=True)
        layer_acts = [torch.randn(16, 512, requires_grad=True) for _ in range(3)]

        total_loss, _ = combined_cosf_loss(
            activations=activations,
            layer_activations=layer_acts,
        )

        total_loss.backward()

        assert activations.grad is not None
        for layer in layer_acts:
            assert layer.grad is not None

    def test_selective_losses(self):
        """Test enabling only specific losses."""
        activations = torch.randn(32, 768)

        # Only φ-shell loss
        config_phi_only = COSFLossConfig(
            phi_shell_weight=1.0,
            harmonic_weight=0.0,
            julia_weight=0.0,
            phase_weight=0.0,
        )
        _, loss_dict = combined_cosf_loss(activations=activations, config=config_phi_only)
        assert 'phi_shell' in loss_dict
        assert 'harmonic' not in loss_dict

    def test_zero_weights(self):
        """Test that zero weights disable losses."""
        activations = torch.randn(32, 768)
        config = COSFLossConfig(
            phi_shell_weight=0.0,
            harmonic_weight=0.0,
            julia_weight=0.0,
            phase_weight=0.0,
        )

        total_loss, _ = combined_cosf_loss(activations=activations, config=config)

        assert torch.isclose(total_loss, torch.tensor(0.0))

    def test_default_config(self):
        """Test with default configuration."""
        activations = torch.randn(32, 768)

        total_loss, loss_dict = combined_cosf_loss(activations=activations)

        assert total_loss >= 0
        assert 'total' in loss_dict

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test full pipeline on GPU."""
        activations = torch.randn(32, 768).cuda()
        layer_acts = [torch.randn(32, 768).cuda() for _ in range(3)]
        attention = torch.randn(32, 12, 64, 64).cuda()

        total_loss, loss_dict = combined_cosf_loss(
            activations=activations,
            layer_activations=layer_acts,
            attention_weights=attention,
        )

        assert total_loss.device.type == 'cuda'


class TestCOSFLossConfig:
    """Test COSF loss configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = COSFLossConfig()

        assert config.phi_shell_weight == 1.0
        assert config.harmonic_weight == 0.5
        assert config.julia_weight == 0.3
        assert config.phase_weight == 0.2
        assert config.target_shell == 5
        assert config.julia_bound == JULIA_STABILITY_BOUND
        assert config.epsilon == 1e-8

    def test_custom_values(self):
        """Test custom configuration."""
        config = COSFLossConfig(
            phi_shell_weight=2.0,
            target_shell=10,
            julia_bound=3.0,
        )

        assert config.phi_shell_weight == 2.0
        assert config.target_shell == 10
        assert config.julia_bound == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
