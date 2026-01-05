"""
Julia Stability Governor: Real-time stability monitoring and correction

Uses Julia stability bounds to ensure reliable inference.
"""

from typing import Dict, List, Optional, Any
from collections import deque
import torch
import numpy as np

from ..core.constants import JULIA_STABILITY_BOUND, PHI


class JuliaGovernor:
    """Real-time stability monitoring and auto-correction.

    Monitors model activations during inference and applies
    φ-decomposition corrections when instability is detected.
    """

    def __init__(
        self,
        model,
        bounds: float = JULIA_STABILITY_BOUND,
        history_size: int = 100,
    ):
        """Initialize governor.

        Args:
            model: COSF transformer model
            bounds: Julia stability bound (default: 2.0)
            history_size: Size of stability history buffer
        """
        self.model = model
        self.bounds = bounds
        self.history = deque(maxlen=history_size)

        self.phi = PHI

        # Correction parameters
        self.temperature_reduction = 0.8
        self.max_corrections = 5
        self.correction_count = 0

    def monitor_inference(
        self,
        activations: torch.Tensor,
        step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Check activations during generation.

        Args:
            activations: Model activation tensor
            step: Optional generation step number

        Returns:
            Stability status dictionary
        """
        # Compute norms
        norms = torch.norm(activations, dim=-1)
        exceed_pct = (norms > self.bounds).float().mean().item()
        max_norm = norms.max().item()
        mean_norm = norms.mean().item()

        # Determine status
        if exceed_pct < 0.2:
            status = 'stable'
        elif exceed_pct < 0.5:
            status = 'warning'
        else:
            status = 'unstable'

        stability = {
            'exceed_pct': exceed_pct,
            'max_norm': max_norm,
            'mean_norm': mean_norm,
            'bounds': self.bounds,
            'status': status,
            'step': step,
        }

        # Add to history
        self.history.append(stability)

        # Apply correction if unstable
        if status == 'unstable':
            self._apply_correction(activations)

        return stability

    def _apply_correction(self, activations: torch.Tensor):
        """Apply φ-decomposition correction on instability.

        Args:
            activations: Activation tensor to correct
        """
        if self.correction_count >= self.max_corrections:
            print("[WARNING] Max corrections reached - may need model reload")
            return

        print(f"[WARNING] Julia bounds exceeded - applying phi-correction (#{self.correction_count + 1})")

        # Method 1: Reduce temperature
        if hasattr(self.model, 'temperature'):
            self.model.temperature *= self.temperature_reduction
            print(f"  Reduced temperature to {self.model.temperature:.3f}")

        # Method 2: Apply φ-normalization to activations
        # Decompose into harmonic components and re-normalize
        with torch.no_grad():
            # Get φ-harmonic components
            phi_components = self._decompose_activation(activations)

            # Reconstruct with bounded components
            corrected = self._reconstruct_bounded(phi_components)

            # Update activations in-place (if possible)
            if hasattr(self.model, 'last_activations'):
                self.model.last_activations = corrected

        self.correction_count += 1

    def _decompose_activation(
        self,
        activation: torch.Tensor
    ) -> List[torch.Tensor]:
        """Decompose activation into φ-harmonic components.

        Args:
            activation: Activation tensor

        Returns:
            List of harmonic component tensors
        """
        components = []
        residual = activation.clone()

        # Decompose into first 5 harmonics
        for i in range(5):
            # Create φ-scaled basis
            scale = self.phi ** i

            # Project onto basis
            projection = residual / scale

            # Clamp to bounds
            projection = torch.clamp(projection, -self.bounds, self.bounds)

            components.append(projection)

            # Update residual
            residual = residual - projection * scale

        return components

    def _reconstruct_bounded(
        self,
        components: List[torch.Tensor]
    ) -> torch.Tensor:
        """Reconstruct activation from bounded components.

        Args:
            components: List of harmonic components

        Returns:
            Reconstructed activation tensor
        """
        result = torch.zeros_like(components[0])

        for i, component in enumerate(components):
            scale = self.phi ** i
            result += component * scale

        # Final clamping
        result = torch.clamp(result, -self.bounds * self.phi, self.bounds * self.phi)

        return result

    def predict_instability(
        self,
        window: int = 10,
        threshold: float = 0.05
    ) -> bool:
        """Predict future instability from trends.

        Args:
            window: Number of recent steps to analyze
            threshold: Trend threshold for prediction

        Returns:
            True if instability predicted
        """
        if len(self.history) < window:
            return False

        # Get recent history
        recent = list(self.history)[-window:]
        exceed_pcts = [h['exceed_pct'] for h in recent]

        # Fit linear trend
        x = np.arange(len(exceed_pcts))
        coeffs = np.polyfit(x, exceed_pcts, deg=1)
        trend = coeffs[0]  # Slope

        # Positive trend = increasing instability
        is_unstable = trend > threshold

        if is_unstable:
            print(f"[ALERT] Instability trend detected: {trend:.4f} > {threshold}")

        return is_unstable

    def get_stability_metrics(self) -> Dict[str, Any]:
        """Get aggregate stability metrics.

        Returns:
            Dictionary of metrics
        """
        if not self.history:
            return {
                'status': 'no_data',
                'sample_count': 0,
            }

        recent = list(self.history)

        return {
            'status': recent[-1]['status'] if recent else 'unknown',
            'sample_count': len(recent),
            'avg_exceed_pct': np.mean([h['exceed_pct'] for h in recent]),
            'max_exceed_pct': max(h['exceed_pct'] for h in recent),
            'avg_max_norm': np.mean([h['max_norm'] for h in recent]),
            'corrections_applied': self.correction_count,
            'is_trending_unstable': self.predict_instability(),
        }

    def reset_corrections(self):
        """Reset correction counter."""
        self.correction_count = 0
        if hasattr(self.model, 'temperature'):
            # Reset temperature to default
            self.model.temperature = 1.0

    def set_bounds(self, new_bounds: float):
        """Update stability bounds.

        Args:
            new_bounds: New Julia stability bound
        """
        old_bounds = self.bounds
        self.bounds = new_bounds
        print(f"Updated Julia bounds: {old_bounds:.2f} → {new_bounds:.2f}")

    def export_history(self, path: str):
        """Export stability history to JSON.

        Args:
            path: Output file path
        """
        import json

        history_list = list(self.history)

        # Convert tensors to floats
        for entry in history_list:
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    entry[key] = value.item()

        with open(path, 'w') as f:
            json.dump(history_list, f, indent=2)

        print(f"Exported {len(history_list)} stability records to {path}")
