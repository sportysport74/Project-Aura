"""
COSF-MI: Julia Set Stability Analysis

Implements stability analysis based on Julia set dynamics.
The iteration z_{n+1} = z_n² + c provides a stability bound:
if |z_n| > 2, the sequence diverges.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum
import numpy as np
import torch
from torch import Tensor

from .constants import JULIA_STABILITY_BOUND


class StabilityClass(Enum):
    """Classification of stability based on Julia analysis."""
    STABLE = "stable"
    MARGINAL = "marginal"
    UNSTABLE = "unstable"
    CHAOTIC = "chaotic"


@dataclass
class JuliaStabilityResult:
    """Results from Julia stability analysis."""
    stability_map: np.ndarray
    escape_iterations: np.ndarray
    max_magnitudes: np.ndarray
    stability_score: float
    class_distribution: dict
    unstable_positions: List[Tuple[int, ...]]
    julia_c: Optional[complex]


@dataclass
class LayerDynamicsResult:
    """Results from layer-to-layer dynamics analysis."""
    layer_magnitudes: np.ndarray
    layer_changes: np.ndarray
    stability_per_layer: np.ndarray
    unstable_layers: List[int]
    trajectory_stable: bool
    lyapunov_estimate: float


class JuliaStabilityAnalyzer:
    """
    Analyzes activation stability using Julia set dynamics.
    
    Example:
        >>> analyzer = JuliaStabilityAnalyzer()
        >>> activations = torch.randn(32, 128, 768)
        >>> result = analyzer.analyze(activations)
        >>> print(f"Stability score: {result.stability_score:.2%}")
    """
    
    def __init__(
        self,
        bound: float = JULIA_STABILITY_BOUND,
        max_iterations: int = 100,
        marginal_threshold: float = 1.8,
        default_c: complex = complex(-0.7, 0.27015),
    ):
        self.bound = bound
        self.max_iterations = max_iterations
        self.marginal_threshold = marginal_threshold
        self.default_c = default_c
    
    def activations_to_complex(self, activations: Tensor) -> np.ndarray:
        """Convert activations to complex numbers for Julia analysis."""
        with torch.no_grad():
            acts = activations.float().cpu().numpy()
        
        original_shape = acts.shape
        hidden = original_shape[-1]
        
        if hidden % 2 == 0:
            reshaped = acts.reshape(*original_shape[:-1], hidden // 2, 2)
            return reshaped[..., 0] + 1j * reshaped[..., 1]
        else:
            truncated = acts[..., :-1]
            reshaped = truncated.reshape(*original_shape[:-1], (hidden-1) // 2, 2)
            return reshaped[..., 0] + 1j * reshaped[..., 1]
    
    def julia_iterate_batch(
        self,
        z_init: np.ndarray,
        c: complex,
        max_iter: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform Julia iteration: z_{n+1} = z_n² + c"""
        if max_iter is None:
            max_iter = self.max_iterations
        
        z = z_init.copy()
        escape_iter = np.full(z.shape, -1, dtype=np.int32)
        max_mags = np.abs(z)
        escaped = np.zeros(z.shape, dtype=bool)
        
        for i in range(max_iter):
            z[~escaped] = z[~escaped] ** 2 + c
            current_mags = np.abs(z)
            max_mags = np.maximum(max_mags, current_mags)
            
            newly_escaped = (current_mags > self.bound) & (~escaped)
            escape_iter[newly_escaped] = i
            escaped |= newly_escaped
            
            if escaped.all():
                break
        
        return escape_iter, max_mags, z
    
    def classify_stability(
        self,
        escape_iterations: np.ndarray,
        max_magnitudes: np.ndarray,
    ) -> np.ndarray:
        """Classify stability: 0=stable, 1=marginal, 2=unstable, 3=chaotic"""
        classes = np.zeros(escape_iterations.shape, dtype=np.int32)
        classes[escape_iterations >= 0] = 2  # UNSTABLE
        classes[(escape_iterations < 0) & (max_magnitudes > self.marginal_threshold)] = 1  # MARGINAL
        classes[(escape_iterations >= 0) & (escape_iterations < 10)] = 3  # CHAOTIC
        return classes
    
    def analyze(self, activations: Tensor, c: complex = None) -> JuliaStabilityResult:
        """Perform full Julia stability analysis."""
        if c is None:
            c = self.default_c
        
        z_init = self.activations_to_complex(activations)
        
        scale = np.mean(np.abs(z_init))
        z_normalized = z_init / scale if scale > 0 else z_init
        
        escape_iter, max_mags, _ = self.julia_iterate_batch(z_normalized, c)
        stability_classes = self.classify_stability(escape_iter, max_mags)
        
        total = stability_classes.size
        class_counts = {
            "stable": int(np.sum(stability_classes == 0)),
            "marginal": int(np.sum(stability_classes == 1)),
            "unstable": int(np.sum(stability_classes == 2)),
            "chaotic": int(np.sum(stability_classes == 3)),
        }
        
        stability_score = class_counts["stable"] / total
        unstable_mask = stability_classes >= 2
        unstable_positions = list(zip(*np.where(unstable_mask)))[:100]
        
        return JuliaStabilityResult(
            stability_map=stability_classes,
            escape_iterations=escape_iter,
            max_magnitudes=max_mags,
            stability_score=stability_score,
            class_distribution=class_counts,
            unstable_positions=unstable_positions,
            julia_c=c,
        )
    
    def analyze_direct(self, activations: Tensor) -> JuliaStabilityResult:
        """Fast stability check without Julia iteration."""
        with torch.no_grad():
            mags = torch.norm(activations.float(), dim=-1).cpu().numpy()
        
        stability_classes = np.zeros(mags.shape, dtype=np.int32)
        stability_classes[mags > self.marginal_threshold] = 1
        stability_classes[mags > self.bound] = 2
        stability_classes[mags > self.bound * 2] = 3
        
        total = stability_classes.size
        class_counts = {
            "stable": int(np.sum(stability_classes == 0)),
            "marginal": int(np.sum(stability_classes == 1)),
            "unstable": int(np.sum(stability_classes == 2)),
            "chaotic": int(np.sum(stability_classes == 3)),
        }
        
        return JuliaStabilityResult(
            stability_map=stability_classes,
            escape_iterations=np.full(mags.shape, -1),
            max_magnitudes=mags,
            stability_score=class_counts["stable"] / total,
            class_distribution=class_counts,
            unstable_positions=list(zip(*np.where(stability_classes >= 2)))[:100],
            julia_c=None,
        )


class LayerDynamicsAnalyzer:
    """Analyzes layer-to-layer dynamics for stability."""
    
    def __init__(self, divergence_threshold: float = 2.0):
        self.divergence_threshold = divergence_threshold
    
    def analyze(self, layer_activations: List[Tensor]) -> LayerDynamicsResult:
        """Analyze dynamics across layers."""
        n_layers = len(layer_activations)
        layer_mags = np.zeros(n_layers)
        
        for i, acts in enumerate(layer_activations):
            with torch.no_grad():
                layer_mags[i] = torch.norm(acts.float(), dim=-1).mean().item()
        
        layer_changes = np.diff(layer_mags)
        
        stability_per_layer = np.ones(n_layers, dtype=bool)
        for i in range(1, n_layers):
            ratio = layer_mags[i] / (layer_mags[i-1] + 1e-10)
            if ratio > self.divergence_threshold:
                stability_per_layer[i] = False
        
        unstable_layers = list(np.where(~stability_per_layer)[0])
        
        if n_layers > 1:
            log_ratios = np.log(layer_mags[1:] / (layer_mags[:-1] + 1e-10) + 1e-10)
            lyapunov = np.mean(log_ratios)
        else:
            lyapunov = 0.0
        
        return LayerDynamicsResult(
            layer_magnitudes=layer_mags,
            layer_changes=layer_changes,
            stability_per_layer=stability_per_layer,
            unstable_layers=unstable_layers,
            trajectory_stable=len(unstable_layers) == 0,
            lyapunov_estimate=float(lyapunov),
        )


def quick_stability_check(activations: Tensor) -> dict:
    """Quick stability check for activations."""
    result = JuliaStabilityAnalyzer().analyze_direct(activations)
    return {
        "stability_score": result.stability_score,
        "class_distribution": result.class_distribution,
        "n_unstable": len(result.unstable_positions),
        "max_magnitude": float(result.max_magnitudes.max()),
    }


# GPU-accelerated Julia iteration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def julia_iterate_gpu(
    z_init: np.ndarray,
    c: complex,
    bound: float = 2.0,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated Julia iteration using CuPy."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    z = cp.asarray(z_init)
    c_gpu = cp.complex128(c)
    
    escape_iter = cp.full(z.shape, -1, dtype=cp.int32)
    max_mags = cp.abs(z)
    escaped = cp.zeros(z.shape, dtype=cp.bool_)
    
    for i in range(max_iter):
        mask = ~escaped
        z[mask] = z[mask] ** 2 + c_gpu
        
        current_mags = cp.abs(z)
        max_mags = cp.maximum(max_mags, current_mags)
        
        newly_escaped = (current_mags > bound) & (~escaped)
        escape_iter[newly_escaped] = i
        escaped |= newly_escaped
        
        if escaped.all():
            break
    
    return cp.asnumpy(escape_iter), cp.asnumpy(max_mags)


if __name__ == "__main__":
    print("Testing Julia Stability Analyzer...")
    torch.manual_seed(42)
    
    batch, seq, hidden = 8, 64, 256
    stable_acts = torch.randn(batch, seq, hidden) * 0.5
    spike_mask = torch.rand(batch, seq, 1) > 0.95
    stable_acts = torch.where(spike_mask, stable_acts * 10, stable_acts)
    
    analyzer = JuliaStabilityAnalyzer()
    
    print("\nDirect analysis:")
    result = analyzer.analyze_direct(stable_acts)
    print(f"  Stability score: {result.stability_score:.2%}")
    print(f"  Distribution: {result.class_distribution}")
    
    print("\nFull Julia analysis:")
    result = analyzer.analyze(stable_acts)
    print(f"  Stability score: {result.stability_score:.2%}")
    print(f"  Julia c: {result.julia_c}")
