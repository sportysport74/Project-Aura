"""
COSF-MI: φ-Shell Decomposition Analysis

Groups neural network activations into "Matryoshka shells" based on golden ratio scaling.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from torch import Tensor

from .constants import PHI, PHI_POWERS, MATRYOSHKA_SHELLS


@dataclass
class PhiShellResult:
    """Results from φ-shell decomposition analysis."""
    shell_indices: np.ndarray      # Shell assignment per element
    shell_counts: np.ndarray       # Count per shell
    shell_means: np.ndarray        # Mean magnitude per shell
    boundaries: np.ndarray         # Shell boundaries used
    dominant_shell: int            # Shell with most elements
    phi_coherence: float           # How well data fits φ-scaling [0,1]
    magnitudes: np.ndarray         # Original magnitudes


@dataclass 
class LayerPhiAnalysis:
    """φ-analysis results for a single layer."""
    layer_idx: int
    shell_result: PhiShellResult
    head_shells: Optional[np.ndarray] = None
    head_coherences: Optional[np.ndarray] = None


class PhiDecomposer:
    """
    Decomposes neural activations into φ-scaled Matryoshka shells.
    
    Example:
        >>> decomposer = PhiDecomposer()
        >>> activations = torch.randn(32, 768)
        >>> result = decomposer.analyze(activations)
        >>> print(f"Dominant shell: {result.dominant_shell}")
    """
    
    def __init__(
        self,
        n_shells: int = MATRYOSHKA_SHELLS,
        base_percentile: float = 10.0,
        use_gpu: bool = True,
    ):
        self.n_shells = n_shells
        self.base_percentile = base_percentile
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.phi_multipliers = np.array([PHI ** n for n in range(n_shells + 1)])
    
    def compute_magnitudes(self, activations: Tensor) -> np.ndarray:
        """Compute L2 magnitudes of activation vectors."""
        with torch.no_grad():
            if self.use_gpu and not activations.is_cuda:
                activations = activations.cuda()
            mags = torch.norm(activations.float(), dim=-1)
            return mags.cpu().numpy()
    
    def compute_shell_boundaries(self, magnitudes: np.ndarray) -> np.ndarray:
        """Compute shell boundaries based on data statistics."""
        base = np.percentile(magnitudes[magnitudes > 0], self.base_percentile)
        return base * self.phi_multipliers
    
    def assign_shells(self, magnitudes: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
        """Assign each magnitude to its corresponding φ-shell."""
        shell_indices = np.zeros_like(magnitudes, dtype=np.int32)
        
        for i in range(self.n_shells):
            mask = (magnitudes >= boundaries[i]) & (magnitudes < boundaries[i + 1])
            shell_indices[mask] = i
        
        shell_indices[magnitudes >= boundaries[-1]] = self.n_shells - 1
        shell_indices[magnitudes <= 0] = -1
        return shell_indices
    
    def compute_phi_coherence(
        self,
        magnitudes: np.ndarray,
        shell_indices: np.ndarray,
        boundaries: np.ndarray,
    ) -> float:
        """Compute how well the magnitude distribution follows φ-scaling."""
        if len(magnitudes) == 0:
            return 0.0
        
        shell_centers = np.sqrt(boundaries[:-1] * boundaries[1:])
        valid_mask = shell_indices >= 0
        
        if not np.any(valid_mask):
            return 0.0
        
        valid_mags = magnitudes[valid_mask]
        valid_shells = shell_indices[valid_mask]
        expected = shell_centers[valid_shells]
        deviations = np.abs(valid_mags - expected) / expected
        
        return float(np.exp(-np.mean(deviations)))
    
    def analyze(self, activations: Tensor) -> PhiShellResult:
        """Perform full φ-shell decomposition analysis."""
        original_shape = activations.shape[:-1]
        flat_acts = activations.reshape(-1, activations.shape[-1])
        
        magnitudes = self.compute_magnitudes(flat_acts)
        boundaries = self.compute_shell_boundaries(magnitudes)
        shell_indices = self.assign_shells(magnitudes, boundaries)
        
        shell_counts = np.array([np.sum(shell_indices == i) for i in range(self.n_shells)])
        shell_means = np.array([
            np.mean(magnitudes[shell_indices == i]) if np.any(shell_indices == i) else 0
            for i in range(self.n_shells)
        ])
        
        dominant_shell = int(np.argmax(shell_counts))
        phi_coherence = self.compute_phi_coherence(magnitudes, shell_indices, boundaries)
        
        return PhiShellResult(
            shell_indices=shell_indices.reshape(original_shape),
            shell_counts=shell_counts,
            shell_means=shell_means,
            boundaries=boundaries,
            dominant_shell=dominant_shell,
            phi_coherence=phi_coherence,
            magnitudes=magnitudes.reshape(original_shape),
        )
    
    def analyze_attention_heads(
        self,
        attention_output: Tensor,
        n_heads: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Analyze φ-shell distribution per attention head."""
        if attention_output.dim() == 3:
            batch, seq, hidden = attention_output.shape
            head_dim = hidden // n_heads
            attention_output = attention_output.reshape(batch, seq, n_heads, head_dim)
        
        n_heads_actual = attention_output.shape[2]
        head_shells = np.zeros(n_heads_actual, dtype=np.int32)
        head_coherences = np.zeros(n_heads_actual)
        
        for h in range(n_heads_actual):
            head_acts = attention_output[:, :, h, :]
            result = self.analyze(head_acts)
            head_shells[h] = result.dominant_shell
            head_coherences[h] = result.phi_coherence
        
        return head_shells, head_coherences


def quick_phi_analysis(activations: Tensor) -> dict:
    """Convenience function for quick φ-analysis."""
    decomposer = PhiDecomposer()
    result = decomposer.analyze(activations)
    return {
        "dominant_shell": result.dominant_shell,
        "phi_coherence": result.phi_coherence,
        "shell_distribution": result.shell_counts.tolist(),
        "shell_means": result.shell_means.tolist(),
    }


# GPU-accelerated version using CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class PhiDecomposerGPU(PhiDecomposer):
    """GPU-accelerated φ-decomposer using CuPy."""
    
    def compute_magnitudes(self, activations: Tensor) -> np.ndarray:
        if not CUPY_AVAILABLE:
            return super().compute_magnitudes(activations)
        
        with torch.no_grad():
            if activations.is_cuda:
                cp_array = cp.asarray(activations)
            else:
                cp_array = cp.asarray(activations.numpy())
            mags = cp.linalg.norm(cp_array.astype(cp.float32), axis=-1)
            return cp.asnumpy(mags)


if __name__ == "__main__":
    print("Testing φ-decomposer...")
    torch.manual_seed(42)
    
    base = torch.randn(32, 128, 768)
    for i in range(5):
        mask = torch.rand(32, 128, 1) > 0.8
        base = torch.where(mask, base * (PHI ** i), base)
    
    decomposer = PhiDecomposer()
    result = decomposer.analyze(base)
    
    print(f"\nResults:")
    print(f"  Dominant shell: {result.dominant_shell}")
    print(f"  φ-coherence: {result.phi_coherence:.4f}")
    print(f"  Shell counts: {result.shell_counts}")
