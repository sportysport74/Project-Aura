"""
COSF-MI: Harmonic Decomposition Analysis

Applies Fourier analysis to neural activations to identify
harmonic patterns corresponding to COSF-related frequencies.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import torch
from torch import Tensor
from scipy import signal
from scipy.fft import fft, fftfreq

from .constants import PHI, COSF


@dataclass
class HarmonicPeak:
    """A detected harmonic peak in the frequency spectrum."""
    frequency: float
    magnitude: float
    phase: float
    harmonic_ratio: float
    is_phi_harmonic: bool
    phi_n: Optional[int]


@dataclass
class HarmonicAnalysisResult:
    """Results from harmonic decomposition analysis."""
    peaks: List[HarmonicPeak]
    frequencies: np.ndarray
    magnitudes: np.ndarray
    phases: np.ndarray
    fundamental: Optional[float]
    phi_harmonic_score: float
    detected_cosf_ratio: Optional[float]


class HarmonicAnalyzer:
    """
    Analyzes neural activations for harmonic patterns using FFT.
    
    Example:
        >>> analyzer = HarmonicAnalyzer()
        >>> activations = torch.randn(1, 512, 768)
        >>> result = analyzer.analyze_sequence(activations)
        >>> print(f"φ-harmonic score: {result.phi_harmonic_score}")
    """
    
    def __init__(
        self,
        min_peak_prominence: float = 0.1,
        phi_tolerance: float = 0.05,
        n_top_peaks: int = 10,
    ):
        self.min_peak_prominence = min_peak_prominence
        self.phi_tolerance = phi_tolerance
        self.n_top_peaks = n_top_peaks
        self.phi_powers = np.array([PHI ** n for n in range(18)])
    
    def compute_spectrum(
        self,
        signal_1d: np.ndarray,
        sampling_rate: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute FFT of a 1D signal."""
        n = len(signal_1d)
        window = signal.windows.hann(n)
        windowed = signal_1d * window
        spectrum = fft(windowed)
        
        freqs = fftfreq(n, 1/sampling_rate)[:n//2]
        mags = np.abs(spectrum[:n//2]) * 2 / n
        phases = np.angle(spectrum[:n//2])
        
        return freqs, mags, phases
    
    def find_peaks(
        self,
        frequencies: np.ndarray,
        magnitudes: np.ndarray,
        phases: np.ndarray,
    ) -> List[HarmonicPeak]:
        """Find significant peaks in the frequency spectrum."""
        peak_indices, _ = signal.find_peaks(
            magnitudes,
            prominence=self.min_peak_prominence * np.max(magnitudes),
        )
        
        if len(peak_indices) == 0:
            return []
        
        sorted_indices = np.argsort(magnitudes[peak_indices])[::-1]
        top_indices = peak_indices[sorted_indices[:self.n_top_peaks]]
        
        # Find fundamental
        freq_sorted = np.sort(frequencies[top_indices])
        fundamental = freq_sorted[freq_sorted > 0][0] if np.any(freq_sorted > 0) else None
        
        peaks = []
        for idx in top_indices:
            freq = frequencies[idx]
            mag = magnitudes[idx]
            phase = phases[idx]
            
            harmonic_ratio = freq / fundamental if fundamental and freq > 0 else 1.0
            
            # Check φ^n alignment
            is_phi_harmonic = False
            phi_n = None
            for n, phi_power in enumerate(self.phi_powers):
                if abs(harmonic_ratio - phi_power) / phi_power < self.phi_tolerance:
                    is_phi_harmonic = True
                    phi_n = n
                    break
            
            peaks.append(HarmonicPeak(
                frequency=float(freq),
                magnitude=float(mag),
                phase=float(phase),
                harmonic_ratio=float(harmonic_ratio),
                is_phi_harmonic=is_phi_harmonic,
                phi_n=phi_n,
            ))
        
        return peaks
    
    def compute_phi_harmonic_score(self, peaks: List[HarmonicPeak]) -> float:
        """Compute alignment score with φ^n harmonic series."""
        if not peaks:
            return 0.0
        return sum(1 for p in peaks if p.is_phi_harmonic) / len(peaks)
    
    def compute_cosf_ratio(self, peaks: List[HarmonicPeak]) -> Optional[float]:
        """Check for COSF-like ratio between peaks."""
        if len(peaks) < 2:
            return None
        
        sorted_peaks = sorted(peaks, key=lambda p: p.frequency)
        
        for i, p1 in enumerate(sorted_peaks[:-1]):
            if p1.frequency <= 0:
                continue
            for p2 in sorted_peaks[i+1:]:
                ratio = p2.frequency / p1.frequency
                if abs(ratio - COSF) / COSF < 0.1:
                    return ratio
                if abs(ratio - self.phi_powers[17]) / self.phi_powers[17] < 0.1:
                    return ratio
        return None
    
    def analyze_sequence(self, activations: Tensor, dim: int = 1) -> HarmonicAnalysisResult:
        """Analyze harmonic patterns along the sequence dimension."""
        with torch.no_grad():
            if activations.dim() == 3:
                signal_1d = activations.float().mean(dim=(0, 2)).cpu().numpy()
            elif activations.dim() == 2:
                signal_1d = activations.float().mean(dim=1).cpu().numpy()
            else:
                signal_1d = activations.float().cpu().numpy().flatten()
        
        freqs, mags, phases = self.compute_spectrum(signal_1d)
        peaks = self.find_peaks(freqs, mags, phases)
        
        return HarmonicAnalysisResult(
            peaks=peaks,
            frequencies=freqs,
            magnitudes=mags,
            phases=phases,
            fundamental=min((p.frequency for p in peaks if p.frequency > 0), default=None),
            phi_harmonic_score=self.compute_phi_harmonic_score(peaks),
            detected_cosf_ratio=self.compute_cosf_ratio(peaks),
        )
    
    def analyze_layer_progression(self, layer_activations: List[Tensor]) -> HarmonicAnalysisResult:
        """Analyze harmonic patterns across layers."""
        layer_mags = []
        for acts in layer_activations:
            with torch.no_grad():
                mag = torch.norm(acts.float(), dim=-1).mean().item()
                layer_mags.append(mag)
        
        signal_1d = np.array(layer_mags)
        freqs, mags, phases = self.compute_spectrum(signal_1d)
        peaks = self.find_peaks(freqs, mags, phases)
        
        return HarmonicAnalysisResult(
            peaks=peaks,
            frequencies=freqs,
            magnitudes=mags,
            phases=phases,
            fundamental=min((p.frequency for p in peaks if p.frequency > 0), default=None),
            phi_harmonic_score=self.compute_phi_harmonic_score(peaks),
            detected_cosf_ratio=self.compute_cosf_ratio(peaks),
        )


class MultiScaleHarmonicAnalyzer:
    """Performs harmonic analysis at multiple time scales."""
    
    def __init__(self, scales: List[int] = [4, 8, 16, 32, 64, 128, 256, 512]):
        self.scales = scales
        self.analyzers = {s: HarmonicAnalyzer() for s in scales}
    
    def analyze(self, activations: Tensor) -> dict:
        """Analyze activations at multiple scales."""
        results = {}
        seq_len = activations.shape[1]
        
        for scale in self.scales:
            if scale > seq_len:
                continue
            
            all_peaks = []
            phi_scores = []
            
            for start in range(0, seq_len - scale + 1, scale // 2):
                window = activations[:, start:start+scale, :]
                result = self.analyzers[scale].analyze_sequence(window)
                all_peaks.extend(result.peaks)
                phi_scores.append(result.phi_harmonic_score)
            
            results[scale] = {
                "n_peaks": len(all_peaks),
                "phi_harmonic_peaks": sum(1 for p in all_peaks if p.is_phi_harmonic),
                "mean_phi_score": np.mean(phi_scores) if phi_scores else 0.0,
            }
        
        return results


def quick_harmonic_analysis(activations: Tensor) -> dict:
    """Convenience function for quick harmonic analysis."""
    analyzer = HarmonicAnalyzer()
    result = analyzer.analyze_sequence(activations)
    return {
        "n_peaks": len(result.peaks),
        "fundamental": result.fundamental,
        "phi_harmonic_score": result.phi_harmonic_score,
        "detected_cosf_ratio": result.detected_cosf_ratio,
        "top_peaks": [
            {"freq": p.frequency, "mag": p.magnitude, "phi_n": p.phi_n}
            for p in result.peaks[:5]
        ],
    }


if __name__ == "__main__":
    print("Testing Harmonic Analyzer...")
    torch.manual_seed(42)
    
    seq_len, hidden = 512, 768
    t = torch.linspace(0, 1, seq_len)
    
    signal = torch.zeros(seq_len, hidden)
    for h in range(hidden):
        freq1, freq2, freq3 = 10, 10 * PHI, 10 * PHI ** 2
        signal[:, h] = (
            torch.sin(2 * np.pi * freq1 * t) +
            0.5 * torch.sin(2 * np.pi * freq2 * t) +
            0.25 * torch.sin(2 * np.pi * freq3 * t) +
            0.1 * torch.randn(seq_len)
        )
    
    result = HarmonicAnalyzer().analyze_sequence(signal.unsqueeze(0))
    
    print(f"\nResults:")
    print(f"  Fundamental: {result.fundamental}")
    print(f"  φ-harmonic score: {result.phi_harmonic_score:.4f}")
    print(f"  Top peaks: {[(p.frequency, p.phi_n) for p in result.peaks[:5]]}")
