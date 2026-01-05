"""
COSF-MI: Circuit Extraction and Analysis

Extracts and analyzes neural circuits using COSF principles.
Identifies attention head groupings, φ-coherent pathways, and
stability boundaries within transformer architectures.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import torch
from torch import Tensor

from ..core import (
    PhiDecomposer,
    HarmonicAnalyzer,
    JuliaStabilityAnalyzer,
    PHI,
    COSF,
    MATRYOSHKA_SHELLS,
)
from ..hooks import ModelHooks, ActivationCache


@dataclass
class CircuitNode:
    """A node in a neural circuit."""
    layer_idx: int
    head_idx: Optional[int]  # None for MLP/residual
    node_type: str  # "attention", "mlp", "residual", "embed", "unembed"
    phi_shell: int
    stability_class: str
    coherence_score: float


@dataclass
class CircuitEdge:
    """An edge connecting circuit nodes."""
    source: CircuitNode
    target: CircuitNode
    weight: float
    phase_coherent: bool


@dataclass
class Circuit:
    """A complete circuit extracted from the model."""
    nodes: List[CircuitNode]
    edges: List[CircuitEdge]
    dominant_shell: int
    mean_coherence: float
    stability_score: float
    description: str


@dataclass
class COSFAnalysisResult:
    """Complete COSF analysis result for a model."""
    phi_analysis: Dict[str, any]
    harmonic_analysis: Dict[str, any]
    stability_analysis: Dict[str, any]
    circuits: List[Circuit]
    layer_summaries: List[Dict]
    global_metrics: Dict[str, float]


class CircuitExtractor:
    """
    Extracts neural circuits using COSF analysis principles.
    
    Identifies:
    1. φ-coherent attention head groupings
    2. Harmonic resonance patterns across layers
    3. Stability boundaries (Julia set analysis)
    4. Phase-locked computational pathways
    
    Example:
        >>> extractor = CircuitExtractor(model, tokenizer)
        >>> result = extractor.analyze("The quick brown fox")
        >>> print(f"Found {len(result.circuits)} circuits")
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Optional[any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        self.hooks = ModelHooks(model)
        self.phi_decomposer = PhiDecomposer()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.stability_analyzer = JuliaStabilityAnalyzer()
        
        # Cache for analysis results
        self._activation_cache: Optional[ActivationCache] = None
    
    def _get_model_config(self) -> Dict:
        """Extract model configuration."""
        config = {}
        
        # Try to get config from different model types
        if hasattr(self.model, 'config'):
            cfg = self.model.config
            config['n_layers'] = getattr(cfg, 'n_layer', getattr(cfg, 'num_hidden_layers', 12))
            config['n_heads'] = getattr(cfg, 'n_head', getattr(cfg, 'num_attention_heads', 12))
            config['d_model'] = getattr(cfg, 'n_embd', getattr(cfg, 'hidden_size', 768))
        else:
            # Default values
            config['n_layers'] = 12
            config['n_heads'] = 12
            config['d_model'] = 768
        
        return config
    
    def _tokenize(self, text: str) -> Dict[str, Tensor]:
        """Tokenize input text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text input")
        
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in tokens.items()}
    
    def analyze_layer(
        self,
        layer_activations: Tensor,
        layer_idx: int,
        n_heads: int,
    ) -> Dict:
        """Analyze a single layer's activations."""
        # φ-decomposition
        phi_result = self.phi_decomposer.analyze(layer_activations)
        
        # Harmonic analysis
        harmonic_result = self.harmonic_analyzer.analyze_sequence(layer_activations)
        
        # Stability analysis
        stability_result = self.stability_analyzer.analyze_direct(layer_activations)
        
        # Per-head analysis if attention layer
        head_analysis = None
        if layer_activations.shape[-1] % n_heads == 0:
            head_shells, head_coherences = self.phi_decomposer.analyze_attention_heads(
                layer_activations, n_heads
            )
            head_analysis = {
                "shells": head_shells.tolist(),
                "coherences": head_coherences.tolist(),
            }
        
        return {
            "layer_idx": layer_idx,
            "phi": {
                "dominant_shell": phi_result.dominant_shell,
                "coherence": phi_result.phi_coherence,
                "shell_distribution": phi_result.shell_counts.tolist(),
            },
            "harmonic": {
                "n_peaks": len(harmonic_result.peaks),
                "phi_score": harmonic_result.phi_harmonic_score,
                "cosf_ratio": harmonic_result.detected_cosf_ratio,
            },
            "stability": {
                "score": stability_result.stability_score,
                "distribution": stability_result.class_distribution,
            },
            "heads": head_analysis,
        }
    
    def extract_circuits(
        self,
        layer_analyses: List[Dict],
        coherence_threshold: float = 0.7,
    ) -> List[Circuit]:
        """Extract circuits from layer analyses."""
        circuits = []
        
        # Group layers by dominant φ-shell
        shell_groups: Dict[int, List[int]] = {}
        for analysis in layer_analyses:
            shell = analysis["phi"]["dominant_shell"]
            if shell not in shell_groups:
                shell_groups[shell] = []
            shell_groups[shell].append(analysis["layer_idx"])
        
        # Create circuits for each coherent shell group
        for shell, layer_indices in shell_groups.items():
            if len(layer_indices) < 2:
                continue
            
            # Check coherence across layers
            coherences = [
                layer_analyses[i]["phi"]["coherence"]
                for i in layer_indices
            ]
            mean_coherence = np.mean(coherences)
            
            if mean_coherence < coherence_threshold:
                continue
            
            # Build circuit nodes
            nodes = []
            for layer_idx in layer_indices:
                analysis = layer_analyses[layer_idx]
                stability = "stable" if analysis["stability"]["score"] > 0.8 else "unstable"
                
                nodes.append(CircuitNode(
                    layer_idx=layer_idx,
                    head_idx=None,
                    node_type="residual",
                    phi_shell=shell,
                    stability_class=stability,
                    coherence_score=analysis["phi"]["coherence"],
                ))
            
            # Build edges (sequential for now)
            edges = []
            for i in range(len(nodes) - 1):
                edges.append(CircuitEdge(
                    source=nodes[i],
                    target=nodes[i + 1],
                    weight=1.0,
                    phase_coherent=True,
                ))
            
            # Compute circuit metrics
            stability_scores = [
                layer_analyses[i]["stability"]["score"]
                for i in layer_indices
            ]
            
            circuits.append(Circuit(
                nodes=nodes,
                edges=edges,
                dominant_shell=shell,
                mean_coherence=mean_coherence,
                stability_score=np.mean(stability_scores),
                description=f"φ-shell {shell} circuit across layers {layer_indices}",
            ))
        
        return circuits
    
    def analyze(
        self,
        text: Optional[str] = None,
        input_ids: Optional[Tensor] = None,
    ) -> COSFAnalysisResult:
        """
        Perform complete COSF analysis on input.
        
        Args:
            text: Input text (requires tokenizer)
            input_ids: Pre-tokenized input IDs
            
        Returns:
            COSFAnalysisResult with all analysis components
        """
        # Prepare input
        if text is not None:
            inputs = self._tokenize(text)
        elif input_ids is not None:
            inputs = {"input_ids": input_ids.to(self.device)}
        else:
            raise ValueError("Must provide either text or input_ids")
        
        # Get model config
        config = self._get_model_config()
        
        # Setup hooks
        n_hooks = self.hooks.attach_transformer_layers()
        if n_hooks == 0:
            n_hooks = self.hooks.attach_all_layers()
        
        # Run forward pass with activation capture
        with self.hooks.capture(to_cpu=True) as cache:
            with torch.no_grad():
                self.model(**inputs)
        
        self._activation_cache = cache
        
        # Analyze each layer
        layer_analyses = []
        for name, activations in sorted(cache.activations.items()):
            if "layer" in name.lower():
                try:
                    layer_idx = int(name.split("_")[-1])
                except ValueError:
                    layer_idx = len(layer_analyses)
                
                analysis = self.analyze_layer(
                    activations,
                    layer_idx,
                    config['n_heads'],
                )
                layer_analyses.append(analysis)
        
        # Extract circuits
        circuits = self.extract_circuits(layer_analyses)
        
        # Compute global metrics
        all_phi_coherences = [a["phi"]["coherence"] for a in layer_analyses]
        all_stability_scores = [a["stability"]["score"] for a in layer_analyses]
        all_harmonic_scores = [a["harmonic"]["phi_score"] for a in layer_analyses]
        
        global_metrics = {
            "mean_phi_coherence": float(np.mean(all_phi_coherences)),
            "mean_stability": float(np.mean(all_stability_scores)),
            "mean_harmonic_score": float(np.mean(all_harmonic_scores)),
            "n_circuits": len(circuits),
            "n_layers_analyzed": len(layer_analyses),
        }
        
        # Cleanup
        self.hooks.detach_all()
        
        return COSFAnalysisResult(
            phi_analysis={"layer_coherences": all_phi_coherences},
            harmonic_analysis={"layer_scores": all_harmonic_scores},
            stability_analysis={"layer_scores": all_stability_scores},
            circuits=circuits,
            layer_summaries=layer_analyses,
            global_metrics=global_metrics,
        )
    
    def print_summary(self, result: COSFAnalysisResult) -> None:
        """Print a summary of analysis results."""
        print("=" * 60)
        print("COSF-MI Analysis Summary")
        print("=" * 60)
        
        gm = result.global_metrics
        print(f"\nGlobal Metrics:")
        print(f"  Layers analyzed: {gm['n_layers_analyzed']}")
        print(f"  Circuits found: {gm['n_circuits']}")
        print(f"  Mean φ-coherence: {gm['mean_phi_coherence']:.4f}")
        print(f"  Mean stability: {gm['mean_stability']:.4f}")
        print(f"  Mean harmonic score: {gm['mean_harmonic_score']:.4f}")
        
        if result.circuits:
            print(f"\nCircuits:")
            for i, circuit in enumerate(result.circuits):
                print(f"  [{i}] {circuit.description}")
                print(f"      Coherence: {circuit.mean_coherence:.4f}, Stability: {circuit.stability_score:.4f}")
        
        print("=" * 60)


def quick_analysis(model, tokenizer, text: str) -> Dict:
    """Quick convenience function for COSF analysis."""
    extractor = CircuitExtractor(model, tokenizer)
    result = extractor.analyze(text)
    return {
        "n_circuits": result.global_metrics["n_circuits"],
        "mean_coherence": result.global_metrics["mean_phi_coherence"],
        "mean_stability": result.global_metrics["mean_stability"],
        "circuits": [c.description for c in result.circuits],
    }


if __name__ == "__main__":
    print("Circuit Extractor module loaded.")
    print("Use CircuitExtractor(model, tokenizer) to analyze models.")
