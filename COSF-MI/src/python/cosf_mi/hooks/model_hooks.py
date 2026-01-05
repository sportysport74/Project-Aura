"""
COSF-MI: Model Hooks for Activation Capture

Provides hooks to intercept activations from transformer models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
import torch
from torch import Tensor, nn


@dataclass
class ActivationCache:
    """Cache for storing captured activations."""
    activations: Dict[str, Tensor] = field(default_factory=dict)
    metadata: Dict[str, dict] = field(default_factory=dict)
    detach: bool = True
    to_cpu: bool = False
    max_stored: Optional[int] = None
    
    def store(self, name: str, activation: Tensor, meta: Optional[dict] = None) -> None:
        """Store an activation tensor."""
        if self.max_stored and len(self.activations) >= self.max_stored:
            oldest = next(iter(self.activations))
            del self.activations[oldest]
            if oldest in self.metadata:
                del self.metadata[oldest]
        
        if self.detach:
            activation = activation.detach()
        if self.to_cpu:
            activation = activation.cpu()
        
        self.activations[name] = activation
        if meta:
            self.metadata[name] = meta
    
    def get(self, name: str) -> Optional[Tensor]:
        return self.activations.get(name)
    
    def clear(self) -> None:
        self.activations.clear()
        self.metadata.clear()
    
    def keys(self) -> List[str]:
        return list(self.activations.keys())
    
    def __len__(self) -> int:
        return len(self.activations)
    
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        return sum(t.numel() * t.element_size() for t in self.activations.values())


class ModelHooks:
    """
    Manages hooks for capturing model activations.
    
    Example:
        >>> model = AutoModel.from_pretrained("gpt2")
        >>> hooks = ModelHooks(model)
        >>> hooks.attach_all_layers()
        >>> 
        >>> with hooks.capture() as cache:
        ...     output = model(input_ids)
        >>> 
        >>> for name, acts in cache.activations.items():
        ...     print(f"{name}: {acts.shape}")
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or self._detect_device()
        self._hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._cache: Optional[ActivationCache] = None
        self._processors: Dict[str, Callable] = {}
    
    def _detect_device(self) -> str:
        try:
            return str(next(self.model.parameters()).device)
        except StopIteration:
            return "cpu"
    
    def _make_hook(self, name: str, capture_input: bool = False, capture_output: bool = True) -> Callable:
        def hook(module, input, output):
            if self._cache is None:
                return
            
            if capture_output:
                activation = output[0] if isinstance(output, tuple) else output
                if name in self._processors:
                    activation = self._processors[name](activation)
                self._cache.store(name, activation, {"type": "output", "shape": list(activation.shape)})
            
            if capture_input:
                activation = input[0] if isinstance(input, tuple) else input
                self._cache.store(f"{name}_input", activation, {"type": "input", "shape": list(activation.shape)})
        
        return hook
    
    def attach(self, module: nn.Module, name: str, capture_input: bool = False, capture_output: bool = True) -> None:
        """Attach a hook to a specific module."""
        if name in self._hooks:
            self._hooks[name].remove()
        
        hook_fn = self._make_hook(name, capture_input, capture_output)
        handle = module.register_forward_hook(hook_fn)
        self._hooks[name] = handle
    
    def attach_by_name(self, module_name: str, hook_name: Optional[str] = None, **kwargs) -> bool:
        """Attach hook to module by its name."""
        hook_name = hook_name or module_name
        try:
            module = self.model
            for part in module_name.split("."):
                module = getattr(module, part)
            self.attach(module, hook_name, **kwargs)
            return True
        except AttributeError:
            return False
    
    def attach_all_layers(self, layer_types: Optional[tuple] = None, name_pattern: Optional[str] = None) -> int:
        """Attach hooks to all layers matching criteria."""
        if layer_types is None:
            layer_types = (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)
        
        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types):
                if name_pattern:
                    import re
                    if not re.search(name_pattern, name):
                        continue
                self.attach(module, name)
                count += 1
        return count
    
    def attach_transformer_layers(self) -> int:
        """Attach hooks to standard transformer layer outputs."""
        count = 0
        patterns = [
            ("transformer.h.{i}", "transformer.ln_f"),
            ("bert.encoder.layer.{i}", "bert.pooler"),
            ("model.layers.{i}", "model.norm"),
            ("encoder.layer.{i}", "decoder.layer.{i}"),
        ]
        
        for layer_pattern, final_pattern in patterns:
            i = 0
            while True:
                name = layer_pattern.format(i=i)
                if self.attach_by_name(name, f"layer_{i}"):
                    count += 1
                    i += 1
                else:
                    break
            
            if count > 0:
                self.attach_by_name(final_pattern, "final_norm")
                break
        
        return count
    
    def register_processor(self, name: str, processor: Callable[[Tensor], Tensor]) -> None:
        """Register a processor to transform activations before caching."""
        self._processors[name] = processor
    
    def detach(self, name: str) -> bool:
        """Remove a specific hook."""
        if name in self._hooks:
            self._hooks[name].remove()
            del self._hooks[name]
            return True
        return False
    
    def detach_all(self) -> None:
        """Remove all hooks."""
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()
    
    @contextmanager
    def capture(self, detach: bool = True, to_cpu: bool = False, max_stored: Optional[int] = None):
        """Context manager for capturing activations."""
        self._cache = ActivationCache(detach=detach, to_cpu=to_cpu, max_stored=max_stored)
        try:
            yield self._cache
        finally:
            self._cache = None
    
    def __del__(self):
        self.detach_all()


def capture_single_forward(
    model: nn.Module,
    inputs: Dict[str, Tensor],
    layer_names: Optional[List[str]] = None,
) -> Dict[str, Tensor]:
    """Convenience function for single forward pass capture."""
    hooks = ModelHooks(model)
    
    if layer_names:
        for name in layer_names:
            hooks.attach_by_name(name)
    else:
        hooks.attach_transformer_layers()
    
    with hooks.capture(to_cpu=True) as cache:
        with torch.no_grad():
            model(**inputs)
    
    hooks.detach_all()
    return dict(cache.activations)


# TransformerLens integration
def create_transformerlens_hooks(model_name: str, device: str = "cuda") -> tuple:
    """Create hooks using TransformerLens for deeper introspection."""
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        raise ImportError("Install transformer-lens: pip install transformer-lens")
    
    model = HookedTransformer.from_pretrained(model_name, device=device)
    return model, model.hook_dict


if __name__ == "__main__":
    print("Testing Model Hooks...")
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, 64)
        
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            return self.layer3(x)
    
    model = SimpleModel()
    hooks = ModelHooks(model)
    
    # Attach to all linear layers
    n = hooks.attach_all_layers(layer_types=(nn.Linear,))
    print(f"Attached {n} hooks")
    
    # Capture activations
    x = torch.randn(4, 64)
    with hooks.capture() as cache:
        output = model(x)
    
    print(f"\nCaptured {len(cache)} activations:")
    for name, acts in cache.activations.items():
        print(f"  {name}: {acts.shape}")
    
    hooks.detach_all()
    print("\nHooks cleaned up.")
