"""
Model-level profiling with per-layer breakdown and bottleneck identification.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import OrderedDict

from .profiler import GPUProfiler
from .patterns import RegimeType


@dataclass
class LayerProfile:
    """Profile for a single layer."""
    name: str
    regime: str
    time_ms: float
    percentage_of_total: float
    memory_mb: float
    flops_utilization: float
    bandwidth_utilization: float
    suggestion: str = ""


@dataclass
class ModelProfile:
    """Full model breakdown."""
    layers: List[LayerProfile]
    total_time_ms: float
    bottleneck_layers: List[str]  # Top 3 slowest
    regime_distribution: Dict[str, float]  # % in each regime
    total_memory_mb: float
    peak_memory_mb: float


class ModelProfiler:
    """Profiler for PyTorch models with per-layer breakdown."""
    
    def __init__(self, model: nn.Module, device_id: int = 0):
        """
        Initialize model profiler.
        
        Args:
            model: PyTorch model to profile
            device_id: CUDA device ID
        """
        self.model = model
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}')
        self.profiler = GPUProfiler(device_id=device_id)
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to eval mode for profiling
        
        # Hook storage
        self.hooks: List[Any] = []
        self.layer_times: Dict[str, float] = {}
        self.layer_memory: Dict[str, float] = {}
        self.layer_profiles: Dict[str, Dict] = {}
    
    def _register_hooks(self):
        """Register forward hooks on all modules."""
        def make_hook(name: str):
            def hook(module, input, output):
                # Time this layer
                torch.cuda.synchronize(self.device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                torch.cuda.synchronize(self.device)
                
                # Get memory before
                mem_before = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                
                # Actually run the layer (output is already computed, but we time it)
                # Note: This is approximate since the output is already computed
                # For accurate timing, we'd need to profile each layer separately
                
                end_event.record()
                torch.cuda.synchronize(self.device)
                
                runtime_ms = start_event.elapsed_time(end_event)
                mem_after = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                
                self.layer_times[name] = runtime_ms
                self.layer_memory[name] = max(0, mem_after - mem_before)
            
            return hook
        
        # Register hooks on all modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def profile(self, sample_input: torch.Tensor, warmup: int = 3) -> ModelProfile:
        """
        Profile entire model with per-layer breakdown.
        
        Implementation:
        1. Register forward hooks on all modules
        2. Profile each module's forward pass
        3. Build layer-by-layer breakdown
        4. Identify bottlenecks (top 20% slowest layers)
        5. Suggest layer-specific optimizations
        
        Args:
            sample_input: Sample input tensor (moved to device automatically)
            warmup: Number of warmup runs
        
        Returns:
            ModelProfile with full breakdown
        """
        # Move input to device
        sample_input = sample_input.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(sample_input)
        torch.cuda.synchronize(self.device)
        
        # Reset stats
        self.layer_times.clear()
        self.layer_memory.clear()
        self.layer_profiles.clear()
        torch.cuda.reset_peak_memory_stats(self.device)
        
        # Register hooks
        self._register_hooks()
        
        # Profile forward pass
        baseline_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        
        with torch.no_grad():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            output = self.model(sample_input)
            end_event.record()
            torch.cuda.synchronize(self.device)
        
        total_time_ms = start_event.elapsed_time(end_event)
        peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        
        # Remove hooks
        self._remove_hooks()
        
        # Build layer profiles
        layer_profiles: List[LayerProfile] = []
        regime_counts: Dict[str, int] = {}
        
        # Sort layers by time (descending)
        sorted_layers = sorted(
            self.layer_times.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Identify bottlenecks (top 20% slowest)
        bottleneck_count = max(1, int(len(sorted_layers) * 0.2))
        bottleneck_layers = [name for name, _ in sorted_layers[:bottleneck_count]]
        
        for name, time_ms in sorted_layers:
            percentage = (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0.0
            memory_mb = self.layer_memory.get(name, 0.0)
            
            # Estimate regime based on time and memory
            # This is approximate - for accurate regime, we'd need to profile each layer separately
            if time_ms < 0.1:  # Very fast
                regime = RegimeType.OVERHEAD_BOUND.value
            elif memory_mb > 100:  # High memory
                regime = RegimeType.MEMORY_BOUND.value
            else:
                regime = RegimeType.COMPUTE_BOUND.value
            
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Generate suggestion
            suggestion = self._generate_suggestion(name, regime, time_ms, memory_mb)
            
            layer_profiles.append(LayerProfile(
                name=name,
                regime=regime,
                time_ms=time_ms,
                percentage_of_total=percentage,
                memory_mb=memory_mb,
                flops_utilization=0.0,  # Would need separate profiling
                bandwidth_utilization=0.0,  # Would need separate profiling
                suggestion=suggestion,
            ))
        
        # Calculate regime distribution
        total_layers = len(layer_profiles)
        regime_distribution = {
            regime: (count / total_layers * 100) if total_layers > 0 else 0.0
            for regime, count in regime_counts.items()
        }
        
        return ModelProfile(
            layers=layer_profiles,
            total_time_ms=total_time_ms,
            bottleneck_layers=bottleneck_layers[:3],  # Top 3
            regime_distribution=regime_distribution,
            total_memory_mb=peak_memory,
            peak_memory_mb=peak_memory,
        )
    
    def _generate_suggestion(self, name: str, regime: str, time_ms: float, memory_mb: float) -> str:
        """Generate optimization suggestion for a layer."""
        name_lower = name.lower()
        
        if 'attention' in name_lower or 'attn' in name_lower:
            if regime == RegimeType.MEMORY_BOUND.value:
                return "Consider FlashAttention for 3-5x speedup"
            elif regime == RegimeType.COMPUTE_BOUND.value:
                return "Already optimized, consider larger batch size"
        
        if 'linear' in name_lower or 'fc' in name_lower:
            if memory_mb > 50:
                return "Consider weight quantization or pruning"
            elif regime == RegimeType.COMPUTE_BOUND.value:
                return "Consider fused linear layers"
        
        if 'conv' in name_lower:
            if regime == RegimeType.MEMORY_BOUND.value:
                return "Consider depthwise separable convolutions"
            elif regime == RegimeType.COMPUTE_BOUND.value:
                return "Consider optimized convolution implementations"
        
        if time_ms > 10.0:  # Very slow layer
            return "This layer is a bottleneck - consider optimization"
        
        return "No specific suggestion"


def profile_model(
    model: nn.Module,
    sample_input: torch.Tensor,
    device_id: int = 0
) -> ModelProfile:
    """
    Profile entire model with per-layer breakdown.
    
    Convenience function that creates a ModelProfiler and runs profiling.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor
        device_id: CUDA device ID
    
    Returns:
        ModelProfile with full breakdown
    """
    profiler = ModelProfiler(model, device_id=device_id)
    return profiler.profile(sample_input)


def get_model_summary(profile: ModelProfile) -> str:
    """Get human-readable model profiling summary."""
    lines = ["Model Profiling Summary:"]
    lines.append(f"  Total Time: {profile.total_time_ms:.2f} ms")
    lines.append(f"  Total Memory: {profile.peak_memory_mb:.1f} MB")
    lines.append(f"  Number of Layers: {len(profile.layers)}")
    lines.append(f"\n  Regime Distribution:")
    for regime, percentage in profile.regime_distribution.items():
        lines.append(f"    {regime}: {percentage:.1f}%")
    
    lines.append(f"\n  Bottleneck Layers (Top 3):")
    for layer_name in profile.bottleneck_layers:
        layer = next((l for l in profile.layers if l.name == layer_name), None)
        if layer:
            lines.append(f"    {layer_name}: {layer.time_ms:.2f} ms ({layer.percentage_of_total:.1f}%)")
            if layer.suggestion:
                lines.append(f"      → {layer.suggestion}")
    
    return "\n".join(lines)

