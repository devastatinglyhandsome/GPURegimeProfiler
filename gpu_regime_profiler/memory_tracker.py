"""
Memory tracking module for GPU memory usage analysis and OOM risk prediction.
"""

import torch
from dataclasses import dataclass
from typing import Callable, Any, Optional


@dataclass
class MemoryAnalysis:
    """Memory usage statistics."""
    allocated_mb: float          # Currently allocated
    reserved_mb: float           # Reserved by PyTorch
    peak_allocated_mb: float     # Peak during operation
    total_available_mb: float    # Total GPU memory
    fragmentation_ratio: float  # Reserved / Allocated
    oom_risk: str               # "LOW" | "MEDIUM" | "HIGH"
    headroom_mb: float          # Available - Peak
    usage_percentage: float     # Peak / Total * 100


def analyze_memory(operation_fn: Callable[[], Any], device_id: int = 0) -> MemoryAnalysis:
    """
    Track memory usage during operation.
    
    Implementation:
    1. Record baseline: torch.cuda.memory_allocated()
    2. Run operation
    3. Record peak: torch.cuda.max_memory_allocated()
    4. Calculate fragmentation
    5. Predict OOM risk based on headroom
    
    OOM risk scoring:
    - HIGH: <1GB headroom or >90% memory used
    - MEDIUM: 1-4GB headroom or 70-90% used
    - LOW: >4GB headroom or <70% used
    
    Args:
        operation_fn: Function to execute (no arguments)
        device_id: CUDA device ID
        
    Returns:
        MemoryAnalysis with memory statistics and OOM risk
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for memory tracking")
    
    device = torch.device(f'cuda:{device_id}')
    
    # Reset peak memory stats before operation
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    
    # Record baseline memory
    baseline_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
    baseline_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB
    
    # Run operation
    try:
        result = operation_fn()
        torch.cuda.synchronize(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Get current stats even if OOM occurred
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
            
            return MemoryAnalysis(
                allocated_mb=baseline_allocated,
                reserved_mb=baseline_reserved,
                peak_allocated_mb=peak_allocated,
                total_available_mb=total_memory,
                fragmentation_ratio=baseline_reserved / baseline_allocated if baseline_allocated > 0 else 1.0,
                oom_risk="HIGH",
                headroom_mb=0.0,
                usage_percentage=100.0,
            )
        else:
            raise
    
    # Record peak memory after operation
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    current_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  # MB
    
    # Calculate metrics
    headroom = total_memory - peak_allocated
    usage_percentage = (peak_allocated / total_memory * 100) if total_memory > 0 else 0.0
    fragmentation_ratio = current_reserved / peak_allocated if peak_allocated > 0 else 1.0
    
    # Predict OOM risk
    if headroom < 1024 or usage_percentage > 90:
        oom_risk = "HIGH"
    elif headroom < 4096 or usage_percentage > 70:
        oom_risk = "MEDIUM"
    else:
        oom_risk = "LOW"
    
    return MemoryAnalysis(
        allocated_mb=baseline_allocated,
        reserved_mb=current_reserved,
        peak_allocated_mb=peak_allocated,
        total_available_mb=total_memory,
        fragmentation_ratio=fragmentation_ratio,
        oom_risk=oom_risk,
        headroom_mb=headroom,
        usage_percentage=usage_percentage,
    )


def get_memory_summary(analysis: MemoryAnalysis) -> str:
    """Get human-readable memory summary."""
    return f"""Memory Analysis:
  Allocated: {analysis.allocated_mb:.1f} MB
  Reserved: {analysis.reserved_mb:.1f} MB
  Peak Allocated: {analysis.peak_allocated_mb:.1f} MB
  Total Available: {analysis.total_available_mb:.1f} MB
  Headroom: {analysis.headroom_mb:.1f} MB
  Usage: {analysis.usage_percentage:.1f}%
  Fragmentation Ratio: {analysis.fragmentation_ratio:.2f}
  OOM Risk: {analysis.oom_risk}"""


def suggest_optimization(analysis: MemoryAnalysis) -> Optional[str]:
    """Suggest memory optimization based on analysis."""
    if analysis.oom_risk == "HIGH":
        if analysis.headroom_mb < 512:
            return "CRITICAL: Very low memory headroom. Consider: smaller batch size, gradient checkpointing, or mixed precision training."
        else:
            return "HIGH RISK: Low memory headroom. Consider: reducing batch size, using gradient accumulation, or enabling gradient checkpointing."
    elif analysis.oom_risk == "MEDIUM":
        return "MEDIUM RISK: Monitor memory usage. Consider: gradient checkpointing for memory-intensive layers."
    else:
        return None  # No suggestion needed for LOW risk

