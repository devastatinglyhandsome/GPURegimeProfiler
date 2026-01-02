"""
GPURegimeProfiler - GPU performance profiler with three-regime classification

This package provides tools to analyze GPU operations and classify them as:
- Overhead-bound: Limited by kernel launch overhead
- Memory-bound: Limited by memory bandwidth
- Compute-bound: Limited by computational throughput
"""

from .profiler import GPUProfiler
from .visualizer import create_performance_plots
from .calibrator import GPUCalibrator
from .hardware_detection import detect_gpu_specs, GPUSpecs
from .memory_tracker import analyze_memory, MemoryAnalysis, get_memory_summary, suggest_optimization
from .error_handling import (
    ProfilingError, CUDANotAvailableError, OperationTooFastError, 
    OOMError, safe_profile
)
from .multi_gpu import profile_multi_gpu, MultiGPUAnalysis, get_multi_gpu_summary
from .patterns import profile_attention, AttentionProfile, get_attention_suggestions
from .decorators import profile_regime, GPUProfilerContext, profile_block
from .context_manager import GPUProfilerContext as GPUProfilerCM
from .model_analysis import profile_model, ModelProfile, ModelProfiler, get_model_summary
from .lightning_profiler import GPURegimeProfiler as LightningGPURegimeProfiler
from .thread_safe import ThreadSafeProfiler, create_thread_safe_profiler
from .mixed_precision import (
    detect_precision, Precision, adjust_flops_for_precision,
    get_precision_peak_flops, has_tensor_cores
)

__version__ = "1.0.0"
__all__ = [
    # Core profiler
    "GPUProfiler",
    "GPUCalibrator",
    
    # Hardware detection
    "detect_gpu_specs",
    "GPUSpecs",
    
    # Memory tracking
    "analyze_memory",
    "MemoryAnalysis",
    "get_memory_summary",
    "suggest_optimization",
    
    # Error handling
    "ProfilingError",
    "CUDANotAvailableError",
    "OperationTooFastError",
    "OOMError",
    "safe_profile",
    
    # Multi-GPU
    "profile_multi_gpu",
    "MultiGPUAnalysis",
    "get_multi_gpu_summary",
    
    # Attention profiling
    "profile_attention",
    "AttentionProfile",
    "get_attention_suggestions",
    
    # Pythonic API
    "profile_regime",
    "GPUProfilerContext",
    "GPUProfilerCM",
    "profile_block",
    
    # Model analysis
    "profile_model",
    "ModelProfile",
    "ModelProfiler",
    "get_model_summary",
    
    # Lightning integration
    "LightningGPURegimeProfiler",
    
    # Thread safety
    "ThreadSafeProfiler",
    "create_thread_safe_profiler",
    
    # Mixed precision
    "detect_precision",
    "Precision",
    "adjust_flops_for_precision",
    "get_precision_peak_flops",
    "has_tensor_cores",
    
    # Visualization
    "create_performance_plots",
]

# Dashboard (optional - only available if dependencies installed)
try:
    from .dashboard import start_dashboard_server, start_dashboard
    from .dashboard_client import DashboardClient, get_dashboard_client
    __all__.extend([
        "start_dashboard_server",
        "start_dashboard",
        "DashboardClient",
        "get_dashboard_client",
    ])
except ImportError:
    # Dashboard not available - dependencies not installed
    pass
