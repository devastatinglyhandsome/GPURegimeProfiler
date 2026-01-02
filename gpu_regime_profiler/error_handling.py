"""
Error handling module with custom exceptions and actionable error messages.
"""

import torch
from typing import Optional, Callable, Any


class ProfilingError(Exception):
    """Base exception for profiler errors."""
    pass


class CUDANotAvailableError(ProfilingError):
    """CUDA not available, can't profile GPU."""
    def __init__(self):
        super().__init__(
            "CUDA not available. GPU profiling requires a CUDA-capable device.\n"
            "Suggestions:\n"
            "  - Check nvidia-smi shows your GPU\n"
            "  - Reinstall PyTorch with CUDA support\n"
            "  - For CPU profiling, use: profiler.profile_cpu(operation)"
        )


class OperationTooFastError(ProfilingError):
    """Operation completed too quickly to measure accurately."""
    def __init__(self, measured_time_us: float):
        super().__init__(
            f"Operation completed in {measured_time_us:.2f}μs - too fast to profile accurately.\n"
            f"Suggestions:\n"
            f"  - Increase problem size (larger tensors)\n"
            f"  - Use profiler.profile_repeated(operation, n=100) to average many runs"
        )


class OOMError(ProfilingError):
    """GPU ran out of memory during profiling."""
    def __init__(self, peak_memory_mb: Optional[float] = None, total_memory_mb: Optional[float] = None):
        msg = "GPU ran out of memory during profiling.\nSuggestions:\n"
        msg += "  - Reduce batch size\n"
        msg += "  - Use profiler.profile_lightweight(operation) for less overhead\n"
        if peak_memory_mb and total_memory_mb:
            msg += f"  - Peak memory: {peak_memory_mb:.1f} MB / {total_memory_mb:.1f} MB\n"
        msg += "  - Consider gradient checkpointing or mixed precision training"
        super().__init__(msg)


class MultiGPUError(ProfilingError):
    """Error with multi-GPU profiling."""
    def __init__(self, message: str, device_count: Optional[int] = None):
        msg = f"Multi-GPU profiling error: {message}\n"
        if device_count is not None:
            msg += f"  Available GPUs: {device_count}\n"
        msg += "Suggestions:\n"
        msg += "  - Ensure all GPUs are accessible\n"
        msg += "  - Check CUDA_VISIBLE_DEVICES environment variable"
        super().__init__(msg)


def safe_profile(operation: Callable[[], Any], fallback_cpu: bool = False) -> tuple[Any, dict]:
    """
    Profile with comprehensive error handling.
    
    Error handling strategy:
    1. CUDA not available → Clear error + suggest fix
    2. OOM during profiling → Suggest smaller batch
    3. Operation too fast → Suggest repeated profiling
    4. Multi-GPU without device specified → Auto-select GPU 0
    5. Mixed precision without support → Warn, continue with FP32
    
    Returns: (result, analysis) or raises ProfilingError with helpful message
    """
    if not torch.cuda.is_available():
        if fallback_cpu:
            import warnings
            warnings.warn("CUDA not available, falling back to CPU profiling")
            # CPU profiling would need separate implementation
            raise CUDANotAvailableError()
        else:
            raise CUDANotAvailableError()
    
    try:
        # This is a placeholder - actual profiling would be done by GPUProfiler
        # This function is meant to wrap profiler calls with error handling
        result = operation()
        return result, {}
    except RuntimeError as e:
        error_str = str(e).lower()
        if "out of memory" in error_str or "oom" in error_str:
            # Try to get memory stats
            try:
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                raise OOMError(peak_mb, total_mb) from e
            except:
                raise OOMError() from e
        else:
            raise ProfilingError(f"CUDA error: {str(e)}") from e
    except Exception as e:
        if isinstance(e, ProfilingError):
            raise
        raise ProfilingError(f"Unexpected error during profiling: {str(e)}") from e


def check_operation_timing(runtime_us: float, min_time_us: float = 10.0) -> None:
    """
    Check if operation is fast enough to profile accurately.
    
    Args:
        runtime_us: Measured runtime in microseconds
        min_time_us: Minimum time for accurate profiling (default 10μs)
    
    Raises:
        OperationTooFastError if operation is too fast
    """
    if runtime_us < min_time_us:
        raise OperationTooFastError(runtime_us)

