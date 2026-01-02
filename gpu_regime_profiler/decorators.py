"""
Decorator and context manager APIs for cleaner profiler usage.
"""

import functools
import torch
import time
from typing import Callable, Any, Optional, Dict
from contextlib import contextmanager

from .profiler import GPUProfiler


def profile_regime(log_to=None, show_dashboard=False, send_to_dashboard=False, device_id: int = 0):
    """
    Decorator for automatic profiling.
    
    Usage:
        @profile_regime(log_to=wandb)
        def training_step(batch):
            loss = model(batch)
            loss.backward()
            return loss
        
        # Automatically profiles each call, logs to W&B
    
    Args:
        log_to: Logger object with .log() method (e.g., wandb, tensorboard)
        show_dashboard: If True, show visualization dashboard after each call
        send_to_dashboard: If True, send profiling data to real-time dashboard (default: False)
        device_id: CUDA device ID to use
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use lightweight mode if sending to dashboard to minimize overhead
            lightweight = send_to_dashboard
            
            profiler = GPUProfiler(device_id=device_id)
            result, analysis = profiler.profile_with_result(
                lambda: func(*args, **kwargs),
                lightweight=lightweight
            )
            
            # Send to dashboard (non-blocking, opt-in)
            if send_to_dashboard:
                try:
                    from .dashboard_client import get_dashboard_client
                    client = get_dashboard_client()
                    if client:
                        client.send_profile(analysis, blocking=False)
                except Exception:
                    # Silent failure - don't break profiling
                    pass
            
            if log_to is not None:
                log_data = {
                    'regime': analysis.get('regime', 'UNKNOWN'),
                    'runtime_ms': analysis.get('runtime_ms', 0.0),
                    'achieved_tflops': analysis.get('achieved_flops', 0.0) / 1e12,
                    'flops_utilization': analysis.get('flops_utilization', 0.0),
                    'bandwidth_utilization': analysis.get('bandwidth_utilization', 0.0),
                }
                
                # Add memory info if available
                if 'memory' in analysis:
                    log_data['memory_used_mb'] = analysis['memory'].get('peak_allocated_mb', 0.0)
                    log_data['oom_risk'] = analysis['memory'].get('oom_risk', 'UNKNOWN')
                
                # Try to log - handle different logger interfaces
                try:
                    if hasattr(log_to, 'log'):
                        log_to.log(log_data)
                    elif hasattr(log_to, 'add_scalar'):
                        # TensorBoard-style
                        for key, value in log_data.items():
                            if isinstance(value, (int, float)):
                                log_to.add_scalar(f'profiler/{key}', value)
                    else:
                        # Fallback: print
                        print(f"Profiling {func.__name__}: {log_data}")
                except Exception as e:
                    print(f"Warning: Failed to log profiling data: {e}")
            
            if show_dashboard:
                try:
                    from .visualizer import create_performance_plots
                    create_performance_plots()
                except Exception as e:
                    print(f"Warning: Failed to show dashboard: {e}")
            
            return result
        return wrapper
    return decorator


class GPUProfilerContext:
    """Context manager for profiling code blocks."""
    
    def __init__(self, device_id: int = 0, lightweight: bool = False):
        """
        Initialize profiler context manager.
        
        Args:
            device_id: CUDA device ID to use
            lightweight: If True, use lightweight profiling mode
        """
        self.device_id = device_id
        self.lightweight = lightweight
        self.profiler: Optional[GPUProfiler] = None
        self._analysis: Optional[Dict] = None
    
    def __enter__(self):
        self.profiler = GPUProfiler(device_id=self.device_id)
        torch.cuda.synchronize(self.device_id)
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize(self.device_id)
        self.elapsed = time.perf_counter() - self.start_time
        
        # Store basic timing info
        self._analysis = {
            'runtime_ms': self.elapsed * 1000.0,
            'runtime_s': self.elapsed,
        }
        
        return False  # Don't suppress exceptions
    
    @property
    def analysis(self) -> Dict:
        """Get profiling results."""
        if self._analysis is None:
            return {'runtime_ms': 0.0, 'regime': 'UNKNOWN'}
        return self._analysis
    
    def profile_operation(self, operation: Callable[[], Any]) -> tuple[Any, Dict]:
        """
        Profile an operation within the context.
        
        Usage:
            with GPUProfilerContext() as prof:
                result = prof.profile_operation(lambda: my_operation())
                print(prof.analysis)
        """
        if self.profiler is None:
            raise RuntimeError("Profiler context not entered")
        
        return self.profiler.profile_with_result(operation, lightweight=self.lightweight)


# Alias for convenience
GPUProfiler = GPUProfilerContext


@contextmanager
def profile_block(device_id: int = 0, operation: Optional[Callable[[], Any]] = None):
    """
    Context manager for profiling a code block or operation.
    
    Usage:
        with profile_block() as prof:
            result = my_operation()
        
        print(prof.analysis)
        
        # Or profile a specific operation:
        with profile_block(operation=lambda: my_op()) as prof:
            pass
        print(prof.analysis)
    """
    profiler_ctx = GPUProfilerContext(device_id=device_id)
    
    with profiler_ctx as prof:
        if operation is not None:
            result, analysis = prof.profile_operation(operation)
            prof._analysis = analysis
            prof._result = result
        yield prof

