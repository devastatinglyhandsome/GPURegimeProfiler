"""
Core GPU Profiler - Three Regime Detection System
Based on "Making Deep Learning Go Brrrr" framework
"""

import torch
import time
import numpy as np
from typing import Dict, Tuple, List, Optional

from .hardware_detection import detect_gpu_specs, GPUSpecs
from .empirical_calibration import run_calibration_suite
from .calibration_cache import CalibrationCache, EmpiricalThresholds
from .memory_tracker import analyze_memory, MemoryAnalysis
from .error_handling import (
    ProfilingError, CUDANotAvailableError, OperationTooFastError, 
    OOMError, check_operation_timing
)

class GPUProfiler:
    # Class variables for calibrated thresholds (per GPU)
    _calibration_cache: Dict[str, EmpiricalThresholds] = {}
    _gpu_specs_cache: Dict[int, GPUSpecs] = {}
    ngrok_token: Optional[str] = None  # Class variable for ngrok token
    
    def __init__(self, device_id: int = 0, force_recalibration: bool = False, ngrok_token: Optional[str] = None):
        """
        Initialize GPU profiler with hardware-adaptive calibration.
        
        Args:
            device_id: CUDA device ID to use
            force_recalibration: If True, force recalibration even if cached
            ngrok_token: Optional ngrok auth token for real-time dashboard access.
                        If not provided, real-time dashboarding will not be available.
                        Can be set later via: GPUProfiler.ngrok_token = "your_token"
        """
        if not torch.cuda.is_available():
            raise CUDANotAvailableError()
        
        # Set ngrok token (instance or class level)
        if ngrok_token:
            GPUProfiler.ngrok_token = ngrok_token
        elif not GPUProfiler.ngrok_token:
            print("\n" + "="*60)
            print("NOTE: Real-time dashboarding not available")
            print("="*60)
            print("To enable real-time dashboard access, set ngrok token:")
            print("  Option 1: GPUProfiler(ngrok_token='your_token')")
            print("  Option 2: GPUProfiler.ngrok_token = 'your_token'")
            print("="*60 + "\n")
        
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}')
        self.gpu_props = torch.cuda.get_device_properties(device_id)
        
        # Detect GPU specs (cached per device)
        if device_id not in GPUProfiler._gpu_specs_cache:
            self.gpu_specs = detect_gpu_specs(device_id)
            GPUProfiler._gpu_specs_cache[device_id] = self.gpu_specs
        else:
            self.gpu_specs = GPUProfiler._gpu_specs_cache[device_id]
        
        # Set theoretical peaks from detected specs
        self.peak_flops = self.gpu_specs.peak_fp32_tflops * 1e12  # Convert TFLOPS to FLOPS
        self.peak_bandwidth = self.gpu_specs.memory_bandwidth_gbps * 1e9  # Convert GB/s to bytes/s
        
        # Load or run calibration
        cache = CalibrationCache()
        cached_thresholds = cache.get(self.gpu_specs.name)
        
        if cached_thresholds and not force_recalibration:
            # Use cached calibration
            self._bandwidth_threshold = cached_thresholds.bandwidth_threshold
            self._flops_threshold = cached_thresholds.flops_threshold
            GPUProfiler._calibration_cache[self.gpu_specs.name] = cached_thresholds
        else:
            # Run new calibration
            self._run_calibration(cache)
    
    def _run_calibration(self, cache: CalibrationCache):
        """Run calibration and set class thresholds"""
        print("GPURegimeProfiler Auto-Calibration")
        print("=" * 50)
        print(f"GPU Detected: {self.gpu_specs.name}")
        print("Mission: Classify GPU operations into performance regimes")
        print("   • OVERHEAD_BOUND: Kernel launch dominates")
        print("   • MEMORY_BOUND: Memory bandwidth limits performance") 
        print("   • COMPUTE_BOUND: Math operations limit performance")
        print("\nRunning calibration benchmarks...")
        print("   (This is a one-time process, results will be cached)")
        
        # Run empirical calibration suite
        thresholds = run_calibration_suite(self.gpu_specs, self.device_id, n_samples=100)
        
        # Cache the results
        cache.set(thresholds)
        GPUProfiler._calibration_cache[self.gpu_specs.name] = thresholds
        
        self._bandwidth_threshold = thresholds.bandwidth_threshold
        self._flops_threshold = thresholds.flops_threshold
        
        print(f"\nCalibration Complete!")
        print(f"   Memory Threshold: {thresholds.bandwidth_threshold*100:.1f}%")
        print(f"   Compute Threshold: {thresholds.flops_threshold*100:.1f}%")
        print(f"   Overhead Baseline: {thresholds.overhead_baseline_us:.2f}μs")
        print("=" * 50)
    
        
    def profile_operation(self, operation_func, *args, **kwargs) -> Dict:
        """Profile a single operation and classify regime"""
        
        try:
            # Warm up
            for _ in range(3):
                _ = operation_func(*args, **kwargs)
            torch.cuda.synchronize()
            
            # Track memory during operation (separate from timing)
            def operation_wrapper():
                return operation_func(*args, **kwargs)
            
            try:
                memory_analysis = analyze_memory(operation_wrapper, self.device_id)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                    total_mb = self.gpu_props.total_memory / (1024 ** 2)
                    raise OOMError(peak_mb, total_mb) from e
                raise
            
            # Time the operation (separate run for accurate timing)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            result = operation_func(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()
            
            runtime_ms = start_event.elapsed_time(end_event)
            runtime_us = runtime_ms * 1000.0
            runtime_s = runtime_ms / 1000.0
            
            # Check if operation is too fast
            check_operation_timing(runtime_us, min_time_us=10.0)
            
            # Calculate metrics
            flops = self._estimate_flops(operation_func, args, kwargs)
            bytes_transferred = self._estimate_bytes(args, result)
            
            achieved_flops = flops / runtime_s if runtime_s > 0 else 0
            achieved_bandwidth = bytes_transferred / runtime_s if runtime_s > 0 else 0
            
            # Regime classification
            regime = self._classify_regime(achieved_flops, achieved_bandwidth, runtime_s)
            
            return {
                'runtime_ms': runtime_ms,
                'achieved_flops': achieved_flops,
                'achieved_bandwidth': achieved_bandwidth,
                'flops_utilization': achieved_flops / self.peak_flops,
                'bandwidth_utilization': achieved_bandwidth / self.peak_bandwidth,
                'regime': regime,
                'compute_intensity': flops / bytes_transferred if bytes_transferred > 0 else float('inf'),
                'memory': {
                    'allocated_mb': memory_analysis.allocated_mb,
                    'reserved_mb': memory_analysis.reserved_mb,
                    'peak_allocated_mb': memory_analysis.peak_allocated_mb,
                    'total_available_mb': memory_analysis.total_available_mb,
                    'fragmentation_ratio': memory_analysis.fragmentation_ratio,
                    'oom_risk': memory_analysis.oom_risk,
                    'headroom_mb': memory_analysis.headroom_mb,
                    'usage_percentage': memory_analysis.usage_percentage,
                }
            }
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "oom" in error_str:
                try:
                    peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                    total_mb = self.gpu_props.total_memory / (1024 ** 2)
                    raise OOMError(peak_mb, total_mb) from e
                except:
                    raise OOMError() from e
            else:
                raise ProfilingError(f"CUDA error during profiling: {str(e)}") from e
        except ProfilingError:
            raise
        except Exception as e:
            raise ProfilingError(f"Unexpected error during profiling: {str(e)}") from e
    
    def profile_with_result(self, operation_func, *args, lightweight=False, **kwargs) -> Tuple[any, Dict]:
        """Profile operation and return both the result and profiling data
        
        Args:
            lightweight: If True, skip warmup and detailed metrics for minimal overhead
        """
        
        try:
            if not lightweight:
                # Separate warmup phase - don't time this
                for _ in range(3):
                    _ = operation_func(*args, **kwargs)
                torch.cuda.synchronize()
            
            # Track memory during operation (separate from timing)
            def operation_wrapper():
                return operation_func(*args, **kwargs)
            
            try:
                memory_analysis = analyze_memory(operation_wrapper, self.device_id)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                    total_mb = self.gpu_props.total_memory / (1024 ** 2)
                    raise OOMError(peak_mb, total_mb) from e
                raise
            
            # Time ONLY the actual operation the user cares about
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            operation_result = operation_func(*args, **kwargs)  # This is what user gets
            end_event.record()
            torch.cuda.synchronize()
            
            runtime_ms = start_event.elapsed_time(end_event)
            
            if lightweight:
                profile_data = {
                    'runtime_ms': runtime_ms,
                    'regime': 'OVERHEAD_BOUND' if runtime_ms < 1.0 else 'UNKNOWN',
                    'memory': {
                        'oom_risk': memory_analysis.oom_risk,
                        'peak_allocated_mb': memory_analysis.peak_allocated_mb,
                    }
                }
            else:
                runtime_us = runtime_ms * 1000.0
                check_operation_timing(runtime_us, min_time_us=10.0)
                
                # Do expensive calculations AFTER getting the result
                # This doesn't affect the user's operation timing
                runtime_s = runtime_ms / 1000.0
                flops = self._estimate_flops(operation_func, args, kwargs)
                bytes_transferred = self._estimate_bytes(args, operation_result)
                
                achieved_flops = flops / runtime_s if runtime_s > 0 else 0
                achieved_bandwidth = bytes_transferred / runtime_s if runtime_s > 0 else 0
                regime = self._classify_regime(achieved_flops, achieved_bandwidth, runtime_s)
                
                profile_data = {
                    'runtime_ms': runtime_ms,
                    'achieved_flops': achieved_flops,
                    'achieved_bandwidth': achieved_bandwidth,
                    'flops_utilization': achieved_flops / self.peak_flops,
                    'bandwidth_utilization': achieved_bandwidth / self.peak_bandwidth,
                    'regime': regime,
                    'compute_intensity': flops / bytes_transferred if bytes_transferred > 0 else float('inf'),
                    'memory': {
                        'allocated_mb': memory_analysis.allocated_mb,
                        'reserved_mb': memory_analysis.reserved_mb,
                        'peak_allocated_mb': memory_analysis.peak_allocated_mb,
                        'total_available_mb': memory_analysis.total_available_mb,
                        'fragmentation_ratio': memory_analysis.fragmentation_ratio,
                        'oom_risk': memory_analysis.oom_risk,
                        'headroom_mb': memory_analysis.headroom_mb,
                        'usage_percentage': memory_analysis.usage_percentage,
                    }
                }
            
            return operation_result, profile_data
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "oom" in error_str:
                try:
                    peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                    total_mb = self.gpu_props.total_memory / (1024 ** 2)
                    raise OOMError(peak_mb, total_mb) from e
                except:
                    raise OOMError() from e
            else:
                raise ProfilingError(f"CUDA error during profiling: {str(e)}") from e
        except ProfilingError:
            raise
        except Exception as e:
            raise ProfilingError(f"Unexpected error during profiling: {str(e)}") from e
    
    def _estimate_flops(self, operation_func, args, kwargs) -> int:
        """Estimate FLOPS for common operations"""
        # Simplified FLOP counting - extend for more operations
        if hasattr(operation_func, '__name__'):
            op_name = operation_func.__name__
        else:
            op_name = str(operation_func)
            
        if 'matmul' in op_name or 'mm' in op_name:
            # Matrix multiplication: 2 * M * N * K
            if len(args) >= 2:
                a, b = args[0], args[1]
                return 2 * a.numel() * b.shape[-1]
        elif 'cos' in op_name or 'sin' in op_name:
            # Trigonometric: 1 FLOP per element
            return args[0].numel()
        elif 'add' in op_name or 'mul' in op_name:
            # Element-wise: 1 FLOP per element
            return args[0].numel()
            
        return args[0].numel()  # Default: 1 FLOP per element
    
    def _estimate_bytes(self, args, result) -> int:
        """Estimate bytes transferred (read inputs + write output)"""
        total_bytes = 0
        
        # Input bytes
        for arg in args:
            if torch.is_tensor(arg):
                total_bytes += arg.numel() * arg.element_size()
        
        # Output bytes
        if torch.is_tensor(result):
            total_bytes += result.numel() * result.element_size()
            
        return total_bytes
    
    def _classify_regime(self, achieved_flops: float, achieved_bandwidth: float, runtime_s: float) -> str:
        """Classify operation into one of three regimes"""
        
        if self.peak_flops == 0 or self.peak_bandwidth == 0:
            return "UNKNOWN"
        
        flops_util = achieved_flops / self.peak_flops
        bandwidth_util = achieved_bandwidth / self.peak_bandwidth
        
        # Use calibrated thresholds (fallback to defaults if not calibrated yet)
        flops_threshold = getattr(self, '_flops_threshold', None) or 0.15
        bandwidth_threshold = getattr(self, '_bandwidth_threshold', None) or 0.10
        
        if flops_util > flops_threshold:
            return "COMPUTE_BOUND"
        elif bandwidth_util > bandwidth_threshold:
            return "MEMORY_BOUND"
        else:
            return "OVERHEAD_BOUND"
    
    def overhead_test(self, operation_func, base_size: int, *args, **kwargs) -> bool:
        """Test if operation is overhead-bound by doubling batch size"""
        
        # Create tensors with base size and 2x size
        small_args = [torch.randn(base_size, device=self.device) for _ in args]
        large_args = [torch.randn(base_size * 2, device=self.device) for _ in args]
        
        small_profile = self.profile_operation(operation_func, *small_args, **kwargs)
        large_profile = self.profile_operation(operation_func, *large_args, **kwargs)
        
        runtime_increase = large_profile['runtime_ms'] / small_profile['runtime_ms']
        
        # If runtime increases < 50% when doubling size, it's overhead-bound
        return runtime_increase < 1.5

def test_known_cases():
    """Test profiler on known cases from the blog"""
    profiler = GPUProfiler()
    
    if not torch.cuda.is_available():
        print("No GPU available for testing")
        return
    
    print("Testing known cases:")
    
    # Test 1: x.cos() - should be memory-bound
    x = torch.randn(1000000, device='cuda')
    result = profiler.profile_operation(torch.cos, x)
    print(f"cos() - Regime: {result['regime']}, FLOPS util: {result['flops_utilization']:.3f}")
    
    # Test 2: Large matmul - should be compute-bound
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    result = profiler.profile_operation(torch.matmul, a, b)
    print(f"matmul() - Regime: {result['regime']}, FLOPS util: {result['flops_utilization']:.3f}")
    
    # Test 3: Overhead test
    is_overhead = profiler.overhead_test(torch.cos, 1000)
    print(f"cos() overhead test: {'OVERHEAD_BOUND' if is_overhead else 'NOT_OVERHEAD_BOUND'}")

if __name__ == "__main__":
    test_known_cases()
