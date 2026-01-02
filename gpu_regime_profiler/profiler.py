"""
Core GPU Profiler - Three Regime Detection System
Based on "Making Deep Learning Go Brrrr" framework
"""

import torch
import time
import numpy as np
from typing import Dict, Tuple, List

class GPUProfiler:
    # Class variables for calibrated thresholds
    _bandwidth_threshold = None
    _flops_threshold = None
    _calibrated = False
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        
        # Set theoretical peaks first
        self.peak_flops = 19.5e12  # 19.5 TFLOPS for A100
        self.peak_bandwidth = 1.5e12  # 1.5 TB/s for A100
        
        # Run calibration on first instantiation
        if not GPUProfiler._calibrated and torch.cuda.is_available():
            self._run_calibration()
    
    def _run_calibration(self):
        """Run calibration and set class thresholds"""
        print("GPURegimeProfiler Auto-Calibration")
        print("=" * 50)
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print("Mission: Classify GPU operations into performance regimes")
        print("   • OVERHEAD_BOUND: Kernel launch dominates")
        print("   • MEMORY_BOUND: Memory bandwidth limits performance") 
        print("   • COMPUTE_BOUND: Math operations limit performance")
        print("\nRunning calibration benchmarks...")
        
        # Test memory bandwidth peak
        print("   Testing memory bandwidth...")
        max_bandwidth_util = self._test_memory_peak()
        
        # Test compute peak  
        print("   Testing compute throughput...")
        max_flops_util = self._test_compute_peak()
        
        # Calculate thresholds (70% of observed peaks)
        bandwidth_threshold = max_bandwidth_util * 0.7
        flops_threshold = max_flops_util * 0.7
        
        print(f"\nCalibration Complete!")
        print(f"   Max Memory Bandwidth: {max_bandwidth_util*100:.1f}%")
        print(f"   Max Compute FLOPS: {max_flops_util*100:.1f}%")
        print(f"   Memory Threshold: {bandwidth_threshold*100:.1f}%")
        print(f"   Compute Threshold: {flops_threshold*100:.1f}%")
        print("=" * 50)
        
        GPUProfiler._bandwidth_threshold = bandwidth_threshold
        GPUProfiler._flops_threshold = flops_threshold
        GPUProfiler._calibrated = True
    
    def _test_memory_peak(self) -> float:
        """Find peak memory bandwidth utilization"""
        max_util = 0.0
        for size in [50_000_000, 100_000_000, 200_000_000]:
            try:
                x = torch.randn(size, device=self.device)
                result = self._profile_single_operation(torch.cos, x)
                max_util = max(max_util, result['bandwidth_utilization'])
            except RuntimeError:  # Out of memory
                break
        return max_util
    
    def _test_compute_peak(self) -> float:
        """Find peak compute utilization"""
        max_util = 0.0
        for dim in [4000, 6000, 8000]:
            try:
                a = torch.randn(dim, dim, device=self.device)
                b = torch.randn(dim, dim, device=self.device)
                result = self._profile_single_operation(torch.matmul, a, b)
                max_util = max(max_util, result['flops_utilization'])
            except RuntimeError:  # Out of memory
                break
        return max_util
    
    def _profile_single_operation(self, operation_func, *args, **kwargs) -> Dict:
        """Internal profiling method for calibration - no warmup"""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        result = operation_func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        
        runtime_ms = start_event.elapsed_time(end_event)
        runtime_s = runtime_ms / 1000.0
        
        flops = self._estimate_flops(operation_func, args, kwargs)
        bytes_transferred = self._estimate_bytes(args, result)
        
        achieved_flops = flops / runtime_s if runtime_s > 0 else 0
        achieved_bandwidth = bytes_transferred / runtime_s if runtime_s > 0 else 0
        
        return {
            'bandwidth_utilization': achieved_bandwidth / self.peak_bandwidth,
            'flops_utilization': achieved_flops / self.peak_flops,
        }
        
    def profile_operation(self, operation_func, *args, **kwargs) -> Dict:
        """Profile a single operation and classify regime"""
        
        # Warm up
        for _ in range(3):
            _ = operation_func(*args, **kwargs)
        torch.cuda.synchronize()
        
        # Time the operation
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        result = operation_func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        
        runtime_ms = start_event.elapsed_time(end_event)
        runtime_s = runtime_ms / 1000.0
        
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
            'compute_intensity': flops / bytes_transferred if bytes_transferred > 0 else float('inf')
        }
    
    def profile_with_result(self, operation_func, *args, lightweight=False, **kwargs) -> Tuple[any, Dict]:
        """Profile operation and return both the result and profiling data
        
        Args:
            lightweight: If True, skip warmup and detailed metrics for minimal overhead
        """
        
        if not lightweight:
            # Separate warmup phase - don't time this
            for _ in range(3):
                _ = operation_func(*args, **kwargs)
            torch.cuda.synchronize()
        
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
                'regime': 'OVERHEAD_BOUND' if runtime_ms < 1.0 else 'UNKNOWN'
            }
        else:
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
                'compute_intensity': flops / bytes_transferred if bytes_transferred > 0 else float('inf')
            }
        
        return operation_result, profile_data
    
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
        
        flops_util = achieved_flops / self.peak_flops
        bandwidth_util = achieved_bandwidth / self.peak_bandwidth
        
        # Use calibrated thresholds (fallback to defaults if not calibrated yet)
        flops_threshold = GPUProfiler._flops_threshold or 0.15
        bandwidth_threshold = GPUProfiler._bandwidth_threshold or 0.10
        
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
