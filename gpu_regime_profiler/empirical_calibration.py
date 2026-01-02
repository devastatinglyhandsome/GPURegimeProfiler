"""
Empirical calibration module for deriving regime thresholds from micro-benchmarks.
"""

import torch
import time
from datetime import datetime
from typing import List
from dataclasses import dataclass

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress bar
    def tqdm(iterable, desc=""):
        return iterable

from .hardware_detection import GPUSpecs
from .calibration_cache import EmpiricalThresholds


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    achieved_flops: float
    achieved_bandwidth: float
    runtime_us: float
    flops_utilization: float
    bandwidth_utilization: float


def run_calibration_suite(gpu_specs: GPUSpecs, device_id: int = 0, n_samples: int = 100) -> EmpiricalThresholds:
    """
    Run micro-benchmarks to derive regime thresholds.
    
    Benchmarks to run:
    1. Large matmul (8192x8192) → Expect compute-bound
    2. Element-wise ops (100M elements) → Expect memory-bound
    3. Tiny matmul (32x32) → Expect overhead-bound
    
    Strategy:
    - Run each benchmark n_samples times
    - Measure achieved FLOPS, bandwidth, overhead
    - Derive thresholds as 80th percentile
    
    Returns: EmpiricalThresholds calibrated for this GPU
    
    Time budget: ~30 seconds on first run, cached forever
    """
    device = torch.device(f'cuda:{device_id}')
    results: List[BenchmarkResult] = []
    
    print(f"Running calibration benchmarks for {gpu_specs.name}...")
    print(f"  Peak FP32: {gpu_specs.peak_fp32_tflops:.1f} TFLOPS")
    print(f"  Peak Bandwidth: {gpu_specs.memory_bandwidth_gbps:.1f} GB/s")
    print(f"  Running {n_samples} samples per benchmark...")
    
    # Benchmark 1: Compute-bound baseline (large matmul)
    print("  Benchmark 1/3: Large matmul (compute-bound)...")
    compute_results = _benchmark_large_matmul(device, gpu_specs, n_samples)
    results.extend(compute_results)
    
    # Benchmark 2: Memory-bound baseline (element-wise ops)
    print("  Benchmark 2/3: Element-wise ops (memory-bound)...")
    memory_results = _benchmark_elementwise(device, gpu_specs, n_samples)
    results.extend(memory_results)
    
    # Benchmark 3: Overhead-bound baseline (tiny matmul)
    print("  Benchmark 3/3: Tiny matmul (overhead-bound)...")
    overhead_results = _benchmark_tiny_matmul(device, gpu_specs, n_samples)
    results.extend(overhead_results)
    
    # Derive thresholds from empirical results
    # Use 80th percentile as "good" threshold
    flops_utilizations = [r.flops_utilization for r in results]
    bandwidth_utilizations = [r.bandwidth_utilization for r in results]
    overhead_times = [r.runtime_us for r in overhead_results]
    
    # Sort and get 80th percentile
    flops_utilizations.sort()
    bandwidth_utilizations.sort()
    overhead_times.sort()
    
    percentile_80_idx = int(len(flops_utilizations) * 0.8)
    flops_threshold = flops_utilizations[percentile_80_idx] if percentile_80_idx < len(flops_utilizations) else flops_utilizations[-1]
    bandwidth_threshold = bandwidth_utilizations[percentile_80_idx] if percentile_80_idx < len(bandwidth_utilizations) else bandwidth_utilizations[-1]
    overhead_baseline = overhead_times[percentile_80_idx] if percentile_80_idx < len(overhead_times) else overhead_times[-1]
    
    print(f"  Calibration complete!")
    print(f"    FLOPS threshold: {flops_threshold*100:.1f}%")
    print(f"    Bandwidth threshold: {bandwidth_threshold*100:.1f}%")
    print(f"    Overhead baseline: {overhead_baseline:.2f}μs")
    
    return EmpiricalThresholds(
        bandwidth_threshold=bandwidth_threshold,
        flops_threshold=flops_threshold,
        overhead_baseline_us=overhead_baseline,
        calibration_date=datetime.now().isoformat(),
        sample_size=n_samples * 3,  # 3 benchmarks
        gpu_name=gpu_specs.name,
        peak_fp32_tflops=gpu_specs.peak_fp32_tflops,
        peak_bandwidth_gbps=gpu_specs.memory_bandwidth_gbps,
    )


def _benchmark_large_matmul(device: torch.device, gpu_specs: GPUSpecs, n_samples: int) -> List[BenchmarkResult]:
    """Benchmark large matmul - should saturate compute."""
    results = []
    dim = 8192
    
    # Warmup
    a = torch.randn(dim, dim, device=device, dtype=torch.float32)
    b = torch.randn(dim, dim, device=device, dtype=torch.float32)
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize(device)
    
    # Benchmark
    for _ in tqdm(range(n_samples), desc="    Large matmul", disable=not TQDM_AVAILABLE):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        c = torch.matmul(a, b)
        end_event.record()
        torch.cuda.synchronize(device)
        
        runtime_ms = start_event.elapsed_time(end_event)
        runtime_s = runtime_ms / 1000.0
        runtime_us = runtime_ms * 1000.0
        
        # FLOPS for matmul: 2 * M * N * K
        flops = 2 * dim * dim * dim
        achieved_flops = flops / runtime_s if runtime_s > 0 else 0
        
        # Bytes: read A, read B, write C
        bytes_transferred = (dim * dim * 4) * 3  # 4 bytes per float32
        achieved_bandwidth = bytes_transferred / runtime_s if runtime_s > 0 else 0
        
        peak_flops = gpu_specs.peak_fp32_tflops * 1e12
        peak_bandwidth = gpu_specs.memory_bandwidth_gbps * 1e9
        
        results.append(BenchmarkResult(
            achieved_flops=achieved_flops,
            achieved_bandwidth=achieved_bandwidth,
            runtime_us=runtime_us,
            flops_utilization=achieved_flops / peak_flops if peak_flops > 0 else 0,
            bandwidth_utilization=achieved_bandwidth / peak_bandwidth if peak_bandwidth > 0 else 0,
        ))
    
    return results


def _benchmark_elementwise(device: torch.device, gpu_specs: GPUSpecs, n_samples: int) -> List[BenchmarkResult]:
    """Benchmark element-wise ops - should saturate memory bandwidth."""
    results = []
    size = 100_000_000
    
    # Warmup
    x = torch.randn(size, device=device, dtype=torch.float32)
    for _ in range(3):
        _ = x * 2.0
    torch.cuda.synchronize(device)
    
    # Benchmark
    for _ in tqdm(range(n_samples), desc="    Element-wise", disable=not TQDM_AVAILABLE):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        y = x * 2.0
        end_event.record()
        torch.cuda.synchronize(device)
        
        runtime_ms = start_event.elapsed_time(end_event)
        runtime_s = runtime_ms / 1000.0
        runtime_us = runtime_ms * 1000.0
        
        # FLOPS for element-wise: 1 per element
        flops = size
        achieved_flops = flops / runtime_s if runtime_s > 0 else 0
        
        # Bytes: read x, write y
        bytes_transferred = size * 4 * 2  # 4 bytes per float32
        achieved_bandwidth = bytes_transferred / runtime_s if runtime_s > 0 else 0
        
        peak_flops = gpu_specs.peak_fp32_tflops * 1e12
        peak_bandwidth = gpu_specs.memory_bandwidth_gbps * 1e9
        
        results.append(BenchmarkResult(
            achieved_flops=achieved_flops,
            achieved_bandwidth=achieved_bandwidth,
            runtime_us=runtime_us,
            flops_utilization=achieved_flops / peak_flops if peak_flops > 0 else 0,
            bandwidth_utilization=achieved_bandwidth / peak_bandwidth if peak_bandwidth > 0 else 0,
        ))
    
    return results


def _benchmark_tiny_matmul(device: torch.device, gpu_specs: GPUSpecs, n_samples: int) -> List[BenchmarkResult]:
    """Benchmark tiny matmul - should be overhead-bound."""
    results = []
    dim = 32
    
    # Warmup
    a = torch.randn(dim, dim, device=device, dtype=torch.float32)
    b = torch.randn(dim, dim, device=device, dtype=torch.float32)
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize(device)
    
    # Benchmark
    for _ in tqdm(range(n_samples), desc="    Tiny matmul", disable=not TQDM_AVAILABLE):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        c = torch.matmul(a, b)
        end_event.record()
        torch.cuda.synchronize(device)
        
        runtime_ms = start_event.elapsed_time(end_event)
        runtime_s = runtime_ms / 1000.0
        runtime_us = runtime_ms * 1000.0
        
        # FLOPS for matmul: 2 * M * N * K
        flops = 2 * dim * dim * dim
        achieved_flops = flops / runtime_s if runtime_s > 0 else 0
        
        # Bytes: read A, read B, write C
        bytes_transferred = (dim * dim * 4) * 3
        achieved_bandwidth = bytes_transferred / runtime_s if runtime_s > 0 else 0
        
        peak_flops = gpu_specs.peak_fp32_tflops * 1e12
        peak_bandwidth = gpu_specs.memory_bandwidth_gbps * 1e9
        
        results.append(BenchmarkResult(
            achieved_flops=achieved_flops,
            achieved_bandwidth=achieved_bandwidth,
            runtime_us=runtime_us,
            flops_utilization=achieved_flops / peak_flops if peak_flops > 0 else 0,
            bandwidth_utilization=achieved_bandwidth / peak_bandwidth if peak_bandwidth > 0 else 0,
        ))
    
    return results

