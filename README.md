# GPURegimeProfiler

GPU performance profiler with automatic three-regime classification for identifying performance bottlenecks. Based on the "Making Deep Learning Go Brrrr" framework.

## Installation

```bash
pip install GPURegimeProfiler
```

## Quick Start

```python
import torch
from gpu_regime_profiler import GPUProfiler

# Create profiler (auto-calibrates on first use)
profiler = GPUProfiler()

# Get both operation result and profiling data
a = torch.randn(2000, 2000, device='cuda')
b = torch.randn(2000, 2000, device='cuda')

result, profile = profiler.profile_with_result(torch.matmul, a, b)

# Use the actual result
print(f"Result shape: {result.shape}")

# Check performance analysis
print(f"Runtime: {profile['runtime_ms']:.2f} ms")
print(f"Regime: {profile['regime']}")
print(f"FLOPS utilization: {profile['flops_utilization']*100:.1f}%")
print(f"Bandwidth utilization: {profile['bandwidth_utilization']*100:.1f}%")
```

## Auto-Calibration

The profiler automatically calibrates itself on first instantiation:

```python
profiler = GPUProfiler()  # Runs calibration benchmarks
```

Output:
```
GPURegimeProfiler Auto-Calibration
==================================================
GPU Detected: Tesla T4
Mission: Classify GPU operations into performance regimes
   • OVERHEAD_BOUND: Kernel launch dominates
   • MEMORY_BOUND: Memory bandwidth limits performance
   • COMPUTE_BOUND: Math operations limit performance

Running calibration benchmarks...
   Testing memory bandwidth...
   Testing compute throughput...

Calibration Complete!
   Max Memory Bandwidth: 16.0%
   Max Compute FLOPS: 21.7%
   Memory Threshold: 11.2%
   Compute Threshold: 15.2%
==================================================
```

## Profiling Methods

### Profile with Result (Recommended)
Get both operation result and performance data:

```python
result, profile = profiler.profile_with_result(torch.cos, x)
# Use result normally, check profile for performance insights
```

### Profile Only
Get just performance data:

```python
profile = profiler.profile_operation(torch.cos, x)
```

### Lightweight Profiling
Minimal overhead for production use:

```python
result, profile = profiler.profile_with_result(torch.cos, x, lightweight=True)
# Returns only runtime_ms and basic regime classification
```

## Performance Visualization

```python
from gpu_regime_profiler import create_performance_plots
create_performance_plots()  # Saves gpu_performance_analysis.png
```

Creates a comprehensive 4-panel visualization showing:
- Execution time scaling
- Throughput efficiency  
- Memory bandwidth utilization
- Performance regime classification

## Command Line Interface

```bash
# Create performance visualization
gpu-profile --visualize

# Profile specific operations
gpu-profile --profile cos --size 1000000
gpu-profile --profile matmul --size 1000000
```

## Three Performance Regimes

- **OVERHEAD_BOUND**: Operation too small, dominated by kernel launch overhead
- **MEMORY_BOUND**: Limited by memory bandwidth, math units underutilized  
- **COMPUTE_BOUND**: Limited by computational throughput, optimal GPU usage

## Key Features

- **Hardware-adaptive**: Auto-calibrates thresholds for any GPU
- **Accurate timing**: Isolates profiling overhead from measurements
- **Dual output**: Returns both operation results and performance data
- **Zero interference**: Profiling doesn't affect your computation results
- **Modern visualization**: Dark-themed performance plots
- **Production ready**: Lightweight mode for minimal overhead

## Requirements

- Python 3.7+
- PyTorch with CUDA support
- NVIDIA GPU with CUDA capability
- matplotlib, numpy, seaborn

## Use Cases

- Optimize GPU kernel performance
- Identify performance bottlenecks in ML training
- Guide memory access pattern improvements
- Validate GPU utilization in production code
- Research GPU performance characteristics
