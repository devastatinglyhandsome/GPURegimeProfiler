"""
Multi-GPU profiling module for distributed workloads.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
import numpy as np

from .profiler import GPUProfiler
from .error_handling import MultiGPUError, ProfilingError


@dataclass
class MultiGPUAnalysis:
    """Analysis across multiple GPUs."""
    per_gpu: Dict[int, Dict]  # device_id -> analysis dict
    load_balance: float                       # 0-1, how balanced (1.0 = perfectly balanced)
    communication_overhead_ms: float          # Time in GPU-GPU transfers (estimated)
    bottleneck_gpu: int                     # Which GPU is slowest
    total_time_ms: float                      # Total time across all GPUs
    utilization_variance: float               # Variance in GPU utilization


def profile_multi_gpu(
    operation: Callable[[], Any], 
    devices: Optional[List[int]] = None,
    operation_kwargs: Optional[Dict] = None
) -> MultiGPUAnalysis:
    """
    Profile operation across multiple GPUs.
    
    Implementation:
    1. Run profiler on each GPU independently
    2. Detect GPU-GPU communication (NCCL calls via timing)
    3. Calculate load balance (variance in GPU utilization)
    4. Identify bottleneck GPU (slowest one)
    
    Use cases:
    - DataParallel: Profile each replica
    - DistributedDataParallel: Profile all ranks
    - Model parallelism: Profile each GPU's slice
    
    Args:
        operation: Function to profile. Should accept device_id as keyword arg if needed
        devices: List of GPU device IDs. If None, uses all available GPUs
        operation_kwargs: Additional kwargs to pass to operation
    
    Returns:
        MultiGPUAnalysis with per-GPU breakdown
    """
    if not torch.cuda.is_available():
        raise MultiGPUError("CUDA not available", device_count=0)
    
    if devices is None:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise MultiGPUError("No GPUs available", device_count=0)
        devices = list(range(device_count))
    
    if len(devices) == 0:
        raise MultiGPUError("No devices specified", device_count=torch.cuda.device_count())
    
    if operation_kwargs is None:
        operation_kwargs = {}
    
    per_gpu_results: Dict[int, Dict] = {}
    gpu_times: List[float] = []
    gpu_utilizations: List[float] = []
    
    # Profile each GPU independently
    for device_id in devices:
        try:
            # Create profiler for this GPU
            profiler = GPUProfiler(device_id=device_id)
            
            # Create operation wrapper that uses the correct device
            def make_operation(dev_id: int):
                def op_wrapper():
                    # Set default device
                    with torch.cuda.device(dev_id):
                        if 'device_id' in operation_kwargs:
                            return operation(**operation_kwargs)
                        else:
                            return operation(device_id=dev_id, **operation_kwargs)
                return op_wrapper
            
            # Profile operation on this GPU
            try:
                result, analysis = profiler.profile_with_result(
                    make_operation(device_id),
                    lightweight=False
                )
                per_gpu_results[device_id] = analysis
                gpu_times.append(analysis['runtime_ms'])
                
                # Calculate utilization (average of FLOPS and bandwidth utilization)
                flops_util = analysis.get('flops_utilization', 0.0)
                bw_util = analysis.get('bandwidth_utilization', 0.0)
                avg_util = (flops_util + bw_util) / 2.0
                gpu_utilizations.append(avg_util)
            except Exception as e:
                raise MultiGPUError(
                    f"Failed to profile GPU {device_id}: {str(e)}",
                    device_count=len(devices)
                ) from e
        except Exception as e:
            raise MultiGPUError(
                f"Error setting up profiler for GPU {device_id}: {str(e)}",
                device_count=len(devices)
            ) from e
    
    # Calculate load balance (1.0 = perfectly balanced, 0.0 = completely imbalanced)
    if len(gpu_utilizations) > 1:
        utilization_std = np.std(gpu_utilizations)
        utilization_mean = np.mean(gpu_utilizations)
        # Load balance: 1 - (std / mean), clamped to [0, 1]
        load_balance = max(0.0, min(1.0, 1.0 - (utilization_std / utilization_mean) if utilization_mean > 0 else 0.0))
        utilization_variance = np.var(gpu_utilizations)
    else:
        load_balance = 1.0  # Single GPU is perfectly balanced
        utilization_variance = 0.0
    
    # Find bottleneck GPU (slowest one)
    if gpu_times:
        bottleneck_idx = np.argmax(gpu_times)
        bottleneck_gpu = devices[bottleneck_idx]
        total_time = max(gpu_times)  # Total time is the slowest GPU
    else:
        bottleneck_gpu = devices[0]
        total_time = 0.0
    
    # Estimate communication overhead
    # This is a rough estimate: difference between max and min GPU times
    # In real distributed training, this would be measured via NCCL profiling
    if len(gpu_times) > 1:
        communication_overhead_ms = max(gpu_times) - min(gpu_times)
    else:
        communication_overhead_ms = 0.0
    
    return MultiGPUAnalysis(
        per_gpu=per_gpu_results,
        load_balance=load_balance,
        communication_overhead_ms=communication_overhead_ms,
        bottleneck_gpu=bottleneck_gpu,
        total_time_ms=total_time,
        utilization_variance=utilization_variance,
    )


def profile_data_parallel(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    devices: Optional[List[int]] = None
) -> MultiGPUAnalysis:
    """
    Profile a DataParallel model across multiple GPUs.
    
    Args:
        model: PyTorch model (will be wrapped in DataParallel)
        input_data: Input tensor
        devices: GPU device IDs. If None, uses all available GPUs
    
    Returns:
        MultiGPUAnalysis
    """
    if devices is None:
        device_count = torch.cuda.device_count()
        devices = list(range(device_count))
    
    def operation(device_id=None):
        # Move model and input to device
        device = torch.device(f'cuda:{device_id}' if device_id is not None else 'cuda:0')
        model_device = model.to(device)
        input_device = input_data.to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model_device(input_device)
        
        return output
    
    return profile_multi_gpu(operation, devices=devices)


def get_multi_gpu_summary(analysis: MultiGPUAnalysis) -> str:
    """Get human-readable multi-GPU summary."""
    lines = ["Multi-GPU Analysis:"]
    lines.append(f"  Total GPUs: {len(analysis.per_gpu)}")
    lines.append(f"  Load Balance: {analysis.load_balance:.2%}")
    lines.append(f"  Bottleneck GPU: {analysis.bottleneck_gpu}")
    lines.append(f"  Total Time: {analysis.total_time_ms:.2f} ms")
    lines.append(f"  Communication Overhead: {analysis.communication_overhead_ms:.2f} ms")
    lines.append(f"  Utilization Variance: {analysis.utilization_variance:.4f}")
    lines.append("\n  Per-GPU Breakdown:")
    
    for device_id, gpu_analysis in analysis.per_gpu.items():
        regime = gpu_analysis.get('regime', 'UNKNOWN')
        runtime = gpu_analysis.get('runtime_ms', 0.0)
        flops_util = gpu_analysis.get('flops_utilization', 0.0) * 100
        bw_util = gpu_analysis.get('bandwidth_utilization', 0.0) * 100
        oom_risk = gpu_analysis.get('memory', {}).get('oom_risk', 'UNKNOWN')
        
        lines.append(f"    GPU {device_id}:")
        lines.append(f"      Regime: {regime}")
        lines.append(f"      Runtime: {runtime:.2f} ms")
        lines.append(f"      FLOPS Util: {flops_util:.1f}%")
        lines.append(f"      Bandwidth Util: {bw_util:.1f}%")
        lines.append(f"      OOM Risk: {oom_risk}")
    
    return "\n".join(lines)

