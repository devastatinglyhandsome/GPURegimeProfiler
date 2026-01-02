"""
Mixed precision support for FP16/BF16/FP32 detection and FLOPS adjustment.
"""

import torch
from typing import Optional, Tuple
from enum import Enum

from .hardware_detection import GPUSpecs


class Precision(str, Enum):
    """Tensor precision types."""
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    UNKNOWN = "UNKNOWN"


def detect_precision(tensor: torch.Tensor) -> Precision:
    """Detect tensor precision."""
    dtype = tensor.dtype
    
    if dtype == torch.float32:
        return Precision.FP32
    elif dtype == torch.float16:
        return Precision.FP16
    elif dtype == torch.bfloat16:
        return Precision.BF16
    elif dtype == torch.int8:
        return Precision.INT8
    else:
        return Precision.UNKNOWN


def adjust_flops_for_precision(
    flops_fp32: float,
    precision: Precision,
    gpu_specs: GPUSpecs
) -> float:
    """
    Adjust peak FLOPS based on precision.
    
    FP16/BF16 matmuls on tensor cores:
    - 2x more FLOPS than FP32 (theoretical)
    - Up to 16x on modern GPUs with tensor cores
    
    Args:
        flops_fp32: Peak FLOPS in FP32
        precision: Tensor precision
        gpu_specs: GPU specifications
    
    Returns:
        Adjusted peak FLOPS for the given precision
    """
    if precision == Precision.FP32:
        return flops_fp32
    elif precision == Precision.FP16:
        # Use tensor core FLOPS if available
        return gpu_specs.peak_fp16_tflops * 1e12
    elif precision == Precision.BF16:
        # BF16 typically same as FP16 on tensor cores
        return gpu_specs.peak_fp16_tflops * 1e12
    elif precision == Precision.INT8:
        # INT8 can be 2-4x faster than FP16
        return gpu_specs.peak_fp16_tflops * 1e12 * 2.0
    else:
        return flops_fp32


def get_precision_peak_flops(
    precision: Precision,
    gpu_specs: GPUSpecs
) -> float:
    """
    Get peak FLOPS for a given precision.
    
    Args:
        precision: Tensor precision
        gpu_specs: GPU specifications
    
    Returns:
        Peak FLOPS in FLOPS (not TFLOPS)
    """
    if precision == Precision.FP32:
        return gpu_specs.peak_fp32_tflops * 1e12
    elif precision in [Precision.FP16, Precision.BF16]:
        return gpu_specs.peak_fp16_tflops * 1e12
    elif precision == Precision.INT8:
        # INT8 is typically 2x FP16
        return gpu_specs.peak_fp16_tflops * 1e12 * 2.0
    else:
        return gpu_specs.peak_fp32_tflops * 1e12


def detect_operation_precision(*tensors: torch.Tensor) -> Precision:
    """
    Detect precision of an operation based on input tensors.
    
    Args:
        *tensors: Input tensors to the operation
    
    Returns:
        Detected precision (uses first tensor's precision)
    """
    if not tensors:
        return Precision.UNKNOWN
    
    return detect_precision(tensors[0])


def has_tensor_cores(gpu_specs: GPUSpecs) -> bool:
    """
    Check if GPU has tensor cores.
    
    Tensor cores are available on:
    - Volta (V100) and later
    - Compute capability 7.0+
    
    Args:
        gpu_specs: GPU specifications
    
    Returns:
        True if GPU has tensor cores
    """
    major, minor = gpu_specs.compute_capability
    return major >= 7


def get_tensor_core_speedup(precision: Precision) -> float:
    """
    Get theoretical speedup from tensor cores for a given precision.
    
    Args:
        precision: Tensor precision
    
    Returns:
        Speedup factor (1.0 = no speedup, 16.0 = 16x speedup)
    """
    if precision == Precision.FP32:
        return 1.0
    elif precision in [Precision.FP16, Precision.BF16]:
        return 16.0  # Tensor cores provide ~16x speedup for FP16/BF16
    elif precision == Precision.INT8:
        return 32.0  # INT8 can be even faster
    else:
        return 1.0

