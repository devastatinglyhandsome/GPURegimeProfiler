"""
Hardware detection module for GPU specifications.
Uses pynvml to query GPU hardware specs and looks up theoretical peak FLOPS.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class GPUSpecs:
    """Hardware specifications for any GPU."""
    name: str
    compute_capability: Tuple[int, int]
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_tensor_tflops: float
    memory_bandwidth_gbps: float
    total_memory_gb: float
    device_id: int = 0


# GPU database with theoretical peak FLOPS
# Sources: NVIDIA official specs and public benchmarks
GPU_DATABASE = {
    # Data Center GPUs
    "Tesla T4": {
        "peak_fp32_tflops": 8.1,
        "peak_fp16_tflops": 65.0,
        "peak_tensor_tflops": 130.0,
        "memory_bandwidth_gbps": 320.0,
    },
    "Tesla V100": {
        "peak_fp32_tflops": 15.7,
        "peak_fp16_tflops": 125.0,
        "peak_tensor_tflops": 125.0,
        "memory_bandwidth_gbps": 900.0,
    },
    "A100": {
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,
        "peak_tensor_tflops": 624.0,
        "memory_bandwidth_gbps": 2039.0,
    },
    "A100-SXM4-40GB": {
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,
        "peak_tensor_tflops": 624.0,
        "memory_bandwidth_gbps": 2039.0,
    },
    "A100-SXM4-80GB": {
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,
        "peak_tensor_tflops": 624.0,
        "memory_bandwidth_gbps": 2039.0,
    },
    "H100": {
        "peak_fp32_tflops": 67.0,
        "peak_fp16_tflops": 1000.0,
        "peak_tensor_tflops": 2000.0,
        "memory_bandwidth_gbps": 3000.0,
    },
    "H100-PCIE-80GB": {
        "peak_fp32_tflops": 51.0,
        "peak_fp16_tflops": 756.0,
        "peak_tensor_tflops": 1513.0,
        "memory_bandwidth_gbps": 2000.0,
    },
    "H100-SXM-80GB": {
        "peak_fp32_tflops": 67.0,
        "peak_fp16_tflops": 1000.0,
        "peak_tensor_tflops": 2000.0,
        "memory_bandwidth_gbps": 3000.0,
    },
    # Consumer GPUs
    "NVIDIA GeForce RTX 4090": {
        "peak_fp32_tflops": 83.0,
        "peak_fp16_tflops": 330.0,
        "peak_tensor_tflops": 1321.0,
        "memory_bandwidth_gbps": 1008.0,
    },
    "NVIDIA GeForce RTX 3090": {
        "peak_fp32_tflops": 36.0,
        "peak_fp16_tflops": 142.0,
        "peak_tensor_tflops": 568.0,
        "memory_bandwidth_gbps": 936.0,
    },
    "NVIDIA GeForce RTX 3080": {
        "peak_fp32_tflops": 30.0,
        "peak_fp16_tflops": 120.0,
        "peak_tensor_tflops": 480.0,
        "memory_bandwidth_gbps": 760.0,
    },
    "NVIDIA GeForce RTX 2080 Ti": {
        "peak_fp32_tflops": 13.4,
        "peak_fp16_tflops": 53.8,
        "peak_tensor_tflops": 107.6,
        "memory_bandwidth_gbps": 616.0,
    },
}


def _get_compute_capability_from_pytorch(device_id: int = 0) -> Tuple[int, int]:
    """Get compute capability using PyTorch."""
    if not torch.cuda.is_available():
        return (0, 0)
    
    props = torch.cuda.get_device_properties(device_id)
    major = props.major
    minor = props.minor
    return (major, minor)


def _lookup_gpu_specs(gpu_name: str) -> Optional[dict]:
    """Look up GPU specs from database."""
    # Try exact match first
    if gpu_name in GPU_DATABASE:
        return GPU_DATABASE[gpu_name]
    
    # Try partial matches
    for db_name, specs in GPU_DATABASE.items():
        if db_name.lower() in gpu_name.lower() or gpu_name.lower() in db_name.lower():
            return specs
    
    return None


def _estimate_specs_from_compute_capability(compute_cap: Tuple[int, int], memory_gb: float) -> dict:
    """Estimate specs based on compute capability and memory.
    
    This is a fallback when GPU is not in database.
    Uses conservative estimates based on architecture generation.
    """
    major, minor = compute_cap
    
    # Ampere (8.0) - A100, RTX 30/40 series
    if major == 8:
        if minor >= 6:  # A100
            return {
                "peak_fp32_tflops": 19.5,
                "peak_fp16_tflops": 312.0,
                "peak_tensor_tflops": 624.0,
                "memory_bandwidth_gbps": 2000.0,
            }
        else:  # RTX 30/40 series
            return {
                "peak_fp32_tflops": 30.0,
                "peak_fp16_tflops": 120.0,
                "peak_tensor_tflops": 480.0,
                "memory_bandwidth_gbps": 800.0,
            }
    
    # Turing (7.5) - RTX 20 series, T4
    elif major == 7 and minor == 5:
        return {
            "peak_fp32_tflops": 10.0,
            "peak_fp16_tflops": 40.0,
            "peak_tensor_tflops": 80.0,
            "memory_bandwidth_gbps": 400.0,
        }
    
    # Volta (7.0) - V100
    elif major == 7 and minor == 0:
        return {
            "peak_fp32_tflops": 15.7,
            "peak_fp16_tflops": 125.0,
            "peak_tensor_tflops": 125.0,
            "memory_bandwidth_gbps": 900.0,
        }
    
    # Pascal (6.0, 6.1) - P100, GTX 10 series
    elif major == 6:
        return {
            "peak_fp32_tflops": 10.0,
            "peak_fp16_tflops": 20.0,
            "peak_tensor_tflops": 20.0,
            "memory_bandwidth_gbps": 500.0,
        }
    
    # Default conservative estimate
    return {
        "peak_fp32_tflops": 5.0,
        "peak_fp16_tflops": 10.0,
        "peak_tensor_tflops": 10.0,
        "memory_bandwidth_gbps": 200.0,
    }


def detect_gpu_specs(device_id: int = 0) -> GPUSpecs:
    """
    Query GPU hardware specs using nvidia-ml-py and PyTorch.
    
    Returns: GPUSpecs with all hardware characteristics
    
    Implementation:
    1. Use pynvml to query device properties (if available)
    2. Use PyTorch as fallback
    3. Look up theoretical peak FLOPS from database
    4. Fall back to estimates based on compute capability if not in database
    
    Edge cases:
    - Multiple GPUs: Return specs for specified CUDA device
    - Unknown GPU: Fall back to conservative estimates based on compute capability
    - No CUDA: Raise clear error
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. GPU profiling requires a CUDA-capable device.\n"
            "Suggestions:\n"
            "  - Check nvidia-smi shows your GPU\n"
            "  - Reinstall PyTorch with CUDA support\n"
            "  - For CPU profiling, use a different tool"
        )
    
    # Get GPU name and memory from PyTorch
    props = torch.cuda.get_device_properties(device_id)
    gpu_name = props.name
    total_memory_bytes = props.total_memory
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    compute_capability = (props.major, props.minor)
    
    # Try to get memory bandwidth from pynvml if available
    memory_bandwidth_gbps = None
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            # Memory bandwidth is not directly available, but we can estimate
            # or look it up from database
        except Exception:
            pass  # Fall back to database lookup
    
    # Look up specs from database
    db_specs = _lookup_gpu_specs(gpu_name)
    
    if db_specs:
        peak_fp32_tflops = db_specs["peak_fp32_tflops"]
        peak_fp16_tflops = db_specs["peak_fp16_tflops"]
        peak_tensor_tflops = db_specs["peak_tensor_tflops"]
        memory_bandwidth_gbps = db_specs["memory_bandwidth_gbps"]
    else:
        # Fall back to estimates based on compute capability
        estimated = _estimate_specs_from_compute_capability(compute_capability, total_memory_gb)
        peak_fp32_tflops = estimated["peak_fp32_tflops"]
        peak_fp16_tflops = estimated["peak_fp16_tflops"]
        peak_tensor_tflops = estimated["peak_tensor_tflops"]
        memory_bandwidth_gbps = estimated["memory_bandwidth_gbps"]
    
    # Convert GB/s to bytes/s for internal use
    memory_bandwidth_bytes_per_sec = memory_bandwidth_gbps * 1e9
    
    return GPUSpecs(
        name=gpu_name,
        compute_capability=compute_capability,
        peak_fp32_tflops=peak_fp32_tflops,
        peak_fp16_tflops=peak_fp16_tflops,
        peak_tensor_tflops=peak_tensor_tflops,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        total_memory_gb=total_memory_gb,
        device_id=device_id,
    )

