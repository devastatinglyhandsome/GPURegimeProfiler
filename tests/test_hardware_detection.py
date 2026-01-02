"""
Tests for hardware detection module.
"""

import pytest
import torch
from gpu_regime_profiler.hardware_detection import detect_gpu_specs, GPUSpecs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_detect_gpu_any_hardware():
    """Should work on any GPU without hardcoded values."""
    specs = detect_gpu_specs()
    assert specs.name  # Has a name
    assert specs.peak_fp32_tflops > 0  # Has compute
    assert specs.memory_bandwidth_gbps > 0  # Has bandwidth
    assert specs.total_memory_gb > 0  # Has memory
    assert len(specs.compute_capability) == 2  # Has compute capability


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_specs_structure():
    """Test GPUSpecs dataclass structure."""
    specs = detect_gpu_specs()
    assert isinstance(specs, GPUSpecs)
    assert isinstance(specs.name, str)
    assert isinstance(specs.peak_fp32_tflops, float)
    assert isinstance(specs.peak_fp16_tflops, float)
    assert isinstance(specs.memory_bandwidth_gbps, float)


def test_detect_gpu_no_cuda():
    """Should raise error when CUDA is not available."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            detect_gpu_specs()

