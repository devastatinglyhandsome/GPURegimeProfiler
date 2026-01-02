"""
Tests for decorator and context manager APIs.
"""

import pytest
import torch
from gpu_regime_profiler.decorators import profile_regime, GPUProfilerContext, profile_block


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profile_regime_decorator():
    """Test @profile_regime decorator."""
    @profile_regime()
    def test_function():
        x = torch.randn(100, 100, device='cuda')
        return x * 2
    
    result = test_function()
    assert result is not None
    assert isinstance(result, torch.Tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_profiler_context():
    """Test GPUProfilerContext context manager."""
    with GPUProfilerContext() as prof:
        x = torch.randn(100, 100, device='cuda')
        y = x * 2
    
    assert hasattr(prof, 'analysis')
    assert 'runtime_ms' in prof.analysis


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profile_block():
    """Test profile_block context manager."""
    def operation():
        x = torch.randn(100, 100, device='cuda')
        return x * 2
    
    with profile_block(operation=operation) as prof:
        pass
    
    assert hasattr(prof, 'analysis')
    assert 'runtime_ms' in prof.analysis

