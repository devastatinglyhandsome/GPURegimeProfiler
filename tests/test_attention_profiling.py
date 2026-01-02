"""
Tests for attention profiling.
"""

import pytest
import torch
from gpu_regime_profiler.patterns import profile_attention, AttentionProfile


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profile_attention():
    """Test attention profiling."""
    batch_size = 2
    seq_len = 128
    head_dim = 64
    
    q = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    v = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    
    profile = profile_attention(q, k, v)
    
    assert isinstance(profile, AttentionProfile)
    assert profile.total_time_ms > 0
    assert profile.qk_time_ms > 0
    assert profile.softmax_time_ms > 0
    assert profile.v_time_ms > 0
    assert profile.bottleneck_stage in ['qk_matmul', 'softmax', 'v_matmul']
    assert isinstance(profile.flash_attention_compatible, bool)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attention_profile_structure():
    """Test AttentionProfile dataclass structure."""
    batch_size = 1
    seq_len = 64
    head_dim = 32
    
    q = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    v = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    
    profile = profile_attention(q, k, v)
    
    assert hasattr(profile, 'qk_matmul_regime')
    assert hasattr(profile, 'softmax_regime')
    assert hasattr(profile, 'v_matmul_regime')
    assert hasattr(profile, 'total_time_ms')
    assert hasattr(profile, 'bottleneck_stage')
    assert hasattr(profile, 'flash_attention_compatible')

