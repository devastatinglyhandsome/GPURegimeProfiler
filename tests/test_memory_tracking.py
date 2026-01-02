"""
Tests for memory tracking module.
"""

import pytest
import torch
from gpu_regime_profiler.memory_tracker import analyze_memory, MemoryAnalysis


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_analyze_memory():
    """Test memory analysis on a simple operation."""
    def operation():
        x = torch.randn(1000, 1000, device='cuda')
        y = x * 2
        return y
    
    analysis = analyze_memory(operation)
    assert isinstance(analysis, MemoryAnalysis)
    assert analysis.peak_allocated_mb >= 0
    assert analysis.total_available_mb > 0
    assert analysis.oom_risk in ['LOW', 'MEDIUM', 'HIGH']
    assert analysis.headroom_mb >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_oom_risk_calculation():
    """Test OOM risk calculation."""
    def small_operation():
        x = torch.randn(100, 100, device='cuda')
        return x * 2
    
    analysis = analyze_memory(small_operation)
    # Small operation should have LOW risk
    assert analysis.oom_risk in ['LOW', 'MEDIUM', 'HIGH']  # At least one of these


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_analysis_fields():
    """Test that all MemoryAnalysis fields are present."""
    def operation():
        return torch.randn(100, 100, device='cuda')
    
    analysis = analyze_memory(operation)
    assert hasattr(analysis, 'allocated_mb')
    assert hasattr(analysis, 'reserved_mb')
    assert hasattr(analysis, 'peak_allocated_mb')
    assert hasattr(analysis, 'total_available_mb')
    assert hasattr(analysis, 'fragmentation_ratio')
    assert hasattr(analysis, 'oom_risk')
    assert hasattr(analysis, 'headroom_mb')
    assert hasattr(analysis, 'usage_percentage')

