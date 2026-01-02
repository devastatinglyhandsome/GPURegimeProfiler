"""
Tests for model-level profiling.
"""

import pytest
import torch
import torch.nn as nn
from gpu_regime_profiler.model_analysis import profile_model, ModelProfile, ModelProfiler


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 50)
        self.linear2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profile_model():
    """Test model profiling."""
    model = SimpleModel()
    sample_input = torch.randn(1, 100, device='cuda')
    
    profile = profile_model(model, sample_input)
    
    assert isinstance(profile, ModelProfile)
    assert profile.total_time_ms > 0
    assert len(profile.layers) > 0
    assert len(profile.bottleneck_layers) > 0
    assert profile.peak_memory_mb >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_profiler():
    """Test ModelProfiler class."""
    model = SimpleModel()
    sample_input = torch.randn(1, 100, device='cuda')
    
    profiler = ModelProfiler(model, device_id=0)
    profile = profiler.profile(sample_input)
    
    assert isinstance(profile, ModelProfile)
    assert profile.total_time_ms > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_profile_structure():
    """Test ModelProfile dataclass structure."""
    model = SimpleModel()
    sample_input = torch.randn(1, 100, device='cuda')
    
    profile = profile_model(model, sample_input)
    
    assert hasattr(profile, 'layers')
    assert hasattr(profile, 'total_time_ms')
    assert hasattr(profile, 'bottleneck_layers')
    assert hasattr(profile, 'regime_distribution')
    assert hasattr(profile, 'total_memory_mb')
    assert hasattr(profile, 'peak_memory_mb')

