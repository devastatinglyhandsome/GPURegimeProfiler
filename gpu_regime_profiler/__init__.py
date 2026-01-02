"""
GPURegimeProfiler - GPU performance profiler with three-regime classification

This package provides tools to analyze GPU operations and classify them as:
- Overhead-bound: Limited by kernel launch overhead
- Memory-bound: Limited by memory bandwidth
- Compute-bound: Limited by computational throughput
"""

from .profiler import GPUProfiler
from .visualizer import create_performance_plots
from .calibrator import GPUCalibrator

__version__ = "0.1.0"
__all__ = ["GPUProfiler", "create_performance_plots", "GPUCalibrator"]
