"""
Context manager implementation for GPU profiling.
This module re-exports the context manager from decorators for convenience.
"""

from .decorators import GPUProfilerContext, GPUProfiler, profile_block

__all__ = ['GPUProfilerContext', 'GPUProfiler', 'profile_block']

