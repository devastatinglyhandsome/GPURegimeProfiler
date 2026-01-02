"""
Thread-safe profiler using thread-local storage for DataLoader worker compatibility.
"""

import threading
from typing import Callable, Any, Dict, Tuple

from .profiler import GPUProfiler


class ThreadSafeProfiler:
    """Thread-safe profiler using thread-local storage."""
    
    def __init__(self, device_id: int = 0):
        """
        Initialize thread-safe profiler.
        
        Args:
            device_id: CUDA device ID (each thread will use this device)
        """
        self.device_id = device_id
        self._local = threading.local()
    
    def _get_profiler(self) -> GPUProfiler:
        """Get thread-local profiler instance."""
        if not hasattr(self._local, 'profiler'):
            self._local.profiler = GPUProfiler(device_id=self.device_id)
        return self._local.profiler
    
    def profile(self, operation: Callable[[], Any], *args, **kwargs) -> Tuple[Any, Dict]:
        """
        Profile operation in a thread-safe manner.
        
        Each thread gets its own profiler instance.
        
        Args:
            operation: Function to profile
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation
        
        Returns:
            (result, analysis) tuple
        """
        profiler = self._get_profiler()
        return profiler.profile_with_result(
            lambda: operation(*args, **kwargs)
        )
    
    def profile_operation(self, operation: Callable[[], Any], *args, **kwargs) -> Dict:
        """
        Profile operation and return only analysis (thread-safe).
        
        Args:
            operation: Function to profile
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation
        
        Returns:
            Analysis dictionary
        """
        profiler = self._get_profiler()
        return profiler.profile_operation(
            lambda: operation(*args, **kwargs)
        )


# Convenience function
def create_thread_safe_profiler(device_id: int = 0) -> ThreadSafeProfiler:
    """Create a thread-safe profiler instance."""
    return ThreadSafeProfiler(device_id=device_id)

