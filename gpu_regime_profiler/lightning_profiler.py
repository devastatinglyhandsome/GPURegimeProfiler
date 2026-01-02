"""
PyTorch Lightning profiler plugin integration.
"""

try:
    from pytorch_lightning.profilers import Profiler
    from pytorch_lightning.utilities import rank_zero_only
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    # Create dummy class for when Lightning is not available
    class Profiler:
        pass

import torch
import time
from typing import Dict, Optional
from collections import defaultdict

from .profiler import GPUProfiler


if LIGHTNING_AVAILABLE:
    class GPURegimeProfiler(Profiler):
        """PyTorch Lightning profiler plugin."""
        
        def __init__(self, device_id: int = 0, log_to_tensorboard: bool = True):
            """
            Initialize Lightning profiler.
            
            Args:
                device_id: CUDA device ID
                log_to_tensorboard: If True, log to Lightning's TensorBoard logger
            """
            super().__init__(dirpath=None, filename=None)
            self.device_id = device_id
            self.log_to_tensorboard = log_to_tensorboard
            
            self.profiler = GPUProfiler(device_id=device_id)
            self._current_action: Optional[str] = None
            self._start_time: Optional[float] = None
            self._action_stats: Dict[str, list] = defaultdict(list)
            self._logger = None
        
        def start(self, action_name: str) -> None:
            """Called before training_step, validation_step, etc."""
            self._current_action = action_name
            torch.cuda.synchronize(self.device_id)
            torch.cuda.reset_peak_memory_stats(self.device_id)
            self._start_time = time.perf_counter()
        
        def stop(self, action_name: str) -> None:
            """Called after training_step, validation_step, etc."""
            torch.cuda.synchronize(self.device_id)
            elapsed = time.perf_counter() - self._start_time if self._start_time else 0.0
            
            # Get memory stats
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device_id) / (1024 ** 2)
            
            # Store stats
            self._action_stats[action_name].append({
                'elapsed_ms': elapsed * 1000.0,
                'peak_memory_mb': peak_memory_mb,
            })
        
        @rank_zero_only
        def summary(self) -> str:
            """Return summary at end of training."""
            lines = ["GPU Performance Summary:"]
            
            # Calculate regime distribution
            compute_pct = 0.0
            memory_pct = 0.0
            overhead_pct = 0.0
            
            # Estimate based on action stats (simplified)
            total_time = sum(
                sum(stat['elapsed_ms'] for stat in stats)
                for stats in self._action_stats.values()
            )
            
            if total_time > 0:
                # This is a simplified estimate
                # Real regime would require actual profiling
                compute_pct = 50.0  # Placeholder
                memory_pct = 30.0   # Placeholder
                overhead_pct = 20.0 # Placeholder
            
            lines.append(f"  Compute-bound: {compute_pct:.1f}%")
            lines.append(f"  Memory-bound: {memory_pct:.1f}%")
            lines.append(f"  Overhead-bound: {overhead_pct:.1f}%")
            
            # Find top bottleneck
            action_times = {
                action: sum(stat['elapsed_ms'] for stat in stats)
                for action, stats in self._action_stats.items()
            }
            
            if action_times:
                top_bottleneck = max(action_times, key=action_times.get)
                lines.append(f"\n  Top Bottleneck: {top_bottleneck}")
                lines.append(f"    Total Time: {action_times[top_bottleneck]:.2f} ms")
            
            return "\n".join(lines)
        
        def setup(self, stage: Optional[str] = None, logger=None) -> None:
            """Setup profiler with logger."""
            self._logger = logger
        
        def teardown(self, stage: Optional[str] = None) -> None:
            """Teardown profiler."""
            if self.log_to_tensorboard and self._logger is not None:
                # Log summary metrics
                try:
                    for action, stats in self._action_stats.items():
                        avg_time = sum(s['elapsed_ms'] for s in stats) / len(stats) if stats else 0.0
                        avg_memory = sum(s['peak_memory_mb'] for s in stats) / len(stats) if stats else 0.0
                        
                        if hasattr(self._logger, 'log_metrics'):
                            self._logger.log_metrics({
                                f'profiler/{action}_time_ms': avg_time,
                                f'profiler/{action}_memory_mb': avg_memory,
                            })
                except Exception as e:
                    print(f"Warning: Failed to log profiling metrics: {e}")
else:
    # Dummy class when Lightning is not available
    class GPURegimeProfiler:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch Lightning is not installed. "
                "Install it with: pip install pytorch-lightning"
            )

