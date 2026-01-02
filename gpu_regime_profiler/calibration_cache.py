"""
Persistent cache for GPU calibration results.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class EmpiricalThresholds:
    """Thresholds derived from micro-benchmarks."""
    bandwidth_threshold: float  # Fraction of peak
    flops_threshold: float      # Fraction of peak
    overhead_baseline_us: float # Microseconds
    calibration_date: str
    sample_size: int
    gpu_name: str
    peak_fp32_tflops: float
    peak_bandwidth_gbps: float


class CalibrationCache:
    """Persistent cache of GPU calibration results."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path.home() / '.gpu_profiler'
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def _get_cache_filename(self, gpu_name: str) -> Path:
        """Get cache filename for GPU."""
        # Sanitize GPU name for filename
        safe_name = gpu_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        return self.cache_dir / f"{safe_name}.json"
    
    def get(self, gpu_name: str) -> Optional[EmpiricalThresholds]:
        """Load cached thresholds for this GPU."""
        cache_file = self._get_cache_filename(gpu_name)
        if not cache_file.exists():
            return None
        
        try:
            data = json.loads(cache_file.read_text())
            return EmpiricalThresholds(**data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Cache file corrupted, delete it
            cache_file.unlink()
            return None
    
    def set(self, thresholds: EmpiricalThresholds):
        """Save thresholds to cache."""
        cache_file = self._get_cache_filename(thresholds.gpu_name)
        cache_file.write_text(json.dumps(asdict(thresholds), indent=2))
    
    def clear(self, gpu_name: Optional[str] = None):
        """Clear cache for specific GPU or all GPUs."""
        if gpu_name:
            cache_file = self._get_cache_filename(gpu_name)
            if cache_file.exists():
                cache_file.unlink()
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()

