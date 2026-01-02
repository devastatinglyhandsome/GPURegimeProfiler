"""
Dashboard Client for sending profiling data to the dashboard server.

Designed to be non-blocking and have zero overhead when not in use.
"""

import threading
import time
import json
from typing import Dict, Optional, Callable, Any
from urllib.parse import urljoin
import warnings

try:
    import requests
    CLIENT_AVAILABLE = True
except ImportError:
    CLIENT_AVAILABLE = False
    requests = None

import torch
import pynvml


class DashboardClient:
    """
    Client for sending profiling data to the dashboard server.
    
    All operations are non-blocking and failures are silent to avoid
    affecting profiling performance.
    """
    
    def __init__(self, server_url: str = "http://127.0.0.1:8080"):
        """
        Initialize dashboard client.
        
        Args:
            server_url: URL of the dashboard server (default: http://127.0.0.1:8080)
        """
        self.server_url = server_url.rstrip('/')
        self.api_url = urljoin(self.server_url, "/api/profile")
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.monitoring_interval = 2.5  # Default 2.5 seconds
        self._session = None
        
        # Test connection (non-blocking, silent failure)
        self._test_connection()
    
    def _test_connection(self):
        """Test if server is available (non-blocking, silent failure)."""
        if not CLIENT_AVAILABLE:
            return False
        
        def test():
            try:
                response = requests.get(f"{self.server_url}/api/gpu/status", timeout=1.0)
                return response.status_code == 200
            except Exception:
                return False
        
        # Run in background thread
        thread = threading.Thread(target=test, daemon=True)
        thread.start()
        return None  # Don't wait for result
    
    def send_profile(self, profile_data: Dict, blocking: bool = False):
        """
        Send profiling data to dashboard (non-blocking by default).
        
        Args:
            profile_data: Profile data dictionary (from profiler.profile_with_result)
            blocking: If True, wait for response (default: False for non-blocking)
        """
        if not CLIENT_AVAILABLE:
            return  # Silent failure
        
        # Extract only essential fields to minimize data size
        essential_data = {
            "runtime_ms": profile_data.get("runtime_ms", 0.0),
            "regime": profile_data.get("regime", "UNKNOWN"),
            "flops_utilization": profile_data.get("flops_utilization", 0.0),
            "bandwidth_utilization": profile_data.get("bandwidth_utilization", 0.0),
            "memory": {
                "peak_allocated_mb": profile_data.get("memory", {}).get("peak_allocated_mb", 0.0),
                "oom_risk": profile_data.get("memory", {}).get("oom_risk", "UNKNOWN"),
                "usage_percentage": profile_data.get("memory", {}).get("usage_percentage", 0.0),
            } if "memory" in profile_data else {},
            "timestamp": time.time()
        }
        
        def send():
            try:
                if self._session is None:
                    self._session = requests.Session()
                self._session.post(
                    self.api_url,
                    json=essential_data,
                    timeout=0.5,  # Short timeout to avoid blocking
                    headers={"Content-Type": "application/json"}
                )
            except Exception:
                # Silent failure - don't break profiling
                pass
        
        if blocking:
            send()
        else:
            # Run in background thread (fire-and-forget)
            thread = threading.Thread(target=send, daemon=True)
            thread.start()
    
    def send_profile_async(self, profile_data: Dict):
        """
        Send profiling data asynchronously (fire-and-forget).
        
        This is an alias for send_profile(profile_data, blocking=False).
        """
        self.send_profile(profile_data, blocking=False)
    
    def start_monitoring(self, interval: float = 2.5, device_ids: Optional[list] = None):
        """
        Start background GPU monitoring thread.
        
        Args:
            interval: Polling interval in seconds (default: 2.5)
            device_ids: List of GPU device IDs to monitor (None = all GPUs)
        """
        if self.monitoring_active:
            return  # Already monitoring
        
        if not CLIENT_AVAILABLE:
            warnings.warn("Dashboard client dependencies not available. Install with: pip install 'gpu-regime-profiler[dashboard]'")
            return
        
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available for GPU monitoring")
            return
        
        self.monitoring_interval = interval
        self.monitoring_active = True
        
        def monitor_loop():
            """Background monitoring loop."""
            try:
                pynvml.nvmlInit()
            except Exception:
                self.monitoring_active = False
                return
            
            device_ids_to_monitor = device_ids
            if device_ids_to_monitor is None:
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    device_ids_to_monitor = list(range(device_count))
                except Exception:
                    self.monitoring_active = False
                    pynvml.nvmlShutdown()
                    return
            
            while self.monitoring_active:
                try:
                    # Get GPU metrics
                    gpu_data = {"gpus": [], "timestamp": time.time()}
                    
                    for device_id in device_ids_to_monitor:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                            
                            # Get utilization
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            
                            # Get memory info
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            
                            # Get temperature
                            try:
                                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            except:
                                temp = None
                            
                            # Get power usage
                            try:
                                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                            except:
                                power = None
                            
                            gpu_data["gpus"].append({
                                "device_id": device_id,
                                "gpu_utilization": util.gpu,
                                "memory_utilization": util.memory,
                                "memory_used_mb": mem_info.used / (1024 ** 2),
                                "memory_total_mb": mem_info.total / (1024 ** 2),
                                "memory_free_mb": mem_info.free / (1024 ** 2),
                                "temperature": temp,
                                "power_watts": power,
                            })
                        except Exception:
                            continue  # Skip this GPU if error
                    
                    # Send to dashboard (non-blocking)
                    if gpu_data["gpus"]:
                        self._send_gpu_metrics(gpu_data)
                    
                except Exception:
                    pass  # Silent failure
                
                # Sleep for interval
                time.sleep(self.monitoring_interval)
            
            # Cleanup
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        
        # Start monitoring in background thread
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _send_gpu_metrics(self, gpu_data: Dict):
        """Send GPU metrics to dashboard (non-blocking)."""
        def send():
            try:
                if self._session is None:
                    self._session = requests.Session()
                # Note: This would need a separate endpoint for GPU metrics
                # For now, we'll use the profile endpoint or extend the API
                # This is a placeholder - actual implementation would use WebSocket
                # or a dedicated GPU metrics endpoint
                pass
            except Exception:
                pass
        
        thread = threading.Thread(target=send, daemon=True)
        thread.start()
    
    def stop_monitoring(self):
        """Stop background GPU monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_monitoring()
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass


# Global client instance (lazy initialization)
_global_client: Optional[DashboardClient] = None


def get_dashboard_client(server_url: str = "http://127.0.0.1:8080") -> Optional[DashboardClient]:
    """
    Get or create global dashboard client instance.
    
    Args:
        server_url: URL of the dashboard server
    
    Returns:
        DashboardClient instance or None if dependencies not available
    """
    global _global_client
    
    if not CLIENT_AVAILABLE:
        return None
    
    if _global_client is None:
        try:
            _global_client = DashboardClient(server_url=server_url)
        except Exception:
            return None
    
    return _global_client

