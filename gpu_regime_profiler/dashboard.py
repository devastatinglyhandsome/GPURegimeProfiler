"""
Main entry point for the GPU Regime Profiler Dashboard.

Provides convenient functions to start and manage the dashboard server.
"""

from typing import Optional

try:
    from .dashboard_server import start_dashboard as _start_dashboard, run_dashboard, DashboardServer
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    _start_dashboard = None
    run_dashboard = None
    DashboardServer = None


def start_dashboard_server(port: int = 8080, host: str = "0.0.0.0", blocking: bool = False):
    """
    Start the dashboard server.
    
    Args:
        port: Port to run the server on (default: 8080)
        host: Host to bind to (default: 127.0.0.1)
        blocking: If True, block until server stops (default: False)
    
    Returns:
        Server instance (if blocking=False) or None (if blocking=True)
    
    Example:
        >>> from gpu_regime_profiler import start_dashboard_server
        >>> start_dashboard_server(port=8080)  # Non-blocking
        >>> # Or blocking:
        >>> start_dashboard_server(port=8080, blocking=True)  # Blocks until Ctrl+C
    """
    if not DASHBOARD_AVAILABLE:
        raise ImportError(
            "Dashboard dependencies not installed. Install with: "
            "pip install 'gpu-regime-profiler[dashboard]'"
        )
    
    if blocking:
        run_dashboard(port=port, host=host)
        return None
    else:
        return _start_dashboard(port=port, host=host)


# Alias for convenience
start_dashboard = start_dashboard_server

