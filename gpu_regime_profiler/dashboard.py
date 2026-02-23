"""
Main entry point for the GPU Regime Profiler Dashboard.

Provides convenient functions to start and manage the dashboard server.
"""

from typing import Optional
import threading
import time

try:
    from .dashboard_server import start_dashboard as _start_dashboard, run_dashboard, DashboardServer
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    _start_dashboard = None
    run_dashboard = None
    DashboardServer = None


def start_dashboard_server(port: int = 8080, host: str = "127.0.0.1", blocking: bool = False):
    """
    Start the dashboard server (local only - no ngrok).
    
    Args:
        port: Port to run the server on (default: 8080)
        host: Host to bind to (default: 127.0.0.1 for local access only)
        blocking: If True, block until server stops (default: False)
    
    Returns:
        Server instance (if blocking=False) or None (if blocking=True)
    
    Example:
        >>> from gpu_regime_profiler import start_dashboard_server
        >>> start_dashboard_server(port=8080)
        >>> # Open http://127.0.0.1:8080 in your browser
    """
    if not DASHBOARD_AVAILABLE:
        raise ImportError(
            "Dashboard dependencies not installed. Install with: "
            "pip install gpu-regime-profiler"
        )
    
    print(f"Starting dashboard at http://{host}:{port}")
    print("Open this URL in your browser to view the dashboard")
    
    if blocking:
        run_dashboard(port=port, host=host)
        return None
    else:
        server = _start_dashboard(port=port, host=host)
        thread = threading.Thread(target=server.run, daemon=False)
        thread.start()
        time.sleep(0.5)
        return server


# Alias for convenience
start_dashboard = start_dashboard_server


def start_dashboard_colab(port: int = 8080, host: str = "0.0.0.0", embed: bool = True):
    """
    Start the dashboard on Google Colab (uses Colab's built-in port proxy).

    Args:
        port: Port to run the server on (default: 8080)
        host: Host to bind to (default: 0.0.0.0)
        embed: If True and running in Colab, embed the dashboard in an iframe (default: True)

    Returns:
        The dashboard server

    Example (run in a Colab cell):
        >>> from gpu_regime_profiler import start_dashboard_colab, GPUProfiler, DashboardClient
        >>> start_dashboard_colab(port=8080)
        >>> profiler = GPUProfiler()
        >>> client = DashboardClient(server_url='http://127.0.0.1:8080')
        >>> _, profile = profiler.profile_with_result(torch.matmul, a, b)
        >>> client.send_profile(profile)
    """
    if not DASHBOARD_AVAILABLE:
        raise ImportError(
            "Dashboard dependencies not installed. Install with: "
            "pip install gpu-regime-profiler"
        )
    server = _start_dashboard(port=port, host=host)
    thread = threading.Thread(target=server.run, daemon=False)
    thread.start()
    time.sleep(1)
    in_colab = False
    try:
        import google.colab  # noqa: F401
        in_colab = True
    except ImportError:
        pass
    if in_colab and embed:
        try:
            from google.colab import output
            js = f"""
            (async () => {{
              const url = await google.colab.kernel.proxyPort({port});
              const iframe = document.createElement('iframe');
              iframe.src = url;
              iframe.width = '100%';
              iframe.height = '600';
              iframe.style.border = 'none';
              document.body.appendChild(iframe);
              return url;
            }})()
            """
            output.eval_js(js)
            print(f"Dashboard embedded above (port {port})")
        except Exception as e:
            print(f"Colab embed failed ({e}). Dashboard running on port {port}")
            print("Look for 'Open preview' next to the cell output")
    else:
        print(f"Dashboard running at http://127.0.0.1:{port}")
        if in_colab:
            print("In Colab: look for 'Open preview' next to the cell")
    return server
