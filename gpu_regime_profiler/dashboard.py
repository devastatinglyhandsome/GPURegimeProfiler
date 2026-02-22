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

try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    ngrok = None


def start_dashboard_server(port: int = 8080, host: str = "0.0.0.0", blocking: bool = False):
    """
    Start the dashboard server.
    
    Args:
        port: Port to run the server on (default: 8080)
        host: Host to bind to (default: 0.0.0.0 for Colab compatibility)
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
        server = _start_dashboard(port=port, host=host)
        thread = threading.Thread(target=server.run, daemon=False)
        thread.start()
        # Give the server a moment to bind so the port is listening
        time.sleep(0.5)
        return server


# Alias for convenience
start_dashboard = start_dashboard_server


def setup_ngrok_tunnel(port: int = 8080, auth_token: Optional[str] = None, auth_token_file: Optional[str] = None) -> Optional[str]:
    """
    Set up ngrok tunnel for the dashboard (supports WebSockets properly).
    
    Args:
        port: Local port the server is running on
        auth_token: Optional ngrok auth token (get from https://dashboard.ngrok.com/get-started/your-authtoken)
                    If not provided, will use free tier (limited)
        auth_token_file: Optional path to file containing ngrok auth token (alternative to auth_token)
    
    Returns:
        Public URL from ngrok, or None if setup failed
    
    Example:
        >>> url = setup_ngrok_tunnel(8080)
        >>> print(f"Dashboard: {url}")
    """
    if not NGROK_AVAILABLE:
        raise ImportError(
            "pyngrok not installed. Install with: "
            "pip install 'gpu-regime-profiler[dashboard]'"
        )
    
    try:
        # Read token from file if provided
        if auth_token_file:
            try:
                with open(auth_token_file, 'r') as f:
                    content = f.read().strip()
                    # Handle both raw token and "ngrok config add-authtoken TOKEN" format
                    if 'add-authtoken' in content:
                        token = content.split('add-authtoken')[-1].strip()
                    else:
                        token = content
                    if token:
                        auth_token = token
            except Exception as e:
                print(f"Warning: Could not read auth token from file: {e}")
        
        # Set auth token if provided
        if auth_token:
            ngrok.set_auth_token(auth_token)
        
        # Create tunnel: explicit HTTP addr so ngrok forwards to our server (public URL is HTTPS)
        addr = f"127.0.0.1:{port}"
        tunnel = ngrok.connect(addr, bind_tls=True)
        public_url = tunnel.public_url
        
        print(f"Ngrok tunnel established!")
        print(f"  Public URL: {public_url}")
        print(f"  Local URL: http://127.0.0.1:{port}")
        print(f"  If you see ERR_SSL_PROTOCOL_ERROR: use https:// (not http), try incognito or another browser, or use local dashboard at http://127.0.0.1:{port}")
        
        return public_url
        
    except Exception as e:
        print(f"Ngrok setup error: {e}")
        print("  Note: You may need to set an auth token:")
        print("  Get one from: https://dashboard.ngrok.com/get-started/your-authtoken")
        return None


def start_dashboard_with_ngrok(
    port: int = 8080, 
    host: str = "0.0.0.0",
    auth_token: Optional[str] = None,
    auth_token_file: Optional[str] = None,
    blocking: bool = False
) -> Optional[str]:
    """
    Start dashboard server with ngrok tunnel (for Colab/remote access).
    Simplified pattern matching Colab's Flask example.
    
    Args:
        port: Port to run the server on
        host: Host to bind to (default: 0.0.0.0)
        auth_token: Optional ngrok auth token for better performance
        auth_token_file: Optional path to file containing ngrok auth token
        blocking: If True, block until server stops (default: False)
    
    Returns:
        Public URL if ngrok used, None otherwise
    
    Example:
        >>> url = start_dashboard_with_ngrok(port=8080)
        >>> print(f"Dashboard: {url}")
    """
    if not DASHBOARD_AVAILABLE:
        raise ImportError(
            "Dashboard dependencies not installed. Install with: "
            "pip install 'gpu-regime-profiler[dashboard]'"
        )
    
    # Start server in background thread (like Colab Flask example)
    def run_server():
        run_dashboard(port=port, host=host)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Give server a moment to start (like Colab example)
    time.sleep(2)
    
    # Set up ngrok tunnel (simple, like Colab example)
    # Try to get token from GPUProfiler if not provided
    if not auth_token and not auth_token_file:
        try:
            from .profiler import GPUProfiler
            if GPUProfiler.ngrok_token:
                auth_token = GPUProfiler.ngrok_token
        except:
            pass
    
    try:
        url = setup_ngrok_tunnel(port, auth_token=auth_token, auth_token_file=auth_token_file)
        if url:
            print(f"\n{'='*60}")
            print(f"DASHBOARD URL: {url}")
            print(f"{'='*60}")
            if blocking:
                try:
                    # Keep running until interrupted
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nShutting down...")
                    if NGROK_AVAILABLE:
                        ngrok.kill()
            return url
    except Exception as e:
        print(f"Ngrok setup failed: {e}")
        print(f"Dashboard running locally at: http://{host}:{port}")
    
    if blocking:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            if NGROK_AVAILABLE:
                try:
                    ngrok.kill()
                except:
                    pass
    
    return None


def start_dashboard_colab(port: int = 8080, host: str = "0.0.0.0", embed: bool = True):
    """
    Start the dashboard on Google Colab (no ngrok). Uses Colab's built-in port
    proxy so you can view the dashboard in the notebook or via Colab's preview link.

    Args:
        port: Port to run the server on (default: 8080)
        host: Host to bind to (default: 0.0.0.0)
        embed: If True and running in Colab, try to embed the dashboard in an iframe below the cell (default: True)

    Returns:
        The dashboard server. Use DashboardClient(server_url=f'http://127.0.0.1:{port}') to send profiles.

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
            "pip install 'gpu-regime-profiler[dashboard]'"
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
            print(f"Dashboard embedded above (port {port}). If the iframe is blank or 403, use the 'Open preview' link Colab shows for port {port}, or run the next cell to print the proxy URL.")
        except Exception as e:
            print(f"Colab embed failed ({e}). Dashboard is running on port {port}. Look for 'Open preview' next to the cell output, or run: from google.colab import output; print(output.eval_js(f'google.colab.kernel.proxyPort({port})'))")
    else:
        print(f"Dashboard running at http://127.0.0.1:{port}")
        if in_colab:
            print("In Colab: look for 'Open preview' next to the cell, or run: from google.colab import output; output.eval_js('google.colab.kernel.proxyPort(%d)')" % port)
    return server

