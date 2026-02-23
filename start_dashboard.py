#!/usr/bin/env python3
"""Start the GPU Profiler Dashboard"""

from gpu_regime_profiler import start_dashboard_server

if __name__ == "__main__":
    print("Starting GPU Regime Profiler Dashboard...")
    print("Access at: http://127.0.0.1:8080")
    print("For remote access: ssh -L 8080:localhost:8080 user@host")
    start_dashboard_server(port=8080, blocking=True)
