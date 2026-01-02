#!/usr/bin/env python3
"""
Quick example: Real-time GPU Profiling Dashboard

This script demonstrates the real-time dashboard feature.
"""

import torch
import time
from gpu_regime_profiler import (
    GPUProfiler,
    DashboardClient,
    start_dashboard_with_ngrok
)

# Optional: Set ngrok token for public access
# GPUProfiler.ngrok_token = "your_ngrok_token"

# Start dashboard with ngrok (for remote access)
# For local only: start_dashboard_server(port=8080, blocking=False)
url = start_dashboard_with_ngrok(port=8080, blocking=False)
if url:
    print(f"Dashboard URL: {url}")
else:
    print("Dashboard running at http://127.0.0.1:8080")

time.sleep(2)  # Wait for server to start

# Create profiler and client
profiler = GPUProfiler()
client = DashboardClient(server_url='http://127.0.0.1:8080')

print("\nProfiling operations...")
print("Open the dashboard URL in your browser to see real-time updates!\n")

# Profile various operations
for i in range(20):
    size = 1000 + i * 100
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    _, profile = profiler.profile_with_result(torch.matmul, a, b)
    client.send_profile(profile)
    
    print(f"  [{i+1:2d}/20] Size: {size:4d}x{size:4d} | "
          f"Regime: {profile['regime']:15s} | "
          f"Runtime: {profile['runtime_ms']:6.2f}ms")
    time.sleep(0.3)

print("\nDone! Check the dashboard for visualizations.")

