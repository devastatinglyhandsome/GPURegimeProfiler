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
    start_dashboard_server
)

# Start dashboard (local only - secure)
start_dashboard_server(port=8080, blocking=False)
print("Dashboard running at http://127.0.0.1:8080")
print("Open this URL in your browser to see real-time updates!\n")

time.sleep(2)  # Wait for server to start

# Create profiler and client
profiler = GPUProfiler()
client = DashboardClient(server_url='http://127.0.0.1:8080')

print("Profiling operations...\n")

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

