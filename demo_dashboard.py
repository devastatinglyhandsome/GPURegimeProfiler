#!/usr/bin/env python3
"""
Demo script for the GPU Regime Profiler Dashboard (works without GPU).

This script demonstrates how to use the dashboard with mock profiling data.
"""

import time
import random
from gpu_regime_profiler.dashboard_client import DashboardClient

def generate_mock_profile():
    """Generate a mock profiling result."""
    regimes = ['OVERHEAD_BOUND', 'MEMORY_BOUND', 'COMPUTE_BOUND']
    regime = random.choice(regimes)
    
    # Generate realistic-looking data based on regime
    if regime == 'COMPUTE_BOUND':
        flops_util = random.uniform(0.6, 0.95)
        bandwidth_util = random.uniform(0.2, 0.5)
        runtime = random.uniform(5, 50)
    elif regime == 'MEMORY_BOUND':
        flops_util = random.uniform(0.2, 0.5)
        bandwidth_util = random.uniform(0.7, 0.95)
        runtime = random.uniform(10, 100)
    else:  # OVERHEAD_BOUND
        flops_util = random.uniform(0.05, 0.2)
        bandwidth_util = random.uniform(0.05, 0.2)
        runtime = random.uniform(0.1, 2.0)
    
    oom_risks = ['LOW', 'MEDIUM', 'HIGH']
    oom_risk = random.choice(oom_risks)
    
    if oom_risk == 'HIGH':
        memory_peak = random.uniform(15000, 20000)
        usage_pct = random.uniform(85, 95)
    elif oom_risk == 'MEDIUM':
        memory_peak = random.uniform(10000, 15000)
        usage_pct = random.uniform(70, 85)
    else:
        memory_peak = random.uniform(2000, 8000)
        usage_pct = random.uniform(20, 70)
    
    return {
        'runtime_ms': runtime,
        'regime': regime,
        'flops_utilization': flops_util,
        'bandwidth_utilization': bandwidth_util,
        'memory': {
            'peak_allocated_mb': memory_peak,
            'oom_risk': oom_risk,
            'usage_percentage': usage_pct,
        }
    }

def main():
    print("🎯 GPU Regime Profiler Dashboard Demo")
    print("=" * 50)
    print()
    print("This demo sends mock profiling data to the dashboard.")
    print("Make sure the dashboard server is running:")
    print("  gpu-profile --dashboard")
    print("  OR")
    print("  python3 -c 'from gpu_regime_profiler import start_dashboard_server; start_dashboard_server(blocking=True)'")
    print()
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    # Create dashboard client
    try:
        client = DashboardClient(server_url='http://127.0.0.1:8080')
        print("✅ Connected to dashboard server")
    except Exception as e:
        print(f"❌ Error connecting to dashboard: {e}")
        print("   Make sure the dashboard server is running on http://127.0.0.1:8080")
        return
    
    print()
    print("📊 Sending mock profiling data...")
    print("   (Open http://127.0.0.1:8080 in your browser to see it!)")
    print()
    
    # Send 20 mock profiles
    for i in range(20):
        profile = generate_mock_profile()
        client.send_profile(profile, blocking=False)
        
        print(f"  [{i+1}/20] Sent: {profile['regime']:15s} | "
              f"Runtime: {profile['runtime_ms']:6.2f}ms | "
              f"FLOPS: {profile['flops_utilization']*100:5.1f}% | "
              f"Memory: {profile['memory']['peak_allocated_mb']:6.1f}MB | "
              f"OOM: {profile['memory']['oom_risk']}")
        
        time.sleep(0.5)  # Send every 0.5 seconds
    
    print()
    print("✅ Demo complete!")
    print("   Check your browser at http://127.0.0.1:8080 to see the charts update in real-time!")

if __name__ == '__main__':
    main()

