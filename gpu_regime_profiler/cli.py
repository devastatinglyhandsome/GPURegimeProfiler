#!/usr/bin/env python3
"""
Command-line interface for GPURegimeProfiler
"""

import argparse
import torch
from .profiler import GPUProfiler
from .visualizer import create_performance_plots

def main():
    parser = argparse.ArgumentParser(description='GPU Regime Profiler')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create performance visualization plots')
    parser.add_argument('--profile', type=str, choices=['cos', 'matmul'], 
                       help='Profile specific operation')
    parser.add_argument('--size', type=int, default=1000000,
                       help='Problem size for profiling')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available. GPU profiling requires CUDA.")
        return
    
    if args.visualize:
        print("Creating performance visualization...")
        create_performance_plots()
        print("Visualization saved as 'gpu_performance_analysis.png'")
    
    if args.profile:
        profiler = GPUProfiler()
        
        if args.profile == 'cos':
            x = torch.randn(args.size, device='cuda')
            result = profiler.profile_operation(torch.cos, x)
        elif args.profile == 'matmul':
            dim = int(args.size ** 0.5)
            a = torch.randn(dim, dim, device='cuda')
            b = torch.randn(dim, dim, device='cuda')
            result = profiler.profile_operation(torch.matmul, a, b)
        
        print(f"Operation: {args.profile}")
        print(f"Runtime: {result['runtime_ms']:.3f} ms")
        print(f"Regime: {result['regime']}")
        print(f"FLOPS utilization: {result['flops_utilization']*100:.1f}%")
        print(f"Bandwidth utilization: {result['bandwidth_utilization']*100:.1f}%")

if __name__ == '__main__':
    main()
