#!/usr/bin/env python3
"""
Example usage of GPURegimeProfiler.

This script demonstrates basic usage and shows what happens without a GPU.
"""

import sys

# Check Python version
if sys.version_info < (3, 7):
    print("ERROR: Python 3.7 or higher is required")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: PyTorch (torch) is not installed.")
    print("\nTo install:")
    print("  1. Install the package: pip install -e .")
    print("  2. Or install PyTorch manually: pip install torch")
    print("\nIf you have a GPU, install PyTorch with CUDA:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

try:
    from gpu_regime_profiler import GPUProfiler, CUDANotAvailableError
except ImportError as e:
    print(f"ERROR: Could not import gpu_regime_profiler: {e}")
    print("\nMake sure you've installed the package:")
    print("  pip install -e .")
    sys.exit(1)

def main():
    print("GPURegimeProfiler Example")
    print("=" * 50)
    
    # Check CUDA availability first
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available on this system.")
        print("\nThis profiler requires a CUDA-capable GPU.")
        print("\nTo check your system:")
        print("  1. Run: nvidia-smi")
        print("  2. Check PyTorch CUDA: python -c \"import torch; print(torch.cuda.is_available())\"")
        print("\nIf you have a GPU but PyTorch doesn't see it:")
        print("  - Reinstall PyTorch with CUDA support:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    try:
        # Create profiler (will auto-calibrate on first run)
        print("Creating profiler...")
        profiler = GPUProfiler()
        print("Profiler created successfully!")
        print()
        
        # Example 1: Profile a matrix multiplication
        print("Example 1: Matrix Multiplication")
        print("-" * 50)
        a = torch.randn(2000, 2000, device='cuda')
        b = torch.randn(2000, 2000, device='cuda')
        
        result, profile = profiler.profile_with_result(torch.matmul, a, b)
        
        print(f"Result shape: {result.shape}")
        print(f"Runtime: {profile['runtime_ms']:.2f} ms")
        print(f"Regime: {profile['regime']}")
        print(f"FLOPS utilization: {profile['flops_utilization']*100:.1f}%")
        print(f"Bandwidth utilization: {profile['bandwidth_utilization']*100:.1f}%")
        print(f"Memory OOM risk: {profile['memory']['oom_risk']}")
        print()
        
        # Example 2: Profile element-wise operation
        print("Example 2: Element-wise Operation")
        print("-" * 50)
        x = torch.randn(10_000_000, device='cuda')
        
        result, profile = profiler.profile_with_result(torch.cos, x)
        
        print(f"Result shape: {result.shape}")
        print(f"Runtime: {profile['runtime_ms']:.2f} ms")
        print(f"Regime: {profile['regime']}")
        print(f"FLOPS utilization: {profile['flops_utilization']*100:.1f}%")
        print(f"Bandwidth utilization: {profile['bandwidth_utilization']*100:.1f}%")
        print()
        
        print("=" * 50)
        print("Examples completed successfully!")
        
    except CUDANotAvailableError as e:
        print("\nERROR: CUDA not available")
        print(str(e))
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

