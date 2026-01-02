import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from .profiler import GPUProfiler

# Set modern style
plt.style.use('dark_background')
sns.set_palette("husl")

def create_performance_plots():
    profiler = GPUProfiler()
    
    # Test different operation sizes
    sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
    cos_times = []
    matmul_times = []
    regimes = []
    
    print("Creating epic GPU performance visualization...")
    
    for size in sizes:
        # Profile cos operation
        x = torch.randn(size, device='cuda')
        result = profiler.profile_operation(torch.cos, x)
        cos_times.append(result['runtime_ms'])
        regimes.append(result['regime'])
        
        # Profile matmul operation (square matrices)
        dim = int(np.sqrt(size))
        if dim > 10:
            a = torch.randn(dim, dim, device='cuda')
            b = torch.randn(dim, dim, device='cuda')
            result = profiler.profile_operation(torch.matmul, a, b)
            matmul_times.append(result['runtime_ms'])
        else:
            matmul_times.append(0)
    
    # Create epic 2x2 subplot with modern styling
    fig = plt.figure(figsize=(16, 12), facecolor='#0a0a0a')
    fig.suptitle('GPU Performance Regime Analysis', fontsize=24, color='#00ff88', fontweight='bold', y=0.95)
    
    # Custom color scheme
    colors = {
        'OVERHEAD_BOUND': '#ff4444',    # Red
        'MEMORY_BOUND': '#ffaa00',      # Orange  
        'COMPUTE_BOUND': '#00ff88'      # Green
    }
    
    # Plot 1: Execution Time Scaling (top-left)
    ax1 = plt.subplot(2, 2, 1, facecolor='#111111')
    ax1.loglog(sizes, cos_times, 'o-', color='#00aaff', linewidth=3, markersize=8, label='cos', alpha=0.8)
    valid_matmul = [t for t in matmul_times if t > 0]
    ax1.loglog(sizes[:len(valid_matmul)], valid_matmul, 's-', color='#ff6600', linewidth=3, markersize=8, label='matmul', alpha=0.8)
    ax1.set_xlabel('Problem Size', fontsize=14, color='white')
    ax1.set_ylabel('Execution Time (ms)', fontsize=14, color='white')
    ax1.set_title('Scaling Performance', fontsize=16, color='#00ff88', fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, color='#333333')
    ax1.tick_params(colors='white')
    
    # Plot 2: Throughput Analysis (top-right)
    ax2 = plt.subplot(2, 2, 2, facecolor='#111111')
    throughput_cos = [s/t*1000 for s, t in zip(sizes, cos_times)]
    throughput_matmul = [s/t*1000 for s, t in zip(sizes, matmul_times) if t > 0]
    
    ax2.semilogx(sizes, throughput_cos, 'o-', color='#00aaff', linewidth=3, markersize=8, label='cos', alpha=0.8)
    ax2.semilogx(sizes[:len(throughput_matmul)], throughput_matmul, 's-', color='#ff6600', linewidth=3, markersize=8, label='matmul', alpha=0.8)
    ax2.set_xlabel('Problem Size', fontsize=14, color='white')
    ax2.set_ylabel('Throughput (elements/sec)', fontsize=14, color='white')
    ax2.set_title('Throughput Efficiency', fontsize=16, color='#00ff88', fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, color='#333333')
    ax2.tick_params(colors='white')
    
    # Plot 3: Memory Bandwidth Heatmap (bottom-left)
    ax3 = plt.subplot(2, 2, 3, facecolor='#111111')
    tesla_t4_bandwidth = 320e9
    cos_bandwidth = [s*4*2/t*1e-6 for s, t in zip(sizes, cos_times)]
    cos_utilization = [bw/tesla_t4_bandwidth*100 for bw in cos_bandwidth]
    
    # Create gradient effect
    colors_grad = plt.cm.plasma(np.linspace(0, 1, len(sizes)))
    scatter = ax3.scatter(sizes, cos_utilization, c=cos_utilization, s=150, cmap='plasma', alpha=0.8, edgecolors='white', linewidth=2)
    ax3.plot(sizes, cos_utilization, '--', color='#666666', alpha=0.5, linewidth=2)
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Problem Size', fontsize=14, color='white')
    ax3.set_ylabel('Memory Bandwidth Utilization (%)', fontsize=14, color='white')
    ax3.set_title('Memory Efficiency', fontsize=16, color='#00ff88', fontweight='bold')
    ax3.axhline(y=50, color='#ff4444', linestyle='--', linewidth=2, alpha=0.7, label='50% Target')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3, color='#333333')
    ax3.tick_params(colors='white')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Bandwidth %', color='white', fontsize=12)
    cbar.ax.tick_params(colors='white')
    
    # Plot 4: Regime Classification with Epic Styling (bottom-right)
    ax4 = plt.subplot(2, 2, 4, facecolor='#111111')
    
    # Create regime-colored scatter plot
    regime_colors_list = [colors[r] for r in regimes]
    scatter2 = ax4.scatter(sizes, cos_times, c=regime_colors_list, s=200, alpha=0.8, edgecolors='white', linewidth=2)
    
    # Add glow effect
    for i, (size, time, regime) in enumerate(zip(sizes, cos_times, regimes)):
        ax4.scatter(size, time, c=colors[regime], s=400, alpha=0.2)  # Glow
    
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Problem Size', fontsize=14, color='white')
    ax4.set_ylabel('Execution Time (ms)', fontsize=14, color='white')
    ax4.set_title('Performance Regimes', fontsize=16, color='#00ff88', fontweight='bold')
    
    # Custom legend
    legend_elements = [
        plt.scatter([], [], c=colors['OVERHEAD_BOUND'], s=100, label='OVERHEAD_BOUND'),
        plt.scatter([], [], c=colors['MEMORY_BOUND'], s=100, label='MEMORY_BOUND'),
        plt.scatter([], [], c=colors['COMPUTE_BOUND'], s=100, label='COMPUTE_BOUND')
    ]
    ax4.legend(handles=legend_elements, fontsize=12, loc='upper left')
    ax4.grid(True, alpha=0.3, color='#333333')
    ax4.tick_params(colors='white')
    
    # Add GPU info text
    gpu_name = torch.cuda.get_device_name(0)
    fig.text(0.02, 0.02, f'GPU: {gpu_name}', fontsize=12, color='#888888', alpha=0.7)
    fig.text(0.98, 0.02, 'GPURegimeProfiler v0.1.0', fontsize=12, color='#888888', alpha=0.7, ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    plt.savefig('gpu_performance_analysis.png', dpi=200, bbox_inches='tight', facecolor='#0a0a0a')
    plt.show()
    
    print("Epic visualization saved as 'gpu_performance_analysis.png'")

if __name__ == "__main__":
    create_performance_plots()
