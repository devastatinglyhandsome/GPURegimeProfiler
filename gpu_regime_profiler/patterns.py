"""
Pattern recognition for common GPU operation patterns, especially attention operations.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

from .profiler import GPUProfiler


class RegimeType(str, Enum):
    """Performance regime types."""
    OVERHEAD_BOUND = "OVERHEAD_BOUND"
    MEMORY_BOUND = "MEMORY_BOUND"
    COMPUTE_BOUND = "COMPUTE_BOUND"
    UNKNOWN = "UNKNOWN"


@dataclass
class AttentionProfile:
    """Specialized profiling for attention operations."""
    qk_matmul_regime: RegimeType      # Q @ K.T
    softmax_regime: RegimeType        # Usually memory-bound
    v_matmul_regime: RegimeType       # @ V
    total_time_ms: float
    qk_time_ms: float
    softmax_time_ms: float
    v_time_ms: float
    bottleneck_stage: str             # "qk_matmul" | "softmax" | "v_matmul"
    flash_attention_compatible: bool  # Could use FlashAttention?
    qk_flops_utilization: float
    v_flops_utilization: float
    softmax_bandwidth_utilization: float


def profile_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device_id: int = 0,
    scale: Optional[float] = None
) -> AttentionProfile:
    """
    Profile attention with stage-by-stage breakdown.
    
    Attention computation: Q @ K.T → scale → softmax → @ V
    
    Implementation:
    1. Profile Q @ K.T separately
    2. Profile softmax separately
    3. Profile @ V separately
    4. Identify bottleneck stage
    5. Check if FlashAttention would help (if softmax is bottleneck)
    
    Args:
        q: Query tensor [batch, seq_len, head_dim] or [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, seq_len, head_dim] or [batch, num_heads, seq_len, head_dim]
        v: Value tensor [batch, seq_len, head_dim] or [batch, num_heads, seq_len, head_dim]
        device_id: CUDA device ID
        scale: Attention scale factor (1/sqrt(head_dim)). If None, auto-calculates
    
    Returns:
        AttentionProfile with stage breakdown + optimization hints
    """
    profiler = GPUProfiler(device_id=device_id)
    
    # Ensure tensors are on correct device
    device = torch.device(f'cuda:{device_id}')
    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    
    # Auto-calculate scale if not provided
    if scale is None:
        if len(q.shape) == 3:
            head_dim = q.shape[-1]
        else:
            head_dim = q.shape[-1]
        scale = 1.0 / (head_dim ** 0.5)
    
    # Stage 1: Q @ K.T
    def qk_matmul():
        return torch.matmul(q, k.transpose(-2, -1))
    
    qk_result, qk_profile = profiler.profile_with_result(qk_matmul)
    qk_time_ms = qk_profile['runtime_ms']
    qk_regime = RegimeType(qk_profile['regime'])
    qk_flops_util = qk_profile.get('flops_utilization', 0.0)
    
    # Stage 2: Scale and Softmax
    def softmax_op():
        scaled = qk_result * scale
        return torch.softmax(scaled, dim=-1)
    
    softmax_result, softmax_profile = profiler.profile_with_result(softmax_op)
    softmax_time_ms = softmax_profile['runtime_ms']
    softmax_regime = RegimeType(softmax_profile['regime'])
    softmax_bw_util = softmax_profile.get('bandwidth_utilization', 0.0)
    
    # Stage 3: @ V
    def v_matmul():
        return torch.matmul(softmax_result, v)
    
    v_result, v_profile = profiler.profile_with_result(v_matmul)
    v_time_ms = v_profile['runtime_ms']
    v_regime = RegimeType(v_profile['regime'])
    v_flops_util = v_profile.get('flops_utilization', 0.0)
    
    total_time_ms = qk_time_ms + softmax_time_ms + v_time_ms
    
    # Identify bottleneck stage
    stage_times = {
        'qk_matmul': qk_time_ms,
        'softmax': softmax_time_ms,
        'v_matmul': v_time_ms,
    }
    bottleneck_stage = max(stage_times, key=stage_times.get)
    
    # Check FlashAttention compatibility
    # FlashAttention helps when:
    # 1. Softmax is the bottleneck (memory-bound)
    # 2. Sequence length is long enough to benefit (>128)
    # 3. Head dimension is suitable (16, 32, 64, 128)
    flash_attention_compatible = False
    if bottleneck_stage == 'softmax' and softmax_regime == RegimeType.MEMORY_BOUND:
        seq_len = q.shape[-2] if len(q.shape) == 3 else q.shape[-2]
        head_dim = q.shape[-1] if len(q.shape) == 3 else q.shape[-1]
        
        if seq_len > 128 and head_dim in [16, 32, 64, 128]:
            flash_attention_compatible = True
    
    return AttentionProfile(
        qk_matmul_regime=qk_regime,
        softmax_regime=softmax_regime,
        v_matmul_regime=v_regime,
        total_time_ms=total_time_ms,
        qk_time_ms=qk_time_ms,
        softmax_time_ms=softmax_time_ms,
        v_time_ms=v_time_ms,
        bottleneck_stage=bottleneck_stage,
        flash_attention_compatible=flash_attention_compatible,
        qk_flops_utilization=qk_flops_util,
        v_flops_utilization=v_flops_util,
        softmax_bandwidth_utilization=softmax_bw_util,
    )


def get_attention_suggestions(profile: AttentionProfile) -> str:
    """Get optimization suggestions based on attention profile."""
    suggestions = []
    
    if profile.bottleneck_stage == "softmax":
        if profile.flash_attention_compatible:
            suggestions.append("Bottleneck: Softmax (memory-bound)")
            suggestions.append("→ Use FlashAttention for 3-5x speedup")
            suggestions.append("  Install: pip install flash-attn")
        else:
            suggestions.append("Bottleneck: Softmax (memory-bound)")
            suggestions.append("→ Consider: chunked attention or gradient checkpointing")
    elif profile.bottleneck_stage == "qk_matmul":
        if profile.qk_matmul_regime == RegimeType.COMPUTE_BOUND:
            suggestions.append("Bottleneck: QK matmul (compute-bound)")
            suggestions.append("→ Already optimized, consider larger batch size")
        else:
            suggestions.append("Bottleneck: QK matmul")
            suggestions.append("→ Consider: fused attention kernels")
    elif profile.bottleneck_stage == "v_matmul":
        if profile.v_matmul_regime == RegimeType.COMPUTE_BOUND:
            suggestions.append("Bottleneck: V matmul (compute-bound)")
            suggestions.append("→ Already optimized")
        else:
            suggestions.append("Bottleneck: V matmul")
            suggestions.append("→ Consider: optimized attention implementations")
    
    return "\n".join(suggestions) if suggestions else "No specific suggestions"


def profile_transformer_block(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device_id: int = 0
) -> dict:
    """
    Profile a transformer block (attention + FFN).
    
    This is a higher-level function that profiles the entire transformer block.
    For detailed attention profiling, use profile_attention().
    
    Args:
        model: Transformer block module
        input_tensor: Input tensor
        device_id: CUDA device ID
    
    Returns:
        Dictionary with profiling results
    """
    profiler = GPUProfiler(device_id=device_id)
    device = torch.device(f'cuda:{device_id}')
    
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    def forward_pass():
        with torch.no_grad():
            return model(input_tensor)
    
    result, analysis = profiler.profile_with_result(forward_pass)
    
    return {
        'total_time_ms': analysis['runtime_ms'],
        'regime': analysis['regime'],
        'flops_utilization': analysis.get('flops_utilization', 0.0),
        'bandwidth_utilization': analysis.get('bandwidth_utilization', 0.0),
        'memory': analysis.get('memory', {}),
    }

