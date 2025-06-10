"""
优化版本的MXFP4量化函数，解决性能瓶颈问题
"""

import torch
import triton
import triton.language as tl

@triton.jit
def quant_mxfp4_optimized_kernel(Input, Output, Scale, L,
                                stride_iz, stride_ih, stride_in,
                                stride_oz, stride_oh, stride_on,
                                stride_sz, stride_sh, stride_sn,
                                sm_scale,
                                C: tl.constexpr, BLK: tl.constexpr):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    offs_n_32 = tl.arange(0, C//32)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + offs_n[:, None] * stride_sn + offs_n_32[None, :]

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale

    # 每32个元素共享一个scale
    x_reshaped = tl.reshape(x, (BLK, C // 32, 32))
    shared_scale = tl.max(tl.abs(x_reshaped), axis=-1, keep_dims=True)
    shared_scale = shared_scale + 1e-7
    
    # 量化过程
    x_quant = x_reshaped / shared_scale
    
    # 预定义候选值，避免重复计算
    candidates = tl.tensor([0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0], dtype=tl.float32)
    
    # 获取符号位
    S = tl.where(x_quant >= 0, 0, 1)
    abs_x = tl.abs(x_quant)
    
    # 判断是否为0
    is_zero = (abs_x == 0)
    
    # 优化：直接在原始维度上计算，避免大的broadcast
    # 计算与候选值的距离，使用更高效的方法
    best_indices = tl.zeros_like(abs_x, dtype=tl.int32)
    
    # 直接找最近的候选值，避免创建大张量
    for i in tl.static_range(8):
        candidate_val = candidates[i]
        current_error = tl.abs(abs_x - candidate_val)
        
        if i == 0:
            min_error = current_error
            best_indices = tl.zeros_like(abs_x, dtype=tl.int32)
        else:
            # 更新最小误差和索引
            is_better = current_error < min_error
            min_error = tl.where(is_better, current_error, min_error)
            best_indices = tl.where(is_better, i, best_indices)
    
    # 处理平局情况 - 优先选择偶数尾数
    for i in tl.static_range(8):
        candidate_val = candidates[i]
        is_tie = tl.abs(tl.abs(abs_x - candidate_val) - min_error) < 1e-6
        is_even_mantissa = (i % 2) == 0
        should_prefer = is_tie & is_even_mantissa
        best_indices = tl.where(should_prefer, i, best_indices)
    
    # 直接计算E和M，避免额外的reshape
    E_selected = best_indices // 2
    M_selected = best_indices % 2
    
    # 处理0值
    E_selected = tl.where(is_zero, 0, E_selected)
    M_selected = tl.where(is_zero, 0, M_selected)
    
    # 组合成最终的量化值
    x_quant_final = (S << 3) | (E_selected << 1) | M_selected
    x_quant_final = x_quant_final.to(tl.uint8)
    
    # 存储结果
    x_uint8 = tl.reshape(x_quant_final, x.shape)
    tl.store(output_ptrs, x_uint8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, shared_scale.reshape(BLK, C // 32))


def quant_mxfp4_optimized(q, k, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"):
    """
    优化版本的MXFP4量化函数
    """
    q_fp4 = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
    k_fp4 = torch.empty(k.shape, dtype=torch.uint8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_fp4.stride(0), q_fp4.stride(1), q_fp4.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_fp4.stride(0), k_fp4.stride(1), k_fp4.stride(2)
    else:
        raise ValueError(f"未支持的tensor布局: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ, head_dim // 32), 
                         device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK, head_dim // 32), 
                         device=k.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # 量化Q
    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_mxfp4_optimized_kernel[grid](
        q, q_fp4, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ
    )

    # 量化K
    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_mxfp4_optimized_kernel[grid](
        k, k_fp4, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK
    )

    return q_fp4, q_scale, k_fp4, k_scale


# 性能对比函数
def benchmark_quant_performance(q, k, num_runs=10):
    """
    对比原版本和优化版本的性能
    """
    import time
    
    # 预热
    for _ in range(3):
        quant_mxfp4_optimized(q, k)
    
    torch.cuda.synchronize()
    
    # 测试优化版本
    start_time = time.time()
    for _ in range(num_runs):
        q_fp4_opt, q_scale_opt, k_fp4_opt, k_scale_opt = quant_mxfp4_optimized(q, k)
        torch.cuda.synchronize()
    end_time = time.time()
    
    opt_time = (end_time - start_time) / num_runs
    print(f"优化版本平均时间: {opt_time*1000:.2f} ms")
    
    return opt_time 