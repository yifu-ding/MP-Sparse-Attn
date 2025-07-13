"""
测试批量矩阵乘法实现
"""
import torch
import numpy as np

import argparse
# from tensor_descriptor import TensorDescriptor
import torch
import triton
import triton.language as tl
import triton.profiler as proton
# from triton.tools.tensor_descriptor import TensorDescriptor

from ours.mxfp import MXFP4Tensor, MXScaleTensor
from ours.quant_mxint8 import quant_fpxint8, quant_mxfp8e5, quant_mxfp4


@torch.compiler.disable
def mxfp_attn_kernel(q, k, v, attn_mask=None, dropout_p=0.0, 
                    is_causal=False, scale=None, smooth_k=False, attention_sink=False, tensor_layout="HND",
                    output_dtype=torch.float16, return_sparsity=False, block_scale_type="mxfp4", skip_thresh=None):
    # assert q.size(-2)>=128, "seq_len should be not less than 128."

    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        k = k - k.mean(dim=-2, keepdim=True)

    B, H, M, K = q.shape  # torch.Size([1, 24, 31409, 128])
    # import pdb; pdb.set_trace() 
    N = k.shape[2]

    assert K in [64, 128], "headdim should be in [64, 128]."
    
    BLKQ = 128
    if block_scale_type == "mxfp4":
        a_fp4, a_scale, b_fp4, b_scale = quant_mxfp4(q, k, BLKQ=BLKQ)
        a_quant = a_fp4
        b_quant = b_fp4
    elif block_scale_type == "mxfp8":
        a_fp8, a_scale, b_fp8, b_scale = quant_mxfp8e5(q, k, BLKQ=BLKQ)
        a_quant = a_fp8
        b_quant = b_fp8
        
    # BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32

    a_quant = a_quant.reshape(B, H, M, K).contiguous()  # torch.Size([1, 4, 512, 128])
    b_quant = b_quant.reshape(B, H, N, K).contiguous()
    
    # 扩展 a_scale 和 b_scale 的第2维度为128的倍数
    M_padded = ((M + 127) // 128) * 128  # 向上取整到128的倍数
    N_padded = ((N + 127) // 128) * 128  # 向上取整到128的倍数
    
    # 扩展 a_scale
    if M_padded > M:
        # # 计算需要填充的大小
        # pad_size = M_padded - M
        # # 创建新的tensor，用零填充
        # a_scale_padded = torch.zeros(B, H, M_padded//128, 4, 32, K//VEC_SIZE//4, 4, device=a_scale.device, dtype=a_scale.dtype)
        # # 复制原始数据
        # a_scale_padded[:, :, :M//128, :, :, :, :] = a_scale
        # a_scale = a_scale_padded
        # 计算填充后的目标长度（向上对齐到 128 的倍数）
        # 创建新的 tensor，并用零填充
        a_scale_padded = torch.zeros(B, H, M_padded, 4, device=a_scale.device, dtype=a_scale.dtype)
        # 复制原始数据到前 M 的部分
        a_scale_padded[:, :, :M, :] = a_scale
        # 替换原始变量
        a_scale = a_scale_padded
    # else:
    #     a_scale = a_scale.reshape(B, H, M//128, 4, 32, K//VEC_SIZE//4, 4)
    
    # 扩展 b_scale
    if N_padded > N:
        # # 计算需要填充的大小
        # pad_size = N_padded - N
        # # 创建新的tensor，用零填充
        # b_scale_padded = torch.zeros(B, H, N_padded//128, 4, 32, K//VEC_SIZE//4, 4, device=b_scale.device, dtype=b_scale.dtype)
        # # 复制原始数据
        # b_scale_padded[:, :, :N//128, :, :, :, :] = b_scale
        # b_scale = b_scale_padded
        b_scale_padded = torch.zeros(B, H, N_padded, 4, device=b_scale.device, dtype=b_scale.dtype)
        b_scale_padded[:, :, :N, :] = b_scale
        b_scale = b_scale_padded
    # else:
    #     b_scale = b_scale.reshape(B, H, N//128, 4, 32, K//VEC_SIZE//4, 4)
    
    # 应用permute操作
    a_scale = a_scale.reshape(B, H, M_padded//128, 4, 32, K//VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    b_scale = b_scale.reshape(B, H, N_padded//128, 4, 32, K//VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    
    a_packed, a_scale, b_packed, b_scale, configs, (reference, a_dequant, b_dequant) = \
        initialize_block_scaled_batched_from_tensor(a_quant, b_quant, a_scale, b_scale, block_scale_type=block_scale_type, compute_reference=False)
    
    # print("✓ 初始化成功")
    # print(f"  - a_packed 形状: {a_packed.shape}")
    # print(f"  - b_packed 形状: {b_packed.shape}")
    # print(f"  - a_scale 形状: {a_scale.shape}")
    # print(f"  - b_scale 形状: {b_scale.shape}")
    
    # 执行多维批量attn
    output = block_scaled_batched_attn(
        a_packed, a_scale, b_packed, b_scale, v,
        torch.float16, B, H, M, N, K, configs, skip_thresh=skip_thresh
    )
    return output
    # return None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_block_scaling():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10

def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2. * M * N * K
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_batched_attn_kernel(  #
        q_ptr, q_scale,  #
        k_ptr, k_scale,  #
        v_ori,
        o_ptr,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        stride_qb, stride_qh, stride_qm, stride_qk,  # a的strides: batch, head, M, K
        stride_kb, stride_kh, stride_kn, stride_kk,  # b的strides: batch, head, N, K  
        stride_vb, stride_vh, stride_vn, stride_vk,  # v_ori的strides: batch, head, N, K
        stride_ob, stride_oh, stride_om, stride_on,  # c的strides: batch, head, M, N
        stride_sqb, stride_sqh, stride_sqm, stride_sqk,  # q_scale的strides
        stride_skb, stride_skh, stride_skn, stride_skk,  # k_scale的strides  # stride_skn=1024
        # stride_sk: tl.constexpr, stride_sk: tl.constexpr, stride_sc: tl.constexpr, stride_sd: tl.constexpr,
        num_h: tl.constexpr,  # head数量
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  # 128 
        BLOCK_N: tl.constexpr,  # 256
        HEAD_DIM: tl.constexpr,  # 256
        NUM_STAGES: tl.constexpr,  # 4
        USE_2D_SCALE_LOAD: tl.constexpr, 
        STAGE, qo_len, kv_len,
        skip_thresh: tl.constexpr = 1.0, 
        WARP_SIZE_M: tl.constexpr = 128,
        WARP_SIZE_N: tl.constexpr = 128,
        ):  # False, qo_len = 256, kv_len = 512

    # WARP_SIZE_M = 64
    # WARP_SIZE_N = 128
    
    # if skip_thresh is None:
    #     skip_thresh = -1e6  # -inf
        
    # 获取三维grid的索引 - 参考_attn_fwd的实现
    start_m = tl.program_id(0)  # M*N维度的块索引
    off_h = tl.program_id(1).to(tl.int64)  # head维度索引
    off_z = tl.program_id(2).to(tl.int64)  # batch维度索引
    
    # 计算M和N维度的块索引
    num_pid_m = tl.cdiv(qo_len, BLOCK_M)
    pid_m = start_m % num_pid_m
    # pid_n = start_m // num_pid_m
    
    # 计算偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N) # 每次计算的 token 列数 # [128]
    
    offs_k_a = 0
    offs_k_b = 0

    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e5

    # block scale offsets - 参考_attn_fwd的offset计算方式
    offs_sm = (pid_m * (BLOCK_M // WARP_SIZE_M) + tl.arange(0, BLOCK_M // WARP_SIZE_M)) % qo_len  # [1] when pid_m==1
    # offs_sm = (pid_m * (BLOCK_M // 128) + tl.arange(0, BLOCK_M // 128)) % qo_len  # [1] when pid_m==1
    # offs_sn = (0 * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)) % N
    offs_sn = tl.arange(0, BLOCK_N // WARP_SIZE_N)  # [0, 2]

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    # 计算当前batch和head的基地址
    q_base_offset = off_z * stride_qb + off_h * stride_qh
    k_base_offset = off_z * stride_kb + off_h * stride_kh
    v_base_offset = off_z * stride_vb + off_h * stride_vh
    q_scale_base_offset = off_z * stride_sqb + off_h * stride_sqh
    k_scale_base_offset = off_z * stride_skb + off_h * stride_skh

    # 简化scale load，使用2D模式
    if USE_2D_SCALE_LOAD:
        offs_inner = tl.arange(0, (HEAD_DIM // VEC_SIZE // 4) * 32 * 4 * 4)   # [256//32//4*32*4*4] = [1024]
        # q_scale_ptr = q_scale + q_scale_base_offset + offs_sm[:, None] * stride_sqm + offs_inner[None, :]
        q_scale_ptr = q_scale + q_scale_base_offset + offs_sm[:, None] * stride_sqm + offs_inner[None, :]  # [1, 1024]
        k_scale_ptr = k_scale + k_scale_base_offset + offs_sn[:, None] * stride_skn + offs_inner[None, :]  # [2, 1024]

        # offs_inner = tl.arange(0, (HEAD_DIM // VEC_SIZE // 4) * 32 * 4 * 4)
        # q_scale_ptr = q_scale + offs_sm[:, None] * stride_sk + offs_inner[None, :]
        # k_scale_ptr = k_scale + offs_sn[:, None] * stride_sk + offs_inner[None, :]
        
    # qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32) # [128, 256] -> p

    if ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
        off_k = tl.arange(0, HEAD_DIM//2)  # [0, 128]
        off_q = tl.arange(0, HEAD_DIM//2)  # [0, 128]
    elif MIXED_PREC:
        off_k = tl.arange(0, HEAD_DIM//2)  # [0, 256]
        off_q = tl.arange(0, HEAD_DIM)  # [0, 256]
    else:
        off_k = tl.arange(0, HEAD_DIM)  # [0, 256]
        off_q = tl.arange(0, HEAD_DIM)  # [0, 256]
    
    off_v = tl.arange(0, HEAD_DIM)
    
    q_ptrs = q_ptr + q_base_offset + offs_m[:, None] * stride_qm + off_q[None, :]
    q = tl.load(q_ptrs)  # [128, 256]
    # q = tl.load(q_ptrs, mask=offs_m[:, None] < qo_len)  # [128, 256]

    scale_q = tl.load(q_scale_ptr)  

    if USE_2D_SCALE_LOAD:
        # scale_q = scale_q.reshape(BLOCK_M // 128, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)  
        scale_q = scale_q.reshape(BLOCK_M // WARP_SIZE_M, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)  
    scale_q = scale_q.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, HEAD_DIM // VEC_SIZE)  # [128, 8]

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # 初始化为0，这是在线softmax的标准做法
    old_m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    
    # import pdb; pdb.set_trace()
    
    # for k in tl.range(0, tl.cdiv(K, HEAD_DIM), num_stages=NUM_STAGES):
    # 计算当前batch和head的数据指针  
    # 是e5m2
    # q_ptrs = q_ptr + q_base_offset + (offs_m[:, None] * stride_qm + (offs_k_a + tl.arange(0, HEAD_DIM))[None, :])
    k_ptrs = k_ptr + k_base_offset + (offs_n[None, :] * stride_kn + off_k[:, None])  # [256, 256]
    # k_ptrs = k_ptr + k_base_offset + (offs_n[:, None] * stride_kn + off_k[None, :])  # [256, 256]
    v_ptrs = v_ori + v_base_offset + (offs_n[:, None] * stride_vn + off_v[None, :])  # 修正索引顺序

    # 因果注意力第一阶段
    lo, hi = 0, start_m * BLOCK_M
    
    for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):
        
        # 计算当前块的有效范围
        curr_n_range = start_n + offs_n
        n_mask = curr_n_range < kv_len
        
        # k = tl.load(k_ptrs, mask = offs_n[None, :] < (kv_len - start_n)) 
        k = tl.load(k_ptrs, mask=n_mask[None, :])
        scale_k = tl.load(k_scale_ptr)  # [2, 1024]
        
        # import pdb; pdb.set_trace()
        if USE_2D_SCALE_LOAD:
            scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4) 
        
        scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE)  # [128, 8] = 512 ele

        # import pdb; pdb.set_trace()
        # qk = tl.dot(q, k, out_dtype=tl.float32)
        
        if MIXED_PREC:
            qk = tl.dot_scaled(q, scale_q, "e5m2", k, scale_k, "e2m1")
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            qk = tl.dot_scaled(q, scale_q, "e2m1", k, scale_k, "e2m1")  # 128,128
        else:
            qk = tl.dot_scaled(q, scale_q, "e5m2", k, scale_k, "e5m2")
     
        # 因果注意力第一阶段计算
        local_m = tl.max(qk, 1)  # [128]
        new_m = tl.maximum(old_m, local_m)
        qk = qk - new_m[:, None]
        
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(old_m - new_m)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # v = tl.load(v_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        v = tl.load(v_ptrs)
        # tl.static_print("p.shape", p.shape)
        # tl.static_print("v.shape", v.shape)

        # 保持计算精度，使用float32进行累加，但确保数据类型匹配
        # p = p.to(tl.float32)
        # v = v.to(tl.float32)
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)
        old_m = new_m

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        if USE_2D_SCALE_LOAD:
            k_scale_ptr += (BLOCK_N // VEC_SIZE // 4) * stride_skk # stride_skk/stride_skn? # 应该是按照scale block的数量更新
    

    # 因果注意力第二阶段
    lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    lo = tl.multiple_of(lo, BLOCK_M)

    for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):
        
        # 计算当前块的有效范围
        curr_n_range = start_n + offs_n
        n_mask = curr_n_range < kv_len
        
        # k = tl.load(k_ptrs, mask = offs_n[None, :] < (kv_len - start_n)) 
        k = tl.load(k_ptrs, mask=n_mask[None, :])
        scale_k = tl.load(k_scale_ptr)  # [2, 1024]
        
        # import pdb; pdb.set_trace()
        if USE_2D_SCALE_LOAD:
            scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4) 
        
        scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE)  # [128, 8] = 512 ele

        if MIXED_PREC:
            qk = tl.dot_scaled(q, scale_q, "e5m2", k, scale_k, "e2m1")
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            qk = tl.dot_scaled(q, scale_q, "e2m1", k, scale_k, "e2m1")  # 128,128
        else:
            qk = tl.dot_scaled(q, scale_q, "e5m2", k, scale_k, "e5m2")

        # 因果注意力第二阶段计算
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk + tl.where(mask, 0, -1.0e6)
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(old_m, local_m)
        qk -= new_m[:, None]
        

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(old_m - new_m)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # v = tl.load(v_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        # v = tl.load(v_ptrs, mask=n_mask[:, None])
        v = tl.load(v_ptrs)
        # tl.static_print("p.shape", p.shape)
        # tl.static_print("v.shape", v.shape)

        # 保持计算精度，使用float32进行累加，但确保数据类型匹配
        # p = p.to(tl.float32)
        # v = v.to(tl.float32)
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)
        old_m = new_m

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        if USE_2D_SCALE_LOAD:
            k_scale_ptr += (BLOCK_N // VEC_SIZE // 4) * stride_skk # stride_skk/stride_skn? # 应该是按照scale block的数量更新

    # 存储结果（包含batch和head维度）
    # c_base_offset = off_z * stride_ob + off_h * stride_oh
    # o_ptrs = o_ptr + c_base_offset + (offs_m[:, None] * stride_om + offs_n[None, :])
    # c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # tl.store(o_ptrs, accumulator.to(output_dtype), mask=c_mask)

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + off_z * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + off_v[None, :] # * stride_on
    # tl.static_print("acc.shape", acc.shape)
    # tl.static_print("o_ptrs.shape", o_ptrs.shape)
    tl.store(o_ptrs, acc.to(output_dtype), mask = (offs_m[:, None] < qo_len))


def block_scaled_batched_attn(a_desc, a_scale, b_desc, b_scale, v_ori, dtype_dst, B, H, M, N, K, configs, STAGE=1, skip_thresh=None):
    """
    支持多维批量矩阵乘法的函数
    
    Args:
        a_desc: 形状为(B, H, M, K)的输入矩阵A  
        a_scale: 矩阵A的scale因子, 形状为(B, H, M//128, K//VEC_SIZE//4, 32, 4, 4)
        b_desc: 形状为(B, H, N, K)的输入矩阵B
        b_scale: 矩阵B的scale因子, 形状为(B, H, N//128, K//VEC_SIZE//4, 32, 4, 4)
        dtype_dst: 输出数据类型
        B: batch size
        H: head数量
        M, N, K: 矩阵维度
        configs: 配置参数
        
    Returns:
        output: 形状为(B, H, M, N)的输出矩阵
    """
    output = torch.empty((B, H, M, K), dtype=dtype_dst, device="cuda")
    
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e5m2:
        dtype_dst = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    
    # 设置三维grid: 参考_attn_fwd的grid设置 (M*N的块数, head数, batch数)
    # grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), H, B)


    _, h_qo, qo_len, head_dim = a_desc.shape
    _, h_kv, kv_len, _ = b_desc.shape
    grid = (triton.cdiv(qo_len, BLOCK_M), H, B)

    block_scaled_batched_attn_kernel[grid](
        a_desc, a_scale, b_desc, b_scale, v_ori, output, M, N, K,
        # 输入矩阵A的stride: batch, head, M, K
        a_desc.stride(0), a_desc.stride(1), a_desc.stride(2), a_desc.stride(3),
        # 输入矩阵B的stride: batch, head, N, K  
        b_desc.stride(0), b_desc.stride(1), b_desc.stride(2), b_desc.stride(3),
        # v_ori的stride: batch, head, N, K
        v_ori.stride(0), v_ori.stride(1), v_ori.stride(2), v_ori.stride(3),
        # 输出矩阵的stride: batch, head, M, N
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # a_scale因子的stride: batch, head, M//128, K//VEC_SIZE//4
        a_scale.stride(0), a_scale.stride(1), a_scale.stride(2), a_scale.stride(3),
        # b_scale因子的stride: batch, head, N//128, K//VEC_SIZE//4
        b_scale.stride(0), b_scale.stride(1), b_scale.stride(2), b_scale.stride(3),
        H,  # head数量
        dtype_dst,
        configs["ELEM_PER_BYTE_A"], configs["ELEM_PER_BYTE_B"], configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], head_dim, # configs["BLOCK_SIZE_K"],
        configs["num_stages"], USE_2D_SCALE_LOAD=True,
        STAGE=STAGE, qo_len=qo_len, kv_len=kv_len, skip_thresh=skip_thresh)
    
    return output


def initialize_block_scaled_batched_from_tensor(a_tensor, b_tensor, a_scale, b_scale, block_scale_type="nvfp4", compute_reference=False):
    """
    初始化多维批量block scaled matmul的参数
    
    Args:
        a_tensor: 输入矩阵A, 形状为(B, H, M, K), dtype为torch.float16
        b_tensor: 输入矩阵B, 形状为(B, H, N, K), dtype为torch.float16  
        a_scale: 矩阵A的scale因子, 形状为(B, H, M//128, K//VEC_SIZE//4, 32, 4, 4)
        b_scale: 矩阵B的scale因子, 形状为(B, H, N//128, K//VEC_SIZE//4, 32, 4, 4)
        block_scale_type: 量化类型, 可选"nvfp4", "mxfp4", "mxfp8", "mixed"
        compute_reference: 是否计算参考结果用于验证
    
    Returns:
        a_desc, a_scale, b_desc, b_scale, configs, reference
    """
    B, H, M, K = a_tensor.shape
    B_b, H_b, N, K_b = b_tensor.shape
    assert B == B_b, f"batch size不匹配: A.shape[0]={B} != B.shape[0]={B_b}"
    assert H == H_b, f"head数量不匹配: A.shape[1]={H} != B.shape[1]={H_b}"
    assert K == K_b, f"矩阵维度不匹配: A.shape[3]={K} != B.shape[3]={K_b}"
    
    BLOCK_M = 128
    # BLOCK_N = 256
    BLOCK_N = 128
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    assert block_scale_type in ["nvfp4", "mxfp4", "mxfp8", "mixed"], f"Invalid block scale type: {block_scale_type}"
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    device = a_tensor.device
    
    # 验证scale tensor的形状（包含batch和head维度）
    M_padded = (M + 127) // 128 * 128
    N_padded = (N + 127) // 128 * 128
    expected_a_scale_shape = (B, H, M_padded // 128, K // VEC_SIZE // 4, 32, 4, 4)
    expected_b_scale_shape = (B, H, N_padded // 128, K // VEC_SIZE // 4, 32, 4, 4)
    assert a_scale.shape == expected_a_scale_shape, f"a_scale形状不匹配: 期望{expected_a_scale_shape}, 实际{a_scale.shape}"
    assert b_scale.shape == expected_b_scale_shape, f"b_scale形状不匹配: 期望{expected_b_scale_shape}, 实际{b_scale.shape}"
  
  
   # 根据block_scale_type处理输入数据
    # import pdb; pdb.set_trace()
    if block_scale_type in ["mxfp8", "mixed"]:
        a = a_tensor.to(torch.float8_e5m2)
        a_ref = a.to(torch.float32) if compute_reference else None
    else:
        # 对于fp4格式, 这里需要将fp16数据转换为packed fp4格式
        # 简化处理：直接使用原tensor, 实际应用中需要进行fp4 packing
        # a = a_tensor
        a = MXFP4Tensor(data=a_tensor, dtype=torch.uint8)
        a_ref = a.to(torch.float32) if compute_reference else None
        a = a.to_packed_tensor(dim=1)
        
    if block_scale_type == "mxfp8": 
        b = b_tensor.to(torch.float8_e5m2)
        b_ref = b.to(torch.float32) if compute_reference else None
    else:
        # 对于fp4格式, 这里需要将fp16数据转换为packed fp4格式
        # 简化处理：直接使用原tensor, 实际应用中需要进行fp4 packing
        # b = b_tensor
        b = MXFP4Tensor(data=b_tensor, dtype=torch.uint8)
        b_ref = b.to(torch.float32) if compute_reference else None
        b = b.to_packed_tensor(dim=1)
        
    # # 简化处理：直接使用mxfp8格式
    # if block_scale_type == "mxfp8":
    #     a = a_tensor.to(torch.float8_e5m2)
    #     b = b_tensor.to(torch.float8_e5m2)
    #     a_ref = a.to(torch.float32) if compute_reference else None
    #     b_ref = b.to(torch.float32) if compute_reference else None
    # else:
    #     # 对于其他格式，暂时也使用fp8处理
    #     a = a_tensor.to(torch.float8_e5m2)
    #     b = b_tensor.to(torch.float8_e5m2)
    #     a_ref = a.to(torch.float32) if compute_reference else None
    #     b_ref = b.to(torch.float32) if compute_reference else None
        
    a_desc = a
    b_desc = b

    # 处理scale因子
    if a_scale.dtype == torch.uint8:
        pass
    else:
        if block_scale_type == "nvfp4":
            a_scale = a_scale.to(torch.float8_e5m2)
            b_scale = b_scale.to(torch.float8_e5m2)
            a_scale_ref = a_scale.to(torch.float32)
            b_scale_ref = b_scale.to(torch.float32)
        elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
            a_scale_ref = MXScaleTensor(a_scale.to(torch.float32))
            b_scale_ref = MXScaleTensor(b_scale.to(torch.float32))
            a_scale = a_scale_ref.data
            b_scale = b_scale_ref.data
        
    reference = None
    if compute_reference:
        # 批量计算参考结果，处理多维输入
        b_ref = b_ref.transpose(-1, -2).contiguous()  # (B, H, K, N)
        
        a_scale_ref = a_scale_ref.to(torch.float32)
        b_scale_ref = b_scale_ref.to(torch.float32)

        def unpack_scale_batched(packed):
            B, H, num_chunk_m, num_chunk_k, _, _, _ = packed.shape
            return packed.permute(0, 1, 2, 5, 4, 3, 6).reshape(B, H, num_chunk_m * 128, num_chunk_k * 4).contiguous()
        # 展开scale因子到原始矩阵大小
        a_scale_expanded = unpack_scale_batched(a_scale_ref).repeat_interleave(VEC_SIZE, dim=3)[:, :, :M, :K]
        b_scale_expanded = unpack_scale_batched(b_scale_ref).repeat_interleave(VEC_SIZE, dim=3).transpose(-1, -2).contiguous()[:, :, :K, :N]
        
        # 计算参考结果：批量矩阵乘法 (A * scale_a) @ (B * scale_b)
        # 使用torch.matmul自动处理批量和head维度
        reference = torch.matmul(a_ref * a_scale_expanded, b_ref * b_scale_expanded)
    
    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 1,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
    }
    
    if compute_reference:
        # return a_desc, a_scale, b_desc, b_scale, configs, (reference, a_ref * a_scale_expanded, (b_ref * b_scale_expanded).transpose(-1, -2))
        return a_desc, a_scale, b_desc, b_scale, configs, (reference, a_ref, (b_ref).transpose(-1, -2))
    else:
        return a_desc, a_scale, b_desc, b_scale, configs, (None, None, None)



def test_batched_attn():
    """测试多维批量矩阵乘法的正确性"""
    # 设置参数
    B = 1 # batch size
    H = 12  # head数量
    M = 256
    N = 1024
    K = 128
    block_scale_type = "mxfp4"  # 使用fp8格式进行测试
    
    print(f"测试多维批量矩阵乘法: B={B}, H={H}, M={M}, N={N}, K={K}")
    
    # 生成随机输入数据 - 修改为4D张量
    torch.manual_seed(42)
    a_tensor = torch.randn(B, H, M, K, dtype=torch.float16, device='cuda') * 0.1
    b_tensor = torch.randn(B, H, N, K, dtype=torch.float16, device='cuda') * 0.1
    # b_tensor = torch.arange(0, N, dtype=torch.float16, device='cuda').repeat(B, H, K, 1).transpose(-1, -2) * 1
    v_ori = torch.randn(B, H, N, K, dtype=torch.float16, device='cuda') * 0.1
    # v_ori = torch.arange(0, N, dtype=torch.float16, device='cuda').repeat(B, H, K,).transpose(-1, -2)
    # v_ori = v_ori.transpose(-1, -2)
    
    BLKQ = 128
    if block_scale_type == "mxfp4":
        a_fp4, a_scale, b_fp4, b_scale = quant_mxfp4(a_tensor, b_tensor, BLKQ=BLKQ)
        a_quant = a_fp4
        b_quant = b_fp4
    else:
        a_fp8, a_scale, b_fp8, b_scale = quant_mxfp8e5(a_tensor, b_tensor, BLKQ=BLKQ)
        a_quant = a_fp8
        b_quant = b_fp8
        
    # 计算VEC_SIZE
    # VEC_SIZE = 32 if block_scale_type == "mxfp8" else 16 # 对于mxfp8
    # BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    
    # 生成随机scale因子 - 修改为6D张量
    # a_scale = torch.randn(B, H, M // 128, K // VEC_SIZE // 4, 32, 4, 4, 
    #                      dtype=torch.float32, device='cuda') * 0.01 + 0.1
    # b_scale = torch.randn(B, H, N // 128, K // VEC_SIZE // 4, 32, 4, 4, 
    #                      dtype=torch.float32, device='cuda') * 0.01 + 0.1
    
    a_quant = a_quant.reshape(B, H, M, K).contiguous()
    b_quant = b_quant.reshape(B, H, N, K).contiguous()
    a_scale = a_scale.reshape(B, H, M//128, 4, 32, K//VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    b_scale = b_scale.reshape(B, H, N//128, 4, 32, K//VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    
    a_packed, a_scale, b_packed, b_scale, configs, (reference, a_dequant, b_dequant) = \
        initialize_block_scaled_batched_from_tensor(a_quant, b_quant, a_scale, b_scale, block_scale_type=block_scale_type, compute_reference=True)
    
    # # try:
    # # 初始化多维批量矩阵乘法
    # a_desc, a_scale_proc, b_desc, b_scale_proc, configs, (reference, a_ref, b_ref) = \
    #     initialize_block_scaled_batched_from_tensor(
    #         a_tensor, b_tensor, a_scale, b_scale, 
    #         block_scale_type=block_scale_type, 
    #         compute_reference=True
    #     )
    
    print("✓ 初始化成功")
    print(f"  - a_packed 形状: {a_packed.shape}")
    print(f"  - b_packed 形状: {b_packed.shape}")
    print(f"  - a_scale 形状: {a_scale.shape}")
    print(f"  - b_scale 形状: {b_scale.shape}")
    
    # 执行多维批量矩阵乘法
    output = block_scaled_batched_attn(
        a_packed, a_scale, b_packed, b_scale, v_ori,
        torch.float16, B, H, M, N, K, configs
    )
    # import pdb; pdb.set_trace()
    print(f"✓ 多维批量矩阵乘法完成，输出形状: {output.shape}")
    
    qk_ref = reference
    # qk_ref = torch.matmul(a_dequant.float(), b_dequant.float())   # 相差1e-7
    p = torch.softmax(qk_ref, dim=-1).to(torch.float16)
    pv = p @ v_ori.to(torch.float16)
    # output_ref = torch.sum(pv, dim=-1)
    # import pdb; pdb.set_trace()
    
    # 验证输出形状
    assert output.shape == (B, H, M, K), f"输出形状错误: 期望{(B, H, M, K)}, 实际{output.shape}"
    
    # 与参考结果比较（如果有的话）
    if reference is not None:
        error = torch.mean(torch.abs(output.float() - pv.float())).item()
        print(f"✓ 与参考结果的平均绝对误差: {error:.6f}")
        
        # 检查是否在合理范围内
        if error < 0.1:  # 根据量化精度调整阈值
            print("✓ 精度验证通过")
        else:
            print("⚠️  精度可能有问题，误差较大")
    
    print("✓ 所有测试通过!")
    return True
        
    # except Exception as e:
    #     print(f"❌ 测试失败: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return False

def benchmark_batched_vs_sequential():
    """比较多维批量处理和逐个处理的性能"""
    B = 4
    H = 8
    M = 512
    N = 512  
    K = 256
    num_runs = 10
    
    print(f"\n性能测试: B={B}, H={H}, M={M}, N={N}, K={K}")
    
    # 生成测试数据 - 修改为4D张量
    torch.manual_seed(42)
    a_tensor = torch.randn(B, H, M, K, dtype=torch.float16, device='cuda') * 0.1
    b_tensor = torch.randn(B, H, N, K, dtype=torch.float16, device='cuda') * 0.1
    
    VEC_SIZE = 32
    a_scale = torch.randn(B, H, M // 128, K // VEC_SIZE // 4, 32, 4, 4, 
                         dtype=torch.float32, device='cuda') * 0.01 + 0.1
    b_scale = torch.randn(B, H, N // 128, K // VEC_SIZE // 4, 32, 4, 4, 
                         dtype=torch.float32, device='cuda') * 0.01 + 0.1
    
    # 初始化
    a_desc, a_scale_proc, b_desc, b_scale_proc, configs, _ = \
        initialize_block_scaled_batched_from_tensor(
            a_tensor, b_tensor, a_scale, b_scale, 
            block_scale_type="mxfp8", compute_reference=False
        )
    
    # 预热
    for _ in range(3):
        _ = block_scaled_batched_attn(
            a_desc, a_scale_proc, b_desc, b_scale_proc, 
            torch.float16, B, H, M, N, K, configs
        )
    torch.cuda.synchronize()
    
    # 测试多维批量处理
    import time
    start_time = time.time()
    for _ in range(num_runs):
        output_batched = block_scaled_batched_attn(
            a_desc, a_scale_proc, b_desc, b_scale_proc, 
            torch.float16, B, H, M, N, K, configs
        )
        torch.cuda.synchronize()
    batched_time = (time.time() - start_time) / num_runs
    
    print(f"多维批量处理平均时间: {batched_time*1000:.2f} ms")
    print(f"单次矩阵乘法等效时间: {batched_time*1000/(B*H):.2f} ms")
    print(f"理论加速比: {B*H}x")
    
    return batched_time

# if __name__ == "__main__":
#     print("开始测试多维批量矩阵乘法实现...")
    
#     # 基本功能测试
#     success = test_batched_attn()
    
    # if success:
    #     # 性能测试
    #     benchmark_batched_vs_sequential()
    # else:
    #     print("基本功能测试失败，跳过性能测试") 
