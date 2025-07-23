"""
测试批量矩阵乘法实现
"""
from asyncio import FastChildWatcher
from sympy.logic.boolalg import false
import torch
import numpy as np

import argparse
import torch
import triton
import triton.language as tl
import triton.profiler as proton

from ours.mxfp import MXFP4Tensor, MXScaleTensor
from ours.quant_kernels import quant_fpxint8, quant_mxfp8e5, quant_mxfp4, quant_nvfp4
import os

DEBUG_MODE = os.environ.get('TRITON_INTERPRET', '0') == '1'

@torch.compiler.disable
def mxfp_attn_kernel(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=False, attention_sink=False, tensor_layout="HND",
                    output_dtype=torch.bfloat16, return_sparsity=False, block_scale_type="mxfp4", return_quant_tensor=False, dual_scale=False,\
                    save_qk=False, quant_granularity="tokenwise"):

    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        k = k - k.mean(dim=-2, keepdim=True)

    B, H, M, K = q.shape 
    N = k.shape[2]
    qo_len = M
    kv_len = N

    assert K in [128], "headdim should be in [128]."

    BLKQ = 128
    BLKK = 128
    ret_dict = None
    a_scale_nvfp4, b_scale_nvfp4 = None, None
    if block_scale_type == "mxfp4":
        pack_along_lastdim = False  # True: kernel fusion support for pack fp4 tensor into uint8 tensor
        a_fp4, a_scale, b_fp4, b_scale, a_scale_2, b_scale_2 = quant_mxfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            pack_along_lastdim=pack_along_lastdim, dual_scale=dual_scale, quant_granularity=quant_granularity)
        if return_quant_tensor:
            ret_dict = {
                "a_fp4": a_fp4,
                "a_scale": a_scale,
                "b_fp4": b_fp4,
                "b_scale": b_scale,
                "a_scale_2": a_scale_2,
                "b_scale_2": b_scale_2
            }
        if not pack_along_lastdim:
            a_fp4 = MXFP4Tensor(data=a_fp4, dtype=torch.uint8)
            a_fp4 = a_fp4.to_packed_tensor(dim=len(a_fp4.data.shape) - 1)
            b_fp4 = MXFP4Tensor(data=b_fp4, dtype=torch.uint8)
            b_fp4 = b_fp4.to_packed_tensor(dim=len(b_fp4.data.shape) - 1)
        a_quant = a_fp4
        b_quant = b_fp4
    elif "mxfp8" in block_scale_type:
        a_fp8, a_scale, b_fp8, b_scale, a_scale_2, b_scale_2 = quant_mxfp8e5(q, k, BLKQ=BLKQ, BLKK=BLKK)
        a_quant = a_fp8
        b_quant = b_fp8

        if 'diag' in block_scale_type:
            pack_along_lastdim = True
            _, a_scale_nvfp4, _, b_scale_nvfp4, _, _ = quant_nvfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, \
                pack_along_lastdim=pack_along_lastdim, dual_scale=dual_scale, quant_granularity=quant_granularity)

    elif "nvfp4" in block_scale_type:
        pack_along_lastdim = True
        a_fp4, a_scale, b_fp4, b_scale, a_scale_2, b_scale_2 = quant_nvfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            pack_along_lastdim=pack_along_lastdim, dual_scale=dual_scale, quant_granularity=quant_granularity)
        if not pack_along_lastdim:
            a_fp4 = MXFP4Tensor(data=a_fp4, dtype=torch.uint8)
            a_fp4 = a_fp4.to_packed_tensor(dim=len(a_fp4.data.shape) - 1)
            b_fp4 = MXFP4Tensor(data=b_fp4, dtype=torch.uint8)
            b_fp4 = b_fp4.to_packed_tensor(dim=len(b_fp4.data.shape) - 1)
        a_quant = a_fp4
        b_quant = b_fp4
    
    a_quant = a_quant.contiguous()
    b_quant = b_quant.contiguous()
    
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32

    # 扩展 a_scale 和 b_scale 的第2维度为128的倍数
    M_padded = ((M + 127) // 128) * 128 
    N_padded = ((N + 127) // 128) * 128 
    # import pdb; pdb.set_trace()
    # 扩展 a_scale
    if M_padded > M:
        a_scale_padded = torch.zeros(B, H, M_padded, K//VEC_SIZE, device=a_scale.device, dtype=a_scale.dtype)
        a_scale_padded[:, :, :M, :] = a_scale
        a_scale = a_scale_padded

        if 'diag' in block_scale_type:
            a_scale_nvfp4_padded = torch.zeros(B, H, M_padded, K //16, device=a_scale_nvfp4.device, dtype=a_scale_nvfp4.dtype)
            a_scale_nvfp4_padded[:, :, :M, :] = a_scale_nvfp4
            a_scale_nvfp4 = a_scale_nvfp4_padded

    # 扩展 b_scale
    if N_padded > N:
        b_scale_padded = torch.zeros(B, H, N_padded, K//VEC_SIZE, device=b_scale.device, dtype=b_scale.dtype)
        b_scale_padded[:, :, :N, :] = b_scale
        b_scale = b_scale_padded

        if 'diag' in block_scale_type:
            b_scale_nvfp4_padded = torch.zeros(B, H, N_padded, K//16, device=b_scale_nvfp4.device, dtype=b_scale_nvfp4.dtype)
            b_scale_nvfp4_padded[:, :, :N, :] = b_scale_nvfp4
            b_scale_nvfp4 = b_scale_nvfp4_padded

    a_scale = a_scale.reshape(B, H, M_padded//128, 4, 32, K //VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    b_scale = b_scale.reshape(B, H, N_padded//128, 4, 32, K //VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()

    if 'diag' in block_scale_type:
        a_scale_nvfp4 = a_scale_nvfp4.reshape(B, H, M_padded//128, 4, 32, K //16//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
        b_scale_nvfp4 = b_scale_nvfp4.reshape(B, H, N_padded//128, 4, 32, K //16//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()

    BLOCK_M = 128
    # BLOCK_N = 256
    BLOCK_N = 128
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if "mxfp8" in block_scale_type else 2

    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 1,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
    }

    output = block_scaled_batched_attn(
        a_quant, a_scale, b_quant, b_scale, a_scale_2, b_scale_2, v,  is_causal,
        output_dtype, B, H, M, N, K, configs, save_qk=save_qk, dual_scale=dual_scale, 
        quant_granularity=quant_granularity, mp_diag=('diag' in block_scale_type),
        a_scale_nvfp4=a_scale_nvfp4, b_scale_nvfp4=b_scale_nvfp4
    )   
    if return_quant_tensor:
        return output, ret_dict
    else:
        return output

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_block_scaling():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


@triton.jit
def _quant_nvfp4_inner(x, scale_nvfp4, BLK, C): 
    x_reshaped = tl.reshape(x, (BLK, C // 16, 16))   # x_reshaped: [BLK, C//16, 16]
    abs_max = tl.max(tl.abs(x_reshaped), axis=-1)    # shape: [BLK, C//16]
    
    # shared_scale = abs_max / 6.0

    shared_scale_broadcast = tl.broadcast_to(
        tl.reshape(scale_nvfp4, (BLK, C // 16, 1)), (BLK, C // 16, 16)
    )
    x_quant = x_reshaped / shared_scale_broadcast
    x_quant = tl.clamp(x_quant, -6.0, 6.0)

    sign = tl.where(x_quant >= 0, 0, 1)
    abs_x = tl.abs(x_quant)
    exp = tl.where(abs_x >= 2.0, 
                   tl.where(abs_x >= 4.0, 3, 2),
                   tl.where(abs_x >= 1.0, 1, 0))
    bias = 1.0
    norm_x = abs_x / (tl.exp2((exp - bias).to(tl.float32)))
    mantissa = tl.where(exp == 0, tl.where(norm_x > 0.25, 1, 0),
                                  tl.where(norm_x > 1.25, 1, 0))
    x_quant = (sign << 3) | (exp << 1) | mantissa

    x_quant_reshaped = tl.reshape(x_quant, (BLK, (C + 1) // 2, 2))
    low, high = tl.split(x_quant_reshaped)
    packed = (high << 4) | low
    # if DEBUG_MODE: import pdb; pdb.set_trace()
    
    # e = tl.floor(tl.log2(shared_scale))
    # e_biased = e + 127
    # # e_biased_int = e_biased
    # e_biased_clamped = tl.clamp(e_biased, 0, 254) #.to(tl.int32)

    return packed.to(tl.uint8)#, shared_scale.to(tl.float8e4nv)



@triton.jit
def _quant_mxfp4_inner(x, BLK, C): 
    x_reshaped = tl.reshape(x, (BLK, C // 32, 32))   # x_reshaped: [BLK, C//16, 16]
    abs_max = tl.max(tl.abs(x_reshaped), axis=-1)    # shape: [BLK, C//16]
    
    # shared_scale = abs_max / 6.0
    emax_elem = 3
    shared_exp = tl.floor(tl.log2(abs_max)) - emax_elem 
    shared_scale = tl.exp2(shared_exp) 

    shared_scale_broadcast = tl.broadcast_to(
        tl.reshape(shared_scale, (BLK, C // 32, 1)), (BLK, C // 32, 32)
    )
    x_quant = x_reshaped / shared_scale_broadcast
    x_quant = tl.clamp(x_quant, -6.0, 6.0)

    sign = tl.where(x_quant >= 0, 0, 1)
    abs_x = tl.abs(x_quant)
    exp = tl.where(abs_x >= 2.0, 
                   tl.where(abs_x >= 4.0, 3, 2),
                   tl.where(abs_x >= 1.0, 1, 0))
    bias = 1.0
    norm_x = abs_x / (tl.exp2((exp - bias).to(tl.float32)))
    mantissa = tl.where(exp == 0, tl.where(norm_x > 0.25, 1, 0),
                                  tl.where(norm_x > 1.25, 1, 0))
    x_quant = (sign << 3) | (exp << 1) | mantissa

    x_quant_reshaped = tl.reshape(x_quant, (BLK, (C + 1) // 2, 2))
    low, high = tl.split(x_quant_reshaped)
    packed = (high << 4) | low
    # if DEBUG_MODE: import pdb; pdb.set_trace()
    
    e = tl.floor(tl.log2(shared_scale))
    e_biased = e + 127
    # e_biased_int = e_biased
    e_biased_clamped = tl.clamp(e_biased, 0, 254) #.to(tl.int32)

    return packed.to(tl.uint8), e_biased_clamped.to(tl.uint8)

@triton.jit
def quant_mxfp8e5_to_nvfp4(q, k, scale_q, scale_k, scale_q_nvfp4, scale_k_nvfp4, BLK, C):
    # inputs: q, k: fp8e5, scale_q, scale_k: e8
    # outputs: q, k: fp4, scale_q, scale_k: fp8e4

    # q/k: 128*128, scale_q/scale_k: 128*1
    # step 1: q >> scale_q, k >> scale_k
    # import pdb; pdb.set_trace()
    
    q = tl.reshape(q, (BLK, C // 32, 32))#.to(tl.float32)
    scale_q = tl.broadcast_to(tl.reshape(scale_q, (BLK, C//32, 1)), (BLK, C//32, 32))
    # 将 scale_q 转成 float32 再做 2 ** (-scale_q)
    scaling_factor = tl.exp2(scale_q.to(tl.float32) - 127)  # 即 2^{-scale_q}
    q_deq = q * scaling_factor                         # 恢复为反量化的 float32

    # k.shape = [128, 128]
    '''k = tl.reshape(k, (C // 32, 32, BLK))# .to(tl.float32)  # [128, 128] --> [4, 32, 128]
    scale_k = tl.broadcast_to(tl.reshape(scale_k, (C//32, 1, BLK)), (C//32, 32, BLK))'''
    k = tl.reshape(k, (BLK, C // 32, 32))
    scale_k = tl.broadcast_to(tl.reshape(scale_k, (BLK, C//32, 1)), (BLK, C//32, 32))
    scaling_factor = tl.exp2(scale_k.to(tl.float32) - 127)  # 即 2^{-scale_k}
    k_deq = k * scaling_factor                         # 恢复为反量化的 float32

    # step 2: quant q, k to nvfp4
    # q_quant = tl.zeros([BLK, C//2], dtype=tl.uint8)  # [128, 64]
    # k_quant = tl.zeros([BLK, C//2], dtype=tl.uint8)  # [128, 64]
    packed_q = _quant_nvfp4_inner(q_deq, scale_q_nvfp4, BLK, C)
    packed_k = _quant_nvfp4_inner(k_deq, scale_k_nvfp4, BLK, C)

    return packed_q, packed_k# , shared_scale_q, shared_scale_k


    '''q_reshaped = tl.reshape(q_deq, (BLK, C // 16, 16))   # x_reshaped: [BLK, C//16, 16]
    abs_max = tl.max(tl.abs(q_reshaped), axis=-1)    # shape: [BLK, C//16]
    shared_scale_q = abs_max / 6.0
    shared_scale_broadcast = tl.broadcast_to(
        tl.reshape(shared_scale_q, (BLK, C // 16, 1)), (BLK, C // 16, 16)
    )
    q_quant = q_reshaped / shared_scale_broadcast
    q_quant = tl.clamp(q_quant, -6.0, 6.0)
    sign = tl.where(q_quant >= 0, 0, 1)
    abs_q = tl.abs(q_quant)
    exp = tl.where(abs_q >= 2.0, 
                   tl.where(abs_q >= 4.0, 3, 2),
                   tl.where(abs_q >= 1.0, 1, 0))
    bias = 1.0
    norm_q = abs_q / (tl.exp2((exp - bias).to(tl.float32)))
    mantissa = tl.where(exp == 0, tl.where(norm_q > 0.25, 1, 0),
                                  tl.where(norm_q > 1.25, 1, 0))
    q_quant = (sign << 3) | (exp << 1) | mantissa

    q_quant_reshaped = tl.reshape(q_quant, (BLK, (C + 1) // 2, 2))
    low, high = tl.split(q_quant_reshaped)
    packed_q = (high << 4) | low

    k_reshaped = tl.reshape(k_deq, (BLK, C // 16, 16))
    abs_max = tl.max(tl.abs(k_reshaped), axis=-1)
    shared_scale_k = abs_max / 6.0
    shared_scale_broadcast = tl.broadcast_to(
        tl.reshape(shared_scale_k, (BLK, C // 16, 1)), (BLK, C // 16, 16)
    )
    k_quant = k_reshaped / shared_scale_broadcast
    k_quant = tl.clamp(k_quant, -6.0, 6.0)
    sign = tl.where(k_quant >= 0, 0, 1)
    abs_k = tl.abs(k_quant)
    exp = tl.where(abs_k >= 2.0, 
                   tl.where(abs_k >= 4.0, 3, 2),
                   tl.where(abs_k >= 1.0, 1, 0))
    bias = 1.0
    norm_k = abs_k / (tl.exp2((exp - bias).to(tl.float32)))
    mantissa = tl.where(exp == 0, tl.where(norm_k > 0.25, 1, 0),
                                  tl.where(norm_k > 1.25, 1, 0))
    k_quant = (sign << 3) | (exp << 1) | mantissa

    k_quant_reshaped = tl.reshape(k_quant, (BLK, (C + 1) // 2, 2))
    low, high = tl.split(k_quant_reshaped)
    packed_k = (high << 4) | low
    
    if DEBUG_MODE: import pdb; pdb.set_trace()
    '''

    # t1 = tl.zeros([BLK, C//16], dtype=tl.float32) + 1.0 
    # t2 = tl.zeros([BLK, C//16], dtype=tl.float32) + 1.0 
    # return packed_q.to(tl.uint8), packed_k.to(tl.uint8), t1.to(tl.float8e4nv), t2.to(tl.float8e4nv)
    

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
        q_ptr, q_scale, q_scale_2,  #
        k_ptr, k_scale, k_scale_2,  #
        v_ori,
        o_ptr,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        stride_qb, stride_qh, stride_qm, stride_qk,  # a的strides: batch, head, M, K
        stride_kb, stride_kh, stride_kn, stride_kk,  # b的strides: batch, head, N, K
        stride_vb, stride_vh, stride_vn, stride_vk,  # v_ori的strides: batch, head, N, K
        stride_ob, stride_oh, stride_om, stride_on,  # c的strides: batch, head, M, N
        stride_sqb, stride_sqh, stride_sqm, stride_sqk,  # q_scale的strides
        stride_sqb_2, stride_sqh_2, stride_sqm_2,  # q_scale_2的strides
        stride_skb, stride_skh, stride_skn, stride_skk, # k_scale的strides  # stride_skn=1024
        stride_skb_2, stride_skh_2, stride_skn_2,  # k_scale_2的strides
        num_h: tl.constexpr,  # head数量
        num_kv_groups: tl.constexpr,
        output_type: tl.constexpr,  #
        is_causal: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr, 
        BLOCK_N: tl.constexpr, 
        HEAD_DIM: tl.constexpr, 
        NUM_STAGES: tl.constexpr, 
        USE_2D_SCALE_LOAD: tl.constexpr,
        qo_len, kv_len,
        save_qk: tl.constexpr,
        WARP_SIZE_M: tl.constexpr = 128,
        WARP_SIZE_N: tl.constexpr = 128,
        dual_scale: tl.constexpr = False,
        quant_granularity: tl.constexpr = 0,
        # mp_diag: tl.constexpr = False,  # 0: mxfp8, 1: mxfp8+nvfp4 (start_m == start_n)
):  # False, qo_len = 256, kv_len = 512

    start_m = tl.program_id(0)  # M*N维度的块索引
    off_h = tl.program_id(1).to(tl.int64)  # head维度索引
    off_z = tl.program_id(2).to(tl.int64)  # batch维度索引

    # 计算M和N维度的块索引
    num_pid_m = tl.cdiv(qo_len, BLOCK_M)
    pid_m = start_m % num_pid_m

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
    elif output_type == 3:
        output_dtype = tl.bfloat16

    # block scale offsets - 参考_attn_fwd的offset计算方式
    offs_sm = (pid_m * (BLOCK_M // WARP_SIZE_M) + tl.arange(0, BLOCK_M // WARP_SIZE_M)) % qo_len  # [1] when pid_m==1
    offs_sn = tl.arange(0, BLOCK_N // WARP_SIZE_N) % kv_len 

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    # 计算当前batch和head的基地址
    q_base_offset = off_z * stride_qb + off_h * stride_qh
    k_base_offset = off_z * stride_kb + off_h * stride_kh
    v_base_offset = off_z * stride_vb + off_h * stride_vh
    q_scale_base_offset = off_z * stride_sqb + off_h * stride_sqh
    k_scale_base_offset = off_z * stride_skb + off_h * stride_skh
    # q_scale_nvfp4_base_offset = q_scale_base_offset * 2
    # k_scale_nvfp4_base_offset = k_scale_base_offset * 2

    # double quantization scale  
    if dual_scale:  
        if quant_granularity == 0: # blockwise - Q, K
            q_scale_2_offset = (off_z * num_h + off_h) * tl.cdiv(qo_len, BLOCK_M)
            k_scale_2_offset = (off_z * (num_h // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
            q_scale_2_ptr = q_scale_2 + q_scale_2_offset + pid_m
            k_scale_2_ptr = k_scale_2 + k_scale_2_offset
            scale_q_2 = tl.load(q_scale_2_ptr)
        elif quant_granularity == 1: # channelwise - K, blockwise - Q
            q_scale_2_offset = (off_z * num_h + off_h) * tl.cdiv(qo_len, BLOCK_M)
            k_scale_2_offset = off_z * stride_skb_2 + off_h * stride_skh_2
            q_scale_2_ptr = q_scale_2 + q_scale_2_offset + pid_m
            k_scale_2_ptr = k_scale_2 + k_scale_2_offset + tl.arange(0, HEAD_DIM)  # HEAD_DIM
            scale_q_2 = tl.load(q_scale_2_ptr)
            # if off_z == 0 and (off_h == 0 and pid_m == 0):
            #     tl.static_print("k_scale_2_ptr.shape", k_scale_2_ptr.shape)
        elif quant_granularity == 2: # tokenwise - Q, K
            q_scale_2_offset = off_z * stride_sqb_2 + off_h * stride_sqh_2
            k_scale_2_offset = off_z * stride_skb_2 + off_h * stride_skh_2
            q_scale_2_ptr = q_scale_2 + q_scale_2_offset + pid_m * stride_sqm_2 + tl.arange(0, BLOCK_M)[:, None]  # BLOCK_M
            k_scale_2_ptr = k_scale_2 + k_scale_2_offset + tl.arange(0, BLOCK_N)[None, :]  # BLOCK_N
            scale_q_2 = tl.load(q_scale_2_ptr)


    # 简化scale load，使用2D模式
    if USE_2D_SCALE_LOAD:
        offs_inner = tl.arange(0, (HEAD_DIM // VEC_SIZE // 4) * 32 * 4 * 4)
        q_scale_ptr = q_scale + q_scale_base_offset + offs_sm[:, None] * stride_sqm + offs_inner[None, :] 
        k_scale_ptr = k_scale + k_scale_base_offset + offs_sn[:, None] * stride_skn + offs_inner[None, :] 
        
        offs_inner_nvfp4 = tl.arange(0, (HEAD_DIM // 16 // 4) * 32 * 4 * 4)  # [2, 32, 4, 4]
        # if mp_diag: 
        #     q_scale_nvfp4_ptr = q_scale_nvfp4 + q_scale_nvfp4_base_offset + offs_sm[:, None] * (stride_sqm * 2) + offs_inner_nvfp4[None, :] 
        #     k_scale_nvfp4_ptr = k_scale_nvfp4 + k_scale_nvfp4_base_offset + offs_sn[:, None] * (stride_skn * 2) + offs_inner_nvfp4[None, :] 

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)  # [128, 256] -> p

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
    q = tl.load(q_ptrs, mask=offs_m[:, None] < qo_len, other=0.0)
    scale_q = tl.load(q_scale_ptr)  # [1, 512]
    # if mp_diag: 
    #     scale_q_nvfp4 = tl.load(q_scale_nvfp4_ptr)  # [1, 512]

    if USE_2D_SCALE_LOAD:
        scale_q = scale_q.reshape(BLOCK_M // WARP_SIZE_M, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)  # 1, 1, 32, 4, 4
        # if mp_diag: 
        #     scale_q_nvfp4 = scale_q_nvfp4.reshape(BLOCK_M // WARP_SIZE_M, HEAD_DIM // 16 // 4, 32, 4, 4)  # 1, 2, 32, 4, 4

    scale_q = scale_q.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, HEAD_DIM // VEC_SIZE)  # [128, 8]
    # if mp_diag:    
    #     scale_q_nvfp4 = scale_q_nvfp4.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, HEAD_DIM // 16)  # [128, 8]

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0 # 初始化为1
    old_m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # k_ptrs = k_ptr + k_base_offset + (offs_n[None, :] * stride_kn + off_k[:, None]) 
    k_ptrs = k_ptr + k_base_offset + (offs_n[:, None] * stride_kn + off_k[None, :]) 
    v_ptrs = v_ori + v_base_offset + (offs_n[:, None] * stride_vn + off_v[None, :]) 
    
    if is_causal:
        # 因果注意力第一阶段
        lo, hi = 0, start_m * BLOCK_M

        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):

            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len

            # k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)  # [headdim, 130-128] 
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)  # [headdim, 130-128] 
            scale_k = tl.load(k_scale_ptr)  # round1: 128*4 --> k: 128*128, round2: 2*4  k: (130-128)*128 = 2*128
            if dual_scale: scale_k_2 = tl.load(k_scale_2_ptr)

            if USE_2D_SCALE_LOAD:
                scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)  # 1, 2, 32, 4, 4
            scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE)  # 128, 128/16 = 128, 8

            # tl.static_print("mp_diag and (start_m != start_n)=", mp_diag and (start_m != start_n))
            # if mp_diag and (start_m != start_n):
            #     scale_k_nvfp4 = tl.load(k_scale_nvfp4_ptr)  # [1, 512]
            #     if USE_2D_SCALE_LOAD:
            #         scale_k_nvfp4 = scale_k_nvfp4.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // 16 // 4, 32, 4, 4)  # 1, 2, 32, 4, 4
            #     scale_k_nvfp4 = scale_k_nvfp4.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // 16)  # 128, 8
            #     q_nvfp4, k_nvfp4 = quant_mxfp8e5_to_nvfp4(q, k, scale_q, scale_k, scale_q_nvfp4, scale_k_nvfp4, BLOCK_M, HEAD_DIM)
            #     qk = tl.dot_scaled(q_nvfp4, scale_q_nvfp4, "e2m1", k_nvfp4.T, scale_k_nvfp4, "e2m1")# causal - stage1
            #     k_scale_nvfp4_ptr += (HEAD_DIM // 16 // 4) * (stride_skk*2)
            # else:
            if MIXED_PREC:
                # tl.static_print("stage1, mixed prec")
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e2m1")
            elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
                # tl.static_print("stage1, elem_per_byte_a == 2 and elem_per_byte_b == 2")
                qk = tl.dot_scaled(q, scale_q, "e2m1", k.T, scale_k, "e2m1")  # 128,128
            else:
                # tl.static_print("stage1, else")
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
            
            if dual_scale:  
                qk = qk * (scale_q_2 * scale_k_2)
                # saved_qk = qk
            
            # 因果注意力第一阶段计算
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])   
            mask_sum = tl.sum(tl.sum(mask, axis=0))
            if True:
            # if mask_sum > 0:   # 好像是对的，但收益不多
                local_m = tl.max(qk, 1)  # [128]
                new_m = tl.maximum(old_m, local_m)
                qk = qk - new_m[:, None]

                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp2(old_m - new_m)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                
                v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
                p = p.to(tl.float16)
                acc += tl.dot(p, v, out_dtype=tl.float16)
                old_m = new_m

                k_ptrs += BLOCK_N * stride_kn
                v_ptrs += BLOCK_N * stride_vn
                if USE_2D_SCALE_LOAD:
                    k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk
                        
                if dual_scale: 
                    if quant_granularity == 0:
                        k_scale_2_ptr += 1
                    elif quant_granularity == 1:
                        k_scale_2_ptr += HEAD_DIM  # seems right?
                    elif quant_granularity == 2:
                        k_scale_2_ptr += BLOCK_N

        # 因果注意力第二阶段
        saved_qk = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)

        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):

            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len

            # k = tl.load(k_ptrs, mask = offs_n[None, :] < (kv_len - start_n))
            # k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0) 
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            scale_k = tl.load(k_scale_ptr)  # [2, 1024]
            if dual_scale: scale_k_2 = tl.load(k_scale_2_ptr)

            if USE_2D_SCALE_LOAD:  
                scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)
            scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE) 

            # if mp_diag and (start_m == start_n):
            #     scale_k_nvfp4 = tl.load(k_scale_nvfp4_ptr)  # [1, 512]
            #     if USE_2D_SCALE_LOAD:  
            #         scale_k_nvfp4 = scale_k_nvfp4.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // 16 // 4, 32, 4, 4)
            #     scale_k_nvfp4 = scale_k_nvfp4.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // 16)  # 128, 8
            #     q_nvfp4, k_nvfp4 = quant_mxfp8e5_to_nvfp4(q, k, scale_q, scale_k, scale_q_nvfp4, scale_k_nvfp4, BLOCK_M, HEAD_DIM)
            #     qk = tl.dot_scaled(q_nvfp4, scale_q_nvfp4, "e2m1", k_nvfp4.T, scale_k_nvfp4, "e2m1") # causal - stage2
            #     k_scale_nvfp4_ptr += (HEAD_DIM // 16 // 4) * (stride_skk*2)
            # else:
            if MIXED_PREC:
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e2m1")
            elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
                qk = tl.dot_scaled(q, scale_q, "e2m1", k.T, scale_k, "e2m1") 
            else:
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")

            if dual_scale: 
                qk = qk * (scale_q_2 * scale_k_2)

            qk = tl.where(n_mask[None, :], qk, -float("inf"))
            # qk = tl.where(offs_m[:, None] < qo_len, qk, -float("inf"))

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

            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            p = p.to(tl.float16)
            acc += tl.dot(p, v, out_dtype=tl.float16)
            old_m = new_m

            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn
            if USE_2D_SCALE_LOAD:
                # if DEBUG_MODE: import pdb; pdb.set_trace()
                k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk
            if dual_scale:  
                if quant_granularity == 0:
                    k_scale_2_ptr += 1
                elif quant_granularity == 1:
                    k_scale_2_ptr += HEAD_DIM  # seems right?
                elif quant_granularity == 2:
                    k_scale_2_ptr += BLOCK_N
    else:
        # saved_qk = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        lo, hi = 0, kv_len
        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):

            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len

            # k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0) 
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            scale_k = tl.load(k_scale_ptr)  # [2, 1024]
            if dual_scale:  scale_k_2 = tl.load(k_scale_2_ptr)

            if USE_2D_SCALE_LOAD:
                scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)
            scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE) 

            if MIXED_PREC:
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e2m1")
            elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
                qk = tl.dot_scaled(q, scale_q, "e2m1", k.T, scale_k, "e2m1")
            else:
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")

            if dual_scale:  
                qk = qk * ( scale_q_2 * scale_k_2)

            qk = tl.where(n_mask[None, :], qk, -float("inf"))

            # 非因果注意力计算
            qk = qk.to(tl.float32)
            local_m = tl.max(qk, 1) 
            new_m = tl.maximum(old_m, local_m)
            # qk = qk - new_m[:, None]
            qk = qk - tl.expand_dims(new_m, axis=1)

            p = tl.math.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(old_m - new_m)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v = tl.load(v_ptrs, mask=n_mask[:, None])
            p = p.to(tl.float32)
            v = v.to(tl.float32)
            acc += tl.dot(p, v, out_dtype=tl.float32)
            old_m = new_m

            k_ptrs += BLOCK_N * stride_kn    # 4bit: seq*head_dim//2, 8bit: seq*head_dim
            v_ptrs += BLOCK_N * stride_vn    
            if USE_2D_SCALE_LOAD:
                # if DEBUG_MODE: import pdb; pdb.set_trace()
                k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk
                # k_scale_nvfp4_ptr += (HEAD_DIM // 16 // 4) * (stride_skk*2)
            if dual_scale:  
                if quant_granularity == 0:
                    k_scale_2_ptr += 1
                elif quant_granularity == 1:
                    k_scale_2_ptr += HEAD_DIM  # seems right?
                elif quant_granularity == 2:
                    k_scale_2_ptr += BLOCK_N

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + off_z * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + off_v[None, :]  
    tl.store(o_ptrs, acc.to(output_dtype), mask=(offs_m[:, None] < qo_len))



@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_batched_attn_kernel_mp_diag(  #
        q_ptr, q_scale, q_scale_2, q_scale_nvfp4,  #
        k_ptr, k_scale, k_scale_2, k_scale_nvfp4,  #
        v_ori,
        o_ptr,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        stride_qb, stride_qh, stride_qm, stride_qk,  # a的strides: batch, head, M, K
        stride_kb, stride_kh, stride_kn, stride_kk,  # b的strides: batch, head, N, K
        stride_vb, stride_vh, stride_vn, stride_vk,  # v_ori的strides: batch, head, N, K
        stride_ob, stride_oh, stride_om, stride_on,  # c的strides: batch, head, M, N
        stride_sqb, stride_sqh, stride_sqm, stride_sqk,  # q_scale的strides
        stride_sqb_2, stride_sqh_2, stride_sqm_2,  # q_scale_2的strides
        stride_skb, stride_skh, stride_skn, stride_skk, # k_scale的strides  # stride_skn=1024
        stride_skb_2, stride_skh_2, stride_skn_2,  # k_scale_2的strides
        num_h: tl.constexpr,  # head数量
        num_kv_groups: tl.constexpr,
        output_type: tl.constexpr,  #
        is_causal: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr, 
        BLOCK_N: tl.constexpr, 
        HEAD_DIM: tl.constexpr, 
        NUM_STAGES: tl.constexpr, 
        USE_2D_SCALE_LOAD: tl.constexpr,
        qo_len, kv_len,
        save_qk: tl.constexpr,
        WARP_SIZE_M: tl.constexpr = 128,
        WARP_SIZE_N: tl.constexpr = 128,
        dual_scale: tl.constexpr = False,
        quant_granularity: tl.constexpr = 0,
):  # False, qo_len = 256, kv_len = 512

    start_m = tl.program_id(0)  # M*N维度的块索引
    off_h = tl.program_id(1).to(tl.int64)  # head维度索引
    off_z = tl.program_id(2).to(tl.int64)  # batch维度索引

    # 计算M和N维度的块索引
    num_pid_m = tl.cdiv(qo_len, BLOCK_M)
    pid_m = start_m % num_pid_m

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
    elif output_type == 3:
        output_dtype = tl.bfloat16

    # block scale offsets - 参考_attn_fwd的offset计算方式
    offs_sm = (pid_m * (BLOCK_M // WARP_SIZE_M) + tl.arange(0, BLOCK_M // WARP_SIZE_M)) % qo_len  # [1] when pid_m==1
    offs_sn = tl.arange(0, BLOCK_N // WARP_SIZE_N) % kv_len 

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    # 计算当前batch和head的基地址
    q_base_offset = off_z * stride_qb + off_h * stride_qh
    k_base_offset = off_z * stride_kb + off_h * stride_kh
    v_base_offset = off_z * stride_vb + off_h * stride_vh
    q_scale_base_offset = off_z * stride_sqb + off_h * stride_sqh
    k_scale_base_offset = off_z * stride_skb + off_h * stride_skh
    q_scale_nvfp4_base_offset = q_scale_base_offset * 2
    k_scale_nvfp4_base_offset = k_scale_base_offset * 2

    # double quantization scale  
    if dual_scale:  
        if quant_granularity == 0: # blockwise - Q, K
            q_scale_2_offset = (off_z * num_h + off_h) * tl.cdiv(qo_len, BLOCK_M)
            k_scale_2_offset = (off_z * (num_h // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
            q_scale_2_ptr = q_scale_2 + q_scale_2_offset + pid_m
            k_scale_2_ptr = k_scale_2 + k_scale_2_offset
            scale_q_2 = tl.load(q_scale_2_ptr)
        elif quant_granularity == 1: # channelwise - K, blockwise - Q
            q_scale_2_offset = (off_z * num_h + off_h) * tl.cdiv(qo_len, BLOCK_M)
            k_scale_2_offset = off_z * stride_skb_2 + off_h * stride_skh_2
            q_scale_2_ptr = q_scale_2 + q_scale_2_offset + pid_m
            k_scale_2_ptr = k_scale_2 + k_scale_2_offset + tl.arange(0, HEAD_DIM)  # HEAD_DIM
            scale_q_2 = tl.load(q_scale_2_ptr)
            # if off_z == 0 and (off_h == 0 and pid_m == 0):
            #     tl.static_print("k_scale_2_ptr.shape", k_scale_2_ptr.shape)
        elif quant_granularity == 2: # tokenwise - Q, K
            q_scale_2_offset = off_z * stride_sqb_2 + off_h * stride_sqh_2
            k_scale_2_offset = off_z * stride_skb_2 + off_h * stride_skh_2
            q_scale_2_ptr = q_scale_2 + q_scale_2_offset + pid_m * stride_sqm_2 + tl.arange(0, BLOCK_M)[:, None]  # BLOCK_M
            k_scale_2_ptr = k_scale_2 + k_scale_2_offset + tl.arange(0, BLOCK_N)[None, :]  # BLOCK_N
            scale_q_2 = tl.load(q_scale_2_ptr)


    # 简化scale load，使用2D模式
    if USE_2D_SCALE_LOAD:
        offs_inner = tl.arange(0, (HEAD_DIM // VEC_SIZE // 4) * 32 * 4 * 4)
        q_scale_ptr = q_scale + q_scale_base_offset + offs_sm[:, None] * stride_sqm + offs_inner[None, :] 
        k_scale_ptr = k_scale + k_scale_base_offset + offs_sn[:, None] * stride_skn + offs_inner[None, :] 
        
        offs_inner_nvfp4 = tl.arange(0, (HEAD_DIM // 16 // 4) * 32 * 4 * 4)  # [2, 32, 4, 4]
        # if mp_diag: 
        q_scale_nvfp4_ptr = q_scale_nvfp4 + q_scale_nvfp4_base_offset + offs_sm[:, None] * (stride_sqm * 2) + offs_inner_nvfp4[None, :] 
        k_scale_nvfp4_ptr = k_scale_nvfp4 + k_scale_nvfp4_base_offset + offs_sn[:, None] * (stride_skn * 2) + offs_inner_nvfp4[None, :] 

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)  # [128, 256] -> p

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
    q = tl.load(q_ptrs, mask=offs_m[:, None] < qo_len, other=0.0)
    scale_q = tl.load(q_scale_ptr)  # [1, 512]
    # if mp_diag: 
    scale_q_nvfp4 = tl.load(q_scale_nvfp4_ptr)  # [1, 512]

    if USE_2D_SCALE_LOAD:
        scale_q = scale_q.reshape(BLOCK_M // WARP_SIZE_M, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)  # 1, 1, 32, 4, 4
        scale_q_nvfp4 = scale_q_nvfp4.reshape(BLOCK_M // WARP_SIZE_M, HEAD_DIM // 16 // 4, 32, 4, 4)  # 1, 2, 32, 4, 4

    scale_q = scale_q.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, HEAD_DIM // VEC_SIZE)  # [128, 8]
    scale_q_nvfp4 = scale_q_nvfp4.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, HEAD_DIM // 16)  # [128, 8]

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0 # 初始化为1
    old_m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # k_ptrs = k_ptr + k_base_offset + (offs_n[None, :] * stride_kn + off_k[:, None]) 
    k_ptrs = k_ptr + k_base_offset + (offs_n[:, None] * stride_kn + off_k[None, :]) 
    v_ptrs = v_ori + v_base_offset + (offs_n[:, None] * stride_vn + off_v[None, :]) 
    
    if is_causal:
        # 因果注意力第一阶段
        lo, hi = 0, start_m * BLOCK_M

        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):

            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len

            # k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)  # [headdim, 130-128] 
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)  # [headdim, 130-128] 
            scale_k = tl.load(k_scale_ptr)  # round1: 128*4 --> k: 128*128, round2: 2*4  k: (130-128)*128 = 2*128
            if dual_scale: scale_k_2 = tl.load(k_scale_2_ptr)

            if USE_2D_SCALE_LOAD:
                scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)  # 1, 2, 32, 4, 4
            scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE)  # 128, 128/16 = 128, 8

            # tl.static_print("mp_diag and (start_m != start_n)=", mp_diag and (start_m != start_n))
            # if True:
            # if start_m == start_n:
            # tl.static_print("causal - stage1, nvfp4")
                # tl.device_print("start_m = {}", start_m)
                # tl.device_print("start_n = {}", start_n)
                # tl.device_print("start_m == start_n = {}", start_m == start_n)
            
            # qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")

            scale_k_nvfp4 = tl.load(k_scale_nvfp4_ptr)  # [1, 512]
            if USE_2D_SCALE_LOAD:
                scale_k_nvfp4 = scale_k_nvfp4.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // 16 // 4, 32, 4, 4)  # 1, 2, 32, 4, 4
            scale_k_nvfp4 = scale_k_nvfp4.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // 16)  # 128, 8
            q_nvfp4, k_nvfp4 = quant_mxfp8e5_to_nvfp4(q, k, scale_q, scale_k, scale_q_nvfp4, scale_k_nvfp4, BLOCK_M, HEAD_DIM)
            qk = tl.dot_scaled(q_nvfp4, scale_q_nvfp4, "e2m1", k_nvfp4.T, scale_k_nvfp4, "e2m1")# causal - stage1
            k_scale_nvfp4_ptr += (HEAD_DIM // 16 // 4) * (stride_skk)
            # tl.device_print("qk", qk)
            # else:
                # if MIXED_PREC:
                #     tl.static_print("stage1, mixed prec")
                #     qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e2m1")
                # elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
                #     tl.static_print("stage1, elem_per_byte_a == 2 and elem_per_byte_b == 2")
                #     qk = tl.dot_scaled(q, scale_q, "e2m1", k.T, scale_k, "e2m1")  # 128,128
                # else:
                #     tl.static_print("stage1, else")
                # tl.device_print("causal - stage1, mxfp8")
                # tl.device_print("start_m = {}", start_m)
                # tl.device_print("start_n = {}", start_n)
                # tl.device_print("start_m == start_n = {}", start_m == start_n)
                # qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
            
            if dual_scale: qk = qk * (scale_q_2 * scale_k_2)
            
            # 因果注意力第一阶段计算
            # mask = offs_m[:, None] >= (start_n + offs_n[None, :])   
            # mask_sum = tl.sum(tl.sum(mask, axis=0))  # no need, already use lower triangular
            local_m = tl.max(qk, 1)  # [128]
            new_m = tl.maximum(old_m, local_m)
            qk = qk - new_m[:, None]

            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(old_m - new_m)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            p = p.to(tl.float16)
            acc += tl.dot(p, v, out_dtype=tl.float16)
            old_m = new_m

            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn
            if USE_2D_SCALE_LOAD:
                k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk
                    
            if dual_scale: 
                if quant_granularity == 0:
                    k_scale_2_ptr += 1
                elif quant_granularity == 1:
                    k_scale_2_ptr += HEAD_DIM  # seems right?
                elif quant_granularity == 2:
                    k_scale_2_ptr += BLOCK_N

        # 因果注意力第二阶段
        saved_qk = tl.zeros([BLOCK_M, 8], dtype=tl.float32)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)

        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):

            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len

            # k = tl.load(k_ptrs, mask = offs_n[None, :] < (kv_len - start_n))
            # k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0) 
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            scale_k = tl.load(k_scale_ptr)  # [2, 1024]
            if dual_scale: scale_k_2 = tl.load(k_scale_2_ptr)

            if USE_2D_SCALE_LOAD:  
                scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)
            scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE) 

            # if start_m != start_n:
            # # if True:
            #     # tl.static_print("causal - stage2, nvfp4")
            #     # tl.static_print("start_m=", start_m, "start_n=", start_n, "start_m == start_n=", start_m == start_n)
            #     scale_k_nvfp4 = tl.load(k_scale_nvfp4_ptr)  # [1, 512]
            #     if USE_2D_SCALE_LOAD:  
            #         scale_k_nvfp4 = scale_k_nvfp4.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // 16 // 4, 32, 4, 4)
            #     scale_k_nvfp4 = scale_k_nvfp4.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // 16)  # 128, 8
            #     q_nvfp4, k_nvfp4 = quant_mxfp8e5_to_nvfp4(q, k, scale_q, scale_k, scale_q_nvfp4, scale_k_nvfp4, BLOCK_M, HEAD_DIM)
            #     qk = tl.dot_scaled(q_nvfp4, scale_q_nvfp4, "e2m1", k_nvfp4.T, scale_k_nvfp4, "e2m1") # causal - stage2
            #     k_scale_nvfp4_ptr += (HEAD_DIM // 16 // 4) * (stride_skk*2)
            # else:
                # if MIXED_PREC:
                #     qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e2m1")
                # elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
                #     qk = tl.dot_scaled(q, scale_q, "e2m1", k.T, scale_k, "e2m1") 
                # else:
            # tl.static_print("causal - stage2, mxfp8")
                # tl.static_print("start_m=", start_m, "start_n=", start_n, "start_m == start_n=", start_m == start_n)

            qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")

            # scale_k_nvfp4 = tl.load(k_scale_nvfp4_ptr)  # [1, 512]
            # if USE_2D_SCALE_LOAD:
            #     scale_k_nvfp4 = scale_k_nvfp4.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // 16 // 4, 32, 4, 4)  # 1, 2, 32, 4, 4
            # scale_k_nvfp4 = scale_k_nvfp4.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // 16)  # 128, 8
            # q_nvfp4, k_nvfp4 = quant_mxfp8e5_to_nvfp4(q, k, scale_q, scale_k, scale_q_nvfp4, scale_k_nvfp4, BLOCK_M, HEAD_DIM)
            # qk = tl.dot_scaled(q_nvfp4, scale_q_nvfp4, "e2m1", k_nvfp4.T, scale_k_nvfp4, "e2m1")# causal - stage2
            # k_scale_nvfp4_ptr += (HEAD_DIM // 16 // 4) * (stride_skk)
            
            # saved_qk = scale_k_nvfp4.to(tl.float32)
            if dual_scale: qk = qk * (scale_q_2 * scale_k_2)

            qk = tl.where(n_mask[None, :], qk, -float("inf"))
            # qk = tl.where(offs_m[:, None] < qo_len, qk, -float("inf"))

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

            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            p = p.to(tl.float16)
            acc += tl.dot(p, v, out_dtype=tl.float16)
            old_m = new_m

            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn
            if USE_2D_SCALE_LOAD:
                k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk
            if dual_scale:  
                if quant_granularity == 0:
                    k_scale_2_ptr += 1
                elif quant_granularity == 1:
                    k_scale_2_ptr += HEAD_DIM  # seems right?
                elif quant_granularity == 2:
                    k_scale_2_ptr += BLOCK_N
    else:
        # saved_qk = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        lo, hi = 0, kv_len
        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):

            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len

            # k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0) 
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            scale_k = tl.load(k_scale_ptr)  # [2, 1024]
            if dual_scale:  scale_k_2 = tl.load(k_scale_2_ptr)

            if USE_2D_SCALE_LOAD:
                scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)
            scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE) 

            if MIXED_PREC:
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e2m1")
            elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
                qk = tl.dot_scaled(q, scale_q, "e2m1", k.T, scale_k, "e2m1")
            else:
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")

            if dual_scale:  
                qk = qk * ( scale_q_2 * scale_k_2)

            qk = tl.where(n_mask[None, :], qk, -float("inf"))

            # 非因果注意力计算
            qk = qk.to(tl.float32)
            local_m = tl.max(qk, 1) 
            new_m = tl.maximum(old_m, local_m)
            # qk = qk - new_m[:, None]
            qk = qk - tl.expand_dims(new_m, axis=1)

            p = tl.math.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(old_m - new_m)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v = tl.load(v_ptrs, mask=n_mask[:, None])
            p = p.to(tl.float32)
            v = v.to(tl.float32)
            acc += tl.dot(p, v, out_dtype=tl.float32)
            old_m = new_m

            k_ptrs += BLOCK_N * stride_kn    # 4bit: seq*head_dim//2, 8bit: seq*head_dim
            v_ptrs += BLOCK_N * stride_vn    
            if USE_2D_SCALE_LOAD:
                if DEBUG_MODE: import pdb; pdb.set_trace()
                k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk
                k_scale_nvfp4_ptr += (HEAD_DIM // 16 // 4) * (stride_skk)
            if dual_scale:  
                if quant_granularity == 0:
                    k_scale_2_ptr += 1
                elif quant_granularity == 1:
                    k_scale_2_ptr += HEAD_DIM  # seems right?
                elif quant_granularity == 2:
                    k_scale_2_ptr += BLOCK_N

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + off_z * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + off_v[None, :]  
    if save_qk:
        o_ptrs = o_ptr + off_z * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + tl.arange(0, 8)[None, :]  
        tl.store(o_ptrs, saved_qk.to(output_dtype), mask=(offs_m[:, None] < qo_len))
    else:
        tl.store(o_ptrs, acc.to(output_dtype), mask=(offs_m[:, None] < qo_len))



def block_scaled_batched_attn(a_desc, a_scale, b_desc, b_scale, a_scale_2, b_scale_2, v_ori, \
    is_causal, dtype_dst, B, H, M, N, K, configs, save_qk=False, dual_scale=False, quant_granularity="blockwise", \
        mp_diag=0, a_scale_nvfp4=None, b_scale_nvfp4=None):
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
    output = torch.zeros((B, H, M, K), dtype=dtype_dst, device="cuda")

    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e5m2:
        dtype_dst = 2
    elif dtype_dst == torch.bfloat16:
        dtype_dst = 3
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    if quant_granularity == "blockwise":
        quant_granularity = 0
    elif quant_granularity == "channelwise":
        quant_granularity = 1
    elif quant_granularity == "tokenwise":
        quant_granularity = 2
    else:
        raise ValueError(f"Unsupported quant_granularity: {quant_granularity}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]

    # 设置三维grid: 参考_attn_fwd的grid设置 (M*N的块数, head数, batch数)
    # grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), H, B)
    _, h_qo, qo_len, _ = a_desc.shape
    _, h_kv, kv_len, _ = b_desc.shape
    
    grid = (triton.cdiv(qo_len, BLOCK_M), H, B)

    num_kv_groups = h_qo // h_kv
    # import pdb; pdb.set_trace()
    # if dual_scale:
    if mp_diag:
        block_scaled_batched_attn_kernel_mp_diag[grid](
        a_desc, a_scale, a_scale_2, a_scale_nvfp4,
        b_desc, b_scale, b_scale_2, b_scale_nvfp4,
        v_ori, output, M, N, K,
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
        a_scale_2.stride(0), a_scale_2.stride(1), a_scale_2.stride(2),
        # b_scale因子的stride: batch, head, N//128, K//VEC_SIZE//4
        b_scale.stride(0), b_scale.stride(1), b_scale.stride(2), b_scale.stride(3),
        b_scale_2.stride(0), b_scale_2.stride(1), b_scale_2.stride(2),
        h_qo, num_kv_groups,  # head数量
        dtype_dst, is_causal,
        configs["ELEM_PER_BYTE_A"], configs["ELEM_PER_BYTE_B"], configs["VEC_SIZE"],
        # configs["BLOCK_SIZE_K"],
        configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], K,
        configs["num_stages"], USE_2D_SCALE_LOAD=True, qo_len=qo_len, kv_len=kv_len, 
        save_qk=save_qk, dual_scale=dual_scale, quant_granularity=quant_granularity)


    else:
        block_scaled_batched_attn_kernel[grid](
            a_desc, a_scale, a_scale_2,
            b_desc, b_scale, b_scale_2,
            v_ori, output, M, N, K,
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
            a_scale_2.stride(0), a_scale_2.stride(1), a_scale_2.stride(2),
            # b_scale因子的stride: batch, head, N//128, K//VEC_SIZE//4
            b_scale.stride(0), b_scale.stride(1), b_scale.stride(2), b_scale.stride(3),
            b_scale_2.stride(0), b_scale_2.stride(1), b_scale_2.stride(2),
            h_qo, num_kv_groups,  # head数量
            dtype_dst, is_causal,
            configs["ELEM_PER_BYTE_A"], configs["ELEM_PER_BYTE_B"], configs["VEC_SIZE"],
            # configs["BLOCK_SIZE_K"],
            configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], K,
            configs["num_stages"], USE_2D_SCALE_LOAD=True, qo_len=qo_len, kv_len=kv_len, 
            save_qk=save_qk, dual_scale=dual_scale, quant_granularity=quant_granularity)

    return output


# def initialize_block_scaled_batched_from_tensor(a_tensor, b_tensor, a_scale, b_scale, block_scale_type="nvfp4", compute_reference=False):
#     """
#     初始化多维批量block scaled matmul的参数

#     Args:
#         a_tensor: 输入矩阵A, 形状为(B, H, M, K), dtype为torch.float16
#         b_tensor: 输入矩阵B, 形状为(B, H, N, K), dtype为torch.float16  
#         a_scale: 矩阵A的scale因子, 形状为(B, H, M//128, K//VEC_SIZE//4, 32, 4, 4)
#         b_scale: 矩阵B的scale因子, 形状为(B, H, N//128, K//VEC_SIZE//4, 32, 4, 4)
#         block_scale_type: 量化类型, 可选"nvfp4", "mxfp4", "mxfp8", "mixed"
#         compute_reference: 是否计算参考结果用于验证

#     Returns:
#         a_desc, a_scale, b_desc, b_scale, configs, reference
#     """
#     B, H, M, K = a_tensor.shape
#     B_b, H_b, N, K_b = b_tensor.shape
#     assert B == B_b, f"batch size不匹配: A.shape[0]={B} != B.shape[0]={B_b}"
#     assert H == H_b, f"head数量不匹配: A.shape[1]={H} != B.shape[1]={H_b}"
#     assert K == K_b, f"矩阵维度不匹配: A.shape[3]={K} != B.shape[3]={K_b}"

#     BLOCK_M = 128
#     # BLOCK_N = 256
#     BLOCK_N = 128
#     BLOCK_K = 256 if "fp4" in block_scale_type else 128
#     VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
#     assert block_scale_type in ["nvfp4", "mxfp4", "mxfp8",
#                                 "mixed"], f"Invalid block scale type: {block_scale_type}"
#     ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
#     ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

#     device = a_tensor.device

#    # 根据block_scale_type处理输入数据
#     if block_scale_type in ["mxfp8", "mixed"]:
#         a = a_tensor.to(torch.float8_e5m2)
#         a_ref = a.to(torch.float32) if compute_reference else None
#     else:
#         # 对于fp4格式, 这里需要将fp16数据转换为packed fp4格式
#         # a = a_tensor
#         a = MXFP4Tensor(data=a_tensor, dtype=torch.uint8)
#         a_ref = a.to(torch.float32) if compute_reference else None
#         a = a.to_packed_tensor(dim=1)

#     if block_scale_type == "mxfp8":
#         b = b_tensor.to(torch.float8_e5m2)
#         b_ref = b.to(torch.float32) if compute_reference else None
#     else:
#         # 对于fp4格式, 这里需要将fp16数据转换为packed fp4格式
#         # b = b_tensor
#         b = MXFP4Tensor(data=b_tensor, dtype=torch.uint8)
#         b_ref = b.to(torch.float32) if compute_reference else None
#         b = b.to_packed_tensor(dim=1)

#     a_desc = a
#     b_desc = b

#     # 处理scale因子
#     if a_scale.dtype == torch.uint8:
#         pass
#     else:
#         if block_scale_type == "nvfp4":
#             a_scale = a_scale.to(torch.float8_e5m2)
#             b_scale = b_scale.to(torch.float8_e5m2)
#             a_scale_ref = a_scale.to(torch.float32)
#             b_scale_ref = b_scale.to(torch.float32)
#         elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
#             a_scale_ref = MXScaleTensor(a_scale.to(torch.float32))
#             b_scale_ref = MXScaleTensor(b_scale.to(torch.float32))
#             a_scale = a_scale_ref.data
#             b_scale = b_scale_ref.data

#     reference = None
#     if compute_reference:
#         # 批量计算参考结果，处理多维输入
#         b_ref = b_ref.transpose(-1, -2).contiguous()  # (B, H, K, N)

#         a_scale_ref = a_scale_ref.to(torch.float32)
#         b_scale_ref = b_scale_ref.to(torch.float32)

#         def unpack_scale_batched(packed):
#             B, H, num_chunk_m, num_chunk_k, _, _, _ = packed.shape
#             return packed.permute(0, 1, 2, 5, 4, 3, 6).reshape(B, H, num_chunk_m * 128, num_chunk_k * 4).contiguous()
#         # 展开scale因子到原始矩阵大小
#         a_scale_expanded = unpack_scale_batched(
#             a_scale_ref).repeat_interleave(VEC_SIZE, dim=3)[:, :, :M, :K]
#         b_scale_expanded = unpack_scale_batched(b_scale_ref).repeat_interleave(
#             VEC_SIZE, dim=3).transpose(-1, -2).contiguous()[:, :, :K, :N]

#         # 计算参考结果：批量矩阵乘法 (A * scale_a) @ (B * scale_b)
#         # 使用torch.matmul自动处理批量和head维度
#         reference = torch.matmul(
#             a_ref * a_scale_expanded, b_ref * b_scale_expanded)

#     configs = {
#         "BLOCK_SIZE_M": BLOCK_M,
#         "BLOCK_SIZE_N": BLOCK_N,
#         "BLOCK_SIZE_K": BLOCK_K,
#         "num_stages": 1,
#         "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
#         "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
#         "VEC_SIZE": VEC_SIZE,
#     }

#     if compute_reference:
#         # return a_desc, a_scale, b_desc, b_scale, configs, (reference, a_ref * a_scale_expanded, (b_ref * b_scale_expanded).transpose(-1, -2))
#         return a_desc, a_scale, b_desc, b_scale, configs, (reference, a_ref, (b_ref).transpose(-1, -2))
#     else:
#         return a_desc, a_scale, b_desc, b_scale, configs, (None, None, None)
