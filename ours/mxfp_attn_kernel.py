
from asyncio import FastChildWatcher
from types import NoneType
from sympy.logic.boolalg import false
import torch
import numpy as np

import argparse
import torch
import triton
import triton.language as tl
import triton.profiler as proton

from ours.mxfp import MXFP4Tensor, MXScaleTensor
# from ours.quant_kernels import quant_fpxint8, quant_mxfp8e5, quant_mxfp4, quant_nvfp4, get_nvfp4_scale, quant_mxfp8e5_nvfp4, quant_mxfp8e4_nvfp4, quant_mxfp8e4
from ours.quant_funcs import quant_mxfp8, quant_mxfp4, quant_nvfp4, get_nvfp4_scale, quant_mxfp8_nvfp4, quant_mxfp8_nvfp4, quant_mxfp8
import os

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
        qk_dtype: tl.constexpr = 0,   # 0 for e5m2, 1 for e4m3
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
        elif quant_granularity == 3: # tensorwise - Q, K
            q_scale_2_offset = off_z * stride_sqb_2 + off_h * stride_sqh_2
            k_scale_2_offset = off_z * stride_skb_2 + off_h * stride_skh_2
            q_scale_2_ptr = q_scale_2 + q_scale_2_offset
            k_scale_2_ptr = k_scale_2 + k_scale_2_offset
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
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e2m1")
            elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
                qk = tl.dot_scaled(q, scale_q, "e2m1", k.T, scale_k, "e2m1")  # 128,128
            else:
                # qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
                if qk_dtype == 0:
                    qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
                else:
                    qk = tl.dot_scaled(q, scale_q, "e4m3", k.T, scale_k, "e4m3")
            
            if dual_scale: qk = qk * (scale_q_2 * scale_k_2)
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
                # qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
                # qk = tl.dot_scaled(q, scale_q, "e4m3", k.T, scale_k, "e4m3")
                if qk_dtype == 0:
                    qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
                else:
                    qk = tl.dot_scaled(q, scale_q, "e4m3", k.T, scale_k, "e4m3")

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
                qk = tl.dot_scaled(q, scale_q, "e4m3", k.T, scale_k, "e4m3")

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
def block_scaled_batched_attn_kernel_mp_diag_pre_quant(  #
        q_ptr, q_scale, q_scale_2, q_nvfp4_ptr, q_scale_nvfp4,  #
        k_ptr, k_scale, k_scale_2, k_nvfp4_ptr, k_scale_nvfp4,  #
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
        diag_tile: tl.constexpr = 1, 
        sink_tile: tl.constexpr = 1,
        qk_dtype: tl.constexpr = 0,   # 0 for e5m2, 1 for e4m3
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
    q_base_offset_nvfp4 = q_base_offset // 2  # 128*128 -> 64*128
    k_base_offset_nvfp4 = k_base_offset // 2  # 128*128 -> 64*128
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
        elif quant_granularity == 3: # tensorwise - Q, K
            q_scale_2_offset = off_z * stride_sqb_2 + off_h * stride_sqh_2
            k_scale_2_offset = off_z * stride_skb_2 + off_h * stride_skh_2
            q_scale_2_ptr = q_scale_2 + q_scale_2_offset
            k_scale_2_ptr = k_scale_2 + k_scale_2_offset
            scale_q_2 = tl.load(q_scale_2_ptr)

    # 简化scale load，使用2D模式
    # if USE_2D_SCALE_LOAD:
    offs_inner = tl.arange(0, (HEAD_DIM // VEC_SIZE // 4) * 32 * 4 * 4)
    q_scale_ptr = q_scale + q_scale_base_offset + offs_sm[:, None] * stride_sqm + offs_inner[None, :] 
    k_scale_ptr = k_scale + k_scale_base_offset + offs_sn[:, None] * stride_skn + offs_inner[None, :] 
    
    offs_inner_nvfp4 = tl.arange(0, (HEAD_DIM // 16 // 4) * 32 * 4 * 4)  # [2, 32, 4, 4]

    q_scale_nvfp4_ptr = q_scale_nvfp4 + q_scale_nvfp4_base_offset + offs_sm[:, None] * (stride_sqm * 2) + offs_inner_nvfp4[None, :] 
    k_scale_nvfp4_ptr = k_scale_nvfp4 + k_scale_nvfp4_base_offset + offs_sn[:, None] * (stride_skn * 2) + offs_inner_nvfp4[None, :] 

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)  # [128, 256] -> p

    off_k_nvfp4 = tl.arange(0, HEAD_DIM//2)  # [0, 128]
    off_q_nvfp4 = tl.arange(0, HEAD_DIM//2)  # [0, 128]

    off_k = tl.arange(0, HEAD_DIM)  # [0, 256]
    off_q = tl.arange(0, HEAD_DIM)  # [0, 256]

    off_v = tl.arange(0, HEAD_DIM)

    q_ptrs = q_ptr + q_base_offset + offs_m[:, None] * stride_qm + off_q[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < qo_len, other=0.0)
    scale_q = tl.load(q_scale_ptr)  # [1, 512]
    # if mp_diag: 

    off_k_nvfp4 = tl.arange(0, HEAD_DIM//2)  # [0, 128]
    off_q_nvfp4 = tl.arange(0, HEAD_DIM//2)  # [0, 128]

    q_ptrs_nvfp4 = q_nvfp4_ptr + q_base_offset_nvfp4 + offs_m[:, None] * (stride_qm//2) + off_q_nvfp4[None, :] # 128*64
    q_nvfp4 = tl.load(q_ptrs_nvfp4, mask=offs_m[:, None] < qo_len, other=0.0)  # 128*64
    scale_q_nvfp4 = tl.load(q_scale_nvfp4_ptr) 

    # if USE_2D_SCALE_LOAD:
    scale_q = scale_q.reshape(BLOCK_M // WARP_SIZE_M, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)  # 1, 1, 32, 4, 4
    scale_q_nvfp4 = scale_q_nvfp4.reshape(BLOCK_M // WARP_SIZE_M, HEAD_DIM // 16 // 4, 32, 4, 4)  # 1, 2, 32, 4, 4

    scale_q = scale_q.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, HEAD_DIM // VEC_SIZE)  # [128, 8]
    scale_q_nvfp4 = scale_q_nvfp4.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, HEAD_DIM // 16)  # [128, 8]

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0 # 初始化为1
    old_m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # k_ptrs = k_ptr + k_base_offset + (offs_n[None, :] * stride_kn + off_k[:, None]) 
    k_ptrs = k_ptr + k_base_offset + (offs_n[:, None] * stride_kn + off_k[None, :]) 
    v_ptrs = v_ori + v_base_offset + (offs_n[:, None] * stride_vn + off_v[None, :]) 
    
    k_ptrs_nvfp4 = k_nvfp4_ptr + k_base_offset_nvfp4 + offs_n[:, None] * (stride_kn//2) + off_k_nvfp4[None, :] # 128*64

    if is_causal:
        lo, hi = 0, sink_tile * BLOCK_M
        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):
            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len

            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            scale_k = tl.load(k_scale_ptr)  # [2, 1024]
            if dual_scale: scale_k_2 = tl.load(k_scale_2_ptr)

            scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)
            k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk
            scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE) 

            if qk_dtype == 0:
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
            else:
                qk = tl.dot_scaled(q, scale_q, "e4m3", k.T, scale_k, "e4m3")
            if dual_scale: qk = qk * scale_q_2 * scale_k_2

            qk = tl.where(n_mask[None, :], qk, -float("inf"))
            # qk = tl.where(offs_m[:, None] < qo_len, qk, -float("inf"))

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
            # acc += tl.dot_scaled(p, None, "fp16", v, None, "fp16")
            old_m = new_m

            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn
            if dual_scale:  
                if quant_granularity == 0:
                    k_scale_2_ptr += 1
                elif quant_granularity == 1:
                    k_scale_2_ptr += HEAD_DIM  # seems right?
                elif quant_granularity == 2:
                    k_scale_2_ptr += BLOCK_N


        # 因果注意力第一阶段
        k_ptrs_nvfp4 += BLOCK_N * (stride_kn//2) * sink_tile # * BLOCK_M//BLOCK_N
        k_scale_nvfp4_ptr += (HEAD_DIM // 16 // 4) * stride_skk * sink_tile
        
        # 因果注意力第一阶段
        lo, hi = sink_tile * BLOCK_M, (start_m+1-diag_tile) * BLOCK_M
        hi = sink_tile * BLOCK_M if hi < sink_tile * BLOCK_M else hi
        lo = hi if lo > hi else lo

        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):

            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len

            # k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)  # [headdim, 130-128] 
            # scale_k = tl.load(k_scale_ptr)  # round1: 128*4 --> k: 128*128, round2: 2*4  k: (130-128)*128 = 2*128
            k_nvfp4 = tl.load(k_ptrs_nvfp4, mask=n_mask[:, None], other=0.0)  # 128*64
            if dual_scale: scale_k_2 = tl.load(k_scale_2_ptr)

            scale_k_nvfp4 = tl.load(k_scale_nvfp4_ptr)  # [1, 512]
            # if USE_2D_SCALE_LOAD:
            scale_k_nvfp4 = scale_k_nvfp4.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // 16 // 4, 32, 4, 4)  # 1, 2, 32, 4, 4
            scale_k_nvfp4 = scale_k_nvfp4.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // 16)  # 128, 8
            qk = tl.dot_scaled(q_nvfp4, scale_q_nvfp4, "e2m1", k_nvfp4.T, scale_k_nvfp4, "e2m1")# causal - stage1
            k_ptrs_nvfp4 += BLOCK_N * (stride_kn//2)
            k_scale_nvfp4_ptr += (HEAD_DIM // 16 // 4) * (stride_skk)
           
            if dual_scale: qk = qk * scale_q_2 * scale_k_2
            
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

            v_ptrs += BLOCK_N * stride_vn

            if dual_scale: 
                if quant_granularity == 0:
                    k_scale_2_ptr += 1
                elif quant_granularity == 1:
                    k_scale_2_ptr += HEAD_DIM  # seems right?
                elif quant_granularity == 2:
                    k_scale_2_ptr += BLOCK_N

        # 因果注意力第二阶段
        k_ptrs += BLOCK_N * stride_kn * (hi//BLOCK_N - sink_tile)
        k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk * (hi//BLOCK_N - sink_tile)
        # k_ptrs += BLOCK_N * stride_kn * (hi//BLOCK_M)
        # k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk * (hi//BLOCK_M)

        # saved_qk = tl.zeros([BLOCK_M, 8], dtype=tl.float32)
        lo, hi = (start_m+1-diag_tile) * BLOCK_M, (start_m + 1) * BLOCK_M
        # lo = tl.multiple_of(lo, BLOCK_M)
        lo = sink_tile * BLOCK_N if lo < sink_tile * BLOCK_N else lo 
        hi = sink_tile * BLOCK_N if hi < sink_tile * BLOCK_N else hi
        
        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):

            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len

            # k = tl.load(k_ptrs, mask = offs_n[None, :] < (kv_len - start_n))
            # k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0) 
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            scale_k = tl.load(k_scale_ptr)  # [2, 1024]
            if dual_scale: scale_k_2 = tl.load(k_scale_2_ptr)

            # if USE_2D_SCALE_LOAD:  
            scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)
            k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk
            scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE) 

            if qk_dtype == 0:
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
            else:
                qk = tl.dot_scaled(q, scale_q, "e4m3", k.T, scale_k, "e4m3")
            if dual_scale: qk = qk * scale_q_2 * scale_k_2

            qk = tl.where(n_mask[None, :], qk, -float("inf"))

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
            # acc += tl.dot_scaled(p, None, "fp16", v, None, "fp16")
            old_m = new_m

            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn
            # if USE_2D_SCALE_LOAD:
            #     k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk
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
                # qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
                qk = tl.dot_scaled(q, scale_q, "e4m3", k.T, scale_k, "e4m3")

            if dual_scale:  
                qk = qk * (scale_q_2 * scale_k_2.T)

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
    # if save_qk:
    #     o_ptrs = o_ptr + off_z * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + tl.arange(0, 8)[None, :]  
    #     tl.store(o_ptrs, saved_qk.to(output_dtype), mask=(offs_m[:, None] < qo_len))
    # else:
    tl.store(o_ptrs, acc.to(output_dtype), mask=(offs_m[:, None] < qo_len))




@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_batched_attn_kernel_mp_sink_pre_quant(  #
        q_ptr, q_scale, q_scale_2, q_nvfp4_ptr, q_scale_nvfp4,  #
        k_ptr, k_scale, k_scale_2, k_nvfp4_ptr, k_scale_nvfp4,  #
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
        sink_tile: tl.constexpr = 1,
        qk_dtype: tl.constexpr = 0,   # 0 for e5m2, 1 for e4m3
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
    q_base_offset_nvfp4 = q_base_offset // 2  # 128*128 -> 64*128
    k_base_offset_nvfp4 = k_base_offset // 2  # 128*128 -> 64*128
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
        elif quant_granularity == 2: # tokenwise - Q, K
            q_scale_2_offset = off_z * stride_sqb_2 + off_h * stride_sqh_2
            k_scale_2_offset = off_z * stride_skb_2 + off_h * stride_skh_2
            q_scale_2_ptr = q_scale_2 + q_scale_2_offset + pid_m * stride_sqm_2 + tl.arange(0, BLOCK_M)[:, None]  # BLOCK_M
            k_scale_2_ptr = k_scale_2 + k_scale_2_offset + tl.arange(0, BLOCK_N)[None, :]  # BLOCK_N
            scale_q_2 = tl.load(q_scale_2_ptr)

    offs_inner = tl.arange(0, (HEAD_DIM // VEC_SIZE // 4) * 32 * 4 * 4)
    q_scale_ptr = q_scale + q_scale_base_offset + offs_sm[:, None] * stride_sqm + offs_inner[None, :] 
    k_scale_ptr = k_scale + k_scale_base_offset + offs_sn[:, None] * stride_skn + offs_inner[None, :] 
    
    offs_inner_nvfp4 = tl.arange(0, (HEAD_DIM // 16 // 4) * 32 * 4 * 4)  # [2, 32, 4, 4]

    q_scale_nvfp4_ptr = q_scale_nvfp4 + q_scale_nvfp4_base_offset + offs_sm[:, None] * (stride_sqm * 2) + offs_inner_nvfp4[None, :] 
    k_scale_nvfp4_ptr = k_scale_nvfp4 + k_scale_nvfp4_base_offset + offs_sn[:, None] * (stride_skn * 2) + offs_inner_nvfp4[None, :] 

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)  # [128, 256] -> p

    off_k_nvfp4 = tl.arange(0, HEAD_DIM//2)  # [0, 128]
    off_q_nvfp4 = tl.arange(0, HEAD_DIM//2)  # [0, 128]

    off_k = tl.arange(0, HEAD_DIM)  # [0, 256]
    off_q = tl.arange(0, HEAD_DIM)  # [0, 256]

    off_v = tl.arange(0, HEAD_DIM)

    q_ptrs = q_ptr + q_base_offset + offs_m[:, None] * stride_qm + off_q[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < qo_len, other=0.0)
    scale_q = tl.load(q_scale_ptr)  # [1, 512]

    off_k_nvfp4 = tl.arange(0, HEAD_DIM//2)  # [0, 128]
    off_q_nvfp4 = tl.arange(0, HEAD_DIM//2)  # [0, 128]

    q_ptrs_nvfp4 = q_nvfp4_ptr + q_base_offset_nvfp4 + offs_m[:, None] * (stride_qm//2) + off_q_nvfp4[None, :] # 128*64
    q_nvfp4 = tl.load(q_ptrs_nvfp4, mask=offs_m[:, None] < qo_len, other=0.0)  # 128*64
    scale_q_nvfp4 = tl.load(q_scale_nvfp4_ptr) 

    scale_q = scale_q.reshape(BLOCK_M // WARP_SIZE_M, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)  # 1, 1, 32, 4, 4
    scale_q_nvfp4 = scale_q_nvfp4.reshape(BLOCK_M // WARP_SIZE_M, HEAD_DIM // 16 // 4, 32, 4, 4)  # 1, 2, 32, 4, 4

    scale_q = scale_q.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, HEAD_DIM // VEC_SIZE)  # [128, 8]
    scale_q_nvfp4 = scale_q_nvfp4.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, HEAD_DIM // 16)  # [128, 8]

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0 # 初始化为1
    old_m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # k_ptrs = k_ptr + k_base_offset + (offs_n[None, :] * stride_kn + off_k[:, None]) 
    k_ptrs = k_ptr + k_base_offset + (offs_n[:, None] * stride_kn + off_k[None, :]) 
    v_ptrs = v_ori + v_base_offset + (offs_n[:, None] * stride_vn + off_v[None, :]) 
    
    k_ptrs_nvfp4 = k_nvfp4_ptr + k_base_offset_nvfp4 + offs_n[:, None] * (stride_kn//2) + off_k_nvfp4[None, :] # 128*64

    if is_causal:
        lo, hi = 0, sink_tile * BLOCK_M
        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):
            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len

            # k = tl.load(k_ptrs, mask = offs_n[None, :] < (kv_len - start_n))
            # k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0) 
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            scale_k = tl.load(k_scale_ptr)  # [2, 1024]
            if dual_scale: scale_k_2 = tl.load(k_scale_2_ptr)

            # if USE_2D_SCALE_LOAD:  
            scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)
            k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk
            scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE) 

            if qk_dtype == 0:
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
            else:
                qk = tl.dot_scaled(q, scale_q, "e4m3", k.T, scale_k, "e4m3")
            if dual_scale: qk = qk * (scale_q_2 * scale_k_2)

            qk = tl.where(n_mask[None, :], qk, -float("inf"))
            # qk = tl.where(offs_m[:, None] < qo_len, qk, -float("inf"))

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
            # acc += tl.dot_scaled(p, None, "fp16", v, None, "fp16")
            old_m = new_m

            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn
            if dual_scale:  
                if quant_granularity == 0: k_scale_2_ptr += 1
                elif quant_granularity == 1: k_scale_2_ptr += HEAD_DIM  # seems right?
                elif quant_granularity == 2: k_scale_2_ptr += BLOCK_N


        # 因果注意力第一阶段
        k_ptrs_nvfp4 += BLOCK_N * (stride_kn//2) * sink_tile # * BLOCK_M//BLOCK_N
        k_scale_nvfp4_ptr += (HEAD_DIM // 16 // 4) * stride_skk * sink_tile
        
        # 因果注意力第一阶段
        lo, hi = sink_tile * BLOCK_M, (start_m) * BLOCK_M
        hi = sink_tile * BLOCK_M if hi < sink_tile * BLOCK_M else hi
        lo = hi if lo > hi else lo

        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):

            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len

            # k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)  # [headdim, 130-128] 
            # scale_k = tl.load(k_scale_ptr)  # round1: 128*4 --> k: 128*128, round2: 2*4  k: (130-128)*128 = 2*128
            k_nvfp4 = tl.load(k_ptrs_nvfp4, mask=n_mask[:, None], other=0.0)  # 128*64
            if dual_scale: scale_k_2 = tl.load(k_scale_2_ptr)

            scale_k_nvfp4 = tl.load(k_scale_nvfp4_ptr)  # [1, 512]
            scale_k_nvfp4 = scale_k_nvfp4.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // 16 // 4, 32, 4, 4)  # 1, 2, 32, 4, 4
            scale_k_nvfp4 = scale_k_nvfp4.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // 16)  # 128, 8
            qk = tl.dot_scaled(q_nvfp4, scale_q_nvfp4, "e2m1", k_nvfp4.T, scale_k_nvfp4, "e2m1")# causal - stage1
            k_ptrs_nvfp4 += BLOCK_N * (stride_kn//2)
            k_scale_nvfp4_ptr += (HEAD_DIM // 16 // 4) * (stride_skk)
           
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

            v_ptrs += BLOCK_N * stride_vn

            if dual_scale: 
                if quant_granularity == 0:
                    k_scale_2_ptr += 1
                elif quant_granularity == 1:
                    k_scale_2_ptr += HEAD_DIM  # seems right?
                elif quant_granularity == 2:
                    k_scale_2_ptr += BLOCK_N

        # 因果注意力第二阶段
        k_ptrs += BLOCK_N * stride_kn * (hi//BLOCK_N - sink_tile)
        k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk * (hi//BLOCK_N - sink_tile)
        # k_ptrs += BLOCK_N * stride_kn * (hi//BLOCK_M)
        # k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk * (hi//BLOCK_M)

        # saved_qk = tl.zeros([BLOCK_M, 8], dtype=tl.float32)
        lo, hi = (start_m) * BLOCK_M, (start_m + 1) * BLOCK_M
        # lo = tl.multiple_of(lo, BLOCK_M)
        lo = sink_tile * BLOCK_N if lo < sink_tile * BLOCK_N else lo 
        hi = sink_tile * BLOCK_N if hi < sink_tile * BLOCK_N else hi
        
        for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):

            # 计算当前块的有效范围
            curr_n_range = start_n + offs_n
            n_mask = curr_n_range < kv_len
 
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            scale_k = tl.load(k_scale_ptr)  # [2, 1024]
            if dual_scale: scale_k_2 = tl.load(k_scale_2_ptr)

            # if USE_2D_SCALE_LOAD:  
            scale_k = scale_k.reshape(BLOCK_N // WARP_SIZE_N, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)
            k_scale_ptr += (HEAD_DIM // VEC_SIZE // 4) * stride_skk
            scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE) 

            if qk_dtype == 0:
                qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
            else:
                qk = tl.dot_scaled(q, scale_q, "e4m3", k.T, scale_k, "e4m3")
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
                # qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
                if qk_dtype == 0:
                    qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
                else:
                    qk = tl.dot_scaled(q, scale_q, "e4m3", k.T, scale_k, "e4m3")

            if dual_scale:  
                qk = qk * scale_q_2 * scale_k_2

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
    # if save_qk:
    #     o_ptrs = o_ptr + off_z * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + tl.arange(0, 8)[None, :]  
    #     tl.store(o_ptrs, saved_qk.to(output_dtype), mask=(offs_m[:, None] < qo_len))
    # else:
    tl.store(o_ptrs, acc.to(output_dtype), mask=(offs_m[:, None] < qo_len))
