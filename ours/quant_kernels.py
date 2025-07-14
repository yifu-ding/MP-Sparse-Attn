"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def quant_fpxint8_kernel(Input, Output, Scale, L,
                        stride_iz, stride_ih, stride_in,
                        stride_oz, stride_oh, stride_on,
                        stride_sz, stride_sh, stride_sn,  # stride_sz: batchsize, stride_sh: headnum, stride_sn: seq_len
                        sm_scale,
                        C: tl.constexpr, BLK: tl.constexpr):   # C: head_dim, BLK: BLKQ 128 or BLKK 64
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
    x_reshaped = tl.reshape(x, (BLK, C // 32, 32))   # x_reshaped: [BLKQ (128), headdim // 32, 32]
    scales = tl.max(tl.abs(x_reshaped), axis=-1) / 127.  # scales: [BLKQ (128), headdim // 32, 1]
    scales = scales + 0.0000001
    # scale 要符合 x_reshaped 的形状？
    scales_broadcast = tl.broadcast_to(tl.reshape(scales, (BLK, C // 32, 1)), (BLK, C // 32, 32))
    x_reshaped = x_reshaped / scales_broadcast
    x_reshaped += 0.5 * tl.where(x_reshaped >= 0, 1, -1)
    x_reshaped = x_reshaped.to(tl.int8)
    x_int8 = tl.reshape(x_reshaped, x.shape)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)

    # 存储每32个元素的scale
    scales = scales.to(tl.float8e5)
    tl.store(scale_ptrs, scales, mask=offs_n[:, None] < L)


def quant_fpxint8(q, k, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    # q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ, 1), device=q.device, dtype=torch.float32)
    # k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK, 1), device=q.device, dtype=torch.float32)
    q_scale = torch.empty((b, h_qo, qo_len, head_dim // 32), device=q.device, dtype=torch.float32)  
    k_scale = torch.empty((b, h_kv, kv_len, head_dim // 32), device=q.device, dtype=torch.float32)
    # q_scale = torch.empty((b, h_qo, qo_len, head_dim // 32), device=q.device, dtype=torch.float8_e5m2)  
    # k_scale = torch.empty((b, h_kv, kv_len, head_dim // 32), device=q.device, dtype=torch.float8_e5m2)
    
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_fpxint8_kernel[grid](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ
    )

    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_fpxint8_kernel[grid](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK
    )

    return q_int8, q_scale, k_int8, k_scale

@triton.jit
def quant_mxfp8e5_kernel(Input, Output, Scale, L,
                    stride_iz, stride_ih, stride_in,
                    stride_oz, stride_oh, stride_on,
                    stride_sz, stride_sh, stride_sn,  # b, h_qo, qo_len, head_dim // 32
                    sm_scale,
                    C: tl.constexpr, BLK: tl.constexpr):   # C: head_dim, BLK: BLKQ 128 or BLKK 64
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

    x_reshaped = tl.reshape(x, (BLK, C // 32, 32))   # x_reshaped: [BLKQ (128), headdim // 32, 32]  --> [BLKQ, 4, 32]
    abs_max = tl.max(tl.abs(x_reshaped), axis=-1)  # abs_max shape: [BLK, C//32] --> [BLKQ, 4]
    
    # 对于float8，emax_elem = 7 for e4m3, 15 for e5m2
    emax_elem = 15
    shared_exp = tl.floor(tl.log2(abs_max)) - emax_elem 
    shared_scale = tl.exp2(shared_exp) 

    shared_scale_broadcast = tl.broadcast_to(tl.reshape(shared_scale, (BLK, C // 32, 1)), (BLK, C // 32, 32))
    x_quant = x_reshaped / shared_scale_broadcast  # x/e-4 = x * e4
    
    # 这里增加一个量化的 scale
    
    # x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)  # 浮点数的四舍五入
    x_quant = tl.clamp(x_quant, -57344, 57344)  # e5m2 range
    # x_quant = tl.clamp(x_quant, -448, 448)  # e4m3 range
    x_quant = x_quant.to(tl.float8e5)
    
    # 存储量化后的值和scale
    x_fp8 = tl.reshape(x_quant, x.shape)
    tl.store(output_ptrs, x_fp8, mask=offs_n[:, None] < L)
    
    # 存储scale
    # is_invalid = torch.isnan(shared_scale) | torch.isinf(shared_scale) | (shared_scale <= 0)
    # tl.store(scale_ptrs, 255, mask = ~is_invalid)
    # valid_values = shared_scale[~is_invalid]
    e = tl.floor(tl.log2(shared_scale))
    e_biased = e + 127
    e_biased_clamped = tl.clamp(e_biased, 0, 254)
    tl.store(scale_ptrs, e_biased_clamped.to(tl.uint8), mask=offs_n[:, None] < L)

def quant_mxfp8e5(q, k, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"):
    q_fp8 = torch.empty(q.shape, dtype=torch.float8_e5m2, device=q.device)
    k_fp8 = torch.empty(k.shape, dtype=torch.float8_e5m2, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_fp8.stride(0), k_fp8.stride(1), k_fp8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_fp8.stride(0), q_fp8.stride(2), q_fp8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_fp8.stride(0), k_fp8.stride(2), k_fp8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, qo_len, head_dim // 32), device=q.device, dtype=torch.uint8)
    k_scale = torch.empty((b, h_kv, kv_len, head_dim // 32), device=q.device, dtype=torch.uint8)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_mxfp8e5_kernel[grid](
        q, q_fp8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ
    )

    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_mxfp8e5_kernel[grid](
        k, k_fp8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK
    )

    return q_fp8, q_scale, k_fp8, k_scale



@triton.jit
def quant_mxfp4_kernel(Input, Output, Scale, L,
                    stride_iz, stride_ih, stride_in,
                    stride_oz, stride_oh, stride_on,
                    stride_sz, stride_sh, stride_sn,
                    sm_scale,
                    C: tl.constexpr, BLK: tl.constexpr, candidates_ptr=None, 
                    pack_along_lastdim = False):   # C: head_dim, BLK: BLKQ 128 or BLKK 64
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    offs_n_32 = tl.arange(0, C//32)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + offs_n[:, None] * stride_sn + offs_n_32[None, :]

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale

    x_reshaped = tl.reshape(x, (BLK, C // 32, 32))   # x_reshaped: [BLKQ (128), headdim // 32, 32]
    abs_max = tl.max(tl.abs(x_reshaped), axis=-1)  # abs_max shape: [BLK, C//32]
    
    emax_elem = 3 
    shared_exp = tl.floor(tl.log2(abs_max)) - emax_elem 
    shared_scale = tl.exp2(shared_exp) 
        
    shared_scale_broadcast = tl.broadcast_to(tl.reshape(shared_scale, (BLK, C // 32, 1)), (BLK, C // 32, 32))
    x_quant = x_reshaped / shared_scale_broadcast  # x/e-4 = x * e4
    
    # 这里增加量化的 scale
    
    x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)  # 浮点数的四舍五入
    # float4 (e2m1) range: [-6, 6]
    x_quant = tl.clamp(x_quant, -6.0, 6.0)
    # float4 (e2m1) 的量化实现
    # 1. 获取符号位
    sign = tl.where(x_quant >= 0, 0, 1)
    abs_x = tl.abs(x_quant)
    # 2. 获取指数位 (2位)
    exp = tl.where(abs_x >= 2.0, 
                  tl.where(abs_x >= 4.0, 3, 2),
                  tl.where(abs_x >= 1.0, 1, 0))
    # 3. 获取尾数位 (1位)
    bias = 1.0
    norm_x = abs_x / (tl.exp2((exp-bias).to(tl.float32)))  # 首先将数值规范化到[1,2)区间
    mantissa = tl.where(exp == 0, tl.where(norm_x > 0.25, 1, 0), tl.where(norm_x > 1.25, 1, 0))  # 平局优先选择偶数尾数
    # 4. 组合成float4 (sign:1, exp:2, mantissa:1)，存在float8里
    x_quant = (sign << 3) | (exp << 1) | mantissa
    
    # 5. pack and store x_quant
    # Pack two e2m1 elements into a single uint8 along the specified dimension.
    # x_uint8 = tl.reshape(x_quant, x.shape)
    if pack_along_lastdim:
        x_quant_reshaped = tl.reshape(x_quant, (BLK, (C + 1) // 2, 2))
        low, high = tl.split(x_quant_reshaped)
        x_uint8_packed = (high << 4) | low
        # mask = (offs_n[:, None] < L) & (offs_k[None, :] < C//2)
        output_ptrs_packed = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + tl.arange(0, C//2)[None, :]
        tl.store(output_ptrs_packed, x_uint8_packed, mask=offs_n[:, None] < L)
    else:
        x_uint8 = tl.reshape(x_quant, x.shape)
        output_ptrs_unpacked = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
        tl.store(output_ptrs_unpacked, x_uint8, mask=offs_n[:, None] < L)

    # 6. store scale
    # is_invalid = torch.isnan(shared_scale) | torch.isinf(shared_scale) | (shared_scale <= 0)
    # tl.store(scale_ptrs, 255, mask = ~is_invalid)
    # valid_values = shared_scale[~is_invalid]
    e = tl.floor(tl.log2(shared_scale))
    e_biased = e + 127
    # e_biased_int = e_biased
    e_biased_clamped = tl.clamp(e_biased, 0, 254) #.to(tl.int32)
    tl.store(scale_ptrs, e_biased_clamped.to(tl.uint8), mask=offs_n[:, None] < L)


def quant_mxfp4(q, k, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND", pack_along_lastdim=False):

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        if pack_along_lastdim:
            assert BLKQ % 2 == 0, "BLKQ must be even for packing along lastdim"
            assert BLKK % 2 == 0, "BLKK must be even for packing along lastdim"
            q_fp4 = torch.empty((b, h_qo, qo_len, (head_dim + 1) // 2), dtype=torch.uint8, device=q.device)
            k_fp4 = torch.empty((b, h_kv, kv_len, (head_dim + 1) // 2), dtype=torch.uint8, device=k.device)
        else:
            q_fp4 = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
            k_fp4 = torch.empty(k.shape, dtype=torch.uint8, device=k.device)

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_fp4.stride(0), q_fp4.stride(1), q_fp4.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_fp4.stride(0), k_fp4.stride(1), k_fp4.stride(2)

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        
        if pack_along_lastdim:
            assert BLKQ % 2 == 0, "BLKQ must be even for packing along lastdim"
            assert BLKK % 2 == 0, "BLKK must be even for packing along lastdim"
            q_fp4 = torch.empty((b, qo_len, h_qo, (head_dim + 1) // 2), dtype=torch.uint8, device=q.device)
            k_fp4 = torch.empty((b, kv_len, h_kv, (head_dim + 1) // 2), dtype=torch.uint8, device=k.device)
        else:
            q_fp4 = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
            k_fp4 = torch.empty(k.shape, dtype=torch.uint8, device=k.device)

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_fp4.stride(0), q_fp4.stride(2), q_fp4.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_fp4.stride(0), k_fp4.stride(2), k_fp4.stride(1)

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, qo_len, head_dim // 32), device=q.device, dtype=torch.uint8)
    k_scale = torch.empty((b, h_kv, kv_len, head_dim // 32), device=q.device, dtype=torch.uint8)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_mxfp4_kernel[grid](
        q, q_fp4, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ,
        pack_along_lastdim=pack_along_lastdim,
    )
    
    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_mxfp4_kernel[grid](
        k, k_fp4, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK,   
        pack_along_lastdim=pack_along_lastdim,
    )

    return q_fp4, q_scale, k_fp4, k_scale