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


@triton.jit
def quant_mxfp8_kernel(Input, Output, Scale, Scale_q, L,
                    stride_iz, stride_ih, stride_in,
                    stride_oz, stride_oh, stride_on,
                    stride_sz, stride_sh, stride_sn,  # b, h_qo, qo_len, head_dim // 32
                    stride_sz_q, stride_sh_q, stride_sn_q,  # b, h_qo, qo_len, head_dim // 32
                    sm_scale,
                    C: tl.constexpr, BLK: tl.constexpr,
                    dual_scale: tl.constexpr=False, dual_scale_type: tl.constexpr=0,
                    qk_dtype: tl.constexpr=0,
                    ):   # C: head_dim, BLK: BLKQ 128 or BLKK 64
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

    if dual_scale:
        if dual_scale_type == 0:
            scale_ptrs_2 = Scale_q + off_b * stride_sz_q + off_h * stride_sh_q + off_blk  # per block scale
            scale = tl.max(tl.abs(x)) / (2**3) # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 1:
            scale_ptrs_2 = Scale_q + off_b * stride_sz_q + off_h * stride_sh_q + off_blk * stride_sn_q + offs_k[None, :]  # per channel scale
            scale = tl.max(tl.abs(x), axis=0, keep_dims=True) / (2**3)   # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 2:
            scale_ptrs_2 = Scale_q + off_b * stride_sz_q + off_h * stride_sh_q + offs_n[:, None] # per token scale
            scale = tl.max(tl.abs(x), axis=1, keep_dims=True) / (2**3) # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale, mask=offs_n[:, None] < L)
        elif dual_scale_type == 3:
            scale_ptrs_2 = Scale_q + off_b * stride_sz_q + off_h * stride_sh_q  # per channel scale
            # scale = tl.max(tl.abs(x), axis=0, keep_dims=True) / (2**3)   # mxfp4 range: [-6, 6]
            # scale += 0.0000001
            scale = tl.load(scale_ptrs_2)
            x = x / scale
            # tl.store(scale_ptrs_2, scale)
        else:
            x = x
            
    
    x_reshaped = tl.reshape(x, (BLK, C // 32, 32))   # x_reshaped: [BLKQ (128), headdim // 32, 32]  --> [BLKQ, 4, 32]
    abs_max = tl.max(tl.abs(x_reshaped), axis=-1)  # abs_max shape: [BLK, C//32] --> [BLKQ, 4]
    
    # 对于float8，emax_elem = 7 for e4m3, 15 for e4m2
    emax_elem = 7 if qk_dtype == 1 else 15
    shared_exp = tl.floor(tl.log2(abs_max)) - emax_elem 
    shared_scale = tl.exp2(shared_exp) 

    shared_scale_broadcast = tl.broadcast_to(tl.reshape(shared_scale, (BLK, C // 32, 1)), (BLK, C // 32, 32))
    x_quant = x_reshaped / shared_scale_broadcast  # x/e-4 = x * e4
    
    # x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)  # 浮点数的四舍五入
    # x_quant = tl.clamp(x_quant, -57344, 57344)  # e4m2 range
    x_quant = tl.clamp(x_quant, -448, 448) if qk_dtype == 1 else tl.clamp(x_quant, -57344, 57344) 
    x_quant = x_quant.to(tl.float8e4nv) if qk_dtype == 1 else x_quant.to(tl.float8e5)
    
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



@triton.jit
def quant_mxfp8_nvfp4_kernel(Input, Output_fp8, Output_fp4, Scale_fp8, Scale_fp4, Scale_q, L,
                    stride_iz, stride_ih, stride_in,
                    stride_oz, stride_oh, stride_on,
                    stride_oz_2, stride_oh_2, stride_on_2,
                    stride_sz, stride_sh, stride_sn,  # b, h_qo, qo_len, head_dim // 32
                    stride_sz_2, stride_sh_2, stride_sn_2,  # b, h_qo, qo_len, head_dim // 16
                    stride_sz_q, stride_sh_q, stride_sn_q,  # b, h_qo, qo_len, head_dim // 32
                    sm_scale, 
                    C: tl.constexpr, BLK: tl.constexpr,   # C: head_dim, BLK: BLKQ 128 or BLKK 64
                    dual_scale_type: tl.constexpr = 0, dual_scale: tl.constexpr = False,
                    fuse_pack: tl.constexpr = False, qk_dtype: tl.constexpr = 0,
                    ):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output_fp8 + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale

    if dual_scale:
        if dual_scale_type == 0:
            scale_ptrs_2 = Scale_q + off_b * stride_sz_q + off_h * stride_sh_q + off_blk  # per block scale
            scale = tl.max(tl.abs(x)) / (2**11) # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 1:
            scale_ptrs_2 = Scale_q + off_b * stride_sz_q + off_h * stride_sh_q + off_blk * stride_sn_q + offs_k[None, :]  # per channel scale
            scale = tl.max(tl.abs(x), axis=0, keep_dims=True) / (448*6)  # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 2:
            scale_ptrs_2 = Scale_q + off_b * stride_sz_q + off_h * stride_sh_q + offs_n[:, None] # per token scale
            scale = tl.max(tl.abs(x), axis=1, keep_dims=True) / (2**11) # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale, mask=offs_n[:, None] < L)
        elif dual_scale_type == 3:
            scale_ptrs_2 = Scale_q + off_b * stride_sz_q + off_h * stride_sh_q  # per channel scale
            # scale = tl.max(tl.abs(x), axis=0, keep_dims=True) / (2**3)   # mxfp4 range: [-6, 6]
            # scale += 0.0000001
            scale = tl.load(scale_ptrs_2)
            x = x / scale
        else:
            x = x
    else:
        x = x
    ########################################################################

    offs_n_16 = tl.arange(0, C//16)
    scale_ptrs = Scale_fp4 + off_b * stride_sz_2 + off_h * stride_sh_2 + offs_n[:, None] * stride_sn_2 + offs_n_16[None, :]

    x_reshaped = tl.reshape(x, (BLK, C // 16, 16))   # x_reshaped: [BLKQ (128), headdim // 16, 16]
    abs_max = tl.max(tl.abs(x_reshaped), axis=-1)  # abs_max shape: [BLK, C//16]
    
    shared_scale = abs_max / 6
        
    shared_scale_broadcast = tl.broadcast_to(tl.reshape(shared_scale, (BLK, C // 16, 1)), (BLK, C // 16, 16))
    x_quant = x_reshaped / shared_scale_broadcast  # x/e-4 = x * e4     x = x / ss = x / (abs_max / 6) = x * 6 / abs_max
    
    # x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)  # 浮点数的四舍五入
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
    # 4. 组合成float4 (sign:1, exp:2, mantissa:1)，存在int8里
    x_quant = (sign << 3) | (exp << 1) | mantissa
    
    # 5. pack and store x_quant
    # Pack two e2m1 elements into a single uint8 along the specified dimension.
    if fuse_pack:
        x_quant_reshaped = tl.reshape(x_quant, (BLK, (C + 1) // 2, 2))
        low, high = tl.split(x_quant_reshaped)
        x_uint8_packed = (high << 4) | low
        # mask = (offs_n[:, None] < L) & (offs_k[None, :] < C//2)
        output_ptrs_packed = Output_fp4 + off_b * stride_oz_2 + off_h * stride_oh_2 + offs_n[:, None] * stride_on_2 + tl.arange(0, C//2)[None, :]
        tl.store(output_ptrs_packed, x_uint8_packed, mask=offs_n[:, None] < L)
    else:
        x_uint8 = tl.reshape(x_quant, x.shape)
        output_ptrs_unpacked = Output_fp4 + off_b * stride_oz_2 + off_h * stride_oh_2 + offs_n[:, None] * stride_on_2 + offs_k[None, :]
        tl.store(output_ptrs_unpacked, x_uint8, mask=offs_n[:, None] < L)

    tl.store(scale_ptrs, shared_scale.to(tl.float8e4nv), mask=offs_n[:, None] < L)

    ########################################################################
    offs_n_32 = tl.arange(0, C//32)
    scale_ptrs = Scale_fp8 + off_b * stride_sz + off_h * stride_sh + offs_n[:, None] * stride_sn + offs_n_32[None, :]

    x_reshaped = tl.reshape(x, (BLK, C // 32, 32))   # x_reshaped: [BLKQ (128), headdim // 32, 32]  --> [BLKQ, 4, 32]
    # abs_max = tl.reshape(abs_max, (BLK, C//32, 2)) 
    # abs_max = tl.max(tl.abs(abs_max), axis=-1)  # abs_max shape: [BLK, C//32] --> [BLKQ, 4]
    abs_max = tl.max(tl.abs(x_reshaped), axis=-1) 

    # 对于float8，emax_elem = 7 for e4m3, 15 for e5m2
    emax_elem = 7 if qk_dtype == 1 else 15
    shared_exp = tl.floor(tl.log2(abs_max)) - emax_elem 
    shared_scale = tl.exp2(shared_exp) 

    shared_scale_broadcast = tl.broadcast_to(tl.reshape(shared_scale, (BLK, C // 32, 1)), (BLK, C // 32, 32))
    x_quant = x_reshaped / shared_scale_broadcast  # x/e-4 = x * e4
    
    # x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)  # 浮点数的四舍五入
    # x_quant = tl.clamp(x_quant, -57344, 57344)  # e5m2 range
    x_quant = tl.clamp(x_quant, -448, 448) if qk_dtype == 1 else tl.clamp(x_quant, -57344, 57344)  # e4m3 range
    x_quant = x_quant.to(tl.float8e4nv) if qk_dtype == 1 else x_quant.to(tl.float8e5)
    
    # 存储量化后的值和scale
    x_fp8 = tl.reshape(x_quant, x.shape)
    tl.store(output_ptrs, x_fp8, mask=offs_n[:, None] < L)
    
    # 存储scale
    e = tl.floor(tl.log2(shared_scale))
    e_biased = e + 127
    e_biased_clamped = tl.clamp(e_biased, 0, 254)
    tl.store(scale_ptrs, e_biased_clamped.to(tl.uint8), mask=offs_n[:, None] < L)




@triton.jit
def quant_mxfp4_kernel(Input, Output, Scale, Scale_2, L,
                    stride_iz, stride_ih, stride_in,
                    stride_oz, stride_oh, stride_on,
                    stride_sz, stride_sh, stride_sn,
                    stride_sz_2, stride_sh_2, stride_sn_2,
                    sm_scale,
                    C: tl.constexpr, BLK: tl.constexpr, candidates_ptr=None, 
                    fuse_pack = False,
                    dual_scale_type = 0,
                    dual_scale = False):   # C: head_dim, BLK: BLKQ 128 or BLKK 64
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    offs_n_32 = tl.arange(0, C//32)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + offs_n[:, None] * stride_sn + offs_n_32[None, :]

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)  # x.shape: [BLK, C] -> [token num, head_dim]
    x = x.to(tl.float32)
    x *= sm_scale

    # import pdb; pdb.set_trace()    
    if dual_scale:
        if dual_scale_type == 0:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + off_blk  # per block scale
            scale = tl.max(tl.abs(x)) / 6  # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 1:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + off_blk * stride_sn_2 + offs_k[None, :]  # per channel scale
            scale = tl.max(tl.abs(x), axis=0, keep_dims=True) / 6   # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 2:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + offs_n[:, None] # per token scale
            scale = tl.max(tl.abs(x), axis=1, keep_dims=True) / 6 # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale, mask=offs_n[:, None] < L)
    
    x_reshaped = tl.reshape(x, (BLK, C // 32, 32))   # x_reshaped: [BLKQ (128), headdim // 32, 32]
    abs_max = tl.max(tl.abs(x_reshaped), axis=-1)  # abs_max shape: [BLK, C//32]
    
    emax_elem = 3
    shared_exp = tl.floor(tl.log2(abs_max)) - emax_elem 
    shared_scale = tl.exp2(shared_exp) 
        
    shared_scale_broadcast = tl.broadcast_to(tl.reshape(shared_scale, (BLK, C // 32, 1)), (BLK, C // 32, 32))
    x_quant = x_reshaped / shared_scale_broadcast  # x/e-4 = x * e4
    
    # x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)  # 浮点数的四舍五入
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
    # 4. 组合成float4 (sign:1, exp:2, mantissa:1)，存在int8里
    x_quant = (sign << 3) | (exp << 1) | mantissa
    
    # 5. pack and store x_quant
    # Pack two e2m1 elements into a single uint8 along the specified dimension.
    # x_uint8 = tl.reshape(x_quant, x.shape)
    if fuse_pack:
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


@triton.jit
def quant_mxfp4_per_channel_kernel(Input, Output, Scale, Scale_2, L,
                    stride_iz, stride_ih, stride_in,
                    stride_oz, stride_oh, stride_on,
                    stride_sz, stride_sh, stride_sn,
                    stride_sz_2, stride_sh_2, stride_sn_2,
                    sm_scale,
                    C: tl.constexpr, BLK: tl.constexpr, candidates_ptr=None, 
                    fuse_pack = False,
                    dual_scale_type = 0,
                    dual_scale = False):   # C: head_dim, BLK: BLKQ 128 or BLKK 64
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    offs_n_32 = off_blk * BLK//32 + tl.arange(0, BLK//32)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + offs_n_32[:, None] * stride_sn + offs_k[None, :]

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L, other=0.0)  # x.shape: [BLK, C] -> [token num, head_dim]
    x = x.to(tl.float32)
    x *= sm_scale
    
    if dual_scale:
        if dual_scale_type == 0:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + off_blk  # per block scale
            scale = tl.max(tl.abs(x)) / 6  # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 1:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + off_blk * stride_sn_2 + offs_k[None, :] # per channel scale
            scale = tl.max(tl.abs(x), axis=0, keep_dims=True) / 6  # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 2:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + offs_n[:, None] # per token scale
            scale = tl.max(tl.abs(x), axis=1, keep_dims=True) / 6  # mxfp4 range: [-6, 6]  x.shape [BLK, C]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)

    # x_trans = tl.permute(x, (1, 0))
    # import pdb; pdb.set_trace()
    x_reshaped = tl.reshape(x, (BLK//32, 32, C))   # x_reshaped: [BLKQ (128), headdim // 32, 32]
    abs_max = tl.max(tl.abs(x_reshaped), axis=1)  # abs_max shape: [BLK//32, C]  8,128
    
    emax_elem = 3
    shared_exp = tl.floor(tl.log2(abs_max)) - emax_elem 
    shared_scale = tl.exp2(shared_exp) # [BLK//32, C]  # 4, 128
    # shared_scale = abs_max / 6

    shared_scale_broadcast = tl.broadcast_to(tl.reshape(shared_scale, (BLK//32, 1, C)), (BLK//32, 32, C))  # 4, 32, 128
    x_quant = x_reshaped / shared_scale_broadcast  # x/e-4 = x * e4
    
    # x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)  # 浮点数的四舍五入
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
    # 4. 组合成float4 (sign:1, exp:2, mantissa:1)，存在int8里
    x_quant = (sign << 3) | (exp << 1) | mantissa
    
    # 5. pack and store x_quant
    # Pack two e2m1 elements into a single uint8 along the specified dimension.
    # x_uint8 = tl.reshape(x_quant, x.shape)
    if fuse_pack:
        x_quant_reshaped = tl.reshape(x_quant, (BLK, (C + 1) // 2, 2))
        low, high = tl.split(x_quant_reshaped)
        x_uint8_packed = (high << 4) | low
        # mask = (offs_n[:, None] < L) & (offs_k[None, :] < C//2)
        output_ptrs_packed = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + tl.arange(0, C//2)[None, :]
        tl.store(output_ptrs_packed, x_uint8_packed, mask=offs_n[:, None] < L)
    else:
        x_uint8 = tl.reshape(x_quant, x.shape)
        # x_uint8 = tl.permute(x_uint8, (1, 0))
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
    # e_biased_clamped = tl.permute(e_biased_clamped, (1, 0))
    # tl.store(scale_ptrs, e_biased_clamped.to(tl.uint8)) # , mask=offs_n_32[:, None] < (L - off_blk * BLK)//32)
    # t = shared_scale.to(tl.float8e4nv)
    tl.store(scale_ptrs, e_biased_clamped.to(tl.uint8), mask=offs_n_32[:, None] < (L+31)//32)


@triton.jit
def quant_nvfp4_kernel(Input, Output, Scale, Scale_2, L,
                    stride_iz, stride_ih, stride_in,
                    stride_oz, stride_oh, stride_on,
                    stride_sz, stride_sh, stride_sn,
                    stride_sz_2, stride_sh_2, stride_sn_2,
                    sm_scale,
                    C: tl.constexpr, BLK: tl.constexpr, candidates_ptr=None, 
                    fuse_pack = False,
                    dual_scale_type = 0,
                    dual_scale = False):   # C: head_dim, BLK: BLKQ 128 or BLKK 64
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    offs_n_16 = tl.arange(0, C//16)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + offs_n[:, None] * stride_sn + offs_n_16[None, :]

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)  # x.shape: [BLK, C] -> [token num, head_dim]
    x = x.to(tl.float32)
    x *= sm_scale


    if dual_scale:
        if dual_scale_type == 0:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + off_blk  # per block scale
            scale = tl.max(tl.abs(x)) / (2**3)  # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 1:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + off_blk * stride_sn_2 + offs_k[None, :]  # per channel scale
            scale = tl.max(tl.abs(x), axis=0, keep_dims=True) / (2**3)   # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 2:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + offs_n[:, None] # per token scale
            scale = tl.max(tl.abs(x), axis=1, keep_dims=True) /  (2**3) # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale, mask=offs_n[:, None] < L)
    
    x_reshaped = tl.reshape(x, (BLK, C // 16, 16))   # x_reshaped: [BLKQ (128), headdim // 16, 16]
    abs_max = tl.max(tl.abs(x_reshaped), axis=-1)  # abs_max shape: [BLK, C//16]
    
    # emax_elem = 3
    # shared_exp = tl.floor(tl.log2(abs_max)) - emax_elem 
    # shared_scale = tl.exp2(shared_exp) 

    shared_scale = abs_max / 6
        
    shared_scale_broadcast = tl.broadcast_to(tl.reshape(shared_scale, (BLK, C // 16, 1)), (BLK, C // 16, 16))
    x_quant = x_reshaped / shared_scale_broadcast  # x/e-4 = x * e4     x = x / ss = x / (abs_max / 6) = x * 6 / abs_max
    
    # x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)  # 浮点数的四舍五入
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
    # 4. 组合成float4 (sign:1, exp:2, mantissa:1)，存在int8里
    x_quant = (sign << 3) | (exp << 1) | mantissa
    
    # 5. pack and store x_quant
    # Pack two e2m1 elements into a single uint8 along the specified dimension.
    # x_uint8 = tl.reshape(x_quant, x.shape)
    if fuse_pack:
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
    # e = tl.floor(tl.log2(shared_scale))
    # e_biased = e + 127
    # # e_biased_int = e_biased
    # e_biased_clamped = tl.clamp(e_biased, 0, 254) #.to(tl.int32)
    # import pdb; pdb.set_trace()
    t = shared_scale.to(tl.float8e4nv)
    tl.store(scale_ptrs, t, mask=offs_n[:, None] < L)



@triton.jit
def get_nvfp4_scale_kernel(Input, Scale, L,
                    stride_iz, stride_ih, stride_in,
                    stride_sz, stride_sh, stride_sn,
                    sm_scale,
                    C: tl.constexpr, BLK: tl.constexpr):   # C: head_dim, BLK: BLKQ 128 or BLKK 64
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    offs_n_16 = tl.arange(0, C//16)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + offs_n[:, None] * stride_sn + offs_n_16[None, :]

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)  # x.shape: [BLK, C] -> [token num, head_dim]
    x = x.to(tl.float32)
    x *= sm_scale

    x_reshaped = tl.reshape(x, (BLK, C // 16, 16))   # x_reshaped: [BLKQ (128), headdim // 16, 16]
    abs_max = tl.max(tl.abs(x_reshaped), axis=-1)  # abs_max shape: [BLK, C//16]
    
    shared_scale = abs_max / 6
    tl.store(scale_ptrs, shared_scale.to(tl.float8e4nv), mask=offs_n[:, None] < L)


@triton.jit
def quant_nvfp4_per_channel_kernel(Input, Output, Scale, Scale_2, L,
                    stride_iz, stride_ih, stride_in,
                    stride_oz, stride_oh, stride_on,
                    stride_sz, stride_sh, stride_sn,
                    stride_sz_2, stride_sh_2, stride_sn_2,
                    sm_scale,
                    C: tl.constexpr, BLK: tl.constexpr, candidates_ptr=None, 
                    fuse_pack = False,
                    dual_scale_type = 0,
                    dual_scale = False):   # C: head_dim, BLK: BLKQ 128 or BLKK 64
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    offs_n_16 = off_blk * BLK//16 + tl.arange(0, BLK//16)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + offs_n_16[:, None] * stride_sn + offs_k[None, :]

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L, other=0.0)  # x.shape: [BLK, C] -> [token num, head_dim]
    x = x.to(tl.float32)
    x *= sm_scale
    
    if dual_scale:
        if dual_scale_type == 0:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + off_blk  # per block scale
            scale = tl.max(tl.abs(x)) / 6  # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 1:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + off_blk * stride_sn_2 + offs_k[None, :] # per channel scale
            scale = tl.max(tl.abs(x), axis=0, keep_dims=True) / 6  # mxfp4 range: [-6, 6]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)
        elif dual_scale_type == 2:
            scale_ptrs_2 = Scale_2 + off_b * stride_sz_2 + off_h * stride_sh_2 + offs_n[:, None] # per token scale
            scale = tl.max(tl.abs(x), axis=1, keep_dims=True) / 6  # mxfp4 range: [-6, 6]  x.shape [BLK, C]
            scale += 0.0000001
            x = x / scale
            tl.store(scale_ptrs_2, scale)

    # x_trans = tl.permute(x, (1, 0))
    # import pdb; pdb.set_trace()
    x_reshaped = tl.reshape(x, (BLK//16, 16, C))   # x_reshaped: [BLKQ (128), headdim // 32, 32]
    abs_max = tl.max(tl.abs(x_reshaped), axis=1)  # abs_max shape: [BLK//16, C]  8,128
    
    # emax_elem = 3
    # shared_exp = tl.floor(tl.log2(abs_max)) - emax_elem 
    # shared_scale = tl.exp2(shared_exp) # [BLK//16, C]  # 4, 128
    # import pdb; pdb.set_trace()
    shared_scale = abs_max / 6

    shared_scale_broadcast = tl.broadcast_to(tl.reshape(shared_scale, (BLK//16, 1, C)), (BLK//16, 16, C))  # 4, 32, 128
    x_quant = x_reshaped / shared_scale_broadcast  # x/e-4 = x * e4
    
    # x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)  # 浮点数的四舍五入
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
    # 4. 组合成float4 (sign:1, exp:2, mantissa:1)，存在int8里
    x_quant = (sign << 3) | (exp << 1) | mantissa
    
    # 5. pack and store x_quant
    # Pack two e2m1 elements into a single uint8 along the specified dimension.
    # x_uint8 = tl.reshape(x_quant, x.shape)
    if fuse_pack:
        x_quant_reshaped = tl.reshape(x_quant, (BLK, (C + 1) // 2, 2))
        low, high = tl.split(x_quant_reshaped)
        x_uint8_packed = (high << 4) | low
        # mask = (offs_n[:, None] < L) & (offs_k[None, :] < C//2)
        output_ptrs_packed = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + tl.arange(0, C//2)[None, :]
        tl.store(output_ptrs_packed, x_uint8_packed, mask=offs_n[:, None] < L)
    else:
        x_uint8 = tl.reshape(x_quant, x.shape)
        # x_uint8 = tl.permute(x_uint8, (1, 0))
        output_ptrs_unpacked = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
        tl.store(output_ptrs_unpacked, x_uint8, mask=offs_n[:, None] < L)

    # 6. store scale
    # is_invalid = torch.isnan(shared_scale) | torch.isinf(shared_scale) | (shared_scale <= 0)
    # tl.store(scale_ptrs, 255, mask = ~is_invalid)
    # valid_values = shared_scale[~is_invalid]
    # e = tl.floor(tl.log2(shared_scale))
    # e_biased = e + 127
    # # e_biased_int = e_biased
    # e_biased_clamped = tl.clamp(e_biased, 0, 254) #.to(tl.int32)
    # # e_biased_clamped = tl.permute(e_biased_clamped, (1, 0))
    # tl.store(scale_ptrs, e_biased_clamped.to(tl.uint8)) # , mask=offs_n_32[:, None] < (L - off_blk * BLK)//32)
    t = shared_scale.to(tl.float8e4nv)
    tl.store(scale_ptrs, t, mask=offs_n_16[:, None] < (L+15)//16)
    



    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        if fuse_pack:
            assert BLKQ % 2 == 0, "BLKQ must be even for packing along lastdim"
            assert BLKK % 2 == 0, "BLKK must be even for packing along lastdim"
            # q_fp4 = torch.empty((b, h_qo, qo_len, (head_dim + 1) // 2), dtype=torch.uint8, device=q.device)
            # k_fp4 = torch.empty((b, h_kv, kv_len, (head_dim + 1) // 2), dtype=torch.uint8, device=k.device)
            q_fp4 = torch.empty((b, h_qo, (qo_len+1)//2, head_dim), dtype=torch.uint8, device=q.device)
            k_fp4 = torch.empty((b, h_kv, (kv_len+1)//2, head_dim), dtype=torch.uint8, device=k.device)
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

        
        if fuse_pack:
            assert BLKQ % 2 == 0, "BLKQ must be even for packing along lastdim"
            assert BLKK % 2 == 0, "BLKK must be even for packing along lastdim"
            # q_fp4 = torch.empty((b, qo_len, h_qo, (head_dim + 1) // 2), dtype=torch.uint8, device=q.device)
            # k_fp4 = torch.empty((b, kv_len, h_kv, (head_dim + 1) // 2), dtype=torch.uint8, device=k.device)
            q_fp4 = torch.empty((b, (qo_len+1)//2, h_qo, head_dim), dtype=torch.uint8, device=q.device)
            k_fp4 = torch.empty((b, (kv_len+1)//2, h_kv, head_dim), dtype=torch.uint8, device=k.device)
        else:
            q_fp4 = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
            k_fp4 = torch.empty(k.shape, dtype=torch.uint8, device=k.device)

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_fp4.stride(0), q_fp4.stride(2), q_fp4.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_fp4.stride(0), k_fp4.stride(2), k_fp4.stride(1)

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    # q_scale = torch.empty((b, h_qo, qo_len, head_dim // 32), device=q.device, dtype=torch.uint8)
    # k_scale = torch.empty((b, h_kv, kv_len, head_dim // 32), device=q.device, dtype=torch.uint8)
    q_scale = torch.empty((b, h_qo, (qo_len+15)//16, head_dim), device=q.device, dtype=torch.float8_e4m3fn)
    k_scale = torch.empty((b, h_kv, (kv_len+15)//16, head_dim), device=q.device, dtype=torch.float8_e4m3fn)

    # dual scale: channelwise, blockwise, tokenwise
    dual_scale_type_q = 0 # 0: blockwise, 1: channelwise, 2: tokenwise
    dual_scale_type_k = 0 # 0: blockwise, 1: channelwise, 2: tokenwise
    if quant_granularity == "blockwise":
        dual_scale_type_q = 0
        dual_scale_type_k = 0
        q_scale_2 = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ, 1), device=q.device, dtype=torch.float32)
        k_scale_2 = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK, 1), device=q.device, dtype=torch.float32) 
    elif quant_granularity == "channelwise": # channelwise in blockwise
        dual_scale_type_q = 0
        dual_scale_type_k = 1
        q_scale_2 = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ, 1), device=q.device, dtype=torch.float32)
        k_scale_2 = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK, head_dim), device=q.device, dtype=torch.float32)    
    elif quant_granularity == "tokenwise":
        dual_scale_type_q = 2
        dual_scale_type_k = 2
        q_scale_2 = torch.empty((b, h_qo, qo_len, 1), device=q.device, dtype=torch.float32)
        k_scale_2 = torch.empty((b, h_kv, kv_len, 1), device=q.device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown quant granularity: {quant_granularity}")

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # import pdb; pdb.set_trace()

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_nvfp4_per_channel_kernel[grid](
        q, q_fp4, q_scale, q_scale_2, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        q_scale_2.stride(0), q_scale_2.stride(1), q_scale_2.stride(2),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ,
        fuse_pack=fuse_pack,
        dual_scale_type=dual_scale_type_q,
        dual_scale=dual_scale, 
    )
    
    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_nvfp4_per_channel_kernel[grid](
        k, k_fp4, k_scale, k_scale_2, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        k_scale_2.stride(0), k_scale_2.stride(1), k_scale_2.stride(2),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK,   
        fuse_pack=fuse_pack,
        dual_scale_type=dual_scale_type_k,
        dual_scale=dual_scale, 
    )

    # if dual_scale:
    return q_fp4, q_scale, k_fp4, k_scale, q_scale_2, k_scale_2
    # else:
    #     return q_fp4, q_scale, k_fp4, k_scale, None, None