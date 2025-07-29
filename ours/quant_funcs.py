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
from ours.quant_kernels import quant_fpxint8_kernel, quant_mxfp8_kernel, quant_mxfp8_nvfp4_kernel, quant_nvfp4_kernel, quant_mxfp4_kernel
from ours.mxfp import MXFP4Tensor, MXFP8Tensor

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


def quant_mxfp8(q, k, BLKQ=128, BLKK=128, sm_scale=None, tensor_layout="HND", quant_granularity="blockwise", \
    dual_scale=False, qk_dtype="e5m2", pack_along_lastdim=False):
    q_fp8 = torch.empty(q.shape, dtype=torch.float8_e5m2 if qk_dtype == "e5m2" else torch.float8_e4m3fn, device=q.device)
    k_fp8 = torch.empty(k.shape, dtype=torch.float8_e5m2 if qk_dtype == "e5m2" else torch.float8_e4m3fn, device=k.device)

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

    # q_scale_2 = torch.empty((b, h_qo, qo_len, 1), device=q.device, dtype=torch.float32)
    # k_scale_2 = torch.empty((b, h_kv, kv_len, 1), device=q.device, dtype=torch.float32)

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
    elif quant_granularity == "tensorwise":
        dual_scale_type_q = 3
        dual_scale_type_k = 3
        # import pdb; pdb.set_trace()
        # q_scale_2 = torch.empty((b, h_qo, 1, 1), device=q.device, dtype=torch.float32)
        q_scale_2 = torch.abs(torch.max(q.reshape(b, h_qo, -1), dim=-1, keepdim=True).values.to(torch.float32)) / (2**3)
        # k_scale_2 = torch.empty((b, h_kv, 1, 1), device=q.device, dtype=torch.float32)
        k_scale_2 = torch.abs(torch.max(k.reshape(b, h_kv, -1), dim=-1, keepdim=True).values.to(torch.float32)) / (2**3)
    else:
        raise ValueError(f"Unknown quant granularity: {quant_granularity}")

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    qk_dtype = 0 if qk_dtype == "e5m2" else 1

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_mxfp8_kernel[grid](
        q, q_fp8, q_scale, q_scale_2, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        q_scale_2.stride(0), q_scale_2.stride(1), q_scale_2.stride(2),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ, 
        dual_scale=dual_scale, dual_scale_type=dual_scale_type_q, qk_dtype=qk_dtype
    )

    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_mxfp8_kernel[grid](
        k, k_fp8, k_scale, k_scale_2, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        k_scale_2.stride(0), k_scale_2.stride(1), k_scale_2.stride(2),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK, 
        dual_scale=dual_scale, dual_scale_type=dual_scale_type_k, qk_dtype=qk_dtype
    )

    return q_fp8, q_scale, k_fp8, k_scale, q_scale_2, k_scale_2


   
def quant_mxfp8_nvfp4(q, k, BLKQ=128, BLKK=128, sm_scale=None, tensor_layout="HND", pack_along_lastdim=False, \
    dual_scale=False, v_quant=False, v=None, quant_granularity="tokenwise", qk_dtype="e5m2"):

    q_fp8 = torch.empty(q.shape, dtype=torch.float8_e5m2 if qk_dtype == "e5m2" else torch.float8_e4m3fn, device=q.device)
    k_fp8 = torch.empty(k.shape, dtype=torch.float8_e5m2 if qk_dtype == "e5m2" else torch.float8_e4m3fn, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_fp8.stride(0), k_fp8.stride(1), k_fp8.stride(2)

        if pack_along_lastdim:
            assert BLKQ % 2 == 0, "BLKQ must be even for packing along lastdim"
            assert BLKK % 2 == 0, "BLKK must be even for packing along lastdim"
            q_fp4 = torch.empty((b, h_qo, qo_len, (head_dim + 1) // 2), dtype=torch.uint8, device=q.device)
            k_fp4 = torch.empty((b, h_kv, kv_len, (head_dim + 1) // 2), dtype=torch.uint8, device=k.device)
        else:
            q_fp4 = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
            k_fp4 = torch.empty(k.shape, dtype=torch.uint8, device=k.device)

    # elif tensor_layout == "NHD":
    #     b, qo_len, h_qo, head_dim = q.shape
    #     _, kv_len, h_kv, _ = k.shape

    #     stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
    #     stride_bz_qo, stride_h_qo, stride_seq_qo = q_fp8.stride(0), q_fp8.stride(2), q_fp8.stride(1)
    #     stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
    #     stride_bz_ko, stride_h_ko, stride_seq_ko = k_fp8.stride(0), k_fp8.stride(2), k_fp8.stride(1)
    # else:
    #     raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale_fp8 = torch.empty((b, h_qo, qo_len, head_dim // 32), device=q.device, dtype=torch.uint8)
    k_scale_fp8 = torch.empty((b, h_kv, kv_len, head_dim // 32), device=q.device, dtype=torch.uint8)

    q_scale_fp4 = torch.empty((b, h_qo, qo_len, head_dim // 16), device=q.device, dtype=torch.float8_e4m3fn)
    k_scale_fp4 = torch.empty((b, h_kv, kv_len, head_dim // 16), device=q.device, dtype=torch.float8_e4m3fn)

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
    elif quant_granularity == "tensorwise":
        dual_scale_type_q = 3
        dual_scale_type_k = 3
        # q_scale_2 = torch.empty((b, h_qo, 1, 1), device=q.device, dtype=torch.float32)
        q_scale_2 = torch.abs(torch.max(q.reshape(b, h_qo, -1), dim=-1, keepdim=True).values) / (2**11)
        # k_scale_2 = torch.empty((b, h_kv, 1, 1), device=q.device, dtype=torch.float32)
        k_scale_2 = torch.abs(torch.max(k.reshape(b, h_kv, -1), dim=-1, keepdim=True).values) / (2**11)
    else:
        raise ValueError(f"Unknown quant granularity: {quant_granularity}")

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    qk_dtype = 0 if qk_dtype == "e5m2" else 1

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_mxfp8_nvfp4_kernel[grid](
        q, q_fp8, q_fp4, q_scale_fp8, q_scale_fp4, q_scale_2, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2), 
        q_fp4.stride(0), q_fp4.stride(1), q_fp4.stride(2), 
        q_scale_fp8.stride(0), q_scale_fp8.stride(1), q_scale_fp8.stride(2),
        q_scale_fp4.stride(0), q_scale_fp4.stride(1), q_scale_fp4.stride(2),
        q_scale_2.stride(0), q_scale_2.stride(1), q_scale_2.stride(2),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ,
        dual_scale_type=dual_scale_type_q, dual_scale=dual_scale,
        pack_along_lastdim=pack_along_lastdim, 
        qk_dtype=qk_dtype
    )

    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_mxfp8_nvfp4_kernel[grid](
        k, k_fp8, k_fp4, k_scale_fp8, k_scale_fp4, k_scale_2, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        k_fp8.stride(0), k_fp8.stride(1), k_fp8.stride(2), 
        k_fp4.stride(0), k_fp4.stride(1), k_fp4.stride(2),
        k_scale_fp8.stride(0), k_scale_fp8.stride(1), k_scale_fp8.stride(2),
        k_scale_fp4.stride(0), k_scale_fp4.stride(1), k_scale_fp4.stride(2),
        k_scale_2.stride(0), k_scale_2.stride(1), k_scale_2.stride(2),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK,
        dual_scale_type=dual_scale_type_k, dual_scale=dual_scale,
        pack_along_lastdim=pack_along_lastdim,
        qk_dtype=qk_dtype
    )

    if not pack_along_lastdim:
        q_fp4 = MXFP4Tensor(data=q_fp4, dtype=torch.uint8)
        q_fp4 = q_fp4.to_packed_tensor(dim=len(q_fp4.data.shape) - 1)
        k_fp4 = MXFP4Tensor(data=k_fp4, dtype=torch.uint8)
        k_fp4 = k_fp4.to_packed_tensor(dim=len(k_fp4.data.shape) - 1)    

    return q_fp8, q_scale_fp8, k_fp8, k_scale_fp8, q_fp4, q_scale_fp4, k_fp4, k_scale_fp4, q_scale_2, k_scale_2



def quant_mxfp4(q, k, BLKQ=128, BLKK=128, sm_scale=None, tensor_layout="HND", VEC_SIZE=32, \
    pack_along_lastdim=False, dual_scale=False, quant_granularity="blockwise", qk_dtype=None):

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

    q_scale = torch.empty((b, h_qo, qo_len, head_dim // VEC_SIZE), device=q.device, dtype=torch.uint8)
    k_scale = torch.empty((b, h_kv, kv_len, head_dim // VEC_SIZE), device=q.device, dtype=torch.uint8)

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
    elif quant_granularity == "tensorwise":
        dual_scale_type_q = 3
        dual_scale_type_k = 3
        # q_scale_2 = torch.empty((b, h_qo, 1, 1), device=q.device, dtype=torch.float32)
        q_scale_2 = torch.abs(torch.max(q.reshape(b, h_qo, -1), dim=-1, keepdim=True).values) / (2**3)
        # k_scale_2 = torch.empty((b, h_kv, 1, 1), device=q.device, dtype=torch.float32)
        k_scale_2 = torch.abs(torch.max(k.reshape(b, h_kv, -1), dim=-1, keepdim=True).values) / (2**3)
    else:
        raise ValueError(f"Unknown quant granularity: {quant_granularity}")

    if sm_scale is None:
        sm_scale = head_dim**-0.5


    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_mxfp4_kernel[grid](
        q, q_fp4, q_scale, q_scale_2, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        q_scale_2.stride(0), q_scale_2.stride(1), q_scale_2.stride(2),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ,
        pack_along_lastdim=pack_along_lastdim,
        dual_scale_type=dual_scale_type_q,
        dual_scale=dual_scale, 
    )
    
    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_mxfp4_kernel[grid](
        k, k_fp4, k_scale, k_scale_2, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        k_scale_2.stride(0), k_scale_2.stride(1), k_scale_2.stride(2),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK,   
        pack_along_lastdim=pack_along_lastdim,
        dual_scale_type=dual_scale_type_k,
        dual_scale=dual_scale, 
    )


    if not pack_along_lastdim:
        q_fp4 = MXFP4Tensor(data=q_fp4, dtype=torch.uint8)
        q_fp4 = q_fp4.to_packed_tensor(dim=len(q_fp4.data.shape) - 1)
        k_fp4 = MXFP4Tensor(data=k_fp4, dtype=torch.uint8)
        k_fp4 = k_fp4.to_packed_tensor(dim=len(k_fp4.data.shape) - 1)

    return q_fp4, q_scale, k_fp4, k_scale, q_scale_2, k_scale_2
    # else:
    #     return q_fp4, q_scale, k_fp4, k_scale, None, None


def quant_mxfp4_per_channel(q, k, BLKQ=128, BLKK=128, sm_scale=None, tensor_layout="HND", pack_along_lastdim=False, dual_scale=False, quant_granularity="blockwise"):

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        if pack_along_lastdim:
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

        
        if pack_along_lastdim:
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
    q_scale = torch.empty((b, h_qo, (qo_len+31)//32, head_dim), device=q.device, dtype=torch.uint8)
    k_scale = torch.empty((b, h_kv, (kv_len+31)//32, head_dim), device=q.device, dtype=torch.uint8)

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
    elif quant_granularity == "tensorwise":
        dual_scale_type_q = 3
        dual_scale_type_k = 3
        # q_scale_2 = torch.empty((b, h_qo, 1, 1), device=q.device, dtype=torch.float32)
        q_scale_2 = torch.abs(torch.max(q.reshape(b, h_qo, -1), dim=-1, keepdim=True).values)
        # k_scale_2 = torch.empty((b, h_kv, 1, 1), device=q.device, dtype=torch.float32)
        k_scale_2 = torch.abs(torch.max(k.reshape(b, h_kv, -1), dim=-1, keepdim=True).values)
    else:
        raise ValueError(f"Unknown quant granularity: {quant_granularity}")

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # import pdb; pdb.set_trace()

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_mxfp4_per_channel_kernel[grid](
        q, q_fp4, q_scale, q_scale_2, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        q_scale_2.stride(0), q_scale_2.stride(1), q_scale_2.stride(2),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ,
        pack_along_lastdim=pack_along_lastdim,
        dual_scale_type=dual_scale_type_q,
        dual_scale=dual_scale, 
    )
    
    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_mxfp4_per_channel_kernel[grid](
        k, k_fp4, k_scale, k_scale_2, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        k_scale_2.stride(0), k_scale_2.stride(1), k_scale_2.stride(2),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK,   
        pack_along_lastdim=pack_along_lastdim,
        dual_scale_type=dual_scale_type_k,
        dual_scale=dual_scale, 
    )

    # if dual_scale:
    return q_fp4, q_scale, k_fp4, k_scale, q_scale_2, k_scale_2
    # else:
    #     return q_fp4, q_scale, k_fp4, k_scale, None, None


def quant_nvfp4(q, k, BLKQ=128, BLKK=128, sm_scale=None, tensor_layout="HND", VEC_SIZE=16, \
    pack_along_lastdim=False, dual_scale=False, quant_granularity="blockwise", qk_dtype=None):

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

    q_scale = torch.empty((b, h_qo, qo_len, head_dim // VEC_SIZE), device=q.device, dtype=torch.float8_e4m3fn)
    k_scale = torch.empty((b, h_kv, kv_len, head_dim // VEC_SIZE), device=q.device, dtype=torch.float8_e4m3fn)

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
    elif quant_granularity == "tensorwise":
        dual_scale_type_q = 3
        dual_scale_type_k = 3
        # q_scale_2 = torch.empty((b, h_qo, 1, 1), device=q.device, dtype=torch.float32)
        q_scale_2 = torch.abs(torch.max(q.reshape(b, h_qo, -1), dim=-1, keepdim=True))
        # k_scale_2 = torch.empty((b, h_kv, 1, 1), device=q.device, dtype=torch.float32)
        k_scale_2 = torch.abs(torch.max(k.reshape(b, h_kv, -1), dim=-1, keepdim=True))
    else:
        raise ValueError(f"Unknown quant granularity: {quant_granularity}")

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # import pdb; pdb.set_trace()
    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_nvfp4_kernel[grid](
        q, q_fp4, q_scale, q_scale_2, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        q_scale_2.stride(0), q_scale_2.stride(1), q_scale_2.stride(2),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ,
        pack_along_lastdim=pack_along_lastdim,
        dual_scale_type=dual_scale_type_q,
        dual_scale=dual_scale, 
    )
    
    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_nvfp4_kernel[grid](
        k, k_fp4, k_scale, k_scale_2, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        k_scale_2.stride(0), k_scale_2.stride(1), k_scale_2.stride(2),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK,   
        pack_along_lastdim=pack_along_lastdim,
        dual_scale_type=dual_scale_type_k,
        dual_scale=dual_scale, 
    )

    if not pack_along_lastdim:
        q_fp4 = MXFP4Tensor(data=q_fp4, dtype=torch.uint8)
        q_fp4 = q_fp4.to_packed_tensor(dim=len(q_fp4.data.shape) - 1)
        k_fp4 = MXFP4Tensor(data=k_fp4, dtype=torch.uint8)
        k_fp4 = k_fp4.to_packed_tensor(dim=len(k_fp4.data.shape) - 1)

    return q_fp4, q_scale, k_fp4, k_scale, q_scale_2, k_scale_2


def get_nvfp4_scale(q, k, BLKQ=128, BLKK=128, sm_scale=None, tensor_layout="HND", VEC_SIZE=16):

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        # if pack_along_lastdim:
        #     assert BLKQ % 2 == 0, "BLKQ must be even for packing along lastdim"
        #     assert BLKK % 2 == 0, "BLKK must be even for packing along lastdim"
        #     q_fp4 = torch.empty((b, h_qo, qo_len, (head_dim + 1) // 2), dtype=torch.uint8, device=q.device)
        #     k_fp4 = torch.empty((b, h_kv, kv_len, (head_dim + 1) // 2), dtype=torch.uint8, device=k.device)
        # else:
        #     q_fp4 = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
        #     k_fp4 = torch.empty(k.shape, dtype=torch.uint8, device=k.device)

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        # stride_bz_qo, stride_h_qo, stride_seq_qo = q_fp4.stride(0), q_fp4.stride(1), q_fp4.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        # stride_bz_ko, stride_h_ko, stride_seq_ko = k_fp4.stride(0), k_fp4.stride(1), k_fp4.stride(2)

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        # if pack_along_lastdim:
        #     assert BLKQ % 2 == 0, "BLKQ must be even for packing along lastdim"
        #     assert BLKK % 2 == 0, "BLKK must be even for packing along lastdim"
        #     q_fp4 = torch.empty((b, qo_len, h_qo, (head_dim + 1) // 2), dtype=torch.uint8, device=q.device)
        #     k_fp4 = torch.empty((b, kv_len, h_kv, (head_dim + 1) // 2), dtype=torch.uint8, device=k.device)
        # else:
        #     q_fp4 = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
        #     k_fp4 = torch.empty(k.shape, dtype=torch.uint8, device=k.device)

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        # stride_bz_qo, stride_h_qo, stride_seq_qo = q_fp4.stride(0), q_fp4.stride(2), q_fp4.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        # stride_bz_ko, stride_h_ko, stride_seq_ko = k_fp4.stride(0), k_fp4.stride(2), k_fp4.stride(1)

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, qo_len, head_dim // VEC_SIZE), device=q.device, dtype=torch.float8_e4m3fn)
    k_scale = torch.empty((b, h_kv, kv_len, head_dim // VEC_SIZE), device=q.device, dtype=torch.float8_e4m3fn)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    get_nvfp4_scale_kernel[grid](
        q, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim, BLK=BLKQ,
        
    )
    
    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    get_nvfp4_scale_kernel[grid](
        k, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        sm_scale=1.0,
        C=head_dim, BLK=BLKK,   
    )
    return q_scale, k_scale

def quant_nvfp4_per_channel(q, k, BLKQ=128, BLKK=128, sm_scale=None, tensor_layout="HND", pack_along_lastdim=False, dual_scale=False, quant_granularity="blockwise"):

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        if pack_along_lastdim:
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

        
        if pack_along_lastdim:
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
        pack_along_lastdim=pack_along_lastdim,
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
        pack_along_lastdim=pack_along_lastdim,
        dual_scale_type=dual_scale_type_k,
        dual_scale=dual_scale, 
    )

    # if dual_scale:
    return q_fp4, q_scale, k_fp4, k_scale, q_scale_2, k_scale_2
    # else:
    #     return q_fp4, q_scale, k_fp4, k_scale, None, None