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

import torch, math
import triton
import triton.language as tl
import torch.nn.functional as F
from spas_sage_attn.utils import hyperparameter_check, get_block_map_meansim
from spas_sage_attn.quant_per_block import per_block_int8
import pdb


@torch.compiler.disable
def online_routing_attn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False, skip_thresh=None):
    assert q.size(-2)>=128, "seq_len should be not less than 128."

    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        k = k - k.mean(dim=-2, keepdim=True)
    # k_block_indices = get_block_map_meansim(q, k, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, attention_sink=attention_sink)  # 
    headdim = q.size(-1)

    assert headdim in [64, 128], "headdim should be in [64, 96, 128]."

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)  # 量化
    # pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    # k_block_indices[:] = 1
    o = forward(q_int8, k_int8, v, q_scale, k_scale,  is_causal=is_causal, tensor_layout="HND", output_dtype=dtype, skip_thresh=skip_thresh)

    return o

def thresh_1(start_n, lo, hi, BLOCK_N):
    return 1 / ((start_n - lo) / BLOCK_N + 1)

def thresh_2(start_n, lo, hi, BLOCK_N):
    return 1 / (((start_n - lo) / BLOCK_N + 1)*2)


@triton.jit
def _attn_fwd_inner(acc, l_i, old_m, q, q_scale, kv_len,
                    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, start_m,  
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    skip_thresh=None):
    if skip_thresh is None:
        skip_thresh = -1e6  # -inf
        
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // BLOCK_N
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    elif STAGE == 3:
        lo, hi = 0, kv_len
    # import pdb; pdb.set_trace()
    # tl.static_print(f"STAGE={STAGE}")
    sink_qk_max = 0.
    for start_n in range(lo, hi, BLOCK_N):
        
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 
        if start_n == lo:
            # baseblock
            # sink_qk_mean = tl.sum(qk) / (BLOCK_M * BLOCK_N)
            sink_qk_max = tl.sum(tl.max(qk, 1)) / BLOCK_M
            cur_qk_max = sink_qk_max
            # tl.static_print(f"sink_qk_max: {sink_qk_max}")
        else:
            # cur_qk_max = tl.max(qk, 1)
            cur_qk_max = tl.sum(tl.max(qk, 1)) / BLOCK_M
            # tl.static_print(f"cur_qk_max: {cur_qk_max}")
            
            # # 根据阈值过滤qk矩阵
            # mask = cur_qk_max > sink_qk_max * thresh_1(start_n, lo, hi, BLOCK_N)
            # # 根据mask筛选出需要保留的qk值
            # # valid_qk = tl.zeros([BLOCK_M, tl.sum(mask)], dtype=tl.float32)
            # # 使用 tl.cumsum 来避免循环
            # mask_cumsum = tl.cumsum(mask.to(tl.int32))
            # valid_qk = qk[mask_cumsum - 1, :]
            # qk = valid_qk
            
            # if cur_qk_max < sink_qk_max * thresh_1(start_n, lo, hi, BLOCK_N):
            # if cur_qk_max >= sink_qk_max * skip_thresh:
                # tl.static_print(f"cur_qk_max: {cur_qk_max}, sink_qk_max: {sink_qk_max}")
                # continue
            
        if start_n == lo or cur_qk_max >= sink_qk_max * skip_thresh: 
            if STAGE == 2:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = qk + tl.where(mask, 0, -1.0e6)
                local_m = tl.max(qk, 1)
                new_m = tl.maximum(old_m, local_m)
                qk -= new_m[:, None]
            else:
                local_m = tl.max(qk, 1)
                new_m = tl.maximum(old_m, local_m)
                qk = qk - new_m[:, None]
            
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(old_m - new_m)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            # if tl.min(new_m - local_m) < pvthreshd:
            v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))

            p = p.to(tl.float16)
            acc += tl.dot(p, v, out_dtype=tl.float16)
            old_m = new_m
            
            K_ptrs += BLOCK_N * stride_kn
            K_scale_ptr += 1
            V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, old_m

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out,  
              stride_qz, stride_qh, stride_qn, # stride_qz: head_num, stride_qh: seq_len, stride_qn: head_dim
              stride_kz, stride_kh, stride_kn,
              stride_vz, stride_vh, stride_vn,  
              stride_oz, stride_oh, stride_on,  
            #   stride_kbidq, stride_kbidk,
              qo_len, kv_len, H:tl.constexpr, num_kv_groups:tl.constexpr,   # qo_len == kv_len 
              HEAD_DIM: tl.constexpr,  # 128
              BLOCK_M: tl.constexpr,  # 128
              BLOCK_N: tl.constexpr,  # 64
              STAGE: tl.constexpr,     # 1, 3
              skip_thresh=None):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    # tl.static_print(f"off_z(batch_idx): {off_z}, off_h(head_idx): {off_h}, start_m(block_idx, {BLOCK_M} token/block): {start_m}")
    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    # k_bid_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * stride_kbidq
    # pvthreshd = tl.load(PVThreshd+off_h)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    # K_bid_ptr = K_blkid + k_bid_offset + start_m * stride_kbidk 
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, 
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n, skip_thresh   # STAGE=1,3 --> 3,1
                                    )
    if STAGE != 1:  # STAGE=3 --> STAGE=2
        acc, l_i, _ = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                       start_m,  
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  
                                        2, offs_m, offs_n, skip_thresh
                                        )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))


def forward(q, k, v, q_scale, k_scale, pvthreshd=None, is_causal=False, tensor_layout="HND", output_dtype=torch.float16, skip_thresh=None):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 3 if is_causal else 1
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":  # default
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    
    assert qo_len == kv_len, "qo_len and kv_len must be equal for causal attention"

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv
    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b   )
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o,  
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        # k_block_id.stride(1), k_block_id.stride(2),
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage,  
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4, skip_thresh=skip_thresh)
    return o
