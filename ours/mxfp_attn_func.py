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
from ours.quant_funcs import quant_mxfp8, quant_mxfp4, quant_nvfp4, get_nvfp4_scale, quant_mxfp8_nvfp4, quant_mxfp8_nvfp4, quant_mxfp8, quant_nvfp4_per_channel
import os
from ours.mxfp_attn_kernel import block_scaled_batched_attn_kernel, block_scaled_batched_attn_kernel_mp_diag_pre_quant, block_scaled_batched_attn_kernel_mp_sink_pre_quant


@torch.compiler.disable
def mxfp_attn_kernel(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=False, attention_sink=False, tensor_layout="HND",
                    output_dtype=torch.bfloat16,  block_scale_type="mxfp4", dual_scale=False,\
                    save_qk=False, quant_granularity="tokenwise", fuse_mp_quant=True, pre_quant=True, diag_tile=0, sink_tile=0, qk_dtype='e5m2', \
                    fuse_pack=True):

    if diag_tile == 0 and sink_tile == 0 and block_scale_type == "mixed":
        block_scale_type = "nvfp4"
    elif block_scale_type != "mixed":
        diag_tile = 0
        sink_tile = 0

    torch.cuda.set_device(v.device)
    if 'diag' in block_scale_type and (not pre_quant):
        assert fuse_mp_quant == False, "fuse_mp_quant must be False when using oneline quant"

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
    q_scale_fp4, k_scale_fp4, q_fp4, k_fp4 = None, None, None, None
    
    kwargs = {
        "BLKQ": BLKQ,
        "BLKK": BLKK,
        "quant_granularity": quant_granularity,
        "dual_scale": dual_scale,
        "qk_dtype": qk_dtype,
        "fuse_pack": fuse_pack,
    }
   
    if block_scale_type == "mxfp4":
        q_quant, q_scale, k_quant, k_scale, q_scale_2, k_scale_2 = quant_mxfp4(q, k, **kwargs)

    elif block_scale_type == "nvfp4":
        q_quant, q_scale, k_quant, k_scale, q_scale_2, k_scale_2 = quant_nvfp4(q, k, **kwargs)

    elif block_scale_type == "mxfp8":
        q_quant, q_scale, k_quant, k_scale, q_scale_2, k_scale_2 = quant_mxfp8(q, k, **kwargs)

    elif block_scale_type == 'mixed' and fuse_mp_quant:
        assert diag_tile + sink_tile > 0, "diag_tile + sink_tile must be greater than 0"
        q_quant, q_scale, k_quant, k_scale, q_fp4, q_scale_fp4, k_fp4, k_scale_fp4, q_scale_2, k_scale_2 = \
            quant_mxfp8_nvfp4(q, k, **kwargs)

        q_fp4, k_fp4 = q_fp4.contiguous(), k_fp4.contiguous()

    elif block_scale_type == 'mixed' and not fuse_mp_quant:
        assert diag_tile + sink_tile > 0, "diag_tile + sink_tile must be greater than 0"
        q_quant, q_scale, k_quant, k_scale, q_scale_2, k_scale_2 = quant_mxfp8(q, k, **kwargs)

        q_fp4, q_scale_fp4, k_fp4, k_scale_fp4, q_scale_2, k_scale_2 = quant_nvfp4(q, k, **kwargs)
        q_fp4, k_fp4 = q_fp4.contiguous(), k_fp4.contiguous()
    
    q_quant = q_quant.contiguous()
    k_quant = k_quant.contiguous()
    
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    q_scale = pad_reshape_scale_factor(q_scale, B, H, M, K, VEC_SIZE, ((M + 127) // 128) * 128)
    k_scale = pad_reshape_scale_factor(k_scale, B, H, N, K, VEC_SIZE, ((N + 127) // 128) * 128)
    if block_scale_type == 'mixed':
        q_scale_fp4 = pad_reshape_scale_factor(q_scale_fp4, B, H, M, K, 16, ((M + 127) // 128) * 128)
        k_scale_fp4 = pad_reshape_scale_factor(k_scale_fp4, B, H, N, K, 16, ((N + 127) // 128) * 128)

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
            q_quant, q_scale, k_quant, k_scale, \
            q_fp4, q_scale_fp4, k_fp4, k_scale_fp4, \
            q_scale_2, k_scale_2, \
            v, is_causal, output_dtype, B, H, M, N, K, configs, save_qk=save_qk, \
            dual_scale=dual_scale, quant_granularity=quant_granularity, \
            diag_tile=diag_tile, sink_tile=sink_tile
    )   
    return output


def pad_reshape_scale_factor(q_scale, B, H, M, K, VEC_SIZE, M_padded):
    # 扩展 q_scale 和 k_scale 的第2维度为128的倍数
    # M_padded = ((M + 127) // 128) * 128 
    # N_padded = ((N + 127) // 128) * 128 
    # 扩展 q_scale
    if M_padded > M:
        q_scale_padded = torch.zeros(B, H, M_padded, K//VEC_SIZE, device=q_scale.device, dtype=q_scale.dtype)
        q_scale_padded[:, :, :M, :] = q_scale
        q_scale = q_scale_padded
       
    q_scale = q_scale.reshape(B, H, M_padded//128, 4, 32, K//VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    return q_scale

def block_scaled_batched_attn(a_desc, a_scale, b_desc, b_scale,  \
                            a_nvfp4, a_scale_nvfp4, b_nvfp4, b_scale_nvfp4, \
                            a_scale_2, b_scale_2, \
                            v_ori, is_causal, dtype_dst, B, H, M, N, K, configs, save_qk=False, \
                            dual_scale=False, quant_granularity="blockwise", \
                            diag_tile=1, sink_tile=1):
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
    elif quant_granularity == "tensorwise":
        quant_granularity = 3
    else:
        raise ValueError(f"Unsupported quant_granularity: {quant_granularity}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]

    # 设置三维grid: 参考_attn_fwd的grid设置 (M*N的块数, head数, batch数)
    # grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), H, B)
    _, h_qo, qo_len, _ = a_desc.shape
    _, h_kv, kv_len, _ = b_desc.shape

    qk_dtype = 0 if a_desc.dtype == torch.float8_e5m2 else 1
    
    grid = (triton.cdiv(qo_len, BLOCK_M), H, B)

    num_kv_groups = h_qo // h_kv
    if diag_tile > 0:
        block_scaled_batched_attn_kernel_mp_diag_pre_quant[grid](
            a_desc, a_scale, a_scale_2, a_nvfp4, a_scale_nvfp4,
            b_desc, b_scale, b_scale_2, b_nvfp4, b_scale_nvfp4,
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
            save_qk=save_qk, dual_scale=dual_scale, quant_granularity=quant_granularity,
            diag_tile=diag_tile, sink_tile=sink_tile, qk_dtype=qk_dtype)
    elif sink_tile > 0:
        # print("block_scaled_batched_attn_kernel_mp_sink_pre_quant")
        block_scaled_batched_attn_kernel_mp_sink_pre_quant[grid](
            a_desc, a_scale, a_scale_2, a_nvfp4, a_scale_nvfp4,
            b_desc, b_scale, b_scale_2, b_nvfp4, b_scale_nvfp4,
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
            save_qk=save_qk, dual_scale=dual_scale, quant_granularity=quant_granularity,
            sink_tile=sink_tile, qk_dtype=qk_dtype)
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
            save_qk=save_qk, dual_scale=dual_scale, quant_granularity=quant_granularity, 
            qk_dtype=qk_dtype)

    return output
