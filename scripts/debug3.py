import torch
import os
from ours.mxfp_attn_kernel import mxfp_attn_kernel
# from ours.modify_mxfp_attn import precision_metric
from torch.nn import functional as F
from tests.test_quant import quant_mxfp8e5, test_quant_mxfp4_input_quant_tensor
from ours.quant_kernels import quant_mxfp4, quant_nvfp4_per_channel, quant_nvfp4, quant_mxfp4_per_channel
from tests.flash_attn_triton import _flash_attn_forward, flash_attn_func #, flash_attn_varlen_func
from tests.flash_attn_triton_residual import _flash_attn_forward_residual
from einops import rearrange, repeat
import math

def mxfp_attn_debug(q, k, v, is_causal=False, output_dtype=torch.float32, block_scale_type="mxfp8+nvfp4", smooth_k=False):
    '''q = q.to(torch.float16)
    k = k.to(torch.float16)
    v = v.to(torch.float16)
    '''
    # sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale = 1.0 / math.sqrt(q.shape[-1]))
    # return sdpa.to(output_dtype)

    torch.cuda.set_device(v.device)

    # dtype = q.dtype
    # if dtype == torch.float32 or dtype == torch.float16:
    #     q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    # else:
    #     q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    B, H, M, K = q.shape 
    N = k.shape[2]
    qo_len = M
    kv_len = N
    '''
    qk = torch.matmul(q, k.transpose(-2, -1)) * 1 / math.sqrt(q.shape[-1])
    mask = torch.full((M, N), float('-inf'), device=q.device)
    mask = torch.triu(mask, diagonal=1)  # 上三角为 -inf
    qk = qk + mask  # broadcasting 会自动扩展成 (B, H, M, N)

    qk_softmax = torch.nn.functional.softmax(qk.to(torch.float32), dim=-1)
    # 计算 qkv 结果
    qkv = torch.matmul(qk_softmax.to(v.dtype), v)
    
    # precision_metric(qkv, sdpa)

    '''
    '''q = q.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    k = k.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    v = v.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    
    fa = flash_attn_func(q, k, v, None, True, 1.0 / math.sqrt(K))
    fa = fa.permute(0, 2, 1, 3).contiguous()

    # precision_metric(fa, qkv)

    return fa.to(output_dtype)'''
    # return qkv.to(output_dtype)


    # import pdb; pdb.set_trace()
    # qk = torch.matmul(q, k.transpose(-2, -1)) * (K ** -0.5)
    # 构建 causal mask，下三角为 True，其他为 False
    # mask = torch.tril(torch.ones(M, N, dtype=torch.bool, device=q.device))  # (M, N)
    # mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, M, N)
    # qk = qk.masked_fill(~mask, float('-inf'))  # 只允许当前 token 及其之前的参与 attention

    assert K in [128], "headdim should be in [128]."

    BLKQ = 128
    BLKK = 128
    ret_dict = None
    # VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    if block_scale_type == "mxfp4":
        # q_mean = torch.mean(q, dim=-2, keepdim=True)
        if smooth_k: k = k - k.mean(dim=-2, keepdim=True)
        # q = q - q_mean
        pack_along_lastdim = False  # True: kernel fusion support for pack fp4 tensor into uint8 tensor
        a_fp4, a_scale, b_fp4, b_scale, a_scale_2, b_scale_2 = quant_mxfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            pack_along_lastdim=pack_along_lastdim, dual_scale=False, quant_granularity="tokenwise")
        qk_quant, a_deq, b_deq = compute_reference(a_fp4, a_scale, b_fp4, b_scale, VEC_SIZE=32, M=M, K=K, N=N, per_channel=False)
        a_deq = a_deq / ((K ** -0.5) * 1.44269504)

        a_deq = a_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        b_deq = b_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        v = v.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        # out, lse, softmax_scale = _flash_attn_forward(a_deq, b_deq, v, bias=None, causal=True, softmax_scale=None)
        out = flash_attn_func(a_deq, b_deq, v, None, True, None)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out.to(output_dtype)
    elif block_scale_type == "nvfp4":
        # q_mean = torch.mean(q, dim=-2, keepdim=True)
        if smooth_k: k = k - k.mean(dim=-2, keepdim=True)
        # q = q - q_mean
        pack_along_lastdim = False  # True: kernel fusion support for pack fp4 tensor into uint8 tensor
        a_fp4, a_scale, b_fp4, b_scale, a_scale_2, b_scale_2 = quant_nvfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            pack_along_lastdim=pack_along_lastdim, dual_scale=True, quant_granularity="tokenwise")
        qk_quant, a_deq, b_deq = compute_reference_nvfp4(a_fp4, a_scale, b_fp4, b_scale, VEC_SIZE=16, M=M, K=K, N=N, per_channel=False)
        a_deq = a_deq / ((K ** -0.5) * 1.44269504)
        a_deq *= a_scale_2
        b_deq *= b_scale_2

        a_deq = a_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        b_deq = b_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        v = v.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        out = flash_attn_func(a_deq, b_deq, v, None, True, None)
        # out, lse, softmax_scale = _flash_attn_forward_residual(a_deq, a_deq, b_deq, b_deq, v, bias=None, causal=True, softmax_scale=None)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out.to(output_dtype)

    elif block_scale_type == "mxfp8+nvfp4":
        # mxfp8 quantization
        if smooth_k: k = k - k.mean(dim=-2, keepdim=True)
        a_fp8, a_scale, b_fp8, b_scale = quant_mxfp8e5(q, k, BLKQ=BLKQ, BLKK=BLKK)
        qk_quant, a_fp8_deq, b_fp8_deq = compute_reference_mxfp8(a_fp8, a_scale, b_fp8, b_scale, VEC_SIZE=32, M=M, K=K, N=N, per_channel=False)
        a_fp8_deq /= ((K ** -0.5) * 1.44269504)

        # nvfp4 quantization    
        pack_along_lastdim = False  # True: kernel fusion support for pack fp4 tensor into uint8 tensor
        a_fp4, a_scale, b_fp4, b_scale, a_scale_2, b_scale_2 = quant_nvfp4(a_fp8_deq, b_fp8_deq, BLKQ=BLKQ, BLKK=BLKK, \
            pack_along_lastdim=pack_along_lastdim, dual_scale=True, quant_granularity="tokenwise")
        qk_quant, a_fp4_deq, b_fp4_deq = compute_reference_nvfp4(a_fp4, a_scale, b_fp4, b_scale, VEC_SIZE=16, M=M, K=K, N=N, per_channel=False)
        a_fp4_deq *= a_scale_2
        b_fp4_deq *= b_scale_2
        a_fp4_deq /= ((K ** -0.5) * 1.44269504)

        a_fp8_deq = a_fp8_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        b_fp8_deq = b_fp8_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        a_fp4_deq = a_fp4_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        b_fp4_deq = b_fp4_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        v = v.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)

        out, lse, softmax_scale = _flash_attn_forward_residual(a_fp8_deq, a_fp4_deq, b_fp8_deq, b_fp4_deq, v, bias=None, causal=True, softmax_scale=None)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out.to(output_dtype)

    elif block_scale_type == "mxfp8":
        if smooth_k: k = k - k.mean(dim=-2, keepdim=True)
        a_fp8, a_scale, b_fp8, b_scale = quant_mxfp8e5(q, k, BLKQ=BLKQ, BLKK=BLKK)
        qk_quant, a_fp8_deq, b_fp8_deq = compute_reference_mxfp8(a_fp8, a_scale, b_fp8, b_scale, VEC_SIZE=32, M=M, K=K, N=N, per_channel=False)
        a_fp8_deq /= ((K ** -0.5) * 1.44269504)
        a_fp8_deq = a_fp8_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        b_fp8_deq = b_fp8_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        v = v.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        
        out = flash_attn_func(a_fp8_deq, b_fp8_deq, v, None, True, None)
        # out, lse, softmax_scale = _flash_attn_forward_residual(a_deq, a_deq, b_deq, b_deq, v, bias=None, causal=True, softmax_scale=None)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out.to(output_dtype)
        
    else:
        raise ValueError(f"Unknown block scale type: {block_scale_type}")

    # a_deq *= a_scale_2
    # b_deq *= b_scale_2

    # precision_metric(a_deq, q)
    # precision_metric(b_deq, k)

    '''
    residual_q = q - a_deq
    residual_k = k - b_deq

    # residual_q = residual_q - resq_deq
    # residual_k = residual_k - resk_deq
    # a_deq += resq_deq
    # b_deq += resk_deq

    dim_sum_q = torch.sum((residual_q), dim=-2) # + torch.sum((residual_k), dim=-2)  # over M
    dim_sum_k = torch.sum((residual_k), dim=-2)

    # 取每个 head 内残差最大的 10 行索引: shape [B, H, 10]
    top10_indices_q = torch.topk(dim_sum_q, k=16, dim=-1).indices  # along last dim (D)
    top10_indices_k = torch.topk(dim_sum_k, k=16, dim=-1).indices  # along last dim (D)

    # 创建一个mask来标记top10的位置
    B, H, D = dim_sum_q.shape
    mask = torch.zeros((B, H, D), dtype=torch.bool, device=dim_sum_q.device)
    
    # 在每个batch和head中标记top10的位置为True
    mask.scatter_(2, top10_indices_q, True)
    mask.scatter_(2, top10_indices_k, True)

    # 扩展mask维度以匹配输入tensor
    mask = mask.unsqueeze(-2)  # [B, H, 1, D]
    
    # 使用where函数: 如果在top10 indices内保留原值,否则置零
    # top10_q0 = torch.where(mask, a_deq, torch.zeros_like(a_deq))
    # top10_k0 = torch.where(mask, b_deq, torch.zeros_like(b_deq))
    top10_resq = torch.where(mask, residual_q, torch.zeros_like(residual_q))
    top10_resk = torch.where(mask, residual_k, torch.zeros_like(residual_k))

    # top10_resq_fp4, top10_resq_scale, top10_resk_fp4, top10_resk_scale, _, _ = quant_mxfp4(top10_resq, top10_resk, BLKQ=BLKQ, BLKK=BLKK, \
    #         pack_along_lastdim=pack_along_lastdim, dual_scale=False, quant_granularity="tokenwise")
    # top10_resq_quant, top10_resq_deq, top10_resk_deq = compute_reference(top10_resq_fp4, top10_resq_scale, top10_resk_fp4, top10_resk_scale, VEC_SIZE=VEC_SIZE, M=M, K=K, N=N)
    # top10_resq_deq = top10_resq_deq / ((K ** -0.5) * 1.44269504)

    # resq_fp4, resq_scale, resk_fp4, resk_scale = quant_mxfp8e5(top10_resq, top10_resk, BLKQ=BLKQ, BLKK=BLKK)
    # # resq_quant, resq_deq, resk_deq = compute_reference(resq_fp4, resq_scale, resk_fp4, resk_scale, VEC_SIZE=VEC_SIZE, M=M, K=K, N=N)
    # resq_deq, resk_deq = compute_dequant_mxfp8e5(resq_fp4, resq_scale, resk_fp4, resk_scale)
    # resq_deq = resq_deq / ((K ** -0.5) * 1.44269504)

    # resq_deq = resq_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    # resk_deq = resk_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    top10_resq = top10_resq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    top10_resk = top10_resk.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    # top10_resq_deq = top10_resq_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    # top10_resk_deq = top10_resk_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    
    a_deq = a_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    b_deq = b_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    # q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    v = v.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    residual_q = residual_q.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    residual_k = residual_k.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    '''

    # q, k, v must permute before fa triton kernel 
    a_fp8_deq = a_fp8_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    b_fp8_deq = b_fp8_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    a_fp4_deq = a_fp4_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    b_fp4_deq = b_fp4_deq.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    v = v.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)

    # out = flash_attn_func(a_deq, b_deq, v, None, True, 1.0 / math.sqrt(K))
    # out = out.permute(0, 2, 1, 3).contiguous()

    # q_mean = q_mean.repeat(1, 1, M, 1).permute(0, 2, 1, 3).to(torch.bfloat16)
    # out, lse, softmax_scale = _flash_attn_forward(a_deq, b_deq, v, bias=None, causal=True, softmax_scale=None)
    out, lse, softmax_scale = _flash_attn_forward_residual(a_fp8_deq, a_fp4_deq, b_fp8_deq, b_fp4_deq, v, bias=None, causal=True, softmax_scale=None)
    out = out.permute(0, 2, 1, 3).contiguous()
    # precision_metric(out, sdpa)
    # import pdb; pdb.set_trace()
    return out.to(output_dtype)

    
    # qk_res1 = residual_q @ b_deq.transpose(-2, -1)
    # qk_res2 = a_deq @ residual_k.transpose(-2, -1) 
    # qk_res3 = residual_q @ residual_k.transpose(-2, -1)
    # qk_quant += qk_res3
    '''qk_quant += q_mean.float() @ b_deq.float().transpose(-2, -1) * (K ** -0.5)   # sage2'''
    # qk_quant += qk_res1 + qk_res2 + qk_res3

    # import pdb; pdb.set_trace()
    '''qk = torch.matmul(q, k.transpose(-2, -1)) * (K ** -0.5)
    mask = torch.full((M, N), float('-inf'), device=q.device)
    mask = torch.triu(mask, diagonal=1)  # 上三角为 -inf
    qk = qk_quant + mask  # broadcasting 会自动扩展成 (B, H, M, N)

    qk_softmax = torch.nn.functional.softmax(qk, dim=-1)
    # 计算 qkv 结果
    qkv = torch.matmul(qk_softmax.to(torch.float16), v.to(torch.float16))

    return qkv.to(output_dtype)'''
           
           

def test_add_resisual_flash_attn(q, k, v, is_causal=False, output_dtype=torch.float32, block_scale_type="mxfp4"):
    # return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    B, H, M, K = q.shape 
    N = k.shape[2]
    qo_len = M
    kv_len = N

    # import pdb; pdb.set_trace()
    # qk = torch.matmul(q, k.transpose(-2, -1)) * (K ** -0.5)
    # 构建 causal mask，下三角为 True，其他为 False
    # mask = torch.tril(torch.ones(M, N, dtype=torch.bool, device=q.device))  # (M, N)
    # mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, M, N)
    # qk = qk.masked_fill(~mask, float('-inf'))  # 只允许当前 token 及其之前的参与 attention

    assert K in [128], "headdim should be in [128]."

    BLKQ = 128
    BLKK = 128
    ret_dict = None
    if block_scale_type == "mxfp4":
        q_mean = torch.mean(q, dim=-2, keepdim=True)
        q = q - q_mean
        pack_along_lastdim = False  # True: kernel fusion support for pack fp4 tensor into uint8 tensor
        a_fp4, a_scale, b_fp4, b_scale, a_scale_2, b_scale_2 = quant_mxfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            pack_along_lastdim=pack_along_lastdim, dual_scale=False)

    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32

    qk_quant, a_deq, b_deq = compute_reference(a_fp4, a_scale, b_fp4, b_scale, VEC_SIZE=VEC_SIZE, M=M, K=K, N=N)
    # residual_q = q * (K ** -0.5) * 1.44269504 - a_deq
    # residual_k = k - b_deq

    # qk_res1 = residual_q @ b_deq.transpose(-2, -1)
    # qk_res2 = a_deq @ residual_k.transpose(-2, -1) 
    # qk_res3 = residual_q @ residual_k.transpose(-2, -1)
    # qk_quant += qk_res3
    qk_quant += q_mean.float() @ b_deq.float().transpose(-2, -1) * (K ** -0.5)   # sage2
    # qk_quant += qk_res1 + qk_res2 + qk_res3

    # import pdb; pdb.set_trace()
    # qk = torch.matmul(q, k.transpose(-2, -1)) * (K ** -0.5)
    mask = torch.full((M, N), float('-inf'), device=q.device)
    mask = torch.triu(mask, diagonal=1)  # 上三角为 -inf
    qk = qk_quant + mask  # broadcasting 会自动扩展成 (B, H, M, N)

    qk_softmax = torch.nn.functional.softmax(qk, dim=-1)
    # 计算 qkv 结果
    qkv = torch.matmul(qk_softmax.to(torch.float16), v.to(torch.float16))

    return qkv.to(output_dtype)

def compute_dequant_mxfp8e5(q_fp8, q_scale, k_fp8, k_scale):
    # import pdb; pdb.set_trace()
    q_scale = (2**(q_scale.to(torch.float32)-127))
    k_scale = (2**(k_scale.to(torch.float32)-127))

    q_scale_expanded = q_scale.repeat_interleave(32, dim=-1)
    k_scale_expanded = k_scale.repeat_interleave(32, dim=-1)
    q_dequant = q_fp8.float() * q_scale_expanded
    k_dequant = k_fp8.float() * k_scale_expanded
    return q_dequant, k_dequant


def compute_reference_mxfp8(a_fp8, a_scale, b_fp8, b_scale, VEC_SIZE=32, M=128, K=128, N=128, per_channel=False):
    a_deq, b_deq = compute_dequant_mxfp8e5(a_fp8, a_scale, b_fp8, b_scale)
    reference = torch.matmul(a_deq, b_deq.transpose(-1, -2))
    return reference, a_deq, b_deq

def compute_reference_per_channel(a_ref, a_scale_ref, b_ref, b_scale_ref, VEC_SIZE=32, M=128, K=128, N=128):
    # 批量计算参考结果，处理多维输入
    # Extract sign, exp, mantissa from uint8
    sign = (a_ref >> 3) & 0x1  # Highest bit is sign bit
    exp = (a_ref >> 1) & 0x3   # Next 2 bits are exponent bits
    mantissa = a_ref & 0x1  # Last bit is mantissa bit
    
    # Rebuild float value
    # 1. Calculate mantissa part
    mantissa_value = torch.where(exp == 0, mantissa.float() * 0.5, 1.0 + mantissa.float() * 0.5)
    # 2. Calculate exponent part: 2^exp
    bias = 1.0
    exp_value = torch.pow(2.0, exp.float() - bias)
    # 3. Apply sign bit
    a_ref_float = mantissa_value * exp_value
    a_ref = torch.where(sign.bool(), -a_ref_float, a_ref_float)

    sign = (b_ref >> 3) & 0x1  # Highest bit is sign bit
    exp = (b_ref >> 1) & 0x3   # Next 2 bits are exponent bits
    mantissa = b_ref & 0x1  # Last bit is mantissa bit
    
    # Rebuild float value
    # 1. Calculate mantissa part
    mantissa_value = torch.where(exp == 0, mantissa.float() * 0.5, 1.0 + mantissa.float() * 0.5)
    # 2. Calculate exponent part: 2^exp
    bias = 1.0
    exp_value = torch.pow(2.0, exp.float() - bias)
    # 3. Apply sign bit
    b_ref_float = mantissa_value * exp_value
    b_ref = torch.where(sign.bool(), -b_ref_float, b_ref_float)
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    a_scale_ref = (2**(a_scale_ref.to(torch.float32)-127))
    b_scale_ref = (2**(b_scale_ref.to(torch.float32)-127))

    # def unpack_scale_batched(packed):
        # B, H, num_chunk_m, num_chunk_k, _, _, _ = packed.shape
        # return packed.permute(0, 1, 2, 5, 4, 3, 6).reshape(B, H, num_chunk_m * 128, num_chunk_k * 4).contiguous()

    # 展开scale因子到原始矩阵大小
    # a_scale_expanded = unpack_scale_batched(
    #     a_scale_ref).repeat_interleave(VEC_SIZE, dim=3)[:, :, :M, :K]
    # b_scale_expanded = unpack_scale_batched(b_scale_ref).repeat_interleave(
    #     VEC_SIZE, dim=3).transpose(-1, -2).contiguous()[:, :, :K, :N]

    a_scale_expanded = a_scale_ref.repeat_interleave(VEC_SIZE, dim=2)[:, :, :M, :]
    b_scale_expanded = b_scale_ref.repeat_interleave(VEC_SIZE, dim=2)[:, :, :N, :]

    # 计算参考结果: 批量矩阵乘法 (A * scale_a) @ (B * scale_b)
    # 使用torch.matmul自动处理批量和head维度
    a_deq = a_ref * a_scale_expanded
    b_deq = b_ref * b_scale_expanded
    reference = torch.matmul(a_deq, b_deq.transpose(-1, -2))

    return reference, a_deq, b_deq


def compute_reference_nvfp4(a_ref, a_scale_ref, b_ref, b_scale_ref, VEC_SIZE=16, M=128, K=128, N=128, per_channel=False):

    # a_scale_ref = a_scale_ref.reshape(1, 24, M//128, 4, 32, K //VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    # b_scale_ref = b_scale_ref.reshape(1, 24, N//128, 4, 32, K //VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()

    # 批量计算参考结果，处理多维输入
    # Extract sign, exp, mantissa from uint8
    sign = (a_ref >> 3) & 0x1  # Highest bit is sign bit
    exp = (a_ref >> 1) & 0x3   # Next 2 bits are exponent bits
    mantissa = a_ref & 0x1  # Last bit is mantissa bit
    
    # Rebuild float value
    # 1. Calculate mantissa part
    mantissa_value = torch.where(exp == 0, mantissa.float() * 0.5, 1.0 + mantissa.float() * 0.5)
    # 2. Calculate exponent part: 2^exp
    bias = 1.0
    exp_value = torch.pow(2.0, exp.float() - bias)
    # 3. Apply sign bit
    a_ref_float = mantissa_value * exp_value
    a_ref = torch.where(sign.bool(), -a_ref_float, a_ref_float)

    sign = (b_ref >> 3) & 0x1  # Highest bit is sign bit
    exp = (b_ref >> 1) & 0x3   # Next 2 bits are exponent bits
    mantissa = b_ref & 0x1  # Last bit is mantissa bit
    
    # Rebuild float value
    # 1. Calculate mantissa part
    mantissa_value = torch.where(exp == 0, mantissa.float() * 0.5, 1.0 + mantissa.float() * 0.5)
    # 2. Calculate exponent part: 2^exp
    bias = 1.0
    exp_value = torch.pow(2.0, exp.float() - bias)
    # 3. Apply sign bit
    b_ref_float = mantissa_value * exp_value
    b_ref = torch.where(sign.bool(), -b_ref_float, b_ref_float)
    # import pdb; pdb.set_trace()

    # a_scale_ref = (2**(a_scale_ref.to(torch.float32)-127))
    # b_scale_ref = (2**(b_scale_ref.to(torch.float32)-127))

    # def unpack_scale_batched(packed):
        # B, H, num_chunk_m, num_chunk_k, _, _, _ = packed.shape
        # return packed.permute(0, 1, 2, 5, 4, 3, 6).reshape(B, H, num_chunk_m * 128, num_chunk_k * 4).contiguous()

    # 展开scale因子到原始矩阵大小
    # a_scale_expanded = unpack_scale_batched(
    #     a_scale_ref).repeat_interleave(VEC_SIZE, dim=3)[:, :, :M, :K]
    # b_scale_expanded = unpack_scale_batched(b_scale_ref).repeat_interleave(
    #     VEC_SIZE, dim=3).transpose(-1, -2).contiguous()[:, :, :K, :N]
    # import pdb; pdb.set_trace()
    if per_channel:
        a_scale_expanded = a_scale_ref.repeat_interleave(VEC_SIZE, dim=2)[:, :, :M, :].to(torch.float32)
        b_scale_expanded = b_scale_ref.repeat_interleave(VEC_SIZE, dim=2)[:, :, :N, :].to(torch.float32)
    else:
        a_scale_expanded = a_scale_ref.repeat_interleave(VEC_SIZE, dim=3).to(torch.float32)
        b_scale_expanded = b_scale_ref.repeat_interleave(VEC_SIZE, dim=3).to(torch.float32)

    # 计算参考结果: 批量矩阵乘法 (A * scale_a) @ (B * scale_b)
    # 使用torch.matmul自动处理批量和head维度
    a_deq = a_ref * a_scale_expanded
    b_deq = b_ref * b_scale_expanded
    reference = torch.matmul(a_deq, b_deq.transpose(-1, -2))

    return reference, a_deq, b_deq


def compute_reference(a_ref, a_scale_ref, b_ref, b_scale_ref, VEC_SIZE=32, M=128, K=128, N=128, per_channel=False):

    # a_scale_ref = a_scale_ref.reshape(1, 24, M//128, 4, 32, K //VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    # b_scale_ref = b_scale_ref.reshape(1, 24, N//128, 4, 32, K //VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()

    # 批量计算参考结果，处理多维输入
    # Extract sign, exp, mantissa from uint8
    sign = (a_ref >> 3) & 0x1  # Highest bit is sign bit
    exp = (a_ref >> 1) & 0x3   # Next 2 bits are exponent bits
    mantissa = a_ref & 0x1  # Last bit is mantissa bit
    
    # Rebuild float value
    # 1. Calculate mantissa part
    mantissa_value = torch.where(exp == 0, mantissa.float() * 0.5, 1.0 + mantissa.float() * 0.5)
    # 2. Calculate exponent part: 2^exp
    bias = 1.0
    exp_value = torch.pow(2.0, exp.float() - bias)
    # 3. Apply sign bit
    a_ref_float = mantissa_value * exp_value
    a_ref = torch.where(sign.bool(), -a_ref_float, a_ref_float)

    sign = (b_ref >> 3) & 0x1  # Highest bit is sign bit
    exp = (b_ref >> 1) & 0x3   # Next 2 bits are exponent bits
    mantissa = b_ref & 0x1  # Last bit is mantissa bit
    
    # Rebuild float value
    # 1. Calculate mantissa part
    mantissa_value = torch.where(exp == 0, mantissa.float() * 0.5, 1.0 + mantissa.float() * 0.5)
    # 2. Calculate exponent part: 2^exp
    bias = 1.0
    exp_value = torch.pow(2.0, exp.float() - bias)
    # 3. Apply sign bit
    b_ref_float = mantissa_value * exp_value
    b_ref = torch.where(sign.bool(), -b_ref_float, b_ref_float)
    # import pdb; pdb.set_trace()

    a_scale_ref = (2**(a_scale_ref.to(torch.float32)-127))
    b_scale_ref = (2**(b_scale_ref.to(torch.float32)-127))

    # def unpack_scale_batched(packed):
        # B, H, num_chunk_m, num_chunk_k, _, _, _ = packed.shape
        # return packed.permute(0, 1, 2, 5, 4, 3, 6).reshape(B, H, num_chunk_m * 128, num_chunk_k * 4).contiguous()

    # 展开scale因子到原始矩阵大小
    # a_scale_expanded = unpack_scale_batched(
    #     a_scale_ref).repeat_interleave(VEC_SIZE, dim=3)[:, :, :M, :K]
    # b_scale_expanded = unpack_scale_batched(b_scale_ref).repeat_interleave(
    #     VEC_SIZE, dim=3).transpose(-1, -2).contiguous()[:, :, :K, :N]
    if per_channel:
        a_scale_expanded = a_scale_ref.repeat_interleave(VEC_SIZE, dim=2)[:, :, :M, :].to(torch.float32)
        b_scale_expanded = b_scale_ref.repeat_interleave(VEC_SIZE, dim=2)[:, :, :N, :].to(torch.float32)
    else:
        a_scale_expanded = a_scale_ref.repeat_interleave(VEC_SIZE, dim=3).to(torch.float32)
        b_scale_expanded = b_scale_ref.repeat_interleave(VEC_SIZE, dim=3).to(torch.float32)

    # 计算参考结果: 批量矩阵乘法 (A * scale_a) @ (B * scale_b)
    # 使用torch.matmul自动处理批量和head维度
    a_deq = a_ref * a_scale_expanded
    b_deq = b_ref * b_scale_expanded
    reference = torch.matmul(a_deq, b_deq.transpose(-1, -2))

    return reference, a_deq, b_deq


def load_attention_states():
    """
    加载保存的注意力状态数据
    """
    file_path = 'saved_files/low_sim_attn_states.pth'
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return None
    
    # 加载保存的数据
    saved_data = torch.load(file_path)
    
    # 提取各个状态
    query_states = saved_data['query_states'].to(device='cuda', dtype=torch.float16)[:, :, :128, :]
    key_states = saved_data['key_states'].to(device='cuda', dtype=torch.float16)[:, :, :128, :]
    value_states = saved_data['value_states'].to(device='cuda', dtype=torch.float16)[:, :, :128, :]
    # attn_output = saved_data['attn_output'].to(device='cuda', dtype=torch.float16)[:, :, 128:256, :]
    # o = saved_data['o'].to(device='cuda', dtype=torch.float16)[:, :, 128:256, :]

    B, H, L, D = query_states.shape

    # for debug
    # value_states = torch.randn_like(value_states)
    
    # out_sdpa = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, \
    #     attn_mask=None, dropout_p=0.0, is_causal=False)
    # o = out_sdpa

    qk_ref = torch.matmul(query_states.float(), key_states.float().transpose(-2, -1)) * (D ** -0.5)
    qk_ref_softmax = torch.nn.functional.softmax(qk_ref, dim=-1)
    qkv_ref = torch.matmul(qk_ref_softmax.to(torch.float32), value_states.to(torch.float32))
    # o = qk_ref

    print(f"q shape: {query_states.shape}")
    print(f"k shape: {key_states.shape}")
    print(f"v shape: {value_states.shape}")
    # print(f"attn_output shape: {attn_output.shape}")
    # print(f"o shape: {o.shape}")

    test_dual_scale = False
    qk_mxfp, ret_dict = mxfp_attn_kernel(query_states, key_states, value_states, is_causal=False, \
        block_scale_type="mxfp4", output_dtype=torch.float32, return_quant_tensor=True, smooth_k=False, \
            save_qk=True, dual_scale=test_dual_scale, quant_granularity="blockwise")  # saved qk
           
    # scaleq_scalek = qk_mxfp
    # saved_qk = qk_mxfp

    # qk_mxfp, ret_dict = mxfp_attn_kernel(query_states, key_states, value_states, is_causal=False, \
    #     block_scale_type="mxfp4", output_dtype=torch.float32, return_quant_tensor=True, smooth_k=False, \
    #         save_qk=True, dual_scale=False)  # saved qk

    a_fp4 = ret_dict['a_fp4']
    a_scale = ret_dict['a_scale']
    b_fp4 = ret_dict['b_fp4']
    b_scale = ret_dict['b_scale']
    a_scale_2 = ret_dict['a_scale_2']
    b_scale_2 = ret_dict['b_scale_2']

    qk = qk_mxfp

    # scale_q_mamual = torch.max(torch.abs(query_states * (D ** -0.5)*1.44269504), dim=-1).values / 6
    # scale_k_mamual = torch.max(torch.abs(key_states), dim=-1).values / 6
    # scale_q_mamual = scale_q_mamual.unsqueeze(-1)
    # scale_k_mamual = scale_k_mamual.unsqueeze(-1)
    # manual_scale = scale_q_mamual @ scale_k_mamual.transpose(-2, -1)
    
    qk_quant, a_deq, b_deq = compute_reference(a_fp4, a_scale, b_fp4, b_scale)
    print("a_deq vs. query_states")
    precision_metric(a_deq, query_states * (D ** -0.5) * 1.44269504)
    print("b_deq vs. key_states")
    b_deq = b_deq.transpose(-1, -2).contiguous()
    precision_metric(b_deq, key_states)
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # qk_quant *= (a_scale_2 * b_scale_2.transpose(-2, -1))

    # q_q = query_states / scale_q_mamual
    # q_k = key_states / scale_k_mamual

    # qk_ref = torch.matmul(q_q, q_k.transpose(-2, -1)) * manual_scale
    
    # precision_metric(qk_mxfp, qk_quant)
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # qk = qk * manual_scale
    # print("qk_quant vs. qk_ref")
    # precision_metric(qk_quant, qk_ref)
    # precision_metric(qk_mxfp, qk_ref)
    print("*"*20)
    # import pdb; pdb.set_trace()
    # print("draw residual qk")
    residual_q = query_states * (D ** -0.5) * 1.44269504 - a_deq
    # residual_q = query_states * (D ** -0.5) - a_deq
    # residual_q = query_states - a_deq
    residual_k = key_states - b_deq
    # # 按照head绘制热力图
    # import matplotlib.pyplot as plt
    
    # # 获取每个head的残差
    # for head_idx in range(qk_ref.shape[1]):
    #     residual_head = residual_k[0, head_idx]  # 取绝对值
        
    #     plt.figure(figsize=(10, 8))
    #     residual_data = residual_head.cpu().numpy()
    #     vmax = abs(residual_data).max()
    #     plt.imshow(residual_data, cmap='RdBu', vmin=-vmax, vmax=vmax)
    #     plt.colorbar()
    #     plt.title(f'Head {head_idx} residual K')
    #     plt.xlabel('head_dim')
    #     plt.ylabel('token')
    #     plt.savefig(f'saved_figs/head_k_{head_idx}_residual_heatmap.png')
    #     plt.close()
    
    # print("*"*20)
    # exit()

    # 计算 head_dim 的残差绝对值总和: shape [B, H, D]
    dim_sum = torch.sum(torch.abs(residual_q), dim=-2) + torch.sum(torch.abs(residual_k), dim=-2)  # over M

    # 取每个 head 内残差最大的 10 行索引: shape [B, H, 10]
    top10_indices = torch.topk(dim_sum, k=128, dim=-1).indices  # along last dim (D)

    # 扩展维度用于 gather: 从 M 维上选取 top10 的 token 行
    top10_indices_exp = top10_indices.unsqueeze(-2).expand(-1, -1, query_states.size(-2), -1)  # [B, H, M, 10]

    # Gather 相应的 top10 token 特征，结果形状为 [B, H, D, 10]
    # top10_q      = query_states.gather(dim=2, index=top10_indices_exp) * ((D ** -0.5) * 1.44269504)  # Q
    # top10_k      = key_states.gather(dim=2, index=top10_indices_exp)                  # K
    top10_q0     = a_deq.gather(dim=-1, index=top10_indices_exp)                       # Q_0
    top10_k0     = b_deq.gather(dim=-1, index=top10_indices_exp)                       # K_0
    # import pdb; pdb.set_trace()
    # 残差项
    top10_resq   = residual_q.gather(dim=-1, index=top10_indices_exp)   # R_Q
    top10_resk   = residual_k.gather(dim=-1, index=top10_indices_exp)   # R_K

    # 三项残差贡献
    qk_res1 = top10_resq @ top10_k0.transpose(-2, -1)  # R_Q @ K_0^T
    qk_res2 = top10_q0  @ top10_resk.transpose(-2, -1) # Q_0 @ R_K^T
    qk_res3 = top10_resq @ top10_resk.transpose(-2, -1)  # R_Q @ R_K^T

    # qk_res1 = residual_q @ b_deq.transpose(-2, -1)
    # qk_res2 = a_deq @ residual_k.transpose(-2, -1) 
    # qk_res3 = residual_q @ residual_k.transpose(-2, -1)

    qk_mxfp2 = qk_mxfp + qk_res1 + qk_res2 + qk_res3

    print("qk_mxfp vs. qk_ref")
    precision_metric(qk_mxfp, qk_ref)

    # print("分别评估三项贡献 vs. qk_ref")
    print("qk_mxfp2 vs. qk_ref")
    precision_metric(qk_mxfp + qk_res1, qk_ref)
    precision_metric(qk_mxfp + qk_res2, qk_ref)
    precision_metric(qk_mxfp + qk_res3, qk_ref)
    precision_metric(qk_mxfp2, qk_ref)

    print("*"*20)
    import pdb; pdb.set_trace()

    # 计算各个残差项的 softmax 和 qkv 结果
    qk_mxfp_softmax = torch.nn.functional.softmax(qk_mxfp, dim=-1)
    qk_res1_softmax = torch.nn.functional.softmax(qk_mxfp + qk_res1, dim=-1)
    qk_res2_softmax = torch.nn.functional.softmax(qk_mxfp + qk_res2, dim=-1)
    qk_res3_softmax = torch.nn.functional.softmax(qk_mxfp + qk_res3, dim=-1)
    qk_mxfp2_softmax = torch.nn.functional.softmax(qk_mxfp2, dim=-1)

    # 计算 qkv 结果
    qkv_mxfp = torch.matmul(qk_mxfp_softmax.to(torch.float16), value_states.to(torch.float16))
    qkv_res1 = torch.matmul(qk_res1_softmax.to(torch.float16), value_states.to(torch.float16))
    qkv_res2 = torch.matmul(qk_res2_softmax.to(torch.float16), value_states.to(torch.float16))
    qkv_res3 = torch.matmul(qk_res3_softmax.to(torch.float16), value_states.to(torch.float16))
    qkv_mxfp2 = torch.matmul(qk_mxfp2_softmax.to(torch.float16), value_states.to(torch.float16))

    print("\n各残差项的 QKV 精度 vs. qkv_ref:")
    print("qkv_mxfp_softmax vs qkv_ref:")
    precision_metric(qkv_mxfp, qkv_ref)
    print("qkv_res1 vs qkv_ref:")
    precision_metric(qkv_res1, qkv_ref)
    print("qkv_res2 vs qkv_ref:")
    precision_metric(qkv_res2, qkv_ref)
    print("qkv_res3 vs qkv_ref:")
    precision_metric(qkv_res3, qkv_ref)
    print("qkv_mxfp2 vs qkv_ref:")
    precision_metric(qkv_mxfp2, qkv_ref)

    import pdb; pdb.set_trace()
    exit()
    # 假设数据类型为 float32, qk 的维度是 [1, 24, 128, 128]
    BLOCK_M = qk.shape[2]
    BLOCK_N = qk.shape[3]
    
    # log-sum-exp trick
    old_m = torch.full((1, 24, BLOCK_M), float("-inf"), dtype=torch.float32, device=qk.device)
    l_i = torch.ones((1, 24, BLOCK_M), dtype=torch.float32, device=qk.device)
    acc = torch.zeros((1, 24, BLOCK_M, 128), dtype=torch.float32, device=qk.device)

    local_m = torch.max(qk, dim=-1).values
    new_m = torch.maximum(old_m, local_m)
    qk_shifted = qk - new_m.unsqueeze(-1)

    p = torch.exp(qk_shifted)  # 改为 exp 而非 exp2
    l_ij = torch.sum(p, dim=-1)

    alpha = torch.exp(old_m - new_m)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha.unsqueeze(-1)

    acc = acc + torch.matmul(p.to(torch.float16), value_states.to(torch.float16))  # 提高精度

    output = acc / l_i.unsqueeze(-1)
    # qk_ref_softmax = torch.nn.functional.softmax(qk, dim=-1)
    # qk_mxfp_manual_softmax = torch.nn.functional.softmax(qk_mxfp_manual, dim=-1)
    # qkv_manual = torch.matmul(output.to(torch.float16), value_states.to(torch.float16))
    qkv_manual = output

    qk_softmax = torch.nn.functional.softmax(qk, dim=-1)
    qkv_quant1 = torch.matmul(qk_softmax.to(torch.float16), value_states.to(torch.float16))
    precision_metric(qkv_quant1, qkv_manual)

    qkv_mxfp, ret_dict = mxfp_attn_kernel(query_states, key_states, value_states, is_causal=False, \
        block_scale_type="mxfp4", output_dtype=torch.float32, return_quant_tensor=True, smooth_k=False, \
            save_qk=False, dual_scale=test_dual_scale, quant_granularity="blockwise")  # saved qk
           
    precision_metric(qkv_ref, qkv_manual)
    precision_metric(qkv_manual, qkv_mxfp)



    import pdb; pdb.set_trace()

   
    # print("qkv_quant vs. qkv_ref")
    # precision_metric(qkv_ref, qkv_quant1)

    # # 手动 per token 计算
    
    # print("scale_q_mamual, scale_k_mamual")
    # precision_metric(scale_q_mamual, a_scale_2)
    # precision_metric(scale_k_mamual, b_scale_2)
    # manual_scale = scale_q_mamual @ scale_k_mamual.transpose(-2, -1)
    # # precision_metric(scaleq_scalek, manual_scale)

    # # 计算 qk
    # q_q = query_states / scale_q_mamual
    # q_k = key_states / scale_k_mamual
    # qk_mxfp, ret_dict = mxfp_attn_kernel(q_q, q_k, value_states, is_causal=False, \
    #     block_scale_type="mxfp4", output_dtype=torch.float32, return_quant_tensor=True, smooth_k=False, \
    #         save_qk=True, dual_scale=False)  # saved qk
    # qk_mxfp_manual = qk_mxfp 

    # print("saved_qk_mxfp vs. qk_mxfp_manual")
    # ref_quant_qk = q_q@q_k.transpose(-2, -1) * (D ** -0.5)
    # # precision_metric(ref_quant_qk, saved_qk) # Cossim: 0.789000, L1: 0.821900, RMSE:2.939200
    # # precision_metric(ref_quant_qk, qk_mxfp_manual)  # Cossim: 0.878300, L1: 1.032800, RMSE:0.430000
    # # import pdb; pdb.set_trace()

    # qk_mxfp_manual = qk_mxfp_manual * manual_scale
    # precision_metric(ref_quant_qk, qk_mxfp_manual)  # Cossim: 0.906200, L1: 7.080900, RMSE:3.120000
    
    '''qk = saved_qk
    # 假设数据类型为 float32, qk 的维度是 [1, 24, 128, 128]
    BLOCK_M = qk.shape[2]
    BLOCK_N = qk.shape[3]
    
    # log-sum-exp trick
    old_m = torch.full((1, 24, BLOCK_M), float("-inf"), dtype=torch.float32, device=qk.device)
    l_i = torch.ones((1, 24, BLOCK_M), dtype=torch.float32, device=qk.device)
    acc = torch.zeros((1, 24, BLOCK_M, 128), dtype=torch.float32, device=qk.device)

    local_m = torch.max(qk, dim=-1).values
    new_m = torch.maximum(old_m, local_m)
    qk_shifted = qk - new_m.unsqueeze(-1)

    p = torch.exp(qk_shifted)  # 改为 exp 而非 exp2
    l_ij = torch.sum(p, dim=-1)

    alpha = torch.exp(old_m - new_m)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha.unsqueeze(-1)

    acc = acc + torch.matmul(p.to(torch.float16), value_states.to(torch.float16))  # 提高精度

    output = acc / l_i.unsqueeze(-1)
    # qk_ref_softmax = torch.nn.functional.softmax(qk, dim=-1)
    # qk_mxfp_manual_softmax = torch.nn.functional.softmax(qk_mxfp_manual, dim=-1)
    # qkv_manual = torch.matmul(output.to(torch.float16), value_states.to(torch.float16))
    qkv_manual = output'''
    # print("qkv_manual vs. qkv_ref")
    # precision_metric(qkv_ref, qkv_manual)

    # precision_metric(scale_q_mamual @ scale_k_mamual.transpose(-2, -1), scaleq_scalek)
    print("256")
    out_mxfp_dual, ret_dict = mxfp_attn_kernel(query_states, key_states, value_states, is_causal=True, \
        block_scale_type="mxfp4", output_dtype=torch.float32, return_quant_tensor=True, smooth_k=False, \
            save_qk=True, dual_scale=True, quant_granularity="tokenwise") 
    # print("dual_scale=True, quant_granularity=tokenwise")
    qk_block2 = out_mxfp_dual
    print("128")
    out_mxfp_dual, ret_dict = mxfp_attn_kernel(query_states, key_states, value_states, is_causal=True, \
        block_scale_type="mxfp4", output_dtype=torch.float32, return_quant_tensor=True, smooth_k=False, \
            save_qk=False, dual_scale=True, quant_granularity="tokenwise") 
    # qk_block2_2 = out_mxfp_dual
    precision_metric(out_mxfp_dual, qkv_ref)
    # print("dual_scale=True, quant_granularity=blockwise")
    # precision_metric(manual_scale, qk_block2)
    # precision_metric(qk_block2_2, qk_block2) # sim=1

    # import pdb; pdb.set_trace()
    out_mxfp, ret_dict = mxfp_attn_kernel(query_states, key_states, value_states, is_causal=True, \
        block_scale_type="mxfp4", output_dtype=torch.float32, return_quant_tensor=True, smooth_k=False, \
            save_qk=False, dual_scale=False, quant_granularity="tokenwise") 
    print("dual_scale=False")
    precision_metric(out_mxfp, qkv_ref)



def precision_metric(quant_o, fa2_o, verbose=True, round_num=4): 
    if quant_o.shape[-2] > 200000:
        quant_o, fa2_o = quant_o.cpu(), fa2_o.cpu()
    x, xx = quant_o.float(), fa2_o.float() 
    sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    l1 =   ( (x - xx).abs().sum() / xx.abs().sum() ).item()
    rmse = torch.sqrt(torch.mean((x -xx) ** 2)).item()
    sim = round(sim, round_num)
    l1 = round(l1, round_num)
    rmse = round(rmse, round_num)
    if verbose: print(f'Cossim: {sim:.6f}, L1: {l1:.6f}, RMSE:{rmse:.6f}')
    return {"Cossim": sim, "L1": l1, "RMSE": rmse}

  
if __name__ == "__main__":
   load_attention_states()
    