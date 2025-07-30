# for ablation study
import torch
import time
# from spas_sage_attn import spas_sage_attn_meansim_cuda
# from spas_sage_attn.utils import get_block_map_meansim
# from spas_sage_attn.triton_kernel_example import spas_sage_attn_meansim, per_block_int8, forward as forward_triton
# from flash_attn.flash_attn_triton import flash_attn_func
import numpy as np
from helper import print_result_as_md
from ours.mxfp_attn_func import mxfp_attn_kernel, block_scaled_batched_attn
# from ours.batched_block_scaled_matmul import test_batched_matmul, initialize_block_scaled_batched_from_tensor
from ours.quant_funcs import quant_mxfp8, quant_mxfp4, quant_nvfp4, quant_mxfp8_nvfp4, quant_mxfp8_nvfp4, quant_mxfp8_nvfp4, quant_mxfp8_nvfp4
import random
import os
from ours.modify_mxfp_attn import precision_metric
from ours.mxfp import MXFP4Tensor, MXScaleTensor, MXFP8Tensor
import triton
import triton.language as tl
from tests.tile_size_ablation import load_attention_states
from tests.time_profiler import time_profiler

iter_times = 10

def measure_time(func, *args, **kwargs):
    # # warmup
    for _ in range(3):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    # testing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for test_time in range(iter_times):
        result = func(*args, **kwargs)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)

    # 计算 ops
    ops = iter_times/elapsed_time # * 10e-12
    avg_time = elapsed_time / iter_times

    return result, avg_time, ops

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def test_kernel_fusion_benchmark():

    seed_all(41)

    batch_size = 1
    num_heads = 24
    # qo_len = 11110
    # kv_len = 11110
    head_dim = 128
    is_causal = True
    block_scale_type = "mixed"
    # block_scale_type = "nvfp4"
    smooth_k = True
    dual_scale = False
    quant_granularity = "blockwise"
    qk_dtype = "e4m3"

    query_states, key_states, value_states = load_attention_states(2048*4)
    q = query_states
    k = key_states
    v = value_states

    qo_len = q.shape[2]
    kv_len = k.shape[2]
    # q = torch.randn(batch_size, num_heads, qo_len, head_dim,
    #                 device='cuda', dtype=torch.float16)
    # k = torch.randn(batch_size, num_heads, kv_len, head_dim, 
    #                 device='cuda', dtype=torch.float16)
    # v = torch.randn(batch_size, num_heads, kv_len, head_dim,
    #                 device='cuda', dtype=torch.float16)

    # s_q = torch.randn(batch_size, num_heads, qo_len, head_dim//16,
    #                 device='cuda', dtype=torch.float16)
    # s_k = torch.randn(batch_size, num_heads, kv_len, head_dim//16, 
    #                 device='cuda', dtype=torch.float16)

    print("Start performance test...")
    

    # out_ours, ours_time_total, ours_ops = measure_time(
    #     mxfp_attn_kernel, q, k, v, is_causal=is_causal, block_scale_type=block_scale_type, \
    #         smooth_k=smooth_k, dual_scale=dual_scale, quant_granularity=quant_granularity, \
    #         fuse_mp_quant=True, pre_quant=True, # save_qk=True
    # )
    BLKQ = 128
    BLKK = 128
    fuse_pack = True
    sm_scale = (head_dim)**(-0.5)* 1.44269504

    # not fused at all
    def not_fused_mxfp8_quant(x, sm_scale):
        x = x.reshape(batch_size, num_heads, qo_len, head_dim//32, 32)
        x *= sm_scale
        emax_elem = 15
        x_abs_max = torch.max(torch.abs(x), dim=-1).values  # [1, 24, 4096, 4]
        shared_exp = torch.floor(torch.log2(x_abs_max)) - emax_elem
        shared_scale = torch.exp2(shared_exp)
        # import pdb; pdb.set_trace()
        shared_scale_broadcast = shared_scale.reshape(batch_size, num_heads, qo_len, head_dim//32, 1).repeat(1, 1, 1, 1, 32)
        x = x / shared_scale_broadcast
        x_quant = torch.clamp(x, -57344, 57344)
        x_fp8 = MXFP8Tensor(x_quant, device=x.device)
        shared_scale = MXScaleTensor(shared_scale, device=x.device)
        return x_fp8, shared_scale

    def not_fused_nvfp4_quant(x, sm_scale):
        x = x.reshape(batch_size, num_heads, qo_len, head_dim//16, 16)
        x *= sm_scale
        x_abs = torch.abs(x)
        quant_scale = torch.max(x_abs, dim=-1, keepdim=True).values / (448*6)
        x_abs = x_abs / quant_scale
        shared_scale = torch.max(x_abs, dim=-1, keepdim=True).values / 6.0
        shared_scale_broadcast = shared_scale.reshape(batch_size, num_heads, qo_len, head_dim//16, 1).repeat(1, 1, 1, 1, 16)
        x_abs = x_abs / shared_scale_broadcast
        x_quant = torch.clamp(x_abs, -6, 6)
        x_fp4 = MXFP4Tensor(x_quant, device=x.device)
        x_fp4 = x_fp4.to_packed_tensor(dim=len(x_fp4.data.shape) - 1)
        quant_scale = MXScaleTensor(quant_scale, device=x.device)
        return x_fp4, quant_scale

    def not_fused_at_all(q, k):
        q_fp8, q_scale = not_fused_mxfp8_quant(q, sm_scale)
        k_fp8, k_scale = not_fused_mxfp8_quant(k, 1)
        q_fp4, q_scale = not_fused_nvfp4_quant(q, sm_scale)
        k_fp4, k_scale = not_fused_nvfp4_quant(k, 1)

        return q_fp8, q_scale, k_fp8, k_scale, q_fp4, k_fp4


    @triton.jit
    def quant_mxfp8e5_kernel_not_fuse_scalar(Input, Output, Scale, L,
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
        
        # x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)  # 浮点数的四舍五入
        x_quant = tl.clamp(x_quant, -57344, 57344)  # e5m2 range
        # x_quant = tl.clamp(x_quant, -448, 448)  # e4m3 range
        x_quant = x_quant.to(tl.float8e5)
        
        # 存储量化后的值和scale
        x_fp8 = tl.reshape(x_quant, x.shape)
        tl.store(output_ptrs, x_fp8, mask=offs_n[:, None] < L)

        scales = shared_scale.to(tl.float32)
        tl.store(scale_ptrs, scales, mask=offs_n[:, None] < L)


    def quant_mxfp8e5_not_fuse_scalar(q, k, BLKQ=128, BLKK=128, sm_scale=None, tensor_layout="HND"):
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

        q_scale = torch.empty((b, h_qo, qo_len, head_dim // 32), device=q.device, dtype=torch.float32)
        k_scale = torch.empty((b, h_kv, kv_len, head_dim // 32), device=q.device, dtype=torch.float32)

        q_scale_2 = torch.empty((b, h_qo, qo_len, 1), device=q.device, dtype=torch.float32)
        k_scale_2 = torch.empty((b, h_kv, kv_len, 1), device=q.device, dtype=torch.float32)

        if sm_scale is None:
            sm_scale = head_dim**-0.5

        grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
        quant_mxfp8e5_kernel_not_fuse_scalar[grid](
            q, q_fp8, q_scale, qo_len,
            stride_bz_q, stride_h_q, stride_seq_q,
            stride_bz_qo, stride_h_qo, stride_seq_qo,
            q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
            sm_scale=(sm_scale * 1.44269504),
            C=head_dim, BLK=BLKQ
        )

        grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
        quant_mxfp8e5_kernel_not_fuse_scalar[grid](
            k, k_fp8, k_scale, kv_len,
            stride_bz_k, stride_h_k, stride_seq_k,
            stride_bz_ko, stride_h_ko, stride_seq_ko,
            k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
            sm_scale=1.0,
            C=head_dim, BLK=BLKK
        )

        return q_fp8, q_scale, k_fp8, k_scale, q_scale_2, k_scale_2

    @triton.jit
    def quant_nvfp4_kernel_not_fuse_scalar(Input, Output, Scale, Scale_2, L,
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
        # t = shared_scale.to(tl.float8e4nv)
        tl.store(scale_ptrs, shared_scale.to(tl.float32), mask=offs_n[:, None] < L)

    def quant_nvfp4_not_fuse_scalar(q, k, BLKQ=128, BLKK=128, sm_scale=None, tensor_layout="HND", VEC_SIZE=16, \
        fuse_pack=False, dual_scale=False, quant_granularity="blockwise"):

        if tensor_layout == "HND":
            b, h_qo, qo_len, head_dim = q.shape
            _, h_kv, kv_len, _ = k.shape

            if fuse_pack:
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

            if fuse_pack:
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

        q_scale = torch.empty((b, h_qo, qo_len, head_dim // VEC_SIZE), device=q.device, dtype=torch.float32)
        k_scale = torch.empty((b, h_kv, kv_len, head_dim // VEC_SIZE), device=q.device, dtype=torch.float32)

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
        quant_nvfp4_kernel_not_fuse_scalar[grid](
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
        quant_nvfp4_kernel_not_fuse_scalar[grid](
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
        # import pdb; pdb.set_trace()
        # if dual_scale:
        return q_fp4, q_scale, k_fp4, k_scale, q_scale_2, k_scale_2
        # else:
        #     return q_fp4, q_scale, k_fp4, k_scale, None, None


    # fuse quant, not fuse pack, not fuse scale cvt
    def fuse_quant_not_fuse_pack_not_fuse_scale_cvt(q, k):
        q_fp8, q_scale, k_fp8, k_scale, q_scale_2, k_scale_2 = quant_mxfp8e5_not_fuse_scalar(q, k, BLKQ=BLKQ, BLKK=BLKK)
        q_scale = MXScaleTensor(q_scale, device=q_scale.device)
        k_scale = MXScaleTensor(k_scale, device=k_scale.device)

        q_fp4, q_scale, k_fp4, k_scale, q_scale_2, k_scale_2 = quant_nvfp4_not_fuse_scalar(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            fuse_pack=False, dual_scale=True, quant_granularity=quant_granularity)
        q_fp4 = MXFP4Tensor(data=q_fp4, dtype=torch.uint8)
        q_fp4 = q_fp4.to_packed_tensor(dim=len(q_fp4.data.shape) - 1)
        k_fp4 = MXFP4Tensor(data=k_fp4, dtype=torch.uint8)
        k_fp4 = k_fp4.to_packed_tensor(dim=len(k_fp4.data.shape) - 1)
        
        q_scale = MXScaleTensor(q_scale, device=q_scale.device)
        k_scale = MXScaleTensor(k_scale, device=k_scale.device)

        # return q_fp8, q_scale, k_fp8, k_scale, q_fp4, k_fp4, q_scale_2, k_scale_2

    # fuse quant, fuse pack, not fuse scale cvt
    def fuse_quant_fuse_pack_not_fuse_scale_cvt(q, k):
        q_fp8, q_scale, k_fp8, k_scale, q_scale_2, k_scale_2 = quant_mxfp8e5_not_fuse_scalar(q, k, BLKQ=BLKQ, BLKK=BLKK)  # not fuse scalar
        q_scale = MXScaleTensor(q_scale, device=q_scale.device)
        k_scale = MXScaleTensor(k_scale, device=k_scale.device)

        q_fp4, q_scale, k_fp4, k_scale, q_scale_2, k_scale_2 = quant_nvfp4_not_fuse_scalar(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            fuse_pack=True, dual_scale=dual_scale, quant_granularity=quant_granularity)

        q_scale = MXScaleTensor(q_scale, device=q_scale.device)
        k_scale = MXScaleTensor(k_scale, device=k_scale.device)

        # return q_fp8, q_scale, k_fp8, k_scale, q_fp4, k_fp4, q_scale_2, k_scale_2


    def fuse_quant_pack_scale_cvt_not_fuse_mp_quant(q, k):
        q_fp8, q_scale, k_fp8, k_scale, q_scale_2, k_scale_2 = quant_mxfp8(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            qk_dtype=qk_dtype, dual_scale=dual_scale, quant_granularity=quant_granularity)
        torch.cuda.synchronize()
        q_fp4, q_scale, k_fp4, k_scale, q_scale_2, k_scale_2 = quant_nvfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            fuse_pack=True, dual_scale=dual_scale, quant_granularity=quant_granularity)
        torch.cuda.synchronize()

    def fuse_all(q, k):
        quant_mxfp8_nvfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, fuse_pack=True, v_quant=False, v=None, \
            dual_scale=dual_scale, quant_granularity=quant_granularity, qk_dtype=qk_dtype)


    dict_func = {
        "not_fused_at_all": not_fused_at_all,
        "fuse_quant_not_fuse_pack_not_fuse_scale_cvt": fuse_quant_not_fuse_pack_not_fuse_scale_cvt,
        "fuse_quant_fuse_pack_not_fuse_scale_cvt": fuse_quant_fuse_pack_not_fuse_scale_cvt,
        "fuse_quant_pack_scale_cvt_not_fuse_mp_quant": fuse_quant_pack_scale_cvt_not_fuse_mp_quant,
        "fuse_all": fuse_all,
    }

    results = {
        "not_fused_at_all": [],
        "fuse_quant_not_fuse_pack_not_fuse_scale_cvt": [],
        "fuse_quant_fuse_pack_not_fuse_scale_cvt": [],
        "fuse_quant_pack_scale_cvt_not_fuse_mp_quant": [],
        "fuse_all": [],
    }

    # time_profiler(fuse_quant_pack_scale_cvt_not_fuse_mp_quant, q, k)
    # exit()

    for func_name, func in dict_func.items():
        print(f"testing {func_name}...")
        out, time_total, ops = measure_time(
            func, q, k
        )
        results[func_name] = {'Avg Time (us)': time_total*1000, 'OPS': ops}

    print("testing torch.nn.functional.scaled_dot_product_attention...")
    def test_sdpa(q, k, v, is_causal=False):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )

    out_torch, time_total_sdpa, ops_sdpa = measure_time(
        test_sdpa, q, k, v, is_causal=is_causal
    )


    # # 打印结果
    # print(f"{'Function':<45} | {'Time (us)':>10} | {'ops':>10}")
    # print("-" * 70)

    # for func_name, (out, avg_time, ops) in results.items():
    #     print(f"{func_name:<45} | {avg_time * 1000 :10.3f} | {ops:10.3f}")

    print_result_as_md(results)


def test_attn_time_breakdown():
    batch_size = 1
    num_heads = 24
    qo_len = 4096
    kv_len = 4096
    head_dim = 128
    is_causal = True
    block_scale_type = "mixed"  # mixed, nvfp4, mxfp8, mxfp4
    qk_dtype = 'e4m3'  # e4m3, e5m2
    smooth_k = True
    dual_scale = True
    quant_granularity = "tensorwise"  # tokenwise, blockwise, tensorwise
    tile_size = 1
    sink_size = 1
    # q_fuse_pack = False
    # k_fuse_pack = False

    print(f"block_scale_type: {block_scale_type}, qk_dtype: {qk_dtype}, \
        dual_scale: {dual_scale}, quant_granularity: {quant_granularity}")

    # q = torch.randn(batch_size, num_heads, qo_len, head_dim,
    #                 device='cuda', dtype=torch.float16)
    # k = torch.randn(batch_size, num_heads, kv_len, head_dim, 
    #                 device='cuda', dtype=torch.float16)
    # v = torch.randn(batch_size, num_heads, kv_len, head_dim,
    #                 device='cuda', dtype=torch.float16)
    query_states, key_states, value_states = load_attention_states()
    q = query_states
    k = key_states
    v = value_states

    kwargs = {
        "is_causal": is_causal,
        "smooth_k": smooth_k,
        "block_scale_type": block_scale_type,
        "dual_scale": dual_scale,
        "quant_granularity": quant_granularity,
        "fuse_mp_quant": True,
        "pre_quant": True,
        "fuse_pack": True,
        "diag_tile": tile_size,
        "sink_tile": sink_size,
        "qk_dtype": qk_dtype,
    }
    time_profiler(mxfp_attn_kernel, q, k, v, **kwargs)


if __name__ == "__main__":
    test_kernel_fusion_benchmark()
    # test_attn_time_breakdown()