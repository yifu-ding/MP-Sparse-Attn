import torch
import time
# from spas_sage_attn import spas_sage_attn_meansim_cuda
# from spas_sage_attn.utils import get_block_map_meansim
# from spas_sage_attn.triton_kernel_example import spas_sage_attn_meansim, per_block_int8, forward as forward_triton
# from flash_attn.flash_attn_triton import flash_attn_func
import numpy as np
from ours.mxfp_attn_kernel import mxfp_attn_kernel, block_scaled_batched_attn
from ours.batched_block_scaled_matmul import test_batched_matmul, initialize_block_scaled_batched_from_tensor    
from ours.quant_mxint8 import quant_fpxint8, quant_mxfp8e5, quant_mxfp4

iter_times = 5
def measure_time(func, *args, **kwargs):
    # warmup
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
    
    return result, elapsed_time

def test_performance():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size = 1
    num_heads = 4
    seq_len = 333
    head_dim = 128
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    print("Start performance test...")
    simthreshd=0.001
    cdfthreshd=0.519
    # cdfthreshd = 0
    
    # 得到一组稀疏率比较好的simthreshd, cdfthreshd
    # for simthreshd in range(1,1000,1):
    #     simthreshd = simthreshd / 1000.
    #     print(f"simthreshd={simthreshd}, cdfthreshd={cdfthreshd}")
    #     for cdfthreshd in range(1,1000,1):
    #         cdfthreshd = cdfthreshd / 1000.
    #         # 测试 spas_sage_attn_meansim
    #         # print("\n测试 spas_sage_attn_meansim:")
    #         # print(f"simthreshd={simthreshd}, cdfthreshd={cdfthreshd}")
    #         # 分步时间
    #         k_block_indices, time_block_map = measure_time(
    #             get_block_map_meansim, q, k, is_causal=False, simthreshd1=simthreshd, cdfthreshd=cdfthreshd
    #         )
    #         sparsity = k_block_indices.flatten().sum() / k_block_indices.numel()
    #         if sparsity < 0.9 and sparsity > 0.5: 
    #             print(f"sparsity={sparsity}")
    #             print(f"simthreshd={simthreshd}, cdfthreshd={cdfthreshd}")
    #             exit()
    
    
    # k_block_indices, time_block_map = measure_time(
    #     get_block_map_meansim, q, k, is_causal=False, simthreshd1=simthreshd, cdfthreshd=cdfthreshd
    # )

    # sparsity = k_block_indices.flatten().sum() / k_block_indices.numel()
    # print(f"sparsity={sparsity}")
    
    # # import pdb; pdb.set_trace()
    # (q_int8, q_scale, k_int8, k_scale), time_int8 = measure_time(
    #     per_block_int8, q, k
    # )
    
    # pvthreshd = torch.tensor([50.0], device='cuda')
    # output, time_forward = measure_time(
    #     forward_triton, q_int8, k_int8, k_block_indices, v, q_scale, k_scale, pvthreshd,
    #     is_causal=False, tensor_layout="HND", output_dtype=torch.float16
    # )
    
    # print("testing spas_sage_attn_meansim...")
    # out_spas, time_total_spas = measure_time(
    #     spas_sage_attn_meansim, q, k, v, is_causal=False, simthreshd1=simthreshd, cdfthreshd=cdfthreshd
    # )
    
    # print("testing spas_sage_attn_meansim...")
    # out_spas_full, time_total_spas_full = measure_time(
    #     spas_sage_attn_meansim, q, k, v, is_causal=False, simthreshd1=0.1, cdfthreshd=0.9
    # )

    # # 测试 spas_sage_attn_meansim_cuda
    # print("testing spas_sage_attn_meansim_cuda...")
    # (out_spas_cuda, qk_sparsity), time_total_spas_cuda = measure_time(
    #     spas_sage_attn_meansim_cuda, q, k, v, is_causal=False, simthreshd1=simthreshd, cdfthreshd=cdfthreshd, return_sparsity=True
    # )
    # print(f"qk_sparsity={qk_sparsity}")
    
    # (out_spas_cuda_full, qk_sparsity), time_total_spas_cuda_full = measure_time(
    #     spas_sage_attn_meansim_cuda, q, k, v, is_causal=False, simthreshd1=0.1, cdfthreshd=0.9, return_sparsity=True
    # )
    # print(f"qk_sparsity_full={qk_sparsity}")
    
    # 总时间
    # print("testing mxfp_attn_kernel...")
    block_scale_type = "mxfp8" 
    out_mxfp = None
    out_mxfp, time_total_mxfp = measure_time(
        mxfp_attn_kernel, q, k, v, is_causal=False, block_scale_type=block_scale_type, skip_thresh=10
    )
    
    # torch.cuda.empty_cache()
    # import gc; gc.collect()
    
    # print("mxfp quant...")
    # if block_scale_type == "mxfp4":
    #     (a_fp4, a_scale, b_fp4, b_scale), time_step_quant_mxfp = measure_time(
    #         quant_mxfp4, q, k, BLKQ=128
    #     )   
    #     # a_fp4, a_scale, b_fp4, b_scale = quant_mxfp4(a_tensor, b_tensor, BLKQ=128)
    #     b_quant = b_fp4.to(torch.uint8)
    #     del a_fp4, b_fp4
    # else:
    #     (a_fp8, a_scale, b_fp8, b_scale), time_step_quant_mxfp = measure_time(
    #         quant_mxfp8e5, q, k, BLKQ=128
    #     )
    #     # a_fp8, a_scale, b_fp8, b_scale = quant_mxfp8e5(a_tensor, b_tensor, BLKQ=128)
    #     a_quant = a_fp8
    #     b_quant = b_fp8
    #     del a_fp8, b_fp8

    # torch.cuda.empty_cache()
    # import gc; gc.collect()
        
    # # BLOCK_K = 256 if "fp4" in block_scale_type else 128
    # VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32

    # a_quant = a_quant.reshape(batch_size, num_heads, seq_len, head_dim).contiguous()
    # b_quant = b_quant.reshape(batch_size, num_heads, seq_len, head_dim).contiguous()
    # a_scale = a_scale.reshape(batch_size, num_heads, seq_len//128, 4, 32, head_dim//VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    # b_scale = b_scale.reshape(batch_size, num_heads, seq_len//128, 4, 32, head_dim//VEC_SIZE//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    
    # out_rand_mxfp, time_total_rand_mxfp = measure_time(
    #     test_batched_matmul, a_quant, b_quant, a_scale, b_scale, configs=None, block_scale_type=block_scale_type, B=batch_size, H=num_heads, M=seq_len, N=seq_len, K=head_dim
    # )  
    
    # 初始化多维批量矩阵乘法
    # print("testing initialize_block_scaled_batched_from_tensor...")
    # (a_desc, a_scale_proc, b_desc, b_scale_proc, configs, reference), time_step_mxfp_init = \
    #     measure_time(
    #         initialize_block_scaled_batched_from_tensor,
    #         a_quant, b_quant, a_scale, b_scale, 
    #         block_scale_type=block_scale_type, 
    #         compute_reference=False
    #     )
    # del a_quant, b_quant, a_scale, b_scale
    
    # print("testing block_scaled_batched_matmul...")
    # out_mxfp, time_step_mxfp_attn = measure_time(
    #     block_scaled_batched_attn, 
    #     a_desc, a_scale_proc, b_desc, b_scale_proc, v,
    #     torch.float16, batch_size, num_heads, seq_len, seq_len, head_dim, configs
    # )
    # del a_desc, a_scale_proc, b_desc, b_scale_proc, configs
    # torch.cuda.empty_cache()
    # import gc; gc.collect()
    
    # print("testing block_scaled_batched_attn...")
    # out_mxfp, time_total_mxfp = measure_time(
    #     block_scaled_batched_attn, a_desc, a_scale_proc, b_desc, b_scale_proc, v,
    #     torch.float16, batch_size, num_heads, seq_len, seq_len, head_dim, configs
    # )
    
    # 测试 torch.nn.functional.scaled_dot_product_attention
    print("testing torch.nn.functional.scaled_dot_product_attention...")
    def test_sdpa(q, k, v, is_causal=False):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )
    
    out_torch, time_total_sdpa = measure_time(
        test_sdpa, q, k, v, is_causal=False
    )
    
    
    # 测试 flash-attention triton
    # print("testing flash-attention triton...")
    # def test_flash_attn_triton(q, k, v, is_causal=False):
    #     out = torch.empty_like(q)
    #     # 准备metadata
    #     # metadata = type('Metadata', (), {
    #     #     'sm_scale': 1.0 / (head_dim ** 0.5),
    #     #     'alibi_slopes': None,
    #     #     'causal': is_causal,
    #     #     'layout': 'HND',
    #     #     'cu_seqlens_q': torch.tensor([0, seq_len], device='cuda', dtype=torch.int32),
    #     #     'cu_seqlens_k': torch.tensor([0, seq_len], device='cuda', dtype=torch.int32),
    #     #     'max_seqlens_q': seq_len,
    #     #     'max_seqlens_k': seq_len,
    #     #     'cache_seqlens': None,
    #     #     'cache_batch_idx': None,
    #     #     'dropout_p': 0.0,
    #     #     'philox_seed': 0,
    #     #     'philox_offset': 0,
    #     #     'return_scores': False,
    #     #     'use_exp2': False
    #     # })
        
    #     out = flash_attn_func(
    #         q, k, v,
    #         None,
    #         is_causal,
    #         1.0 / (head_dim ** 0.5)
    #     )
    #     return out
    
    # out_fa, time_total_flash = measure_time(
    #     test_flash_attn_triton, q, k, v, is_causal=False
    # )
    
    # 打印结果
    print(f"\n{' Performance Test ':=^50}")
    print(f"**** Shape ****\nbatch_size: {batch_size}, num_heads: {num_heads}, seq_len: {seq_len}, head_dim: {head_dim}")
    # print("**** spas_sage_attn_meansim (triton) kernel 分析 ****")
    # print(f"get_block_map_meansim: {time_block_map:.4f} ms ({time_block_map/time_total_spas*100:.4f}%)")
    # print(f"per_block_int8: {time_int8:.4f} ms ({time_int8/time_total_spas*100:.4f}%)")
    # print(f"forward: {time_forward:.4f} ms ({time_forward/time_total_spas*100:.4f}%), average time: {time_forward/iter_times:.4f} ms")
    
    # print("**** 总体算子时间对比 ****")
    print(f"mxfp_attn_kernel ({block_scale_type} triton) total: {time_total_mxfp:.4f} ms, average time: {time_total_mxfp/iter_times:.4f} ms")
    # print(f"mxfp_attn_kernel (triton) quant: {time_step_quant_mxfp:.4f} ms, average time: {time_step_quant_mxfp/iter_times:.4f} ms")
    # print(f"mxfp_attn_kernel (triton) init: {time_step_mxfp_init:.4f} ms, average time: {time_step_mxfp_init/iter_times:.4f} ms")
    # print(f"mxfp_attn_kernel (triton) attn: {time_step_mxfp_attn:.4f} ms, average time: {time_step_mxfp_attn/iter_times:.4f} ms")
    # print(f"\nspas (triton): {time_total_spas:.4f} ms, average time: {time_total_spas/iter_times:.4f} ms")
    # print(f"spas_full (triton): {time_total_spas_full:.4f} ms, average time: {time_total_spas_full/iter_times:.4f} ms")
    
    # print(f"spas_sage_attn_meansim_cuda (cuda): {time_total_spas_cuda:.4f} ms, average time: {time_total_spas_cuda/iter_times:.4f} ms")
    # print(f"spas_sage_attn_meansim_cuda_full (cuda): {time_total_spas_cuda_full:.4f} ms, average time: {time_total_spas_cuda_full/iter_times:.4f} ms")
    
    print(f"torch.nn.functional.scaled_dot_product_attention (torch): {time_total_sdpa:.4f} ms, average time: {time_total_sdpa/iter_times:.4f} ms")
    # print(f"flash-attention (triton): {time_total_flash:.4f} ms, average time: {time_total_flash/iter_times:.4f} ms")
    
    # 计算mse
    if out_mxfp is not None:
        mse_mxfp = torch.nn.functional.mse_loss(out_mxfp, out_torch)
    else:
        mse_mxfp = None
    # mse_spas = torch.nn.functional.mse_loss(out_spas, out_torch)
    # mse_spas_full = torch.nn.functional.mse_loss(out_spas_full, out_torch)
    # mse_spas_cuda = torch.nn.functional.mse_loss(out_spas_cuda, out_torch)
    # mse_spas_cuda_full = torch.nn.functional.mse_loss(out_spas_cuda_full, out_torch)   
    # mse_fa = torch.nn.functional.mse_loss(out_fa, out_torch)
    
    # import pdb; pdb.set_trace()
    if mse_mxfp is not None:
        print(f"mse_mxfp {block_scale_type}: {mse_mxfp:.6f}")
    # print(f"mse_spas: {mse_spas:.6f}, mse_spas_full: {mse_spas_full:.6f}")
    # print(f"mse_spas_cuda: {mse_spas_cuda:.6f}, mse_spas_cuda_full: {mse_spas_cuda_full:.6f}")
    # print(f"mse_fa: {mse_fa:.6f}")

if __name__ == "__main__":
    test_performance() 
