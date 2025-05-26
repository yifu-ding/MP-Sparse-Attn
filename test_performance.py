import torch
import time
from spas_sage_attn import spas_sage_attn_meansim_cuda
from spas_sage_attn.utils import get_block_map_meansim
from spas_sage_attn.triton_kernel_example import spas_sage_attn_meansim, per_block_int8, forward as forward_triton
from flash_attn.flash_attn_triton import flash_attn_func
import numpy as np

iter_times = 100
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
    num_heads = 24
    seq_len = 1024
    head_dim = 128
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    print("开始性能测试...")
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
    
    
    k_block_indices, time_block_map = measure_time(
        get_block_map_meansim, q, k, is_causal=False, simthreshd1=simthreshd, cdfthreshd=cdfthreshd
    )

    sparsity = k_block_indices.flatten().sum() / k_block_indices.numel()
    print(f"sparsity={sparsity}")
    
    # import pdb; pdb.set_trace()
    (q_int8, q_scale, k_int8, k_scale), time_int8 = measure_time(
        per_block_int8, q, k
    )
    
    pvthreshd = torch.tensor([50.0], device='cuda')
    output, time_forward = measure_time(
        forward_triton, q_int8, k_int8, k_block_indices, v, q_scale, k_scale, pvthreshd,
        is_causal=False, tensor_layout="HND", output_dtype=torch.float16
    )
    
    
    # 总时间
    print("testing spas_sage_attn_meansim...")
    _, time_total_spas = measure_time(
        spas_sage_attn_meansim, q, k, v, is_causal=False, simthreshd1=simthreshd, cdfthreshd=cdfthreshd
    )
    
    print("testing spas_sage_attn_meansim...")
    _, time_total_spas_full = measure_time(
        spas_sage_attn_meansim, q, k, v, is_causal=False, simthreshd1=0.1, cdfthreshd=0.9
    )

    # 测试 spas_sage_attn_meansim_cuda
    print("testing spas_sage_attn_meansim_cuda...")
    (_, qk_sparsity), time_total_spas_cuda = measure_time(
        spas_sage_attn_meansim_cuda, q, k, v, is_causal=False, simthreshd1=simthreshd, cdfthreshd=cdfthreshd, return_sparsity=True
    )
    print(f"qk_sparsity={qk_sparsity}")
    
    (_, qk_sparsity), time_total_spas_cuda_full = measure_time(
        spas_sage_attn_meansim_cuda, q, k, v, is_causal=False, simthreshd1=0.1, cdfthreshd=0.9, return_sparsity=True
    )
    print(f"qk_sparsity_full={qk_sparsity}")
    
    # 测试 torch.nn.functional.scaled_dot_product_attention
    print("testing torch.nn.functional.scaled_dot_product_attention...")
    def test_sdpa(q, k, v, is_causal=False):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )
    
    _, time_total_sdpa = measure_time(
        test_sdpa, q, k, v, is_causal=False
    )
    
    # 测试 flash-attention triton
    print("testing flash-attention triton...")
    def test_flash_attn_triton(q, k, v, is_causal=False):
        out = torch.empty_like(q)
        # 准备metadata
        metadata = type('Metadata', (), {
            'sm_scale': 1.0 / (head_dim ** 0.5),
            'alibi_slopes': None,
            'causal': is_causal,
            'layout': 'HND',
            'cu_seqlens_q': torch.tensor([0, seq_len], device='cuda', dtype=torch.int32),
            'cu_seqlens_k': torch.tensor([0, seq_len], device='cuda', dtype=torch.int32),
            'max_seqlens_q': seq_len,
            'max_seqlens_k': seq_len,
            'cache_seqlens': None,
            'cache_batch_idx': None,
            'dropout_p': 0.0,
            'philox_seed': 0,
            'philox_offset': 0,
            'return_scores': False,
            'use_exp2': False
        })
        
        out = flash_attn_func(
            q, k, v,
            None,
            metadata.causal,
            metadata.sm_scale
        )
        return out
    
    _, time_total_flash = measure_time(
        test_flash_attn_triton, q, k, v, is_causal=False
    )
    
    # 打印结果
    print(f"\n{'性能测试结果':=^50}")
    print(f"batch_size: {batch_size}, num_heads: {num_heads}, seq_len: {seq_len}, head_dim: {head_dim}")
    print("\n*** spas_sage_attn_meansim:")
    print(f"get_block_map_meansim: {time_block_map:.2f} ms ({time_block_map/time_total_spas*100:.2f}%)")
    print(f"per_block_int8: {time_int8:.2f} ms ({time_int8/time_total_spas*100:.2f}%)")
    print(f"forward: {time_forward:.2f} ms ({time_forward/time_total_spas*100:.2f}%)")

    print(f"\ntime_total_spas: {time_total_spas:.2f} ms")
    print(f"time_total_spas_full: {time_total_spas_full:.2f} ms")
    
    print(f"spas_sage_attn_meansim_cuda: {time_total_spas_cuda:.2f} ms")
    print(f"spas_sage_attn_meansim_cuda_full: {time_total_spas_cuda_full:.2f} ms")
    
    print(f"torch.nn.functional.scaled_dot_product_attention: {time_total_sdpa:.2f} ms")
    
    print(f"flash-attention triton: {time_total_flash:.2f} ms")
    
    # print("\n性能对比:")
    # print(f"spas_sage_attn_meansim / scaled_dot_product_attention = {time_total_sdpa/time_total_spas:.2f}")
    # print(f"spas_sage_attn_meansim / flash_attn_triton = {time_total_flash/time_total_spas:.2f}")
    # print(f"spas_sage_attn_meansim / spas_sage_attn_meansim_cuda = {time_total_spas_cuda/time_total_spas:.2f}")
    # print(f"scaled_dot_product_attention / flash_attn_triton = {time_total_flash/time_total_sdpa:.2f}")
    # print(f"scaled_dot_product_attention / spas_sage_attn_meansim_cuda = {time_total_spas_cuda/time_total_sdpa:.2f}")
    # print(f"flash_attn_triton / spas_sage_attn_meansim_cuda = {time_total_spas_cuda/time_total_flash:.2f}")
    
if __name__ == "__main__":
    test_performance() 