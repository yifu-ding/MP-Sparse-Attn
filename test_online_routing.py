import torch
import time
from ours.online_routing import online_routing_attn
import triton
import pdb
import os

def measure_time(func, *args, **kwargs):
    # Warmup
    for _ in range(3):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Measure time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    result = func(*args, **kwargs)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    
    return result, elapsed_time

def test_performance():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create test data
    batch_size = 1
    num_heads = 24
    seq_len = 2048
    head_dim = 128
    
    # 从本地加载保存的qkv数据
    save_dir = "./results/saved_qkv"
    save_name = f"qkv_bsz{batch_size}_qlen2083_layer0.pt"  # 使用第0层的数据
    save_path = os.path.join(save_dir, save_name)
    
    # 加载数据
    qkv_data = torch.load(save_path)
    q = qkv_data['query_states'][:, :, :seq_len, :].to(device='cuda', dtype=torch.float16)
    k = qkv_data['key_states'][:, :, :seq_len, :].to(device='cuda', dtype=torch.float16)
    v = qkv_data['value_states'][:, :, :seq_len, :].to(device='cuda', dtype=torch.float16)
    
    # Test online_routing_attn
    print("\nTesting online_routing_attn:")
    output, time_total = measure_time(
        online_routing_attn, q, k, v, 
        is_causal=False, 
        simthreshd1=0.3, 
        cdfthreshd=0.96,
        pvthreshd=20,
        attention_sink=False
    )
    
    # Print results
    print(f"\n{'Performance Test Results':=^50}")
    print(f"batch_size: {batch_size}, num_heads: {num_heads}, seq_len: {seq_len}, head_dim: {head_dim}")
    print(f"online_routing_attn total time: {time_total:.2f} ms")
    
    # Verify output shape
    print(f"\nOutput shape: {output.shape}")
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    
    # Check for NaN values
    if torch.isnan(output).any():
        print("Warning: Output contains NaN values!")
    else:
        print("Output does not contain NaN values")
    
    # Check for inf values
    if torch.isinf(output).any():
        print("Warning: Output contains inf values!")
    else:
        print("Output does not contain inf values")

if __name__ == "__main__":
    test_performance() 