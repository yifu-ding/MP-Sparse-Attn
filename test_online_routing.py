from httpx import head
import torch
import time
from ours.online_routing import online_routing_attn
import triton
import pdb
import os

iter_time = 1
def measure_time(func, *args, **kwargs):
    # Warmup
    for _ in range(3):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Measure time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iter_time):
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
    num_heads = 4
    seq_len = 512
    head_dim = 128
    
    # 从本地加载保存的qkv数据
    save_dir = "./results/saved_qkv"
    save_name = f"qkv_bsz{batch_size}_qlen2083_layer0.pt"  # 使用第0层的数据
    save_path = os.path.join(save_dir, save_name)
    
    # 加载数据
    qkv_data = torch.load(save_path)
    q = qkv_data['query_states'][:batch_size, :num_heads, :seq_len, :head_dim].to(device='cuda', dtype=torch.float16)
    k = qkv_data['key_states'][:batch_size, :num_heads, :seq_len, :head_dim].to(device='cuda', dtype=torch.float16)
    v = qkv_data['value_states'][:batch_size, :num_heads, :seq_len, :head_dim].to(device='cuda', dtype=torch.float16)
    
    # Test online_routing_attn
    print("\nTesting online_routing_attn:")
    # skip_thresh = 0.5
    for skip_thresh in range(0,10):
        skip_thresh = skip_thresh / 10.0
        
        output, time_total = measure_time(
            online_routing_attn, q, k, v, 
            is_causal=False, 
            attention_sink=False,
            skip_thresh=skip_thresh
        )
    
        # output_nosparse, time_total_nosparse = measure_time(
        #     online_routing_attn, q, k, v, 
        #     is_causal=False, 
        #     attention_sink=False,
        #     skip_thresh=None
        # )
        
        print("testing torch.nn.functional.scaled_dot_product_attention...")
        def test_sdpa(q, k, v, is_causal=False):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        
        output_sdpa, time_total_sdpa = measure_time(
            test_sdpa, q, k, v, is_causal=False
        )
        
        # Print results
        print(f"\n{'Performance Test Results':=^50}")
        print(f"batch_size: {batch_size}, num_heads: {num_heads}, seq_len: {seq_len}, head_dim: {head_dim}")
        print(f"online_routing_attn total time: {time_total:.2f} ms, average time: {time_total/iter_time:.2f} ms")
        # print(f"online_routing_attn (nosparse) total time: {time_total_nosparse:.2f} ms, average time: {time_total_nosparse/iter_time:.2f} ms")
        print(f"torch.nn.functional.scaled_dot_product_attention total time: {time_total_sdpa:.2f} ms, average time: {time_total_sdpa/iter_time:.2f} ms")
        
        # Verify output shape
        print(f"\nInput shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        print(f"Output shape: {output.shape}, is correct: {output.shape == output_sdpa.shape}")

        mse_loss = torch.nn.functional.mse_loss(output, output_sdpa)
        l1_loss = torch.nn.functional.l1_loss(output, output_sdpa)
        sum_abs_loss = torch.sum(torch.abs(output - output_sdpa))
        print(f"thresh: {skip_thresh}, MSE Loss: {mse_loss:.6f}, L1 Loss: {l1_loss:.6f}, Sum of abs: {sum_abs_loss:.6f}")
        
    # # Check for NaN values
    # if torch.isnan(output).any():
    #     print("Warning: Output contains NaN values!")
    # else:
    #     print("Output does not contain NaN values")
    
    # # Check for inf values
    # if torch.isinf(output).any():
    #     print("Warning: Output contains inf values!")
    # else:
    #     print("Output does not contain inf values")

if __name__ == "__main__":
    test_performance() 