# for ablation study
import torch
import time
# from spas_sage_attn import spas_sage_attn_meansim_cuda
# from spas_sage_attn.utils import get_block_map_meansim
# from spas_sage_attn.triton_kernel_example import spas_sage_attn_meansim, per_block_int8, forward as forward_triton
# from flash_attn.flash_attn_triton import flash_attn_func
import numpy as np
from ours.mxfp_attn_kernel import mxfp_attn_kernel, block_scaled_batched_attn
# from ours.batched_block_scaled_matmul import test_batched_matmul, initialize_block_scaled_batched_from_tensor
from ours.quant_kernels import quant_fpxint8, quant_mxfp8e5, quant_mxfp4, quant_mxfp8e5_nvfp4, quant_nvfp4
import random
import os
from ours.modify_mxfp_attn import precision_metric
from ours.mxfp import MXFP4Tensor, MXScaleTensor, MXFP8Tensor
import triton
import triton.language as tl
import torch.nn.functional as F

iter_times = 1

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


def precision_metric(quant_o, fa2_o, verbose=False, round_num=4): 
    if quant_o.shape[-2] > 200000:
        quant_o, fa2_o = quant_o.cpu(), fa2_o.cpu()
    x, xx = quant_o.float(), fa2_o.float() 
    sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    l1 =   ( (x - xx).abs().sum() / xx.abs().sum() ).item()
    rmse = torch.sqrt(torch.mean((x -xx) ** 2)).item()
    sim = round(sim, round_num)
    l1 = round(l1, round_num)
    rmse = round(rmse, round_num)
    psnr = 10 * np.log10((10 ** 2) / rmse**2)
    if verbose: print(f'Cossim: {sim:.6f}, L1: {l1:.6f}, RMSE:{rmse:.6f}, PSNR:{psnr:.6f}')
    return {"Cossim": sim, "L1": l1, "RMSE": rmse, "PSNR": psnr}

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
    query_states = saved_data['query_states'].to(device='cuda', dtype=torch.float16)[:, :, :, :]
    key_states = saved_data['key_states'].to(device='cuda', dtype=torch.float16)[:, :, :, :]
    value_states = saved_data['value_states'].to(device='cuda', dtype=torch.float16)[:, :, :, :]
    return query_states, key_states, value_states
    
def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test_tile_size_ablation():

    seed_all(41)

    batch_size = 1
    num_heads = 24
    qo_len = 4096
    kv_len = 4096
    head_dim = 128
    is_causal = True
    block_scale_type = "mxfp8_diag"
    qk_dtype = 'e4m3'
    smooth_k = True
    dual_scale = True
    quant_granularity = "tokenwise"

    print(f"block_scale_type: {block_scale_type}, qk_dtype: {qk_dtype}, dual_scale: {dual_scale}, quant_granularity: {quant_granularity}")

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

    print("Start performance test...")
    
    def test_sdpa(q, k, v, is_causal=False):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )

    out_torch = test_sdpa(q, k, v)

    results = {}
    # try:
    for sink_size in [0,1,2,4,6,8,16]:
        for tile_size in [0,1,2,4,6,8,16]:# 4, 8, 16]: #  8, 16, 32]:
    # for sink_size in [0]:
    #     for tile_size in [0]:# 4, 8, 16]: #  8, 16, 32]:
            print(f"fp8_tile_num={tile_size}, sink_size={sink_size}")
            results[f"{tile_size}_{sink_size}"] = {"time_total": 0, "ops": 0, "Cossim": 0, "L1": 0, "RMSE": 0, "PSNR": 0}
            out, time_total, ops = measure_time(
                mxfp_attn_kernel, q, k, v, is_causal=is_causal, block_scale_type=block_scale_type, \
                    smooth_k=smooth_k, dual_scale=dual_scale, quant_granularity=quant_granularity, \
                    fuse_mp_quant=True, pre_quant=True, fp8_tile_num=tile_size, sink_size=sink_size, \
                    qk_dtype=qk_dtype  # save_qk=True
            )
            # import pdb; pdb.set_trace()
            measures = precision_metric(out, out_torch)  # {"Cossim": sim, "L1": l1, "RMSE": rmse, "PSNR": psnr}
            for key, value in measures.items():
                results[f"{tile_size}_{sink_size}"][key] = value
            results[f"{tile_size}_{sink_size}"]["time_total"] = time_total * 1000
            results[f"{tile_size}_{sink_size}"]["ops"] = ops
    # except Exception as e:
    #     print(e)
    #     pass

    # 打印结果
    print(f"| {'tile size':<10} | {'sink size':<10} | {'Time (us)':>10} | {'ops':>10} | {'Cossim':>10} | {'L1':>10} | {'RMSE':>10} | {'PSNR':>10} |")
    print("|" + "------|" * 8)

    for size, measures in results.items():
        tile_size, sink_size = size.split("_")
        print(f"| {tile_size:<10} | {sink_size:<10} | {measures['time_total'] :10.3f} | {measures['ops']:10.3f} | {measures['Cossim']:10.3f} | {measures['L1']:10.3f} | {measures['RMSE']:10.3f} | {measures['PSNR']:10.3f} |")

if __name__ == "__main__":
    test_tile_size_ablation()