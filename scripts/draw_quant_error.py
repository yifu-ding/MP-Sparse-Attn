from sympy import false
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
from ours.mxfp import MXFP4Tensor
import matplotlib.pyplot as plt
from scripts.debug3 import compute_reference, compute_reference_nvfp4, compute_reference_mxfp8
import numpy as np

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

def draw_heatmap(residual, x_label, y_label, title, vmax=None):
    
    fontfamily = "DejaVu Serif"

    for head_idx in range(residual.shape[1]):
        residual_head = residual[0, head_idx] #, :256, :256]
        
        plt.figure(figsize=(10,8))
        plt.xlabel(x_label, fontfamily=fontfamily, fontsize=14)
        plt.ylabel(y_label, fontfamily=fontfamily, fontsize=14)
        plt.xticks(fontfamily=fontfamily, fontsize=12)
        plt.yticks(fontfamily=fontfamily, fontsize=12)
        # plt.legend(prop={"family": fontfamily, "size": 12}, loc="upper right")	

        residual_data = residual_head.cpu().numpy()
        vmax = abs(residual_data).max() if vmax is None else vmax
        print(abs(residual_data).max())
        # vmax = 12
        # vmax = residual_data.max()
        # vmin = residual_data.min()
        # midpoint = (vmax + vmin) / 2
        # mid = torch.full(residual_data.shape, midpoint)
        # mid = torch.triu(mid, diagonal=1) 
        # residual_data += mid.cpu().numpy()
        # plt.imshow(residual_data, cmap='RdBu', vmin=-vmax, vmax=vmax)
        plt.imshow(residual_data, cmap='seismic', vmin=-vmax, vmax=vmax)
        # plt.colorbar()
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=12)  # 设置刻度字体大小
        for t in cbar.ax.get_yticklabels():
            t.set_fontfamily(fontfamily)

        plt.title(f'{title} Head {head_idx}')
        file_name = title.replace(" ", "_")
        if not os.path.exists(f'saved_figs/128/head{head_idx}'):
            os.makedirs(f'saved_figs/128/head{head_idx}')
        plt.savefig(f'saved_figs/128/head{head_idx}/nondual_{file_name}.png')
        # plt.savefig(f'saved_figs/quant_error/{file_name}_head{head_idx}.png')
        plt.tight_layout()
        plt.close()
        return 

    print(f"Saved {file_name} heatmap to saved_figs/")
    print("*"*20)


def combine_mp_matrix(qk_quant8, q_deq8, k_deq8, qk_quant4, q_deq4, k_deq4):
    window_T = 128
    B, H, M, N = qk_quant4.shape
    
    qk_quant = qk_quant4.clone()
    q_deq = q_deq4.clone() 
    k_deq = k_deq4.clone()

    # 对每个batch和head进行处理
    for b in range(B):
        for h in range(H):
            # 按window_T大小遍历对角块
            # for i in range(0, M, window_T):
            #     # for j in range(0, N, window_T):
            #     # if i == j:  # 在对角线上的块
            #         # 计算当前块的实际大小(处理边界情况)
            #     curr_M = min(window_T, M-i)
            #     curr_N = min(window_T, N-i)
                
            #     # 替换对角块为8-bit版本
            #     qk_quant[b,h,i:i+curr_M,i:i+curr_N] = qk_quant8[b,h,i:i+curr_M,i:i+curr_N]
            #     q_deq[b,h,i:i+curr_M,:] = q_deq8[b,h,i:i+curr_M,:]
            #     k_deq[b,h,i:i+curr_N,:] = k_deq8[b,h,i:i+curr_N,:]

            qk_quant[b,h,:,:window_T] = qk_quant8[b,h,:,:window_T]
    return qk_quant, q_deq, k_deq


def prepare_tensor(q, k, v, block_scale_type="mxfp4", dual_scale = True):
    BLKQ = 128
    BLKK = 128
  
    B, H, M, K = q.shape 
    N = k.shape[2]
    qo_len = M
    kv_len = N
    
    pack_along_lastdim = False  # True: kernel fusion support for pack fp4 tensor into uint8 tensor

    # save_qk = True
    # out_mxfp, ret_dict = mxfp_attn_kernel(q, k, v, is_causal=is_causal, \
    # block_scale_type=block_scale_type, output_dtype=torch.float32, return_quant_tensor=True, smooth_k=True, \
    #     save_qk=save_qk, dual_scale=False, quant_granularity="tokenwise")  # saved qk

    # q_fp = ret_dict['q_fp']
    # q_scale = ret_dict['q_scale']
    # k_fp = ret_dict['k_fp'] 
    # k_scale = ret_dict['k_scale']
    # q_scale_2 = ret_dict['q_scale_2']
    # k_scale_2 = ret_dict['k_scale_2']

    if block_scale_type == "mxfp4":
        q_fp, q_scale, k_fp, k_scale, q_scale_2, k_scale_2 = quant_mxfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            pack_along_lastdim=pack_along_lastdim, dual_scale=dual_scale, quant_granularity="tokenwise")
        qk_quant, q_deq, k_deq = compute_reference(q_fp, q_scale, k_fp, k_scale, VEC_SIZE=32, M=M, K=K, N=N, per_channel=False)
    elif block_scale_type == "nvfp4":
        q_fp, q_scale, k_fp, k_scale, q_scale_2, k_scale_2 = quant_nvfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            pack_along_lastdim=pack_along_lastdim, dual_scale=dual_scale, quant_granularity="tokenwise")
        qk_quant, q_deq, k_deq = compute_reference_nvfp4(q_fp, q_scale, k_fp, k_scale, VEC_SIZE=16, M=M, K=K, N=N, per_channel=False)
        if dual_scale:
            q_deq = q_deq * q_scale_2
            k_deq = k_deq * k_scale_2
            qk_quant = qk_quant * q_scale_2 * k_scale_2.transpose(-1, -2)
    elif block_scale_type == "mxfp8":
        q_fp, q_scale, k_fp, k_scale, q_scale_2, k_scale_2 = quant_mxfp8e5(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            dual_scale=dual_scale, quant_granularity="tokenwise")
        qk_quant, q_deq, k_deq = compute_reference_mxfp8(q_fp, q_scale, k_fp, k_scale, VEC_SIZE=32, M=M, K=K, N=N, per_channel=False)
    elif block_scale_type == "ours":
        q_fp8, q_scale8, k_fp8, k_scale8, q_scale_28, k_scale_28 = quant_mxfp8e5(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            dual_scale=False, quant_granularity="tokenwise")
        qk_quant8, q_deq8, k_deq8 = compute_reference_mxfp8(q_fp8, q_scale8, k_fp8, k_scale8, VEC_SIZE=32, M=M, K=K, N=N, per_channel=False)

        q_fp4, q_scale4, k_fp4, k_scale4, q_scale_24, k_scale_24 = quant_nvfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, \
            pack_along_lastdim=pack_along_lastdim, dual_scale=dual_scale, quant_granularity="tokenwise")
        qk_quant4, q_deq4, k_deq4 = compute_reference_nvfp4(q_fp4, q_scale4, k_fp4, k_scale4, VEC_SIZE=16, M=M, K=K, N=N, per_channel=False)

        if dual_scale:
            q_deq4 = q_deq4 * q_scale_24
            k_deq4 = k_deq4 * k_scale_24
            qk_quant4 = qk_quant4 * q_scale_24 * k_scale_24.transpose(-1, -2)

        qk_quant, q_deq, k_deq = combine_mp_matrix(qk_quant8, q_deq8, k_deq8, qk_quant4, q_deq4, k_deq4)


    q_deq = q_deq / ((K ** -0.5) * 1.44269504)

    
    return qk_quant, q_deq, k_deq

def cal_and_draw(q, k, v, qk_quant, q_deq, k_deq, block_scale_type="mxfp4"):

    B, H, M, K = q.shape 
    N = k.shape[2]

    qk_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (K ** -0.5)
    mask = torch.full((M, N), 1, device=q.device)  # 给 qk 加 0
    mask = 1 - torch.triu(mask, diagonal=1)  # 下三角为 1   # 给 qk 加 0
    qk_ref = qk_ref * mask  # broadcasting 会自动扩展成 (B, H, M, N)
    qk_quant = qk_quant * mask  # broadcasting 会自动扩展成 (B, H, M, N)
   
    mxfp4_vmax = [4.7684274, 0.40252528, 0.75223505, 7.868451, 7.4296875]
    nvfp4_nondual_vmax = [6.1319537, 0.19816749, 0.2692886, 1.447371, 1.3789062]
    nvfp4_dual_vmax = [5.498338, 0.20012604, 0.27224413, 1.4453123, 1.3818359]
    draw_heatmap(qk_ref - qk_quant, "K sequence", "Q sequence", f"QK {block_scale_type}", vmax=mxfp4_vmax[0])

    mask = torch.full((M, N), float('-inf'), device=q.device)  # 给 softmax 的 qk 加上 mask
    mask = torch.triu(mask, diagonal=1) # 给 softmax 的 qk 加上 mask
    qk_ref = qk_ref + mask  # broadcasting 会自动扩展成 (B, H, M, N)
    qk_quant = qk_quant + mask  # broadcasting 会自动扩展成 (B, H, M, N)
    qk_ref_softmax = torch.nn.functional.softmax(qk_ref, dim=-1)
    qk_mxfp_softmax = torch.nn.functional.softmax(qk_quant, dim=-1)
    qkv_ref = torch.matmul(qk_ref_softmax.to(torch.float32), v.to(torch.float32))
    qkv_mxfp = torch.matmul(qk_mxfp_softmax.to(torch.float32), v.to(torch.float32))

    draw_heatmap(qk_ref_softmax - qk_mxfp_softmax, "K sequence", "Q sequence", f"QK sm {block_scale_type}", vmax=mxfp4_vmax[1])
    draw_heatmap(qkv_ref - qkv_mxfp, "Head Dim", "Q sequence", f"QKV {block_scale_type}", vmax=mxfp4_vmax[2])
    draw_heatmap(q - q_deq, "Head Dim", "Q sequence", f"Q {block_scale_type}", vmax=mxfp4_vmax[3])
    draw_heatmap(k - k_deq, "Head Dim", "K sequence", f"K {block_scale_type}", vmax=mxfp4_vmax[4])


def cal_cos_sim(q, k, v, qk_quant, q_deq, k_deq, block_scale_type="mxfp4"):
    B, H, M, K = q.shape 
    N = k.shape[2]

    q_sim = precision_metric(q, q_deq)
    k_sim = precision_metric(k, k_deq)

    qk_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (K ** -0.5)
    qk_sim = precision_metric(qk_ref, qk_quant)
    
    mask = torch.full((M, N), 1, device=q.device)  # 给 qk 加 0
    mask = 1 - torch.triu(mask, diagonal=1)  # 下三角为 1   # 给 qk 加 0
    qk_ref = qk_ref * mask  # broadcasting 会自动扩展成 (B, H, M, N)
    qk_quant = qk_quant * mask  # broadcasting 会自动扩展成 (B, H, M, N)

    qk_ref_softmax = torch.nn.functional.softmax(qk_ref, dim=-1)
    qk_mxfp_softmax = torch.nn.functional.softmax(qk_quant, dim=-1)

    qk_sm_sim = precision_metric(qk_ref_softmax, qk_mxfp_softmax)

    qkv_ref = torch.matmul(qk_ref_softmax.to(torch.float32), v.to(torch.float32))
    qkv_mxfp = torch.matmul(qk_mxfp_softmax.to(torch.float32), v.to(torch.float32))

    qkv_sim = precision_metric(qkv_ref, qkv_mxfp)

    # 格式化打印输出，保留3位小数
    # 行是 Cossim, L1 , RMSE， PSNR，列是q, k, qk, qk_sm, qkv

    def print_similarity_metrics(q_sim, k_sim, qk_sim, qk_sm_sim, qkv_sim):
        metrics = ["Cossim", "L1", "RMSE", "PSNR"]
        variables = {
            "q_sim": q_sim,
            "k_sim": k_sim,
            "qk_sim": qk_sim,
            "qk_sm_sim": qk_sm_sim,
            "qkv_sim": qkv_sim,
        }

        # 打印表头
        header = "{:<10}".format("") + "".join([f"{name:<12}" for name in variables.keys()])
        print(header)
        print("-" * len(header))

        # 每一行打印一个 metric
        for metric in metrics:
            row = f"{metric:<10}"
            for var in variables.values():
                val = var.get(metric, float("nan"))
                row += f"{val:<12.3f}"
            print(row)

    print_similarity_metrics(q_sim, k_sim, qk_sim, qk_sm_sim, qkv_sim)


if __name__ == "__main__":
    file_path = 'saved_files/low_sim_attn_states.pth'
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        exit()
    
    # 加载保存的数据
    saved_data = torch.load(file_path)
    
    # for tile in range(1,2):
    query_states = saved_data['query_states'].to(device='cuda', dtype=torch.float16)[:, :, :2048, :]
    key_states = saved_data['key_states'].to(device='cuda', dtype=torch.float16)[:, :, :2048, :]
    value_states = saved_data['value_states'].to(device='cuda', dtype=torch.float16)[:, :, :2048, :]

    smooth_k = True
    if smooth_k: key_states = key_states - key_states.mean(dim=-2, keepdim=True)

    block_scale_type = "ours" 
    dual_scale = True
    qk_quant, q_deq, k_deq = prepare_tensor(query_states, key_states, value_states, block_scale_type=block_scale_type, dual_scale=dual_scale)

    # cal_and_draw(query_states, key_states, value_states, qk_quant, q_deq, k_deq, block_scale_type=block_scale_type)

    cal_cos_sim(query_states, key_states, value_states, qk_quant, q_deq, k_deq, block_scale_type=block_scale_type)