import torch
import os
from ours.mxfp_attn_kernel import mxfp_attn_kernel
from ours.modify_mxfp_attn import precision_metric
from torch.nn import functional as F
from tests.test_quant import test_quant_mxfp4_input_quant_tensor

def load_attention_states():
    """
    加载保存的注意力状态数据
    """
    file_path = 'saved_files/low_sim_attn_states.pth'
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    # 加载保存的数据
    saved_data = torch.load(file_path)
    
    # 提取各个状态
    query_states = saved_data['query_states'].to(device='cuda', dtype=torch.float16)[:, :, :128, :]
    key_states = saved_data['key_states'].to(device='cuda', dtype=torch.float16)[:, :, 256:384, :]
    value_states = saved_data['value_states'].to(device='cuda', dtype=torch.float16)[:, :, 256:384, :]
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
    
    qk_mxfp, ret_dict = mxfp_attn_kernel(query_states, key_states, value_states, is_causal=False, \
        block_scale_type="mxfp4", output_dtype=torch.float32, return_quant_tensor=True, smooth_k=False, \
            save_qk=True, dual_scale=False)  # saved qk

    a_fp4 = ret_dict['a_fp4']
    a_scale = ret_dict['a_scale']
    b_fp4 = ret_dict['b_fp4']
    b_scale = ret_dict['b_scale']

    # qk_quant1 = test_quant_mxfp4_input_quant_tensor(a_fp4, b_fp4, a_scale, b_scale, head_dim=128)
    # qk_quant1_softmax = torch.nn.functional.softmax(qk_quant1, dim=-1)
    # qkv_quant1 = torch.matmul(qk_quant1_softmax.to(torch.float16), value_states.to(torch.float16))

    print("matmul qk vs. mxfp qk")
    precision_metric(qk_mxfp, qk_ref)
    # import pdb; pdb.set_trace()

    out_mxfp, ret_dict = mxfp_attn_kernel(query_states, key_states, value_states, is_causal=False, \
        block_scale_type="mxfp4", output_dtype=torch.float32, return_quant_tensor=True, smooth_k=False, \
            save_qk=False, dual_scale=False)  # saved qk

    qk = qk_mxfp
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

    qk_ref_softmax = torch.nn.functional.softmax(qk, dim=-1)
    qkv_ref = torch.matmul(qk_ref_softmax.to(torch.float32), value_states.to(torch.float32))
    print("onlinesoftmax(torch) output vs. mxfp output")
    precision_metric(output, out_mxfp)
    print("qkv(torch matmul) output vs. mxfp output")
    precision_metric(qkv_ref, out_mxfp)
    import pdb; pdb.set_trace()

    # sim = F.cosine_similarity(out_mxfp.reshape(1, -1), o.reshape(1, -1)).item()
    # print(f"sim: {sim}")
    # mse_mxfp = torch.nn.functional.mse_loss(out_mxfp, o)
    # print(f"mse_mxfp: {mse_mxfp}")
    # 逐个 head 计算相似度
    num_heads = query_states.shape[1]
    # print("\n按 head 计算相似度:")
    for head_idx in range(num_heads):
        # 按序列长度计算相似度
        seq_len = query_states.shape[2]
        # print(f"\nHead {head_idx} 按序列位置计算相似度:")
        for seq_idx in range(100):
            out_mxfp_seq = out_mxfp[:, head_idx:head_idx+1, seq_idx:seq_idx+1, :]
            o_seq = o[:, head_idx:head_idx+1, seq_idx:seq_idx+1, :]
            sim_seq = F.cosine_similarity(out_mxfp_seq.reshape(1, -1), o_seq.reshape(1, -1)).item()
            mse_seq = torch.nn.functional.mse_loss(out_mxfp_seq, o_seq)
            if sim_seq < 0.7:
                print(f"位置 {seq_idx}: sim = {sim_seq:.6f}, mse = {mse_seq:.6f}")
                # import pdb; pdb.set_trace()
                # 取出当前head的query_states, key_states, value_states
                q_tmp = query_states[:, head_idx:head_idx+1, seq_idx:seq_idx+1, :]
                k_tmp = key_states[:, head_idx:head_idx+1, :, :]
                v_tmp = value_states[:, head_idx:head_idx+1, :, :]
                print(f"q_tmp shape: {q_tmp.shape}")
                print(f"k_tmp shape: {k_tmp.shape}")
                print(f"v_tmp shape: {v_tmp.shape}")
                out_mxfp_tmp, ret_dict = mxfp_attn_kernel(q_tmp, k_tmp, v_tmp, is_causal=False, block_scale_type="mxfp4", \
                    output_dtype=torch.float16, return_quant_tensor=True)

                ori_a_fp4 = a_fp4[:, head_idx:head_idx+1, seq_idx:seq_idx+1, :]
                ori_a_scale = a_scale[:, head_idx:head_idx+1, seq_idx:seq_idx+1, :]
                ori_b_fp4 = b_fp4[:, head_idx:head_idx+1, :, :]
                ori_b_scale = b_scale[:, head_idx:head_idx+1, :, :]

                qk_quant1 = test_quant_mxfp4_input_quant_tensor(ori_a_fp4, ori_b_fp4, ori_a_scale, ori_b_scale, head_dim=128)
                qk_quant1_softmax = torch.nn.functional.softmax(qk_quant1, dim=-1)
                qkv_quant1 = torch.matmul(qk_quant1_softmax.to(torch.float16), v_tmp.to(torch.float16))

                # a_fp4_tmp = ret_dict['a_fp4']
                # a_scale_tmp = ret_dict['a_scale']
                # b_fp4_tmp = ret_dict['b_fp4']
                # b_scale_tmp = ret_dict['b_scale']
                # qk_quant2 = test_quant_mxfp4_input_quant_tensor(a_fp4_tmp, b_fp4_tmp, a_scale_tmp, b_scale_tmp, head_dim=128)

                # sim_qk = F.cosine_similarity(qk_quant1.reshape(1, -1), qk_quant2.reshape(1, -1)).item()
                # mse_qk = torch.nn.functional.mse_loss(qk_quant1, qk_quant2)
                # print(f"位置 {seq_idx}: sim = {sim_qk:.6f}, mse = {mse_qk:.6f}")
                sim_qkv = F.cosine_similarity(qkv_quant1.reshape(1, -1), o_seq.reshape(1, -1)).item()
                mse_qkv = torch.nn.functional.mse_loss(qkv_quant1, o_seq)
                print(f"外部算 pv: 位置 {seq_idx}: sim = {sim_qkv:.6f}, mse = {mse_qkv:.6f}")

                sim_tmp = F.cosine_similarity(out_mxfp_tmp.reshape(1, -1), o_seq.reshape(1, -1)).item()
                mse_tmp = torch.nn.functional.mse_loss(out_mxfp_tmp, o_seq)
                print(f"kernel 内算 pv: 位置 {seq_idx}: sim = {sim_tmp:.6f}, mse = {mse_tmp:.6f}")
                
                # ori_a_fp4 = a_fp4[:, head_idx:head_idx+1, seq_idx:seq_idx+1, :]
                # ori_a_scale = a_scale[:, head_idx:head_idx+1, seq_idx:seq_idx+1, :]
                # ori_b_fp4 = b_fp4[:, head_idx:head_idx+1, seq_idx:seq_idx+1, :]
                # ori_b_scale = b_scale[:, head_idx:head_idx+1, seq_idx:seq_idx+1, :]

                import pdb; pdb.set_trace()


        # out_mxfp_head = out_mxfp[:, head_idx:head_idx+1, :, :]
        # o_head = o[:, head_idx:head_idx+1, :, :]
        # sim_head = F.cosine_similarity(out_mxfp_head.reshape(1, -1), o_head.reshape(1, -1)).item()
        # mse_head = torch.nn.functional.mse_loss(out_mxfp_head, o_head)
        # print(f"Head {head_idx}: sim = {sim_head:.6f}, mse = {mse_head:.6f}")

if __name__ == "__main__":
   load_attention_states()
    
   