
# 修复批量矩阵乘法kernel - 支持多维输入 (b, h, M, K) 和 (b, h, N, K)
from Levenshtein import quickmedian


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_batched_matmul_kernel(  #
        q_ptr, q_scale,  #
        k_ptr, k_scale,  #
        v_ori,
        o_ptr,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        stride_qb, stride_qh, stride_qm, stride_qk,  # a的strides: batch, head, M, K
        stride_kb, stride_kh, stride_kn, stride_kk,  # b的strides: batch, head, N, K  
        stride_ob, stride_oh, stride_om, stride_on,  # c的strides: batch, head, M, N
        stride_sqb, stride_sqh, stride_sqm, stride_sqk,  # q_scale的strides
        stride_skb, stride_skh, stride_skn, stride_skk,  # k_scale的strides
        # stride_sk: tl.constexpr, stride_sk: tl.constexpr, stride_sc: tl.constexpr, stride_sd: tl.constexpr,
        num_h: tl.constexpr,  # head数量
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  # 128 
        BLOCK_N: tl.constexpr,  # 256
        HEAD_DIM: tl.constexpr,  # 128
        NUM_STAGES: tl.constexpr,  # 4
        USE_2D_SCALE_LOAD: tl.constexpr, 
        STAGE, qo_len, kv_len):  # False

    # 获取三维grid的索引 - 参考_attn_fwd的实现
    start_m = tl.program_id(0)  # M*N维度的块索引
    off_h = tl.program_id(1).to(tl.int64)  # head维度索引
    off_z = tl.program_id(2).to(tl.int64)  # batch维度索引
    
    # 计算M和N维度的块索引
    num_pid_m = tl.cdiv(qo_len, BLOCK_M)
    pid_m = start_m % num_pid_m
    # pid_n = start_m // num_pid_m
    
    # 计算偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N) # 每次计算的 token 列数
    
    offs_k_a = 0
    offs_k_b = 0

    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e5

    # block scale offsets - 参考_attn_fwd的offset计算方式
    offs_sm = (pid_m * (BLOCK_M // 128) + tl.arange(0, BLOCK_M // 128)) % M
    # offs_sn = (pid_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)) % N
    offs_sn = tl.arange(0, BLOCK_N)

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    # 计算当前batch和head的基地址
    q_base_offset = off_z * stride_qb + off_h * stride_qh
    k_base_offset = off_z * stride_kb + off_h * stride_kh
    q_scale_base_offset = off_z * stride_sqb + off_h * stride_sqh
    k_scale_base_offset = off_z * stride_skb + off_h * stride_skh

    # 简化scale load，使用2D模式
    if USE_2D_SCALE_LOAD:
        offs_inner = tl.arange(0, (HEAD_DIM // VEC_SIZE // 4) * 32 * 4 * 4)
        # q_scale_ptr = q_scale + q_scale_base_offset + offs_sm[:, None] * stride_sqm + offs_inner[None, :]
        q_scale_ptr = q_scale + q_scale_base_offset + offs_sm[:, None] * stride_sqm + offs_inner[None, :]
        k_scale_ptr = k_scale + k_scale_base_offset + offs_sn[:, None] * stride_skn + offs_inner[None, :]

        # offs_inner = tl.arange(0, (HEAD_DIM // VEC_SIZE // 4) * 32 * 4 * 4)
        # q_scale_ptr = q_scale + offs_sm[:, None] * stride_sk + offs_inner[None, :]
        # k_scale_ptr = k_scale + offs_sn[:, None] * stride_sk + offs_inner[None, :]
        
    # qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        k_scale_ptr += lo // BLOCK_N
        k_ptrs += stride_kn * lo
        v_ptrs += stride_kn * lo
    elif STAGE == 3:
        lo, hi = 0, kv_len
    
    off_k = tl.arange(0, HEAD_DIM)
    
    q_ptrs = q_ptr + q_base_offset + offs_m[:, None] * stride_qm + off_k[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < qo_len)

    scale_q = tl.load(q_scale_ptr)

    if USE_2D_SCALE_LOAD:
        scale_q = scale_q.reshape(BLOCK_M // 128, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)
    scale_q = scale_q.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, HEAD_DIM // VEC_SIZE)

    # for k in tl.range(0, tl.cdiv(K, HEAD_DIM), num_stages=NUM_STAGES):
    for start_n in range(lo, hi, BLOCK_N):
        # 计算当前batch和head的数据指针  
        # 是e5m2
        
        # q_ptrs = q_ptr + q_base_offset + (offs_m[:, None] * stride_qm + (offs_k_a + tl.arange(0, HEAD_DIM))[None, :])
        k_ptrs = k_ptr + k_base_offset + (offs_n[:, None] * stride_kn + off_k[None, :])
        v_ptrs = v_ori + k_base_offset + (offs_n[:, None] * stride_kn + off_k[None, :])
        
        # 加载数据
        # a = tl.load(q_ptrs)
        k = tl.load(k_ptrs)
        
        # 加载scale因子
        scale_k = tl.load(k_scale_ptr)
        
        if USE_2D_SCALE_LOAD:
            scale_k = scale_k.reshape(BLOCK_N // 128, HEAD_DIM // VEC_SIZE // 4, 32, 4, 4)
        scale_k = scale_k.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, HEAD_DIM // VEC_SIZE)

        qk = tl.dot_scaled(q, scale_q, "e5m2", k.T, scale_k, "e5m2")
        
        if STAGE == 2:   # is_causal
            mask = offs_m[:, None] >= (k + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            local_m = tl.max(qk, 1)
            new_m = tl.maximum(old_m, local_m)
            qk -= new_m[:, None]
        else:
            local_m = tl.max(qk, 1)
            new_m = tl.maximum(old_m, local_m)
            qk = qk - new_m[:, None]
        
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(old_m - new_m)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # if tl.min(new_m - local_m) < pvthreshd:
        v = tl.load(v_ptrs, mask = offs_n[:, None] < (kv_len - start_n))

        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)
        old_m = new_m

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_kn
        
        if USE_2D_SCALE_LOAD:
            k_scale_ptr += BLOCK_N * stride_skn
            
            
    # 存储结果（包含batch和head维度）
    # c_base_offset = off_z * stride_ob + off_h * stride_oh
    # o_ptrs = o_ptr + c_base_offset + (offs_m[:, None] * stride_om + offs_n[None, :])
    # c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # tl.store(o_ptrs, accumulator.to(output_dtype), mask=c_mask)

    acc = acc / l_i[:, None]
    tl.store(o_ptr, acc.to(output_dtype), mask = (offs_m[:, None] < qo_len))
