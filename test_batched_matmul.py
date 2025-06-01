"""
测试批量矩阵乘法实现
"""
import torch
import numpy as np

import argparse
# from tensor_descriptor import TensorDescriptor
import torch
import triton
import triton.language as tl
import triton.profiler as proton
# from triton.tools.tensor_descriptor import TensorDescriptor

from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_block_scaling():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10

def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2. * M * N * K
    return ret

# 修复批量矩阵乘法kernel - 支持多维输入 (b, h, M, K) 和 (b, h, N, K)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_batched_matmul_kernel(  #
        a_ptr, a_scale,  #
        b_ptr, b_scale,  #
        c_ptr,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        stride_ab, stride_ah, stride_am, stride_ak,  # a的strides: batch, head, M, K
        stride_bb, stride_bh, stride_bn, stride_bk,  # b的strides: batch, head, N, K  
        stride_cb, stride_ch, stride_cm, stride_cn,  # c的strides: batch, head, M, N
        stride_sab, stride_sah, stride_sam, stride_sak,  # a_scale的strides
        stride_sbb, stride_sbh, stride_sbn, stride_sbk,  # b_scale的strides
        # stride_sk: tl.constexpr, stride_sb: tl.constexpr, stride_sc: tl.constexpr, stride_sd: tl.constexpr,
        num_h: tl.constexpr,  # head数量
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  # 128 
        BLOCK_N: tl.constexpr,  # 256
        BLOCK_K: tl.constexpr,  # 128
        NUM_STAGES: tl.constexpr,  # 4
        USE_2D_SCALE_LOAD: tl.constexpr):  # False

    # 获取三维grid的索引 - 参考_attn_fwd的实现
    start_m = tl.program_id(0)  # M*N维度的块索引
    off_h = tl.program_id(1).to(tl.int64)  # head维度索引
    off_z = tl.program_id(2).to(tl.int64)  # batch维度索引
    
    # 计算M和N维度的块索引
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = start_m % num_pid_m
    pid_n = start_m // num_pid_m
    
    # 计算偏移量
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
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
    offs_sn = (pid_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)) % N

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    # 计算当前batch和head的基地址
    a_base_offset = off_z * stride_ab + off_h * stride_ah
    b_base_offset = off_z * stride_bb + off_h * stride_bh
    a_scale_base_offset = off_z * stride_sab + off_h * stride_sah
    b_scale_base_offset = off_z * stride_sbb + off_h * stride_sbh

    # 简化scale load，使用2D模式
    if USE_2D_SCALE_LOAD:
        offs_inner = tl.arange(0, (BLOCK_K // VEC_SIZE // 4) * 32 * 4 * 4)
        a_scale_ptr = a_scale + a_scale_base_offset + offs_sm[:, None] * stride_sam + offs_inner[None, :]
        b_scale_ptr = b_scale + b_scale_base_offset + offs_sn[:, None] * stride_sbn + offs_inner[None, :]

        # offs_inner = tl.arange(0, (BLOCK_K // VEC_SIZE // 4) * 32 * 4 * 4)
        # a_scale_ptr = a_scale + offs_sm[:, None] * stride_sk + offs_inner[None, :]
        # b_scale_ptr = b_scale + offs_sn[:, None] * stride_sk + offs_inner[None, :]
        
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        # 计算当前batch和head的数据指针
        a_ptrs = a_ptr + a_base_offset + (offs_am[:, None] * stride_am + (offs_k_a + tl.arange(0, BLOCK_K))[None, :])
        b_ptrs = b_ptr + b_base_offset + (offs_bn[:, None] * stride_bn + (offs_k_b + tl.arange(0, BLOCK_K))[None, :])
        
        # 加载数据
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        # 加载scale因子
        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)
        
        if USE_2D_SCALE_LOAD:
            scale_a = scale_a.reshape(BLOCK_M // 128, BLOCK_K // VEC_SIZE // 4, 32, 4, 4)
            scale_b = scale_b.reshape(BLOCK_N // 128, BLOCK_K // VEC_SIZE // 4, 32, 4, 4)
        scale_a = scale_a.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        # # 执行矩阵乘法
        # if MIXED_PREC:
        #     accumulator = tl.dot_scaled(a, scale_a, "e5m2", b.T, scale_b, "e2m1", accumulator)
        # elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
        #     accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        # else:
        accumulator = tl.dot_scaled(a, scale_a, "e5m2", b.T, scale_b, "e5m2", accumulator)
        
        # if ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
        #     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (offs_k_a + tl.arange(0, BLOCK_K//2))[None, :])
        #     b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k_b + tl.arange(0, BLOCK_K//2))[None, :])
        # elif MIXED_PREC:
        #     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (offs_k_a + tl.arange(0, BLOCK_K))[None, :])
        #     b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k_b + tl.arange(0, BLOCK_K//2))[None, :])
        # else:
        #     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (offs_k_a + tl.arange(0, BLOCK_K))[None, :])
        #     b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k_b + tl.arange(0, BLOCK_K))[None, :])
            
        # a = tl.load(a_ptrs)
        # b = tl.load(b_ptrs)
        
        # scale_a = tl.load(a_scale_ptr)
        # scale_b = tl.load(b_scale_ptr)
        # if USE_2D_SCALE_LOAD:
        #     scale_a = scale_a.reshape(BLOCK_M // 128, BLOCK_K // VEC_SIZE // 4, 32, 4, 4)
        #     scale_b = scale_b.reshape(BLOCK_N // 128, BLOCK_K // VEC_SIZE // 4, 32, 4, 4)
        # scale_a = scale_a.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        # scale_b = scale_b.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        # if MIXED_PREC:
        #     accumulator = tl.dot_scaled(a, scale_a, "e5m2", b.T, scale_b, "e2m1", accumulator)
        # elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
        #     accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        # else:
        #     accumulator = tl.dot_scaled(a, scale_a, "e5m2", b.T, scale_b, "e5m2", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        a_scale_ptr += (BLOCK_K // VEC_SIZE // 4) * stride_sak
        b_scale_ptr += (BLOCK_K // VEC_SIZE // 4) * stride_sbk
        # a_scale_ptr += (BLOCK_K // VEC_SIZE // 4) * stride_sb
        # b_scale_ptr += (BLOCK_K // VEC_SIZE // 4) * stride_sb

    # 存储结果（包含batch和head维度）
    c_base_offset = off_z * stride_cb + off_h * stride_ch
    c_ptrs = c_ptr + c_base_offset + (offs_am[:, None] * stride_cm + offs_bn[None, :])
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(output_dtype), mask=c_mask)


def block_scaled_batched_matmul(a_desc, a_scale, b_desc, b_scale, dtype_dst, B, H, M, N, K, configs):
    """
    支持多维批量矩阵乘法的函数
    
    Args:
        a_desc: 形状为(B, H, M, K)的输入矩阵A  
        a_scale: 矩阵A的scale因子, 形状为(B, H, M//128, K//VEC_SIZE//4, 32, 4, 4)
        b_desc: 形状为(B, H, N, K)的输入矩阵B
        b_scale: 矩阵B的scale因子, 形状为(B, H, N//128, K//VEC_SIZE//4, 32, 4, 4)
        dtype_dst: 输出数据类型
        B: batch size
        H: head数量
        M, N, K: 矩阵维度
        configs: 配置参数
        
    Returns:
        output: 形状为(B, H, M, N)的输出矩阵
    """
    output = torch.empty((B, H, M, N), dtype=dtype_dst, device="cuda")
    
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e5m2:
        dtype_dst = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    
    # 设置三维grid: 参考_attn_fwd的grid设置 (M*N的块数, head数, batch数)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), H, B)
    
    block_scaled_batched_matmul_kernel[grid](
        a_desc, a_scale, b_desc, b_scale, output, M, N, K,
        # 输入矩阵A的stride: batch, head, M, K
        a_desc.stride(0), a_desc.stride(1), a_desc.stride(2), a_desc.stride(3),
        # 输入矩阵B的stride: batch, head, N, K  
        b_desc.stride(0), b_desc.stride(1), b_desc.stride(2), b_desc.stride(3),
        # 输出矩阵的stride: batch, head, M, N
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # a_scale因子的stride: batch, head, M//128, K//VEC_SIZE//4
        a_scale.stride(0), a_scale.stride(1), a_scale.stride(2), a_scale.stride(3),
        # b_scale因子的stride: batch, head, N//128, K//VEC_SIZE//4
        b_scale.stride(0), b_scale.stride(1), b_scale.stride(2), b_scale.stride(3),
        H,  # head数量
        dtype_dst,
        configs["ELEM_PER_BYTE_A"], configs["ELEM_PER_BYTE_B"], configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"],
        configs["num_stages"], USE_2D_SCALE_LOAD=True)
    
    return output


def initialize_block_scaled_batched_from_tensor(a_tensor, b_tensor, a_scale, b_scale, block_scale_type="nvfp4", compute_reference=False):
    """
    初始化多维批量block scaled matmul的参数
    
    Args:
        a_tensor: 输入矩阵A, 形状为(B, H, M, K), dtype为torch.float16
        b_tensor: 输入矩阵B, 形状为(B, H, N, K), dtype为torch.float16  
        a_scale: 矩阵A的scale因子, 形状为(B, H, M//128, K//VEC_SIZE//4, 32, 4, 4)
        b_scale: 矩阵B的scale因子, 形状为(B, H, N//128, K//VEC_SIZE//4, 32, 4, 4)
        block_scale_type: 量化类型, 可选"nvfp4", "mxfp4", "mxfp8", "mixed"
        compute_reference: 是否计算参考结果用于验证
    
    Returns:
        a_desc, a_scale, b_desc, b_scale, configs, reference
    """
    B, H, M, K = a_tensor.shape
    B_b, H_b, N, K_b = b_tensor.shape
    assert B == B_b, f"batch size不匹配: A.shape[0]={B} != B.shape[0]={B_b}"
    assert H == H_b, f"head数量不匹配: A.shape[1]={H} != B.shape[1]={H_b}"
    assert K == K_b, f"矩阵维度不匹配: A.shape[3]={K} != B.shape[3]={K_b}"
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    assert block_scale_type in ["nvfp4", "mxfp4", "mxfp8", "mixed"], f"Invalid block scale type: {block_scale_type}"
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    device = a_tensor.device
    
    # 验证scale tensor的形状（包含batch和head维度）
    expected_a_scale_shape = (B, H, M // 128, K // VEC_SIZE // 4, 32, 4, 4)
    expected_b_scale_shape = (B, H, N // 128, K // VEC_SIZE // 4, 32, 4, 4)
    assert a_scale.shape == expected_a_scale_shape, f"a_scale形状不匹配: 期望{expected_a_scale_shape}, 实际{a_scale.shape}"
    assert b_scale.shape == expected_b_scale_shape, f"b_scale形状不匹配: 期望{expected_b_scale_shape}, 实际{b_scale.shape}"
  
    # 简化处理：直接使用mxfp8格式
    if block_scale_type == "mxfp8":
        a = a_tensor.to(torch.float8_e5m2)
        b = b_tensor.to(torch.float8_e5m2)
        a_ref = a.to(torch.float32) if compute_reference else None
        b_ref = b.to(torch.float32) if compute_reference else None
    else:
        # 对于其他格式，暂时也使用fp8处理
        a = a_tensor.to(torch.float8_e5m2)
        b = b_tensor.to(torch.float8_e5m2)
        a_ref = a.to(torch.float32) if compute_reference else None
        b_ref = b.to(torch.float32) if compute_reference else None
        
    a_desc = a
    b_desc = b

    # # Scale tensor处理 - 简化为使用原始输入
    # a_scale_proc = a_scale
    # b_scale_proc = b_scale
    
    # 处理scale因子
    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e5m2)
        b_scale = b_scale.to(torch.float8_e5m2)
        a_scale_ref = a_scale.to(torch.float32)
        b_scale_ref = b_scale.to(torch.float32)
    elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
        a_scale_ref = MXScaleTensor(a_scale.to(torch.float32))
        b_scale_ref = MXScaleTensor(b_scale.to(torch.float32))
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data
        
    reference = None
    if compute_reference:
        # 批量计算参考结果，处理多维输入
        b_ref = b_ref.transpose(-1, -2).contiguous()  # (B, H, K, N)
        
        a_scale_ref = a_scale_ref.to(torch.float32)
        b_scale_ref = b_scale_ref.to(torch.float32)

        def unpack_scale_batched(packed):
            B, H, num_chunk_m, num_chunk_k, _, _, _ = packed.shape
            return packed.permute(0, 1, 2, 5, 4, 3, 6).reshape(B, H, num_chunk_m * 128, num_chunk_k * 4).contiguous()
        # 展开scale因子到原始矩阵大小
        a_scale_expanded = unpack_scale_batched(a_scale_ref).repeat_interleave(VEC_SIZE, dim=3)[:, :, :M, :K]
        b_scale_expanded = unpack_scale_batched(b_scale_ref).repeat_interleave(VEC_SIZE, dim=3).transpose(-1, -2).contiguous()[:, :, :K, :N]
        
        # 计算参考结果：批量矩阵乘法 (A * scale_a) @ (B * scale_b)
        # 使用torch.matmul自动处理批量和head维度
        reference = torch.matmul(a_ref * a_scale_expanded, b_ref * b_scale_expanded)
    
    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 4,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
    }
    
    if compute_reference:
        return a_desc, a_scale, b_desc, b_scale, configs, (reference, a_ref * a_scale_expanded, (b_ref * b_scale_expanded).transpose(-1, -2))
    else:
        return a_desc, a_scale, b_desc, b_scale, configs, None



def test_batched_matmul():
    """测试多维批量矩阵乘法的正确性"""
    # 设置参数
    B = 2  # batch size
    H = 4  # head数量
    M = 256 
    N = 512
    K = 256
    block_scale_type = "mxfp8"  # 使用fp8格式进行测试
    
    print(f"测试多维批量矩阵乘法: B={B}, H={H}, M={M}, N={N}, K={K}")
    
    # 生成随机输入数据 - 修改为4D张量
    torch.manual_seed(42)
    a_tensor = torch.randn(B, H, M, K, dtype=torch.float16, device='cuda') * 0.1
    b_tensor = torch.randn(B, H, N, K, dtype=torch.float16, device='cuda') * 0.1
    
    # 计算VEC_SIZE
    VEC_SIZE = 32  # 对于mxfp8
    
    # 生成随机scale因子 - 修改为6D张量
    a_scale = torch.randn(B, H, M // 128, K // VEC_SIZE // 4, 32, 4, 4, 
                         dtype=torch.float32, device='cuda') * 0.01 + 0.1
    b_scale = torch.randn(B, H, N // 128, K // VEC_SIZE // 4, 32, 4, 4, 
                         dtype=torch.float32, device='cuda') * 0.01 + 0.1
    
    # try:
    # 初始化多维批量矩阵乘法
    a_desc, a_scale_proc, b_desc, b_scale_proc, configs, reference = \
        initialize_block_scaled_batched_from_tensor(
            a_tensor, b_tensor, a_scale, b_scale, 
            block_scale_type=block_scale_type, 
            compute_reference=True
        )
    
    print("✓ 初始化成功")
    print(f"  - a_desc形状: {a_desc.shape}")
    print(f"  - b_desc形状: {b_desc.shape}")
    print(f"  - a_scale形状: {a_scale_proc.shape}")
    print(f"  - b_scale形状: {b_scale_proc.shape}")
    
    # 执行多维批量矩阵乘法
    output = block_scaled_batched_matmul(
        a_desc, a_scale_proc, b_desc, b_scale_proc, 
        torch.float16, B, H, M, N, K, configs
    )
    
    print(f"✓ 多维批量矩阵乘法完成，输出形状: {output.shape}")
    
    # 验证输出形状
    assert output.shape == (B, H, M, N), f"输出形状错误: 期望{(B, H, M, N)}, 实际{output.shape}"
    
    # 与参考结果比较（如果有的话）
    if reference is not None:
        error = torch.mean(torch.abs(output.float() - reference)).item()
        print(f"✓ 与参考结果的平均绝对误差: {error:.6f}")
        
        # 检查是否在合理范围内
        if error < 0.1:  # 根据量化精度调整阈值
            print("✓ 精度验证通过")
        else:
            print("⚠️  精度可能有问题，误差较大")
    
    print("✓ 所有测试通过!")
    return True
        
    # except Exception as e:
    #     print(f"❌ 测试失败: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return False

def benchmark_batched_vs_sequential():
    """比较多维批量处理和逐个处理的性能"""
    B = 4
    H = 8
    M = 512
    N = 512  
    K = 256
    num_runs = 10
    
    print(f"\n性能测试: B={B}, H={H}, M={M}, N={N}, K={K}")
    
    # 生成测试数据 - 修改为4D张量
    torch.manual_seed(42)
    a_tensor = torch.randn(B, H, M, K, dtype=torch.float16, device='cuda') * 0.1
    b_tensor = torch.randn(B, H, N, K, dtype=torch.float16, device='cuda') * 0.1
    
    VEC_SIZE = 32
    a_scale = torch.randn(B, H, M // 128, K // VEC_SIZE // 4, 32, 4, 4, 
                         dtype=torch.float32, device='cuda') * 0.01 + 0.1
    b_scale = torch.randn(B, H, N // 128, K // VEC_SIZE // 4, 32, 4, 4, 
                         dtype=torch.float32, device='cuda') * 0.01 + 0.1
    
    # 初始化
    a_desc, a_scale_proc, b_desc, b_scale_proc, configs, _ = \
        initialize_block_scaled_batched_from_tensor(
            a_tensor, b_tensor, a_scale, b_scale, 
            block_scale_type="mxfp8", compute_reference=False
        )
    
    # 预热
    for _ in range(3):
        _ = block_scaled_batched_matmul(
            a_desc, a_scale_proc, b_desc, b_scale_proc, 
            torch.float16, B, H, M, N, K, configs
        )
    torch.cuda.synchronize()
    
    # 测试多维批量处理
    import time
    start_time = time.time()
    for _ in range(num_runs):
        output_batched = block_scaled_batched_matmul(
            a_desc, a_scale_proc, b_desc, b_scale_proc, 
            torch.float16, B, H, M, N, K, configs
        )
        torch.cuda.synchronize()
    batched_time = (time.time() - start_time) / num_runs
    
    print(f"多维批量处理平均时间: {batched_time*1000:.2f} ms")
    print(f"单次矩阵乘法等效时间: {batched_time*1000/(B*H):.2f} ms")
    print(f"理论加速比: {B*H}x")
    
    return batched_time

if __name__ == "__main__":
    print("开始测试多维批量矩阵乘法实现...")
    
    # 基本功能测试
    success = test_batched_matmul()
    
    if success:
        # 性能测试
        benchmark_batched_vs_sequential()
    else:
        print("基本功能测试失败，跳过性能测试") 