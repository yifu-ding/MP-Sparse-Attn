"""
Block Scaled Matrix Multiplication
==================================
This tutorial demonstrates a Triton implementation of block scaled matrix multiplication
which is generic over FP4 and FP8 formats. The formats supported in the tutorial are the OCP microscaling
formats, including mxfp4 and mxfp8, as well as NVIDIA's nvfp4 format. These matrix multiplications
are accelerated by fifth generation tensor core instructions on CUDA devices with compute capability 10.

Users can run the tutorial with each of the supported formats by passing the `--format`
argument and can benchmark the performance of each by specifying matrix dimensions
and iteration steps.

.. code-block:: bash

    # FP4
    python 10-block-scaled-matmul.py --format nvfp4
    python 10-block-scaled-matmul.py --format mxfp4 --K_range 512 8192 --bench

    # FP8
    python 10-block-scaled-matmul.py --format mxfp8 --K_range 8192 16384 --K_step 2048 --bench

Future updates to this tutorial which support mixed precision block scaled matmul are planned.
"""

# %%
# Background
# ----------
#
# CUDA devices that support PTX 8.7 and later can utlize block scaled matrix multiply
# instructions. In order for low latency access to these scale factors in the fast
# inner loop over tensor core MMAs, it is important to ensure that the blocked
# scale factors are stored in a contiguous memory layout according to their access
# pattern.
#
# The block scaled matmul tensor core instructions compute the following product:
#
#     C = (A * scale_a) @ (B * scale_b)
#
# where scale_a and scale_b are the blocked scale factors for the A and B matrices.
# Under block scaled matmul, each scale factor is broadcast and multiplied across a
# vector of elements from the A and B matrices, usually along their respective K axes.
# The number of elements of A and B over which each scale factor is broadcast is herein
# refered to as the vector size (VEC_SIZE).
#
# In a linear row-major layout, the scale factors would take the shape
#
#     (M, K // VEC_SIZE) and (N, K // VEC_SIZE)   [1]
#
# in global memory. However, to avoid non-contiguous memory access, it is beneficial to
# instead store the scale factors in a packed block layout. For the LHS matrix this layout
# is given by
#
#     (M // 32 // 4, K // VEC_SIZE // 4, 32, 4, 4)   [2].
#
# In this way, each tensor core MMA in the fast inner loop over K blocks can achieve contiguous
# access of a block of 128 rows of scale factors along the M axis, for each BLOCK_M x BLOCK_K
# subtile of the matrix A.
#
# In order to conform with Triton's language semantics for dot_scaled, the scale factors
# are prepared in the above 5D layout [2], but are then logically transposed and reshaped into
# the 2D layout [1] expected by tl.dot_scaled.
#
# For more detailed information on the scale factor layout, see
#  1. https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
#  2. https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
#


# from dataclasses import dataclass
# from typing import List, Any


# @dataclass
# class TensorDescriptor:
#     base: Any
#     shape: List[int]
#     strides: List[int]
#     block_shape: List[int]

#     def __post_init__(self):
#         rank = len(self.shape)
#         assert len(self.strides) == rank, f"rank mismatch: {self}"
#         assert len(self.block_shape) == rank, f"rank mismatch: {self}"

#     @property
#     def dtype(self):
#         """Return the dtype of the underlying tensor."""
#         return self.base.dtype

#     def data_ptr(self):
#         """Return the data pointer of the underlying tensor."""
#         return self.base.data_ptr()

#     @staticmethod
#     def from_tensor(tensor: Any, block_shape: List[int]):
#         return TensorDescriptor(
#             tensor,
#             tensor.shape,
#             tensor.stride(),
#             block_shape,
#         )

#     def load(self, offsets):
#         """Load method for compatibility with Triton's tl.load operations."""
#         # This is a placeholder - actual implementation depends on Triton's requirements
#         import triton.language as tl
#         return tl.load(self.base + offsets[0] * self.strides[0] + offsets[1] * self.strides[1])

#     def store(self, offsets, value):
#         """Store method for compatibility with Triton's tl.store operations."""
#         # This is a placeholder - actual implementation depends on Triton's requirements
#         import triton.language as tl
#         return tl.store(self.base + offsets[0] * self.strides[0] + offsets[1] * self.strides[1], value)


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


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(  #
        a_ptr, a_scale,  #
        b_ptr, b_scale,  #
        c_ptr,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
        stride_sk: tl.constexpr, stride_sb: tl.constexpr, stride_sc: tl.constexpr, stride_sd: tl.constexpr,
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  # 128 
        BLOCK_N: tl.constexpr,  # 256
        BLOCK_K: tl.constexpr,  # 128
        NUM_STAGES: tl.constexpr,  # 4
        USE_2D_SCALE_LOAD: tl.constexpr):  # False


    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    # offs_am = pid_m * BLOCK_M
    # offs_bn = pid_n * BLOCK_N
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 128
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # 256

    offs_k_a = 0
    offs_k_b = 0

    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e5


    ## block scale offsets
    offs_sm = (pid_m * (BLOCK_M // 128) + tl.arange(0, BLOCK_M // 128)) % M
    offs_sn = (pid_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)) % N

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    # For now it is recommended to use 2D scale loads for better performance.
    # In the future we will bring additional optimizations to either allow 5D loads,
    # the use of TMAs for scale factors, or both.
    if USE_2D_SCALE_LOAD:
        offs_inner = tl.arange(0, (BLOCK_K // VEC_SIZE // 4) * 32 * 4 * 4)
        a_scale_ptr = a_scale + offs_sm[:, None] * stride_sk + offs_inner[None, :]
        b_scale_ptr = b_scale + offs_sn[:, None] * stride_sk + offs_inner[None, :]
    else:
        offs_sk = tl.arange(0, (BLOCK_K // VEC_SIZE // 4))
        # MN spatial offsets for 32 element blocking
        offs_sc = tl.arange(0, 32)
        # offsets for both scale factor column ID (along K)
        # and spatial block column ID (along MN)
        offs_sd = tl.arange(0, 4)
        a_scale_ptr = a_scale + (offs_sm[:, None, None, None, None] * stride_sk + offs_sk[None, :, None, None, None] *
                                 stride_sb + offs_sc[None, None, :, None, None] * stride_sc +
                                 offs_sd[None, None, None, :, None] * stride_sd + offs_sd[None, None, None, None, :])
        b_scale_ptr = b_scale + (offs_sn[:, None, None, None, None] * stride_sk + offs_sk[None, :, None, None, None] *
                                 stride_sb + offs_sc[None, None, :, None, None] * stride_sc +
                                 offs_sd[None, None, None, :, None] * stride_sd + offs_sd[None, None, None, None, :])

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        # a = a_desc.load([offs_am, offs_k_a])
        # b = b_desc.load([offs_bn, offs_k_b])

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (offs_k_a + tl.arange(0, BLOCK_K//2))[None, :])
        b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k_b + tl.arange(0, BLOCK_K//2))[None, :])
        
        # 使用mask处理边界情况
        # a_mask = (offs_am[:, None] < M) & ((offs_k_a + tl.arange(0, BLOCK_K))[None, :] < K)
        # b_mask = (offs_bn[:, None] < N) & ((offs_k_b + tl.arange(0, BLOCK_K))[None, :] < K)
        
        # a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)
        if USE_2D_SCALE_LOAD:
            scale_a = scale_a.reshape(BLOCK_M // 128, BLOCK_K // VEC_SIZE // 4, 32, 4, 4)
            scale_b = scale_b.reshape(BLOCK_N // 128, BLOCK_K // VEC_SIZE // 4, 32, 4, 4)
        scale_a = scale_a.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        if MIXED_PREC:
            accumulator = tl.dot_scaled(a, scale_a, "e5m2", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e5m2", b.T, scale_b, "e5m2", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        a_scale_ptr += (BLOCK_K // VEC_SIZE // 4) * stride_sb
        b_scale_ptr += (BLOCK_K // VEC_SIZE // 4) * stride_sb

    # 存储结果
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :])
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(output_dtype), mask=c_mask)

    # c_ptr.store([offs_am, offs_bn], accumulator.to(output_dtype))
   
def block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, dtype_dst, M, N, K, configs):
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
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
    # c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])
    c_desc = output
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](a_desc, a_scale, b_desc, b_scale, c_desc, M, N, K, 
                                     a_desc.stride(0), a_desc.stride(1),
                                     b_desc.stride(0), b_desc.stride(1),
                                     c_desc.stride(0), c_desc.stride(1),
                                     a_scale.stride(0), a_scale.stride(1), a_scale.stride(2), a_scale.stride(3), dtype_dst,
                                     configs["ELEM_PER_BYTE_A"], configs["ELEM_PER_BYTE_B"], configs["VEC_SIZE"],
                                     configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"],
                                     configs["num_stages"], USE_2D_SCALE_LOAD=True)
    return output


def initialize_block_scaled_from_tensor(a_tensor, b_tensor, a_scale, b_scale, block_scale_type="nvfp4", compute_reference=False):
    """
    初始化block scaled matmul的参数
    
    Args:
        a_tensor: 输入矩阵A, 形状为(M, K), dtype为torch.float16
        b_tensor: 输入矩阵B, 形状为(K, N), dtype为torch.float16  
        a_scale: 矩阵A的scale因子, 形状为(M//128, K//VEC_SIZE//4, 32, 4, 4)
        b_scale: 矩阵B的scale因子, 形状为(N//128, K//VEC_SIZE//4, 32, 4, 4)
        block_scale_type: 量化类型, 可选"nvfp4", "mxfp4", "mxfp8", "mixed"
        compute_reference: 是否计算参考结果用于验证
    
    Returns:
        a_desc, a_scale, b_desc, b_scale, configs, reference
    """
    M, K = a_tensor.shape
    N, K_b = b_tensor.shape
    assert K == K_b, f"矩阵维度不匹配: A.shape[1]={K} != B.shape[0]={K_b}"
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    assert block_scale_type in ["nvfp4", "mxfp4", "mxfp8", "mixed"], f"Invalid block scale type: {block_scale_type}"
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    device = a_tensor.device
    
    # 验证scale tensor的形状
    expected_a_scale_shape = (M // 128, K // VEC_SIZE // 4, 32, 4, 4)
    expected_b_scale_shape = (N // 128, K // VEC_SIZE // 4, 32, 4, 4)
    assert a_scale.shape == expected_a_scale_shape, f"a_scale形状不匹配: 期望{expected_a_scale_shape}, 实际{a_scale.shape}"
    assert b_scale.shape == expected_b_scale_shape, f"b_scale形状不匹配: 期望{expected_b_scale_shape}, 实际{b_scale.shape}"
  
    # 根据block_scale_type处理输入数据
    if block_scale_type in ["mxfp8", "mixed"]:
        a = a_tensor.to(torch.float8_e5m2)
        a_ref = a.to(torch.float32)
    else:
        # 对于fp4格式, 这里需要将fp16数据转换为packed fp4格式
        # 简化处理：直接使用原tensor, 实际应用中需要进行fp4 packing
        # a = a_tensor
        a = MXFP4Tensor(data=a_tensor.to(torch.float32))
        a_ref = a.to(torch.float32)
        a = a.to_packed_tensor(dim=1)
    a_desc = a

    if block_scale_type == "mxfp8": 
        b = b_tensor.to(torch.float8_e5m2)
        b_ref = b.to(torch.float32)
    else:
        # 对于fp4格式, 这里需要将fp16数据转换为packed fp4格式
        # 简化处理：直接使用原tensor, 实际应用中需要进行fp4 packing
        # b = b_tensor
        b = MXFP4Tensor(data=b_tensor.to(torch.float32))
        b_ref = b.to(torch.float32)
        b = b.to_packed_tensor(dim=1)
    b_desc = b

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
        
        b_ref = b_ref.T.contiguous()
        
        a_scale_ref = a_scale_ref.to(torch.float32)
        b_scale_ref = b_scale_ref.to(torch.float32)

        def unpack_scale(packed):
            num_chunk_m, num_chunk_k, _, _, _ = packed.shape
            return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()
        
        # 展开scale因子到原始矩阵大小
        a_scale_expanded = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
        b_scale_expanded = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
        
        # 计算参考结果：(A * scale_a) @ (B * scale_b)
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
    
    return a_desc, a_scale, b_desc, b_scale, configs, reference


def validate_block_scaled_from_tensor(M, N, K, block_scale_type="nvfp4"):
    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    if block_scale_type == "mixed":
        # This is needed for TMA with the descriptor created on the device.
        # TMA load for mixed-precision fp4 is supported only by device TMA.
        triton.set_allocator(alloc_fn)
        
    # 创建测试用的tensor
    device = "cuda"
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    
    # 创建输入矩阵 (fp16)
    a_tensor = torch.randn((M, K), dtype=torch.float16, device=device)
    b_tensor = torch.randn((N, K), dtype=torch.float16, device=device)
    
    # 创建scale因子 (fp32), 添加小的epsilon避免零值
    epsilon = 1e-8
    a_scale_tensor = torch.rand((M // 128, K // VEC_SIZE // 4, 32, 4, 4), device=device) + epsilon  # dtype=float32
    b_scale_tensor = torch.rand((N // 128, K // VEC_SIZE // 4, 32, 4, 4), device=device) + epsilon

    a_desc, a_scale, b_desc, b_scale, configs, reference = initialize_block_scaled_from_tensor(
        a_tensor, b_tensor, a_scale_tensor, b_scale_tensor, block_scale_type, compute_reference=True
    )
    
    output = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float16, M, N, K, configs)
    # print("Reference shape:", reference.shape)
    # print("Output shape:", output.shape)
    print(reference)
    print(output)
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)
    print(f"✅ (pass {block_scale_type})")


def bench_block_scaled_from_tensor(M, N, K, block_scale_type="nvfp4", reps=10):
    assert K % 128 == 0
    print(f"Problem Shape = {M}x{N}x{K}")

    # 创建测试用的tensor
    device = "cuda"
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    
    # 创建输入矩阵 (fp16)
    a_tensor = torch.randn((M, K), dtype=torch.float16, device=device)
    b_tensor = torch.randn((N, K), dtype=torch.float16, device=device)
    
    # 创建scale因子 (fp16), 添加小的epsilon避免零值
    epsilon = 1e-8
    a_scale_tensor = torch.rand((M // 128, K // VEC_SIZE // 4, 32, 4, 4), device=device) + epsilon  # dtype=float32
    b_scale_tensor = torch.rand((N // 128, K // VEC_SIZE // 4, 32, 4, 4), device=device) + epsilon

    a_desc, a_scale, b_desc, b_scale, configs, _ = initialize_block_scaled_from_tensor(
        a_tensor, b_tensor, a_scale_tensor, b_scale_tensor, block_scale_type, compute_reference=False
    )
    
    _ = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float16, M, N, K, configs)

    proton.activate(0)
    for _ in range(reps):
        _ = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float16, M, N, K, configs)
    proton.deactivate(0)
    print("Done benchmarking")
    

def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer
    metric_names = ["time/ms"]
    metric_names = ["tflop/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--bench", action="store_true", default=False)
    parser.add_argument("--format", type=str, choices=["mxfp4", "nvfp4", "mxfp8", "mixed"], default="nvfp4")
    args = parser.parse_args()

    # if not supports_block_scaling():
    if False:
        print("⛔ This example requires GPU support for block scaled matmul")
    else:
        if args.K and args.K_range is None:
            args.K_range = [args.K, args.K]
            args.K_step = 1  # doesn't matter as long as it's not 0

        torch.manual_seed(42)

        validate_block_scaled_from_tensor(128, 256, 256, block_scale_type=args.format)

        if args.bench:
            proton.start("block_scaled_matmul", hook="triton")
            proton.deactivate(0)  # Skip argument creation
            for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
                bench_block_scaled_from_tensor(M=8192, N=8192, K=K, reps=10000, block_scale_type=args.format)
            proton.finalize()
            show_profile("block_scaled_matmul")
