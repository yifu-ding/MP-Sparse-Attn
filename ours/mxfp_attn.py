"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch, math
import triton
import triton.language as tl
import torch.nn.functional as F
# from spas_sage_attn.utils import hyperparameter_check, get_block_map_meansim
# from spas_sage_attn.quant_per_block import per_block_int8
import pdb
from spas_sage_attn.quant_mxint8 import quant_fpxint8, quant_mxfp8e5, quant_mxfp4

from spas_sage_attn.block_scaled_matmul import initialize_block_scaled_from_tensor, block_scaled_matmul
from spas_sage_attn.batched_block_scaled_matmul import initialize_block_scaled_batched_from_tensor, block_scaled_batched_matmul


@torch.compiler.disable
def mxfp_attn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, \
    attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False, skip_thresh=None, \
    block_scale_type="mxfp4", BLKQ=128, compute_reference=False):
    assert q.size(-2)>=128, "seq_len should be not less than 128."

    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        k = k - k.mean(dim=-2, keepdim=True)
    # headdim = q.size(-1)
    batch_size, num_heads, seq_len, head_dim = q.shape

    assert head_dim in [64, 128], "headdim should be in [64, 96, 128]."

    # Quantize to float4 (MX format)
    if block_scale_type == "mxfp4":
        q_fp4, q_scale, k_fp4, k_scale = quant_mxfp4(q, k, BLKQ=BLKQ)
        q_quant = q_fp4
        k_quant = k_fp4
    else:
        q_fp8, q_scale, k_fp8, k_scale = quant_mxfp8e5(q, k, BLKQ=BLKQ)
        q_quant = q_fp8
        k_quant = k_fp8
        
    q_quant = q_quant.reshape(batch_size, num_heads, seq_len, head_dim)
    k_quant = k_quant.reshape(batch_size, num_heads, seq_len, head_dim)
    q_scale = q_scale.reshape(batch_size, num_heads, seq_len//128, 4, 32, head_dim//32//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    k_scale = k_scale.reshape(batch_size, num_heads, seq_len//128, 4, 32, head_dim//32//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    
    q_packed, q_scale, k_packed, k_scale, configs, (reference, q_dequant, k_dequant) = \
        initialize_block_scaled_batched_from_tensor(q_quant, k_quant, q_scale, k_scale, block_scale_type=block_scale_type, compute_reference=compute_reference)
    
    output = block_scaled_batched_matmul(q_packed, q_scale, k_packed, k_scale, output_dtype, batch_size, num_heads, seq_len, seq_len, head_dim, configs)
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)
    print(f"✅ (pass {block_scale_type} block scaled)")
    
    # q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)  # 量化
    # # pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    # # k_block_indices[:] = 1
    # o = forward(q_int8, k_int8, v, q_scale, k_scale,  is_causal=is_causal, tensor_layout="HND", output_dtype=dtype, skip_thresh=skip_thresh)

    return output
