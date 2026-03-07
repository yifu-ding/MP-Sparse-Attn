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

import torch
import torch.nn as nn
from ours.mxfp_attn_func import mxfp_attn_kernel
# 
from tools.gpu_process import GPUProcessPoolExecutor
executor = GPUProcessPoolExecutor()

class MXFPAttention(nn.Module):
    def __init__(self, rearrange_kwargs={}, 
                 layer_idx=-1, verbose=False, kernel_name=None, mxfp_bw=None, smooth_k=False, \
                    dual_scale=False, pre_quant=False, fuse_mp_quant=False, diag_tile=1, sink_tile=1, \
                        quant_granularity=None, qk_dtype=None):
        super(MXFPAttention, self).__init__()
        self.layer_idx = layer_idx
        self.head_num = None

        self.verbose = verbose
        self.kernel_name = kernel_name
        self.mxfp_bw = mxfp_bw
        self.smooth_k = smooth_k
        self.dual_scale = dual_scale
        self.pre_quant = pre_quant
        self.fuse_mp_quant = fuse_mp_quant
        self.diag_tile = diag_tile
        self.sink_tile = sink_tile
        self.quant_granularity = quant_granularity
        self.qk_dtype = qk_dtype
  
    def kernel_selection(self, kernel_name=None):
        if kernel_name == "mxfp_attn":
            return mxfp_attn_kernel
        elif kernel_name == "native":
            return torch.nn.functional.scaled_dot_product_attention
        else:
            raise ValueError(f"not support kernel name: {kernel_name}")

    @torch.no_grad()
    def forward(
        self,
        q,
        k,
        v,
        mask=None,
        is_causal=False,
        scale=None,
        tensor_layout="HND",
        tune_mode=False,
        smooth_k=True,
        return_sparsity=False,
        output_dtype=torch.float16,
    ):
        assert len(q.shape) == 4, "q should be 4-d tensor with B, H, L, D"
      
        kernel = self.kernel_selection(self.kernel_name)
        if self.kernel_name == "mxfp_attn":
            # def mxfp_attn_kernel(q, k, v, attn_mask=None, dropout_p=0.0, 
            #     is_causal=False, scale=None, smooth_k=False, attention_sink=False, tensor_layout="HND",
            #     output_dtype=torch.float16, return_sparsity=False, block_scale_type="mxfp4", skip_thresh=None):
            o = kernel(
                q,
                k,
                v,
                mask,
                is_causal=is_causal,
                scale=scale,
                tensor_layout=tensor_layout,
                attention_sink=False,
                block_scale_type=self.mxfp_bw,
                output_dtype=output_dtype,
                smooth_k=self.smooth_k,
                dual_scale=self.dual_scale,
                pre_quant=self.pre_quant,
                fuse_mp_quant=self.fuse_mp_quant,
                diag_tile=self.diag_tile,
                sink_tile=self.sink_tile,
                quant_granularity=self.quant_granularity,
                qk_dtype=self.qk_dtype,
                # skip_thresh=self.skip_thresh,
            )
        elif self.kernel_name == "native":
            o = kernel(
                q,
                k,
                v,
                mask,
                is_causal=is_causal,
                scale=scale,
            )
        else:
            raise ValueError(f"not support kernel name: {self.kernel_name}")
        
        o = o.to(output_dtype)
        if return_sparsity:
            o, total_sparsity = o
            return o, total_sparsity
        else:
            return o
