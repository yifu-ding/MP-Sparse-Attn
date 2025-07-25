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
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
# from spas_sage_attn.utils import precision_metric
# from spas_sage_attn import spas_sage_attn_meansim_cuda, spas_sage2_attn_meansim_cuda
# from spas_sage_attn.triton_kernel_example import spas_sage_attn_meansim
import warnings
# from einops import rearrange
from ours.online_routing import online_routing_attn
from ours.mxfp_attn_kernel import mxfp_attn_kernel
from scripts.debug3 import mxfp_attn_debug

# def extract_sparse_attention_state_dict(model, verbose=False):
#     saved_state_dict = {}
#     for k, v in model.named_modules(): # enumerate all nn.Module instance in the model
#         if isinstance(v, SparseAttentionMeansim):
#             if verbose: print(k, 'is an instance of SparseAttentionMeansim')
#             for model_key, model_param in model.state_dict().items(): # find the corresponding state_dict item
#                 if k in model_key:
#                     if verbose: print(f'{model_key} is a substate_dict of {k}, we will save it.')
#                     saved_state_dict[model_key] = model_param
#     return saved_state_dict


# def load_sparse_attention_state_dict(model, saved_state_dict, multigpu=False, verbose=False):
#     if not multigpu:
#         device = next(model.parameters()).device
#         dtype = next(model.parameters()).dtype
#     for k, v in model.named_modules():
#         if isinstance(v, SparseAttentionMeansim): # find each SparseAttentionMeansim instance
#             if verbose: print(k, 'is an instance of SparseAttentionMeansim, but it is empty now.')
#             for sk, sv in saved_state_dict.items():
#                 if k in sk:
#                     if verbose: print(f'{sk} is a substate_dict of {k}, we will load it.')
#                     sub_name = sk.split(k)[1][1:]
#                     if multigpu:
#                         sv= sv.to(device=v.device)
#                     else:
#                         sv = sv.to(device=device, dtype=dtype)
#                     setattr(v, sub_name, nn.Parameter(sv, requires_grad=False))
#     if not multigpu:
#         model = model.to(device)
#     return model


# def partition_points_into_line(points, block_size, min_dim1=-1, max_dim1=1):
#     blocks = {}
#     for point in points:
#         dim1 = point['simthreshd1']
#         # Calculate block indices for dim1 and dim2
#         block_index_dim1 = int((dim1 - min_dim1) // block_size)
#         key = (block_index_dim1,)
#         # Initialize the block if it doesn't exist
#         if key not in blocks:
#             blocks[key] = []
#         blocks[key].append(point)
#     return blocks

# 
from tools.gpu_process import GPUProcessPoolExecutor
executor = GPUProcessPoolExecutor()

class MXFPAttention(nn.Module):
    def __init__(self, rearrange_kwargs={}, 
                 layer_idx=-1, verbose=False, kernel_name=None, mxfp_bw=None, smooth_k=False, \
                    dual_scale=False, pre_quant=False, fuse_mp_quant=False, fp8_tile_num=1):
        super(MXFPAttention, self).__init__()
        self.layer_idx = layer_idx
        self.head_num = None
        # assert l1 >= 0 and cos_sim <= 1 and rmse >= 0, "l1, cos_sim, rmse should be legal"
        # assert pv_l1 > l1, 'pv l1 must greater than l1'
        # self.l1 = l1
        # self.pv_l1 = pv_l1
        # self.cos_sim = cos_sim
        # self.rmse = rmse
        # self.is_sparse = None  # bool, shape of head number, decide whether to use sparse attention for each head
        # self.cdfthreshd = None  # float, shape of head number, decide the threshold of cdf for each head
        # self.simthreshd1 = None
        # self.simthreshd2 = None
        # self.pvthreshd = None
        # self.tuning_sparsity = None
        # self.num_data_passed = 0
        # self.hyperparams_cache = {}
        # self.sim_rule = sim_rule
        # self.rearrange_kwargs = rearrange_kwargs
        # self.tune_pv = tune_pv
        self.verbose = verbose
        self.kernel_name = kernel_name
        self.mxfp_bw = mxfp_bw
        self.smooth_k = smooth_k
        self.dual_scale = dual_scale
        self.pre_quant = pre_quant
        self.fuse_mp_quant = fuse_mp_quant
        self.fp8_tile_num = fp8_tile_num
    # def is_sim(self, o_gt, o_sparse):
    #     if self.sim_rule == "cosine":
    #         return precision_metric(o_sparse, o_gt, verbose=False)["Cossim"] > self.cos_sim
    #     elif self.sim_rule == "rmse":
    #         return precision_metric(o_sparse, o_gt, verbose=False)["RMSE"] < self.rmse
    #     elif self.sim_rule == "l1":
    #         return precision_metric(o_sparse, o_gt, verbose=False)["L1"] < self.l1
    #     else:
    #         raise ValueError("sim_rule should be one of ['cosine', 'rmse', 'l1']")
    
    # def init_hyperparams(self, head_num, device):
    #     self.head_num = head_num
    #     self.is_sparse = nn.Parameter(
    #         torch.ones(self.head_num, dtype=torch.bool, device=device),
    #         requires_grad=False,
    #     )
    #     self.cdfthreshd = nn.Parameter(
    #         torch.ones(self.head_num, device=device) * 0.1,
    #         requires_grad=False,
    #     )
    #     self.simthreshd1 = nn.Parameter(
    #         torch.ones(self.head_num, device=device) * -1,
    #         requires_grad=False,
    #     )
    #     self.simthreshd2 = nn.Parameter(
    #         torch.zeros(self.head_num, device=device),
    #         requires_grad=False,
    #     )
    #     self.pvthreshd = nn.Parameter(
    #         torch.ones(self.head_num, device=device) * 20,
    #         requires_grad=False,
    #     )
    #     self.tuning_sparsity = torch.zeros(self.head_num, device=device)
    #     self.num_data_passed = 0
    #     self.hyperparams_cache = {}

    def kernel_selection(self, kernel_name=None):
        if kernel_name == "online_routing":
            return online_routing_attn
        elif kernel_name == "mxfp_attn":
            return mxfp_attn_kernel
        elif kernel_name == "mxfp_attn_debug":
            return mxfp_attn_debug
        elif kernel_name == "native":
            return torch.nn.functional.scaled_dot_product_attention
        else:
            raise ValueError(f"not support kernel name: {kernel_name}")

    # @torch.no_grad()
    # def autotune(self, qi, ki, vi, head_idx, mask=None, is_causal=False, smooth_k=True):
    #     qi = qi.to(torch.cuda.current_device())
    #     ki = ki.to(torch.cuda.current_device())
    #     vi = vi.to(torch.cuda.current_device())
    #     all_hyperparams = []
    #     granularity = 16
    #     for simthreshd1 in range(int(-1 * granularity), int(1 * granularity)):
    #         simthreshd1 = simthreshd1 / granularity
    #         cur_cdfthreshd, sparsity = self.tune_cdfthreshd(
    #             qi,
    #             ki,
    #             vi,
    #             mask,
    #             is_causal=is_causal,
    #             smooth_k=smooth_k,
    #             simthreshd1=simthreshd1,
    #         )
    #         if self.tune_pv:
    #             pvthreshd, _ = self.tune_pvthreshd(
    #                 qi,
    #                 ki,
    #                 vi,
    #                 mask,
    #                 is_causal=is_causal,
    #                 smooth_k=smooth_k,
    #                 simthreshd1=simthreshd1,
    #                 cdfthreshd=cur_cdfthreshd,
    #             )
    #         else:
    #             pvthreshd = 20
    #         all_hyperparams.append({
    #             "simthreshd1": simthreshd1,
    #             "cdfthreshd": cur_cdfthreshd,
    #             'pvthreshd': pvthreshd,
    #             "sparsity": sparsity,
    #             'data_idx': self.num_data_passed
    #         })
    #         if sparsity < 0.1:
    #             break  # no need to continue to raise threshold bound
    #     if self.hyperparams_cache.get(head_idx) is None:
    #         self.hyperparams_cache[head_idx] = []
    #     cache_hyper = self.hyperparams_cache[head_idx]
    #     all_hyperparams = all_hyperparams + cache_hyper
    #     self.hyperparams_cache[head_idx] = all_hyperparams
        
    #     grid = partition_points_into_line(all_hyperparams, 2/granularity)
    #     groups = list(grid.values())
    #     # sort by sum of sparsity, local smoothing
    #     groups = sorted(groups, key=lambda x: sum([y['sparsity'] for y in x]), reverse=True)
    #     final_group = groups[0]
    #     final_simthreshd1 = np.max([x['simthreshd1'] for x in final_group]).item()
    #     final_cdfthreshd = np.max([x['cdfthreshd'] for x in final_group]).item()
    #     final_pvthreshd = np.max([x['pvthreshd'] for x in final_group]).item()
    #     mean_sparsity = np.mean([x['sparsity'] for x in final_group]).item()
    #     return {
    #         'final_simthreshd1': final_simthreshd1,
    #         'final_cdfthreshd': final_cdfthreshd,
    #         'final_pvthreshd': final_pvthreshd,
    #         'mean_sparsity': mean_sparsity,
    #         'head_idx': head_idx
    #     }
        
    # def fill_results(self, rtdict):
    #     head_idx = rtdict['head_idx']
    #     self.simthreshd1[head_idx] = rtdict['final_simthreshd1']
    #     self.cdfthreshd[head_idx] = rtdict['final_cdfthreshd']
    #     self.pvthreshd[head_idx] = rtdict['final_pvthreshd']
    #     self.is_sparse[head_idx] = rtdict['mean_sparsity'] > 0.1 and self.is_sparse[head_idx]
    #     self.tuning_sparsity[head_idx] = rtdict['mean_sparsity']
    #     if not self.is_sparse[head_idx]:
    #         self.cdfthreshd[head_idx] = 1
    #         self.simthreshd1[head_idx] = 1
        
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
            
        # if os.environ.get("TUNE_MODE", "") == "1" or tune_mode:
        #     if tensor_layout == 'NHD':
        #         q = rearrange(q, '... L H D -> ... H L D')
        #         k = rearrange(k, '... L H D -> ... H L D')
        #         v = rearrange(v, '... L H D -> ... H L D')
        #     if self.is_sparse is None:  # init per head hyper parameters
        #         self.init_hyperparams(q.shape[1], q.device)
        #     if os.environ.get('PARALLEL_TUNE', '') == '':
        #         for i in tqdm(range(self.head_num), desc="Heads#L"+str(self.layer_idx)):
        #             if not self.is_sparse[i].item():
        #                 continue
        #             qi, ki, vi = q[:, i : i + 1], k[:, i : i + 1], v[:, i : i + 1]
        #             rtdict = self.autotune(qi, ki, vi, head_idx=i, mask=mask, is_causal=is_causal, smooth_k=smooth_k)
        #             self.fill_results(rtdict)
        #     else:
        #         futures = []
        #         for i in range(self.head_num):
        #             if not self.is_sparse[i].item():
        #                 continue
        #             qi, ki, vi = q[:, i : i + 1], k[:, i : i + 1], v[:, i : i + 1]
        #             future = executor.submit(self.autotune, qi, ki, vi, head_idx=i, mask=mask, is_causal=is_causal, smooth_k=smooth_k)
        #             futures.append(future)
        #         for future in tqdm(futures):
        #             rtdict = future.result()
        #             self.fill_results(rtdict)
                    
                
        #     self.num_data_passed += 1
        #     if self.verbose:
        #         print(f'{self.cdfthreshd=}')
        #         print(f'{self.simthreshd1=}')
        #         print(f'{self.is_sparse=}')
        #         print(f'{self.pvthreshd=}')
        #         print(f'{self.tuning_sparsity=}')
        #         print(f'mean sparsity:{self.tuning_sparsity.mean().item()}, layer_idx:{self.layer_idx}')
        #     o = F.scaled_dot_product_attention(q, k, v, mask, is_causal=is_causal)
        #     if tensor_layout == 'NHD':
        #         o = rearrange(o, '... H L D -> ... L H D')
        #     torch.cuda.empty_cache()
        # else:
            # assert self.cdfthreshd is not None, "attention hyperparameters should be tuned first"

        kernel = self.kernel_selection(self.kernel_name)
        if self.kernel_name == "online_routing":
            o = kernel(
                q,
                k,
                v,
                mask,
                is_causal=is_causal,
                skip_thresh=self.skip_thresh
            )
        elif self.kernel_name == "mxfp_attn":
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
                fp8_tile_num=self.fp8_tile_num,
                # skip_thresh=self.skip_thresh,
            )
        elif self.kernel_name == "mxfp_attn_debug":
            o = kernel(
                q,
                k,
                v,
                is_causal=is_causal,
                output_dtype=output_dtype,
                block_scale_type=self.mxfp_bw,
                smooth_k=self.smooth_k,
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
