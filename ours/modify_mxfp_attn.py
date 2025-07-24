import torch, pickle
from typing import List, Optional, Tuple, Union
from types import MethodType
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
from transformers.cache_utils import Cache
import torch.nn.functional as F
# from spas_sage_attn import spas_sage_attn
# from spas_sage_attn.utils import precision_metric
# from spas_sage_attn.autotune import SparseAttentionMeansim
# , SparseAttention, extract_sparse_attention_state_dict, load_sparse_attention_state_dict
# from functools import partial
import types
from ours.kernel_selection import MXFPAttention
from scripts.debug3 import mxfp_attn_debug

def set_mxfp_attn_llama(model, verbose=False, kernel_name=None, mxfp_bw=None, smooth_k=False, \
    dual_scale=False, pre_quant=False, fuse_mp_quant=False, fp8_tile_num=1):
    for layer_id, layer in enumerate(model.model.layers):

        setattr(layer.self_attn, 'mxfp_attention', MXFPAttention(layer_idx=layer_id, 
                                                                    verbose=verbose,
                                                                    kernel_name=kernel_name,
                                                                    mxfp_bw=mxfp_bw, 
                                                                    smooth_k=smooth_k,
                                                                    dual_scale=dual_scale,
                                                                    pre_quant=pre_quant,
                                                                    fuse_mp_quant=fuse_mp_quant,
                                                                    fp8_tile_num=fp8_tile_num))
        # layer.self_attn.sparse_attention.device = next(layer.self_attn.parameters()).device

        old_forward = layer.self_attn.forward

        def new_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
                
            assert not output_attentions, "Output attentions not supported"
            assert attention_mask is None, "Attention mask not supported"

            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            if position_embeddings is None:
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            # Move tensors to same device as query_states
            cos = cos.to(query_states.device)
            sin = sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            if q_len == 1:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                causal_mask = attention_mask
                if attention_mask is not None:
                    causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

                # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
                # Reference: https://github.com/pytorch/pytorch/issues/112577.
                if query_states.device.type == "cuda" and causal_mask is not None:
                    query_states = query_states.contiguous()
                    key_states = key_states.contiguous()
                    value_states = value_states.contiguous()

                # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
                # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
                is_causal = True if causal_mask is None and q_len > 1 else False

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=causal_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(bsz, q_len, -1)
                attn_output = self.o_proj(attn_output)
            else:
                key_states = key_states.repeat_interleave(query_states.size(-3)//key_states.size(-3), -3)
                value_states = value_states.repeat_interleave(query_states.size(-3)//value_states.size(-3), -3)
                attn_out_dtype = self.o_proj.weight.dtype
                attn_output = self.mxfp_attention(query_states, key_states, value_states, is_causal=True, output_dtype=attn_out_dtype)
                # attn_output = test_add_resisual(query_states, key_states, value_states, is_causal=True, output_dtype=attn_out_dtype)

                if verbose:
                    o = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)
                    sim_dict = precision_metric(attn_output, o)
                    if sim_dict['Cossim'] < 0.5:
                        # import pdb; pdb.set_trace()
                        # torch.save({
                        #     'query_states': query_states.detach().cpu(),
                        #     'key_states': key_states.detach().cpu(), 
                        #     'value_states': value_states.detach().cpu(),
                        #     'attn_output': attn_output.detach().cpu(),
                        #     'o': o.detach().cpu()
                        # }, 'saved_files/low_sim_attn_states.pth')
                        # print(f"save low sim attn states to saved_files/low_sim_attn_states.pth")
                        exit()

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(bsz, q_len, -1)
                attn_output = self.o_proj(attn_output)

            # return attn_output, None, past_key_value   # for transformers==4.46.3
            return attn_output, None   # for transformers==4.52.3

        layer.self_attn.forward = types.MethodType(new_forward, layer.self_attn)
    

def precision_metric(quant_o, fa2_o, verbose=True, round_num=4): 
    if quant_o.shape[-2] > 200000:
        quant_o, fa2_o = quant_o.cpu(), fa2_o.cpu()
    x, xx = quant_o.float(), fa2_o.float() 
    sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    l1 =   ( (x - xx).abs().sum() / xx.abs().sum() ).item()
    rmse = torch.sqrt(torch.mean((x -xx) ** 2)).item()
    sim = round(sim, round_num)
    l1 = round(l1, round_num)
    rmse = round(rmse, round_num)
    if verbose: print(f'Cossim: {sim:.6f}, L1: {l1:.6f}, RMSE:{rmse:.6f}')
    return {"Cossim": sim, "L1": l1, "RMSE": rmse}

