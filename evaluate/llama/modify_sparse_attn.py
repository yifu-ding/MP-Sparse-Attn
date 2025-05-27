import torch, pickle
from typing import List, Optional, Tuple, Union
from types import MethodType
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
from transformers.cache_utils import Cache
import torch.nn.functional as F
# from spas_sage_attn import spas_sage_attn
from spas_sage_attn.utils import precision_metric
from spas_sage_attn.autotune import SparseAttentionMeansim
# , SparseAttention, extract_sparse_attention_state_dict, load_sparse_attention_state_dict
# from functools import partial
import types


def set_spas_sage_attn_llama(model, l1=0.06, pv_l1=0.065, verbose=False):
    for layer_id, layer in enumerate(model.model.layers):

        setattr(layer.self_attn, 'sparse_attention', SparseAttentionMeansim(l1=l1, 
                                                                            pv_l1=pv_l1, 
                                                                            layer_idx=layer_id, 
                                                                            verbose=verbose,
                                                                            kernel_name="online_routing"))
                                                                            # kernel_name=None))
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
                # import pdb; pdb.set_trace()  # no?
            else:
                key_states = key_states.repeat_interleave(query_states.size(-3)//key_states.size(-3), -3)
                value_states = value_states.repeat_interleave(query_states.size(-3)//value_states.size(-3), -3)
                
                # # 创建保存目录
                # import os
                # save_dir = "./results/saved_qkv"
                # os.makedirs(save_dir, exist_ok=True)

                # # 生成文件名
                # save_name = f"qkv_bsz{bsz}_qlen{q_len}_layer{self.layer_idx}.pt"
                # save_path = os.path.join(save_dir, save_name)

                # # 保存 query_states, key_states, value_states
                # torch.save({
                #     'query_states': query_states,
                #     'key_states': key_states, 
                #     'value_states': value_states
                # }, save_path)

                # if self.layer_idx > 3:
                #     exit()
                
                # attn_output = spas_sage_attn(query_states, key_states, value_states, is_causal=True, )  # need to add other parameters
                attn_output = self.sparse_attention(query_states, key_states, value_states, is_causal=True, )
                if verbose:
                    o = F.scaled_dot_product_attention(query_states, key_states, value_states,  is_causal=True)
                    precision_metric(attn_output, o)

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(bsz, q_len, -1)
                attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value   # for transformers==4.46.3
            # return attn_output, None   # for transformers==4.52.3

        layer.self_attn.forward = types.MethodType(new_forward, layer.self_attn)
        
     
