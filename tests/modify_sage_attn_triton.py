import torch, pickle
from typing import List, Optional, Tuple, Union
from types import MethodType
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
from transformers.cache_utils import Cache
import types
import torch.nn as nn
from tests.flash_attn_triton import flash_attn_func
# from flash_attn.flash_attn_triton import _flash_attn_forward
from tests.sageattn_core import sageattn, sageattn_qk_int8_pv_fp16_triton
        
def LlamaSageAttnForward(
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
    # global forward_time
    # st_forward = time.perf_counter()
    assert not output_attentions, "Output attentions not supported"
    assert attention_mask is None, "Attention mask not supported"
    # assert self.num_key_value_groups == 1, "GQA will be supported in near future"
    # [batch_size, len, 4096]
    bsz, q_len, _ = hidden_states.size()

    # print(f"hidden_states.shape: {hidden_states.shape}")
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    # q, k, v -> [batch, len, 4096], [batch, len, 1024], [batch, len, 1024]
    # print(f"query_states.shape: {query_states.shape}, key_states.shape: {key_states.shape}, value_states.shape: {value_states.shape}")

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
    # key_states = repeat_kv(key_states, self.num_key_value_groups)
    # value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    # global gen_time
    # global atten_time
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
        
        # st_gen = time.perf_counter()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        # ed_gen = time.perf_counter()
        # gen_time += ed_gen - st_gen
        # attn_output = sageattn(
        #     query_states,
        #     key_states,
        #     value_states,
        #     # attn_mask=causal_mask,
        #     # dropout_p=self.attention_dropout if self.training else 0.0,
        #     is_causal=True,
        # )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
    else:
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)
        key_states = key_states.repeat_interleave(query_states.size(-3)//key_states.size(-3), -3)
        value_states = value_states.repeat_interleave(query_states.size(-3)//value_states.size(-3), -3)
        
        # attn_output = sageattn(
        #     query_states,
        #     key_states,
        #     value_states,
        #     # attn_mask=causal_mask,
        #     # dropout_p=self.attention_dropout if self.training else 0.0,
        #     is_causal=True,
        # )

        attn_output = sageattn_qk_int8_pv_fp16_triton(
            query_states,
            key_states,
            value_states,
            # attn_mask=causal_mask,
            # dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

    return attn_output, None


def set_sage_attn_triton_qk8_pv16(model):
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(LlamaSageAttnForward, layer.self_attn)

    