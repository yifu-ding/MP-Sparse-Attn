from ours.mxfp_attn_func import mxfp_attn_kernel, block_scaled_batched_attn
from tests.tile_size_ablation import load_attention_states
from tests.time_profiler import time_profiler
from tests.flash_attn_triton import flash_attn_func

def test_attn_time_breakdown():
    batch_size = 1
    num_heads = 24
    qo_len = 4096
    kv_len = 4096
    head_dim = 128
    is_causal = True
    block_scale_type = "mixed"  # mixed, nvfp4, mxfp8, mxfp4
    qk_dtype = 'e4m3'  # e4m3, e5m2
    smooth_k = True
    dual_scale = False
    quant_granularity = "tokenwise"  # tokenwise, blockwise, tensorwise
    tile_size = 2
    sink_size = 2
    # q_pack_along_lastdim = False
    # k_pack_along_lastdim = False
    if "fp4" in block_scale_type: qk_dtype = "e2m1"

    text = f"block_scale_type: {block_scale_type}, qk_dtype: {qk_dtype}, dual_scale: {dual_scale}"
    if dual_scale:
        text += f", granularity: {quant_granularity}"
    if block_scale_type == "mixed":
        text += f", tile_size: {tile_size}, sink_size: {sink_size}"
    print(text)

    # q = torch.randn(batch_size, num_heads, qo_len, head_dim,
    #                 device='cuda', dtype=torch.float16)
    # k = torch.randn(batch_size, num_heads, kv_len, head_dim, 
    #                 device='cuda', dtype=torch.float16)
    # v = torch.randn(batch_size, num_heads, kv_len, head_dim,
    #                 device='cuda', dtype=torch.float16)
    query_states, key_states, value_states = load_attention_states(8192)
    q = query_states
    k = key_states
    v = value_states

    kwargs = {
        "is_causal": is_causal,
        "smooth_k": smooth_k,
        "block_scale_type": block_scale_type,
        "dual_scale": dual_scale,
        "quant_granularity": quant_granularity,
        "fuse_mp_quant": True,
        "pre_quant": True,
        "fuse_pack": True,
        "diag_tile": tile_size,
        "sink_tile": sink_size,
        "qk_dtype": qk_dtype,
    }

    def test_flash_attn_func(q, k, v, **kwargs):
        q = q.transpose(1, 2).contiguous()  
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        flash_attn_func(q, k, v, None, True, None)

    time_profiler(mxfp_attn_kernel, q, k, v, **kwargs)
    time_profiler(test_flash_attn_func, q, k, v, **kwargs)


if __name__ == "__main__":
    test_attn_time_breakdown()