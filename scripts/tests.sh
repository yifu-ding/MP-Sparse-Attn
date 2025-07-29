export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:$(pwd)

# test our kernel
# TRITON_INTERPRET=1 python test_online_routing.py
# python test_online_routing.py

# test speedup on kernels (sparge, torch.nn, fa)
# python test_performance.py

# TRITON_INTERPRET=1 python spas_sage_attn/test_quant.py
# TRITON_INTERPRET=1 
# python spas_sage_attn/test_dot_scaled.py

# TRITON_INTERPRET=1
# python spas_sage_attn/test_quant.py --format mxfp4
# --K_range 8192 16384 --K_step 2048 --bench


# TRITON_INTERPRET=1 python test_batched_matmul.py

export TRITON_INTERPRET=1
export TRITON_DEBUG=1
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
export TRITON_IR_VERBOSE=1
# export TRITON_PTXAS_VERBOSE=1
# python ours/mxfp_attn_kernel.py
if [ "$1" = "1" ]; then
    python tests/test_performance.py
elif [ "$1" = "2" ]; then
    python scripts/debug2.py
elif [ "$1" = "3" ]; then
    python scripts/debug3.py
elif [ "$1" = "4" ]; then
    python tests/test_quant.py
elif [ "$1" = "5" ]; then
    python tests/tile_size_ablation.py
fi
