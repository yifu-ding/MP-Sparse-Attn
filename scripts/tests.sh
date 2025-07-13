export CUDA_VISIBLE_DEVICES=4

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

# export TRITON_INTERPRET=1 
# export TRITON_DEBUG=1
# export TRITON_IR_VERBOSE=1
# export TRITON_PTXAS_VERBOSE=1
# python ours/mxfp_attn_kernel.py
python test_performance.py