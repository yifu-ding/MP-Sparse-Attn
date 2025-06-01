export CUDA_VISIBLE_DEVICES=0

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


TRITON_INTERPRET=1 python test_batched_matmul.py
