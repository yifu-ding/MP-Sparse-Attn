export CUDA_VISIBLE_DEVICES=0

# test our kernel
# TRITON_INTERPRET=1 python test_online_routing.py
python test_online_routing.py

# test speedup on kernels (sparge, torch.nn, fa)
# python test_performance.py