# export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=6
export WANDB_API_KEY='0472ad3924bc84e9db9a77d63ac636eb0e13a49d'
# wandb offline

# ours
# python evaluate/llama/llama_main.py \
#     --model Llama-3.2-3B-Instruct \
#     --device cuda \
#     --output_path ./results \
#     --test_speedup \
#     --test_accuracy \
#     --skip_thresh 0.5 \
#     --kernel_name online_routing \
#     --use_wandb \
    # --num_fewshots 50 \
    # --test_dataset_name dureader \

# sparge attn triton 
# python evaluate/llama/llama_main.py \
#     --model Llama-3.2-3B-Instruct \
#     --device cuda \
#     --output_path ./results \
#     --model_out_path  ./evaluate/models_dict/llama-3.2-3b-instruct_l1_0.08_pv_l1_0.09-20shots.pt \
#     --l1 0.08 \
#     --pv_l1 0.09 \
#     --test_speedup \
#     --test_accuracy \
#     --kernel_name spargeattn_triton \
#     --use_wandb \
#     # --test_dataset_name dureader \
#     # --num_fewshots 50 \

# mxfp attn
python evaluate/llama/llama_main.py \
    --model Llama-3.2-3B-Instruct \
    --device cuda \
    --output_path ./results \
    --test_accuracy \
    --kernel_name mxfp_attn \
    --mxfp_bw mxfp8 \
    2>&1 | tee ./logs/mxfp_attn_llama-3.2-3b-instruct_mxfp8_test_accuracy.log
    # --test_speedup \
    # --use_wandb \


# # baseline: fp16 transformer attention
# python evaluate/llama/llama_main.py \
#     --model Llama-3.2-3B-Instruct \
#     --device cuda \
#     --output_path ./results \
#     --num_fewshots 50 \
#     --test_speedup \
#     --test_accuracy \
#     --test_dataset_name dureader \
#     # --use_wandb \
