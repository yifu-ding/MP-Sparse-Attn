export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY='0472ad3924bc84e9db9a77d63ac636eb0e13a49d'

# 基础 attn 推理, tune mode
python evaluate/llama/llama_main.py \
    --model Llama-3.2-3B-Instruct \
    --device cuda \
    --output_path ./results \
    --model_out_path  ./evaluate/models_dict/llama-3.2-3b-instruct_l1_0.08_pv_l1_0.09-5shots.pt \
    --l1 0.08 \
    --pv_l1 0.09 \
    --num_fewshots 1 \
    --sparse_attention \
    --test_speedup \
    --test_accuracy \
    --test_dataset_name dureader \
    # --tune \
    # --use_wandb \
    # \

# 使用稀疏注意力推理
# python evaluate/llama/llama_main.py \
#     --model Llama-3.2-3B-Instruct  \
#     --device cuda \
#     --max_length 512 \
#     --sparse_attention  \
#     --block_size 50 \
#     --output_path ./results \
#     --test_dataset_name dureader \
#     --model_out_path  ./evaluate/models_dict

# # 在LongBench-E数据集上评估
# python evaluate/llama/llama_main.py --model llama2-7b-chat-4k --device cuda --max_length 512 --e
