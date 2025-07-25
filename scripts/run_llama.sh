# export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=4
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
# MODEL_NAME=llama3-3-70b
MODEL_NAME=llama-3.2-3b-instruct
# MODEL_NAME=llama-3.1-8b-instruct
KERNEL_NAME=flash_attn
BITWIDTH=fp16
PRE_QUANT=True
FUSE_MP_QUANT=True
DATASET_NAME=dureader
FP8_TILE_NUM=1
OUTPUT_PATH=./results/${KERNEL_NAME}_${BITWIDTH}_${MODEL_NAME}_${DATASET_NAME}_PRE_${PRE_QUANT}_FUSE_${FUSE_MP_QUANT}_TILE_${FP8_TILE_NUM}

rm -rf $OUTPUT_PATH
mkdir -p $OUTPUT_PATH
echo "re-create $OUTPUT_PATH"

python evaluate/llama/llama_main.py \
    --model $MODEL_NAME \
    --device cuda \
    --output_path $OUTPUT_PATH \
    --test_dataset_name $DATASET_NAME \
    --mxfp_bw $BITWIDTH \
    --kernel_name $KERNEL_NAME \
    --smooth_k \
    --pre_quant $PRE_QUANT \
    --fuse_mp_quant $FUSE_MP_QUANT \
    --fp8_tile_num $FP8_TILE_NUM \
    --get_pred \
    --compute_accuracy \
    --num_fewshots 5 \
    2>&1 | tee $OUTPUT_PATH/all_tasks.log
    # --test_speedup \
    # --get_pred \
    # --compute_accuracy \
    # --dual_scale \
    # --verbose \
    # --smooth_k \
    # --use_wandb \
      # --get_pred \
    # --compute_accuracy \


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
