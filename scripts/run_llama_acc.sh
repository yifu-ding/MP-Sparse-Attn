# export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3
export WANDB_API_KEY='0472ad3924bc84e9db9a77d63ac636eb0e13a49d'

# MODEL_NAME=llama3-3-70b
# MODEL_NAME=llama-3.2-3b-instruct
# MODEL_NAME=qwen2.5-7b-instruct
MODEL_NAME=qwen2.5-14b-instruct
# MODEL_NAME=llama-3.1-8b-instruct
KERNEL_NAME=mxfp_attn
BITWIDTH=mixed
DATASET_NAME=all  # 2wikimqa
DIAG_TILE=1
SINK_TILE=1
QUANT_GRANULARITY=blockwise
QK_DTYPE=e4m3
OUTPUT_PATH=./results/${KERNEL_NAME}_${BITWIDTH}_${MODEL_NAME}_${DATASET_NAME}_TILE_${DIAG_TILE}+${SINK_TILE}_QK_${QK_DTYPE}_${QUANT_GRANULARITY}
# OUTPUT_PATH=./results/${KERNEL_NAME}_${BITWIDTH}_${MODEL_NAME}_${DATASET_NAME}_TILE_${DIAG_TILE}+${SINK_TILE}_QK_${QK_DTYPE}
#_${QUANT_GRANULARITY}

rm -rf $OUTPUT_PATH
mkdir -p $OUTPUT_PATH
echo "output path: $OUTPUT_PATH"

python evaluate/llama/llama_main.py \
    --model $MODEL_NAME \
    --device cuda \
    --output_path $OUTPUT_PATH \
    --test_dataset_name $DATASET_NAME \
    --mxfp_bw $BITWIDTH \
    --kernel_name $KERNEL_NAME \
    --smooth_k \
    --pre_quant \
    --fuse_mp_quant \
    --diag_tile $DIAG_TILE \
    --sink_tile $SINK_TILE \
    --quant_granularity $QUANT_GRANULARITY \
    --dual_scale \
    --get_pred \
    --compute_accuracy \
    --qk_dtype $QK_DTYPE \
    --num_fewshots 5 \
    2>&1 | tee $OUTPUT_PATH/all_tasks.log
    # --num_fewshots 10 \

sh scripts/send_email.sh $OUTPUT_PATH

    # --num_fewshots 5 \
    # --get_pred \
    # --compute_accuracy \
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
