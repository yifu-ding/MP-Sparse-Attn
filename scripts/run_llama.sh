# export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=2
export WANDB_API_KEY='0472ad3924bc84e9db9a77d63ac636eb0e13a49d'

# MODEL_NAME=llama3-3-70b
# MODEL_NAME=llama-3.2-3b-instruct
MODEL_NAME=qwen2.5-7b-instruct
# MODEL_NAME=llama-3.1-8b-instruct
KERNEL_NAME=mxfp_attn
BITWIDTH=mixed
DATASET_NAME=narrativeqa  # 2wikimqa
DIAG_TILE=1
SINK_TILE=1
QUANT_GRANULARITY=tokenwise
QK_DTYPE=e4m3
# OUTPUT_PATH=./results/speed_${KERNEL_NAME}_${BITWIDTH}_${MODEL_NAME}_${DATASET_NAME}_TILE_${DIAG_TILE}+${SINK_TILE}_QK_${QK_DTYPE}_${QUANT_GRANULARITY}
OUTPUT_PATH=./results/speed_${KERNEL_NAME}_${BITWIDTH}_${MODEL_NAME}_${DATASET_NAME}_TILE_${DIAG_TILE}+${SINK_TILE}_QK_${QK_DTYPE}
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
    --qk_dtype $QK_DTYPE \
    --test_speedup \
    --dual_scale \
    --num_fewshots 1 \
    2>&1 | tee $OUTPUT_PATH/all_tasks.log

# sh scripts/send_email.sh $OUTPUT_PATH
