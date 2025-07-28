# export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=5
export WANDB_API_KEY='0472ad3924bc84e9db9a77d63ac636eb0e13a49d'
# wandb offline


MODEL_NAME=llama-3.1-8b-instruct
KERNEL_NAME=native
# BITWIDTH=mxfp8 
DATASET_NAME="all"
OUTPUT_PATH=./results/${KERNEL_NAME}_${BITWIDTH}_${MODEL_NAME}_${DATASET_NAME}

rm -rf $OUTPUT_PATH
mkdir -p $OUTPUT_PATH
echo "re-create $OUTPUT_PATH, log in $OUTPUT_PATH/native.log"

python evaluate/llama/llama_eval_dataset.py \
    --model $MODEL_NAME \
    --device cuda \
    --output_path $OUTPUT_PATH \
    --get_pred \
    --compute_accuracy \
    --test_dataset_name $DATASET_NAME \
    --kernel_name $KERNEL_NAME \
    2>&1 | tee $OUTPUT_PATH/native.log
    # --smooth_k \
    # --mxfp_bw $BITWIDTH \
