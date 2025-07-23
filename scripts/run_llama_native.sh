# export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=2
export WANDB_API_KEY='0472ad3924bc84e9db9a77d63ac636eb0e13a49d'
# wandb offline


MODEL_NAME=llama-3.2-3b-instruct
KERNEL_NAME=native
# BITWIDTH=mxfp8 
DATASET_NAME="dureader"
OUTPUT_PATH=./results/${KERNEL_NAME}_${MODEL_NAME}_${DATASET_NAME}

rm -rf $OUTPUT_PATH
mkdir -p $OUTPUT_PATH
echo "re-create $OUTPUT_PATH, log in $OUTPUT_PATH/native.log"

python evaluate/llama/llama_main.py \
    --model $MODEL_NAME \
    --device cuda \
    --output_path $OUTPUT_PATH \
    --get_pred \
    --compute_accuracy \
    --test_dataset_name $DATASET_NAME \
    --kernel_name $KERNEL_NAME \
    2>&1 | tee $OUTPUT_PATH/native.log
    # --mxfp_bw $BITWIDTH \
    # --smooth_k \
