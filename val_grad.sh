MODEL_LOG="TinyLlama/TinyLlama-1.1B-Chat-v1.0-p0.1-lora-seed3"

TASK=mmlu

DATA_DIR=data
DIMS="4096" # We use 8192 as our default projection dimension 

# Get the GPU argument, default to using GPU 0 and 1
GPU=${1:-0,1}

for CKPT in 23 46 70 92; do
    echo "Processing checkpoint ${CKPT}"

    MODEL_PATH=../out/${MODEL_LOG}/checkpoint-${CKPT}
    OUTPUT_PATH=../grads/${MODEL_LOG}/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
    CUDA_VISIBLE_DEVICES=$GPU ./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"
done
