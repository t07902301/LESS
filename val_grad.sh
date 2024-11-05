
CKPT=92
MODEL_LOG="TinyLlama/TinyLlama-1.1B-Chat-v1.0-p0.1-lora-seed3"
MODEL_PATH=../out/${MODEL_LOG}/checkpoint-${CKPT}
TASK=mmlu
OUTPUT_PATH=../grads/${MODEL_LOG}/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
DATA_DIR=data
DIMS="8192" # We use 8192 as our default projection dimension 

./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"