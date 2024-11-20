TARGET_TASK_NAME="mmlu"
PERCENTAGE=0.5
TRAIN_FILES=selected_data/filtered/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
JOB_NAME=${MODEL_PATH}-p${PERCENTAGE}-lora-clf-filtered

GPU_ID=${1:-all}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU $CUDA_VISIBLE_DEVICES"
./less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 