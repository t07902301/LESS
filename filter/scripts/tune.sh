TARGET_TASK_NAME="mmlu"
PERCENTAGE=0.35
TRAIN_FILES=selected_data/filtered/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
JOB_NAME=${MODEL_PATH}-p${PERCENTAGE}-lora-clf-filtered

./less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 