CKPT=92

TRAINING_DATA_NAME=dolly
TRAINING_DATA_FILE=data/train/processed/dolly/val_dolly_data.jsonl # when changing data name, change the data path accordingly
GRADIENT_TYPE="adam"
MODEL_LOG="TinyLlama/TinyLlama-1.1B-Chat-v1.0-p0.1-lora-seed3"
MODEL_PATH=../out/${MODEL_LOG}/checkpoint-${CKPT}
OUTPUT_PATH=../grads/${MODEL_LOG}/fake_val/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

START_TIME=$(date +%s)

# Your script or command here
./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))
echo "Execution time: $EXECUTION_TIME seconds"