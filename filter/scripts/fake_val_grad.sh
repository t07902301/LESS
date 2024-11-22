# This script is used to get the gradients of sampled training data to build data selection filter. 
TRAINING_DATA_NAME=dolly
TRAINING_DATA_FILE=~/data/llm/train/processed/dolly/sampled_val_dolly_data.jsonl # when changing data name, change the data path accordingly
GRADIENT_TYPE="adam"
MODEL_LOG="TinyLlama/TinyLlama-1.1B-Chat-v1.0-p0.1-lora-seed3"
DIMS="4096" # Dimension of Projected Gradient Vectors

START_TIME=$(date +%s)
GPU_ID=${1:-all}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU $CUDA_VISIBLE_DEVICES"
for CKPT in 23 46 70 92; do

    MODEL_PATH=~/out/${MODEL_LOG}/checkpoint-${CKPT}
    OUTPUT_PATH=~/grads/${MODEL_LOG}/fake_val/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}

    ./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
done

END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))
echo "Execution time: $EXECUTION_TIME seconds"