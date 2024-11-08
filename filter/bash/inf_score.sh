DIM=8192 # decide which dimension to use
MODEL_LOG="TinyLlama/TinyLlama-1.1B-Chat-v1.0-p0.1-lora-seed3"
TRAIN_FILE_NAMES="dolly"
CKPTS="92" # checkpoing index
CHECKPOINT_WEIGHTS="7.7030e-06" # average lr of the epoch
TARGET_TASK_NAMES="mmlu"
VALIDATION_GRADIENT_PATH=../grads/${MODEL_LOG}/${TARGET_TASK_NAMES}-ckpt${CKPTS}-sgd/dim${DIM}
GRADIENT_PATH=../grads/${MODEL_LOG}/filtered/${TRAIN_FILE_NAMES}-ckpt${CKPTS}-adam/dim${DIM}

SELECTED_DATA_OUTPUT_PATH="selected_data/filtered"

./less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"