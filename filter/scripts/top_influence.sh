TRAIN_FILE_NAMES="dolly"
TARGET_TASK_NAMES="mmlu"

SELECTED_DATA_OUTPUT_PATH="selected_data/filtered/"

GPU_ID=${1:-all}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU $CUDA_VISIBLE_DEVICES"

python3 -m less.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files ~/data/llm/train/processed/dolly/train_dolly_data.jsonl \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage 0.5