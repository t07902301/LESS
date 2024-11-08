train_files=data/train/processed/dolly/train_dolly_data.jsonl
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
PERCENTAGE=0.1 # percentage of the full data to train, you can specify the training file you want to use in the script
DATA_SEED=3
JOB_NAME=${MODEL_PATH}-p${PERCENTAGE}-lora-seed${DATA_SEED}

./less/scripts/train/warmup_lora_train.sh "$train_files" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"

# DATA_DIR=data
# MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
# PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
# DATA_SEED=3
# JOB_NAME=tiny-llama-p${PERCENTAGE}-lora-seed${DATA_SEED}

# ./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"
