GPU_ID=${1:-all}
export CUDA_VISIBLE_DEVICES=$GPU_ID
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir ~/data/llm/eval/mmlu \
    --save_dir result/filtered/clf \
    --model_name_or_path ~/out/TinyLlama/TinyLlama-1.1B-Chat-v1.0-p0.5-lora-clf-filtered \
    --tokenizer_name_or_path ~/out/TinyLlama/TinyLlama-1.1B-Chat-v1.0-p0.5-lora-clf-filtered \
    --eval_batch_size 4 \
    --convert_to_bf16