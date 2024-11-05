
CUDA_VISIBLE_DEVICES=1 python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir ../data/eval/mmlu \
    --save_dir result \
    --model_name_or_path ../../out/TinyLlama/TinyLlama-1.1B-Chat-v1.0-p0.1-lora \
    --tokenizer_name_or_path ../../out/TinyLlama/TinyLlama-1.1B-Chat-v1.0-p0.1-lora \
    --eval_batch_size 4 \
    --convert_to_bf16