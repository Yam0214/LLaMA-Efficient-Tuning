save_dir="saves/LLaMA2-7B-Chat/zero_shot/trecis-if"
mkdir -p ${save_dir}
max_samples=80000000
# max_samples=2

# # predict on val set
# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --overwrite_cache False \
#     --predict_with_generate True \
#     --finetuning_type freeze \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset trecis_mc_val_top \
#     --cutoff_len 3072 \
#     --max_samples 100000 \
#     --per_device_eval_batch_size 2 \
#     --max_new_tokens 1024 \
#     --top_p 0.7 \
#     --temperature 0.01 \
#     --output_dir ${save_dir}/results_val_top \
#     --do_predict True  > ${save_dir}/log.val_top

# # predict on val set
# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --overwrite_cache False \
#     --predict_with_generate True \
#     --finetuning_type freeze \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset trecis_mc_sampled_5000_test_top \
#     --cutoff_len 3072 \
#     --max_samples 100000 \
#     --per_device_eval_batch_size 2 \
#     --max_new_tokens 1024 \
#     --top_p 0.7 \
#     --temperature 0.01 \
#     --output_dir ${save_dir}/results_test_top \
#     --do_predict True > ${save_dir}/log.test_top


# # predict on val set
# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --overwrite_cache False \
#     --predict_with_generate True \
#     --finetuning_type freeze \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset trecis_mc_trecis_val_high \
#     --cutoff_len 3072 \
#     --max_samples 100000 \
#     --per_device_eval_batch_size 2 \
#     --max_new_tokens 128 \
#     --top_p 0.7 \
#     --temperature 0.01 \
#     --output_dir ${save_dir}/results_val_high \
#     --do_predict True  > ${save_dir}/log.val_high

# # predict on val set
# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --overwrite_cache False \
#     --predict_with_generate True \
#     --finetuning_type freeze \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset trecis_mc_sampled_5000_test_high \
#     --cutoff_len 3072 \
#     --max_samples 100000 \
#     --per_device_eval_batch_size 2 \
#     --max_new_tokens 128 \
#     --top_p 0.7 \
#     --temperature 0.01 \
#     --output_dir ${save_dir}/results_test_high \
#     --do_predict True > ${save_dir}/log.test_high

# predict on val set
# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --overwrite_cache False \
#     --predict_with_generate True \
#     --finetuning_type freeze \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset trecis_if_val_high_0 \
#     --cutoff_len 3072 \
#     --max_samples ${max_samples} \
#     --per_device_eval_batch_size 2 \
#     --max_new_tokens 128 \
#     --top_p 0.7 \
#     --temperature 0.01 \
#     --output_dir ${save_dir}/results_if_val_high_0 \
#     --do_predict True  > ${save_dir}/log.val_high_0

# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --overwrite_cache False \
#     --predict_with_generate True \
#     --finetuning_type freeze \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset trecis_if_val_high_1 \
#     --cutoff_len 3072 \
#     --max_samples ${max_samples} \
#     --per_device_eval_batch_size 2 \
#     --max_new_tokens 128 \
#     --top_p 0.7 \
#     --temperature 0.01 \
#     --output_dir ${save_dir}/results_if_val_high_1 \
#     --do_predict True  > ${save_dir}/log.val_high_1

# predict on sampled test set
# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --overwrite_cache False \
#     --predict_with_generate True \
#     --finetuning_type freeze \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset trecis_if_sampled_5000_test_high_0 \
#     --cutoff_len 3072 \
#     --max_samples ${max_samples} \
#     --per_device_eval_batch_size 2 \
#     --max_new_tokens 128 \
#     --top_p 0.7 \
#     --temperature 0.01 \
#     --output_dir ${save_dir}/results_if_test_high_0 \
#     --do_predict True > ${save_dir}/log.test_high_0

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
    --overwrite_cache False \
    --predict_with_generate True \
    --finetuning_type freeze \
    --template llama2 \
    --dataset_dir data \
    --dataset trecis_if_sampled_5000_test_high_1 \
    --cutoff_len 3072 \
    --max_samples ${max_samples} \
    --per_device_eval_batch_size 2 \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.01 \
    --output_dir ${save_dir}/results_if_test_high_1 \
    --do_predict True > ${save_dir}/log.test_high_1