max_samples=80000000
train_ds="trecis-mc_tag-train-random-1"
save_dir="saves/LLaMA2-7B-Chat/lora/${train_ds}"
mkdir -p ${save_dir}


# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --do_train True \
#     --overwrite_output_dir True \
#     --overwrite_cache False \
#     --finetuning_type lora \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset ${train_ds} \
#     --cutoff_len 3072 \
#     --learning_rate 5e-05 \
#     --num_train_epochs 1.0 \
#     --max_samples ${max_samples} \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --max_grad_norm 1.0 \
#     --logging_steps 5 \
#     --save_steps 100 \
#     --warmup_steps 0 \
#     --flash_attn False \
#     --lora_rank 8 \
#     --lora_dropout 0.1 \
#     --lora_target q_proj,v_proj \
#     --output_dir ${save_dir} \
#     --fp16 True \
#     --plot_loss True > ${save_dir}/log.train

# # predict on top level
# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --overwrite_output_dir True \
#     --overwrite_cache False \
#     --checkpoint_dir ${save_dir} \
#     --predict_with_generate True \
#     --finetuning_type lora \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset try_trecis_val_top \
#     --cutoff_len 3072 \
#     --max_samples ${max_samples} \
#     --per_device_eval_batch_size 2 \
#     --max_new_tokens 1024 \
#     --top_p 0.7 \
#     --temperature 0.01 \
#     --output_dir ${save_dir}/results_val_top \
#     --do_predict True  > ${save_dir}/log.val

# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --overwrite_output_dir True \
#     --overwrite_cache False \
#     --checkpoint_dir ${save_dir} \
#     --predict_with_generate True \
#     --finetuning_type lora \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset try_sampled_5000_test_top \
#     --cutoff_len 3072 \
#     --max_samples ${max_samples} \
#     --per_device_eval_batch_size 2 \
#     --max_new_tokens 1024 \
#     --top_p 0.7 \
#     --temperature 0.01 \
#     --output_dir ${save_dir}/results_test_top \
#     --do_predict True > ${save_dir}/log.test


# # predict on high level
# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --overwrite_output_dir True \
#     --overwrite_cache False \
#     --checkpoint_dir ${save_dir} \
#     --predict_with_generate True \
#     --finetuning_type lora \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset try_trecis_val_high \
#     --cutoff_len 3072 \
#     --max_samples ${max_samples} \
#     --per_device_eval_batch_size 2 \
#     --max_new_tokens 1024 \
#     --top_p 0.7 \
#     --temperature 0.01 \
#     --output_dir ${save_dir}/results_val_high \
#     --do_predict True  > ${save_dir}/log.val

# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
#     --overwrite_output_dir True \
#     --overwrite_cache False \
#     --checkpoint_dir ${save_dir} \
#     --predict_with_generate True \
#     --finetuning_type lora \
#     --template llama2 \
#     --dataset_dir data \
#     --dataset try_sampled_5000_test_high \
#     --cutoff_len 3072 \
#     --max_samples ${max_samples} \
#     --per_device_eval_batch_size 2 \
#     --max_new_tokens 1024 \
#     --top_p 0.7 \
#     --temperature 0.01 \
#     --output_dir ${save_dir}/results_test_high \
#     --do_predict True > ${save_dir}/log.test


# predict on high level with inference template
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
    --overwrite_output_dir True \
    --overwrite_cache False \
    --checkpoint_dir ${save_dir} \
    --predict_with_generate True \
    --finetuning_type lora \
    --template llama2 \
    --dataset_dir data \
    --dataset trecis_if_val_high_0 \
    --cutoff_len 3072 \
    --max_samples ${max_samples} \
    --per_device_eval_batch_size 2 \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.01 \
    --output_dir ${save_dir}/results_if_val_high_0 \
    --do_predict True  > ${save_dir}/log.val_0

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
    --overwrite_output_dir True \
    --overwrite_cache False \
    --checkpoint_dir ${save_dir} \
    --predict_with_generate True \
    --finetuning_type lora \
    --template llama2 \
    --dataset_dir data \
    --dataset trecis_if_val_high_1 \
    --cutoff_len 3072 \
    --max_samples ${max_samples} \
    --per_device_eval_batch_size 2 \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.01 \
    --output_dir ${save_dir}/results_if_val_high_1 \
    --do_predict True  > ${save_dir}/log.val_1

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
    --overwrite_output_dir True \
    --overwrite_cache False \
    --checkpoint_dir ${save_dir} \
    --predict_with_generate True \
    --finetuning_type lora \
    --template llama2 \
    --dataset_dir data \
    --dataset trecis_if_sampled_5000_test_high_0 \
    --cutoff_len 3072 \
    --max_samples ${max_samples} \
    --per_device_eval_batch_size 2 \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.01 \
    --output_dir ${save_dir}/results_if_test_high_0 \
    --do_predict True > ${save_dir}/log.test_0

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path llm_collections/Llama-2-7b-chat-hf \
    --overwrite_output_dir True \
    --overwrite_cache False \
    --checkpoint_dir ${save_dir} \
    --predict_with_generate True \
    --finetuning_type lora \
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
    --do_predict True > ${save_dir}/log.test_1