#!/bin/bash
#SBATCH --output=/data/jp_rivera/model_transmit_v2/Qwen/recipes/finetune/deepspeed/slurm_files/slurm_output/%x_%j.out
#SBATCH --error=/data/jp_rivera/model_transmit_v2/Qwen/recipes/finetune/deepspeed/slurm_files/slurm_output/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=45:00:00
#SBATCH --job-name=finetune_qwen
#SBATCH --mail-user=jprivera44@gmail.com
#SBATCH --mail-type=ALL

# Load the environment
source /data/jp_rivera/miniconda3/etc/profile.d/conda.sh
conda activate zion
# --data_path "../Belle_sampled_qwen.json" \
# Launch fine-tuning using torchrun
#--model_name_or_path "/data/public_models/Qwen2-0.5B" \
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=6601 ../../../../finetune.py \
    --model_name_or_path "/data/public_models/compressed__7b_nc2_ogs32_igs1_sb1" \
    --bf16 True \
    --dataset_name "redpajama" \
    --output_dir "output_qwen" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --subset_size 9000 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --wandb_project "qwen_2-1-5-compressed_run" \
    --wandb_run_name "first_test" \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed "../../../../finetune/ds_config_zero2.json"
