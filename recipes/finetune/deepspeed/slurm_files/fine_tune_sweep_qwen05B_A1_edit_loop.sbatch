#!/bin/bash
#SBATCH --job-name=finetune_qwen_smooth_loss
#SBATCH --output=/data/jp_rivera/model_transmit_v2/Qwen/recipes/finetune/deepspeed/slurm_files/slurm_output/A_%x_%j.out
#SBATCH --error=/data/jp_rivera/model_transmit_v2/Qwen/recipes/finetune/deepspeed/slurm_files/slurm_output/A_%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=36:00:00
#SBATCH --mail-user=jprivera44@gmail.com
#SBATCH --mail-type=ALL

# ------------------------------------------------------------------------------
# 1) Load environment
# ------------------------------------------------------------------------------
source /data/jp_rivera/miniconda3/etc/profile.d/conda.sh
conda activate zion

# ------------------------------------------------------------------------------
# 2) Set hyperparameters
# ------------------------------------------------------------------------------
LR=3e-6
WARM=0.15
SCHED="polynomial"
GC=0.1
MAGPIE_SUBSET=5000
RP_SUBSET=5000
EPOCHS=1
PER_DEV_BATCH=8
GRAD_ACC=16
PORT=47400

# Get current date/time for run names
DATETIME=$(date "+%m%d_%H")


# First run with Magpie
RUN_NAME="MAG_${DATETIME}_epoch${EPOCHS}_lr${LR}_warm${WARM}_${SCHED}_gc${GC}_batch${PER_DEV_BATCH}"
OUTDIR="/data/public_models/jp_rivera/05_test/output_qwen_MAG_test${RUN_NAME}"

echo "=== Starting Magpie training run with LR=$LR, Warmup=$WARM, Sched=$SCHED, GC=$GC, EPOCHS=$EPOCHS ==="
echo "=== Magpie subset = $MAGPIE_SUBSET ==="


torchrun \
  --nproc_per_node=4 --nnodes=1 \
  --master_addr=localhost --master_port="$PORT" \
  ../../../../finetune.py \
  --model_name_or_path "/data/public_models/compressed_Qwen2-0.5B-Instruct_5b_nc2_ogs8_igs1_sb2" \
  --dataset_name "magpie" \
  --magpie_subset_size "${MAGPIE_SUBSET}" \
  --output_dir "${OUTDIR}" \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size "${PER_DEV_BATCH}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 2 \
  --learning_rate "${LR}" \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_epsilon 1e-8 \
  --max_grad_norm "${GC}" \
  --warmup_ratio "${WARM}" \
  --warmup_steps 100 \
  --lr_scheduler_type "${SCHED}" \
  --logging_steps 1 \
  --logging_first_step true \
  --report_to "wandb" \
  --wandb_project "compressed_qwen_05B_instruct_A1" \
  --wandb_run_name "${RUN_NAME}" \
  --model_max_length 512 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --deepspeed "../../../../finetune/ds_config_zero2.json" \
  --optim "adamw_torch" \
  --gradient_clipping "${GC}" \
  --seed 42 \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory true

echo "=== Finished single-run job with LR=$LR, Warmup=$WARM, Sched=$SCHED, GC=$GC, EPOCHS=$EPOCHS ==="





# # Get the last checkpoint from the first run
# LAST_CHECKPOINT=$(ls -d ${OUTDIR}/checkpoint-* | sort -V | tail -n 1)
# echo "Last checkpoint from Magpie run: ${LAST_CHECKPOINT}"




# RUN_NAME="RP_${DATETIME}_epoch${EPOCHS}_lr${LR}_warm${WARM}_${SCHED}_gc${GC}_batch${PER_DEV_BATCH}"
# OUTDIR="/data/public_models/jp_rivera/05_test/output_qwen_RP_${RUN_NAME}"

# echo "=== Starting RedPajama training run with LR=$LR, Warmup=$WARM, Sched=$SCHED, GC=$GC, EPOCHS=$EPOCHS ==="
# echo "=== RedPajama subset = $RP_SUBSET ==="

# # Second training run (Magpie)
# torchrun \
#   --nproc_per_node=4 --nnodes=1 \
#   --master_addr=localhost --master_port="$PORT" \
#   ../../../../finetune.py \
#   --model_name_or_path "${LAST_CHECKPOINT}" \
#   --dataset_name "redpajama" \
#   --rp_subset_size "${RP_SUBSET}" \
#   --output_dir "${OUTDIR}" \
#   --num_train_epochs "${EPOCHS}" \
#   --per_device_train_batch_size "${PER_DEV_BATCH}" \
#   --per_device_eval_batch_size 1 \
#   --gradient_accumulation_steps "${GRAD_ACC}" \
#   --evaluation_strategy "no" \
#   --save_strategy "steps" \
#   --save_steps 500 \
#   --save_total_limit 2 \
#   --learning_rate "${LR}" \
#   --weight_decay 0.01 \
#   --adam_beta1 0.9 \
#   --adam_beta2 0.999 \
#   --adam_epsilon 1e-8 \
#   --max_grad_norm "${GC}" \
#   --warmup_ratio "${WARM}" \
#   --warmup_steps 100 \
#   --lr_scheduler_type "${SCHED}" \
#   --logging_steps 1 \
#   --logging_first_step true \
#   --report_to "wandb" \
#   --wandb_project "compressed_qwen_05B_instruct_A1" \
#   --wandb_run_name "${RUN_NAME}" \
#   --model_max_length 512 \
#   --gradient_checkpointing True \
#   --lazy_preprocess True \
#   --deepspeed "../../../../finetune/ds_config_zero2.json" \
#   --optim "adamw_torch" \
#   --gradient_clipping "${GC}" \
#   --seed 42 \
#   --dataloader_num_workers 4 \
#   --dataloader_pin_memory true

# echo "=== Finished single-run job with LR=$LR, Warmup=$WARM, Sched=$SCHED, GC=$GC, EPOCHS=$EPOCHS ==="


# echo "=== Finished both training runs ==="