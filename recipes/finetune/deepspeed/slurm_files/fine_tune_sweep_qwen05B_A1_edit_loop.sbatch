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
LR=3e-6           # learning rate
WARM=0.15         # warmup ratio
SCHED="polynomial"
GC=0.1            # gradient clipping
MAGPIE_SUBSET=5000
RP_SUBSET=5000
EPOCHS=1
PER_DEV_BATCH=8
GRAD_ACC=16

RUN_NAME="test_loop_new_epoch${EPOCHS}_lr${LR}_warm${WARM}_${SCHED}_gc${GC}_batch${PER_DEV_BATCH}"
OUTDIR="/data/public_models/jp_rivera/05_test/output_qwen_05_A1_1${RUN_NAME}"
PORT=47400  # or pick another free port

echo "=== Starting single-run job with LR=$LR, Warmup=$WARM, Sched=$SCHED, GC=$GC, EPOCHS=$EPOCHS ==="
echo "=== Magpie subset = $MAGPIE_SUBSET, RedPajama subset = $RP_SUBSET ==="
echo "=== Per-device batch = $PER_DEV_BATCH, Grad accum = $GRAD_ACC, Port=$PORT ==="

# ------------------------------------------------------------------------------
# 3) Launch training
# ------------------------------------------------------------------------------
torchrun \
  --nproc_per_node=4 --nnodes=1 \
  --master_addr=localhost --master_port="$PORT" \
  ../../../../finetune.py \
    --model_name_or_path "/data/public_models/jp_rivera/output_qwen_05_A1_1new_epoch1_lr3e-6_warm0.15_polynomial_gc0.1_batch8/" \
    --bf16 True \
    --dataset_name "magpie" \
    --magpie_subset_size "${MAGPIE_SUBSET}" \
    --rp_subset_size "${RP_SUBSET}" \
    --output_dir "${OUTDIR}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${PER_DEV_BATCH}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
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
LAST_CHECKPOINT=$(ls -d ${OUTDIR}/checkpoint-* | sort -V | tail -n 1)
echo "Last checkpoint from Magpie run: ${LAST_CHECKPOINT}"

echo "=== Starting RedPajama training run with LR=$LR, Warmup=$WARM, Sched=$SCHED, GC=$GC, EPOCHS=$EPOCHS ==="
echo "=== RedPajama subset = $RP_SUBSET ==="

# ------------------------------------------------------------------------------
# 3) Launch training
# ------------------------------------------------------------------------------
torchrun \
  --nproc_per_node=4 --nnodes=1 \
  --master_addr=localhost --master_port="$PORT" \
  ../../../../finetune.py \
    --model_name_or_path "${LAST_CHECKPOINT}"  \
    --bf16 True \
    --dataset_name "redpajama" \
    --magpie_subset_size "${MAGPIE_SUBSET}" \
    --rp_subset_size "${RP_SUBSET}" \
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

