#!/bin/bash
#SBATCH --job-name=finetune_qwen_retry_failed
#SBATCH --output=/data/jp_rivera/model_transmit_v2/Qwen/recipes/finetune/deepspeed/slurm_files/slurm_output/%x_%A_%a.out
#SBATCH --error=/data/jp_rivera/model_transmit_v2/Qwen/recipes/finetune/deepspeed/slurm_files/slurm_output/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mail-user=jprivera44@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=0-4   # 5 combos total (indices 0..4)

# ------------------------------------------------------------------------------
# 1) Load environment
# ------------------------------------------------------------------------------
source /data/jp_rivera/miniconda3/etc/profile.d/conda.sh
conda activate zion

# ------------------------------------------------------------------------------
# 2) List only the combos that FAILED or CRASHED
#    (Adjust these combos to match your actual failures.)
# ------------------------------------------------------------------------------
# Format each line: "lr warm sched gradclip"
COMBOS=(
  "5e-5 0.05 linear 1.0"       # e.g. epoch1_lr5e-5_warm0.05_linear_gc1.0
  "5e-5 0.01 linear 1.0"       # e.g. epoch1_lr5e-5_warm0.01_linear_gc1.0
  "5e-5 0.05 linear 0.5"       # e.g. epoch1_lr5e-5_warm0.05_linear_gc0.5
  "1e-5 0.01 cosine 0.5"       # e.g. epoch1_lr1e-5_warm0.01_cosine_gc0.5
  "1e-5 0.01 cosine 1.0"       # e.g. epoch1_lr1e-5_warm0.01_cosine_gc1.0
)

# ------------------------------------------------------------------------------
# 3) Pick the combo based on $SLURM_ARRAY_TASK_ID
# ------------------------------------------------------------------------------
IDX=$SLURM_ARRAY_TASK_ID
read LR WARM SCHED GC <<< "${COMBOS[$IDX]}"

# ------------------------------------------------------------------------------
# 4) Construct run name, output dir, subset size
#    We'll do 1 epoch, subset=18000 (double the 9000).
# ------------------------------------------------------------------------------
SUBSET_SIZE=18000
RUN_NAME="epoch1_lr${LR}_warm${WARM}_${SCHED}_gc${GC}_retry"
OUTDIR="output_qwen_${RUN_NAME}"

# ------------------------------------------------------------------------------
# 5) Pick a unique master port to avoid collisions
# ------------------------------------------------------------------------------
PORT=$((6600 + IDX))  # e.g. 6600, 6601, 6602, ...

echo "=== Starting retry job $IDX with LR=$LR, Warmup=$WARM, Scheduler=$SCHED, GC=$GC, Subset=$SUBSET_SIZE, Port=$PORT ==="

# ------------------------------------------------------------------------------
# 6) Launch training
# ------------------------------------------------------------------------------
torchrun \
  --nproc_per_node=4 --nnodes=1 \
  --master_addr=localhost --master_port="$PORT" \
  ../../../../finetune.py \
    --model_name_or_path "/data/public_models/compressed__7b_nc2_ogs32_igs1_sb1" \
    --bf16 True \
    --dataset_name "redpajama" \
    --output_dir "${OUTDIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate "${LR}" \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio "${WARM}" \
    --lr_scheduler_type "${SCHED}" \
    --logging_steps 1 \
    --report_to "wandb" \
    --wandb_project "compressed_qwen_sweep_reduce_spikes" \
    --wandb_run_name "${RUN_NAME}" \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --subset_size "${SUBSET_SIZE}" \
    --deepspeed "../../../../finetune/ds_config_zero2.json" \
    --optim "adamw_torch" \
    --gradient_clipping "${GC}"

echo "=== Finished retry job $IDX with LR=$LR, Warmup=$WARM, Scheduler=$SCHED, GC=$GC ==="
