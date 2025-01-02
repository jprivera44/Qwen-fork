#!/bin/bash
#SBATCH --job-name=finetune_qwen_retry_failed
#SBATCH --output=.../%x_%A_%a.out
#SBATCH --error=.../%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mail-user=jprivera44@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=0-4

# --save_steps 1000 \
# --save_total_limit 10 \

source /data/jp_rivera/miniconda3/etc/profile.d/conda.sh
conda activate zion

# Format each line: "lr warm sched gradclip"
# e.g. Lower LR to 3e-5, bigger warmup, smaller GC if you like
COMBOS=(
  "3e-5 0.1 linear 1.0"
  "3e-5 0.05 linear 1.0"
  "1e-5 0.05 linear 0.5"
  "1e-5 0.1 cosine 0.5"
  "1e-5 0.1 cosine 1.0"
)

IDX=$SLURM_ARRAY_TASK_ID
read LR WARM SCHED GC <<< "${COMBOS[$IDX]}"

SUBSET_SIZE=18000
EPOCHS=2  # Increase from 1 to 2
RUN_NAME="epoch${EPOCHS}_lr${LR}_warm${WARM}_${SCHED}_gc${GC}_retry"
OUTDIR="output_qwen_${RUN_NAME}"
PORT=$((6600 + IDX))

echo "=== Starting job $IDX with LR=$LR, Warmup=$WARM, Sched=$SCHED, GC=$GC, EPOCHS=$EPOCHS, Subset=$SUBSET_SIZE ==="

torchrun \
  --nproc_per_node=4 --nnodes=1 \
  --master_addr=localhost --master_port="$PORT" \
  ../../../../finetune.py \
    --model_name_or_path "/data/public_models/compressed__7b_nc2_ogs32_igs1_sb1" \
    --bf16 True \
    --dataset_name "redpajama" \
    --output_dir "${OUTDIR}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
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

echo "=== Finished job $IDX with LR=$LR, Warmup=$WARM, Sched=$SCHED, GC=$GC, EPOCHS=$EPOCHS ==="
