#!/bin/bash
#SBATCH --job-name=finetune_qwen_sweep
#SBATCH --output=/data/jp_rivera/model_transmit_v2/Qwen/recipes/finetune/deepspeed/slurm_files/slurm_output/%x_%A_%a.out
#SBATCH --error=/data/jp_rivera/model_transmit_v2/Qwen/recipes/finetune/deepspeed/slurm_files/slurm_output/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --mail-user=jprivera44@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=0-15  # 16 parallel jobs (indices 0..15)

# ------------------------------------------------------------------------------
# 1) Load environment
# ------------------------------------------------------------------------------
source /data/jp_rivera/miniconda3/etc/profile.d/conda.sh
conda activate zion

# ------------------------------------------------------------------------------
# 2) Define hyperparam grids
#    We'll produce 16 combos by combining 2x2x2x2 from the arrays below.
# ------------------------------------------------------------------------------
LR_LIST=("5e-5" "1e-5" "5e-6" "1e-6" "5e-7")
WARMUP_LIST=("0.01" "0.05")      # 2 options
SCHED_LIST=("linear" "cosine")   # 2 options
GCLIP_LIST=("0.5" "1.0")         # 2 options

# If you also want to sweep over grad_accum_steps, you could define an array
# and expand your array dimension. E.g. GACC_LIST=("8" "16") → 2 more combos 
# => 32 total. For now, let's keep it simpler.

# ------------------------------------------------------------------------------
# 3) Get the array indices
# ------------------------------------------------------------------------------
# We'll interpret the bits of $SLURM_ARRAY_TASK_ID to pick each dimension:
IDX=$SLURM_ARRAY_TASK_ID

# We'll do a little math to map 0..15 → the 4 arrays
# Each dimension has 2 possibilities, so we can decode with integer division/mod.
LR_IDX=$(( (IDX / 8) % 2 ))     # top bit
WARMUP_IDX=$(( (IDX / 4) % 2 )) # next bit
SCHED_IDX=$(( (IDX / 2) % 2 ))  
GCLIP_IDX=$(( IDX % 2 ))        # last bit

LR=${LR_LIST[$LR_IDX]}
WR=${WARMUP_LIST[$WARMUP_IDX]}
SCHED=${SCHED_LIST[$SCHED_IDX]}
GCLIP=${GCLIP_LIST[$GCLIP_IDX]}

# (If you'd rather define each combination manually in arrays, that's also fine.)

echo "=== Starting job $IDX with LR=$LR, Warmup=$WR, Scheduler=$SCHED, GradClip=$GCLIP ==="

# ------------------------------------------------------------------------------
# 4) Construct unique run name & output directory
# ------------------------------------------------------------------------------
RUN_NAME="epoch1_lr${LR}_warm${WR}_${SCHED}_gc${GCLIP}"
OUTDIR="output_qwen_${RUN_NAME}"

# ------------------------------------------------------------------------------
# 5) Launch training
#    - We fix num_train_epochs=1
#    - We pass in gradient_clipping and lr_scheduler_type, etc.
# ------------------------------------------------------------------------------
torchrun --nproc_per_node=4 --nnodes=1 --master_addr=localhost --master_port=6601 \
  ../../../../finetune.py \
    --model_name_or_path "/data/public_models/Qwen2-1.5B-Instruct" \
    --bf16 True \
    --dataset_name "redpajama" \
    --output_dir "${OUTDIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate "${LR}" \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio "${WR}" \
    --lr_scheduler_type "${SCHED}" \
    --logging_steps 1 \
    --report_to "wandb" \
    --wandb_project "qwen_sweep_reduce_spikes" \
    --wandb_run_name "${RUN_NAME}" \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --subset_size 9000 \
    --deepspeed "../../../../finetune/ds_config_zero2.json" \
    --optim "adamw_torch" \
    --gradient_clipping "${GCLIP}"

# Note: The above assumes your fine-tune script can accept:
#   --gradient_clipping <float>
# If not, you might need to do something like
#   --max_grad_norm <float>
# or adapt your code. 
# With DeepSpeed, you can also set gradient_clipping in ds_config, 
# but passing --gradient_clipping "auto" might be overridden by the CLI.

echo "=== Finished job $IDX with LR=$LR, Warmup=$WR, Scheduler=$SCHED, GradClip=$GCLIP ==="
