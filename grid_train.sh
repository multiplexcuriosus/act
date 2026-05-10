#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  /home/dyros/Data/jg_data/hdf5/_20260509_172409
  /home/dyros/Data/jg_data/hdf5/_20260509_173147
)

BASE_CKPT="/home/dyros/Data/jg_data/ckpts/grid_combined_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_CKPT"

# Save this script into the run folder for reproducibility
cp "$0" "$BASE_CKPT/run_grid.sh"

for LR in 1e-5 2e-5; do
  for BS in 8 16; do
    for KL in 5 10; do
      RUN_NAME="lr_${LR}_bs_${BS}_kl_${KL}"
      CKPT_DIR="${BASE_CKPT}/${RUN_NAME}"
      mkdir -p "$CKPT_DIR"

      echo "=== Starting ${RUN_NAME} ==="
      echo "Checkpoint dir: ${CKPT_DIR}"

      python imitate_episodes.py \
        --policy_class ACT \
        --task_name real_franka_ball_ring \
        --ckpt_dir "$CKPT_DIR" \
        --dataset_dirs "${DATASETS[@]}" \
        --camera_names rgb \
        --data_mode joint \
        --state_dim 8 \
        --action_dim 7 \
        --batch_size "$BS" \
        --num_epochs 5000 \
        --lr "$LR" \
        --chunk_size 30 \
        --hidden_dim 512 \
        --dim_feedforward 3200 \
        --kl_weight "$KL" \
        --seed 0 \
        2>&1 | tee "${CKPT_DIR}/train.log"

      echo "=== Finished ${RUN_NAME} ==="
      echo
    done
  done
done

echo "All grid runs finished."
echo "Results saved in: ${BASE_CKPT}"
