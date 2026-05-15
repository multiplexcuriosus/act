#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  /home/dyros/Data/jg_data/hdf5/recording_20260514_155523/
  /home/dyros/Data/jg_data/hdf5/recording_20260514_160133/
  /home/dyros/Data/jg_data/hdf5/recording_20260514_160705/
)

BASE_CKPT="/home/dyros/Data/jg_data/ckpts/grid_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_CKPT"

# Save this script into the run folder for reproducibility
cp "$0" "$BASE_CKPT/run_grid.sh"

for LR in 1e-5 2e-5; do
  for BS in 8 16; do
    for KL in 5 10; do
      for CAM in "event" "rgb"; do
        RUN_NAME="cam_${CAM}_lr_${LR}_bs_${BS}_kl_${KL}"
        CKPT_DIR="${BASE_CKPT}/${RUN_NAME}"
        mkdir -p "$CKPT_DIR"

        echo "=== Starting ${RUN_NAME} ==="
        echo "Checkpoint dir: ${CKPT_DIR}"
        echo "Camera: ${CAM}"
        echo "LR: ${LR}, BS: ${BS}, KL: ${KL}"
        echo

        python imitate_episodes.py \
          --policy_class ACT \
          --task_name real_franka_ball_ring \
          --ckpt_dir "$CKPT_DIR" \
          --dataset_dirs "${DATASETS[@]}" \
          --camera_names "$CAM" \
          --data_mode joint \
          --state_dim 8 \
          --action_dim 7 \
          --batch_size "$BS" \
          --num_epochs 10000 \
          --lr "$LR" \
          --chunk_size 30 \
          --hidden_dim 512 \
          --dim_feedforward 3200 \
          --kl_weight "$KL" \
          --seed 0 \
          --checkpoint_interval 0 \
          2>&1 | tee "${CKPT_DIR}/train.log"

        echo "=== Finished ${RUN_NAME} ==="
        echo
      done
    done
  done
done

echo "All grid runs finished."
echo "Results saved in: ${BASE_CKPT}"