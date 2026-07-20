#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
/home/dyros/Data/jg_data/hdf5/recording_20260716_161544_intercept_pipeline_test
)

# Must equal /observations/qpos.shape[1].
STATE_DIM=7

BASE_CKPT="/home/dyros/Data/jg_data/ckpts/intercept_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_CKPT"

if [[ -f "$0" ]]; then
  cp "$0" "$BASE_CKPT/run_grid.sh"
fi

# The current intercept HDF5 contains RGB only.
CAMERA_MODES=(rgb)

for IMAGE_SIZE in 320; do
  for LR in 1e-5 2e-5; do
    for BS in 4; do
      for KL in 1 10; do
        for CAM_MODE in "${CAMERA_MODES[@]}"; do

          case "$CAM_MODE" in
            rgb)
              CAM_NAMES=(rgb)
              CAM_TAG="rgb"
              ;;
            *)
              echo "Unsupported CAM_MODE for this dataset: $CAM_MODE" >&2
              exit 1
              ;;
          esac

          RUN_NAME="intercept_cam_${CAM_TAG}_lr_${LR}_bs_${BS}_kl_${KL}_imgsize_${IMAGE_SIZE}"
          CKPT_DIR="${BASE_CKPT}/${RUN_NAME}"
          mkdir -p "$CKPT_DIR"

          echo "=== Starting ${RUN_NAME} ==="
          echo "Checkpoint dir: ${CKPT_DIR}"
          echo "Datasets: ${DATASETS[*]}"
          echo "Cameras: ${CAM_NAMES[*]}"
          echo "State dim: ${STATE_DIM}"
          echo "Action: [goto_s, is_commanded]"
          echo "LR: ${LR}, BS: ${BS}, KL: ${KL}, IMAGE_SIZE: ${IMAGE_SIZE}"
          echo

          CMD=(
            python imitate_episodes.py
            --policy_class ACT
            --task_name real_franka_ball_intercept
            --ckpt_dir "$CKPT_DIR"
            --dataset_dirs "${DATASETS[@]}"
            --camera_names "${CAM_NAMES[@]}"

            --data_mode intercept
            --state_dim "$STATE_DIM"
            --action_dim 2
            --use_bce_last_action_dim

            --batch_size "$BS"
            --num_epochs 5000
            --lr "$LR"
            --chunk_size 30
            --hidden_dim 512
            --dim_feedforward 3200
            --kl_weight "$KL"
            --seed 0

            --checkpoint_interval 1000
            --save_extra_checkpoints
            --profile_memory
            --image_size "$IMAGE_SIZE"
          )

          "${CMD[@]}" 2>&1 | tee "${CKPT_DIR}/train.log"

          echo "=== Finished ${RUN_NAME} ==="
          echo
        done
      done
    done
  done
done

echo "All intercept runs finished."
echo "Results saved in: ${BASE_CKPT}"