#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  "/home/dyros/Data/jg_data/hdf5/event_only/hdf5_20260728_225008"
  "/home/dyros/Data/jg_data/hdf5/event_only/hdf5_20260727_215525"
)

# HDF5 qpos remains (T, 7), but the loader concatenates:
# qpos[t-6], qpos[t-3], qpos[t] -> model state dimension 21.
RAW_QPOS_DIM=7
STATE_DIM=21
ACTION_DIM=1

VISUAL_HISTORY_FRAMES=3
CHUNK_SIZE=30

DATASET_NAME="$(basename "${DATASETS[0]}")"
BASE_CKPT="/home/dyros/Data/jg_data/ckpts/${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_CKPT"

if [[ -f "$0" ]]; then
  cp "$0" "$BASE_CKPT/run_grid.sh"
fi

CAMERA_MODE="event"
CAMERA_MODES=("${CAMERA_MODE}")

for IMAGE_SIZE in 320; do
  for LR in 1e-5 2e-5; do
    for BS in 8 16 32; do
      for KL in 1 5 10; do
        for CAM_MODE in "${CAMERA_MODES[@]}"; do

          case "$CAM_MODE" in
            rgb)
              CAM_NAMES=(rgb)
              CAM_TAG="rgb_hist3"
              ;;
            event)
              CAM_NAMES=(event)
              CAM_TAG="event_3chef_hist3"
              ;;
            *)
              echo "Unsupported CAM_MODE: $CAM_MODE" >&2
              exit 1
              ;;
          esac

          RUN_NAME="intercept_delta_s_cam_${CAM_TAG}_hist_${VISUAL_HISTORY_FRAMES}_chunk_${CHUNK_SIZE}_lr_${LR}_bs_${BS}_kl_${KL}_imgsize_${IMAGE_SIZE}"
          CKPT_DIR="${BASE_CKPT}/${RUN_NAME}"
          mkdir -p "$CKPT_DIR"

          echo "=== Starting ${RUN_NAME} ==="
          echo "Checkpoint dir: ${CKPT_DIR}"
          echo "Datasets: ${DATASETS[*]}"
          echo "Cameras: ${CAM_NAMES[*]}"
          echo "Raw qpos dim: ${RAW_QPOS_DIM}"
          echo "Model state dim: ${STATE_DIM}"
          echo "Action dim: ${ACTION_DIM}"
          echo "RGB/qpos history offsets: [-6, -3, 0]"
          echo "Action: delta_s[k] = tcp_s(t+k+1) - tcp_s(t)"
          echo "Chunk size: ${CHUNK_SIZE}"
          echo "LR: ${LR}, BS: ${BS}, KL: ${KL}, IMAGE_SIZE: ${IMAGE_SIZE}"
          echo

          CMD=(
            python3 imitate_episodes.py
            --policy_class ACT
            --task_name real_franka_ball_intercept
            --ckpt_dir "$CKPT_DIR"
            --dataset_dirs "${DATASETS[@]}"
            --camera_names "${CAM_NAMES[@]}"

            --data_mode intercept
            --state_dim "$STATE_DIM"
            --action_dim "$ACTION_DIM"
            --visual_history_frames "$VISUAL_HISTORY_FRAMES"
            --no_use_bce_last_action_dim

            --batch_size "$BS"
            --num_epochs 5000
            --lr "$LR"
            --chunk_size "$CHUNK_SIZE"
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

echo "All delta-s interception runs finished."
echo "Results saved in: ${BASE_CKPT}"