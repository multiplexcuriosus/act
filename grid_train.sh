#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
/home/dyros/Data/jg_data/hdf5/recording_20260619_161711_oval-from-circle-and-hex-into-oval
/home/dyros/Data/jg_data/hdf5/recording_20260619_170405_oval-from-hex-and-circle-into-oval
)

BASE_CKPT="/home/dyros/Data/jg_data/ckpts/grid_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_CKPT"

# Save this script into the run folder for reproducibility.
# This only works if the script is run from a file, not pasted through stdin.
if [[ -f "$0" ]]; then
  cp "$0" "$BASE_CKPT/run_grid.sh"
fi

# Camera modes to train in one grid run.
# NOTE:
#   event     -> --camera_names event
#   rgb       -> --camera_names rgb
#   rgb_event -> --camera_names rgb event
#
# Do not use "event rgb"; model camera slots depend on this order.
CAMERA_MODES=(rgb event rgb_event)

for IMAGE_SIZE in 320; do
  for LR in 2e-5; do
    for BS in 8; do
      for KL in 10; do
        for CAM_MODE in "${CAMERA_MODES[@]}"; do

          case "$CAM_MODE" in
            event)
              CAM_NAMES=(event)
              CAM_TAG="event"
              USE_EVENT=true
              ;;
            rgb)
              CAM_NAMES=(rgb)
              CAM_TAG="rgb"
              USE_EVENT=false
              ;;
            rgb_event)
              CAM_NAMES=(rgb event)
              CAM_TAG="rgb_event"
              USE_EVENT=true
              ;;
            *)
              echo "Unknown CAM_MODE: $CAM_MODE" >&2
              exit 1
              ;;
          esac

          RUN_NAME="cam_${CAM_TAG}_lr_${LR}_bs_${BS}_kl_${KL}_imgsize_${IMAGE_SIZE}"
          CKPT_DIR="${BASE_CKPT}/${RUN_NAME}"
          mkdir -p "$CKPT_DIR"

          echo "=== Starting ${RUN_NAME} ==="
          echo "Checkpoint dir: ${CKPT_DIR}"
          echo "Cameras: ${CAM_NAMES[*]}"
          echo "LR: ${LR}, BS: ${BS}, KL: ${KL}, IMAGE_SIZE: ${IMAGE_SIZE}"
          echo

          CMD=(
            python imitate_episodes.py
            --policy_class ACT
            --task_name real_franka_ball_ring
            --ckpt_dir "$CKPT_DIR"
            --dataset_dirs "${DATASETS[@]}"
            --camera_names "${CAM_NAMES[@]}"
            --data_mode joint
            --state_dim 8
            --action_dim 7
            --batch_size "$BS"
            --num_epochs 20000
            --lr "$LR"
            --chunk_size 30
            --hidden_dim 256
            --dim_feedforward 3200
            --kl_weight "$KL"
            --seed 0
            --checkpoint_interval 5000
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

echo "All grid runs finished."
echo "Results saved in: ${BASE_CKPT}"
