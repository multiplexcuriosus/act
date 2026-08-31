#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET_DIRS=(
  "/home/dyros/Data/jg_data/hdf5/bumpy/recording_20260824_150734_bumpybase1_hdf5_mwindow/"
  "/home/dyros/Data/jg_data/hdf5/bumpy/recording_20260824_194237_bumpybase2_hdf5_mwindow/"
)

POLICY_RATE_HZ=60
HISTORY_LENGTH=3
STATE_DIM=21

RUN_ROOT="${RUN_ROOT:-/home/dyros/Data/jg_data/ckpts/intercept_sparse_legacy_bumpy_$(date +%Y%m%d_%H%M%S)}"

for DATASET_DIR in "${DATASET_DIRS[@]}"; do
  if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Dataset directory does not exist: $DATASET_DIR" >&2
    exit 1
  fi
done

mkdir -p "$RUN_ROOT"

cp -- "$SCRIPT_PATH" "$RUN_ROOT/train_sparse_ball_bumpy_legacy3shot.sh"

for SPARSE_SOURCE in rgb event; do
  for LR in 1e-5; do
    for BS in 8; do
      for KL in 1; do

        RUN_NAME="sparse_${SPARSE_SOURCE}_uvnorm_valid_age_legacy3shot_${POLICY_RATE_HZ}hz_lr${LR}_bs${BS}_kl${KL}"
        CKPT_DIR="${RUN_ROOT}/${SPARSE_SOURCE}/${RUN_NAME}"

        mkdir -p "$CKPT_DIR"

        echo
        echo "============================================================"
        echo "Starting: ${RUN_NAME}"
        echo "Checkpoint directory: ${CKPT_DIR}"
        echo "Training data: bumpybase1 + bumpybase2"
        echo "History: legacy 3 snapshots (~[-200, -100, 0] ms)"
        echo "============================================================"

        CMD=(
          "$PYTHON_BIN" imitate_episodes.py

          --policy_class ACT
          --task_name real_intercept
          --data_mode intercept

          --dataset_dirs "${DATASET_DIRS[@]}"
          --ckpt_dir "$CKPT_DIR"

          --camera_names sparse_ball
          --input_modality sparse_ball
          --sparse_source "$SPARSE_SOURCE"
          --sparse_feature_dim 4

          # Legacy mode is the default when m_window is not specified.
          --sparse_history_length "$HISTORY_LENGTH"
          --max_observation_age_sec 0.10

          --policy_rate_hz "$POLICY_RATE_HZ"

          --state_dim "$STATE_DIM"
          --action_dim 1
          --chunk_size "$POLICY_RATE_HZ"

          --hidden_dim 512
          --dim_feedforward 3200

          --batch_size "$BS"
          --num_epochs 5000
          --lr "$LR"
          --kl_weight "$KL"
          --seed 0

          --no_use_bce_last_action_dim

          --checkpoint_interval 1000
          --save_extra_checkpoints
          --profile_memory
        )

        "${CMD[@]}" 2>&1 | tee "${CKPT_DIR}/train.log"

        echo "Finished: ${RUN_NAME}"
      done
    done
  done
done

echo
echo "All bumpy-trained legacy 3-snapshot RGB/event runs completed."
echo "Results: ${RUN_ROOT}"