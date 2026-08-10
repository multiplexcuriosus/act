#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 DATASET_DIR CHECKPOINT_DIR" >&2
  exit 2
fi
python3 imitate_episodes.py \
  --data_mode intercept --dataset_dir "$1" --ckpt_dir "$2" \
  --task_name ball_interception --policy_class ACT --camera_names sparse_ball \
  --action_dim 1 --state_dim 21 --chunk_size 30 --visual_history_frames 3 \
  --no_use_bce_last_action_dim --kl_weight 10 --hidden_dim 512 \
  --dim_feedforward 3200 --batch_size 8 --seed 0 --num_epochs 2000 --lr 2e-5
