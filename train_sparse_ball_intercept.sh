#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ( "$3" != "rgb" && "$3" != "event" ) ]]; then
    echo "usage: $0 DATASET_DIR CHECKPOINT_DIR rgb|event" >&2
    exit 2
fi

python3 imitate_episodes.py \
    --dataset_dir "$1" \
    --ckpt_dir "$2" \
    --sparse_source "$3" \
    --input_modality sparse_ball \
    --camera_names sparse_ball \
    --sparse_feature_dim 4 \
    --sparse_history_length 3 \
    --max_observation_age_sec 0.10 \
    --data_mode intercept \
    --policy_class ACT \
    --task_name real_intercept \
    --batch_size 8 \
    --seed 0 \
    --num_epochs 5000 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 30 \
    --hidden_dim 512 \
    --dim_feedforward 3200
