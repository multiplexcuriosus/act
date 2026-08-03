#!/usr/bin/env python3
import argparse
import os
import pickle

import h5py
import numpy as np
import torch

from intercept_rollout_contract import (
    INTERCEPT_HISTORY_OFFSETS,
    absolute_s_from_anchor,
    build_visual_history_tensor,
    compute_history_indices,
    denormalize_delta_chunk,
    validate_intercept_stats_and_config,
)
from policy import ACTPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--camera_name", default="rgb", choices=["rgb", "event"])
    ap.add_argument("--chunk_size", type=int, default=30)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dim_feedforward", type=int, default=3200)
    ap.add_argument("--kl_weight", type=int, default=10)
    ap.add_argument("--state_dim", type=int, default=21)
    ap.add_argument("--action_dim", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    if args.state_dim != 21:
        raise ValueError(f"Interception checkpoint expects state_dim=21, got {args.state_dim}")
    if args.action_dim != 1:
        raise ValueError(f"Interception checkpoint expects action_dim=1, got {args.action_dim}")
    if args.chunk_size != 30:
        raise ValueError(f"Interception checkpoint expects chunk_size=30, got {args.chunk_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": [args.camera_name],
        "input_modality": args.camera_name,
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "use_bce_last_action_dim": False,
        "rgb_history_frames": 3,
        "visual_history_frames": 3,
        "visual_history_offsets": list(INTERCEPT_HISTORY_OFFSETS),
        "channels_per_visual_frame": 3,
        "visual_frame_order": "oldest_to_newest",
        "image_normalization": "shifted_3chef_centered" if args.camera_name == "event" else "imagenet",
        "image_channels": 9,
        "image_size": args.image_size,
    }

    policy = ACTPolicy(cfg)
    policy.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "policy_val_best.ckpt"), map_location=device))
    policy.to(device).eval()

    with open(os.path.join(args.ckpt_dir, "dataset_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    stats_arrays = validate_intercept_stats_and_config(stats, cfg, expected_chunk_size=args.chunk_size)

    with h5py.File(args.hdf5, "r") as f:
        rgb = np.asarray(f[f"/observations/images/{args.camera_name}"][()], dtype=np.uint8)
        qpos = np.asarray(f["/observations/qpos"][()], dtype=np.float32)
        tcp_s_abs = np.asarray(f["/action"][()], dtype=np.float32).reshape(-1)

    print("t | anchor_s | pred_abs[0] | gt_abs(t+1) | pred_abs[-1]")
    with torch.inference_mode():
        max_t = min(len(tcp_s_abs) - 1, 120)
        for t in range(6, max_t):
            history_indices = compute_history_indices(t, INTERCEPT_HISTORY_OFFSETS)

            q_hist = qpos[history_indices].reshape(-1)
            q_norm = (q_hist - stats_arrays["qpos_mean"]) / stats_arrays["qpos_std"]
            q_tensor = torch.from_numpy(q_norm).float().to(device).unsqueeze(0)

            rgb_frames = [rgb[index] for index in history_indices]
            image_np = build_visual_history_tensor(rgb_frames, args.image_size, modality=args.camera_name)
            image_tensor = torch.from_numpy(image_np).float().to(device)

            raw = policy(q_tensor, image_tensor)
            if tuple(raw.shape) != (1, args.chunk_size, 1):
                raise RuntimeError(
                    f"Policy output shape mismatch: expected {(1, args.chunk_size, 1)}, got {tuple(raw.shape)}"
                )

            norm_delta = raw[0, :, 0].detach().cpu().numpy()
            delta_s = denormalize_delta_chunk(norm_delta, stats_arrays["action_mean"], stats_arrays["action_std"])
            absolute_s = absolute_s_from_anchor(float(tcp_s_abs[t]), delta_s)

            gt_next = float(tcp_s_abs[t + 1])
            if t % 5 == 0:
                print(
                    f"{t:03d} | "
                    f"{tcp_s_abs[t]:+.5f} | "
                    f"{absolute_s[0]:+.5f} | "
                    f"{gt_next:+.5f} | "
                    f"{absolute_s[-1]:+.5f}"
                )

if __name__ == "__main__":
    main()