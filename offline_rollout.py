#!/usr/bin/env python3
import os, pickle, argparse, h5py
import numpy as np
import torch
from policy import ACTPolicy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--camera_name", default="rgb")
    ap.add_argument("--chunk_size", type=int, default=30)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dim_feedforward", type=int, default=3200)
    ap.add_argument("--kl_weight", type=int, default=10)
    ap.add_argument("--state_dim", type=int, default=8)
    ap.add_argument("--action_dim", type=int, default=7)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

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
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "use_bce_last_action_dim": True,
    }

    policy = ACTPolicy(cfg)
    policy.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "policy_val_best.ckpt"), map_location=device))
    policy.to(device).eval()

    with open(os.path.join(args.ckpt_dir, "dataset_stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    with h5py.File(args.hdf5, "r") as f:
        imgs = f[f"/observations/images/{args.camera_name}"][:]
        qpos = f["/observations/qpos"][:]
        gt_action = f["/action"][:]

    qpos_mean = stats["qpos_mean"]
    qpos_std = np.clip(stats["qpos_std"], 1e-6, None)
    action_mean = stats["action_mean"]
    action_std = stats["action_std"]

    all_actions = None
    print("t | pred_denorm[:3] | gt[:3] | pred_grip_prob | gt_grip")
    with torch.inference_mode():
        for t in range(min(len(qpos), 120)):
            q = (qpos[t] - qpos_mean) / qpos_std
            q = torch.from_numpy(q).float().to(device).unsqueeze(0)

            img = torch.from_numpy(imgs[t] / 255.0).float()
            img = img.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)

            if t % args.chunk_size == 0:
                all_actions = policy(q, img)

            raw = all_actions[:, t % args.chunk_size].squeeze(0).cpu().numpy()
            denorm = raw * action_std + action_mean
            grip_prob = 1.0 / (1.0 + np.exp(-raw[-1]))

            if t % 5 == 0:
                print(
                    f"{t:03d} | "
                    f"{denorm[:3]} | "
                    f"{gt_action[t, :3]} | "
                    f"{grip_prob:.3f} | "
                    f"{gt_action[t, -1]:.1f}"
                )

if __name__ == "__main__":
    main()