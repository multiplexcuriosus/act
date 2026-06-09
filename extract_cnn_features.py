#!/usr/bin/env python3
import os
import argparse
import pickle
import h5py
import numpy as np
import torch

from policy import ACTPolicy


def load_policy(args, device):
    policy_config = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "nheads": args.nheads,
        "camera_names": args.camera_names,
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "use_bce_last_action_dim": args.use_bce_last_action_dim,
    }

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    ckpt_obj = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj

    policy = ACTPolicy(policy_config)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    stats_path = os.path.join(args.ckpt_dir, args.stats_name)
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    return policy, stats


def preprocess_qpos(qpos, stats):
    qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
    qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
    return (qpos - qpos_mean) / np.clip(qpos_std, 1e-6, None)


def load_image_from_hdf5_episode(f, camera_names, t):
    imgs = []

    for cam in camera_names:
        # Adjust these paths if your HDF5 schema differs.
        img = f[f"/observations/images/{cam}"][t]

        # Expected HDF5 image could be HWC uint8 or CHW.
        if img.ndim == 3 and img.shape[0] == 3:
            img_chw = img
        elif img.ndim == 3 and img.shape[2] == 3:
            img_chw = np.transpose(img, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected image shape for camera {cam}: {img.shape}")

        imgs.append(img_chw)

    image = np.stack(imgs, axis=0)              # [num_cam, 3, H, W]
    image = image.astype(np.float32) / 255.0
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--ckpt_name", default="policy_val_best.ckpt")
    parser.add_argument("--stats_name", default="dataset_stats.pkl")
    parser.add_argument("--episode_paths", nargs="+", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--camera_names", nargs="+", required=True)
    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)

    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--kl_weight", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--dim_feedforward", type=int, required=True)

    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--use_bce_last_action_dim", action="store_true", default=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, stats = load_policy(args, device)

    all_cnn = []
    all_decoder = []
    all_pred = []
    all_action_now = []
    all_episode_id = []
    all_timestep = []

    with torch.inference_mode():
        for ep_idx, ep_path in enumerate(args.episode_paths):
            with h5py.File(ep_path, "r") as f:
                qpos_arr = np.asarray(f["/observations/qpos"], dtype=np.float32)
                action_arr = np.asarray(f["/action"], dtype=np.float32)

                T = len(qpos_arr)

                for t in range(T):
                    qpos_np = preprocess_qpos(qpos_arr[t], stats)
                    image_np = load_image_from_hdf5_episode(f, args.camera_names, t)

                    qpos = torch.from_numpy(qpos_np).float().unsqueeze(0).to(device)
                    image = torch.from_numpy(image_np).float().unsqueeze(0).to(device)

                    a_hat, _, _, features = policy.extract_features(
                        qpos,
                        image,
                        actions=None,
                        is_pad=None,
                    )

                    all_cnn.append(features["cnn_proj_pooled"].squeeze(0).cpu().numpy())
                    all_decoder.append(features["decoder_hs"].squeeze(0).cpu().numpy())
                    all_pred.append(a_hat.squeeze(0).cpu().numpy())
                    all_action_now.append(action_arr[t])
                    all_episode_id.append(ep_idx)
                    all_timestep.append(t)

    np.savez(
        args.out,
        cnn_proj_pooled=np.stack(all_cnn, axis=0),
        decoder_hs=np.stack(all_decoder, axis=0),
        pred_action_chunk=np.stack(all_pred, axis=0),
        action_now=np.stack(all_action_now, axis=0),
        episode_id=np.asarray(all_episode_id, dtype=np.int64),
        timestep=np.asarray(all_timestep, dtype=np.int64),
        camera_names=np.asarray(args.camera_names),
    )

    print(f"Saved features to {args.out}")


if __name__ == "__main__":
    main()