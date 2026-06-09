#!/usr/bin/env python3
import os
import math
import argparse
import pickle
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from einops import rearrange

from policy import ACTPolicy


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_std(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(x, eps, None)


def pre_process_qpos(qpos: np.ndarray, qpos_mean: np.ndarray, qpos_std: np.ndarray) -> np.ndarray:
    return (qpos - qpos_mean) / safe_std(qpos_std)


def post_process_action(raw_action: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return raw_action * action_std + action_mean


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    pred, gt: [N, 2] in DENORMALIZED action space
    """
    assert pred.ndim == 2 and pred.shape[1] == 2
    assert gt.ndim == 2 and gt.shape[1] == 2

    err = pred - gt
    abs_err = np.abs(err)
    sq_err = err ** 2

    metrics = {}

    metrics["mse_x"] = float(np.mean(sq_err[:, 0]))
    metrics["mse_y"] = float(np.mean(sq_err[:, 1]))
    metrics["mae_x"] = float(np.mean(abs_err[:, 0]))
    metrics["mae_y"] = float(np.mean(abs_err[:, 1]))

    sign_pred_x = np.sign(pred[:, 0])
    sign_gt_x = np.sign(gt[:, 0])
    sign_pred_y = np.sign(pred[:, 1])
    sign_gt_y = np.sign(gt[:, 1])

    metrics["sign_acc_x"] = float(np.mean(sign_pred_x == sign_gt_x))
    metrics["sign_acc_y"] = float(np.mean(sign_pred_y == sign_gt_y))

    def safe_corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    metrics["corr_x"] = safe_corr(pred[:, 0], gt[:, 0])
    metrics["corr_y"] = safe_corr(pred[:, 1], gt[:, 1])

    metrics["mean_pred_x"] = float(np.mean(pred[:, 0]))
    metrics["mean_pred_y"] = float(np.mean(pred[:, 1]))
    metrics["mean_gt_x"] = float(np.mean(gt[:, 0]))
    metrics["mean_gt_y"] = float(np.mean(gt[:, 1]))

    return metrics


def save_scatter(pred: np.ndarray, gt: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(gt, pred, s=4, alpha=0.35)
    mn = min(np.min(gt), np.min(pred))
    mx = max(np.max(gt), np.max(pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("ground truth")
    plt.ylabel("prediction")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def print_metrics(name: str, metrics: Dict[str, float]):
    print(f"\n===== {name} =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            if math.isnan(v):
                print(f"{k:>16s}: nan")
            else:
                print(f"{k:>16s}: {v:.6f}")
        else:
            print(f"{k:>16s}: {v}")


def load_stats(stats_path: str):
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    required = ["qpos_mean", "qpos_std", "action_mean", "action_std"]
    for k in required:
        if k not in stats:
            raise KeyError(f'Missing key "{k}" in stats file: {stats_path}')

    qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
    qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats["action_std"], dtype=np.float32)

    print("Loaded stats:")
    print("  qpos_mean  :", qpos_mean)
    print("  qpos_std   :", qpos_std)
    print("  action_mean:", action_mean)
    print("  action_std :", action_std)

    return qpos_mean, qpos_std, action_mean, action_std


def load_episode_paths(dataset_dir: str, num_episodes: int):
    paths = []
    for i in range(num_episodes):
        p = os.path.join(dataset_dir, f"episode_{i}.hdf5")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing episode file: {p}")
        paths.append(p)
    return paths


def split_episode_paths(paths, val_ratio=0.2, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)

    n_val = max(1, int(round(len(paths) * val_ratio)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_paths = [paths[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    return train_paths, val_paths


def read_episode_steps(h5_path: str, image_key: str, qpos_key: str, action_key: str):
    with h5py.File(h5_path, "r") as f:
        images = f[image_key][()]   # [T, H, W, C] expected
        qpos = f[qpos_key][()]      # [T, D]
        actions = f[action_key][()] # [T, 2] or maybe [T, chunk, 2] but likely [T, 2]

    return images, qpos, actions


def build_policy_input_from_arrays(
    image_hwc: np.ndarray,
    qpos_row: np.ndarray,
    qpos_mean: np.ndarray,
    qpos_std: np.ndarray,
    device: torch.device,
):
    # match your player logic:
    # qpos = first two state entries only
    # image stays BGR / 255.0 and becomes [1, 1, C, H, W]
    qpos_numpy = np.asarray(qpos_row[:2], dtype=np.float32)
    qpos_norm = pre_process_qpos(qpos_numpy, qpos_mean, qpos_std)
    qpos_t = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)

    image_float = torch.from_numpy(image_hwc / 255.0).float()
    curr_image = rearrange(image_float, "h w c -> c h w").unsqueeze(0).unsqueeze(0).to(device)

    return qpos_t, curr_image


@torch.no_grad()
def eval_split(
    policy,
    episode_paths,
    qpos_mean,
    qpos_std,
    action_mean,
    action_std,
    device,
    image_key,
    qpos_key,
    action_key,
    max_episodes=None,
):
    pred_all = []
    gt_all = []

    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]

    for ep_idx, h5_path in enumerate(episode_paths):
        print(f"[eval] episode {ep_idx+1}/{len(episode_paths)}: {h5_path}", flush=True)
        images, qpos, actions = read_episode_steps(h5_path, image_key, qpos_key, action_key)

        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError(
                f"Expected actions [T, 2] in {h5_path}, got shape {actions.shape}"
            )

        T = len(actions)
        for t in range(T):
            if t % 50 == 0:
                print(f"  step {t}/{T}", flush=True)
            qpos_t, curr_image_t = build_policy_input_from_arrays(
                images[t], qpos[t], qpos_mean, qpos_std, device
            )

            all_actions = policy(qpos_t, curr_image_t)

            # match your player default debug/static behavior:
            # take first action of returned chunk
            raw_action = all_actions[:, 0]
            raw_action = raw_action.squeeze(0).detach().cpu().numpy()

            pred_action = post_process_action(raw_action, action_mean, action_std)
            gt_action = np.asarray(actions[t], dtype=np.float32)

            pred_all.append(pred_action)
            gt_all.append(gt_action)

    pred_all = np.asarray(pred_all, dtype=np.float32)
    gt_all = np.asarray(gt_all, dtype=np.float32)
    return pred_all, gt_all

def compute_label_distribution(episode_paths, qpos_key, action_key):
    all_actions = []
    all_qpos = []
    all_target = []

    for h5_path in episode_paths:
        with h5py.File(h5_path, "r") as f:
            actions = f[action_key][()]              # [T, 2]
            qpos = f[qpos_key][()]                  # [T, 2]
            target = f["/observations/target_pos"][()]  # [T, 2]

        all_actions.append(actions)
        all_qpos.append(qpos)
        all_target.append(target)

    actions = np.concatenate(all_actions, axis=0)
    qpos = np.concatenate(all_qpos, axis=0)
    target = np.concatenate(all_target, axis=0)

    dx = target[:, 0] - qpos[:, 0]
    dy = target[:, 1] - qpos[:, 1]

    ax = actions[:, 0]
    ay = actions[:, 1]

    def summarize(name, d, a):
        print(f"\n--- {name} ---")
        print(f"{name} > 0: {(d > 0).mean():.3f}")
        print(f"{name} < 0: {(d < 0).mean():.3f}")
        print(f"action_{name[-1]} > 0: {(a > 0).mean():.3f}")
        print(f"action_{name[-1]} < 0: {(a < 0).mean():.3f}")
        print(f"mean {name}: {d.mean():.3f}")
        print(f"mean action_{name[-1]}: {a.mean():.3f}")

    summarize("dx", dx, ax)
    summarize("dy", dy, ay)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)

    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="policy_val_best.ckpt")
    parser.add_argument("--stats_name", type=str, default="dataset_stats.pkl")

    parser.add_argument("--out_dir", type=str, default="eval_action_dims_out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_train_episodes", type=int, default=None)
    parser.add_argument("--max_val_episodes", type=int, default=None)

    # policy config: aligned with your player file
    parser.add_argument("--state_dim", type=int, default=2)
    parser.add_argument("--camera_name", type=str, default="toy")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--kl_weight", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--dim_feedforward", type=int, required=True)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)

    # HDF5 dataset keys: likely need only these adjusted if your file layout differs
    parser.add_argument("--image_key", type=str, default="/observations/images/main")
    parser.add_argument("--qpos_key", type=str, default="/observations/qpos")
    parser.add_argument("--action_key", type=str, default="/action")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    stats_path = os.path.join(args.ckpt_dir, args.stats_name)

    qpos_mean, qpos_std, action_mean, action_std = load_stats(stats_path)

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
        "camera_names": [args.camera_name],
        "state_dim": args.state_dim,
    }

    policy = ACTPolicy(policy_config)
    load_status = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Checkpoint load status: {load_status}")
    policy.to(device)
    policy.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    episode_paths = load_episode_paths(args.dataset_dir, args.num_episodes)
    train_paths, val_paths = split_episode_paths(
        episode_paths, val_ratio=args.val_ratio, seed=args.seed
    )

    print(f"Train episodes: {len(train_paths)}")
    print(f"Val episodes  : {len(val_paths)}")

    pred_train, gt_train = eval_split(
        policy=policy,
        episode_paths=train_paths,
        qpos_mean=qpos_mean,
        qpos_std=qpos_std,
        action_mean=action_mean,
        action_std=action_std,
        device=device,
        image_key=args.image_key,
        qpos_key=args.qpos_key,
        action_key=args.action_key,
        max_episodes=args.max_train_episodes,
    )

    pred_val, gt_val = eval_split(
        policy=policy,
        episode_paths=val_paths,
        qpos_mean=qpos_mean,
        qpos_std=qpos_std,
        action_mean=action_mean,
        action_std=action_std,
        device=device,
        image_key=args.image_key,
        qpos_key=args.qpos_key,
        action_key=args.action_key,
        max_episodes=args.max_val_episodes,
    )

    train_metrics = compute_metrics(pred_train, gt_train)
    val_metrics = compute_metrics(pred_val, gt_val)

    print_metrics("TRAIN", train_metrics)
    print_metrics("VAL", val_metrics)

    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write("===== TRAIN =====\n")
        for k, v in train_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n===== VAL =====\n")
        for k, v in val_metrics.items():
            f.write(f"{k}: {v}\n")

    save_scatter(
        pred_train[:, 0], gt_train[:, 0],
        os.path.join(args.out_dir, "scatter_train_x.png"),
        "Train: pred_x vs gt_x"
    )
    save_scatter(
        pred_train[:, 1], gt_train[:, 1],
        os.path.join(args.out_dir, "scatter_train_y.png"),
        "Train: pred_y vs gt_y"
    )
    save_scatter(
        pred_val[:, 0], gt_val[:, 0],
        os.path.join(args.out_dir, "scatter_val_x.png"),
        "Val: pred_x vs gt_x"
    )
    save_scatter(
        pred_val[:, 1], gt_val[:, 1],
        os.path.join(args.out_dir, "scatter_val_y.png"),
        "Val: pred_y vs gt_y"
    )

    print("\n===== TRAIN LABEL DISTRIBUTION =====")
    compute_label_distribution(train_paths, args.qpos_key, args.action_key)

    print("\n===== VAL LABEL DISTRIBUTION =====")
    compute_label_distribution(val_paths, args.qpos_key, args.action_key)

    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()