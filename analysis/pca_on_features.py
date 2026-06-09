#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


DEFAULT_RGB_PATH = Path("/home/jau/dyros/data/features/rgb_cnn_features.npz")
DEFAULT_EVENT_PATH = Path("/home/jau/dyros/data/features/event_cnn_features.npz")
FIRST_N = 40


def print_basic_stats(name, arr):
    print(f"\n{name}")
    print(f"  shape: {arr.shape}")
    print(f"  mean:  {np.mean(arr):.6f}")
    print(f"  std:   {np.std(arr):.6f}")
    print(f"  min:   {np.min(arr):.6f}")
    print(f"  max:   {np.max(arr):.6f}")


def safe_spearman(x, y):
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan
    return spearmanr(x, y)


def pairwise_feature_action_corr(X, A):
    if len(X) < 3:
        return np.nan, np.nan

    feature_dists = pdist(X, metric="euclidean")
    action_dists = pdist(A, metric="euclidean")
    return safe_spearman(feature_dists, action_dists)


def velocity_corr(X, A):
    if len(X) < 3:
        return np.nan, np.nan

    dX = np.linalg.norm(np.diff(X, axis=0), axis=1)
    dA = np.linalg.norm(np.diff(A, axis=0), axis=1)
    return safe_spearman(dX, dA)


def sort_loaded_data(d):
    X = d["cnn_proj_pooled"]
    t = d["timestep"]

    # Prefer episode-aware sorting if available.
    # This avoids mixing episode 0 timestep 1 with episode 1 timestep 1.
    if "episode_id" in d:
        episode_id = d["episode_id"]
        order = np.lexsort((t, episode_id))
    else:
        episode_id = np.zeros_like(t)
        order = np.argsort(t)

    out = {
        "X": X[order],
        "t": t[order],
        "episode_id": episode_id[order],
    }

    if "action_now" in d:
        out["A"] = d["action_now"][order]

    if "pred_action_chunk" in d:
        out["P"] = d["pred_action_chunk"][order]

    return out


def load_modality(path, label):
    d = np.load(path)
    data = sort_loaded_data(d)
    data["label"] = label
    data["path"] = path

    print(f"\n{'=' * 80}")
    print(f"Loaded {label}: {path}")
    print_basic_stats(f"{label} cnn_proj_pooled", data["X"])

    norms = np.linalg.norm(data["X"], axis=1)
    print_basic_stats(f"{label} feature norm ||h_t||", norms)

    dX = compute_feature_velocity_episodewise(data["X"], data["episode_id"])
    print_basic_stats(f"{label} feature velocity ||h_t - h_(t-1)||", dX)

    return data


def compute_feature_velocity_episodewise(X, episode_id):
    values = []
    for ep in np.unique(episode_id):
        idx = np.where(episode_id == ep)[0]
        if len(idx) < 2:
            continue
        values.append(np.linalg.norm(np.diff(X[idx], axis=0), axis=1))

    if not values:
        return np.array([])
    return np.concatenate(values, axis=0)


def compute_action_velocity_episodewise(A, episode_id):
    values = []
    for ep in np.unique(episode_id):
        idx = np.where(episode_id == ep)[0]
        if len(idx) < 2:
            continue
        values.append(np.linalg.norm(np.diff(A[idx], axis=0), axis=1))

    if not values:
        return np.array([])
    return np.concatenate(values, axis=0)


def compute_velocity_pairs_episodewise(X, A, episode_id):
    dX_all = []
    dA_all = []
    t_next_all = []

    for ep in np.unique(episode_id):
        idx = np.where(episode_id == ep)[0]
        if len(idx) < 2:
            continue

        dX = np.linalg.norm(np.diff(X[idx], axis=0), axis=1)
        dA = np.linalg.norm(np.diff(A[idx], axis=0), axis=1)

        dX_all.append(dX)
        dA_all.append(dA)
        t_next_all.append(idx[1:])

    if not dX_all:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(dX_all, axis=0),
        np.concatenate(dA_all, axis=0),
        np.concatenate(t_next_all, axis=0),
    )


def run_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X2 = pca.fit_transform(X)
    return pca, X2


def print_pca_stats(label, pca):
    print(f"\n{label} PCA")
    print("  Explained variance ratio:", pca.explained_variance_ratio_)
    print("  Total explained variance:", pca.explained_variance_ratio_.sum())


def plot_pca_side_by_side(rgb, event, first_n=None):
    if first_n is None:
        title_suffix = "all timesteps"
        rgb_mask = np.ones(len(rgb["X"]), dtype=bool)
        event_mask = np.ones(len(event["X"]), dtype=bool)
    else:
        title_suffix = f"first {first_n} timesteps"
        rgb_mask = rgb["t"] < first_n
        event_mask = event["t"] < first_n

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=False, sharey=False)

    for ax, data, mask, name in [
        (axes[0], rgb, rgb_mask, "RGB"),
        (axes[1], event, event_mask, "Event"),
    ]:
        X = data["X"][mask]
        t = data["t"][mask]

        if len(X) < 3:
            ax.set_title(f"{name}: not enough samples")
            continue

        pca, X2 = run_pca(X)
        print_pca_stats(f"{name} {title_suffix}", pca)

        ax.plot(X2[:, 0], X2[:, 1], linewidth=0.8, alpha=0.5)
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=t, s=12)
        ax.set_title(
            f"{name} PCA, {title_suffix}\n"
            f"PC1+PC2={pca.explained_variance_ratio_.sum():.3f}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label="timestep")

    fig.tight_layout()
    plt.show()


def plot_pca_colored_by_action_side_by_side(rgb, event):
    if "A" not in rgb or "A" not in event:
        print("\nSkipping PCA colored by action magnitude: missing action_now.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=False, sharey=False)

    for ax, data, name in [
        (axes[0], rgb, "RGB"),
        (axes[1], event, "Event"),
    ]:
        X = data["X"]
        A = data["A"]
        A_cont = A[:, :6] if A.shape[1] >= 7 else A
        action_mag = np.linalg.norm(A_cont, axis=1)

        pca, X2 = run_pca(X)

        ax.plot(X2[:, 0], X2[:, 1], linewidth=0.8, alpha=0.4)
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=action_mag, s=12)
        ax.set_title(f"{name} PCA colored by ||continuous action||")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label=r"$||a_{0:6}||$")

    fig.tight_layout()
    plt.show()


def plot_pca_colored_by_gripper_side_by_side(rgb, event):
    if "A" not in rgb or "A" not in event:
        print("\nSkipping PCA colored by gripper: missing action_now.")
        return

    if rgb["A"].shape[1] < 7 or event["A"].shape[1] < 7:
        print("\nSkipping PCA colored by gripper: action has fewer than 7 dims.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=False, sharey=False)

    for ax, data, name in [
        (axes[0], rgb, "RGB"),
        (axes[1], event, "Event"),
    ]:
        X = data["X"]
        gripper = data["A"][:, 6]

        pca, X2 = run_pca(X)

        ax.plot(X2[:, 0], X2[:, 1], linewidth=0.8, alpha=0.4)
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=gripper, s=12)
        ax.set_title(f"{name} PCA colored by gripper action")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label="gripper")

    fig.tight_layout()
    plt.show()


def plot_norms_same_axis(rgb, event):
    rgb_norm = np.linalg.norm(rgb["X"], axis=1)
    event_norm = np.linalg.norm(event["X"], axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(rgb["t"], rgb_norm, label="RGB", linewidth=1.0)
    plt.plot(event["t"], event_norm, label="Event", linewidth=1.0)
    plt.xlabel("timestep")
    plt.ylabel(r"$||h_t||$")
    plt.title("CNN feature norm over time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_velocity_same_axis(rgb, event):
    plt.figure(figsize=(8, 4))

    for data, name in [(rgb, "RGB"), (event, "Event")]:
        X = data["X"]
        t = data["t"]
        episode_id = data["episode_id"]

        for ep in np.unique(episode_id):
            idx = np.where(episode_id == ep)[0]
            if len(idx) < 2:
                continue

            dX = np.linalg.norm(np.diff(X[idx], axis=0), axis=1)
            label = name if ep == np.unique(episode_id)[0] else None
            plt.plot(t[idx][1:], dX, label=label, linewidth=1.0)

    plt.xlabel("timestep")
    plt.ylabel(r"$||h_t - h_{t-1}||$")
    plt.title("CNN feature velocity over time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_action_velocity_once(data):
    if "A" not in data:
        print("\nSkipping action velocity: missing action_now.")
        return

    A = data["A"]
    A_cont = A[:, :6] if A.shape[1] >= 7 else A
    t = data["t"]
    episode_id = data["episode_id"]

    plt.figure(figsize=(8, 4))

    for ep in np.unique(episode_id):
        idx = np.where(episode_id == ep)[0]
        if len(idx) < 2:
            continue

        dA = np.linalg.norm(np.diff(A_cont[idx], axis=0), axis=1)
        label = "continuous action" if ep == np.unique(episode_id)[0] else None
        plt.plot(t[idx][1:], dA, label=label, linewidth=1.0)

    plt.xlabel("timestep")
    plt.ylabel(r"$||a_t - a_{t-1}||$")
    plt.title("Action velocity over time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_vs_action_velocity_side_by_side(rgb, event):
    if "A" not in rgb or "A" not in event:
        print("\nSkipping feature/action velocity scatter: missing action_now.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, data, name in [
        (axes[0], rgb, "RGB"),
        (axes[1], event, "Event"),
    ]:
        A = data["A"]
        A_cont = A[:, :6] if A.shape[1] >= 7 else A

        dX, dA, _ = compute_velocity_pairs_episodewise(
            data["X"],
            A_cont,
            data["episode_id"],
        )

        rho, p = safe_spearman(dX, dA)

        ax.scatter(dX, dA, s=12)
        ax.set_xlabel(r"feature velocity $||h_t - h_{t-1}||$")
        ax.set_ylabel(r"action velocity $||a_t - a_{t-1}||$")
        ax.set_title(f"{name}: rho={rho:.3f}, p={p:.2e}")

    fig.suptitle("Feature velocity vs action velocity")
    fig.tight_layout()
    plt.show()


def plot_cumulative_pca_variance(rgb, event, max_components=50):
    plt.figure(figsize=(8, 4))

    for data, name in [(rgb, "RGB"), (event, "Event")]:
        X = data["X"]
        n_components = min(max_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        plt.plot(
            np.arange(1, n_components + 1),
            cumvar,
            label=name,
            linewidth=1.0,
        )

        print(f"\n{name} cumulative PCA variance")
        for k in [2, 5, 10, 20, 50]:
            if k <= n_components:
                print(f"  {k:>3} PCs explain {cumvar[k - 1]:.6f}")

    plt.xlabel("number of principal components")
    plt.ylabel("cumulative explained variance")
    plt.title("Cumulative PCA explained variance")
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_duplicate_feature_diffs(data):
    X = data["X"]
    episode_id = data["episode_id"]
    label = data["label"]

    zero_count = 0
    near_zero_count = 0
    total = 0

    for ep in np.unique(episode_id):
        idx = np.where(episode_id == ep)[0]
        if len(idx) < 2:
            continue
        dX = np.linalg.norm(np.diff(X[idx], axis=0), axis=1)
        zero_count += int(np.sum(dX < 1e-8))
        near_zero_count += int(np.sum(dX < 1e-4))
        total += len(dX)

    print(f"\n{label} duplicate/near-duplicate feature diffs")
    print(f"  zero diffs:      {zero_count} / {total}")
    print(f"  near-zero diffs: {near_zero_count} / {total}")


def print_action_consistency(rgb, event):
    if "A" not in rgb or "A" not in event:
        return

    if rgb["A"].shape != event["A"].shape:
        print("\nWARNING: RGB and event action_now shapes differ.")
        print("  RGB:", rgb["A"].shape)
        print("  Event:", event["A"].shape)
        return

    max_abs_diff = np.max(np.abs(rgb["A"] - event["A"]))
    print(f"\nAction consistency between RGB/Event files")
    print(f"  max |A_rgb - A_event|: {max_abs_diff:.9f}")

    if max_abs_diff > 1e-6:
        print("  WARNING: action_now differs between files. They may not come from the exact same HDF5 ordering.")


def print_timestep_consistency(rgb, event):
    same_t = np.array_equal(rgb["t"], event["t"])
    same_ep = np.array_equal(rgb["episode_id"], event["episode_id"])

    print(f"\nTimestep/episode consistency")
    print(f"  same timestep array:  {same_t}")
    print(f"  same episode_id array: {same_ep}")

    if not same_t or not same_ep:
        print("  WARNING: RGB/Event files are not aligned after sorting.")


def print_all_correlations(data, first_n):
    label = data["label"]
    X = data["X"]

    if "A" not in data:
        print(f"\n{label}: no action_now found. Skipping action correlations.")
        return

    A = data["A"]
    A_cont = A[:, :6] if A.shape[1] >= 7 else A

    print(f"\n{'-' * 80}")
    print(f"{label} action correlations")

    rho, p = pairwise_feature_action_corr(X, A)
    print(f"All timesteps, full action: rho={rho:.6f}, p={p:.6e}")

    rho, p = pairwise_feature_action_corr(X, A_cont)
    print(f"All timesteps, continuous action dims only: rho={rho:.6f}, p={p:.6e}")

    dX, dA, _ = compute_velocity_pairs_episodewise(X, A_cont, data["episode_id"])
    rho, p = safe_spearman(dX, dA)
    print(f"All timesteps, feature velocity vs action velocity: rho={rho:.6f}, p={p:.6e}")

    mask_first = data["t"] < first_n
    if np.sum(mask_first) >= 3:
        rho, p = pairwise_feature_action_corr(X[mask_first], A[mask_first])
        print(f"First {first_n}, full action: rho={rho:.6f}, p={p:.6e}")

        rho, p = pairwise_feature_action_corr(X[mask_first], A_cont[mask_first])
        print(f"First {first_n}, continuous action dims only: rho={rho:.6f}, p={p:.6e}")

        rho, p = velocity_corr(X[mask_first], A_cont[mask_first])
        print(f"First {first_n}, feature velocity vs action velocity: rho={rho:.6f}, p={p:.6e}")

    if "P" in data:
        P = data["P"]
        if P.ndim == 3:
            P_now = P[:, 0, :]
        else:
            P_now = P

        P_cont = P_now[:, :6] if P_now.shape[1] >= 7 else P_now

        rho, p = pairwise_feature_action_corr(X, P_now)
        print(f"All timesteps, predicted full action: rho={rho:.6f}, p={p:.6e}")

        rho, p = pairwise_feature_action_corr(X, P_cont)
        print(f"All timesteps, predicted continuous action dims only: rho={rho:.6f}, p={p:.6e}")


def print_pc_correlations(data):
    label = data["label"]

    if "A" not in data:
        return

    X = data["X"]
    t = data["t"]
    A = data["A"]
    A_cont = A[:, :6] if A.shape[1] >= 7 else A
    action_mag = np.linalg.norm(A_cont, axis=1)

    pca, X2 = run_pca(X)
    pc1 = X2[:, 0]
    pc2 = X2[:, 1]
    feature_norm = np.linalg.norm(X, axis=1)

    variables = [
        ("timestep", t),
        ("action_mag", action_mag),
        ("feature_norm", feature_norm),
    ]

    if A.shape[1] >= 7:
        variables.append(("gripper", A[:, 6]))

    print(f"\n{label} PC correlation checks")
    for name, y in variables:
        rho1, p1 = safe_spearman(pc1, y)
        rho2, p2 = safe_spearman(pc2, y)
        print(f"  corr(PC1, {name}) = {rho1:.3f}, p={p1:.3e}")
        print(f"  corr(PC2, {name}) = {rho2:.3f}, p={p2:.3e}")


def print_top10_pca_variance(data):
    label = data["label"]
    X = data["X"]

    n_components = min(10, X.shape[0], X.shape[1])
    if n_components < 1:
        print(f"\n{label}: not enough samples for PCA.")
        return

    pca_full = PCA(n_components=n_components)
    pca_full.fit(X)

    print(f"\n{label} explained variance ratio first {n_components} PCs:")
    for i, v in enumerate(pca_full.explained_variance_ratio_, start=1):
        print(f"PC{i}: {v:.4f}")

    print("Cumulative:")
    print(np.cumsum(pca_full.explained_variance_ratio_))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB_PATH)
    parser.add_argument("--event", type=Path, default=DEFAULT_EVENT_PATH)
    parser.add_argument("--first_n", type=int, default=FIRST_N)
    args = parser.parse_args()

    rgb = load_modality(args.rgb, "RGB")
    event = load_modality(args.event, "Event")

    print_timestep_consistency(rgb, event)
    print_action_consistency(rgb, event)

    print_duplicate_feature_diffs(rgb)
    print_duplicate_feature_diffs(event)

    print_all_correlations(rgb, args.first_n)
    print_all_correlations(event, args.first_n)

    print_pc_correlations(rgb)
    print_pc_correlations(event)
    print_top10_pca_variance(rgb)
    print_top10_pca_variance(event)

    # Plots from previous script, now RGB/Event compared.
    plot_pca_side_by_side(rgb, event, first_n=None)
    plot_pca_side_by_side(rgb, event, first_n=args.first_n)

    plot_pca_colored_by_action_side_by_side(rgb, event)
    plot_pca_colored_by_gripper_side_by_side(rgb, event)

    plot_norms_same_axis(rgb, event)
    plot_feature_velocity_same_axis(rgb, event)

    # Same for RGB/event because both files should come from same HDF5.
    plot_action_velocity_once(rgb)

    plot_feature_vs_action_velocity_side_by_side(rgb, event)
    plot_cumulative_pca_variance(rgb, event)


if __name__ == "__main__":
    main()