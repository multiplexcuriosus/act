import numpy as np
import torch
import os
import glob
import re
import h5py
import cv2
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

import IPython
e = IPython.embed

color_transform = transforms.ColorJitter(brightness=0.5,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1,
                          )

DEFAULT_IMAGE_SIZE = (320, 320)


def canonical_image_size(image_size=None):
    if image_size is None:
        return DEFAULT_IMAGE_SIZE

    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        return (image_size, image_size)

    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        h = int(image_size[0])
        w = int(image_size[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"image_size must contain positive values, got {image_size}")
        return (h, w)

    raise ValueError(f"Unsupported image_size={image_size}. Expected None, int, or (h, w)")


DEFAULT_JOINT_DATA_CONFIG = {
    "qpos_dim": None,
    "action_dim": None,
    "qpos_indices": None,
    "action_indices": None,
    "action_key": "/action",
}


def _build_joint_data_config(
    qpos_dim=None,
    action_dim=None,
    qpos_indices=None,
    action_indices=None,
    action_key=None,
):
    cfg = dict(DEFAULT_JOINT_DATA_CONFIG)
    cfg["qpos_dim"] = qpos_dim
    cfg["action_dim"] = action_dim
    cfg["action_key"] = action_key if action_key is not None else cfg["action_key"]

    # qpos/action indexing is intentionally decoupled.
    if qpos_indices is not None:
        cfg["qpos_indices"] = list(qpos_indices)
    elif qpos_dim is not None:
        cfg["qpos_indices"] = list(range(qpos_dim))

    if action_indices is not None:
        cfg["action_indices"] = list(action_indices)
    elif action_dim is not None:
        cfg["action_indices"] = list(range(action_dim))

    return cfg

def rotate_n_crop_transform(img, size=(360, 480), angle=None, top=None):
    if angle is None:
        angle = np.random.random() * 10 - 5
    if top is None:
        w, h = img.size
        top_h = np.random.randint(0, max(1, h - size[0] + 1))
        top_w = np.random.randint(0, max(1, w - size[1] + 1))
        top = [top_h, top_w]

    img = transforms.functional.rotate(img, angle)
    img = transforms.functional.crop(img, *top, *size)
    return img


def _is_grayscale_image(img):
    if img.ndim == 2:
        return True
    if img.ndim == 3 and img.shape[-1] == 1:
        return True
    return False


def _is_event_frame(cam_name, img):
    return ('event' in cam_name.lower()) or _is_grayscale_image(img)

def ensure_hwc3(image, cam_name):
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape for camera '{cam_name}': {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape for camera '{cam_name}': {image.shape}")
    return image


def maybe_resize_for_rgb_event(image, cam_name, camera_names, target_size):
    target_h, target_w = target_size
    if camera_names == ["rgb", "event"]:
        if image.shape[:2] != target_size:
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if image.shape[:2] != target_size:
            raise ValueError(
                f"RGB+event mode requires {target_h}x{target_w} images. camera={cam_name}, got={image.shape}"
            )
    return image


def _print_image_pipeline_info(camera_names, target_size, event_input_channels=3, event_channel_index=2):
    camera_list = ', '.join(camera_names)
    event_camera_names = [name for name in camera_names if 'event' in name.lower()]
    target_h, target_w = target_size

    print(f"[INFO] Camera names: {camera_list}")
    if event_camera_names:
        for event_camera_name in event_camera_names:
            print(f"[INFO] Event camera detected: {event_camera_name}")
    else:
        print("[INFO] Event camera detected: none (grayscale auto-detect still enabled)")
    print("[INFO] Event preprocessing enabled: True (camera-name or grayscale auto-detect)")
    print(f"[INFO] Resizing all images to {target_h}x{target_w}")
    if camera_names == ["event"] and event_input_channels == 1:
        print("[INFO] Event frames kept as single-channel input")
        print(f"[INFO] event_channel_index={event_channel_index} (used when raw event image has 3 channels)")
    else:
        print("[INFO] Event frames converted to fake RGB for ResNet compatibility")


def _extract_single_event_channel(image, event_channel_index, cam_name):
    image = np.asarray(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        if image.shape[-1] == 1:
            return image[..., 0]
        if image.shape[-1] >= 3:
            if not (0 <= event_channel_index < image.shape[-1]):
                raise ValueError(
                    f"Invalid event_channel_index={event_channel_index} for camera '{cam_name}' with image shape {image.shape}"
                )
            return image[..., event_channel_index]

    raise ValueError(f"Unsupported event image shape for camera '{cam_name}': {image.shape}")


def _resize_single_channel_image(image_hw, target_size):
    target_h, target_w = target_size
    if image_hw.shape[:2] == target_size:
        return image_hw
    return cv2.resize(image_hw, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _natural_episode_sort_key(dataset_path):
    basename = os.path.basename(dataset_path)
    match = re.match(r'^episode_(\d+)\.hdf5$', basename)
    if match is not None:
        return (0, int(match.group(1)), basename)
    return (1, basename)


def _extract_episode_index(dataset_path):
    basename = os.path.basename(dataset_path)
    match = re.match(r'^episode_(\d+)\.hdf5$', basename)
    if match is None:
        return None
    return int(match.group(1))


def _warn_if_episode_indices_noncontiguous(episode_paths):
    episode_indices_by_dir = {}
    for dataset_path in episode_paths:
        episode_index = _extract_episode_index(dataset_path)
        if episode_index is None:
            continue
        dataset_dir = os.path.dirname(dataset_path)
        episode_indices_by_dir.setdefault(dataset_dir, []).append(episode_index)

    for dataset_dir, episode_indices in episode_indices_by_dir.items():
        unique_indices = sorted(set(episode_indices))
        if not unique_indices:
            continue
        expected_indices = list(range(unique_indices[0], unique_indices[-1] + 1))
        if unique_indices != expected_indices:
            print(f"[WARN] Non-contiguous episode indices detected in {dataset_dir}: {unique_indices}")


def collect_episode_paths(dataset_dirs):
    if isinstance(dataset_dirs, str):
        normalized_dataset_dirs = [dataset_dirs]
    else:
        normalized_dataset_dirs = list(dataset_dirs)

    if len(normalized_dataset_dirs) == 0:
        raise ValueError("No dataset directories were provided.")

    episode_paths = []
    for dataset_dir in normalized_dataset_dirs:
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset path is not a directory: {dataset_dir}")
        dataset_episode_paths = glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))
        if len(dataset_episode_paths) == 0:
            raise FileNotFoundError(f"No episode_*.hdf5 files found in dataset directory: {dataset_dir}")
        dataset_episode_paths.sort(key=_natural_episode_sort_key)
        episode_paths.extend(dataset_episode_paths)

    if len(episode_paths) == 0:
        joined_dirs = ', '.join(normalized_dataset_dirs)
        raise FileNotFoundError(f"No episode_*.hdf5 files found in dataset directories: {joined_dirs}")

    return episode_paths


# =========================
# split-selection helpers
# =========================

def _compute_episode_action_stats_joint(episode_paths, action_indices=None, action_key='/action'):
    """
    Per-episode stats for joint/action datasets.
    Uses only action distribution, since this generic loader does not know dx/dy.
    """
    episode_stats = []
    for episode_idx, dataset_path in enumerate(episode_paths):
        with h5py.File(dataset_path, 'r') as root:
            action = root[action_key][()]
            if action_indices is not None:
                action = action[:, action_indices]
        action = np.asarray(action, dtype=np.float32)

        ep_stat = {
            "episode_idx": episode_idx,
            "num_steps": int(action.shape[0]),
            "mean_action": action.mean(axis=0),
            "pos_frac_action": (action > 0).mean(axis=0),
        }
        episode_stats.append(ep_stat)
    return episode_stats


def _compute_episode_action_stats_pose(episode_paths):
    """
    Per-episode stats for pose/action datasets.
    """
    episode_stats = []
    for episode_idx, dataset_path in enumerate(episode_paths):
        with h5py.File(dataset_path, 'r') as root:
            action = root['/ee_action_global'][()]
        action = np.asarray(action, dtype=np.float32)

        ep_stat = {
            "episode_idx": episode_idx,
            "num_steps": int(action.shape[0]),
            "mean_action": action.mean(axis=0),
            "pos_frac_action": (action > 0).mean(axis=0),
        }
        episode_stats.append(ep_stat)
    return episode_stats


def _aggregate_episode_stats(episode_stats, indices):
    """
    Weighted aggregation over episodes using episode length as weight.
    """
    if len(indices) == 0:
        raise ValueError("Cannot aggregate empty split.")

    weights = np.array([episode_stats[i]["num_steps"] for i in indices], dtype=np.float64)
    weights = weights / weights.sum()

    mean_action = np.stack([episode_stats[i]["mean_action"] for i in indices], axis=0)
    pos_frac_action = np.stack([episode_stats[i]["pos_frac_action"] for i in indices], axis=0)

    return {
        "mean_action": (weights[:, None] * mean_action).sum(axis=0),
        "pos_frac_action": (weights[:, None] * pos_frac_action).sum(axis=0),
        "num_selected": len(indices),
        "num_steps": int(sum(episode_stats[i]["num_steps"] for i in indices)),
    }


def _compute_global_episode_stats(episode_stats):
    all_indices = list(range(len(episode_stats)))
    return _aggregate_episode_stats(episode_stats, all_indices)


def _score_split(train_stats, val_stats, global_stats):
    """
    Lower is better.

    We want:
    - train close to val
    - both close to global distribution

    Mean mismatch is normalized by a rough scale from global positivity spread.
    """
    mean_scale = 1.0

    train_val_mean_gap = np.mean(np.abs(train_stats["mean_action"] - val_stats["mean_action"])) / mean_scale
    train_val_pos_gap = np.mean(np.abs(train_stats["pos_frac_action"] - val_stats["pos_frac_action"]))

    val_global_mean_gap = np.mean(np.abs(val_stats["mean_action"] - global_stats["mean_action"])) / mean_scale
    val_global_pos_gap = np.mean(np.abs(val_stats["pos_frac_action"] - global_stats["pos_frac_action"]))

    train_global_mean_gap = np.mean(np.abs(train_stats["mean_action"] - global_stats["mean_action"])) / mean_scale
    train_global_pos_gap = np.mean(np.abs(train_stats["pos_frac_action"] - global_stats["pos_frac_action"]))

    # Emphasize val being representative, since checkpoint selection uses val.
    score = (
        1.0 * train_val_mean_gap +
        2.0 * train_val_pos_gap +
        1.5 * val_global_mean_gap +
        2.5 * val_global_pos_gap +
        0.5 * train_global_mean_gap +
        1.0 * train_global_pos_gap
    )
    return float(score)


def _print_split_summary(train_indices, val_indices, episode_stats, header="Chosen split"):
    train_stats = _aggregate_episode_stats(episode_stats, train_indices)
    val_stats = _aggregate_episode_stats(episode_stats, val_indices)
    global_stats = _compute_global_episode_stats(episode_stats)

    print(f"\n===== {header} =====")
    print(f"train episodes: {len(train_indices)} | val episodes: {len(val_indices)}")
    print(f"train steps   : {train_stats['num_steps']} | val steps   : {val_stats['num_steps']}")

    dim = len(global_stats["mean_action"])
    for d in range(dim):
        print(f"\n--- action dim {d} ---")
        print(f"global mean        : {global_stats['mean_action'][d]: .6f}")
        print(f"train mean         : {train_stats['mean_action'][d]: .6f}")
        print(f"val mean           : {val_stats['mean_action'][d]: .6f}")
        print(f"global pos frac    : {global_stats['pos_frac_action'][d]: .3f}")
        print(f"train pos frac     : {train_stats['pos_frac_action'][d]: .3f}")
        print(f"val pos frac       : {val_stats['pos_frac_action'][d]: .3f}")
        print(f"|train-val mean|   : {abs(train_stats['mean_action'][d] - val_stats['mean_action'][d]): .6f}")
        print(f"|train-val posfrac|: {abs(train_stats['pos_frac_action'][d] - val_stats['pos_frac_action'][d]): .3f}")
    print("")


def _choose_balanced_episode_split(
    episode_stats,
    train_ratio=0.8,
    num_trials=100,
    seed=0,
    verbose=True,
):
    """
    Option A:
    Try many random episode-level splits and keep the most balanced one.
    """
    episode_count = len(episode_stats)
    if episode_count < 2:
        raise ValueError("Need at least 2 episodes to create train/val split.")

    num_train = int(train_ratio * episode_count)
    num_train = max(1, min(num_train, episode_count - 1))

    global_stats = _compute_global_episode_stats(episode_stats)
    rng = np.random.RandomState(seed)

    best_score = np.inf
    best_train_indices = None
    best_val_indices = None

    for _ in range(num_trials):
        shuffled = rng.permutation(episode_count)
        train_indices = np.sort(shuffled[:num_train])
        val_indices = np.sort(shuffled[num_train:])

        train_stats = _aggregate_episode_stats(episode_stats, train_indices)
        val_stats = _aggregate_episode_stats(episode_stats, val_indices)
        score = _score_split(train_stats, val_stats, global_stats)

        if score < best_score:
            best_score = score
            best_train_indices = train_indices
            best_val_indices = val_indices

    if verbose:
        print(f"\nBalanced split search: {num_trials} trials, best score = {best_score:.6f}")
        _print_split_summary(best_train_indices, best_val_indices, episode_stats, header="Balanced split summary")

    return best_train_indices, best_val_indices


class EpisodicJointDataset(torch.utils.data.Dataset):
    def __init__(self, episode_paths, camera_names, chunk_size, norm_stats, qpos_indices=None, action_indices=None, action_key='/action', img_aug=False, image_size=None, event_input_channels=3, event_channel_index=2):
        super(EpisodicJointDataset).__init__()
        self.qpos_indices = qpos_indices
        self.action_indices = action_indices
        self.action_key = action_key
        self.episode_paths = list(episode_paths)
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self.img_aug = img_aug
        self.image_size = canonical_image_size(image_size)
        self.event_input_channels = int(event_input_channels)
        self.event_channel_index = int(event_channel_index)
        self.is_sim = None
        self._printed_image_debug = False
        if self.event_input_channels not in (1, 3):
            raise ValueError(f"event_input_channels must be 1 or 3, got {self.event_input_channels}")
        if self.event_input_channels == 1 and self.camera_names != ["event"]:
            raise ValueError(
                "event_input_channels=1 is only supported for camera_names=['event'] in current training pipeline."
            )
        _print_image_pipeline_info(
            self.camera_names,
            self.image_size,
            event_input_channels=self.event_input_channels,
            event_channel_index=self.event_channel_index,
        )
        print(f"[INFO] Selected canonical image size: {self.image_size}")
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_paths)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode
        
        dataset_path = self.episode_paths[index]
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            episode_len = root[self.action_key].shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            if self.qpos_indices is not None:
                qpos = qpos[self.qpos_indices]
            image_dict = dict()
            for cam_name in self.camera_names:
                if cam_name.endswith('stereo'):
                    left_img = root[f'/observations/images/{cam_name[:-6]}left'][start_ts]
                    right_img = root[f'/observations/images/{cam_name[:-6]}right'][start_ts]
                    left_is_event = _is_event_frame(cam_name, left_img)
                    right_is_event = _is_event_frame(cam_name, right_img)
                    left_img = transforms.functional.to_pil_image(left_img)
                    right_img = transforms.functional.to_pil_image(right_img)
                    if self.img_aug:
                        angle = np.random.random() * 10 - 5
                        top_h = np.random.randint(0, 120)
                        top_w = np.random.randint(0, 160)
                        if not left_is_event:
                            left_img = color_transform(left_img)
                        left_img = rotate_n_crop_transform(left_img, [480, 640], angle, (top_h, top_w))
                        if not right_is_event:
                            right_img = color_transform(right_img)
                        right_img = rotate_n_crop_transform(right_img, [480, 640], angle, (top_h, top_w))
                    left_img = np.asarray(left_img)
                    right_img = np.asarray(right_img)
                    left_img = ensure_hwc3(left_img, f'{cam_name}_left')
                    right_img = ensure_hwc3(right_img, f'{cam_name}_right')
                    left_img = maybe_resize_for_rgb_event(left_img, f'{cam_name}_left', self.camera_names, self.image_size)
                    right_img = maybe_resize_for_rgb_event(right_img, f'{cam_name}_right', self.camera_names, self.image_size)
                    stereo_img = np.concatenate([left_img, right_img], axis=1) # width dimension
                    if self.camera_names != ["rgb", "event"]:
                        stereo_img = cv2.resize(
                            stereo_img,
                            (self.image_size[1], self.image_size[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    image_dict[cam_name] = ensure_hwc3(stereo_img, cam_name)
                else:
                    img = root[f'/observations/images/{cam_name}'][start_ts]
                    is_event = _is_event_frame(cam_name, img)
                    img = transforms.functional.to_pil_image(img)
                    if self.img_aug:
                        if not is_event:
                            img = color_transform(img)
                        img = rotate_n_crop_transform(img)
                    img = np.asarray(img)
                    if self.camera_names == ["event"] and self.event_input_channels == 1:
                        img = _extract_single_event_channel(img, self.event_channel_index, cam_name)
                        img = _resize_single_channel_image(img, self.image_size)
                        img = img[..., None]
                    else:
                        img = ensure_hwc3(img, cam_name)
                        if self.camera_names != ["rgb", "event"]:
                            img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
                        img = maybe_resize_for_rgb_event(img, cam_name, self.camera_names, self.image_size)
                    image_dict[cam_name] = img
            # get all actions after and including start_ts
            action = root[self.action_key][start_ts:min(start_ts+self.chunk_size, episode_len)]
            if self.action_indices is not None:
                action = action[:, self.action_indices]
            action_len, action_dof = action.shape

        self.is_sim = is_sim
        padded_action = np.zeros((self.chunk_size, action_dof), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # new axis for different cameras in exact camera_names order
        all_cam_images = []
        for cam_name in self.camera_names:
            cam_image = image_dict[cam_name]
            if self.camera_names == ["event"] and self.event_input_channels == 1:
                cam_image = _extract_single_event_channel(cam_image, self.event_channel_index, cam_name)
                cam_image = _resize_single_channel_image(cam_image, self.image_size)
                cam_image = cam_image[..., None]
            else:
                cam_image = ensure_hwc3(cam_image, cam_name)
                cam_image = maybe_resize_for_rgb_event(cam_image, cam_name, self.camera_names, self.image_size)
            cam_image = np.transpose(cam_image, (2, 0, 1))
            all_cam_images.append(cam_image)
        image_data = torch.from_numpy(np.stack(all_cam_images, axis=0))
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        assert image_data.ndim == 4, image_data.shape
        assert image_data.shape[0] == len(self.camera_names), image_data.shape
        expected_channels = 1 if (self.camera_names == ["event"] and self.event_input_channels == 1) else 3
        assert image_data.shape[1] == expected_channels, image_data.shape
        if self.camera_names == ["event"]:
            assert image_data.shape == (1, expected_channels, self.image_size[0], self.image_size[1]), image_data.shape
        if self.camera_names == ["rgb", "event"]:
            assert image_data.shape == (2, 3, self.image_size[0], self.image_size[1]), image_data.shape
        if not self._printed_image_debug:
            print(f"[DEBUG] camera_names={self.camera_names}, image_data.shape={tuple(image_data.shape)}")
            self._printed_image_debug = True

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_joint_norm_stats(episode_paths, qpos_indices=None, action_indices=None, action_key='/action'):
    all_qpos_data = []
    all_action_data = []

    for dataset_path in episode_paths:
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root[action_key][()]

        if qpos_indices is not None:
            qpos = qpos[:, qpos_indices]
        if action_indices is not None:
            action = action[:, action_indices]

        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    qpos_mean = all_qpos_data.mean(dim=0).numpy()
    qpos_std = all_qpos_data.std(dim=0).numpy().clip(1e-2, np.inf)

    action_mean = all_action_data.mean(dim=0).numpy()
    action_std = all_action_data.std(dim=0).numpy().clip(1e-2, np.inf)

    stats = {
        "action_mean": action_mean,
        "action_std": action_std,
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "example_qpos": qpos,
    }

    return stats


def load_joint_data(
    dataset_dirs,
    camera_names,
    chunk_size,
    batch_size_train,
    batch_size_val,
    model_dof=None,
    img_aug=False,
    split_num_trials=500,
    split_seed=0,
    qpos_dim=None,
    action_dim=None,
    qpos_indices=None,
    action_indices=None,
    action_key='/action',
    image_size=None,
    event_input_channels=3,
    event_channel_index=2,
):
    # Backward-compatible default: model_dof applies to both if explicit dims are not provided.
    if qpos_dim is None and model_dof is not None:
        qpos_dim = model_dof
    if action_dim is None and model_dof is not None:
        action_dim = model_dof

    joint_data_cfg = _build_joint_data_config(
        qpos_dim=qpos_dim,
        action_dim=action_dim,
        qpos_indices=qpos_indices,
        action_indices=action_indices,
        action_key=action_key,
    )

    episode_paths = collect_episode_paths(dataset_dirs)
    total_episode_count = len(episode_paths)
    if total_episode_count < 2:
        raise ValueError("Need at least 2 episodes for train/val split")

    if isinstance(dataset_dirs, str):
        source_dataset_dirs = [dataset_dirs]
    else:
        source_dataset_dirs = list(dataset_dirs)
    
    print('\nSource dataset dirs:')
    for source_dataset_dir in source_dataset_dirs:
        print(source_dataset_dir)
    print(f'Total discovered episodes: {total_episode_count}')
    print('First few episode paths:')
    for dataset_path in episode_paths[:min(5, total_episode_count)]:
        print(dataset_path)
    print('')
    _warn_if_episode_indices_noncontiguous(episode_paths)

    # obtain train/val split using balanced random search
    # train_ratio = 0.8
    # episode_stats = _compute_episode_action_stats_joint(
    #     episode_paths,
    #     action_indices=joint_data_cfg['action_indices'],
    #     action_key=joint_data_cfg['action_key'],
    # )
    # train_indices, val_indices = _choose_balanced_episode_split(
    #     episode_stats=episode_stats,
    #     train_ratio=train_ratio,
    #     num_trials=split_num_trials,
    #     seed=split_seed,
    #     verbose=True,
    # )
    episode_indices = list(range(total_episode_count))
    rng = np.random.RandomState(split_seed)
    rng.shuffle(episode_indices)

    split_idx = int(0.8 * total_episode_count)
    split_idx = max(1, min(split_idx, total_episode_count - 1))

    train_indices = episode_indices[:split_idx]
    val_indices = episode_indices[split_idx:]

    print(f'Train episode count: {len(train_indices)}')
    print(f'Val episode count: {len(val_indices)}')

    # obtain normalization stats for qpos and action
    norm_stats = get_joint_norm_stats(
        episode_paths,
        qpos_indices=joint_data_cfg['qpos_indices'],
        action_indices=joint_data_cfg['action_indices'],
        action_key=joint_data_cfg['action_key'],
    )

    # construct dataset and dataloader
    train_dataset = EpisodicJointDataset(
        [episode_paths[index] for index in train_indices],
        camera_names,
        chunk_size,
        norm_stats,
        qpos_indices=joint_data_cfg['qpos_indices'],
        action_indices=joint_data_cfg['action_indices'],
        action_key=joint_data_cfg['action_key'],
        img_aug=img_aug,
        image_size=image_size,
        event_input_channels=event_input_channels,
        event_channel_index=event_channel_index,
    )
    val_dataset = EpisodicJointDataset(
        [episode_paths[index] for index in val_indices],
        camera_names,
        chunk_size,
        norm_stats,
        qpos_indices=joint_data_cfg['qpos_indices'],
        action_indices=joint_data_cfg['action_indices'],
        action_key=joint_data_cfg['action_key'],
        img_aug=img_aug,
        image_size=image_size,
        event_input_channels=event_input_channels,
        event_channel_index=event_channel_index,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


class EpisodicPoseDataset(torch.utils.data.Dataset):
    def __init__(self, episode_paths, camera_names, chunk_size, norm_stats, img_aug, image_size=None, event_input_channels=3, event_channel_index=2):
        super(EpisodicPoseDataset).__init__()
        self.episode_paths = list(episode_paths)
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self.img_aug = img_aug
        self.image_size = canonical_image_size(image_size)
        self.event_input_channels = int(event_input_channels)
        self.event_channel_index = int(event_channel_index)
        self.is_sim = None
        self._printed_image_debug = False
        if self.event_input_channels not in (1, 3):
            raise ValueError(f"event_input_channels must be 1 or 3, got {self.event_input_channels}")
        if self.event_input_channels == 1 and self.camera_names != ["event"]:
            raise ValueError(
                "event_input_channels=1 is only supported for camera_names=['event'] in current training pipeline."
            )
        _print_image_pipeline_info(
            self.camera_names,
            self.image_size,
            event_input_channels=self.event_input_channels,
            event_channel_index=self.event_channel_index,
        )
        print(f"[INFO] Selected canonical image size: {self.image_size}")
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_paths)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        dataset_path = self.episode_paths[index]
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            episode_len = root['/observations/ee_pose_global'].shape[0] - 120  # hardcode for TOCABI data, do not train moving to ready pose
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/ee_pose_global'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                if cam_name.endswith('stereo'):
                    left_img = root[f'/observations/images/{cam_name[:-6]}left'][start_ts]
                    right_img = root[f'/observations/images/{cam_name[:-6]}right'][start_ts]
                    left_is_event = _is_event_frame(cam_name, left_img)
                    right_is_event = _is_event_frame(cam_name, right_img)
                    left_img = transforms.functional.to_pil_image(left_img)
                    right_img = transforms.functional.to_pil_image(right_img)
                    if self.img_aug:
                        angle = np.random.random() * 10 - 5
                        top_h = np.random.randint(0, 120)
                        top_w = np.random.randint(0, 160)
                        if not left_is_event:
                            left_img = color_transform(left_img)
                        left_img = rotate_n_crop_transform(left_img, [480, 640], angle, (top_h, top_w))
                        if not right_is_event:
                            right_img = color_transform(right_img)
                        right_img = rotate_n_crop_transform(right_img, [480, 640], angle, (top_h, top_w))
                    left_img = np.asarray(left_img)
                    right_img = np.asarray(right_img)
                    left_img = ensure_hwc3(left_img, f'{cam_name}_left')
                    right_img = ensure_hwc3(right_img, f'{cam_name}_right')
                    left_img = maybe_resize_for_rgb_event(left_img, f'{cam_name}_left', self.camera_names, self.image_size)
                    right_img = maybe_resize_for_rgb_event(right_img, f'{cam_name}_right', self.camera_names, self.image_size)
                    stereo_img = np.concatenate([left_img, right_img], axis=1) # width dimension
                    if self.camera_names != ["rgb", "event"]:
                        stereo_img = cv2.resize(
                            stereo_img,
                            (self.image_size[1], self.image_size[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    image_dict[cam_name] = ensure_hwc3(stereo_img, cam_name)
                else:
                    img = root[f'/observations/images/{cam_name}'][start_ts]
                    is_event = _is_event_frame(cam_name, img)
                    img = transforms.functional.to_pil_image(img)
                    if self.img_aug:
                        if not is_event:
                            img = color_transform(img)
                        img = rotate_n_crop_transform(img)
                    img = np.asarray(img)
                    if self.camera_names == ["event"] and self.event_input_channels == 1:
                        img = _extract_single_event_channel(img, self.event_channel_index, cam_name)
                        img = _resize_single_channel_image(img, self.image_size)
                        img = img[..., None]
                    else:
                        img = ensure_hwc3(img, cam_name)
                        if self.camera_names != ["rgb", "event"]:
                            img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
                        img = maybe_resize_for_rgb_event(img, cam_name, self.camera_names, self.image_size)
                    image_dict[cam_name] = img
            # get all actions after and including start_ts
            action = root['/ee_action_global'][start_ts:min(start_ts+self.chunk_size, episode_len)]
            action_len, action_dof = action.shape

        self.is_sim = is_sim
        padded_action = np.zeros((self.chunk_size, action_dof), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # new axis for different cameras in exact camera_names order
        all_cam_images = []
        for cam_name in self.camera_names:
            cam_image = image_dict[cam_name]
            if self.camera_names == ["event"] and self.event_input_channels == 1:
                cam_image = _extract_single_event_channel(cam_image, self.event_channel_index, cam_name)
                cam_image = _resize_single_channel_image(cam_image, self.image_size)
                cam_image = cam_image[..., None]
            else:
                cam_image = ensure_hwc3(cam_image, cam_name)
                cam_image = maybe_resize_for_rgb_event(cam_image, cam_name, self.camera_names, self.image_size)
            cam_image = np.transpose(cam_image, (2, 0, 1))
            all_cam_images.append(cam_image)
        image_data = torch.from_numpy(np.stack(all_cam_images, axis=0))
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        assert image_data.ndim == 4, image_data.shape
        assert image_data.shape[0] == len(self.camera_names), image_data.shape
        expected_channels = 1 if (self.camera_names == ["event"] and self.event_input_channels == 1) else 3
        assert image_data.shape[1] == expected_channels, image_data.shape
        if self.camera_names == ["event"]:
            assert image_data.shape == (1, expected_channels, self.image_size[0], self.image_size[1]), image_data.shape
        if self.camera_names == ["rgb", "event"]:
            assert image_data.shape == (2, 3, self.image_size[0], self.image_size[1]), image_data.shape
        if not self._printed_image_debug:
            print(f"[DEBUG] camera_names={self.camera_names}, image_data.shape={tuple(image_data.shape)}")
            self._printed_image_debug = True

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_pose_norm_stats(episode_paths):
    all_qpos_data = []
    all_action_data = []
    for dataset_path in episode_paths:
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/ee_pose_global'][()]
            action = root['/ee_action_global'][()]
        all_qpos_data.append(torch.from_numpy(qpos[:,9:12])) # do not normalize 9D roation & binary gripper state
        all_action_data.append(torch.from_numpy(action[:,9:12]))
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = np.zeros(13, dtype=np.float32)
    action_mean[9:12] = all_action_data.mean(dim=0).numpy()
    action_std = np.ones(13, dtype=np.float32)
    action_std[9:12] = all_action_data.std(dim=0).numpy()
    action_std = action_std.clip(1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = np.zeros(13, dtype=np.float32)
    qpos_mean[9:12] = all_qpos_data.mean(dim=0).numpy()
    qpos_std = np.ones(13, dtype=np.float32)
    qpos_std[9:12] = all_qpos_data.std(dim=0).numpy()
    qpos_std = qpos_std.clip(1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean, "action_std": action_std,
             "qpos_mean": qpos_mean, "qpos_std": qpos_std,
             "example_qpos": qpos}

    return stats


def load_pose_data(
    dataset_dirs,
    camera_names,
    chunk_size,
    batch_size_train,
    batch_size_val,
    img_aug=False,
    split_num_trials=100,
    split_seed=0,
    image_size=None,
    event_input_channels=3,
    event_channel_index=2,
):
    episode_paths = collect_episode_paths(dataset_dirs)
    total_episode_count = len(episode_paths)
    if total_episode_count < 2:
        raise ValueError("Need at least 2 episodes for train/val split")

    if isinstance(dataset_dirs, str):
        source_dataset_dirs = [dataset_dirs]
    else:
        source_dataset_dirs = list(dataset_dirs)

    print('\nSource dataset dirs:')
    for source_dataset_dir in source_dataset_dirs:
        print(source_dataset_dir)
    print(f'Total discovered episodes: {total_episode_count}')
    print('First few episode paths:')
    for dataset_path in episode_paths[:min(5, total_episode_count)]:
        print(dataset_path)
    print('')
    _warn_if_episode_indices_noncontiguous(episode_paths)

    # obtain train/val split using balanced random search
    train_ratio = 0.8
    episode_stats = _compute_episode_action_stats_pose(episode_paths)
    train_indices, val_indices = _choose_balanced_episode_split(
        episode_stats=episode_stats,
        train_ratio=train_ratio,
        num_trials=split_num_trials,
        seed=split_seed,
        verbose=True,
    )

    print(f'Train episode count: {len(train_indices)}')
    print(f'Val episode count: {len(val_indices)}')

    # obtain normalization stats for qpos and action
    norm_stats = get_pose_norm_stats(episode_paths)

    # construct dataset and dataloader
    train_dataset = EpisodicPoseDataset(
        [episode_paths[index] for index in train_indices],
        camera_names,
        chunk_size,
        norm_stats,
        img_aug,
        image_size=image_size,
        event_input_channels=event_input_channels,
        event_channel_index=event_channel_index,
    )
    val_dataset = EpisodicPoseDataset(
        [episode_paths[index] for index in val_indices],
        camera_names,
        chunk_size,
        norm_stats,
        img_aug,
        image_size=image_size,
        event_input_channels=event_input_channels,
        event_channel_index=event_channel_index,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)