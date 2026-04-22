import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.v2 as transforms

import IPython
e = IPython.embed

color_transform = transforms.ColorJitter(brightness=0.5,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1,
                          )

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


# =========================
# split-selection helpers
# =========================

def _compute_episode_action_stats_joint(dataset_dir, num_episodes, active_joints):
    """
    Per-episode stats for joint/action datasets.
    Uses only action distribution, since this generic loader does not know dx/dy.
    """
    episode_stats = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            action = root['/action'][:, active_joints][()]
        action = np.asarray(action, dtype=np.float32)

        ep_stat = {
            "episode_idx": episode_idx,
            "num_steps": int(action.shape[0]),
            "mean_action": action.mean(axis=0),
            "pos_frac_action": (action > 0).mean(axis=0),
        }
        episode_stats.append(ep_stat)
    return episode_stats


def _compute_episode_action_stats_pose(dataset_dir, num_episodes):
    """
    Per-episode stats for pose/action datasets.
    """
    episode_stats = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
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
        "num_episodes": len(indices),
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
    num_episodes,
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
    if num_episodes < 2:
        raise ValueError("Need at least 2 episodes to create train/val split.")

    num_train = int(train_ratio * num_episodes)
    num_train = max(1, min(num_train, num_episodes - 1))

    global_stats = _compute_global_episode_stats(episode_stats)
    rng = np.random.RandomState(seed)

    best_score = np.inf
    best_train_indices = None
    best_val_indices = None

    for _ in range(num_trials):
        shuffled = rng.permutation(num_episodes)
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
    def __init__(self, episode_ids, dataset_dir, camera_names, chunk_size, norm_stats, active_joints, img_aug=False):
        super(EpisodicJointDataset).__init__()
        self.active_joints = active_joints
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self.img_aug = img_aug
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode
        
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            episode_len = root['/action'].shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts, self.active_joints]
            image_dict = dict()
            for cam_name in self.camera_names:
                if cam_name.endswith('stereo'):
                    left_img = root[f'/observations/images/{cam_name[:-6]}left'][start_ts]
                    right_img = root[f'/observations/images/{cam_name[:-6]}right'][start_ts]
                    left_img = transforms.functional.to_pil_image(left_img)
                    right_img = transforms.functional.to_pil_image(right_img)
                    if self.img_aug:
                        angle = np.random.random() * 10 - 5
                        top_h = np.random.randint(0, 120)
                        top_w = np.random.randint(0, 160)
                        left_img = color_transform(left_img)
                        left_img = rotate_n_crop_transform(left_img, [480, 640], angle, (top_h, top_w))
                        right_img = color_transform(right_img)
                        right_img = rotate_n_crop_transform(right_img, [480, 640], angle, (top_h, top_w))
                    left_img = transforms.functional.resize(left_img, [480, 640])
                    right_img = transforms.functional.resize(right_img, [480, 640])
                    image_dict[cam_name] = np.concatenate([left_img, right_img], axis=1) # width dimension
                else:
                    img = root[f'/observations/images/{cam_name}'][start_ts]
                    img = transforms.functional.to_pil_image(img)
                    if self.img_aug:
                        img = color_transform(img)
                        img = rotate_n_crop_transform(img)
                    #img = transforms.functional.resize(img, [480, 640])
                    image_dict[cam_name] = img
            # get all actions after and including start_ts
            action = root['/action'][start_ts:min(start_ts+self.chunk_size, episode_len), self.active_joints]
            action_len, action_dof = action.shape

        self.is_sim = is_sim
        padded_action = np.zeros((self.chunk_size, action_dof), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_joint_norm_stats(dataset_dir, num_episodes, active_joints):
    all_qpos_data = []
    all_action_data = []

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][:, active_joints]
            action = root['/action'][:, active_joints]

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
    dataset_dir,
    num_episodes,
    camera_names,
    chunk_size,
    batch_size_train,
    batch_size_val,
    model_dof,
    img_aug=False,
    split_num_trials=500,
    split_seed=0,
):
    active_joints = list(range(model_dof))
    
    print(f'\nData from: {dataset_dir}\n')

    # obtain train/val split using balanced random search
    train_ratio = 0.8
    episode_stats = _compute_episode_action_stats_joint(dataset_dir, num_episodes, active_joints)
    train_indices, val_indices = _choose_balanced_episode_split(
        num_episodes=num_episodes,
        episode_stats=episode_stats,
        train_ratio=train_ratio,
        num_trials=split_num_trials,
        seed=split_seed,
        verbose=True,
    )

    # obtain normalization stats for qpos and action
    norm_stats = get_joint_norm_stats(dataset_dir, num_episodes, active_joints)

    # construct dataset and dataloader
    train_dataset = EpisodicJointDataset(train_indices, dataset_dir, camera_names, chunk_size, norm_stats, active_joints, img_aug)
    val_dataset = EpisodicJointDataset(val_indices, dataset_dir, camera_names, chunk_size, norm_stats, active_joints, img_aug)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


class EpisodicPoseDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, chunk_size, norm_stats, img_aug):
        super(EpisodicPoseDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self.img_aug = img_aug
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
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
                    left_img = transforms.functional.to_pil_image(left_img)
                    right_img = transforms.functional.to_pil_image(right_img)
                    if self.img_aug:
                        angle = np.random.random() * 10 - 5
                        top_h = np.random.randint(0, 120)
                        top_w = np.random.randint(0, 160)
                        left_img = color_transform(left_img)
                        left_img = rotate_n_crop_transform(left_img, [480, 640], angle, (top_h, top_w))
                        right_img = color_transform(right_img)
                        right_img = rotate_n_crop_transform(right_img, [480, 640], angle, (top_h, top_w))
                    left_img = transforms.functional.resize(left_img, [480, 640])
                    right_img = transforms.functional.resize(right_img, [480, 640])
                    image_dict[cam_name] = np.concatenate([left_img, right_img], axis=1) # width dimension
                else:
                    img = root[f'/observations/images/{cam_name}'][start_ts]
                    img = transforms.functional.to_pil_image(img)
                    if self.img_aug:
                        img = color_transform(img)
                        img = rotate_n_crop_transform(img)
                    #img = transforms.functional.resize(img, [480, 640])
                    image_dict[cam_name] = img
            # get all actions after and including start_ts
            action = root['/ee_action_global'][start_ts:min(start_ts+self.chunk_size, episode_len)]
            action_len, action_dof = action.shape

        self.is_sim = is_sim
        padded_action = np.zeros((self.chunk_size, action_dof), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_pose_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
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
    dataset_dir,
    num_episodes,
    camera_names,
    chunk_size,
    batch_size_train,
    batch_size_val,
    img_aug=False,
    split_num_trials=100,
    split_seed=0,
):
    print(f'\nData from: {dataset_dir}\n')

    # obtain train/val split using balanced random search
    train_ratio = 0.8
    episode_stats = _compute_episode_action_stats_pose(dataset_dir, num_episodes)
    train_indices, val_indices = _choose_balanced_episode_split(
        num_episodes=num_episodes,
        episode_stats=episode_stats,
        train_ratio=train_ratio,
        num_trials=split_num_trials,
        seed=split_seed,
        verbose=True,
    )

    # obtain normalization stats for qpos and action
    norm_stats = get_pose_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicPoseDataset(train_indices, dataset_dir, camera_names, chunk_size, norm_stats, img_aug)
    val_dataset = EpisodicPoseDataset(val_indices, dataset_dir, camera_names, chunk_size, norm_stats, img_aug)
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