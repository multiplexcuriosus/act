import os
import sys
import tempfile

import h5py
import numpy as np
import pytest
import torch

HELPERS_DIR = os.path.dirname(__file__)
ACT_DIR = os.path.dirname(HELPERS_DIR)
if ACT_DIR not in sys.path:
    sys.path.insert(0, ACT_DIR)

import detr.models.backbone as backbone_mod
from policy import ACTPolicy, _build_image_normalizer
from utils import (
    INTERCEPT_EVENT_WINDOWS_MS_DEFAULT,
    INTERCEPT_HISTORY_OFFSETS_DEFAULT,
    EpisodicInterceptDataset,
    _prepare_visual_history_frames,
    _validate_intercept_episode_structure,
    compute_history_indices,
    get_intercept_norm_stats,
    load_intercept_data,
)


def _write_intercept_episode(path, mode="event", steps=12, metadata_override=None, data_override=None):
    metadata_override = metadata_override or {}
    data_override = data_override or {}

    qpos = np.stack(
        [np.linspace(0.0 + i, 0.6 + i, 7, dtype=np.float32) for i in range(steps)],
        axis=0,
    )
    action_abs = np.linspace(0.2, 0.2 + 0.01 * (steps - 1), steps, dtype=np.float32).reshape(-1, 1)
    timestamps = np.linspace(1.0, 1.0 + 0.033 * (steps - 1), steps, dtype=np.float64)

    rgb = np.stack(
        [np.full((24, 24, 3), fill_value=min(255, i * 7), dtype=np.uint8) for i in range(steps)],
        axis=0,
    )
    event = np.stack(
        [np.full((24, 24, 3), fill_value=128 + ((i % 3) - 1) * 20, dtype=np.uint8) for i in range(steps)],
        axis=0,
    )

    event_source_timestamps = timestamps - 0.001
    event_source_age_sec = timestamps - event_source_timestamps
    event_count_per_channel = np.full((steps, 3), fill_value=5, dtype=np.int64)

    qpos = np.asarray(data_override.get("qpos", qpos), dtype=np.float32)
    action_abs = np.asarray(data_override.get("action", action_abs), dtype=np.float32)
    timestamps = np.asarray(data_override.get("timestamps", timestamps), dtype=np.float64)
    rgb = np.asarray(data_override.get("rgb", rgb))
    event = np.asarray(data_override.get("event", event))
    event_source_timestamps = np.asarray(
        data_override.get("event_source_timestamps", event_source_timestamps),
        dtype=np.float64,
    )
    event_source_age_sec = np.asarray(
        data_override.get("event_source_age_sec", event_source_age_sec), dtype=np.float64
    )
    event_count_per_channel = np.asarray(
        data_override.get("event_count_per_channel", event_count_per_channel)
    )

    with h5py.File(path, "w") as root:
        root.attrs["sim"] = False
        root.attrs["action_type"] = metadata_override.get("action_type", "measured_tcp_s_absolute")
        root.attrs["action_representation"] = metadata_override.get("action_representation", "absolute")
        root.attrs["action_positive_direction"] = metadata_override.get(
            "action_positive_direction", "robot_base_positive_x"
        )

        if mode == "event":
            root.attrs["event_representation"] = metadata_override.get(
                "event_representation", "shifted_3chef_signed"
            )
            root.attrs["event_frame_mode"] = metadata_override.get("event_frame_mode", "shifted")
            root.attrs["event_frame_windows_ms"] = np.asarray(
                metadata_override.get("event_frame_windows_ms", INTERCEPT_EVENT_WINDOWS_MS_DEFAULT),
                dtype=np.float64,
            )
            root.attrs["event_channel_order"] = metadata_override.get(
                "event_channel_order", "recent_to_oldest"
            )
            root.attrs["event_scaling"] = metadata_override.get(
                "event_scaling", "signed_log1p_fixed_clip"
            )
            root.attrs["event_clip_count"] = float(metadata_override.get("event_clip_count", 3.0))
            root.attrs["event_neutral_u8"] = int(metadata_override.get("event_neutral_u8", 128))
            root.attrs["event_sampling_policy"] = metadata_override.get(
                "event_sampling_policy", "latest_packet_at_or_before_grid_time"
            )

        root.create_dataset("/observations/qpos", data=qpos)
        root.create_dataset("/action", data=action_abs)
        root.create_dataset("/observations/timestamps", data=timestamps)
        root.create_dataset("/observations/images/rgb", data=rgb)
        if mode == "event":
            root.create_dataset("/observations/images/event", data=event)
            root.create_dataset("/event_source_timestamps", data=event_source_timestamps)
            root.create_dataset("/event_source_age_sec", data=event_source_age_sec)
            root.create_dataset("/event_count_per_channel", data=event_count_per_channel)


def _simple_policy_config(image_normalization="shifted_3chef_centered"):
    return {
        "lr": 1e-4,
        "num_queries": 4,
        "kl_weight": 1,
        "hidden_dim": 64,
        "dim_feedforward": 128,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 1,
        "dec_layers": 1,
        "nheads": 4,
        "camera_names": ["event"],
        "state_dim": 21,
        "action_dim": 1,
        "use_bce_last_action_dim": False,
        "device": "cpu",
        "image_size": 64,
        "image_channels": 9,
        "visual_history_frames": 3,
        "visual_history_offsets": list(INTERCEPT_HISTORY_OFFSETS_DEFAULT),
        "channels_per_visual_frame": 3,
        "image_normalization": image_normalization,
    }


def test_event_episode_validation_accepts_valid_episode(tmp_path):
    ep = tmp_path / "episode_0.hdf5"
    _write_intercept_episode(str(ep), mode="event")

    with h5py.File(ep, "r") as root:
        qpos, action, metadata = _validate_intercept_episode_structure(
            root, str(ep), modality="event", expected_event_metadata=None
        )
    assert qpos.shape[1] == 7
    assert action.shape[1] == 1
    assert metadata["event_frame_mode"] == "shifted"


def test_event_episode_validation_rejects_bad_event_shape(tmp_path):
    ep = tmp_path / "episode_0.hdf5"
    _write_intercept_episode(
        str(ep),
        mode="event",
        data_override={"event": np.zeros((12, 24, 24), dtype=np.uint8)},
    )
    with h5py.File(ep, "r") as root:
        with pytest.raises(ValueError):
            _validate_intercept_episode_structure(root, str(ep), modality="event")


def test_event_episode_validation_rejects_missing_event_attr(tmp_path):
    ep = tmp_path / "episode_0.hdf5"
    _write_intercept_episode(str(ep), mode="event", metadata_override={"event_frame_mode": "foo"})
    with h5py.File(ep, "r") as root:
        with pytest.raises(ValueError):
            _validate_intercept_episode_structure(root, str(ep), modality="event")


def test_event_episode_validation_rejects_noncausal_source_timestamps(tmp_path):
    ep = tmp_path / "episode_0.hdf5"
    ts = np.linspace(1.0, 1.0 + 0.033 * 11, 12, dtype=np.float64)
    _write_intercept_episode(
        str(ep),
        mode="event",
        data_override={"timestamps": ts, "event_source_timestamps": ts + 0.001},
    )
    with h5py.File(ep, "r") as root:
        with pytest.raises(ValueError):
            _validate_intercept_episode_structure(root, str(ep), modality="event")


def test_event_episode_validation_rejects_incorrect_age_consistency(tmp_path):
    ep = tmp_path / "episode_0.hdf5"
    ts = np.linspace(1.0, 1.0 + 0.033 * 11, 12, dtype=np.float64)
    source_ts = ts - 0.001
    wrong_age = np.full((12,), 0.999, dtype=np.float64)
    _write_intercept_episode(
        str(ep),
        mode="event",
        data_override={
            "timestamps": ts,
            "event_source_timestamps": source_ts,
            "event_source_age_sec": wrong_age,
        },
    )
    with h5py.File(ep, "r") as root:
        with pytest.raises(ValueError):
            _validate_intercept_episode_structure(root, str(ep), modality="event")


def test_prepare_visual_history_event_rejects_photometric_aug():
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    with pytest.raises(ValueError):
        _prepare_visual_history_frames(
            frames,
            modality="event",
            camera_names=["event"],
            image_size=(8, 8),
            photometric_aug=True,
            spatial_aug=False,
        )


def test_prepare_visual_history_event_rejects_spatial_aug():
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    with pytest.raises(ValueError):
        _prepare_visual_history_frames(
            frames,
            modality="event",
            camera_names=["event"],
            image_size=(8, 8),
            photometric_aug=False,
            spatial_aug=True,
        )


def test_history_indices_clamp_and_order():
    assert compute_history_indices(0, INTERCEPT_HISTORY_OFFSETS_DEFAULT) == [0, 0, 0]
    assert compute_history_indices(5, INTERCEPT_HISTORY_OFFSETS_DEFAULT) == [0, 2, 5]


def test_event_dataset_sample_shape_and_qpos_shape(tmp_path):
    ep = tmp_path / "episode_0.hdf5"
    _write_intercept_episode(str(ep), mode="event")

    stats = {
        "action_mean": np.zeros((1,), dtype=np.float32),
        "action_std": np.ones((1,), dtype=np.float32),
        "qpos_mean": np.zeros((21,), dtype=np.float32),
        "qpos_std": np.ones((21,), dtype=np.float32),
    }

    with h5py.File(ep, "r") as root:
        _, _, event_meta = _validate_intercept_episode_structure(root, str(ep), modality="event")

    ds = EpisodicInterceptDataset(
        [str(ep)],
        ["event"],
        chunk_size=4,
        norm_stats=stats,
        image_size=16,
        input_modality="event",
        expected_event_metadata=event_meta,
    )
    image, qpos, action, is_pad = ds[0]
    assert tuple(image.shape) == (1, 9, 16, 16)
    assert tuple(qpos.shape) == (21,)
    assert action.shape[1] == 1
    assert is_pad.shape[0] == 4


def test_load_intercept_data_rejects_invalid_camera_names(tmp_path):
    for i in range(2):
        _write_intercept_episode(str(tmp_path / f"episode_{i}.hdf5"), mode="event")
    with pytest.raises(ValueError):
        load_intercept_data(
            str(tmp_path),
            ["rgb", "event"],
            chunk_size=4,
            batch_size_train=1,
            batch_size_val=1,
        )


def test_load_intercept_data_rejects_mixed_event_metadata(tmp_path):
    _write_intercept_episode(str(tmp_path / "episode_0.hdf5"), mode="event")
    _write_intercept_episode(
        str(tmp_path / "episode_1.hdf5"),
        mode="event",
        metadata_override={"event_channel_order": "oldest_to_recent"},
    )
    with pytest.raises(ValueError):
        load_intercept_data(
            str(tmp_path),
            ["event"],
            chunk_size=4,
            batch_size_train=1,
            batch_size_val=1,
            rgb_history_frames=3,
            visual_history_frames=3,
        )


def test_get_intercept_norm_stats_event_requires_metadata(tmp_path):
    ep = tmp_path / "episode_0.hdf5"
    _write_intercept_episode(str(ep), mode="event")
    with pytest.raises(ValueError):
        get_intercept_norm_stats([str(ep)], chunk_size=4, input_modality="event", event_metadata=None)


def test_get_intercept_norm_stats_event_includes_metadata(tmp_path):
    ep = tmp_path / "episode_0.hdf5"
    _write_intercept_episode(str(ep), mode="event")

    with h5py.File(ep, "r") as root:
        _, _, event_meta = _validate_intercept_episode_structure(root, str(ep), modality="event")
    stats = get_intercept_norm_stats(
        [str(ep)],
        chunk_size=4,
        input_modality="event",
        event_metadata=event_meta,
    )
    assert stats["input_modality"] == "event"
    assert stats["image_normalization"] == "shifted_3chef_centered"
    assert stats["event_representation"] == "shifted_3chef_signed"


def test_normalizer_shifted_3chef_centered_values():
    norm = _build_image_normalizer(9, normalization_mode="shifted_3chef_centered")
    assert len(norm.mean) == 9
    assert len(norm.std) == 9
    assert np.isclose(norm.mean[0], 128.0 / 255.0)
    assert np.isclose(norm.std[0], 127.0 / 255.0)


def test_normalizer_imagenet_repeats_for_nine_channels():
    norm = _build_image_normalizer(9, normalization_mode="imagenet")
    assert len(norm.mean) == 9
    assert np.isclose(norm.mean[0], 0.485)
    assert np.isclose(norm.mean[3], 0.485)


def test_event_and_rgb_split_membership_consistency(tmp_path):
    event_dir = tmp_path / "event"
    rgb_dir = tmp_path / "rgb"
    event_dir.mkdir()
    rgb_dir.mkdir()

    for i in range(6):
        _write_intercept_episode(str(event_dir / f"episode_{i}.hdf5"), mode="event")
        _write_intercept_episode(str(rgb_dir / f"episode_{i}.hdf5"), mode="rgb")

    train_event, val_event, stats_event, _ = load_intercept_data(
        str(event_dir), ["event"], chunk_size=4, batch_size_train=2, batch_size_val=2, split_seed=7
    )
    train_rgb, val_rgb, stats_rgb, _ = load_intercept_data(
        str(rgb_dir), ["rgb"], chunk_size=4, batch_size_train=2, batch_size_val=2, split_seed=7
    )

    event_train_names = sorted(os.path.basename(p) for p in train_event.dataset.episode_paths)
    rgb_train_names = sorted(os.path.basename(p) for p in train_rgb.dataset.episode_paths)
    assert event_train_names == rgb_train_names
    assert len(val_event.dataset.episode_paths) == len(val_rgb.dataset.episode_paths)
    assert stats_event["qpos_mean"].shape == stats_rgb["qpos_mean"].shape == (21,)
    assert stats_event["action_mean"].shape == stats_rgb["action_mean"].shape == (1,)


def test_act_event_forward_one_minibatch_smoke():
    backbone_mod.is_main_process = lambda: False
    policy = ACTPolicy(_simple_policy_config())
    batch_size = 2
    qpos = torch.zeros((batch_size, 21), dtype=torch.float32)
    image = torch.zeros((batch_size, 1, 9, 64, 64), dtype=torch.float32)
    actions = torch.zeros((batch_size, 4, 1), dtype=torch.float32)
    is_pad = torch.zeros((batch_size, 4), dtype=torch.bool)

    out = policy(qpos, image, actions, is_pad)
    assert "loss" in out
    pred = policy(qpos, image)
    assert tuple(pred.shape) == (batch_size, 4, 1)


def test_act_event_backbone_conv1_is_nine_channels():
    backbone_mod.is_main_process = lambda: False
    policy = ACTPolicy(_simple_policy_config())
    conv_channels = []
    for backbone in getattr(policy.model, "backbones", []):
        conv_channels.append(int(backbone[0].body.conv1.in_channels))
    assert conv_channels
    assert all(ch == 9 for ch in conv_channels)
