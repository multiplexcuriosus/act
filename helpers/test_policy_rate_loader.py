"""Synthetic 30/60 Hz sparse interception loader contract tests."""

import h5py
import numpy as np
import pytest

from utils import load_intercept_data


def _write_sparse_episode(path, rate):
    steps = 70
    timestamps = np.arange(steps, dtype=np.float64) / rate
    with h5py.File(path, "w") as root:
        root.attrs.update(
            action_type="measured_tcp_s_absolute",
            action_representation="absolute",
            action_positive_direction="robot_base_positive_x",
            policy_rate_hz=rate,
            rgb_width_px=1280,
            rgb_height_px=720,
        )
        root.create_dataset("/action", data=np.linspace(0, 1, steps, dtype=np.float32)[:, None])
        root.create_dataset("/observations/qpos", data=np.zeros((steps, 7), dtype=np.float32))
        root.create_dataset("/observations/timestamps", data=timestamps)
        sparse = root.require_group("/observations/sparse_tracking")
        sparse.create_dataset("rgb_2d_px", data=np.tile([640.0, 360.0], (steps, 1)))
        sparse.create_dataset("rgb_valid", data=np.ones(steps, dtype=np.uint8))
        sparse.create_dataset("rgb_source_timestamps", data=timestamps)


def _dataset_dir(tmp_path, rate):
    directory = tmp_path / f"rate_{rate}"
    directory.mkdir()
    for index in range(2):
        _write_sparse_episode(directory / f"episode_{index}.hdf5", rate)
    return directory


def _load(directory, rate, offsets):
    return load_intercept_data(
        str(directory), ["sparse_ball"], rate, 1, 1,
        input_modality="sparse_ball", sparse_source="rgb",
        history_offsets=offsets, policy_rate_hz=rate,
    )


@pytest.mark.parametrize("rate,offsets", [(30, (-6, -3, 0)), (60, (-12, -6, 0))])
def test_sparse_loader_accepts_matching_rate_and_offsets(tmp_path, rate, offsets):
    train, _, stats, _ = _load(_dataset_dir(tmp_path, rate), rate, offsets)
    assert train.dataset.qpos_history_offsets == offsets
    assert stats["qpos_history_offsets"] == list(offsets)


def test_60_hz_loader_rejects_30_hz_hdf5(tmp_path):
    with pytest.raises(ValueError, match="Regenerate.*bag_to_il_intercept.py"):
        _load(_dataset_dir(tmp_path, 30), 60, (-12, -6, 0))


def test_sparse_loader_rejects_wrong_offsets(tmp_path):
    with pytest.raises(
        ValueError,
        match=r"modality=sparse_ball.*policy_rate_hz=60.*expected=.*received=",
    ):
        _load(_dataset_dir(tmp_path, 60), 60, (-6, -3, 0))
