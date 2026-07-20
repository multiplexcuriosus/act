#!/usr/bin/env python3
"""Focused validation for intercept action schema and loader wiring.

This script validates:
1) Dense (N,2) action generation for nonzero and zero targets.
2) Timestamp-to-grid command alignment and transition behavior.
3) Invalid-episode rejection cases.
4) HDF5 schema emitted by write_episode().
5) load_intercept_data() canonical path and old split-schema rejection.
6) Generic joint-mode helper sanity remains available.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _install_ros_stubs() -> None:
    # bag_to_il_intercept imports ROS modules at import time.
    if "rosbag2_py" not in sys.modules:
        rosbag2_py = types.ModuleType("rosbag2_py")

        class _Dummy:
            def __init__(self, *args, **kwargs):
                pass

        rosbag2_py.SequentialReader = _Dummy
        rosbag2_py.StorageOptions = _Dummy
        rosbag2_py.ConverterOptions = _Dummy
        sys.modules["rosbag2_py"] = rosbag2_py

    if "rclpy" not in sys.modules:
        rclpy = types.ModuleType("rclpy")
        serialization = types.ModuleType("rclpy.serialization")

        def _deserialize_message(raw, msg_cls):
            return raw

        serialization.deserialize_message = _deserialize_message
        rclpy.serialization = serialization
        sys.modules["rclpy"] = rclpy
        sys.modules["rclpy.serialization"] = serialization

    if "rosidl_runtime_py" not in sys.modules:
        rosidl_runtime_py = types.ModuleType("rosidl_runtime_py")
        utilities = types.ModuleType("rosidl_runtime_py.utilities")

        def _get_message(_):
            return object

        utilities.get_message = _get_message
        rosidl_runtime_py.utilities = utilities
        sys.modules["rosidl_runtime_py"] = rosidl_runtime_py
        sys.modules["rosidl_runtime_py.utilities"] = utilities


def _install_ml_stubs() -> None:
    if "IPython" not in sys.modules:
        ipython = types.ModuleType("IPython")
        ipython.embed = lambda *args, **kwargs: None
        sys.modules["IPython"] = ipython

    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.manual_seed = lambda *_args, **_kwargs: None

        torch_utils = types.ModuleType("torch.utils")
        torch_utils_data = types.ModuleType("torch.utils.data")

        class _Dataset:
            pass

        class _DataLoader:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        torch_utils_data.Dataset = _Dataset
        torch_utils_data.DataLoader = _DataLoader
        torch_utils.data = torch_utils_data
        torch.utils = torch_utils

        sys.modules["torch"] = torch
        sys.modules["torch.utils"] = torch_utils
        sys.modules["torch.utils.data"] = torch_utils_data

    if "torchvision" not in sys.modules:
        torchvision = types.ModuleType("torchvision")
        transforms_mod = types.ModuleType("torchvision.transforms")
        transforms_v2 = types.ModuleType("torchvision.transforms.v2")

        class _ColorJitter:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, image):
                return image

        functional = types.SimpleNamespace(
            to_pil_image=lambda x: x,
            resize=lambda x, *_args, **_kwargs: x,
            rotate=lambda x, *_args, **_kwargs: x,
            crop=lambda x, *_args, **_kwargs: x,
        )
        transforms_v2.ColorJitter = _ColorJitter
        transforms_v2.functional = functional

        transforms_mod.v2 = transforms_v2
        torchvision.transforms = transforms_mod

        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.transforms"] = transforms_mod
        sys.modules["torchvision.transforms.v2"] = transforms_v2


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fake_rgb_msg(value: int = 0) -> SimpleNamespace:
    image = np.full((2, 2, 3), value, dtype=np.uint8)
    return SimpleNamespace(
        height=2,
        width=2,
        encoding="rgb8",
        step=6,
        data=image.tobytes(),
    )


def _episode_data(command_time: float, target_s: float) -> dict:
    rgb_msgs = [_fake_rgb_msg(10), _fake_rgb_msg(20), _fake_rgb_msg(30), _fake_rgb_msg(40)]
    return {
        "rgb_t": [0.0, 1.0, 2.0, 3.0],
        "rgb_msg": rgb_msgs,
        "joint_t": [0.0, 1.0, 2.0, 3.0],
        "qpos": [
            np.zeros(7, dtype=np.float32),
            np.ones(7, dtype=np.float32),
            np.full(7, 2.0, dtype=np.float32),
            np.full(7, 3.0, dtype=np.float32),
        ],
        "goto_s_t": [command_time],
        "goto_s": [target_s],
        "target_base_t": [],
        "target_base": [],
    }


def _assert_raises(fn, expected_substring: str) -> None:
    try:
        fn()
    except RuntimeError as exc:
        message = str(exc)
        if expected_substring not in message:
            raise AssertionError(
                f"Expected RuntimeError containing '{expected_substring}', got: {message}"
            ) from exc
        return
    raise AssertionError("Expected RuntimeError but no exception was raised")


def _write_episode_file(path: str, action: np.ndarray, include_old_flag: bool = False) -> None:
    n = int(action.shape[0])
    with h5py.File(path, "w") as h5:
        h5.attrs["sim"] = False
        h5.create_dataset("/action", data=action.astype(np.float32), dtype=np.float32)
        if include_old_flag:
            h5.create_dataset(
                "/action_is_commanded",
                data=np.zeros((n, 1), dtype=np.uint8),
                dtype=np.uint8,
            )
        h5.create_dataset("/observations/qpos", data=np.zeros((n, 7), dtype=np.float32), dtype=np.float32)
        h5.create_dataset(
            "/observations/images/rgb",
            data=np.zeros((n, 2, 2, 3), dtype=np.uint8),
            dtype=np.uint8,
        )


def main() -> None:
    _install_ros_stubs()
    _install_ml_stubs()

    bag_converter = _load_module(
        "bag_to_il_intercept",
        os.path.join(REPO_ROOT, "act", "helpers", "bag_to_il_intercept.py"),
    )
    utils = _load_module(
        "act_utils",
        os.path.join(REPO_ROOT, "act", "utils.py"),
    )

    episode = bag_converter.EpisodeWindow(source_idx=0, output_idx=0, start=0.0, end=3.0)

    # 1) Nonzero target: dense col0 constant, col1 transitions at first grid >= command_time.
    arrays = bag_converter.sample_episode(
        data=_episode_data(command_time=2.0, target_s=-0.0884),
        episode=episode,
        fps=1.0,
    )
    action = arrays["action"]
    assert action.shape == (4, 2)
    assert np.allclose(action[:, 0], -0.0884, atol=1e-6)
    assert np.array_equal(action[:, 1], np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

    # 2) Zero target: column 0 remains zero while execute flag still transitions.
    arrays_zero = bag_converter.sample_episode(
        data=_episode_data(command_time=2.0, target_s=0.0),
        episode=episode,
        fps=1.0,
    )
    action_zero = arrays_zero["action"]
    assert np.array_equal(action_zero[:, 0], np.zeros(4, dtype=np.float32))
    assert np.array_equal(action_zero[:, 1], np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

    # 3) Alignment checks.
    assert len(arrays["timestamps"]) == len(arrays["rgb"]) == len(arrays["qpos"]) == len(arrays["action"])
    assert arrays["action"].shape == (len(arrays["timestamps"]), 2)

    # 4) Invalid episodes.
    def _no_command():
        data = _episode_data(command_time=2.0, target_s=0.1)
        data["goto_s_t"] = []
        data["goto_s"] = []
        bag_converter.sample_episode(data=data, episode=episode, fps=1.0)

    def _multiple_commands():
        data = _episode_data(command_time=2.0, target_s=0.1)
        data["goto_s_t"] = [1.0, 2.0]
        data["goto_s"] = [0.1, 0.2]
        bag_converter.sample_episode(data=data, episode=episode, fps=1.0)

    def _command_after_interval():
        bag_converter.sample_episode(
            data=_episode_data(command_time=10.0, target_s=0.1),
            episode=episode,
            fps=1.0,
        )

    def _no_precommand_samples():
        bag_converter.sample_episode(
            data=_episode_data(command_time=0.0, target_s=0.1),
            episode=episode,
            fps=1.0,
        )

    _assert_raises(_no_command, "expected exactly one GOTO_S command")
    _assert_raises(_multiple_commands, "incompatible with multiple commands")
    _assert_raises(_command_after_interval, "after the causally usable RGB/joint interval")
    _assert_raises(_no_precommand_samples, "no causally usable pre-command samples")

    # 5) HDF5 schema emitted by converter.
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "episode_0.hdf5")
        bag_converter.write_episode(
            output_path=out_path,
            arrays=arrays,
            episode=episode,
            topics=bag_converter.Topics(
                rgb="/top_cam/camera/color/image_raw",
                joint="/joint_states",
                episode="/episode/control",
                goto_s="/trajectory_executor/executed_goto_s",
                goto_s_target_base="/trajectory_executor/executed_goto_s_target_base",
            ),
            fps=1.0,
            compression="none",
            overwrite=True,
        )
        with h5py.File(out_path, "r") as h5:
            assert "/action" in h5
            assert h5["/action"].shape[1] == 2
            assert h5["/action"].dtype == np.float32
            assert "/commands/goto_s/timestamps" in h5
            assert "/commands/goto_s/values" in h5
            assert "/observations/qpos" in h5
            assert "/observations/images/rgb" in h5
            assert "/action_is_commanded" not in h5

    # 6) Loader: canonical intercept path reads /action directly and enforces new schema.
    with tempfile.TemporaryDirectory() as tmpdir:
        episode_path = os.path.join(tmpdir, "episode_0.hdf5")
        _write_episode_file(episode_path, action=np.zeros((5, 2), dtype=np.float32), include_old_flag=False)

        sentinel = object()
        with mock.patch.object(utils, "load_joint_data", return_value=sentinel) as patched_loader:
            result = utils.load_intercept_data(
                dataset_dirs=tmpdir,
                camera_names=["rgb"],
                chunk_size=4,
                batch_size_train=1,
                batch_size_val=1,
                qpos_dim=7,
            )
            assert result is sentinel
            _, kwargs = patched_loader.call_args
            assert kwargs["action_key"] == "/action"
            assert kwargs["action_dim"] == 2
            assert "command_flag_key" not in kwargs

    with tempfile.TemporaryDirectory() as tmpdir:
        old_episode_path = os.path.join(tmpdir, "episode_0.hdf5")
        _write_episode_file(old_episode_path, action=np.zeros((5, 2), dtype=np.float32), include_old_flag=True)
        _assert_raises(
            lambda: utils.load_intercept_data(
                dataset_dirs=tmpdir,
                camera_names=["rgb"],
                chunk_size=4,
                batch_size_train=1,
                batch_size_val=1,
                qpos_dim=7,
            ),
            "Old split intercept schema detected",
        )

    # 7) Backward compatibility outside intercept mode: generic joint loader API remains available.
    joint_signature = inspect.signature(utils.load_joint_data)
    assert "action_key" in joint_signature.parameters
    assert "dataset_dirs" in joint_signature.parameters

    print("All intercept schema validations passed.")


if __name__ == "__main__":
    main()
