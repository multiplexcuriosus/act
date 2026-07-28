#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

HERE = os.path.dirname(__file__)
ACT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ACT_ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, ACT_ROOT)

from utils import (  # noqa: E402
    EpisodicInterceptDataset,
    INTERCEPT_TARGET_DESIRED_GOAL,
    INTERCEPT_TARGET_MEASURED_TCP_TRAJECTORY,
    get_intercept_norm_stats,
    load_intercept_data,
    _validate_intercept_episode_structure,
)


class InterceptTargetModeTests(unittest.TestCase):
    def write_episode(
        self,
        path: str,
        action_values: np.ndarray,
        desired_goal: float | None = 0.10,
        selected_goal_values: tuple[float, ...] = (0.10,),
        include_goal_attrs: bool = True,
    ) -> None:
        action = np.asarray(action_values, dtype=np.float32).reshape(-1, 1)
        T = int(action.shape[0])
        rgb = np.stack(
            [np.full((4, 4, 3), fill_value=10 + index, dtype=np.uint8) for index in range(T)],
            axis=0,
        )
        qpos = np.stack(
            [np.full(7, fill_value=float(index), dtype=np.float32) for index in range(T)],
            axis=0,
        )
        with h5py.File(path, "w") as h5:
            h5.attrs["sim"] = False
            h5.attrs["action_type"] = "measured_tcp_s_absolute"
            h5.attrs["action_representation"] = "absolute"
            h5.attrs["action_coordinate"] = "captured_interception_line"
            h5.attrs["action_origin"] = "captured_line_center"
            h5.attrs["action_positive_direction"] = "robot_base_positive_x"
            h5.attrs["action_units"] = "m"
            if include_goal_attrs:
                h5.attrs["desired_goal_source_topic"] = "/interception_controller/selected_goto_s"
                h5.attrs["desired_goal_resolution"] = "single_selected_goto_s_event"
                h5.attrs["desired_goal_coordinate"] = "captured_interception_line"
                h5.attrs["desired_goal_positive_direction"] = "robot_base_positive_x"
                h5.attrs["desired_goal_units"] = "m"

            h5.create_dataset("/action", data=action, dtype=np.float32)
            h5.create_dataset("/observations/qpos", data=qpos, dtype=np.float32)
            h5.create_dataset("/observations/images/rgb", data=rgb, dtype=np.uint8)

            commands = h5.create_group("commands")
            selected = commands.create_group("selected_goto_s")
            selected.create_dataset(
                "timestamps",
                data=np.asarray([0.1 + 0.1 * index for index in range(len(selected_goal_values))], dtype=np.float64),
                dtype=np.float64,
            )
            selected.create_dataset(
                "values",
                data=np.asarray(selected_goal_values, dtype=np.float32).reshape(-1, 1),
                dtype=np.float32,
            )

            targets = h5.create_group("targets")
            if desired_goal is not None:
                targets.create_dataset(
                    "desired_intercept_s",
                    data=np.asarray([desired_goal], dtype=np.float32),
                    dtype=np.float32,
                )

    def make_stats(self, intercept_target: str, action_mean: float = 0.0, action_std: float = 1.0):
        return {
            "action_mean": np.asarray([action_mean], dtype=np.float32),
            "action_std": np.asarray([action_std], dtype=np.float32),
            "qpos_mean": np.zeros(21, dtype=np.float32),
            "qpos_std": np.ones(21, dtype=np.float32),
            "data_mode": "intercept",
            "raw_qpos_dim": 7,
            "state_dim": 21,
            "action_dim": 1,
            "rgb_history_frames": 3,
            "rgb_history_offsets": [-6, -3, 0],
            "qpos_history_offsets": [-6, -3, 0],
            "rgb_frame_order": "oldest_to_newest",
            "qpos_flatten_order": "oldest_to_newest",
            "image_channels": 9,
            "intercept_target": intercept_target,
            "action_type": "measured_tcp_s_delta"
            if intercept_target == INTERCEPT_TARGET_MEASURED_TCP_TRAJECTORY
            else "desired_intercept_s_delta",
            "action_representation": "future_delta_relative_to_anchor"
            if intercept_target == INTERCEPT_TARGET_MEASURED_TCP_TRAJECTORY
            else "episode_goal_delta_relative_to_anchor_tcp",
            "action_anchor_offset": 0,
            "action_first_target_offset": 1
            if intercept_target == INTERCEPT_TARGET_MEASURED_TCP_TRAJECTORY
            else 0,
            "action_query_semantics": "future_measured_tcp_trajectory"
            if intercept_target == INTERCEPT_TARGET_MEASURED_TCP_TRAJECTORY
            else "replicated_current_goal",
            "action_positive_direction": "robot_base_positive_x",
            "action_units": "m",
            "goal_source_topic": "/interception_controller/selected_goto_s"
            if intercept_target == INTERCEPT_TARGET_DESIRED_GOAL
            else None,
        }

    def make_dataset(self, paths, intercept_target):
        stats = self.make_stats(intercept_target=intercept_target)
        return EpisodicInterceptDataset(
            paths,
            ["rgb"],
            chunk_size=30,
            norm_stats=stats,
            intercept_target=intercept_target,
            history_offsets=(-6, -3, 0),
            photometric_aug=False,
            spatial_aug=False,
            image_size=4,
        )

    def test_measured_mode_output_exactly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "episode_0.hdf5")
            self.write_episode(path, action_values=[-0.02, 0.10, 0.20, 0.30], selected_goal_values=(0.10,))
            dataset = self.make_dataset([path], INTERCEPT_TARGET_MEASURED_TCP_TRAJECTORY)
            with mock.patch("numpy.random.randint", return_value=0):
                _image, _qpos, action, is_pad = dataset[0]

            self.assertEqual(action.shape, (30, 1))
            np.testing.assert_allclose(action[:3, 0].numpy(), np.asarray([0.12, 0.22, 0.32], dtype=np.float32), atol=1e-6)
            self.assertTrue(is_pad[3:].all().item())

    def test_desired_goal_replicates_residual_and_no_padding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "episode_0.hdf5")
            self.write_episode(path, action_values=[-0.02, 0.00, 0.02], desired_goal=0.10, selected_goal_values=(0.10,))
            dataset = self.make_dataset([path], INTERCEPT_TARGET_DESIRED_GOAL)
            with mock.patch("numpy.random.randint", return_value=0):
                _image, _qpos, action, is_pad = dataset[0]

            self.assertEqual(action.shape, (30, 1))
            np.testing.assert_allclose(action[:, 0].numpy(), np.full(30, 0.12, dtype=np.float32), atol=1e-6)
            self.assertFalse(is_pad.any().item())

    def test_desired_goal_anchor_t_minus_one_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "episode_0.hdf5")
            self.write_episode(path, action_values=[-0.20, -0.10, 0.02], desired_goal=-0.10, selected_goal_values=(-0.10,))
            dataset = self.make_dataset([path], INTERCEPT_TARGET_DESIRED_GOAL)
            with mock.patch("numpy.random.randint", return_value=2):
                _image, _qpos, action, is_pad = dataset[0]

            np.testing.assert_allclose(action[:, 0].numpy(), np.full(30, -0.12, dtype=np.float32), atol=1e-6)
            self.assertFalse(is_pad.any().item())

    def test_goal_residual_norm_uses_training_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "episode_0.hdf5")
            val_path = os.path.join(tmpdir, "episode_1.hdf5")
            self.write_episode(train_path, action_values=[-0.02, 0.00, 0.02], desired_goal=0.10, selected_goal_values=(0.10,))
            self.write_episode(val_path, action_values=[0.20, 0.22, 0.24], desired_goal=0.10, selected_goal_values=(0.10,))
            stats = get_intercept_norm_stats([train_path], chunk_size=30, intercept_target=INTERCEPT_TARGET_DESIRED_GOAL)

            self.assertAlmostEqual(float(stats["action_mean"][0]), 0.10, places=6)
            self.assertGreater(float(stats["action_std"][0]), 0.0)

    def test_selected_goal_validation_failures(self):
        cases = [
            ("missing_targets", dict(desired_goal=None), "Missing required dataset: /targets/desired_intercept_s"),
            ("zero_count", dict(selected_goal_values=()), "expected exactly one finite selected_goto_s event"),
            ("multiple", dict(selected_goal_values=(0.10, 0.20)), "expected exactly one finite selected_goto_s event"),
            ("non_finite", dict(selected_goal_values=(np.nan,)), "expected exactly one finite selected_goto_s event"),
        ]

        for _, kwargs, message in cases:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "episode_0.hdf5")
                self.write_episode(
                    path,
                    action_values=[-0.02, 0.00, 0.02],
                    **kwargs,
                )
                with h5py.File(path, "r") as h5:
                    with self.assertRaises(ValueError) as ctx:
                        _validate_intercept_episode_structure(
                            h5,
                            path,
                            INTERCEPT_TARGET_DESIRED_GOAL,
                        )
                self.assertIn(message, str(ctx.exception))

    def test_load_intercept_data_preserves_target_mode_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path0 = os.path.join(tmpdir, "episode_0.hdf5")
            path1 = os.path.join(tmpdir, "episode_1.hdf5")
            self.write_episode(path0, action_values=[-0.02, 0.00, 0.02], desired_goal=0.10, selected_goal_values=(0.10,))
            self.write_episode(path1, action_values=[0.02, 0.04, 0.06], desired_goal=0.10, selected_goal_values=(0.10,))

            train_loader, _val_loader, stats, _ = load_intercept_data(
                tmpdir,
                ["rgb"],
                chunk_size=30,
                batch_size_train=1,
                batch_size_val=1,
                intercept_target=INTERCEPT_TARGET_DESIRED_GOAL,
                image_size=4,
                rgb_history_frames=3,
                history_offsets=(-6, -3, 0),
            )

            self.assertEqual(stats["intercept_target"], INTERCEPT_TARGET_DESIRED_GOAL)
            batch_image, batch_qpos, batch_action, batch_is_pad = next(iter(train_loader))
            self.assertEqual(batch_action.shape[1:], (30, 1))
            self.assertFalse(batch_is_pad.any().item())
            self.assertTrue(np.isfinite(batch_action.numpy()).all())


if __name__ == "__main__":
    unittest.main()
