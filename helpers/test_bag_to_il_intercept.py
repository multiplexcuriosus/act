#!/usr/bin/env python3

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import cv2
import h5py
import numpy as np


def load_converter_module():
    if "rosbag2_py" not in sys.modules:
        rosbag2_py = types.ModuleType("rosbag2_py")

        class _DummySequentialReader:
            def open(self, *args, **kwargs):
                raise NotImplementedError

            def get_all_topics_and_types(self):
                return []

            def has_next(self):
                return False

            def read_next(self):
                raise StopIteration

            def set_filter(self, *_args, **_kwargs):
                return None

        class _DummyStorageFilter:
            def __init__(self, topics):
                self.topics = list(topics)

        rosbag2_py.SequentialReader = _DummySequentialReader
        rosbag2_py.StorageFilter = _DummyStorageFilter
        rosbag2_py.StorageOptions = lambda **kwargs: kwargs
        rosbag2_py.ConverterOptions = lambda **kwargs: kwargs
        sys.modules["rosbag2_py"] = rosbag2_py

    if "rclpy" not in sys.modules:
        rclpy = types.ModuleType("rclpy")
        sys.modules["rclpy"] = rclpy

    if "rclpy.serialization" not in sys.modules:
        serialization = types.ModuleType("rclpy.serialization")
        serialization.deserialize_message = lambda raw, cls: raw
        sys.modules["rclpy.serialization"] = serialization

    if "rosidl_runtime_py" not in sys.modules:
        rosidl_runtime_py = types.ModuleType("rosidl_runtime_py")
        sys.modules["rosidl_runtime_py"] = rosidl_runtime_py

    if "rosidl_runtime_py.utilities" not in sys.modules:
        utilities = types.ModuleType("rosidl_runtime_py.utilities")
        utilities.get_message = lambda type_name: object
        sys.modules["rosidl_runtime_py.utilities"] = utilities

    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)
    module_path = os.path.join(here, "bag_to_il_intercept.py")
    spec = importlib.util.spec_from_file_location(
        "bag_to_il_intercept_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BagToIlInterceptTests(unittest.TestCase):
    def test_policy_grids_are_exact_for_30_and_60_hz(self):
        for rate in (30, 60):
            grid = self.mod.policy_grid_ns(10.0, 11.0, rate)
            self.assertEqual(len(grid), rate + 1)
            self.assertEqual(int(grid[0]), 10_000_000_000)
            self.assertEqual(int(grid[-1]), 11_000_000_000)
            self.assertTrue(np.all(np.diff(grid) > 0))

    @classmethod
    def setUpClass(cls):
        cls.mod = load_converter_module()

    @staticmethod
    def make_raw_msg_bgr(image_bgr: np.ndarray):
        h, w, _ = image_bgr.shape
        return types.SimpleNamespace(
            height=h,
            width=w,
            encoding="bgr8",
            step=w * 3,
            data=image_bgr.astype(np.uint8).tobytes(),
        )

    @staticmethod
    def make_compressed_msg(image_bgr: np.ndarray, ext: str):
        ok, encoded = cv2.imencode(ext, image_bgr)
        assert ok
        return types.SimpleNamespace(
            format=ext,
            data=encoded.tobytes(),
        )

    def make_topics(self, rgb_topic="/rgb"):
        return self.mod.Topics(
            rgb=rgb_topic,
            joint="/joint_states",
            episode="/episode/control",
            current_tcp_s="/middle_line/current_tcp_s",
            goto_s="/trajectory_executor/executed_goto_s",
            goto_s_target_base="/trajectory_executor/executed_goto_s_target_base",
        )

    def make_episode(self, start=0.0, end=0.4):
        return self.mod.EpisodeWindow(source_idx=0, output_idx=0, start=start, end=end)

    def make_sampling_data(
        self,
        tcp_times,
        tcp_values,
        goto_values=None,
        base_time=0.0,
    ):
        rgb0 = np.array([[[0, 0, 255]]], dtype=np.uint8)
        rgb1 = np.array([[[0, 255, 0]]], dtype=np.uint8)
        rgb2 = np.array([[[255, 0, 0]]], dtype=np.uint8)
        rgb_msgs = [
            self.make_raw_msg_bgr(rgb0),
            self.make_raw_msg_bgr(rgb1),
            self.make_raw_msg_bgr(rgb2),
        ]
        data = {
            "rgb_t": [base_time + 0.0, base_time + 0.1, base_time + 0.2],
            "rgb_msg": rgb_msgs,
            "rgb_2d_t": [],
            "rgb_2d_source_t": [],
            "rgb_2d_px": [],
            "joint_t": [base_time + 0.0, base_time + 0.1, base_time + 0.2],
            "qpos": [
                np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
                np.asarray([2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
                np.asarray([3, 4, 5, 6, 7, 8, 9], dtype=np.float32),
            ],
            "current_tcp_s_t": list(tcp_times),
            "current_tcp_s": list(tcp_values),
            "goto_s_t": [],
            "goto_s": [],
            "target_base_t": [],
            "target_base": [],
        }
        if goto_values is not None:
            data["goto_s_t"] = [
                base_time + 0.05 + 0.1 * i for i in range(len(goto_values))
            ]
            data["goto_s"] = list(goto_values)
        return data

    def test_rgb_2d_sampling_is_causal_and_validated(self):
        data = self.make_sampling_data([0.0, 0.1, 0.2], [0.0, 0.1, 0.2])
        data["rgb_2d_t"] = [0.05, 0.15]
        data["rgb_2d_source_t"] = [0.04, 0.14]
        data["rgb_2d_px"] = [[0.25, 0.75], [1.0, 0.5]]
        arrays = self.mod.sample_episode(
            data, self.make_episode(end=0.2), fps=10.0,
            max_current_tcp_s_age_sec=1.0, rgb_2d_enabled=True,
            max_rgb_2d_age_sec=0.10,
        )
        np.testing.assert_array_equal(arrays["rgb_valid"], [0, 1, 0])
        np.testing.assert_array_equal(
            arrays["rgb_2d_px"], [[0.0, 0.0], [0.25, 0.75], [0.0, 0.0]])
        self.assertEqual(arrays["rgb_2d_px"].dtype, np.dtype("f4"))
        self.assertEqual(arrays["rgb_valid"].dtype, np.dtype("u1"))
        np.testing.assert_allclose(
            arrays["rgb_source_timestamps"], [np.nan, 0.04, 0.14],
            equal_nan=True,
        )

        data["rgb_2d_t"] = [0.0, 0.1]
        data["rgb_2d_source_t"] = [0.0, 0.09]
        data["rgb_2d_px"] = [[0.5, 0.5], [np.nan, np.inf]]
        arrays = self.mod.sample_episode(
            data, self.make_episode(end=0.2), fps=10.0,
            max_current_tcp_s_age_sec=1.0, rgb_2d_enabled=True,
            max_rgb_2d_age_sec=0.05,
        )
        np.testing.assert_array_equal(arrays["rgb_valid"], [1, 0, 0])
        np.testing.assert_array_equal(
            arrays["rgb_2d_px"], [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]])

    def test_rgb_2d_write_default_and_opt_out_schema(self):
        data = self.make_sampling_data([0.0, 0.1, 0.2], [0.0, 0.1, 0.2])
        data["rgb_2d_t"] = []
        data["rgb_2d_source_t"] = []
        data["rgb_2d_px"] = []
        episode = self.make_episode(end=0.2)
        topics = self.make_topics()
        with tempfile.TemporaryDirectory() as directory:
            for enabled, filename in ((True, "default.hdf5"), (False, "optout.hdf5")):
                arrays = self.mod.sample_episode(
                    data, episode, fps=10.0, max_current_tcp_s_age_sec=1.0,
                    rgb_2d_enabled=enabled,
                )
                path = os.path.join(directory, filename)
                self.mod.write_episode(
                    path, arrays, episode, topics, fps=10.0,
                    compression="none", overwrite=False,
                )
                with h5py.File(path, "r") as result:
                    if enabled:
                        sparse = result["observations/sparse_tracking"]
                        self.assertEqual(sparse["rgb_2d_px"].shape, (3, 2))
                        self.assertEqual(sparse["rgb_valid"].shape, (3,))
                        self.assertEqual(sparse["rgb_source_timestamps"].shape, (3,))
                        self.assertEqual(sparse["rgb_source_timestamps"].dtype, np.dtype("f8"))
                        self.assertTrue(np.isnan(sparse["rgb_source_timestamps"][:]).all())
                        self.assertEqual(int(np.sum(sparse["rgb_valid"][:])), 0)
                    else:
                        self.assertNotIn("sparse_tracking", result["observations"])

    def test_rgb_timestamp_migration_changes_only_timestamp_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "episode_0.hdf5")
            with h5py.File(path, "w") as target:
                observations = target.create_group("observations")
                observations.create_dataset("timestamps", data=[1.0, 1.1])
                sparse = observations.create_group("sparse_tracking")
                sparse.attrs["sentinel"] = "keep"
                sparse.create_dataset("rgb_2d_px", data=np.asarray([[1, 2], [3, 4]], "f4"))
                sparse.create_dataset("rgb_valid", data=np.asarray([1, 0], "u1"))
                sparse.create_dataset("event_sentinel", data=np.asarray([7], "i2"))
            self.mod.add_rgb_source_timestamps_dataset(path, [0.99, np.nan])
            with h5py.File(path, "r") as target:
                sparse = target["observations/sparse_tracking"]
                np.testing.assert_allclose(
                    sparse["rgb_source_timestamps"][:], [0.99, np.nan], equal_nan=True
                )
                np.testing.assert_array_equal(sparse["rgb_2d_px"][:], [[1, 2], [3, 4]])
                np.testing.assert_array_equal(sparse["rgb_valid"][:], [1, 0])
                np.testing.assert_array_equal(sparse["event_sentinel"][:], [7])
                self.assertEqual(sparse.attrs["sentinel"], "keep")

    @staticmethod
    def make_raw_events_h5(
        path: str,
        event_t_us=None,
        packet_ros_t_ns=None,
        packet_start_event_idx=None,
        packet_end_event_idx=None,
    ):
        if event_t_us is None:
            event_t_us = [780_000, 950_000, 1_050_000, 1_220_000]
        if packet_ros_t_ns is None:
            packet_ros_t_ns = [
                780_000_000,
                950_000_000,
                1_050_000_000,
                1_220_000_000,
            ]
        if packet_start_event_idx is None:
            packet_start_event_idx = [0, 1, 2, 3]
        if packet_end_event_idx is None:
            packet_end_event_idx = [1, 2, 3, 4]

        with h5py.File(path, "w") as h5:
            h5.attrs["width"] = 1
            h5.attrs["height"] = 1
            events = h5.create_group("events")
            packets = h5.create_group("packets")

            events.create_dataset("type", data=np.asarray([1] * len(event_t_us), dtype=np.uint8))
            events.create_dataset("x", data=np.asarray([0] * len(event_t_us), dtype=np.int16))
            events.create_dataset("y", data=np.asarray([0] * len(event_t_us), dtype=np.int16))
            events.create_dataset(
                "t_us",
                data=np.asarray(event_t_us, dtype=np.int64),
            )

            packets.create_dataset(
                "ros_t_ns",
                data=np.asarray(packet_ros_t_ns, dtype=np.int64),
            )
            packets.create_dataset(
                "start_event_idx",
                data=np.asarray(packet_start_event_idx, dtype=np.int64),
            )
            packets.create_dataset(
                "end_event_idx",
                data=np.asarray(packet_end_event_idx, dtype=np.int64),
            )

    def test_decode_raw_image_msg(self):
        image_bgr = np.array([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
        msg = self.make_raw_msg_bgr(image_bgr)
        rgb = self.mod.image_msg_to_rgb(msg)
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(rgb.shape, (1, 2, 3))
        np.testing.assert_array_equal(rgb[0, 0], np.array([255, 0, 0], dtype=np.uint8))
        np.testing.assert_array_equal(rgb[0, 1], np.array([0, 255, 0], dtype=np.uint8))

    def test_decode_compressed_image_msg_png_and_jpeg(self):
        image_bgr = np.array(
            [
                [[0, 0, 255], [0, 255, 0]],
                [[255, 0, 0], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )

        png_msg = self.make_compressed_msg(image_bgr, ".png")
        png_rgb = self.mod.image_msg_to_rgb(png_msg)
        self.assertEqual(png_rgb.dtype, np.uint8)
        self.assertEqual(png_rgb.shape, (2, 2, 3))
        expected_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(png_rgb, expected_rgb)

        jpeg_msg = self.make_compressed_msg(image_bgr, ".jpg")
        jpeg_rgb = self.mod.image_msg_to_rgb(jpeg_msg)
        self.assertEqual(jpeg_rgb.dtype, np.uint8)
        self.assertEqual(jpeg_rgb.shape, (2, 2, 3))

    def test_rgb_auto_selection(self):
        raw_only = {
            "/top_cam/camera/color/image_raw": self.mod.RAW_IMAGE_TYPE,
        }
        compressed_only = {
            "/top_cam/camera/color/image_raw/compressed": self.mod.COMPRESSED_IMAGE_TYPE,
        }
        both = {
            "/top_cam/camera/color/image_raw": self.mod.RAW_IMAGE_TYPE,
            "/top_cam/camera/color/image_raw/compressed": self.mod.COMPRESSED_IMAGE_TYPE,
        }

        self.assertEqual(
            self.mod.resolve_rgb_topic(raw_only, "auto"),
            "/top_cam/camera/color/image_raw",
        )
        self.assertEqual(
            self.mod.resolve_rgb_topic(compressed_only, "auto"),
            "/top_cam/camera/color/image_raw/compressed",
        )
        self.assertEqual(
            self.mod.resolve_rgb_topic(both, "auto"),
            "/top_cam/camera/color/image_raw/compressed",
        )

    def test_current_tcp_s_required(self):
        topics = self.make_topics()
        types_map = {
            "/rgb": self.mod.RAW_IMAGE_TYPE,
            "/joint_states": "sensor_msgs/msg/JointState",
            "/episode/control": "std_msgs/msg/UInt8",
        }
        with self.assertRaises(RuntimeError):
            self.mod.validate_topics(
                types_map,
                topics,
                collect_goto_s_debug=True,
                collect_target_base_debug=True,
            )

    def test_executed_goto_s_not_required(self):
        topics = self.make_topics()
        types_map = {
            "/rgb": self.mod.RAW_IMAGE_TYPE,
            "/joint_states": "sensor_msgs/msg/JointState",
            "/episode/control": "std_msgs/msg/UInt8",
            "/middle_line/current_tcp_s": self.mod.FLOAT64_TYPE,
        }
        collect_goto_s, collect_target_base = self.mod.validate_topics(
            types_map,
            topics,
            collect_goto_s_debug=True,
            collect_target_base_debug=True,
        )
        self.assertFalse(collect_goto_s)
        self.assertFalse(collect_target_base)

    def test_causal_tcp_sampling_latest_at_or_before(self):
        data = self.make_sampling_data(
            tcp_times=[0.0, 0.1, 0.2],
            tcp_values=[-0.5, 0.25, 0.75],
        )
        arrays = self.mod.sample_episode(
            data=data,
            episode=self.make_episode(),
            fps=10.0,
            max_current_tcp_s_age_sec=0.2,
        )
        np.testing.assert_allclose(
            arrays["timestamps"], np.array([0.0, 0.1, 0.2], dtype=np.float64)
        )
        np.testing.assert_allclose(
            arrays["action"][:, 0], np.array([-0.5, 0.25, 0.75], dtype=np.float32)
        )

    def test_current_tcp_s_constrains_usable_interval(self):
        data = self.make_sampling_data(
            tcp_times=[0.1, 0.2],
            tcp_values=[1.0, 2.0],
        )
        arrays = self.mod.sample_episode(
            data=data,
            episode=self.make_episode(),
            fps=10.0,
            max_current_tcp_s_age_sec=0.2,
        )
        np.testing.assert_allclose(
            arrays["timestamps"], np.array([0.1, 0.2], dtype=np.float64)
        )

    def test_stale_current_tcp_s_rejected(self):
        data = self.make_sampling_data(
            tcp_times=[0.0],
            tcp_values=[0.0],
        )
        with self.assertRaises(RuntimeError):
            self.mod.sample_episode(
                data=data,
                episode=self.make_episode(),
                fps=10.0,
                max_current_tcp_s_age_sec=0.05,
            )

    def test_non_finite_tcp_rejected(self):
        data = self.make_sampling_data(
            tcp_times=[0.0, 0.1, 0.2],
            tcp_values=[0.0, np.nan, 1.0],
        )
        with self.assertRaises(RuntimeError):
            self.mod.sample_episode(
                data=data,
                episode=self.make_episode(),
                fps=10.0,
                max_current_tcp_s_age_sec=0.2,
            )

    def test_negative_and_positive_values_unchanged_not_clamped(self):
        data = self.make_sampling_data(
            tcp_times=[0.0, 0.1, 0.2],
            tcp_values=[-5.0, 0.0, 7.5],
        )
        arrays = self.mod.sample_episode(
            data=data,
            episode=self.make_episode(),
            fps=10.0,
            max_current_tcp_s_age_sec=0.2,
        )
        np.testing.assert_allclose(
            arrays["action"][:, 0], np.array([-5.0, 0.0, 7.5], dtype=np.float32)
        )

    def test_episode_without_goto_debug_still_writes(self):
        data = self.make_sampling_data(
            tcp_times=[0.0, 0.1, 0.2],
            tcp_values=[-1.0, 0.5, 2.0],
            goto_values=None,
        )
        arrays = self.mod.sample_episode(
            data=data,
            episode=self.make_episode(),
            fps=10.0,
            max_current_tcp_s_age_sec=0.2,
        )
        self.assertEqual(arrays["command_timestamps"].size, 0)

        topics = self.make_topics()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "episode_0.hdf5")
            self.mod.write_episode(
                output_path=out_path,
                arrays=arrays,
                episode=self.make_episode(),
                topics=topics,
                fps=10.0,
                compression="none",
                overwrite=False,
            )
            with h5py.File(out_path, "r") as h5:
                self.assertIn("commands", h5)
                self.assertIn("goto_s", h5["commands"])
                self.assertEqual(int(h5.attrs["command_count"]), 0)

    def test_written_action_shape_and_metadata_and_legacy_absence(self):
        data = self.make_sampling_data(
            tcp_times=[0.0, 0.1, 0.2],
            tcp_values=[-3.0, 1.0, 4.0],
            goto_values=None,
        )
        arrays = self.mod.sample_episode(
            data=data,
            episode=self.make_episode(),
            fps=10.0,
            max_current_tcp_s_age_sec=0.2,
        )

        topics = self.make_topics()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "episode_0.hdf5")
            self.mod.write_episode(
                output_path=out_path,
                arrays=arrays,
                episode=self.make_episode(),
                topics=topics,
                fps=10.0,
                compression="none",
                overwrite=False,
            )
            with h5py.File(out_path, "r") as h5:
                self.assertEqual(h5["action"].shape, (3, 1))
                self.assertIn("action_source_timestamps", h5)
                self.assertIn("action_source_age_sec", h5)
                self.assertNotIn("action_is_commanded", h5)

                self.assertNotIn("precommand_action_policy", h5.attrs)
                self.assertNotIn("precommand_action_value", h5.attrs)
                self.assertEqual(
                    h5.attrs["action_positive_direction"],
                    "robot_base_positive_x",
                )

                self.assertEqual(h5["action"].shape[0], h5["observations/timestamps"].shape[0])
                self.assertEqual(h5["action"].shape[0], h5["observations/qpos"].shape[0])
                self.assertEqual(
                    h5["action"].shape[0],
                    h5["observations/images/rgb"].shape[0],
                )

    def test_event_sidecar_shape_dtype_and_non_negative_age(self):
        data = self.make_sampling_data(
            tcp_times=[1.0, 1.1, 1.2],
            tcp_values=[-0.5, 0.25, 0.75],
            base_time=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = os.path.join(tmpdir, "raw_events.h5")
            self.make_raw_events_h5(sidecar_path)
            store = self.mod.RawEventStore(sidecar_path, logger=lambda *_: None)
            try:
                arrays = self.mod.sample_episode(
                    data=data,
                    episode=self.make_episode(start=1.0, end=1.4),
                    fps=10.0,
                    max_current_tcp_s_age_sec=0.2,
                    raw_event_store=store,
                    event_frame_windows_ms=(50.0, 100.0, 200.0),
                    event_frame_mode="shifted",
                    event_clip_count=4.0,
                )
            finally:
                store.close()

        self.assertIn("event", arrays)
        self.assertEqual(arrays["event"].shape, (3, 1, 1, 3))
        self.assertEqual(arrays["event"].dtype, np.uint8)
        self.assertTrue(np.all(arrays["event_source_age_sec"] >= 0.0))
        self.assertEqual(arrays["event_count_per_channel"].shape, (3, 3))

    def test_xyt_sidecar_storage_and_metadata(self):
        data = self.make_sampling_data(
            tcp_times=[1.0, 1.1, 1.2],
            tcp_values=[-0.5, 0.25, 0.75],
            base_time=1.0,
        )
        episode = self.make_episode(start=1.0, end=1.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = os.path.join(tmpdir, "raw_events.h5")
            self.make_raw_events_h5(sidecar_path)
            store = self.mod.RawEventStore(sidecar_path, logger=lambda *_: None)
            try:
                arrays = self.mod.sample_episode(
                    data=data,
                    episode=episode,
                    fps=10.0,
                    max_current_tcp_s_age_sec=0.2,
                    raw_event_store=store,
                    event_representation="xyt_signed_voxel_v1",
                    event_horizon_ms=200.0,
                    event_temporal_bins=9,
                    event_output_height=2,
                    event_output_width=3,
                    event_clip_count=16.0,
                )
            finally:
                store.close()

            self.assertEqual(arrays["event"].shape, (3, 2, 3, 9))
            self.assertEqual(arrays["event"].dtype, np.uint8)
            self.assertEqual(arrays["event_count_per_channel"].shape, (3, 9))

            out_path = os.path.join(tmpdir, "episode_0.hdf5")
            self.mod.write_episode(
                output_path=out_path,
                arrays=arrays,
                episode=episode,
                topics=self.make_topics(),
                fps=10.0,
                compression="none",
                overwrite=False,
                raw_events_h5=sidecar_path,
                event_clip_count=16.0,
                event_representation="xyt_signed_voxel_v1",
                event_horizon_ms=200.0,
                event_temporal_bins=9,
                event_output_height=2,
                event_output_width=3,
            )
            with h5py.File(out_path, "r") as h5:
                self.assertEqual(h5["observations/images/event"].shape, (3, 2, 3, 9))
                self.assertEqual(h5.attrs["event_representation"], "xyt_signed_voxel_v1")
                self.assertAlmostEqual(h5.attrs["event_bin_width_ms"], 200.0 / 9.0)
                self.assertEqual(h5.attrs["event_channel_order"], "oldest_to_newest")
                self.assertEqual(h5.attrs["event_spatial_height"], 2)
                self.assertEqual(h5.attrs["event_spatial_width"], 3)
                np.testing.assert_array_equal(h5.attrs["visual_history_offsets"], [0])
                np.testing.assert_array_equal(h5.attrs["qpos_history_offsets"], [-6, -3, 0])
                self.assertEqual(h5.attrs["image_channels"], 9)

    def test_event_empty_window_produces_neutral_128(self):
        data = self.make_sampling_data(
            tcp_times=[1.0, 1.1, 1.2],
            tcp_values=[-0.5, 0.25, 0.75],
            base_time=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = os.path.join(tmpdir, "raw_events.h5")
            self.make_raw_events_h5(
                sidecar_path,
                # Include one packet event that is too old for shifted middle bin
                # while preserving required sidecar coverage interval.
                event_t_us=[700_000, 1_000_000, 1_200_000],
                packet_ros_t_ns=[800_000_000, 1_000_000_000, 1_200_000_000],
                packet_start_event_idx=[0, 1, 2],
                packet_end_event_idx=[1, 2, 3],
            )
            store = self.mod.RawEventStore(sidecar_path, logger=lambda *_: None)
            try:
                arrays = self.mod.sample_episode(
                    data=data,
                    episode=self.make_episode(start=1.0, end=1.4),
                    fps=10.0,
                    max_current_tcp_s_age_sec=0.2,
                    raw_event_store=store,
                    event_frame_windows_ms=(50.0, 100.0, 200.0),
                    event_frame_mode="shifted",
                    event_clip_count=4.0,
                )
            finally:
                store.close()

        # For the first sampled frame, shifted channel 1 has no events and stays neutral.
        self.assertEqual(int(arrays["event"][0, 0, 0, 1]), 128)

    def test_existing_arrays_unchanged_when_event_sidecar_added(self):
        data = self.make_sampling_data(
            tcp_times=[1.0, 1.1, 1.2],
            tcp_values=[-0.5, 0.25, 0.75],
            base_time=1.0,
        )
        episode = self.make_episode(start=1.0, end=1.4)
        without_events = self.mod.sample_episode(
            data=data,
            episode=episode,
            fps=10.0,
            max_current_tcp_s_age_sec=0.2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = os.path.join(tmpdir, "raw_events.h5")
            self.make_raw_events_h5(sidecar_path)
            store = self.mod.RawEventStore(sidecar_path, logger=lambda *_: None)
            try:
                with_events = self.mod.sample_episode(
                    data=data,
                    episode=episode,
                    fps=10.0,
                    max_current_tcp_s_age_sec=0.2,
                    raw_event_store=store,
                    event_frame_windows_ms=(50.0, 100.0, 200.0),
                    event_frame_mode="shifted",
                    event_clip_count=4.0,
                )
            finally:
                store.close()

        np.testing.assert_allclose(
            without_events["timestamps"],
            with_events["timestamps"],
        )
        np.testing.assert_array_equal(
            without_events["rgb"],
            with_events["rgb"],
        )
        np.testing.assert_allclose(
            without_events["qpos"],
            with_events["qpos"],
        )
        np.testing.assert_allclose(
            without_events["action"],
            with_events["action"],
        )
        np.testing.assert_allclose(
            without_events["action_source_timestamps"],
            with_events["action_source_timestamps"],
        )
        np.testing.assert_allclose(
            without_events["action_source_age_sec"],
            with_events["action_source_age_sec"],
        )
        np.testing.assert_allclose(
            without_events["command_timestamps"],
            with_events["command_timestamps"],
        )
        np.testing.assert_array_equal(
            without_events["command_values"],
            with_events["command_values"],
        )

    def test_write_episode_with_event_sidecar_writes_expected_schema(self):
        data = self.make_sampling_data(
            tcp_times=[1.0, 1.1, 1.2],
            tcp_values=[-0.5, 0.25, 0.75],
            base_time=1.0,
        )
        topics = self.make_topics()
        episode = self.make_episode(start=1.0, end=1.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = os.path.join(tmpdir, "raw_events.h5")
            self.make_raw_events_h5(sidecar_path)
            store = self.mod.RawEventStore(sidecar_path, logger=lambda *_: None)
            try:
                arrays = self.mod.sample_episode(
                    data=data,
                    episode=episode,
                    fps=10.0,
                    max_current_tcp_s_age_sec=0.2,
                    raw_event_store=store,
                    event_frame_windows_ms=(50.0, 100.0, 200.0),
                    event_frame_mode="shifted",
                    event_clip_count=16.0,
                )
            finally:
                store.close()

            out_path = os.path.join(tmpdir, "episode_0.hdf5")
            self.mod.write_episode(
                output_path=out_path,
                arrays=arrays,
                episode=episode,
                topics=topics,
                fps=10.0,
                compression="gzip",
                overwrite=False,
                raw_events_h5=sidecar_path,
                event_frame_windows_ms=(50.0, 100.0, 200.0),
                event_frame_mode="shifted",
                event_clip_count=16.0,
            )

            with h5py.File(out_path, "r") as h5:
                self.assertIn("observations/images/event", h5)
                self.assertEqual(
                    h5["observations/images/event"].dtype,
                    np.dtype(np.uint8),
                )
                self.assertEqual(
                    h5["observations/images/event"].shape[0],
                    h5["observations/timestamps"].shape[0],
                )
                self.assertIn("event_source_timestamps", h5)
                self.assertIn("event_source_age_sec", h5)
                self.assertIn("event_count_per_channel", h5)
                self.assertEqual(
                    h5.attrs["event_sampling_policy"],
                    "latest_packet_at_or_before_grid_time",
                )
                self.assertEqual(h5.attrs["event_neutral_u8"], 128)
                self.assertEqual(h5.attrs["event_representation"], "shifted_3chef_signed")
                self.assertEqual(h5.attrs["event_frame_mode"], "shifted")

    def test_sidecar_coverage_start_too_late_is_rejected(self):
        data = self.make_sampling_data(
            tcp_times=[1.0, 1.1, 1.2],
            tcp_values=[0.0, 0.1, 0.2],
            base_time=1.0,
        )
        episode = self.make_episode(start=1.0, end=1.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = os.path.join(tmpdir, "raw_events.h5")
            # First packet at 0.85s, but required start is 0.8s.
            self.make_raw_events_h5(
                sidecar_path,
                packet_ros_t_ns=[850_000_000, 950_000_000, 1_050_000_000],
                packet_start_event_idx=[0, 1, 2],
                packet_end_event_idx=[1, 2, 3],
                event_t_us=[850_000, 950_000, 1_050_000],
            )
            store = self.mod.RawEventStore(sidecar_path, logger=lambda *_: None)
            try:
                with self.assertRaises(RuntimeError):
                    self.mod.sample_episode(
                        data=data,
                        episode=episode,
                        fps=10.0,
                        max_current_tcp_s_age_sec=0.2,
                        raw_event_store=store,
                        event_frame_windows_ms=(50.0, 100.0, 200.0),
                        event_frame_mode="shifted",
                        event_clip_count=4.0,
                    )
            finally:
                store.close()

    def test_sidecar_coverage_end_too_early_is_rejected(self):
        data = self.make_sampling_data(
            tcp_times=[1.0, 1.1, 1.2],
            tcp_values=[0.0, 0.1, 0.2],
            base_time=1.0,
        )
        episode = self.make_episode(start=1.0, end=1.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = os.path.join(tmpdir, "raw_events.h5")
            self.make_raw_events_h5(
                sidecar_path,
                packet_ros_t_ns=[780_000_000, 950_000_000, 1_150_000_000],
                packet_start_event_idx=[0, 1, 2],
                packet_end_event_idx=[1, 2, 3],
                event_t_us=[780_000, 950_000, 1_150_000],
            )
            store = self.mod.RawEventStore(sidecar_path, logger=lambda *_: None)
            try:
                with self.assertRaises(RuntimeError):
                    self.mod.sample_episode(
                        data=data,
                        episode=episode,
                        fps=10.0,
                        max_current_tcp_s_age_sec=0.2,
                        raw_event_store=store,
                        event_frame_windows_ms=(50.0, 100.0, 200.0),
                        event_frame_mode="shifted",
                        event_clip_count=4.0,
                    )
            finally:
                store.close()

    def test_missing_causal_packet_raises_instead_of_faking_source_timestamp(self):
        data = self.make_sampling_data(
            tcp_times=[1.0, 1.1, 1.2],
            tcp_values=[0.0, 0.1, 0.2],
            base_time=1.0,
        )
        episode = self.make_episode(start=1.0, end=1.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = os.path.join(tmpdir, "raw_events.h5")
            # Coverage exists at both ends, but there is a large gap causing no causal packet
            # for mid-grid timestamps when search window starts after the earlier packet.
            self.make_raw_events_h5(
                sidecar_path,
                packet_ros_t_ns=[800_000_000, 1_300_000_000],
                packet_start_event_idx=[0, 1],
                packet_end_event_idx=[1, 2],
                event_t_us=[800_000, 1_300_000],
            )
            store = self.mod.RawEventStore(sidecar_path, logger=lambda *_: None)
            try:
                with self.assertRaises(RuntimeError):
                    self.mod.sample_episode(
                        data=data,
                        episode=episode,
                        fps=10.0,
                        max_current_tcp_s_age_sec=0.2,
                        raw_event_store=store,
                        event_frame_windows_ms=(50.0, 100.0, 200.0),
                        event_frame_mode="shifted",
                        event_clip_count=4.0,
                    )
            finally:
                store.close()

    def test_event_source_timestamps_are_causal_and_ages_match_difference(self):
        data = self.make_sampling_data(
            tcp_times=[1.0, 1.1, 1.2],
            tcp_values=[0.0, 0.1, 0.2],
            base_time=1.0,
        )
        episode = self.make_episode(start=1.0, end=1.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = os.path.join(tmpdir, "raw_events.h5")
            self.make_raw_events_h5(sidecar_path)
            store = self.mod.RawEventStore(sidecar_path, logger=lambda *_: None)
            try:
                arrays = self.mod.sample_episode(
                    data=data,
                    episode=episode,
                    fps=10.0,
                    max_current_tcp_s_age_sec=0.2,
                    raw_event_store=store,
                    event_frame_windows_ms=(50.0, 100.0, 200.0),
                    event_frame_mode="shifted",
                    event_clip_count=4.0,
                )
            finally:
                store.close()

        grid_ts = arrays["timestamps"]
        src_ts = arrays["event_source_timestamps"]
        age_sec = arrays["event_source_age_sec"]
        packet_ts_set_ns = {780_000_000, 950_000_000, 1_050_000_000, 1_220_000_000}
        for idx in range(grid_ts.size):
            self.assertLessEqual(src_ts[idx], grid_ts[idx])
            src_ns = int(np.rint(src_ts[idx] * 1e9))
            self.assertIn(src_ns, packet_ts_set_ns)
            self.assertAlmostEqual(float(age_sec[idx]), float(grid_ts[idx] - src_ts[idx]), places=6)

    def test_cumulative_mode_is_rejected_by_parse_args(self):
        argv = list(sys.argv)
        try:
            sys.argv = [
                "bag_to_il_intercept.py",
                "--bag",
                "/tmp/bag",
                "--out_dir",
                "/tmp/out",
                "--event_frame_mode",
                "cumulative",
            ]
            with self.assertRaises(SystemExit):
                self.mod.parse_args()
        finally:
            sys.argv = argv

    def test_xyt_representation_arguments_are_accepted(self):
        argv = list(sys.argv)
        try:
            sys.argv = [
                "bag_to_il_intercept.py",
                "--bag",
                "/tmp/bag",
                "--out_dir",
                "/tmp/out",
                "--event_representation",
                "xyt_signed_voxel_v1",
                "--event_horizon_ms",
                "200",
                "--event_temporal_bins",
                "9",
                "--event_output_height",
                "320",
                "--event_output_width",
                "320",
                "--event_clip_count",
                "16",
            ]
            args = self.mod.parse_args()
            self.assertEqual(args.event_representation, "xyt_signed_voxel_v1")
            self.assertEqual(args.event_temporal_bins, 9)
            self.assertEqual(args.event_output_height, 320)
            self.assertEqual(args.event_output_width, 320)
        finally:
            sys.argv = argv

    def test_fps_and_event_clip_count_defaults(self):
        argv = list(sys.argv)
        try:
            sys.argv = [
                "bag_to_il_intercept.py",
                "--bag", "/tmp/bag",
                "--out_dir", "/tmp/out",
            ]
            args = self.mod.parse_args()
            self.assertEqual(args.fps, 30.0)
            self.assertEqual(args.event_clip_count, 16.0)
        finally:
            sys.argv = argv

    def test_out_dir_is_the_exact_episode_destination(self):
        argv = list(sys.argv)
        try:
            sys.argv = [
                "bag_to_il_intercept.py",
                "--rec_dir",
                "/data/recording_20260727_215525",
                "--out_dir",
                "/data/position_only/recording_20260727_215525_position_only",
            ]
            args = self.mod.parse_args()
            self.assertEqual(
                self.mod.resolve_output_dir(args.out_dir),
                "/data/position_only/recording_20260727_215525_position_only",
            )
        finally:
            sys.argv = argv

    def test_write_episode_rejects_non_shifted_event_metadata_mode(self):
        data = self.make_sampling_data(
            tcp_times=[1.0, 1.1, 1.2],
            tcp_values=[0.0, 0.1, 0.2],
            base_time=1.0,
        )
        episode = self.make_episode(start=1.0, end=1.4)
        topics = self.make_topics()

        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = os.path.join(tmpdir, "raw_events.h5")
            self.make_raw_events_h5(sidecar_path)
            store = self.mod.RawEventStore(sidecar_path, logger=lambda *_: None)
            try:
                arrays = self.mod.sample_episode(
                    data=data,
                    episode=episode,
                    fps=10.0,
                    max_current_tcp_s_age_sec=0.2,
                    raw_event_store=store,
                    event_frame_windows_ms=(50.0, 100.0, 200.0),
                    event_frame_mode="shifted",
                    event_clip_count=4.0,
                )
            finally:
                store.close()

            with self.assertRaises(RuntimeError):
                self.mod.write_episode(
                    output_path=os.path.join(tmpdir, "episode_0.hdf5"),
                    arrays=arrays,
                    episode=episode,
                    topics=topics,
                    fps=10.0,
                    compression="none",
                    overwrite=False,
                    raw_events_h5=sidecar_path,
                    event_frame_windows_ms=(50.0, 100.0, 200.0),
                    event_frame_mode="cumulative",
                    event_clip_count=4.0,
                )

    def test_resolve_input_paths_allows_rgb_only_when_sidecar_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_dir = os.path.join(tmpdir, "recording_20260728_225008")
            os.makedirs(rec_dir)
            bag_path = os.path.join(rec_dir, "recording_20260728_225008_bag")
            os.makedirs(bag_path)

            args = types.SimpleNamespace(
                bag=None,
                rec_dir=rec_dir,
                raw_events_h5=None,
                event_clip_count=16.0,
            )

            with mock.patch.object(
                self.mod,
                "resolve_recording_dir",
                return_value=(bag_path, None, "recording_20260728_225008"),
            ):
                resolved_bag_path, resolved_sidecar_path = self.mod.resolve_input_paths(args)

            self.assertEqual(resolved_bag_path, bag_path)
            self.assertIsNone(resolved_sidecar_path)

    def test_apply_storage_filter_sets_expected_topics(self):
        captured = {}

        class Reader:
            def set_filter(self, storage_filter):
                captured["topics"] = list(storage_filter.topics)

        filtered = self.mod.apply_storage_filter(
            Reader(),
            ["/episode/control", "/episode/control", "/joint_states"],
        )
        self.assertEqual(filtered, ["/episode/control", "/joint_states"])
        self.assertEqual(captured["topics"], ["/episode/control", "/joint_states"])

    def test_validate_non_overlapping_windows_rejects_overlap(self):
        windows = [
            self.mod.EpisodeWindow(source_idx=0, output_idx=0, start=1.0, end=2.0),
            self.mod.EpisodeWindow(source_idx=1, output_idx=1, start=2.0, end=3.0),
        ]
        with self.assertRaises(RuntimeError):
            self.mod.validate_non_overlapping_windows(windows)

    def test_main_streams_across_two_episodes_with_boundary_and_gap_routing(self):
        class FakeReader:
            def __init__(self, messages, types_map):
                self._messages = list(messages)
                self._index = 0
                self._allowed_topics = None
                self.types_map = dict(types_map)
                self.applied_filters = []

            def open(self, *_args, **_kwargs):
                return None

            def get_all_topics_and_types(self):
                return [
                    types.SimpleNamespace(name=name, type=type_name)
                    for name, type_name in self.types_map.items()
                ]

            def set_filter(self, storage_filter):
                self._allowed_topics = set(storage_filter.topics)
                self.applied_filters.append(list(storage_filter.topics))

            def _next_visible_index(self):
                i = self._index
                while i < len(self._messages):
                    topic = self._messages[i][0]
                    if self._allowed_topics is None or topic in self._allowed_topics:
                        return i
                    i += 1
                return None

            def has_next(self):
                return self._next_visible_index() is not None

            def read_next(self):
                nxt = self._next_visible_index()
                if nxt is None:
                    raise StopIteration
                self._index = nxt + 1
                return self._messages[nxt]

        types_map = {
            "/rgb": self.mod.RAW_IMAGE_TYPE,
            "/joint_states": "sensor_msgs/msg/JointState",
            "/episode/control": "std_msgs/msg/UInt8",
            "/middle_line/current_tcp_s": self.mod.FLOAT64_TYPE,
            "/trajectory_executor/executed_goto_s": self.mod.FLOAT64_TYPE,
            "/trajectory_executor/executed_goto_s_target_base": "geometry_msgs/msg/PointStamped",
        }

        marker_msgs = [
            ("/episode/control", types.SimpleNamespace(data=1), int(1.0e9)),
            ("/episode/control", types.SimpleNamespace(data=2), int(2.0e9)),
            ("/episode/control", types.SimpleNamespace(data=1), int(4.0e9)),
            ("/episode/control", types.SimpleNamespace(data=2), int(5.0e9)),
        ]

        joint_msg = lambda offset: types.SimpleNamespace(name=list(self.mod.ARM_JOINT_NAMES), position=[offset + i for i in range(7)])
        target_msg = lambda: types.SimpleNamespace(point=types.SimpleNamespace(x=0.1, y=0.2, z=0.3))
        rgb_msg = self.make_raw_msg_bgr(np.array([[[0, 0, 255]]], dtype=np.uint8))

        data_msgs = [
            ("/rgb", rgb_msg, int(0.5e9)),
            ("/joint_states", joint_msg(0.5), int(0.5e9)),
            ("/middle_line/current_tcp_s", types.SimpleNamespace(data=0.5), int(0.5e9)),
            ("/rgb", rgb_msg, int(1.0e9)),
            ("/joint_states", joint_msg(1.0), int(1.0e9)),
            ("/middle_line/current_tcp_s", types.SimpleNamespace(data=1.0), int(1.0e9)),
            ("/trajectory_executor/executed_goto_s", types.SimpleNamespace(data=1.0), int(1.0e9)),
            ("/trajectory_executor/executed_goto_s_target_base", target_msg(), int(1.0e9)),
            ("/rgb", rgb_msg, int(2.0e9)),
            ("/joint_states", joint_msg(2.0), int(2.0e9)),
            ("/middle_line/current_tcp_s", types.SimpleNamespace(data=2.0), int(2.0e9)),
            ("/rgb", rgb_msg, int(3.0e9)),
            ("/joint_states", joint_msg(3.0), int(3.0e9)),
            ("/middle_line/current_tcp_s", types.SimpleNamespace(data=3.0), int(3.0e9)),
            ("/rgb", rgb_msg, int(4.0e9)),
            ("/joint_states", joint_msg(4.0), int(4.0e9)),
            ("/middle_line/current_tcp_s", types.SimpleNamespace(data=4.0), int(4.0e9)),
            ("/rgb", rgb_msg, int(5.0e9)),
            ("/joint_states", joint_msg(5.0), int(5.0e9)),
            ("/middle_line/current_tcp_s", types.SimpleNamespace(data=5.0), int(5.0e9)),
        ]

        marker_reader = FakeReader(marker_msgs, types_map)
        data_reader = FakeReader(data_msgs, types_map)
        finalize_calls = []

        def fake_finalize_episode(**kwargs):
            ep = kwargs["episode"]
            data = kwargs["data"]
            finalize_calls.append(
                {
                    "episode": ep.output_idx,
                    "rgb_t": list(data["rgb_t"]),
                    "joint_t": list(data["joint_t"]),
                    "tcp_t": list(data["current_tcp_s_t"]),
                }
            )
            data.clear()
            return len(finalize_calls)

        args = types.SimpleNamespace(
            bag="/tmp/fake_bag",
            rec_dir=None,
            out_dir="/tmp/fake_out",
            storage_id="mcap",
            fps=30.0,
            min_duration=0.0,
            max_episodes=None,
            max_current_tcp_s_age_sec=0.10,
            no_target_base=False,
            compression="none",
            overwrite=False,
            raw_events_h5=None,
            event_frame_windows_ms=[50.0, 100.0, 200.0],
            event_frame_mode="shifted",
            event_clip_count=None,
            event_packet_margin_ms=50.0,
            rgb_topic="/rgb",
            joint_topic="/joint_states",
            episode_topic="/episode/control",
            current_tcp_s_topic="/middle_line/current_tcp_s",
            goto_s_topic="/trajectory_executor/executed_goto_s",
            goto_s_target_base_topic="/trajectory_executor/executed_goto_s_target_base",
        )

        with mock.patch.object(self.mod, "parse_args", return_value=args), \
             mock.patch.object(self.mod, "resolve_input_paths", return_value=("/tmp/fake_bag", None)), \
             mock.patch.object(self.mod, "open_reader", side_effect=[marker_reader, data_reader]) as open_reader_mock, \
             mock.patch.object(self.mod, "finalize_episode", side_effect=fake_finalize_episode), \
             mock.patch.object(self.mod, "deserialize_message", side_effect=lambda raw, _cls: raw), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os, "makedirs", return_value=None):
            self.mod.main()

        self.assertEqual(open_reader_mock.call_count, 2)
        self.assertEqual(marker_reader.applied_filters[-1], ["/episode/control"])
        self.assertEqual(
            data_reader.applied_filters[-1],
            [
                "/rgb",
                "/joint_states",
                "/middle_line/current_tcp_s",
                "/trajectory_executor/executed_goto_s",
                "/trajectory_executor/executed_goto_s_target_base",
            ],
        )
        self.assertEqual(len(finalize_calls), 2)
        self.assertEqual(finalize_calls[0]["episode"], 0)
        self.assertEqual(finalize_calls[1]["episode"], 1)
        self.assertEqual(finalize_calls[0]["rgb_t"], [1.0, 2.0])
        self.assertEqual(finalize_calls[1]["rgb_t"], [4.0, 5.0])
        self.assertNotIn(3.0, finalize_calls[0]["rgb_t"])
        self.assertNotIn(3.0, finalize_calls[1]["rgb_t"])

    def test_openmv_api_and_default_config_resolution(self):
        api = self.mod.import_openmv_tracker_api(
            "/home/dyros/jg_ws/src/openmv_cam"
        )
        for name in self.mod.OPENMV_REQUIRED_API:
            self.assertTrue(hasattr(api, name))
        args = types.SimpleNamespace(
            event_tracker_config=None, rec_dir="/tmp/no-recording-config"
        )
        config, config_json, pre_roll_ms, source = (
            self.mod.resolve_event_tracker_config(args, "recording_test", api)
        )
        self.assertEqual(config["width"], 320)
        self.assertEqual(config["height"], 320)
        self.assertEqual(pre_roll_ms, 0.0)
        self.assertIn("offline_tracker_example.json", source)
        self.assertEqual(json.loads(config_json), config)

    def test_event_alignment_is_causal_held_fresh_and_duplicate_stable(self):
        api = self.mod.import_openmv_tracker_api(
            "/home/dyros/jg_ws/src/openmv_cam"
        )

        def update(t, valid, x, reason):
            return types.SimpleNamespace(
                available_ros_t_ns=t, packet_id=t, sensor_window_start_us=0,
                sensor_window_end_us=1, x_px=x, y_px=x + 1,
                vx_px_s=2, vy_px_s=3, speed_px_s=4, confidence=.8,
                valid=valid, velocity_valid=True, window_event_count=10,
                candidate_count=1, blob_area_px=20, blob_event_count=10,
                blob_width_px=4, blob_height_px=5, circularity=.5,
                rejection_reason=reason,
            )

        updates = [
            update(100, True, 1, ""),
            update(150, False, 0, "noise"),
            update(150, True, 2, ""),  # last duplicate wins
            update(400, True, 9, ""),  # future for all tested grid rows
        ]
        aligned = api.align_tracker_updates_to_policy_grid(
            updates, np.asarray([90, 120, 160, 300], dtype=np.int64), 1e-7
        )
        np.testing.assert_array_equal(aligned["event_valid"], [0, 1, 1, 0])
        np.testing.assert_array_equal(aligned["event_2d_px"][1:3], [[1, 2], [2, 3]])
        self.assertEqual(aligned["event_latest_update_timestamp_ns"][2], 150)
        self.assertNotIn(9, aligned["event_2d_px"][:, 0])

    def test_rgb_and_event_sparse_streams_write_additively(self):
        api = self.mod.import_openmv_tracker_api(
            "/home/dyros/jg_ws/src/openmv_cam"
        )
        data = self.make_sampling_data([0.0, 0.1, 0.2], [0.0, 0.1, 0.2])
        data["rgb_2d_t"] = [0.0]
        data["rgb_2d_source_t"] = [0.0]
        data["rgb_2d_px"] = [[0.0, 0.0]]
        row = api.TrackerUpdate(
            "episode_0", 0, 0, 0, 1, 0, 1, 1.0, 1.0, 0.0, 0.0,
            0.0, 1.0, True, False, 10, 1, 4, 10, 2, 2, 1.0, "",
        )
        arrays = self.mod.sample_episode(
            data, self.make_episode(end=0.2), fps=10.0,
            max_current_tcp_s_age_sec=1.0, rgb_2d_enabled=True,
            event_tracker_updates=[row],
            event_tracker_aligner=api.align_tracker_updates_to_policy_grid,
            max_observation_age_sec=0.1,
        )
        metadata = {
            "schema_version": "event_tracker_updates_v2",
            "sensor_width": 320, "sensor_height": 320,
            "tracker_config_json": "{}", "tracker_config_hash": "hash",
            "raw_events_h5": "/raw.h5", "tracker_code_version": "5336fa8",
            "availability_timestamp_domain": "packet_ros_t_ns",
            "sensor_timestamp_domain": "genx320_microseconds",
            "pre_roll_ms": 0.0, "max_observation_age_sec": 0.1,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "episode_0.hdf5")
            self.mod.write_episode(
                output, arrays, self.make_episode(end=0.2), self.make_topics(),
                10.0, "none", False, event_tracker_metadata=metadata,
            )
            with h5py.File(output, "r") as target:
                sparse = target["observations/sparse_tracking"]
                for name in ("rgb_2d_px", "rgb_valid", "event_2d_px",
                             "event_valid", "event_latest_rejection_reason"):
                    self.assertIn(name, sparse)
                self.assertIn("images/rgb", target["observations"])
                self.assertIn("processing/event_tracker", target)
                self.assertEqual(len(sparse["rgb_valid"]), len(sparse["event_valid"]))
                np.testing.assert_array_equal(
                    sparse["event_2d_px"][:][sparse["event_valid"][:] == 0], 0
                )

    def test_failed_write_removes_temporary_file(self):
        arrays = {
            "timestamps": np.asarray([0.0]),
            "timestamps_ns": np.asarray([0], dtype=np.int64),
            "rgb": np.zeros((1, 1, 1, 3), dtype=np.uint8),
            "qpos": np.zeros((1, 7), dtype=np.float32),
            "action": np.zeros((1, 1), dtype=np.float32),
            "action_source_timestamps": np.asarray([0.0]),
            "action_source_age_sec": np.asarray([0.0], dtype=np.float32),
            "command_timestamps": np.empty(0),
            "command_values": np.empty((0, 1), dtype=np.float32),
            "target_base_timestamps": np.empty(0),
            "target_base_points": np.empty((0, 3), dtype=np.float32),
            "event": np.zeros((1, 1, 1, 3), dtype=np.uint8),
        }
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "episode_0.hdf5")
            with self.assertRaises(RuntimeError):
                self.mod.write_episode(
                    output, arrays, self.make_episode(), self.make_topics(),
                    30.0, "none", False, raw_events_h5=None,
                )
            self.assertFalse(os.path.exists(output + ".tmp"))

    def test_cache_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = os.path.join(directory, "raw.h5")
            open(raw, "wb").close()
            cache = os.path.join(directory, "cache.json")
            with open(cache, "w", encoding="utf-8") as stream:
                json.dump({
                    "schema_version": self.mod.EVENT_TRACKER_CACHE_SCHEMA,
                    "raw_events_h5": raw, "tracker_config_hash": "wrong",
                    "episodes": {"episode_0": []},
                }, stream)
            with self.assertRaisesRegex(RuntimeError, "configuration hash"):
                self.mod.load_event_tracker_cache(
                    cache, raw, "expected", ["episode_0"], object
                )


if __name__ == "__main__":
    unittest.main()
