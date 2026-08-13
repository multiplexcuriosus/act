#!/usr/bin/env python3

import importlib.util
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

    def test_policy_grids_are_exact_for_30_and_60_hz(self):
        for rate, expected_period in ((30, 33_333_333), (60, 16_666_667)):
            grid = self.mod.policy_grid_ns(1.0, 11.0, rate)
            self.assertEqual(grid.dtype, np.int64)
            self.assertEqual(int(round(np.mean(np.diff(grid)))), expected_period)
            indices = np.arange(len(grid), dtype=np.int64)
            expected = 1_000_000_000 + np.rint(
                indices.astype(np.float64) * 1e9 / rate
            ).astype(np.int64)
            np.testing.assert_array_equal(grid, expected)

    def test_rgb_point_uses_header_timestamp_and_aligns_causally(self):
        topics = self.make_topics()
        topics.rgb_2d = "/ball_tracker2/ball_2d_px"
        data = self.mod.create_episode_buffer()
        msg = types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(sec=1, nanosec=20_000_000)
            ),
            point=types.SimpleNamespace(x=100.0, y=200.0, z=0.0),
        )
        self.mod.ingest_episode_message(data, topics.rgb_2d, msg, 9.0, topics)
        aligned = self.mod.align_point_detections(
            data["rgb_2d"], np.asarray([1_010_000_000, 1_020_000_000], "i8")
        )
        self.assertEqual(aligned["rgb_valid"].tolist(), [0, 1])
        self.assertEqual(aligned["rgb_source_timestamps_ns"].tolist(),
                         [-1, 1_020_000_000])
        self.assertTrue(np.isnan(aligned["rgb_source_timestamps"][0]))

    def test_event_tracker_alignment_holds_valid_across_invalid_update(self):
        openmv_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "openmv_cam"
        ))
        _, TrackerUpdate, _, align = self.mod.import_openmv_tracker_api(
            openmv_root
        )

        def update(t_ns, valid, x, reason=""):
            return TrackerUpdate(
                "episode_0", 0, 0, t_ns, t_ns // 1_000_000,
                10, 20, x, 5.0, 0.0, 0.0, 0.0, .8, valid, False,
                10, int(valid), 2, 3, 2, 2, .7, reason,
            )

        result = align(
            [update(1_000_000_000, True, 10.0),
             update(1_030_000_000, False, 0.0, "no_blob"),
             update(1_060_000_000, True, 20.0)],
            np.asarray([1_030_000_000, 1_055_000_000, 1_060_000_000], "i8"),
            .1,
        )
        self.assertEqual(result["event_2d_px"][:, 0].tolist(), [10, 10, 20])
        self.assertEqual(result["event_valid"].tolist(), [1, 1, 1])
        self.assertEqual(result["event_latest_update_valid"].tolist(), [0, 0, 1])
        self.assertEqual(result["event_latest_rejection_reason"].tolist(),
                         ["no_blob", "no_blob", ""])
        self.assertTrue(np.all(
            result["event_source_timestamps_ns"] <=
            np.asarray([1_030_000_000, 1_055_000_000, 1_060_000_000], "i8")
        ))

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

    def test_resolve_input_paths_defaults_to_recording_sidecar_for_event_mode(self):
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

            expected_sidecar = os.path.join(
                rec_dir,
                "recording_20260728_225008_raw_events.h5",
            )
            self.assertEqual(resolved_bag_path, bag_path)
            self.assertEqual(resolved_sidecar_path, expected_sidecar)

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


if __name__ == "__main__":
    unittest.main()
