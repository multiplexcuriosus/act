#!/usr/bin/env python3

import importlib.util
import os
import sys
import tempfile
import types
import unittest

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

        rosbag2_py.SequentialReader = _DummySequentialReader
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

    def make_episode(self):
        return self.mod.EpisodeWindow(source_idx=0, output_idx=0, start=0.0, end=0.4)

    def make_sampling_data(self, tcp_times, tcp_values, goto_values=None):
        rgb0 = np.array([[[0, 0, 255]]], dtype=np.uint8)
        rgb1 = np.array([[[0, 255, 0]]], dtype=np.uint8)
        rgb2 = np.array([[[255, 0, 0]]], dtype=np.uint8)
        rgb_msgs = [
            self.make_raw_msg_bgr(rgb0),
            self.make_raw_msg_bgr(rgb1),
            self.make_raw_msg_bgr(rgb2),
        ]
        data = {
            "rgb_t": [0.0, 0.1, 0.2],
            "rgb_msg": rgb_msgs,
            "joint_t": [0.0, 0.1, 0.2],
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
            data["goto_s_t"] = [0.05 + 0.1 * i for i in range(len(goto_values))]
            data["goto_s"] = list(goto_values)
        return data

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


if __name__ == "__main__":
    unittest.main()
