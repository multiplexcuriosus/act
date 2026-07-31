#!/usr/bin/env python3

import importlib.util
import os
import sys
import tempfile
import types
import unittest

import h5py
import numpy as np


HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import raw_event_hdf5 as reh5  # noqa: E402


def load_bag_to_hdf5_module():
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
        sys.modules["rclpy"] = types.ModuleType("rclpy")

    if "rclpy.serialization" not in sys.modules:
        serialization = types.ModuleType("rclpy.serialization")
        serialization.deserialize_message = lambda raw, cls: raw
        sys.modules["rclpy.serialization"] = serialization

    if "rosidl_runtime_py" not in sys.modules:
        sys.modules["rosidl_runtime_py"] = types.ModuleType("rosidl_runtime_py")

    if "rosidl_runtime_py.utilities" not in sys.modules:
        utilities = types.ModuleType("rosidl_runtime_py.utilities")
        utilities.get_message = lambda type_name: object
        sys.modules["rosidl_runtime_py.utilities"] = utilities

    module_path = os.path.join(HERE, "bag_to_hdf5.py")
    spec = importlib.util.spec_from_file_location("bag_to_hdf5_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RawEventHdf5Tests(unittest.TestCase):
    @staticmethod
    def write_sidecar(
        path: str,
        event_type,
        event_x,
        event_y,
        event_t_us,
        packet_ros_t_ns,
        packet_start_event_idx,
        packet_end_event_idx,
    ):
        with h5py.File(path, "w") as h5:
            h5.attrs["width"] = 2
            h5.attrs["height"] = 1
            events = h5.create_group("events")
            packets = h5.create_group("packets")
            events.create_dataset("type", data=np.asarray(event_type, dtype=np.uint8))
            events.create_dataset("x", data=np.asarray(event_x, dtype=np.int16))
            events.create_dataset("y", data=np.asarray(event_y, dtype=np.int16))
            events.create_dataset("t_us", data=np.asarray(event_t_us, dtype=np.int64))
            packets.create_dataset(
                "ros_t_ns", data=np.asarray(packet_ros_t_ns, dtype=np.int64)
            )
            packets.create_dataset(
                "start_event_idx",
                data=np.asarray(packet_start_event_idx, dtype=np.int64),
            )
            packets.create_dataset(
                "end_event_idx",
                data=np.asarray(packet_end_event_idx, dtype=np.int64),
            )

    def test_schema_validation_missing_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = os.path.join(tmpdir, "broken.h5")
            with h5py.File(sidecar, "w") as h5:
                h5.create_group("events")
                h5.create_group("packets")
            with self.assertRaises(RuntimeError):
                reh5.RawEventStore(sidecar, logger=lambda *_: None)

    def test_future_packet_is_not_used(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = os.path.join(tmpdir, "raw.h5")
            self.write_sidecar(
                sidecar,
                event_type=[1, 1],
                event_x=[0, 1],
                event_y=[0, 0],
                event_t_us=[100_000, 200_000],
                packet_ros_t_ns=[1_000_000_000, 2_000_000_000],
                packet_start_event_idx=[0, 1],
                packet_end_event_idx=[1, 2],
            )
            store = reh5.RawEventStore(sidecar, logger=lambda *_: None)
            try:
                _, source_ns, counts = store.frame_3chef_with_metadata_at_bag_time(
                    bag_t_sec=1.5,
                    windows_ms=(50.0, 100.0, 200.0),
                    mode="shifted",
                    scaling_mode="signed_log1p_fixed_clip",
                    event_clip_count=8.0,
                    packet_margin_ms=1000.0,
                )
            finally:
                store.close()

        self.assertEqual(source_ns, 1_000_000_000)
        self.assertGreater(int(np.sum(counts)), 0)

    def test_shifted_bin_boundaries_are_exclusive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = os.path.join(tmpdir, "raw.h5")
            self.write_sidecar(
                sidecar,
                event_type=[1, 1, 1],
                event_x=[0, 0, 0],
                event_y=[0, 0, 0],
                event_t_us=[100_000, 150_000, 200_000],
                packet_ros_t_ns=[2_000_000_000],
                packet_start_event_idx=[0],
                packet_end_event_idx=[3],
            )
            store = reh5.RawEventStore(sidecar, logger=lambda *_: None)
            try:
                _, _, counts = store.frame_3chef_with_metadata_at_bag_time(
                    bag_t_sec=2.0,
                    windows_ms=(50.0, 100.0, 200.0),
                    mode="shifted",
                    scaling_mode="signed_log1p_fixed_clip",
                    event_clip_count=8.0,
                )
            finally:
                store.close()

        np.testing.assert_array_equal(counts, np.asarray([2, 1, 0], dtype=np.int32))

    def test_fixed_log_scaling_is_invariant_to_other_pixel_activity(self):
        frame_a = reh5.render_event_frame_from_raw_arrays(
            event_type=np.asarray([1], dtype=np.uint8),
            event_x=np.asarray([0], dtype=np.int16),
            event_y=np.asarray([0], dtype=np.int16),
            width=2,
            height=1,
            scaling_mode="signed_log1p_fixed_clip",
            event_clip_count=32.0,
        )
        frame_b = reh5.render_event_frame_from_raw_arrays(
            event_type=np.asarray([1] + [1] * 20, dtype=np.uint8),
            event_x=np.asarray([0] + [1] * 20, dtype=np.int16),
            event_y=np.asarray([0] + [0] * 20, dtype=np.int16),
            width=2,
            height=1,
            scaling_mode="signed_log1p_fixed_clip",
            event_clip_count=32.0,
        )

        self.assertEqual(int(frame_a[0, 0]), int(frame_b[0, 0]))

    def test_empty_slice_produces_neutral(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = os.path.join(tmpdir, "raw.h5")
            self.write_sidecar(
                sidecar,
                event_type=[1],
                event_x=[0],
                event_y=[0],
                event_t_us=[100_000],
                packet_ros_t_ns=[1_000_000_000],
                packet_start_event_idx=[0],
                packet_end_event_idx=[1],
            )
            store = reh5.RawEventStore(sidecar, logger=lambda *_: None)
            try:
                frame, _, _ = store.frame_3chef_with_metadata_at_bag_time(
                    bag_t_sec=0.1,
                    windows_ms=(50.0, 100.0, 200.0),
                    mode="shifted",
                    scaling_mode="signed_log1p_fixed_clip",
                    event_clip_count=8.0,
                    packet_margin_ms=0.0,
                )
            finally:
                store.close()

        self.assertTrue(np.all(frame == 128))

    def test_bag_to_hdf5_still_exports_shared_loader(self):
        bag_mod = load_bag_to_hdf5_module()
        self.assertIs(bag_mod.RawEventStore, reh5.RawEventStore)
        self.assertIs(bag_mod.resolve_recording_dir, reh5.resolve_recording_dir)


if __name__ == "__main__":
    unittest.main()
