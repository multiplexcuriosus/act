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
        event_type=None,
        event_x=None,
        event_y=None,
        event_t_us=None,
        packet_ros_t_ns=None,
        packet_start_event_idx=None,
        packet_end_event_idx=None,
        width=2,
        height=1,
    ):
        if event_type is None:
            event_type = [1, 1, 1]
        if event_x is None:
            event_x = [0, 1, 0]
        if event_y is None:
            event_y = [0, 0, 0]
        if event_t_us is None:
            event_t_us = [100_000, 150_000, 200_000]
        if packet_ros_t_ns is None:
            packet_ros_t_ns = [900_000_000, 1_100_000_000, 1_300_000_000]
        if packet_start_event_idx is None:
            packet_start_event_idx = [0, 1, 2]
        if packet_end_event_idx is None:
            packet_end_event_idx = [1, 2, 3]

        with h5py.File(path, "w") as h5:
            h5.attrs["width"] = width
            h5.attrs["height"] = height
            events = h5.create_group("events")
            packets = h5.create_group("packets")
            events.create_dataset("type", data=np.asarray(event_type))
            events.create_dataset("x", data=np.asarray(event_x))
            events.create_dataset("y", data=np.asarray(event_y))
            events.create_dataset("t_us", data=np.asarray(event_t_us))
            packets.create_dataset(
                "ros_t_ns", data=np.asarray(packet_ros_t_ns)
            )
            packets.create_dataset(
                "start_event_idx",
                data=np.asarray(packet_start_event_idx),
            )
            packets.create_dataset(
                "end_event_idx",
                data=np.asarray(packet_end_event_idx),
            )

    def test_schema_validation_missing_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = os.path.join(tmpdir, "broken.h5")
            with h5py.File(sidecar, "w") as h5:
                h5.create_group("events")
                h5.create_group("packets")
            with self.assertRaises(ValueError):
                reh5.RawEventStore(sidecar, logger=lambda *_: None)

    def test_event_arrays_with_unequal_lengths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = os.path.join(tmpdir, "bad_events_len.h5")
            self.write_sidecar(
                sidecar,
                event_type=[1, 1],
                event_x=[0, 1, 0],
                event_y=[0, 0, 0],
                event_t_us=[1, 2, 3],
            )
            with self.assertRaises(ValueError):
                reh5.RawEventStore(sidecar, logger=lambda *_: None)

    def test_packet_arrays_with_unequal_lengths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = os.path.join(tmpdir, "bad_packet_len.h5")
            self.write_sidecar(
                sidecar,
                packet_ros_t_ns=[1, 2],
                packet_start_event_idx=[0, 1, 2],
                packet_end_event_idx=[1, 2, 3],
            )
            with self.assertRaises(ValueError):
                reh5.RawEventStore(sidecar, logger=lambda *_: None)

    def test_non_1d_required_arrays_are_rejected(self):
        cases = [
            {"event_type": [[1, 1], [1, 1]]},
            {"event_x": [[0, 1], [0, 1]]},
            {"event_y": [[0, 0], [0, 0]]},
            {"event_t_us": [[1, 2], [3, 4]]},
            {"packet_ros_t_ns": [[1, 2], [3, 4]]},
            {"packet_start_event_idx": [[0, 1], [1, 2]]},
            {"packet_end_event_idx": [[1, 2], [2, 3]]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, kwargs in enumerate(cases):
                sidecar = os.path.join(tmpdir, f"bad_ndim_{idx}.h5")
                self.write_sidecar(sidecar, **kwargs)
                with self.subTest(case=idx):
                    with self.assertRaises(ValueError):
                        reh5.RawEventStore(sidecar, logger=lambda *_: None)

    def test_reversed_packet_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = os.path.join(tmpdir, "bad_packet_time_order.h5")
            self.write_sidecar(sidecar, packet_ros_t_ns=[10, 5, 20])
            with self.assertRaises(ValueError):
                reh5.RawEventStore(sidecar, logger=lambda *_: None)

    def test_negative_packet_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = os.path.join(tmpdir, "bad_packet_time_negative.h5")
            self.write_sidecar(sidecar, packet_ros_t_ns=[-1, 5, 20])
            with self.assertRaises(ValueError):
                reh5.RawEventStore(sidecar, logger=lambda *_: None)

    def test_invalid_packet_indices_are_rejected(self):
        cases = [
            {"packet_start_event_idx": [-1, 1, 2]},
            {"packet_end_event_idx": [-1, 2, 3]},
            {"packet_start_event_idx": [0, 3, 2], "packet_end_event_idx": [1, 2, 3]},
            {"packet_start_event_idx": [0, 1, 2], "packet_end_event_idx": [1, 2, 5]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, kwargs in enumerate(cases):
                sidecar = os.path.join(tmpdir, f"bad_idx_{idx}.h5")
                self.write_sidecar(sidecar, **kwargs)
                with self.subTest(case=idx):
                    with self.assertRaises(ValueError):
                        reh5.RawEventStore(sidecar, logger=lambda *_: None)

    def test_non_monotonic_packet_index_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_start = os.path.join(tmpdir, "bad_start_monotonic.h5")
            self.write_sidecar(sidecar_start, packet_start_event_idx=[0, 2, 1])
            with self.assertRaises(ValueError):
                reh5.RawEventStore(sidecar_start, logger=lambda *_: None)

            sidecar_end = os.path.join(tmpdir, "bad_end_monotonic.h5")
            self.write_sidecar(sidecar_end, packet_end_event_idx=[1, 3, 2])
            with self.assertRaises(ValueError):
                reh5.RawEventStore(sidecar_end, logger=lambda *_: None)

    def test_zero_or_negative_image_dimensions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_w = os.path.join(tmpdir, "bad_width.h5")
            self.write_sidecar(sidecar_w, width=0, height=1)
            with self.assertRaises(ValueError):
                reh5.RawEventStore(sidecar_w, logger=lambda *_: None)

            sidecar_h = os.path.join(tmpdir, "bad_height.h5")
            self.write_sidecar(sidecar_h, width=2, height=-1)
            with self.assertRaises(ValueError):
                reh5.RawEventStore(sidecar_h, logger=lambda *_: None)

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

    def test_bag_to_hdf5_shifted_and_cumulative_modes_remain_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = os.path.join(tmpdir, "raw_modes.h5")
            self.write_sidecar(sidecar)
            store = reh5.RawEventStore(sidecar, logger=lambda *_: None)
            try:
                shifted = store.frame_3chef_at_bag_time(
                    bag_t_sec=1.3,
                    windows_ms=(50.0, 100.0, 200.0),
                    mode="shifted",
                )
                cumulative = store.frame_3chef_at_bag_time(
                    bag_t_sec=1.3,
                    windows_ms=(50.0, 100.0, 200.0),
                    mode="cumulative",
                )
            finally:
                store.close()

        self.assertEqual(shifted.shape, (1, 2, 3))
        self.assertEqual(cumulative.shape, (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
