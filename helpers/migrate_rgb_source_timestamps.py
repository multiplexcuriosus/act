#!/usr/bin/env python3
"""Add only RGB PointStamped header timestamps to existing ACT episodes."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from bag_to_il_intercept import (
    DEFAULT_RGB_2D_TOPIC, add_rgb_source_timestamps_dataset,
    apply_storage_filter, header_stamp_to_sec, open_reader, topic_type_map,
)
from raw_event_hdf5 import resolve_recording_dir


def collect_rgb_stamps(bag_path, topic, storage_id):
    """Collect bag association times and PointStamped header times."""
    reader = open_reader(bag_path, storage_id)
    types = topic_type_map(reader)
    if topic not in types:
        raise RuntimeError(f"RGB sparse topic is absent from bag: {topic}")
    message_class = get_message(types[topic])
    apply_storage_filter(reader, [topic])
    receipt, source = [], []
    while reader.has_next():
        _, raw, timestamp_ns = reader.read_next()
        message = deserialize_message(raw, message_class)
        receipt.append(float(timestamp_ns) * 1e-9)
        source.append(header_stamp_to_sec(message))
    return np.asarray(receipt), np.asarray(source)


def migrate_episode(path, receipt_times, source_times, overwrite=False):
    """Create only ``rgb_source_timestamps`` in one episode."""
    with h5py.File(path, "r") as target:
        for required in (
            "/observations/timestamps",
            "/observations/sparse_tracking/rgb_2d_px",
            "/observations/sparse_tracking/rgb_valid",
        ):
            if required not in target:
                raise ValueError(f"{path}: missing {required}")
        grid = np.asarray(target["/observations/timestamps"][:], dtype=np.float64)
    indices = np.searchsorted(receipt_times, grid, side="right") - 1
    values = np.full(grid.shape, np.nan, dtype=np.float64)
    present = indices >= 0
    values[present] = source_times[indices[present]]
    add_rgb_source_timestamps_dataset(path, values, overwrite=overwrite)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bag")
    inputs.add_argument("--rec-dir")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--topic", default=DEFAULT_RGB_2D_TOPIC)
    parser.add_argument("--storage-id", default="mcap")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    bag_path = args.bag
    if args.rec_dir:
        bag_path, _, _ = resolve_recording_dir(
            os.path.abspath(os.path.expanduser(args.rec_dir)),
            allow_missing_raw_events=True,
        )
    receipt, source = collect_rgb_stamps(bag_path, args.topic, args.storage_id)
    paths = sorted(Path(args.episodes).glob("episode_*.hdf5"))
    if not paths:
        raise ValueError(f"No episode_*.hdf5 files in {args.episodes}")
    for path in paths:
        migrate_episode(path, receipt, source, overwrite=args.overwrite)
        print(f"migrated {path}", flush=True)


if __name__ == "__main__":
    main()
