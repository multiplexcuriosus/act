#!/usr/bin/env python3
"""Add only RGB PointStamped header timestamps to existing ACT episodes."""

import argparse
import os
import time
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


def log(message):
    """Print one immediately visible progress line."""
    print(message, flush=True)


def format_duration(seconds):
    """Format a short wall-clock duration for progress messages."""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {seconds:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes):02d}m {seconds:04.1f}s"


def collect_rgb_stamps(bag_path, topic, storage_id, progress_every=10000):
    """Collect bag association times and PointStamped header times."""
    log(f"[bag] Opening {bag_path}")
    log(f"[bag] Storage: {storage_id}; topic: {topic}")
    started = time.monotonic()
    reader = open_reader(bag_path, storage_id)
    types = topic_type_map(reader)
    if topic not in types:
        raise RuntimeError(f"RGB sparse topic is absent from bag: {topic}")
    log(f"[bag] Message type: {types[topic]}")
    message_class = get_message(types[topic])
    apply_storage_filter(reader, [topic])
    log("[bag] Storage filter applied; reading RGB timestamp messages...")
    receipt, source = [], []
    while reader.has_next():
        _, raw, timestamp_ns = reader.read_next()
        message = deserialize_message(raw, message_class)
        receipt.append(float(timestamp_ns) * 1e-9)
        source.append(header_stamp_to_sec(message))
        count = len(receipt)
        if progress_every and count % progress_every == 0:
            elapsed = time.monotonic() - started
            rate = count / elapsed if elapsed else 0.0
            bag_span = receipt[-1] - receipt[0]
            log(
                f"[bag] Read {count:,} messages | {rate:,.0f} msg/s | "
                f"bag span {format_duration(bag_span)} | "
                f"wall time {format_duration(elapsed)}"
            )
    receipt_array = np.asarray(receipt)
    source_array = np.asarray(source)
    elapsed = time.monotonic() - started
    rate = len(receipt_array) / elapsed if elapsed else 0.0
    log(
        f"[bag] Finished: {len(receipt_array):,} messages in "
        f"{format_duration(elapsed)} ({rate:,.0f} msg/s)"
    )
    if receipt_array.size:
        bag_span = receipt_array[-1] - receipt_array[0]
        lag_ms = (receipt_array - source_array) * 1e3
        log(
            f"[bag] Timestamp span: {format_duration(bag_span)}; "
            f"receipt-source lag: median {np.median(lag_ms):.2f} ms, "
            f"min {np.min(lag_ms):.2f} ms, max {np.max(lag_ms):.2f} ms"
        )
    return receipt_array, source_array


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
    return grid.size, int(np.count_nonzero(present))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bag")
    inputs.add_argument("--rec-dir")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--topic", default=DEFAULT_RGB_2D_TOPIC)
    parser.add_argument("--storage-id", default="mcap")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        metavar="MESSAGES",
        help="print bag-read progress every N messages (0 disables; default: 10000)",
    )
    args = parser.parse_args(argv)
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    log("[setup] RGB source-timestamp migration starting")
    log(f"[setup] Episodes directory: {Path(args.episodes).expanduser().resolve()}")
    log(f"[setup] Existing datasets will be {'replaced' if args.overwrite else 'preserved'}")
    bag_path = args.bag
    if args.rec_dir:
        log(f"[setup] Resolving recording directory: {args.rec_dir}")
        bag_path, _, _ = resolve_recording_dir(
            os.path.abspath(os.path.expanduser(args.rec_dir)),
            allow_missing_raw_events=True,
        )
    log(f"[setup] Resolved bag: {bag_path}")
    receipt, source = collect_rgb_stamps(
        bag_path, args.topic, args.storage_id, args.progress_every
    )
    if not receipt.size:
        raise RuntimeError(f"No messages found on RGB sparse topic: {args.topic}")
    paths = sorted(Path(args.episodes).glob("episode_*.hdf5"))
    if not paths:
        raise ValueError(f"No episode_*.hdf5 files in {args.episodes}")
    log(f"[episodes] Found {len(paths):,} episode files")
    started = time.monotonic()
    total_samples = 0
    total_present = 0
    for number, path in enumerate(paths, start=1):
        episode_started = time.monotonic()
        samples, present = migrate_episode(
            path, receipt, source, overwrite=args.overwrite
        )
        total_samples += samples
        total_present += present
        elapsed = time.monotonic() - started
        rate = number / elapsed if elapsed else 0.0
        remaining = (len(paths) - number) / rate if rate else 0.0
        coverage = 100.0 * present / samples if samples else 0.0
        log(
            f"[episodes] [{number}/{len(paths)}] migrated {path.name} | "
            f"{present:,}/{samples:,} timestamps ({coverage:.1f}%) | "
            f"file {format_duration(time.monotonic() - episode_started)} | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(remaining)}"
        )
    elapsed = time.monotonic() - started
    coverage = 100.0 * total_present / total_samples if total_samples else 0.0
    log(
        f"[done] Migrated {len(paths):,} episodes and {total_samples:,} samples "
        f"in {format_duration(elapsed)}; timestamp coverage "
        f"{total_present:,}/{total_samples:,} ({coverage:.1f}%)"
    )


if __name__ == "__main__":
    main()
