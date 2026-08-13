#!/usr/bin/env python3
"""Convert ROS 2 ball-interception bags into per-episode IL HDF5 files.

The learned action is the continuously measured middle-line TCP coordinate
published on /middle_line/current_tcp_s (std_msgs/msg/Float64):

    action[t, 0] = measured current_tcp_s at sample t

Images, joint states, and current_tcp_s are sampled causally: every sample on
an evenly spaced grid uses the most recent source message whose bag timestamp
is less than or equal to the sample time. Temporal RGB stacking and delta
construction are intentionally deferred to the training loader.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import h5py
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from raw_event_hdf5 import RawEventStore, resolve_recording_dir

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sparse_ball import policy_period_ns, validate_policy_rate


DEFAULT_RGB_TOPIC = "auto"
DEFAULT_RGB_TOPIC_RAW = "/top_cam/camera/color/image_raw"
DEFAULT_RGB_TOPIC_COMPRESSED = "/top_cam/camera/color/image_raw/compressed"
DEFAULT_RGB_2D_TOPIC = "/ball_tracker2/ball_2d_px"
DEFAULT_JOINT_TOPIC = "/joint_states"
DEFAULT_EPISODE_TOPIC = "/episode/control"
DEFAULT_CURRENT_TCP_S_TOPIC = "/middle_line/current_tcp_s"
DEFAULT_GOTO_S_TOPIC = "/trajectory_executor/executed_goto_s"
DEFAULT_GOTO_S_TARGET_BASE_TOPIC = (
    "/trajectory_executor/executed_goto_s_target_base"
)

RAW_IMAGE_TYPE = "sensor_msgs/msg/Image"
COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"
FLOAT64_TYPE = "std_msgs/msg/Float64"
POINT_STAMPED_TYPE = "geometry_msgs/msg/PointStamped"

ARM_JOINT_NAMES = (
    "right_fr3_joint1",
    "right_fr3_joint2",
    "right_fr3_joint3",
    "right_fr3_joint4",
    "right_fr3_joint5",
    "right_fr3_joint6",
    "right_fr3_joint7",
)

SHIFTED_3CHEF_REPRESENTATION = "shifted_3chef_signed"
XYT_REPRESENTATION = "xyt_signed_voxel_v1"


@dataclass
class Topics:
    rgb: str
    joint: str
    episode: str
    current_tcp_s: str
    goto_s: str
    goto_s_target_base: str
    rgb_2d: str = DEFAULT_RGB_2D_TOPIC


@dataclass
class EpisodeWindow:
    source_idx: int
    output_idx: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def log(message: str) -> None:
    print(message, flush=True)


def ns_to_sec(timestamp_ns: int) -> float:
    return float(timestamp_ns) * 1e-9


def policy_grid_ns(start_sec: float, end_sec: float, rate_hz: int) -> np.ndarray:
    """Build a drift-free 30/60 Hz HDF5 grid with rational integer offsets."""
    rate = int(rate_hz)
    if float(rate_hz) != rate or rate <= 0:
        raise ValueError("policy grid rate must be a positive integer")
    origin_ns = int(round(float(start_sec) * 1e9))
    end_ns = int(round(float(end_sec) * 1e9))
    count = max(0, ((end_ns - origin_ns) * rate) // 1_000_000_000 + 1)
    indices = np.arange(count, dtype=np.int64)
    offsets = (indices * 1_000_000_000 + rate // 2) // rate
    grid = origin_ns + offsets
    return grid[grid <= end_ns]


def header_stamp_to_sec(msg: Any) -> float:
    """Return a PointStamped source timestamp, never its bag receipt time."""
    try:
        stamp = msg.header.stamp
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("RGB PointStamped message has no valid header timestamp") from error
    if not np.isfinite(value):
        raise RuntimeError("RGB PointStamped header timestamp is non-finite")
    return value


def add_rgb_source_timestamps_dataset(path, values, overwrite=False):
    """Add or replace only the RGB source-timestamp dataset in an episode."""
    values = np.asarray(values, dtype=np.float64)
    with h5py.File(path, "r+") as target:
        sparse_key = "/observations/sparse_tracking"
        key = f"{sparse_key}/rgb_source_timestamps"
        if sparse_key not in target:
            raise ValueError(f"{path}: missing {sparse_key}")
        if key in target and not overwrite:
            raise FileExistsError(f"{path}: {key} already exists; use --overwrite")
        timestamps = target["/observations/timestamps"]
        if values.shape != timestamps.shape:
            raise ValueError(
                f"{path}: RGB source timestamps must have shape {timestamps.shape}, "
                f"got {values.shape}"
            )
        sparse = target[sparse_key]
        temporary_name = "rgb_source_timestamps.__tmp__"
        if temporary_name in sparse:
            del sparse[temporary_name]
        sparse.create_dataset(temporary_name, data=values, dtype=np.float64)
        if "rgb_source_timestamps" in sparse:
            del sparse["rgb_source_timestamps"]
        sparse.move(temporary_name, "rgb_source_timestamps")


def open_reader(bag_path: str, storage_id: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    return reader


def apply_storage_filter(
    reader: rosbag2_py.SequentialReader, topics: Sequence[str]
) -> List[str]:
    filtered = list(dict.fromkeys(topics))
    if not filtered:
        raise RuntimeError("Storage filter topic list must be non-empty")
    reader.set_filter(rosbag2_py.StorageFilter(topics=filtered))
    return filtered


def open_filtered_reader(
    bag_path: str, storage_id: str, topics: Sequence[str]
) -> rosbag2_py.SequentialReader:
    reader = open_reader(bag_path, storage_id)
    apply_storage_filter(reader, topics)
    return reader


def topic_type_map(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
    return {item.name: item.type for item in reader.get_all_topics_and_types()}


def format_hms(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0.0:
        return "n/a"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def message_classes(
    types: Dict[str, str], selected_topics: Set[str]
) -> Dict[str, Any]:
    return {
        topic: get_message(types[topic])
        for topic in selected_topics
        if topic in types
    }


def resolve_rgb_topic(types: Dict[str, str], requested_rgb_topic: str) -> str:
    if requested_rgb_topic != "auto":
        if requested_rgb_topic not in types:
            raise RuntimeError(
                f"Requested RGB topic is missing: {requested_rgb_topic}"
            )
        return requested_rgb_topic

    if DEFAULT_RGB_TOPIC_COMPRESSED in types:
        return DEFAULT_RGB_TOPIC_COMPRESSED
    if DEFAULT_RGB_TOPIC_RAW in types:
        return DEFAULT_RGB_TOPIC_RAW

    raise RuntimeError(
        "RGB topic auto-selection failed: neither "
        f"{DEFAULT_RGB_TOPIC_COMPRESSED} nor {DEFAULT_RGB_TOPIC_RAW} is present"
    )


def validate_topics(
    types: Dict[str, str],
    topics: Topics,
    collect_goto_s_debug: bool,
    collect_target_base_debug: bool,
    collect_rgb_2d: bool = True,
) -> Tuple[bool, bool]:
    required = {topics.rgb, topics.joint, topics.episode, topics.current_tcp_s}
    missing = sorted(required - set(types))
    if missing:
        raise RuntimeError(f"Missing required topics: {missing}")

    rgb_type = types[topics.rgb]
    if rgb_type not in (RAW_IMAGE_TYPE, COMPRESSED_IMAGE_TYPE):
        raise RuntimeError(
            f"RGB topic {topics.rgb} has unsupported type {rgb_type}. "
            f"Expected {RAW_IMAGE_TYPE} or {COMPRESSED_IMAGE_TYPE}."
        )

    tcp_s_type = types[topics.current_tcp_s]
    if tcp_s_type != FLOAT64_TYPE:
        raise RuntimeError(
            f"Topic {topics.current_tcp_s} must be {FLOAT64_TYPE}, got {tcp_s_type}"
        )

    for topic in sorted(required):
        log(f"[INFO] {topic} :: {types[topic]}")

    log(f"[INFO] selected RGB topic: {topics.rgb} ({rgb_type})")

    goto_s_available = topics.goto_s in types
    target_base_available = topics.goto_s_target_base in types
    rgb_2d_available = topics.rgb_2d in types

    if collect_rgb_2d and rgb_2d_available:
        rgb_2d_type = types[topics.rgb_2d]
        if rgb_2d_type != POINT_STAMPED_TYPE:
            log(
                f"[WARNING] optional RGB 2D topic {topics.rgb_2d} has type "
                f"{rgb_2d_type}, expected {POINT_STAMPED_TYPE}; writing invalid RGB 2D samples"
            )
            rgb_2d_available = False
        else:
            log(f"[INFO] {topics.rgb_2d} :: {rgb_2d_type}")
    elif collect_rgb_2d:
        log(
            f"[INFO] optional RGB 2D topic is absent: {topics.rgb_2d}; "
            "writing invalid RGB 2D samples"
        )

    if collect_goto_s_debug and not goto_s_available:
        log(
            "[INFO] optional GOTO_S debug topic is absent; "
            "commands/goto_s datasets will be empty"
        )
    if collect_target_base_debug and not target_base_available:
        log(
            "[INFO] optional GOTO_S target-base debug topic is absent; "
            "commands/goto_s_target_base datasets will be empty"
        )

    return (
        collect_goto_s_debug and goto_s_available,
        collect_target_base_debug and target_base_available,
    )


def extract_episode_windows(
    reader: rosbag2_py.SequentialReader,
    types: Dict[str, str],
    episode_topic: str,
    min_duration: float,
) -> List[EpisodeWindow]:
    """Apply the recording manager's start/stop/cancel protocol."""
    log("[INFO] Pass 1: scanning episode-control markers")
    if episode_topic not in types:
        raise RuntimeError(f"Episode topic not present: {episode_topic}")
    msg_cls = get_message(types[episode_topic])

    current_start: Optional[float] = None
    committed: List[Tuple[float, float]] = []

    while reader.has_next():
        topic, raw, timestamp_ns = reader.read_next()
        if topic != episode_topic:
            continue

        msg = deserialize_message(raw, msg_cls)
        timestamp = ns_to_sec(timestamp_ns)
        value = int(msg.data)
        log(f"[DEBUG] {episode_topic} at {timestamp:.6f} -> {value}")

        if value == 1:  # start
            if current_start is None:
                current_start = timestamp
            else:
                log("[WARNING] duplicate start while an episode is active; ignored")

        elif value == 2:  # stop and commit
            if current_start is None:
                log("[WARNING] stop while no episode is active; ignored")
            else:
                committed.append((current_start, timestamp))
                current_start = None

        elif value == 3:  # cancel current
            if current_start is None:
                log("[WARNING] cancel-current while no episode is active; ignored")
            else:
                log(f"[INFO] cancelled active episode starting at {current_start:.6f}")
                current_start = None

        elif value == 4:  # cancel last committed
            if current_start is not None:
                log("[WARNING] cancel-last while an episode is active; ignored")
            elif committed:
                removed = committed.pop()
                log(
                    "[INFO] removed last committed episode "
                    f"[{removed[0]:.6f}, {removed[1]:.6f}]"
                )
            else:
                log("[WARNING] cancel-last with no committed episode; ignored")
        else:
            log(f"[WARNING] unknown episode-control value {value}; ignored")

    if current_start is not None:
        log("[WARNING] bag ended during an episode; unfinished episode discarded")

    windows: List[EpisodeWindow] = []
    for source_idx, (start, end) in enumerate(committed):
        duration = end - start
        if duration < min_duration:
            log(
                f"[INFO] dropping source episode {source_idx}: "
                f"{duration:.3f}s < min_duration {min_duration:.3f}s"
            )
            continue
        windows.append(
            EpisodeWindow(
                source_idx=source_idx,
                output_idx=len(windows),
                start=start,
                end=end,
            )
        )

    if not windows:
        raise RuntimeError("No committed episode windows survived filtering")

    log(f"[INFO] retained {len(windows)} of {len(committed)} committed episodes")
    for episode in windows:
        log(
            f"       output {episode.output_idx} (source {episode.source_idx}): "
            f"{episode.start:.6f} .. {episode.end:.6f} "
            f"({episode.duration:.3f}s)"
        )
    return windows


def validate_non_overlapping_windows(windows: Sequence[EpisodeWindow]) -> None:
    if not windows:
        return
    for previous, current in zip(windows, windows[1:]):
        # Inclusive [start, end] windows overlap unless next.start is strictly greater.
        if current.start <= previous.end:
            raise RuntimeError(
                "Selected episode windows overlap under inclusive boundaries: "
                f"output {previous.output_idx} [{previous.start:.6f}, {previous.end:.6f}] "
                f"and output {current.output_idx} [{current.start:.6f}, {current.end:.6f}]. "
                "Refusing to continue because a single message timestamp could belong "
                "to multiple episodes."
            )


def decode_raw_image_to_rgb(msg: Any) -> np.ndarray:
    """Decode sensor_msgs/msg/Image into contiguous RGB uint8 (H, W, 3)."""
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()

    channel_counts = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
    }
    if encoding not in channel_counts:
        raise ValueError(f"Unsupported RGB image encoding: {msg.encoding}")

    channels = channel_counts[encoding]
    row_bytes = width * channels
    step = int(msg.step) if int(msg.step) > 0 else row_bytes
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise RuntimeError(
            f"Image data too short: got {raw.size} bytes, expected at least {expected}"
        )

    packed = raw[:expected].reshape(height, step)[:, :row_bytes]
    if channels == 1:
        mono = packed.reshape(height, width)
        rgb = cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    image = packed.reshape(height, width, channels)
    if encoding == "rgb8":
        return np.ascontiguousarray(image, dtype=np.uint8)
    if encoding == "bgr8":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)
    if encoding == "rgba8":
        rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def decode_compressed_image_to_rgb(msg: Any) -> np.ndarray:
    """Decode sensor_msgs/msg/CompressedImage into contiguous RGB uint8."""
    encoded = np.frombuffer(msg.data, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size == 0:
        raise RuntimeError("CompressedImage decode failed or produced empty output")

    if decoded.ndim == 2:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    elif decoded.ndim == 3 and decoded.shape[2] == 3:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    elif decoded.ndim == 3 and decoded.shape[2] == 4:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
    else:
        raise RuntimeError(
            f"Unsupported decoded CompressedImage shape: {decoded.shape}"
        )

    return np.ascontiguousarray(rgb, dtype=np.uint8)


def image_msg_to_rgb(msg: Any) -> np.ndarray:
    """Decode sensor_msgs/Image or sensor_msgs/CompressedImage into RGB uint8."""
    if hasattr(msg, "encoding") and hasattr(msg, "height") and hasattr(msg, "step"):
        return decode_raw_image_to_rgb(msg)
    if hasattr(msg, "format") and hasattr(msg, "data"):
        return decode_compressed_image_to_rgb(msg)
    raise TypeError(f"Unsupported RGB message object type: {type(msg)}")


def arm_qpos(joint_names: Sequence[str], positions: Sequence[float]) -> np.ndarray:
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    missing = [name for name in ARM_JOINT_NAMES if name not in name_to_index]
    if missing:
        raise RuntimeError(f"Missing required FR3 arm joints: {missing}")
    return np.asarray(
        [positions[name_to_index[name]] for name in ARM_JOINT_NAMES],
        dtype=np.float32,
    )


def required_data_topics(
    topics: Topics,
    collect_goto_s_debug: bool,
    collect_target_base_debug: bool,
    collect_rgb_2d: bool = False,
) -> List[str]:
    tracked = [topics.rgb, topics.joint, topics.current_tcp_s]
    if collect_goto_s_debug:
        tracked.append(topics.goto_s)
    if collect_target_base_debug:
        tracked.append(topics.goto_s_target_base)
    if collect_rgb_2d:
        tracked.append(topics.rgb_2d)
    return tracked

def create_episode_buffer() -> Dict[str, Any]:
    return {
        "rgb_t": [],
        "rgb_msg": [],
        "rgb_2d_t": [],
        "rgb_2d_source_t": [],
        "rgb_2d_px": [],
        "joint_t": [],
        "qpos": [],
        "current_tcp_s_t": [],
        "current_tcp_s": [],
        "goto_s_t": [],
        "goto_s": [],
        "target_base_t": [],
        "target_base": [],
    }


def ingest_episode_message(
    data: Dict[str, Any],
    topic: str,
    msg: Any,
    timestamp: float,
    topics: Topics,
) -> None:
    if topic == topics.rgb:
        data["rgb_t"].append(timestamp)
        data["rgb_msg"].append(msg)
    elif topic == topics.rgb_2d:
        data["rgb_2d_t"].append(timestamp)
        data["rgb_2d_source_t"].append(header_stamp_to_sec(msg))
        data["rgb_2d_px"].append([float(msg.point.x), float(msg.point.y)])
    elif topic == topics.joint:
        data["joint_t"].append(timestamp)
        data["qpos"].append(arm_qpos(msg.name, msg.position))
    elif topic == topics.current_tcp_s:
        data["current_tcp_s_t"].append(timestamp)
        data["current_tcp_s"].append(float(msg.data))
    elif topic == topics.goto_s:
        data["goto_s_t"].append(timestamp)
        data["goto_s"].append(float(msg.data))
    elif topic == topics.goto_s_target_base:
        data["target_base_t"].append(timestamp)
        data["target_base"].append(
            [float(msg.point.x), float(msg.point.y), float(msg.point.z)]
        )


def finalize_episode(
    episode: EpisodeWindow,
    data: Dict[str, Any],
    output_dir: str,
    topics: Topics,
    fps: float,
    max_current_tcp_s_age_sec: float,
    compression: str,
    overwrite: bool,
    collect_started_wall: float,
    rgb_2d_enabled: bool = True,
    max_rgb_2d_age_sec: float = 0.10,
    raw_event_store: Optional[RawEventStore] = None,
    raw_events_h5: Optional[str] = None,
    event_frame_windows_ms: Tuple[float, float, float] = (50.0, 100.0, 200.0),
    event_frame_mode: str = "shifted",
    event_clip_count: Optional[float] = None,
    event_packet_margin_ms: float = 50.0,
    event_representation: str = SHIFTED_3CHEF_REPRESENTATION,
    event_horizon_ms: float = 200.0,
    event_temporal_bins: int = 9,
    event_output_height: int = 320,
    event_output_width: int = 320,
) -> int:
    collect_sec = max(0.0, time.perf_counter() - collect_started_wall)

    log(
        f"[INFO] episode {episode.output_idx}: "
        f"rgb={len(data['rgb_t'])}, joints={len(data['joint_t'])}, "
        f"current_tcp_s={len(data['current_tcp_s_t'])}, "
        f"goto_s_debug={len(data['goto_s_t'])}, "
        f"target_base_debug={len(data['target_base_t'])}"
    )

    if not data["rgb_t"]:
        raise RuntimeError(f"Episode {episode.output_idx}: no RGB frames")
    if not data["joint_t"]:
        raise RuntimeError(f"Episode {episode.output_idx}: no joint states")
    if not data["current_tcp_s_t"]:
        raise RuntimeError(
            f"Episode {episode.output_idx}: no current_tcp_s measurements"
        )

    sample_start = time.perf_counter()
    arrays = sample_episode(
        data=data,
        episode=episode,
        fps=fps,
        max_current_tcp_s_age_sec=max_current_tcp_s_age_sec,
        rgb_2d_enabled=rgb_2d_enabled,
        max_rgb_2d_age_sec=max_rgb_2d_age_sec,
        raw_event_store=raw_event_store,
        event_frame_windows_ms=event_frame_windows_ms,
        event_frame_mode=event_frame_mode,
        event_clip_count=event_clip_count,
        event_packet_margin_ms=event_packet_margin_ms,
        event_representation=event_representation,
        event_horizon_ms=event_horizon_ms,
        event_temporal_bins=event_temporal_bins,
        event_output_height=event_output_height,
        event_output_width=event_output_width,
    )
    sample_sec = max(0.0, time.perf_counter() - sample_start)

    write_start = time.perf_counter()
    output_path = os.path.join(output_dir, f"episode_{episode.output_idx}.hdf5")
    write_episode(
        output_path=output_path,
        arrays=arrays,
        episode=episode,
        topics=topics,
        fps=fps,
        compression=compression,
        overwrite=overwrite,
        max_rgb_2d_age_sec=max_rgb_2d_age_sec,
        raw_events_h5=raw_events_h5,
        event_frame_windows_ms=event_frame_windows_ms,
        event_frame_mode=event_frame_mode,
        event_clip_count=event_clip_count,
        event_representation=event_representation,
        event_horizon_ms=event_horizon_ms,
        event_temporal_bins=event_temporal_bins,
        event_output_height=event_output_height,
        event_output_width=event_output_width,
    )
    write_sec = max(0.0, time.perf_counter() - write_start)
    total_sec = collect_sec + sample_sec + write_sec

    log(
        f"[INFO] episode {episode.output_idx}: "
        f"collect={collect_sec:.3f}s sample/event_render={sample_sec:.3f}s "
        f"write={write_sec:.3f}s total={total_sec:.3f}s"
    )

    length = int(arrays["timestamps"].size)
    del arrays
    data.clear()
    gc.collect()
    return length

def log_data_pass_progress(
    current_timestamp: float,
    completed_episodes: int,
    total_episodes: int,
    pass_start_wall: float,
    selected_start: float,
    selected_end: float,
) -> None:
    elapsed = max(0.0, time.perf_counter() - pass_start_wall)
    if total_episodes == 0:
        progress = 1.0
    elif selected_end <= selected_start:
        progress = 1.0 if completed_episodes >= total_episodes else 0.0
    else:
        clamped = min(max(current_timestamp, selected_start), selected_end)
        progress = (clamped - selected_start) / (selected_end - selected_start)
        progress = float(min(max(progress, 0.0), 1.0))
        if completed_episodes >= total_episodes:
            progress = 1.0

    stable_for_estimate = elapsed >= 15.0 and progress >= 0.05
    if stable_for_estimate:
        estimated_total = elapsed / progress
        remaining = max(0.0, estimated_total - elapsed)
        remaining_text = f"~{format_hms(remaining)}"
        total_text = f"~{format_hms(estimated_total)}"
    else:
        remaining_text = "n/a"
        total_text = "n/a"

    log(
        f"[INFO] Data pass: {progress * 100.0:.1f}% | "
        f"episodes {completed_episodes}/{total_episodes} | "
        f"elapsed {format_hms(elapsed)} | "
        f"remaining {remaining_text} | total {total_text}"
    )


def _validate_monotonic_non_decreasing(name: str, values: np.ndarray) -> None:
    if values.ndim != 1:
        raise RuntimeError(f"{name} timestamps must be 1-D")
    if values.size == 0:
        raise RuntimeError(f"{name} timestamps must be non-empty")
    diffs = np.diff(values)
    if np.any(diffs < 0.0):
        raise RuntimeError(f"{name} timestamps are not monotonic non-decreasing")


def sample_episode(
    data: Dict[str, Any],
    episode: EpisodeWindow,
    fps: float,
    max_current_tcp_s_age_sec: float,
    rgb_2d_enabled: bool = False,
    max_rgb_2d_age_sec: float = 0.10,
    raw_event_store: Optional[RawEventStore] = None,
    event_frame_windows_ms: Tuple[float, float, float] = (50.0, 100.0, 200.0),
    event_frame_mode: str = "shifted",
    event_clip_count: Optional[float] = None,
    event_packet_margin_ms: float = 50.0,
    event_representation: str = SHIFTED_3CHEF_REPRESENTATION,
    event_horizon_ms: float = 200.0,
    event_temporal_bins: int = 9,
    event_output_height: int = 320,
    event_output_width: int = 320,
) -> Dict[str, np.ndarray]:
    """Causally sample observations and measured current_tcp_s action."""
    rgb_times = np.asarray(data["rgb_t"], dtype=np.float64)
    joint_times = np.asarray(data["joint_t"], dtype=np.float64)
    tcp_s_times = np.asarray(data["current_tcp_s_t"], dtype=np.float64)
    tcp_s_values = np.asarray(data["current_tcp_s"], dtype=np.float32)

    _validate_monotonic_non_decreasing("RGB", rgb_times)
    _validate_monotonic_non_decreasing("joint", joint_times)
    _validate_monotonic_non_decreasing("current_tcp_s", tcp_s_times)

    effective_start = max(
        episode.start,
        float(rgb_times[0]),
        float(joint_times[0]),
        float(tcp_s_times[0]),
    )
    effective_end = min(
        episode.end,
        float(rgb_times[-1]),
        float(joint_times[-1]),
        float(tcp_s_times[-1]),
    )
    if effective_end <= effective_start:
        raise RuntimeError(
            f"Episode {episode.output_idx}: RGB/joint/current_tcp_s streams have "
            "no overlapping interval"
        )

    rate_hz = int(fps)
    if float(fps) != rate_hz or rate_hz <= 0:
        raise ValueError("fps must be a positive integer")
    grid_ns = policy_grid_ns(effective_start, effective_end, rate_hz)
    if grid_ns.size == 0:
        raise RuntimeError(f"Episode {episode.output_idx}: empty sampling grid")
    grid = grid_ns.astype(np.float64) * 1e-9

    rgb_indices = np.searchsorted(rgb_times, grid, side="right") - 1
    joint_indices = np.searchsorted(joint_times, grid, side="right") - 1
    tcp_s_indices = np.searchsorted(tcp_s_times, grid, side="right") - 1
    if np.any(rgb_indices < 0) or np.any(joint_indices < 0) or np.any(tcp_s_indices < 0):
        raise AssertionError("Internal error: non-causal source index")

    rgb = np.stack(
        [image_msg_to_rgb(data["rgb_msg"][index]) for index in rgb_indices],
        axis=0,
    ).astype(np.uint8, copy=False)

    rgb_2d_px = np.zeros((grid.size, 2), dtype=np.float32)
    rgb_valid = np.zeros((grid.size,), dtype=np.uint8)
    rgb_source_timestamps = np.full((grid.size,), np.nan, dtype=np.float64)
    if rgb_2d_enabled and data.get("rgb_2d_t"):
        detection_times = np.asarray(data["rgb_2d_t"], dtype=np.float64)
        detections = np.asarray(data["rgb_2d_px"], dtype=np.float32).reshape(-1, 2)
        source_times = np.asarray(data["rgb_2d_source_t"], dtype=np.float64)
        if source_times.shape != detection_times.shape:
            raise RuntimeError("RGB 2D source timestamp count does not match detections")
        _validate_monotonic_non_decreasing("RGB 2D", detection_times)
        detection_indices = np.searchsorted(detection_times, grid, side="right") - 1
        present = detection_indices >= 0
        if np.any(present):
            selected = detection_indices[present]
            points = detections[selected]
            rgb_source_timestamps[present] = source_times[selected]
            ages = grid[present] - detection_times[selected]
            heights = rgb[present].shape[1]
            widths = rgb[present].shape[2]
            valid = (
                np.all(np.isfinite(points), axis=1)
                & (points[:, 0] >= 0.0)
                & (points[:, 0] < widths)
                & (points[:, 1] >= 0.0)
                & (points[:, 1] < heights)
            )
            if max_rgb_2d_age_sec > 0.0:
                valid &= ages <= float(max_rgb_2d_age_sec)
            present_rows = np.flatnonzero(present)
            valid_rows = present_rows[valid]
            rgb_2d_px[valid_rows] = points[valid]
            rgb_valid[valid_rows] = 1
    qpos = np.stack(
        [data["qpos"][index] for index in joint_indices], axis=0
    ).astype(np.float32, copy=False)

    action_values = tcp_s_values[tcp_s_indices]
    if not np.all(np.isfinite(action_values)):
        raise RuntimeError(
            f"Episode {episode.output_idx}: sampled current_tcp_s contains non-finite values"
        )

    action = action_values.reshape(-1, 1).astype(np.float32, copy=False)
    action_source_timestamps = tcp_s_times[tcp_s_indices]
    action_source_age_sec = (grid - action_source_timestamps).astype(np.float32)

    if np.any(action_source_age_sec < 0.0):
        min_age = float(np.min(action_source_age_sec))
        raise RuntimeError(
            f"Episode {episode.output_idx}: negative current_tcp_s source age detected "
            f"(min={min_age:.6e}s)"
        )

    if max_current_tcp_s_age_sec > 0.0:
        too_old = action_source_age_sec > float(max_current_tcp_s_age_sec)
        if np.any(too_old):
            max_age = float(np.max(action_source_age_sec))
            raise RuntimeError(
                f"Episode {episode.output_idx}: stale current_tcp_s source sample "
                f"detected (max age={max_age:.6f}s > "
                f"{max_current_tcp_s_age_sec:.6f}s)."
            )

    if action.shape != (grid.size, 1):
        raise RuntimeError(
            f"Episode {episode.output_idx}: action shape mismatch {action.shape}, "
            f"expected {(grid.size, 1)}"
        )
    if rgb.shape[0] != grid.size or qpos.shape[0] != grid.size:
        raise RuntimeError(
            f"Episode {episode.output_idx}: sampled array length mismatch "
            f"(T={grid.size}, rgb={rgb.shape[0]}, qpos={qpos.shape[0]})"
        )

    event: Optional[np.ndarray] = None
    event_source_timestamps: Optional[np.ndarray] = None
    event_source_age_sec: Optional[np.ndarray] = None
    event_count_per_channel: Optional[np.ndarray] = None
    if raw_event_store is not None:
        if event_clip_count is None or float(event_clip_count) <= 0.0:
            raise RuntimeError(
                "event_clip_count must be explicitly provided and positive when "
                "raw-event sidecar conversion is enabled"
            )
        if event_representation == SHIFTED_3CHEF_REPRESENTATION and event_frame_mode != "shifted":
            raise RuntimeError(
                "Interception event conversion supports only shifted mode"
            )

        if event_representation not in (
            SHIFTED_3CHEF_REPRESENTATION,
            XYT_REPRESENTATION,
        ):
            raise RuntimeError(
                f"Unsupported event_representation: {event_representation}"
            )

        max_window_ms = (
            max(event_frame_windows_ms)
            if event_representation == SHIFTED_3CHEF_REPRESENTATION
            else float(event_horizon_ms)
        )
        max_window_ns = int(np.rint(max_window_ms * 1e6))
        required_start_ns = int(grid_ns[0] - max_window_ns)
        required_end_ns = int(grid_ns[-1])
        required_start_sec = required_start_ns * 1e-9
        required_end_sec = required_end_ns * 1e-9
        available_start_ns = raw_event_store.packet_ros_start_ns
        available_end_ns = raw_event_store.packet_ros_end_ns
        episode_name = f"episode_{episode.output_idx}"

        if available_start_ns is None or available_end_ns is None:
            raise RuntimeError(
                f"{episode_name}: raw-event sidecar has no packet timestamps; "
                f"required interval {required_start_sec:.6f}s..{required_end_sec:.6f}s"
            )
        if available_start_ns > required_start_ns or available_end_ns < required_end_ns:
            available_start_sec = available_start_ns * 1e-9
            available_end_sec = available_end_ns * 1e-9
            raise RuntimeError(
                f"{episode_name}: raw-event sidecar coverage is incomplete; "
                f"required interval={required_start_sec:.6f}s..{required_end_sec:.6f}s, "
                f"available interval={available_start_sec:.6f}s..{available_end_sec:.6f}s"
            )

        event_frames: List[np.ndarray] = []
        event_source_ns = np.empty((grid.size,), dtype=np.int64)
        event_channel_count = (
            3
            if event_representation == SHIFTED_3CHEF_REPRESENTATION
            else int(event_temporal_bins)
        )
        event_counts = np.empty((grid.size, event_channel_count), dtype=np.int32)
        for index, t_g in enumerate(grid):
            if event_representation == SHIFTED_3CHEF_REPRESENTATION:
                frame_u8, source_packet_ros_t_ns, counts = (
                    raw_event_store.frame_3chef_with_metadata_at_bag_time(
                        bag_t_sec=float(t_g),
                        windows_ms=event_frame_windows_ms,
                        mode=event_frame_mode,
                        packet_margin_ms=event_packet_margin_ms,
                        scaling_mode="signed_log1p_fixed_clip",
                        event_clip_count=float(event_clip_count),
                    )
                )
            else:
                frame_u8, source_packet_ros_t_ns, counts = (
                    raw_event_store.xyt_signed_voxel_with_metadata_at_bag_time(
                    bag_t_sec=float(t_g),
                    horizon_ms=event_horizon_ms,
                    temporal_bins=event_temporal_bins,
                    output_height=event_output_height,
                    output_width=event_output_width,
                    packet_margin_ms=event_packet_margin_ms,
                    event_clip_count=float(event_clip_count),
                    )
                )

            event_frames.append(frame_u8)
            event_counts[index, :] = counts
            if source_packet_ros_t_ns is None:
                raise RuntimeError(
                    f"{episode_name}: missing causal event source packet for "
                    f"grid timestamp {float(t_g):.6f}s"
                )

            source_ns = int(source_packet_ros_t_ns)
            if source_ns > int(grid_ns[index]):
                raise RuntimeError(
                    f"{episode_name}: non-causal event source packet detected "
                    f"(source={source_ns}, grid={int(grid_ns[index])})"
                )
            event_source_ns[index] = source_ns

        event = np.stack(event_frames, axis=0).astype(np.uint8, copy=False)
        event_source_timestamps = event_source_ns.astype(np.float64) * 1e-9
        event_age_ns = grid_ns - event_source_ns
        event_source_age_sec = (event_age_ns.astype(np.float64) * 1e-9).astype(
            np.float32
        )
        if np.any(event_source_age_sec < 0.0):
            min_age = float(np.min(event_source_age_sec))
            raise RuntimeError(
                f"Episode {episode.output_idx}: negative event source age detected "
                f"(min={min_age:.6e}s)"
            )
        event_count_per_channel = event_counts

    log(
        f"[INFO] episode {episode.output_idx}: sampled {grid.size} steps at "
        f"{fps:.3f} Hz; current_tcp_s age sec min/mean/max="
        f"{float(np.min(action_source_age_sec)):.6f}/"
        f"{float(np.mean(action_source_age_sec)):.6f}/"
        f"{float(np.max(action_source_age_sec)):.6f}; s min/mean/max="
        f"{float(np.min(action[:, 0])):+.6f}/"
        f"{float(np.mean(action[:, 0])):+.6f}/"
        f"{float(np.max(action[:, 0])):+.6f}"
    )

    command_timestamps = np.asarray(data["goto_s_t"], dtype=np.float64)
    command_values = np.asarray(data["goto_s"], dtype=np.float32).reshape(-1, 1)
    target_base_t = np.asarray(data["target_base_t"], dtype=np.float64)
    if data["target_base"]:
        target_base = np.asarray(data["target_base"], dtype=np.float32).reshape(-1, 3)
    else:
        target_base = np.empty((0, 3), dtype=np.float32)

    arrays = {
        "timestamps": grid,
        "rgb": rgb,
        "qpos": qpos,
        "action": action,
        "action_source_timestamps": action_source_timestamps,
        "action_source_age_sec": action_source_age_sec,
        "command_timestamps": command_timestamps,
        "command_values": command_values,
        "target_base_timestamps": target_base_t,
        "target_base_points": target_base,
    }

    if rgb_2d_enabled:
        arrays["rgb_2d_px"] = rgb_2d_px
        arrays["rgb_valid"] = rgb_valid
        arrays["rgb_source_timestamps"] = rgb_source_timestamps

    if event is not None:
        arrays["event"] = event
        arrays["event_source_timestamps"] = event_source_timestamps
        arrays["event_source_age_sec"] = event_source_age_sec
        arrays["event_count_per_channel"] = event_count_per_channel

    return arrays


def dataset_kwargs(compression: str) -> Dict[str, Any]:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 1}
    return {"compression": "lzf"}


def write_episode(
    output_path: str,
    arrays: Dict[str, np.ndarray],
    episode: EpisodeWindow,
    topics: Topics,
    fps: float,
    compression: str,
    overwrite: bool,
    max_rgb_2d_age_sec: float = 0.10,
    raw_events_h5: Optional[str] = None,
    event_frame_windows_ms: Tuple[float, float, float] = (50.0, 100.0, 200.0),
    event_frame_mode: str = "shifted",
    event_clip_count: Optional[float] = None,
    event_representation: str = SHIFTED_3CHEF_REPRESENTATION,
    event_horizon_ms: float = 200.0,
    event_temporal_bins: int = 9,
    event_output_height: int = 320,
    event_output_width: int = 320,
) -> None:
    if os.path.exists(output_path) and not overwrite:
        raise RuntimeError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    temporary_path = output_path + ".tmp"
    if os.path.exists(temporary_path):
        raise RuntimeError(
            f"Temporary output already exists, perhaps from a failed run: {temporary_path}"
        )

    compression_kwargs = dataset_kwargs(compression)
    try:
        with h5py.File(temporary_path, "w") as h5:
            h5.attrs["sim"] = False
            h5.attrs["task"] = "ball_interception"
            h5.attrs["fps"] = float(fps)
            h5.attrs["policy_rate_hz"] = int(fps)
            h5.attrs["policy_period_ns"] = (
                policy_period_ns(fps)
                if int(fps) in (30, 60)
                else int(round(1_000_000_000 / float(fps)))
            )
            h5.attrs["episode_index"] = int(episode.output_idx)
            h5.attrs["source_episode_index"] = int(episode.source_idx)
            h5.attrs["episode_start"] = float(episode.start)
            h5.attrs["episode_end"] = float(episode.end)
            h5.attrs["effective_episode_start"] = float(arrays["timestamps"][0])
            h5.attrs["effective_episode_end"] = float(arrays["timestamps"][-1])
            h5.attrs["joint_names"] = np.asarray(
                ARM_JOINT_NAMES, dtype=h5py.string_dtype("utf-8")
            )
            h5.attrs["action_type"] = "measured_tcp_s_absolute"
            h5.attrs["action_representation"] = "absolute"
            h5.attrs["action_coordinate"] = "captured_interception_line"
            h5.attrs["action_origin"] = "captured_line_center"
            h5.attrs["action_positive_direction"] = "robot_base_positive_x"
            h5.attrs["action_units"] = "m"
            h5.attrs["action_source_topic"] = topics.current_tcp_s
            h5.attrs["action_sampling_policy"] = (
                "latest_message_at_or_before_grid_time"
            )
            h5.attrs["delta_action_construction"] = "deferred_to_training_loader"
            h5.attrs["rgb_source_topic"] = topics.rgb
            h5.attrs["joint_source_topic"] = topics.joint
            h5.attrs["rgb_temporal_stacking"] = "deferred_to_training_loader"
            h5.attrs["command_count"] = int(arrays["command_timestamps"].size)

            has_event_sidecar = "event" in arrays
            if has_event_sidecar:
                if raw_events_h5 is None:
                    raise RuntimeError(
                        "event data present but raw_events_h5 metadata path is missing"
                    )
                if event_clip_count is None:
                    raise RuntimeError(
                        "event data present but event_clip_count is missing"
                    )
                if (
                    event_representation == SHIFTED_3CHEF_REPRESENTATION
                    and event_frame_mode != "shifted"
                ):
                    raise RuntimeError(
                        "Inconsistent event metadata: interception event sidecar "
                        "must use event_frame_mode='shifted'"
                    )
                h5.attrs["event_representation"] = event_representation
                if event_representation == SHIFTED_3CHEF_REPRESENTATION:
                    h5.attrs["event_frame_windows_ms"] = np.asarray(
                        event_frame_windows_ms,
                        dtype=np.float32,
                    )
                    h5.attrs["event_frame_mode"] = event_frame_mode
                    h5.attrs["event_channel_order"] = "recent_to_oldest"
                elif event_representation == XYT_REPRESENTATION:
                    h5.attrs["event_horizon_ms"] = float(event_horizon_ms)
                    h5.attrs["event_temporal_bins"] = int(event_temporal_bins)
                    h5.attrs["event_bin_width_ms"] = float(event_horizon_ms) / int(
                        event_temporal_bins
                    )
                    h5.attrs["event_spatial_height"] = int(event_output_height)
                    h5.attrs["event_spatial_width"] = int(event_output_width)
                    h5.attrs["event_channel_order"] = "oldest_to_newest"
                    h5.attrs["event_polarity_encoding"] = "signed"
                    h5.attrs["visual_history_frames"] = 1
                    h5.attrs["visual_history_offsets"] = np.asarray([0], dtype=np.int32)
                    h5.attrs["qpos_history_frames"] = 3
                    h5.attrs["qpos_history_offsets"] = np.asarray(
                        [-6, -3, 0], dtype=np.int32
                    )
                    h5.attrs["channels_per_visual_frame"] = int(event_temporal_bins)
                    h5.attrs["image_channels"] = int(event_temporal_bins)
                else:
                    raise RuntimeError(
                        f"Unsupported event_representation: {event_representation}"
                    )
                h5.attrs["event_scaling"] = "signed_log1p_fixed_clip"
                h5.attrs["event_clip_count"] = float(event_clip_count)
                h5.attrs["event_neutral_u8"] = 128
                h5.attrs["event_sampling_policy"] = (
                    "latest_packet_at_or_before_grid_time"
                )
                h5.attrs["raw_events_h5"] = str(raw_events_h5)

            observations = h5.create_group("observations")
            images = observations.create_group("images")
            observations.create_dataset(
                "timestamps", data=arrays["timestamps"], dtype=np.float64
            )
            observations.create_dataset(
                "qpos", data=arrays["qpos"], dtype=np.float32
            )
            if "rgb_2d_px" in arrays:
                sparse = observations.create_group("sparse_tracking")
                sparse.create_dataset(
                    "rgb_2d_px", data=arrays["rgb_2d_px"], dtype=np.float32
                )
                sparse.create_dataset(
                    "rgb_valid", data=arrays["rgb_valid"], dtype=np.uint8
                )
                sparse.create_dataset(
                    "rgb_source_timestamps",
                    data=arrays["rgb_source_timestamps"],
                    dtype=np.float64,
                )
                sparse.attrs["rgb_source_topic"] = topics.rgb_2d
                sparse.attrs["rgb_sampling_policy"] = (
                    "latest_message_at_or_before_observation_timestamp"
                )
                sparse.attrs["rgb_max_source_age_sec"] = float(
                    max_rgb_2d_age_sec
                )
                sparse.attrs["rgb_invalid_coordinate_fill"] = 0.0
                sparse.attrs["raw_coordinate_names"] = np.asarray(
                    ["u_px", "v_px"], dtype=h5py.string_dtype("utf-8")
                )
                sparse.attrs["raw_coordinate_units"] = "pixels"
                sparse.attrs["rgb_width_px"] = int(arrays["rgb"].shape[2])
                sparse.attrs["rgb_height_px"] = int(arrays["rgb"].shape[1])
            images.create_dataset(
                "rgb",
                data=arrays["rgb"],
                dtype=np.uint8,
                chunks=(1, *arrays["rgb"].shape[1:]),
                **compression_kwargs,
            )
            if has_event_sidecar:
                images.create_dataset(
                    "event",
                    data=arrays["event"],
                    dtype=np.uint8,
                    chunks=(1, *arrays["event"].shape[1:]),
                    **compression_kwargs,
                )

            h5.create_dataset("action", data=arrays["action"], dtype=np.float32)
            h5.create_dataset(
                "action_source_timestamps",
                data=arrays["action_source_timestamps"],
                dtype=np.float64,
            )
            h5.create_dataset(
                "action_source_age_sec",
                data=arrays["action_source_age_sec"],
                dtype=np.float32,
            )
            if has_event_sidecar:
                h5.create_dataset(
                    "event_source_timestamps",
                    data=arrays["event_source_timestamps"],
                    dtype=np.float64,
                )
                h5.create_dataset(
                    "event_source_age_sec",
                    data=arrays["event_source_age_sec"],
                    dtype=np.float32,
                )
                h5.create_dataset(
                    "event_count_per_channel",
                    data=arrays["event_count_per_channel"],
                    dtype=np.int32,
                )

            commands = h5.create_group("commands")
            goto_s = commands.create_group("goto_s")
            goto_s.attrs["source_topic"] = topics.goto_s
            goto_s.create_dataset(
                "timestamps", data=arrays["command_timestamps"], dtype=np.float64
            )
            goto_s.create_dataset(
                "values", data=arrays["command_values"], dtype=np.float32
            )

            target_base = commands.create_group("goto_s_target_base")
            target_base.attrs["source_topic"] = topics.goto_s_target_base
            target_base.attrs["frame"] = "message_header_frame_id"
            target_base.create_dataset(
                "timestamps",
                data=arrays["target_base_timestamps"],
                dtype=np.float64,
            )
            target_base.create_dataset(
                "points", data=arrays["target_base_points"], dtype=np.float32
            )

        os.replace(temporary_path, output_path)
    except Exception:
        # Preserve a partially written .tmp file for diagnosis rather than deleting it.
        raise

    log(f"[INFO] wrote {output_path}")


def resolve_input_paths(
    args: argparse.Namespace,
) -> Tuple[str, Optional[str]]:
    if args.bag is not None:
        bag_path = os.path.abspath(os.path.expanduser(args.bag))
        raw_events_h5 = None
        if args.raw_events_h5 is not None:
            raw_events_h5 = os.path.abspath(os.path.expanduser(args.raw_events_h5))
        return bag_path, raw_events_h5

    rec_dir = os.path.abspath(os.path.expanduser(args.rec_dir))
    bag_path, auto_raw_events_h5, recording_name = resolve_recording_dir(
        rec_dir,
        allow_missing_raw_events=True,
        logger=log,
    )
    if args.raw_events_h5 is not None:
        return bag_path, os.path.abspath(os.path.expanduser(args.raw_events_h5))

    # In --rec_dir mode, event conversion is expected to use a colocated sidecar
    # named <recording_name>_raw_events.h5 when --event_clip_count is provided.
    if auto_raw_events_h5 is not None:
        return bag_path, auto_raw_events_h5
    if args.event_clip_count is not None:
        expected_sidecar = os.path.join(
            rec_dir, f"{recording_name}_raw_events.h5"
        )
        return bag_path, expected_sidecar
    return bag_path, None


def resolve_output_dir(out_dir: str) -> str:
    """Resolve --out_dir as the exact episode destination directory."""
    return os.path.abspath(os.path.expanduser(out_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a ROS 2 ball-interception bag to per-episode IL HDF5"
    )
    bag_input_group = parser.add_mutually_exclusive_group(required=True)
    bag_input_group.add_argument("--bag", help="ROS 2 bag directory")
    bag_input_group.add_argument(
        "--rec_dir",
        help=(
            "Recording directory containing a bag subdirectory ending in '_bag'"
        ),
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Exact directory in which episode_*.hdf5 files are written",
    )
    parser.add_argument(
        "--storage_id", choices=("mcap", "sqlite3"), default="mcap"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.5,
        help="Drop committed episodes shorter than this many seconds",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--max_current_tcp_s_age_sec",
        type=float,
        default=0.10,
        help=(
            "Maximum allowed age for causally selected current_tcp_s source "
            "samples. Non-positive disables rejection but age stats are still logged."
        ),
    )
    parser.add_argument(
        "--no_target_base",
        action="store_true",
        help="Do not collect optional executed_goto_s_target_base debug messages",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "lzf", "gzip"),
        default="none",
        help="HDF5 compression for RGB frames and event frames when enabled",
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--raw_events_h5",
        default=None,
        help=(
            "Optional raw-event sidecar HDF5. When provided (or auto-resolved via "
            "--rec_dir), event tensors are rendered from raw packets."
        ),
    )
    parser.add_argument(
        "--event_representation",
        choices=(SHIFTED_3CHEF_REPRESENTATION, XYT_REPRESENTATION),
        default=SHIFTED_3CHEF_REPRESENTATION,
        help=(
            "Raw-event representation. The default preserves shifted-3Chef; "
            "xyt_signed_voxel_v1 writes one causal HWC temporal volume per observation."
        ),
    )
    parser.add_argument("--event_horizon_ms", type=float, default=200.0)
    parser.add_argument("--event_temporal_bins", type=int, default=9)
    parser.add_argument("--event_output_height", type=int, default=320)
    parser.add_argument("--event_output_width", type=int, default=320)
    parser.add_argument(
        "--event_frame_windows_ms",
        type=float,
        nargs=3,
        default=[50.0, 100.0, 200.0],
        help="Three shifted/cumulative event windows in milliseconds.",
    )
    parser.add_argument(
        "--event_frame_mode",
        default="shifted",
        help="Event channel binning mode (shifted only for interception conversion).",
    )
    parser.add_argument(
        "--event_clip_count",
        type=float,
        default=16.0,
        help=(
            "Fixed clip denominator for signed_log1p_fixed_clip rendering. "
            "Default: 16."
        ),
    )
    parser.add_argument(
        "--event_packet_margin_ms",
        type=float,
        default=50.0,
        help="Extra raw-event packet margin before the largest window.",
    )

    parser.add_argument("--rgb_topic", default=DEFAULT_RGB_TOPIC)
    parser.add_argument("--rgb_2d_topic", default=DEFAULT_RGB_2D_TOPIC)
    parser.add_argument(
        "--no-rgb-2d",
        action="store_true",
        help="Do not extract RGB ball detections or write sparse RGB tracking datasets.",
    )
    parser.add_argument(
        "--max_rgb_2d_age_sec",
        type=float,
        default=0.10,
        help=(
            "Maximum age of a causal RGB 2D detection to mark valid. "
            "Non-positive disables the age limit. Default: 0.10 s."
        ),
    )
    parser.add_argument("--joint_topic", default=DEFAULT_JOINT_TOPIC)
    parser.add_argument("--episode_topic", default=DEFAULT_EPISODE_TOPIC)
    parser.add_argument(
        "--current_tcp_s_topic", default=DEFAULT_CURRENT_TCP_S_TOPIC
    )
    parser.add_argument("--goto_s_topic", default=DEFAULT_GOTO_S_TOPIC)
    parser.add_argument(
        "--goto_s_target_base_topic", default=DEFAULT_GOTO_S_TARGET_BASE_TOPIC
    )
    args = parser.parse_args()

    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.min_duration < 0.0:
        parser.error("--min_duration must be non-negative")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max_episodes must be positive")
    if len(args.event_frame_windows_ms) != 3:
        parser.error("--event_frame_windows_ms must contain exactly 3 values")
    if any(w <= 0.0 for w in args.event_frame_windows_ms):
        parser.error("--event_frame_windows_ms values must be positive")
    if any(
        args.event_frame_windows_ms[i] > args.event_frame_windows_ms[i + 1]
        for i in range(2)
    ):
        parser.error(
            "--event_frame_windows_ms must be non-decreasing, e.g. 50 100 200"
        )
    if args.event_clip_count is not None and args.event_clip_count <= 0.0:
        parser.error("--event_clip_count must be positive when provided")
    if args.event_packet_margin_ms < 0.0:
        parser.error("--event_packet_margin_ms must be non-negative")
    if (
        args.event_representation == SHIFTED_3CHEF_REPRESENTATION
        and args.event_frame_mode != "shifted"
    ):
        parser.error(
            "--event_frame_mode must be 'shifted' for interception conversion"
        )
    if args.event_horizon_ms <= 0.0:
        parser.error("--event_horizon_ms must be positive")
    if args.event_temporal_bins <= 0:
        parser.error("--event_temporal_bins must be positive")
    if args.event_output_height <= 0 or args.event_output_width <= 0:
        parser.error("--event_output_height and --event_output_width must be positive")
    if args.event_representation == XYT_REPRESENTATION:
        if not np.isclose(args.event_horizon_ms, 200.0):
            parser.error("xyt_signed_voxel_v1 requires --event_horizon_ms 200")
        if args.event_temporal_bins != 9:
            parser.error("xyt_signed_voxel_v1 requires --event_temporal_bins 9")
        if args.event_clip_count is None or not np.isclose(args.event_clip_count, 16.0):
            parser.error("xyt_signed_voxel_v1 requires --event_clip_count 16")
    return args


def main() -> None:
    conversion_start_wall = time.perf_counter()
    args = parse_args()
    # Preserve compatibility with programmatic callers/tests that provide a
    # pre-XYT argparse namespace.
    for name, default in (
        ("event_representation", SHIFTED_3CHEF_REPRESENTATION),
        ("event_horizon_ms", 200.0),
        ("event_temporal_bins", 9),
        ("event_output_height", 320),
        ("event_output_width", 320),
        ("rgb_2d_topic", DEFAULT_RGB_2D_TOPIC),
        ("no_rgb_2d", False),
        ("max_rgb_2d_age_sec", 0.10),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)
    bag_path, raw_events_h5 = resolve_input_paths(args)
    output_dir = resolve_output_dir(args.out_dir)
    if not os.path.exists(bag_path):
        raise RuntimeError(f"Bag path does not exist: {bag_path}")
    if raw_events_h5 is not None and not os.path.exists(raw_events_h5):
        raise RuntimeError(f"Raw-event sidecar does not exist: {raw_events_h5}")
    if raw_events_h5 is not None and args.event_clip_count is None:
        raise RuntimeError(
            "--event_clip_count is required when event conversion is enabled"
        )

    log(f"[INFO] bag: {bag_path}")
    if raw_events_h5 is not None:
        log(f"[INFO] raw events H5: {raw_events_h5}")
    else:
        log("[INFO] raw events H5: disabled")
    log(f"[INFO] storage: {args.storage_id}")

    marker_pass_start_wall = time.perf_counter()
    marker_reader = open_reader(bag_path, args.storage_id)
    types = topic_type_map(marker_reader)
    selected_rgb_topic = resolve_rgb_topic(types, args.rgb_topic)

    topics = Topics(
        rgb=selected_rgb_topic,
        rgb_2d=args.rgb_2d_topic,
        joint=args.joint_topic,
        episode=args.episode_topic,
        current_tcp_s=args.current_tcp_s_topic,
        goto_s=args.goto_s_topic,
        goto_s_target_base=args.goto_s_target_base_topic,
    )

    collect_goto_s_debug, collect_target_base_debug = validate_topics(
        types,
        topics,
        collect_goto_s_debug=True,
        collect_target_base_debug=not args.no_target_base,
        collect_rgb_2d=not args.no_rgb_2d,
    )
    collect_rgb_2d = (
        not args.no_rgb_2d
        and types.get(topics.rgb_2d) == POINT_STAMPED_TYPE
    )

    marker_filter = apply_storage_filter(marker_reader, [topics.episode])
    log(f"[INFO] marker filter topics: {marker_filter}")

    episodes = extract_episode_windows(
        reader=marker_reader,
        types=types,
        episode_topic=topics.episode,
        min_duration=args.min_duration,
    )
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    validate_non_overlapping_windows(episodes)

    marker_elapsed = max(0.0, time.perf_counter() - marker_pass_start_wall)
    log(
        f"[INFO] marker scan complete: retained {len(episodes)} episodes "
        f"in {format_hms(marker_elapsed)}"
    )

    os.makedirs(output_dir, exist_ok=True)
    log(f"[INFO] output directory: {output_dir}")

    if not episodes:
        raise RuntimeError("No episodes selected for conversion")

    data_topics = required_data_topics(
        topics=topics,
        collect_goto_s_debug=collect_goto_s_debug,
        collect_target_base_debug=collect_target_base_debug,
        collect_rgb_2d=collect_rgb_2d,
    )
    log(f"[INFO] data filter topics: {data_topics}")

    data_reader = open_filtered_reader(
        bag_path=bag_path,
        storage_id=args.storage_id,
        topics=data_topics,
    )
    classes = message_classes(types, set(data_topics))
    missing_classes = [topic for topic in data_topics if topic not in classes]
    if missing_classes:
        raise RuntimeError(
            f"Missing message class definitions for topics: {missing_classes}"
        )

    selected_start = episodes[0].start
    selected_end = episodes[-1].end
    log(
        "[INFO] selected episode time range (sec): "
        f"{selected_start:.6f} .. {selected_end:.6f}"
    )
    log("[INFO] Pass 2: streaming selected episodes with one filtered data reader")

    raw_event_store: Optional[RawEventStore] = None
    if raw_events_h5 is not None:
        raw_event_store = RawEventStore(raw_events_h5, logger=log)
        if episodes:
            selected_start = min(ep.start for ep in episodes)
            selected_end = max(ep.end for ep in episodes)
            if (
                raw_event_store.packet_ros_start_ns is None
                or raw_event_store.packet_ros_end_ns is None
            ):
                raise RuntimeError(
                    "Raw-event sidecar has no packet ROS timestamps and cannot be "
                    "used for event conversion"
                )
            raw_start_sec = raw_event_store.packet_ros_start_ns * 1e-9
            raw_end_sec = raw_event_store.packet_ros_end_ns * 1e-9
            if selected_end < raw_start_sec or selected_start > raw_end_sec:
                raise RuntimeError(
                    "Raw-event sidecar packet ROS timestamps do not overlap selected "
                    "episode windows"
                )

    lengths: List[int] = []
    try:
        total_episodes = len(episodes)
        completed_episodes = 0
        active_index = 0
        active_buffer = create_episode_buffer()
        active_collect_start_wall = time.perf_counter()
        data_pass_start_wall = time.perf_counter()
        last_progress_wall = data_pass_start_wall
        progress_interval_sec = 7.0
        latest_timestamp = selected_start

        while data_reader.has_next() and active_index < total_episodes:
            topic, raw, timestamp_ns = data_reader.read_next()
            timestamp = ns_to_sec(timestamp_ns)
            latest_timestamp = timestamp

            while active_index < total_episodes and timestamp > episodes[active_index].end:
                episode = episodes[active_index]
                lengths.append(
                    finalize_episode(
                        episode=episode,
                        data=active_buffer,
                        output_dir=output_dir,
                        topics=topics,
                        fps=args.fps,
                        max_current_tcp_s_age_sec=args.max_current_tcp_s_age_sec,
                        rgb_2d_enabled=not args.no_rgb_2d,
                        max_rgb_2d_age_sec=args.max_rgb_2d_age_sec,
                        compression=args.compression,
                        overwrite=args.overwrite,
                        collect_started_wall=active_collect_start_wall,
                        raw_event_store=raw_event_store,
                        raw_events_h5=raw_events_h5,
                        event_frame_windows_ms=tuple(args.event_frame_windows_ms),
                        event_frame_mode=args.event_frame_mode,
                        event_clip_count=args.event_clip_count,
                        event_packet_margin_ms=args.event_packet_margin_ms,
                        event_representation=args.event_representation,
                        event_horizon_ms=args.event_horizon_ms,
                        event_temporal_bins=args.event_temporal_bins,
                        event_output_height=args.event_output_height,
                        event_output_width=args.event_output_width,
                    )
                )
                completed_episodes += 1
                active_index += 1
                if active_index < total_episodes:
                    active_buffer = create_episode_buffer()
                    active_collect_start_wall = time.perf_counter()

            if active_index >= total_episodes:
                break

            active_episode = episodes[active_index]
            if timestamp < active_episode.start:
                pass
            elif timestamp <= active_episode.end:
                msg = deserialize_message(raw, classes[topic])
                ingest_episode_message(
                    data=active_buffer,
                    topic=topic,
                    msg=msg,
                    timestamp=timestamp,
                    topics=topics,
                )

            now = time.perf_counter()
            if now - last_progress_wall >= progress_interval_sec:
                log_data_pass_progress(
                    current_timestamp=timestamp,
                    completed_episodes=completed_episodes,
                    total_episodes=total_episodes,
                    pass_start_wall=data_pass_start_wall,
                    selected_start=selected_start,
                    selected_end=selected_end,
                )
                last_progress_wall = now

        while active_index < total_episodes:
            episode = episodes[active_index]
            lengths.append(
                finalize_episode(
                    episode=episode,
                    data=active_buffer,
                    output_dir=output_dir,
                    topics=topics,
                    fps=args.fps,
                    max_current_tcp_s_age_sec=args.max_current_tcp_s_age_sec,
                    rgb_2d_enabled=not args.no_rgb_2d,
                    max_rgb_2d_age_sec=args.max_rgb_2d_age_sec,
                    compression=args.compression,
                    overwrite=args.overwrite,
                    collect_started_wall=active_collect_start_wall,
                    raw_event_store=raw_event_store,
                    raw_events_h5=raw_events_h5,
                    event_frame_windows_ms=tuple(args.event_frame_windows_ms),
                    event_frame_mode=args.event_frame_mode,
                    event_clip_count=args.event_clip_count,
                    event_packet_margin_ms=args.event_packet_margin_ms,
                    event_representation=args.event_representation,
                    event_horizon_ms=args.event_horizon_ms,
                    event_temporal_bins=args.event_temporal_bins,
                    event_output_height=args.event_output_height,
                    event_output_width=args.event_output_width,
                )
            )
            completed_episodes += 1
            active_index += 1
            if active_index < total_episodes:
                active_buffer = create_episode_buffer()
                active_collect_start_wall = time.perf_counter()

        log_data_pass_progress(
            current_timestamp=max(latest_timestamp, selected_end),
            completed_episodes=completed_episodes,
            total_episodes=total_episodes,
            pass_start_wall=data_pass_start_wall,
            selected_start=selected_start,
            selected_end=selected_end,
        )
    finally:
        if raw_event_store is not None:
            raw_event_store.close()

    log("[INFO] conversion complete")
    log(f"       episodes: {len(lengths)}")
    log(
        f"       lengths: min={min(lengths)}, "
        f"mean={float(np.mean(lengths)):.2f}, max={max(lengths)}"
    )
    total_wall = max(0.0, time.perf_counter() - conversion_start_wall)
    log(f"       total wall time: {format_hms(total_wall)}")


if __name__ == "__main__":
    main()
