#!/usr/bin/env python3
"""Convert ROS 2 ball-interception bags into per-episode IL HDF5 files.

The learned action is a two-dimensional dense representation:

    action[:, 0] = executed episode GOTO_S target at every timestep
    action[:, 1] = 0 before the executed command timestamp, 1 at/after it

Images and joint states are sampled causally: every 30 Hz sample uses the most
recent message whose bag timestamp is less than or equal to the sample time.
Temporal RGB stacking (for example rgb[t-1] + rgb[t]) is intentionally left to
the training data loader.
"""

from __future__ import annotations

import argparse
import gc
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import h5py
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


DEFAULT_RGB_TOPIC = "/top_cam/camera/color/image_raw"
DEFAULT_JOINT_TOPIC = "/joint_states"
DEFAULT_EPISODE_TOPIC = "/episode/control"
DEFAULT_GOTO_S_TOPIC = "/trajectory_executor/executed_goto_s"
DEFAULT_GOTO_S_TARGET_BASE_TOPIC = (
    "/trajectory_executor/executed_goto_s_target_base"
)

ARM_JOINT_NAMES = (
    "right_fr3_joint1",
    "right_fr3_joint2",
    "right_fr3_joint3",
    "right_fr3_joint4",
    "right_fr3_joint5",
    "right_fr3_joint6",
    "right_fr3_joint7",
)


@dataclass
class Topics:
    rgb: str
    joint: str
    episode: str
    goto_s: str
    goto_s_target_base: str


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


def topic_type_map(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
    return {item.name: item.type for item in reader.get_all_topics_and_types()}


def message_classes(
    types: Dict[str, str], selected_topics: Set[str]
) -> Dict[str, Any]:
    return {
        topic: get_message(types[topic])
        for topic in selected_topics
        if topic in types
    }


def validate_topics(
    types: Dict[str, str], topics: Topics, collect_target_base: bool
) -> bool:
    required = {topics.rgb, topics.joint, topics.episode, topics.goto_s}
    missing = sorted(required - set(types))
    if missing:
        raise RuntimeError(f"Missing required topics: {missing}")

    for topic in sorted(required):
        log(f"[INFO] {topic} :: {types[topic]}")

    target_base_available = topics.goto_s_target_base in types
    if collect_target_base and not target_base_available:
        log(
            "[WARNING] optional GOTO_S target-base topic is absent; "
            "no target-base debug data will be written"
        )
    return collect_target_base and target_base_available


def extract_episode_windows(
    bag_path: str,
    storage_id: str,
    episode_topic: str,
    min_duration: float,
) -> List[EpisodeWindow]:
    """Apply the recording manager's start/stop/cancel protocol."""
    log("[INFO] Pass 1: scanning episode-control markers")
    reader = open_reader(bag_path, storage_id)
    types = topic_type_map(reader)
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


def image_msg_to_rgb(msg: Any) -> np.ndarray:
    """Decode common sensor_msgs/Image encodings and return RGB uint8."""
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
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)

    image = packed.reshape(height, width, channels)
    if encoding == "rgb8":
        return image.copy()
    if encoding == "bgr8":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)


def arm_qpos(joint_names: Sequence[str], positions: Sequence[float]) -> np.ndarray:
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    missing = [name for name in ARM_JOINT_NAMES if name not in name_to_index]
    if missing:
        raise RuntimeError(f"Missing required FR3 arm joints: {missing}")
    return np.asarray(
        [positions[name_to_index[name]] for name in ARM_JOINT_NAMES],
        dtype=np.float32,
    )


def collect_episode(
    bag_path: str,
    storage_id: str,
    episode: EpisodeWindow,
    topics: Topics,
    collect_target_base: bool,
) -> Dict[str, Any]:
    """Read only sensor and command messages inside one episode window."""
    log(
        f"[INFO] Pass 2: collecting output episode {episode.output_idx} "
        f"[{episode.start:.6f}, {episode.end:.6f}]"
    )
    reader = open_reader(bag_path, storage_id)
    types = topic_type_map(reader)

    tracked = {topics.rgb, topics.joint, topics.goto_s}
    if collect_target_base:
        tracked.add(topics.goto_s_target_base)
    classes = message_classes(types, tracked)

    data: Dict[str, Any] = {
        "rgb_t": [],
        "rgb_msg": [],
        "joint_t": [],
        "qpos": [],
        "goto_s_t": [],
        "goto_s": [],
        "target_base_t": [],
        "target_base": [],
    }

    while reader.has_next():
        topic, raw, timestamp_ns = reader.read_next()
        if topic not in tracked:
            continue

        timestamp = ns_to_sec(timestamp_ns)
        if timestamp < episode.start:
            continue
        if timestamp > episode.end:
            # Bags are time ordered, so no later tracked message can be in-window.
            break

        msg = deserialize_message(raw, classes[topic])
        if topic == topics.rgb:
            data["rgb_t"].append(timestamp)
            data["rgb_msg"].append(msg)
        elif topic == topics.joint:
            data["joint_t"].append(timestamp)
            data["qpos"].append(arm_qpos(msg.name, msg.position))
        elif topic == topics.goto_s:
            data["goto_s_t"].append(timestamp)
            data["goto_s"].append(float(msg.data))
        elif topic == topics.goto_s_target_base:
            data["target_base_t"].append(timestamp)
            data["target_base"].append(
                [float(msg.point.x), float(msg.point.y), float(msg.point.z)]
            )

    log(
        f"[INFO] episode {episode.output_idx}: "
        f"rgb={len(data['rgb_t'])}, joints={len(data['joint_t'])}, "
        f"goto_s={len(data['goto_s_t'])}, "
        f"target_base={len(data['target_base_t'])}"
    )

    if not data["rgb_t"]:
        raise RuntimeError(f"Episode {episode.output_idx}: no RGB frames")
    if not data["joint_t"]:
        raise RuntimeError(f"Episode {episode.output_idx}: no joint states")
    if not data["goto_s_t"]:
        raise RuntimeError(
            f"Episode {episode.output_idx}: no executed GOTO_S command; "
            "cancel the episode or inspect command-topic timing"
        )
    return data


def sample_episode(
    data: Dict[str, Any],
    episode: EpisodeWindow,
    fps: float,
) -> Dict[str, np.ndarray]:
    """Causally sample observations and build dense (intercept_s, execute_flag)."""
    command_count = len(data["goto_s_t"])
    if command_count == 0:
        raise RuntimeError(
            f"Episode {episode.output_idx}: expected exactly one GOTO_S command, "
            f"found {command_count}"
        )
    if command_count > 1:
        raise RuntimeError(
            f"Episode {episode.output_idx}: expected exactly one GOTO_S command, "
            f"found {command_count}. Dense episode-target action schema is "
            "incompatible with multiple commands per episode."
        )

    # Starting at the first available RGB and joint message avoids using a future
    # observation for the first grid sample. Ending at the last available messages
    # likewise ensures every sample has causal data from both observation streams.
    effective_start = max(
        episode.start,
        float(data["rgb_t"][0]),
        float(data["joint_t"][0]),
    )
    effective_end = min(
        episode.end,
        float(data["rgb_t"][-1]),
        float(data["joint_t"][-1]),
    )
    if effective_end <= effective_start:
        raise RuntimeError(
            f"Episode {episode.output_idx}: RGB/joint streams have no overlapping interval"
        )

    dt = 1.0 / fps
    grid = np.arange(effective_start, effective_end + 1e-9, dt, dtype=np.float64)
    if grid.size == 0:
        raise RuntimeError(f"Episode {episode.output_idx}: empty sampling grid")

    rgb_times = np.asarray(data["rgb_t"], dtype=np.float64)
    joint_times = np.asarray(data["joint_t"], dtype=np.float64)
    command_times = np.asarray(data["goto_s_t"], dtype=np.float64)
    command_values = np.asarray(data["goto_s"], dtype=np.float32)
    command_time = float(command_times[0])
    target_s = np.float32(command_values[0])
    if not np.isfinite(command_time):
        raise RuntimeError(
            f"Episode {episode.output_idx}: non-finite GOTO_S command timestamp"
        )
    if not np.isfinite(target_s):
        raise RuntimeError(
            f"Episode {episode.output_idx}: non-finite GOTO_S target value"
        )

    rgb_indices = np.searchsorted(rgb_times, grid, side="right") - 1
    joint_indices = np.searchsorted(joint_times, grid, side="right") - 1
    if np.any(rgb_indices < 0) or np.any(joint_indices < 0):
        raise AssertionError("Internal error: non-causal observation index")

    rgb = np.stack(
        [image_msg_to_rgb(data["rgb_msg"][index]) for index in rgb_indices],
        axis=0,
    ).astype(np.uint8, copy=False)
    qpos = np.stack(
        [data["qpos"][index] for index in joint_indices], axis=0
    ).astype(np.float32, copy=False)

    commanded = grid >= command_time
    commanded_indices = np.flatnonzero(commanded)
    if commanded_indices.size == 0:
        raise RuntimeError(
            f"Episode {episode.output_idx}: GOTO_S command at {command_time:.6f} occurs "
            "after the causally usable RGB/joint interval"
        )
    first_grid_command_index = int(commanded_indices[0])
    if first_grid_command_index == 0:
        raise RuntimeError(
            f"Episode {episode.output_idx}: no causally usable pre-command samples "
            "remain in the effective RGB/joint interval"
        )

    action = np.empty((grid.size, 2), dtype=np.float32)
    action[:, 0] = target_s
    action[:, 1] = commanded.astype(np.float32)

    if action.shape != (grid.size, 2):
        raise AssertionError("Internal error: action shape mismatch")
    if not np.isfinite(action).all():
        raise RuntimeError(f"Episode {episode.output_idx}: non-finite dense action")
    execute_values = action[:, 1]
    if not np.all(np.logical_or(execute_values == 0.0, execute_values == 1.0)):
        raise RuntimeError(
            f"Episode {episode.output_idx}: execute_flag must be binary 0/1"
        )
    if not np.all(np.diff(execute_values) >= 0.0):
        raise RuntimeError(
            f"Episode {episode.output_idx}: execute_flag must be monotonically nondecreasing"
        )
    if not np.any(execute_values == 0.0) or not np.any(execute_values == 1.0):
        raise RuntimeError(
            f"Episode {episode.output_idx}: expected both pre-command and post-command samples"
        )

    log(
        f"[INFO] episode {episode.output_idx}: sampled {grid.size} steps at "
        f"{fps:.3f} Hz; first commanded sample index {first_grid_command_index}, "
        f"command timestamp={command_time:.6f}, dense target s={target_s:+.6f} m"
    )

    target_base_t = np.asarray(data["target_base_t"], dtype=np.float64)
    if data["target_base"]:
        target_base = np.asarray(data["target_base"], dtype=np.float32).reshape(-1, 3)
    else:
        target_base = np.empty((0, 3), dtype=np.float32)

    return {
        "timestamps": grid,
        "rgb": rgb,
        "qpos": qpos,
        "action": action,
        "command_timestamps": command_times,
        "command_values": command_values.reshape(-1, 1),
        "target_base_timestamps": target_base_t,
        "target_base_points": target_base,
    }


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
            h5.attrs["episode_index"] = int(episode.output_idx)
            h5.attrs["source_episode_index"] = int(episode.source_idx)
            h5.attrs["episode_start"] = float(episode.start)
            h5.attrs["episode_end"] = float(episode.end)
            h5.attrs["effective_episode_start"] = float(arrays["timestamps"][0])
            h5.attrs["effective_episode_end"] = float(arrays["timestamps"][-1])
            h5.attrs["joint_names"] = np.asarray(
                ARM_JOINT_NAMES, dtype=h5py.string_dtype("utf-8")
            )
            h5.attrs["action_type"] = "dense_intercept_target_with_execute_flag"
            h5.attrs["action_coordinate"] = "captured_interception_line"
            h5.attrs["action_origin"] = "captured_line_center"
            h5.attrs["intercept_s_units"] = "m"
            h5.attrs[
                "intercept_s_semantics"
            ] = "episode target repeated at every effective timestep"
            h5.attrs[
                "execute_flag_semantics"
            ] = "0 before executed GOTO_S timestamp, 1 at/after timestamp"
            h5.attrs["action_layout"] = np.asarray(
                ["intercept_s", "execute_flag"],
                dtype=h5py.string_dtype("utf-8"),
            )
            h5.attrs["action_source_topic"] = topics.goto_s
            h5.attrs["rgb_source_topic"] = topics.rgb
            h5.attrs["joint_source_topic"] = topics.joint
            h5.attrs["sampling_policy"] = "latest_message_at_or_before_grid_time"
            h5.attrs["rgb_temporal_stacking"] = "deferred_to_training_loader"
            h5.attrs["command_count"] = int(arrays["command_timestamps"].size)

            observations = h5.create_group("observations")
            images = observations.create_group("images")
            observations.create_dataset(
                "timestamps", data=arrays["timestamps"], dtype=np.float64
            )
            observations.create_dataset(
                "qpos", data=arrays["qpos"], dtype=np.float32
            )
            images.create_dataset(
                "rgb",
                data=arrays["rgb"],
                dtype=np.uint8,
                chunks=(1, *arrays["rgb"].shape[1:]),
                **compression_kwargs,
            )

            action_ds = h5.create_dataset(
                "action", data=arrays["action"], dtype=np.float32
            )
            action_ds.attrs["columns"] = np.asarray(
                ["intercept_s", "execute_flag"],
                dtype=h5py.string_dtype("utf-8"),
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


def recording_name(bag_path: str) -> str:
    normalized = os.path.normpath(os.path.abspath(os.path.expanduser(bag_path)))
    basename = os.path.basename(normalized)
    return os.path.splitext(basename)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a ROS 2 ball-interception bag to per-episode IL HDF5"
    )
    parser.add_argument("--bag", required=True, help="ROS 2 bag directory")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Parent output directory; a bag-named child directory is created",
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
        "--no_target_base",
        action="store_true",
        help="Do not collect optional executed_goto_s_target_base debug messages",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "lzf", "gzip"),
        default="none",
        help="HDF5 compression for RGB frames",
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--rgb_topic", default=DEFAULT_RGB_TOPIC)
    parser.add_argument("--joint_topic", default=DEFAULT_JOINT_TOPIC)
    parser.add_argument("--episode_topic", default=DEFAULT_EPISODE_TOPIC)
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
    return args


def main() -> None:
    args = parse_args()
    bag_path = os.path.abspath(os.path.expanduser(args.bag))
    output_parent = os.path.abspath(os.path.expanduser(args.out_dir))
    if not os.path.exists(bag_path):
        raise RuntimeError(f"Bag path does not exist: {bag_path}")

    topics = Topics(
        rgb=args.rgb_topic,
        joint=args.joint_topic,
        episode=args.episode_topic,
        goto_s=args.goto_s_topic,
        goto_s_target_base=args.goto_s_target_base_topic,
    )

    log(f"[INFO] bag: {bag_path}")
    log(f"[INFO] storage: {args.storage_id}")
    reader = open_reader(bag_path, args.storage_id)
    types = topic_type_map(reader)
    collect_target_base = validate_topics(
        types, topics, collect_target_base=not args.no_target_base
    )

    episodes = extract_episode_windows(
        bag_path=bag_path,
        storage_id=args.storage_id,
        episode_topic=topics.episode,
        min_duration=args.min_duration,
    )
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    output_dir = os.path.join(output_parent, recording_name(bag_path))
    os.makedirs(output_dir, exist_ok=True)
    log(f"[INFO] output directory: {output_dir}")

    lengths: List[int] = []
    for episode in episodes:
        data = collect_episode(
            bag_path=bag_path,
            storage_id=args.storage_id,
            episode=episode,
            topics=topics,
            collect_target_base=collect_target_base,
        )
        arrays = sample_episode(
            data=data,
            episode=episode,
            fps=args.fps,
        )
        output_path = os.path.join(
            output_dir, f"episode_{episode.output_idx}.hdf5"
        )
        write_episode(
            output_path=output_path,
            arrays=arrays,
            episode=episode,
            topics=topics,
            fps=args.fps,
            compression=args.compression,
            overwrite=args.overwrite,
        )
        lengths.append(int(arrays["timestamps"].size))
        del data
        del arrays
        gc.collect()

    log("[INFO] conversion complete")
    log(f"       episodes: {len(lengths)}")
    log(
        f"       lengths: min={min(lengths)}, "
        f"mean={float(np.mean(lengths)):.2f}, max={max(lengths)}"
    )


if __name__ == "__main__":
    main()
