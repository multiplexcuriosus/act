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
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
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

from sparse_ball import (
    SPARSE_HISTORY_OFFSETS_SEC, construct_sparse_features, policy_period_ns,
    sparse_history_offsets_frames, validate_policy_rate,
)


DEFAULT_RGB_TOPIC = "auto"
DEFAULT_RGB_TOPIC_RAW = "/top_cam/camera/color/image_raw"
DEFAULT_RGB_TOPIC_COMPRESSED = "/top_cam/camera/color/image_raw/compressed"
DEFAULT_RGB_2D_TOPIC = "/ball_tracker2/ball_2d_px"
EVENT_NATIVE_RGB_2D_TOPIC = "/dryrun_ball_tracker2/ball_2d_px"
DEFAULT_EVENT_UPDATE_TOPIC = "/openmv_cam/event_tracker/update"
DEFAULT_RGB_CAMERA_INFO_TOPIC = "/top_cam/camera/color/camera_info"
DEFAULT_EVENT_CAMERA_INFO_TOPIC = "/event_camera/camera_info"
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
CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"
EVENT_TRACKER_UPDATE_TYPE = "openmv_cam/msg/EventTrackerUpdate"

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
EVENT_TRACKER_CACHE_SCHEMA = "act_event_tracker_cache_v1"
EVENT_TRACKER_PROCESSING_SCHEMA = "event_tracker_updates_v2"
OPENMV_REQUIRED_API = (
    "TrackerUpdate", "run_tracker_updates", "align_tracker_updates_to_policy_grid",
)

GIBIBYTE = 1024 ** 3
DEFAULT_MIN_FREE_DISK_GB = 10.0
DEFAULT_DISK_CHECK_PATH = "/"
DEFAULT_DISK_CHECK_INTERVAL_SEC = 5.0


class LowDiskSpaceError(RuntimeError):
    """Raised in the main thread when the disk-space watchdog trips."""


def available_disk_bytes(path: str) -> int:
    """Return bytes available to an unprivileged process, as reported by df."""
    stats = os.statvfs(path)
    return int(stats.f_bavail) * int(stats.f_frsize)


class DiskSpaceMonitor:
    """Continuously interrupt the main thread when free space is too low."""

    def __init__(self, path: str, minimum_gib: float, interval_sec: float):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.minimum_bytes = int(float(minimum_gib) * GIBIBYTE)
        self.minimum_gib = float(minimum_gib)
        self.interval_sec = float(interval_sec)
        self._stop = threading.Event()
        self._failure_message: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._previous_handler = None

    def _check(self) -> None:
        available = available_disk_bytes(self.path)
        if available < self.minimum_bytes:
            raise LowDiskSpaceError(
                f"Free disk space on {self.path} is {available / GIBIBYTE:.2f} GiB, "
                f"below the required {self.minimum_gib:g} GiB; stopping conversion"
            )

    def _handle_signal(self, _signum, _frame) -> None:
        if self._failure_message is not None:
            raise LowDiskSpaceError(self._failure_message)

    def _watch(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                self._check()
            except (LowDiskSpaceError, OSError) as error:
                self._failure_message = str(error)
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    def __enter__(self):
        self._check()
        self._previous_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._handle_signal)
        self._thread = threading.Thread(
            target=self._watch,
            name="disk-space-monitor",
            daemon=True,
        )
        self._thread.start()
        log(
            f"[INFO] disk guard: path={self.path}, "
            f"minimum_free={self.minimum_gib:g} GiB, "
            f"check_interval={self.interval_sec:g}s"
        )
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._previous_handler is not None:
            signal.signal(signal.SIGUSR1, self._previous_handler)
        return False


@dataclass
class Topics:
    rgb: Optional[str]
    joint: str
    episode: str
    current_tcp_s: str
    goto_s: str
    goto_s_target_base: str
    rgb_2d: str = DEFAULT_RGB_2D_TOPIC
    event_update: Optional[str] = None
    rgb_camera_info: Optional[str] = None
    event_camera_info: Optional[str] = None


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


def import_openmv_tracker_api(openmv_cam_root=None):
    """Resolve and validate the reusable OpenMV offline tracker API."""
    roots = []
    if openmv_cam_root:
        roots.append(("--openmv-cam-root", Path(openmv_cam_root)))
    if os.environ.get("OPENMV_CAM_ROOT"):
        roots.append(("OPENMV_CAM_ROOT", Path(os.environ["OPENMV_CAM_ROOT"])))
    roots.append(("sibling repository", REPOSITORY_ROOT.parent / "openmv_cam"))

    errors = []
    for source, root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            errors.append(f"{source}: {root} is not a directory")
            continue
        candidates = (root, root / "src", root.parent)
        import_root = next(
            (candidate for candidate in candidates
             if (candidate / "openmv_cam" / "offline_dataset.py").is_file()),
            None,
        )
        if import_root is None:
            errors.append(f"{source}: {root} does not contain openmv_cam/offline_dataset.py")
            continue
        import_root_text = str(import_root)
        if import_root_text not in sys.path:
            sys.path.insert(0, import_root_text)
        try:
            module = importlib.import_module("openmv_cam.offline_dataset")
            missing = [name for name in OPENMV_REQUIRED_API if not hasattr(module, name)]
            if missing:
                raise AttributeError(f"missing API: {', '.join(missing)}")
            log(f"[INFO] OpenMV tracker package: {root} ({source})")
            return module
        except (ImportError, ModuleNotFoundError, AttributeError) as error:
            errors.append(f"{source}: {root}: {error}")

    try:
        module = importlib.import_module("openmv_cam.offline_dataset")
        missing = [name for name in OPENMV_REQUIRED_API if not hasattr(module, name)]
        if missing:
            raise AttributeError(f"missing API: {', '.join(missing)}")
        log("[INFO] OpenMV tracker package: installed/PYTHONPATH")
        return module
    except (ImportError, ModuleNotFoundError, AttributeError) as error:
        errors.append(f"installed/PYTHONPATH: {error}")
    raise RuntimeError(
        "Raw-event sparse tracking requires openmv_cam.offline_dataset exposing "
        f"{', '.join(OPENMV_REQUIRED_API)}. Set --openmv-cam-root or "
        "OPENMV_CAM_ROOT to the repository root (accepted default: "
        f"{REPOSITORY_ROOT.parent / 'openmv_cam'}). Attempts: {'; '.join(errors)}"
    )


def resolve_event_tracker_config(args, recording_name, openmv_module):
    """Resolve a validated tracker JSON using the documented priority order."""
    candidates = []
    if args.event_tracker_config:
        candidates.append(("--event-tracker-config", Path(args.event_tracker_config)))
    elif os.environ.get("OPENMV_EVENT_TRACKER_CONFIG"):
        candidates.append((
            "OPENMV_EVENT_TRACKER_CONFIG",
            Path(os.environ["OPENMV_EVENT_TRACKER_CONFIG"]),
        ))
    if args.rec_dir:
        candidates.append((
            "recording-local",
            Path(args.rec_dir) / f"{recording_name}_event_tracker_config.json",
        ))
    module_root = Path(openmv_module.__file__).resolve().parents[1]
    candidates.append((
        "OpenMV repository default", module_root / "config" / "offline_tracker_example.json",
    ))
    for source, path in candidates:
        path = path.expanduser().resolve()
        if not path.is_file():
            if source in ("--event-tracker-config", "OPENMV_EVENT_TRACKER_CONFIG"):
                raise RuntimeError(f"{source} tracker configuration does not exist: {path}")
            continue
        with path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        if not isinstance(config, dict):
            raise RuntimeError(f"Tracker configuration must be a JSON object: {path}")
        pre_roll_ms = float(config.pop("pre_roll_ms", 0.0))
        try:
            openmv_module.tracker_factory(config)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid OpenMV tracker configuration {path}: {error}") from error
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        log(f"[INFO] event tracker configuration: {path} ({source})")
        return config, config_json, pre_roll_ms, str(path)
    raise RuntimeError(
        "No validated event tracker JSON was found. Provide --event-tracker-config, "
        "OPENMV_EVENT_TRACKER_CONFIG, a recording-local config, or the OpenMV "
        "config/offline_tracker_example.json default."
    )


def load_event_tracker_cache(path, raw_events_h5, config_hash, episode_names,
                             TrackerUpdate):
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        cache = json.load(stream)
    if cache.get("schema_version") != EVENT_TRACKER_CACHE_SCHEMA:
        raise RuntimeError("event tracker cache schema version mismatch")
    if Path(cache.get("raw_events_h5", "")).resolve() != Path(raw_events_h5).resolve():
        raise RuntimeError("event tracker cache raw-event path mismatch")
    if cache.get("tracker_config_hash") != config_hash:
        raise RuntimeError("event tracker cache configuration hash mismatch")
    rows = cache.get("episodes")
    if not isinstance(rows, dict) or any(name not in rows for name in episode_names):
        raise RuntimeError("event tracker cache is missing selected episodes")
    try:
        return {name: [TrackerUpdate(**row) for row in rows[name]]
                for name in episode_names}
    except (TypeError, KeyError) as error:
        raise RuntimeError(f"event tracker cache has invalid TrackerUpdate fields: {error}") from error


def save_event_tracker_cache(path, raw_events_h5, config_hash, updates_by_episode):
    cache = {
        "schema_version": EVENT_TRACKER_CACHE_SCHEMA,
        "raw_events_h5": str(Path(raw_events_h5).resolve()),
        "tracker_config_hash": config_hash,
        "episodes": {name: [asdict(update) for update in updates]
                     for name, updates in updates_by_episode.items()},
    }
    target = Path(path).expanduser().resolve()
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(cache, stream, sort_keys=True)
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


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


def resolve_topic_profile(types: Dict[str, str], requested: str) -> str:
    """Select a reproducible sparse topic layout from exact topic names."""
    legacy = DEFAULT_RGB_2D_TOPIC in types
    native = EVENT_NATIVE_RGB_2D_TOPIC in types
    if requested != "auto":
        return requested
    if legacy and native:
        raise RuntimeError(
            "Topic profile is ambiguous: both legacy and event-native RGB sparse "
            "topics are present; pass --topic-profile or --rgb-2d-topic explicitly"
        )
    if native:
        return "event_native"
    return "legacy_rgb_primary"


def validate_topics(
    types: Dict[str, str],
    topics: Topics,
    collect_goto_s_debug: bool,
    collect_target_base_debug: bool,
    collect_rgb_2d: bool = True,
    require_rgb: bool = True,
) -> Tuple[bool, bool]:
    required = {topics.joint, topics.episode, topics.current_tcp_s}
    if require_rgb:
        required.add(topics.rgb)
    missing = sorted(required - set(types))
    if missing:
        raise RuntimeError(f"Missing required topics: {missing}")

    rgb_type = types.get(topics.rgb) if topics.rgb else None
    if require_rgb and rgb_type not in (RAW_IMAGE_TYPE, COMPRESSED_IMAGE_TYPE):
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

    if require_rgb:
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
    tracked = [topic for topic in (topics.rgb, topics.joint, topics.current_tcp_s)
               if topic]
    if collect_goto_s_debug:
        tracked.append(topics.goto_s)
    if collect_target_base_debug:
        tracked.append(topics.goto_s_target_base)
    if collect_rgb_2d:
        tracked.append(topics.rgb_2d)
    for topic in (topics.event_update, topics.rgb_camera_info,
                  topics.event_camera_info):
        if topic:
            tracked.append(topic)
    return tracked

def create_episode_buffer() -> Dict[str, Any]:
    return {
        "rgb_t": [],
        "rgb_msg": [],
        "rgb_2d_t": [],
        "rgb_2d_source_t": [],
        "rgb_2d_px": [],
        "event_update_t": [],
        "event_update_msg": [],
        "rgb_size": None,
        "event_size": None,
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
    elif topic == topics.event_update:
        data["event_update_t"].append(timestamp)
        data["event_update_msg"].append(msg)
    elif topic == topics.rgb_camera_info:
        size = (int(msg.width), int(msg.height))
        if data["rgb_size"] not in (None, size):
            raise RuntimeError("RGB CameraInfo dimensions changed within an episode")
        data["rgb_size"] = size
    elif topic == topics.event_camera_info:
        size = (int(msg.width), int(msg.height))
        if data["event_size"] not in (None, size):
            raise RuntimeError("Event CameraInfo dimensions changed within an episode")
        data["event_size"] = size
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
    event_tracker_updates=None,
    event_tracker_aligner=None,
    event_tracker_metadata: Optional[Dict[str, Any]] = None,
    sparse_source: Optional[str] = None,
    max_observation_age_sec: float = 0.10,
    sparse_only: bool = False,
    rgb_sparse_size: Optional[Tuple[int, int]] = None,
    event_sparse_size: Optional[Tuple[int, int]] = None,
    topic_profile: str = "legacy_rgb_primary",
    recorded_event_rows=None,
) -> int:
    collect_sec = max(0.0, time.perf_counter() - collect_started_wall)

    log(
        f"[INFO] episode {episode.output_idx}: "
        f"rgb={len(data['rgb_t'])}, joints={len(data['joint_t'])}, "
        f"current_tcp_s={len(data['current_tcp_s_t'])}, "
        f"goto_s_debug={len(data['goto_s_t'])}, "
        f"target_base_debug={len(data['target_base_t'])}"
    )

    if not sparse_only and not data["rgb_t"]:
        raise RuntimeError(f"Episode {episode.output_idx}: no RGB frames")
    if not data["joint_t"]:
        raise RuntimeError(f"Episode {episode.output_idx}: no joint states")
    if not data["current_tcp_s_t"]:
        raise RuntimeError(
            f"Episode {episode.output_idx}: no current_tcp_s measurements"
        )
    rgb_sparse_size = rgb_sparse_size or data.get("rgb_size")
    event_sparse_size = event_sparse_size or data.get("event_size")
    if rgb_2d_enabled and rgb_sparse_size is None and sparse_only:
        raise RuntimeError("Sparse-only RGB conversion requires RGB CameraInfo or --rgb-width/--rgb-height")
    if (data.get("event_update_msg") or recorded_event_rows) and event_sparse_size is None:
        raise RuntimeError("Recorded event conversion requires event CameraInfo or --event-width/--event-height")

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
        event_tracker_updates=event_tracker_updates,
        event_tracker_aligner=event_tracker_aligner,
        sparse_source=sparse_source,
        max_observation_age_sec=max_observation_age_sec,
        event_sparse_size=(
            event_sparse_size or (
                int((event_tracker_metadata or {}).get("sensor_width", 320)),
                int((event_tracker_metadata or {}).get("sensor_height", 320)),
            )
        ),
        sparse_only=sparse_only,
        rgb_sparse_size=rgb_sparse_size,
        recorded_event_rows=(recorded_event_rows if recorded_event_rows is not None else recorded_event_update_rows(
            data["event_update_msg"], data["event_update_t"]
        ) if data.get("event_update_msg") else None),
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
        event_tracker_metadata=event_tracker_metadata,
        sparse_source=sparse_source,
        max_observation_age_sec=max_observation_age_sec,
        rgb_sparse_size=rgb_sparse_size,
        event_sparse_size=event_sparse_size,
        topic_profile=topic_profile,
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


def recorded_event_update_rows(messages, receipt_times):
    rows = []
    for msg, receipt in zip(messages, receipt_times):
        row = {
            "availability_timestamp_ns": int(msg.availability_timestamp_ns),
            "bag_availability_timestamp": float(receipt),
            "x_px": float(msg.x_px), "y_px": float(msg.y_px),
            "valid": bool(msg.valid),
        }
        for name in ("vx_px_s", "vy_px_s", "velocity_valid", "confidence",
                     "sensor_window_start_us", "sensor_window_end_us",
                     "window_event_count", "candidate_count", "rejection_reason"):
            if hasattr(msg, name):
                row[name] = getattr(msg, name)
        for name in ("source_packet_id", "tracker_update_id"):
            valid_name = f"{name}_valid"
            row[valid_name] = bool(getattr(msg, valid_name, False))
            if row[valid_name]:
                row[name] = int(getattr(msg, name))
        rows.append(row)
    return rows


def read_recorded_event_updates(bag_path, topic, episodes):
    """Read an MCAP-embedded custom schema without requiring a sourced ROS package."""
    from rosbags.highlevel import AnyReader
    result = {f"episode_{episode.output_idx}": [] for episode in episodes}
    with AnyReader([Path(bag_path)]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise RuntimeError(f"Recorded event update topic is absent: {topic}")
        episode_index = 0
        for connection, timestamp_ns, raw in reader.messages(connections=connections):
            receipt = timestamp_ns * 1e-9
            while episode_index < len(episodes) and receipt > episodes[episode_index].end:
                episode_index += 1
            if episode_index >= len(episodes):
                break
            episode = episodes[episode_index]
            if receipt >= episode.start:
                msg = reader.deserialize(raw, connection.msgtype)
                result[f"episode_{episode.output_idx}"].extend(
                    recorded_event_update_rows([msg], [receipt]))
    return result


def align_recorded_event_updates(rows, grid_ns, max_age_sec):
    receipt_ns = np.asarray(
        [round(row["bag_availability_timestamp"] * 1e9) for row in rows],
        dtype=np.int64,
    )
    indices = np.searchsorted(receipt_ns, grid_ns, side="right") - 1
    points = np.zeros((len(grid_ns), 2), dtype=np.float32)
    valid = np.zeros(len(grid_ns), dtype=np.uint8)
    source = np.full(len(grid_ns), np.nan, dtype=np.float64)
    available = np.full(len(grid_ns), np.nan, dtype=np.float64)
    for out, index in zip(np.flatnonzero(indices >= 0), indices[indices >= 0]):
        row = rows[int(index)]
        source[out] = row["availability_timestamp_ns"] * 1e-9
        available[out] = row["bag_availability_timestamp"]
        age = grid_ns[out] * 1e-9 - source[out]
        if row["valid"] and 0 <= age <= max_age_sec:
            points[out] = row["x_px"], row["y_px"]
            valid[out] = 1
    return {"event_2d_px": points, "event_valid": valid,
            "event_source_timestamps": source,
            "event_availability_timestamps": available}


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
    event_tracker_updates=None,
    event_tracker_aligner=None,
    sparse_source: Optional[str] = None,
    max_observation_age_sec: float = 0.10,
    event_sparse_size: Tuple[int, int] = (320, 320),
    sparse_only: bool = False,
    rgb_sparse_size: Optional[Tuple[int, int]] = None,
    recorded_event_rows=None,
) -> Dict[str, np.ndarray]:
    """Causally sample observations and measured current_tcp_s action."""
    rgb_times = np.asarray(data["rgb_t"], dtype=np.float64)
    joint_times = np.asarray(data["joint_t"], dtype=np.float64)
    tcp_s_times = np.asarray(data["current_tcp_s_t"], dtype=np.float64)
    tcp_s_values = np.asarray(data["current_tcp_s"], dtype=np.float32)

    if not sparse_only:
        _validate_monotonic_non_decreasing("RGB", rgb_times)
    _validate_monotonic_non_decreasing("joint", joint_times)
    _validate_monotonic_non_decreasing("current_tcp_s", tcp_s_times)

    starts = [episode.start, float(joint_times[0]), float(tcp_s_times[0])]
    ends = [episode.end, float(joint_times[-1]), float(tcp_s_times[-1])]
    if not sparse_only:
        starts.append(float(rgb_times[0])); ends.append(float(rgb_times[-1]))
    effective_start, effective_end = max(starts), min(ends)
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

    rgb_indices = (np.searchsorted(rgb_times, grid, side="right") - 1
                   if not sparse_only else None)
    joint_indices = np.searchsorted(joint_times, grid, side="right") - 1
    tcp_s_indices = np.searchsorted(tcp_s_times, grid, side="right") - 1
    if ((rgb_indices is not None and np.any(rgb_indices < 0))
            or np.any(joint_indices < 0) or np.any(tcp_s_indices < 0)):
        raise AssertionError("Internal error: non-causal source index")

    rgb = np.stack(
        [image_msg_to_rgb(data["rgb_msg"][index]) for index in rgb_indices],
        axis=0,
    ).astype(np.uint8, copy=False) if not sparse_only else None

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
            widths, heights = (rgb_sparse_size if rgb_sparse_size is not None
                               else (rgb.shape[2], rgb.shape[1]))
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
    rgb_source_age_sec = np.full((grid.size,), np.nan, dtype=np.float64)
    rgb_has_source = np.isfinite(rgb_source_timestamps)
    rgb_source_age_sec[rgb_has_source] = (
        grid[rgb_has_source] - rgb_source_timestamps[rgb_has_source]
    )
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
    if ((rgb is not None and rgb.shape[0] != grid.size)
            or qpos.shape[0] != grid.size):
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
        "timestamps_ns": grid_ns,
        "qpos": qpos,
        "action": action,
        "action_source_timestamps": action_source_timestamps,
        "action_source_age_sec": action_source_age_sec,
        "command_timestamps": command_timestamps,
        "command_values": command_values,
        "target_base_timestamps": target_base_t,
        "target_base_points": target_base,
    }
    if rgb is not None:
        arrays["rgb"] = rgb

    if rgb_2d_enabled:
        # Legacy in-memory aliases retained for callers; the writer uses the
        # explicit source-qualified keys below.
        arrays["rgb_2d_px"] = rgb_2d_px
        arrays["rgb_valid"] = rgb_valid
        arrays["rgb_source_timestamps"] = rgb_source_timestamps
        arrays["sparse_tracking/rgb_2d_px"] = rgb_2d_px
        arrays["sparse_tracking/rgb_valid"] = rgb_valid
        arrays["sparse_tracking/rgb_source_timestamps"] = rgb_source_timestamps
        arrays["sparse_tracking/rgb_source_age_sec"] = rgb_source_age_sec
        raw_rgb_timestamps = np.asarray(data.get("rgb_2d_source_t", []), dtype=np.float64)
        raw_rgb_points = np.asarray(data.get("rgb_2d_px", []), dtype=np.float32).reshape(-1, 2)
        if raw_rgb_timestamps.shape != (len(raw_rgb_points),):
            raise RuntimeError("raw RGB sparse timestamp/coordinate count mismatch")
        if raw_rgb_timestamps.size:
            _validate_monotonic_non_decreasing("raw RGB sparse", raw_rgb_timestamps)
        width, height = (rgb_sparse_size if rgb_sparse_size is not None
                         else (rgb.shape[2], rgb.shape[1]))
        raw_rgb_valid = (np.isfinite(raw_rgb_points).all(axis=1)
                         & (raw_rgb_points[:, 0] >= 0) & (raw_rgb_points[:, 0] < width)
                         & (raw_rgb_points[:, 1] >= 0) & (raw_rgb_points[:, 1] < height))
        arrays["sparse_tracking/raw_rgb_timestamps"] = raw_rgb_timestamps
        arrays["sparse_tracking/raw_rgb_2d_px"] = raw_rgb_points
        arrays["sparse_tracking/raw_rgb_valid"] = raw_rgb_valid.astype(np.uint8)
        arrays["sparse_tracking/raw_rgb_availability_timestamps"] = np.asarray(
            data.get("rgb_2d_t", []), dtype=np.float64
        )

    if event is not None:
        arrays["event"] = event
        arrays["event_source_timestamps"] = event_source_timestamps
        arrays["event_source_age_sec"] = event_source_age_sec
        arrays["event_count_per_channel"] = event_count_per_channel

    if event_tracker_updates is not None:
        if event_tracker_aligner is None:
            raise RuntimeError("event tracker updates require the OpenMV alignment API")
        event_tracking = event_tracker_aligner(
            event_tracker_updates, grid_ns, max_observation_age_sec
        )
        # OpenMV retains the last valid point for diagnostics after it expires;
        # ACT's model-facing coordinate contract requires invalid rows to be zero.
        event_tracking["event_2d_px"] = np.asarray(
            event_tracking["event_2d_px"], dtype=np.float32
        ).copy()
        event_tracking["event_2d_px"][
            np.asarray(event_tracking["event_valid"]) == 0
        ] = 0.0
        arrays.update({f"sparse_tracking/{key}": value
                       for key, value in event_tracking.items()})
        arrays["event_tracker_updates"] = list(event_tracker_updates)
        raw_event_timestamps = np.asarray(
            [update.available_ros_t_ns for update in event_tracker_updates],
            dtype=np.float64,
        ) * 1e-9
        if raw_event_timestamps.size:
            _validate_monotonic_non_decreasing("raw event sparse", raw_event_timestamps)
        arrays["sparse_tracking/raw_event_timestamps"] = raw_event_timestamps
        arrays["sparse_tracking/raw_event_2d_px"] = np.asarray(
            [[update.x_px, update.y_px] for update in event_tracker_updates],
            dtype=np.float32,
        ).reshape(-1, 2)
        arrays["sparse_tracking/raw_event_valid"] = np.asarray(
            [update.valid for update in event_tracker_updates], dtype=np.uint8
        )
        arrays["sparse_tracking/raw_event_availability_timestamps"] = raw_event_timestamps

    if recorded_event_rows is not None:
        event_tracking = align_recorded_event_updates(
            recorded_event_rows, grid_ns, max_observation_age_sec
        )
        arrays.update({f"sparse_tracking/{key}": value
                       for key, value in event_tracking.items()})
        arrays["recorded_event_update_rows"] = recorded_event_rows
        arrays["sparse_tracking/raw_event_timestamps"] = np.asarray(
            [row["availability_timestamp_ns"] for row in recorded_event_rows],
            dtype=np.float64) * 1e-9
        arrays["sparse_tracking/raw_event_availability_timestamps"] = np.asarray(
            [row["bag_availability_timestamp"] for row in recorded_event_rows],
            dtype=np.float64)
        arrays["sparse_tracking/raw_event_2d_px"] = np.asarray(
            [[row["x_px"], row["y_px"]] for row in recorded_event_rows],
            dtype=np.float32).reshape(-1, 2)
        arrays["sparse_tracking/raw_event_valid"] = np.asarray(
            [row["valid"] for row in recorded_event_rows], dtype=np.uint8)

    if sparse_source is not None:
        prefix = f"sparse_tracking/{sparse_source}"
        required = (f"{prefix}_2d_px", f"{prefix}_valid",
                    f"{prefix}_source_timestamps")
        missing = [name for name in required if name not in arrays]
        if missing:
            raise RuntimeError(
                f"sparse source {sparse_source!r} has no aligned detections: {missing}"
            )
        width, height = ((rgb_sparse_size or (int(rgb.shape[2]), int(rgb.shape[1])))
                         if sparse_source == "rgb" else event_sparse_size)
        arrays["sparse_ball"] = construct_sparse_features(
            grid, arrays[required[0]], arrays[required[1]], arrays[required[2]],
            width, height, max_observation_age_sec,
        )
        arrays["sparse_image_size"] = (int(width), int(height))

    return arrays


def dataset_kwargs(compression: str) -> Dict[str, Any]:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 1}
    return {"compression": "lzf"}


def validate_policy_grid_metadata(h5, requested_fps: float, path: str = "output") -> float:
    """Validate that converter metadata and timestamps describe the requested grid."""
    requested_rate = int(requested_fps)
    if float(requested_fps) != requested_rate or requested_rate <= 0:
        raise RuntimeError(f"{path}: requested FPS must be a positive integer")
    metadata_rate = int(h5.attrs.get("policy_rate_hz", requested_rate))
    timestamps = np.asarray(h5["/observations/timestamps"][()], dtype=np.float64)
    if timestamps.size < 2:
        raise RuntimeError(f"{path}: need at least two policy timestamps to validate cadence")
    median_period = float(np.median(np.diff(timestamps)))
    expected_period = 1.0 / requested_rate
    metadata_period = float(h5.attrs.get("policy_period_ns", 0)) * 1e-9
    if metadata_rate != requested_rate or not np.isclose(
        metadata_period, expected_period, rtol=0.0, atol=1e-9
    ) or not np.isclose(median_period, expected_period, rtol=0.02, atol=1e-6):
        raise RuntimeError(
            f"{path}: output policy-grid mismatch: requested={requested_rate} Hz "
            f"({expected_period:.9f}s), metadata={metadata_rate} Hz/"
            f"{metadata_period:.9f}s, timestamp median={median_period:.9f}s"
        )
    return median_period


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
    event_tracker_metadata: Optional[Dict[str, Any]] = None,
    sparse_source: Optional[str] = None,
    max_observation_age_sec: float = 0.10,
    rgb_sparse_size: Optional[Tuple[int, int]] = None,
    event_sparse_size: Optional[Tuple[int, int]] = None,
    topic_profile: str = "legacy_rgb_primary",
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
    has_sparse_tracking = any(
        key.startswith("sparse_tracking/") for key in arrays
    )
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
            if has_sparse_tracking and int(fps) in (30, 60):
                sparse_offsets = sparse_history_offsets_frames(fps)
                h5.attrs["chunk_size"] = int(fps)
                h5.attrs["qpos_history_frames"] = len(sparse_offsets)
                h5.attrs["qpos_history_offsets"] = np.asarray(
                    sparse_offsets, dtype=np.int32
                )
                h5.attrs["sparse_history_offsets_sec"] = np.asarray(
                    SPARSE_HISTORY_OFFSETS_SEC, dtype=np.float64
                )
                h5.attrs["sparse_history_length"] = len(sparse_offsets)
                h5.attrs["sparse_history_offsets_frames"] = np.asarray(
                    sparse_offsets, dtype=np.int32
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
            h5.attrs["rgb_source_topic"] = topics.rgb or ""
            h5.attrs["topic_profile"] = topic_profile
            h5.attrs["dense_rgb_present"] = "rgb" in arrays
            h5.attrs["dense_event_present"] = "event" in arrays
            h5.attrs["joint_source_topic"] = topics.joint
            h5.attrs["rgb_temporal_stacking"] = "deferred_to_training_loader"
            h5.attrs["command_count"] = int(arrays["command_timestamps"].size)
            h5.attrs["sparse_tracking_sources"] = np.asarray(
                (["rgb", "event"] if event_tracker_metadata else ["rgb"]),
                dtype=h5py.string_dtype("utf-8"),
            )
            if event_tracker_metadata:
                h5.attrs["event_tracker_schema_version"] = (
                    event_tracker_metadata["schema_version"]
                )
                if "tracker_config_hash" in event_tracker_metadata:
                    h5.attrs["event_tracker_config_hash"] = (
                        event_tracker_metadata["tracker_config_hash"]
                    )

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
            images = None
            observations.create_dataset(
                "timestamps", data=arrays["timestamps"], dtype=np.float64
            )
            observations.create_dataset(
                "timestamps_ns", data=arrays["timestamps_ns"], dtype=np.int64
            )
            observations.create_dataset(
                "qpos", data=arrays["qpos"], dtype=np.float32
            )
            tracking_keys = sorted(
                key for key in arrays if key.startswith("sparse_tracking/")
            )
            if tracking_keys:
                sparse = observations.create_group("sparse_tracking")
                for key in tracking_keys:
                    name = key.split("/", 1)[1]
                    values = arrays[key]
                    if name.endswith("rejection_reason"):
                        values = np.asarray(
                            values, dtype=h5py.string_dtype("utf-8", length=256)
                        )
                    sparse.create_dataset(name, data=values)
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
                sparse.attrs["raw_rgb_timestamp_domain"] = "PointStamped ROS header timestamp"
                sparse.attrs["raw_event_timestamp_domain"] = "tracker packet availability ROS timestamp"
                sparse.attrs["raw_stream_order"] = "monotonic_oldest_to_newest"
                rgb_width, rgb_height = (rgb_sparse_size or
                    (int(arrays["rgb"].shape[2]), int(arrays["rgb"].shape[1])))
                sparse.attrs["rgb_width_px"] = int(rgb_width)
                sparse.attrs["rgb_height_px"] = int(rgb_height)
                sparse.attrs["rgb_message_type"] = POINT_STAMPED_TYPE
                sparse.attrs["raw_rgb_availability_timestamp_domain"] = "MCAP bag record time"
                sparse.attrs["event_source_topic"] = (
                    topics.event_update or (event_tracker_metadata or {}).get(
                        "source_topic", "offline_raw_event_sidecar"))
                sparse.attrs["event_message_type"] = (
                    EVENT_TRACKER_UPDATE_TYPE if topics.event_update else "offline_tracker_update"
                )
                sparse.attrs["raw_event_availability_timestamp_domain"] = "MCAP bag record time"
                if event_tracker_metadata:
                    sparse.attrs["event_source_timestamp_domain"] = (
                        "tracker_packet_availability_ros_t_ns"
                    )
                    sparse.attrs["event_valid_semantics"] = (
                        "latest causal valid detection held until max_observation_age_sec"
                    )
                    sparse.attrs["event_width_px"] = int(
                        event_tracker_metadata["sensor_width"]
                    )
                    sparse.attrs["event_height_px"] = int(
                        event_tracker_metadata["sensor_height"]
                    )
                    sparse.attrs["sources"] = np.asarray(
                        ["rgb", "event"], dtype=h5py.string_dtype("utf-8")
                    )
                elif event_sparse_size is not None:
                    sparse.attrs["event_width_px"] = int(event_sparse_size[0])
                    sparse.attrs["event_height_px"] = int(event_sparse_size[1])
                sources = [name for name in ("rgb", "event")
                           if f"raw_{name}_timestamps" in sparse]
                sparse.attrs["sources"] = np.asarray(
                    sources, dtype=h5py.string_dtype("utf-8"))
            if "rgb" in arrays or has_event_sidecar:
                images = observations.create_group("images")
            if "rgb" in arrays:
                images.create_dataset(
                    "rgb", data=arrays["rgb"], dtype=np.uint8,
                    chunks=(1, *arrays["rgb"].shape[1:]), **compression_kwargs)
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

            if "event_tracker_updates" in arrays:
                processing = h5.require_group("processing").create_group(
                    "event_tracker"
                )
                updates = arrays["event_tracker_updates"]
                field_names = tuple(asdict(updates[0]).keys()) if updates else (
                    "available_ros_t_ns", "packet_id", "sensor_window_start_us",
                    "sensor_window_end_us", "x_px", "y_px", "vx_px_s",
                    "vy_px_s", "speed_px_s", "confidence", "valid",
                    "velocity_valid", "window_event_count", "candidate_count",
                    "blob_area_px", "blob_event_count", "blob_width_px",
                    "blob_height_px", "circularity", "rejection_reason",
                )
                excluded = {"episode_name", "episode_index", "source_episode_index"}
                for field in field_names:
                    if field in excluded:
                        continue
                    values = [getattr(update, field) for update in updates]
                    if field == "rejection_reason":
                        values = np.asarray(
                            values, dtype=h5py.string_dtype("utf-8", length=256)
                        )
                    else:
                        values = np.asarray(values)
                    processing.create_dataset(field, data=values)
                for key, value in (event_tracker_metadata or {}).items():
                    processing.attrs[key] = value

            if "recorded_event_update_rows" in arrays:
                processing = h5.require_group("processing").create_group("event_tracker")
                rows = arrays["recorded_event_update_rows"]
                fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
                for field in fields:
                    values = [row.get(field) for row in rows]
                    if field == "rejection_reason":
                        values = np.asarray(values, dtype=h5py.string_dtype("utf-8", length=256))
                    elif any(value is None for value in values):
                        continue
                    processing.create_dataset(field, data=np.asarray(values))
                processing.attrs["origin"] = "recorded_ros_topic"
                processing.attrs["source_topic"] = (
                    topics.event_update or (event_tracker_metadata or {}).get("source_topic", ""))
                processing.attrs["message_type"] = EVENT_TRACKER_UPDATE_TYPE
                processing.attrs["offline_only_fields_present"] = False

            if "sparse_ball" in arrays:
                observations.create_dataset(
                    "sparse_ball", data=arrays["sparse_ball"], dtype=np.float32
                )
                width, height = arrays["sparse_image_size"]
                h5.attrs["sparse_source"] = sparse_source
                h5.attrs["sparse_feature_names"] = np.asarray(
                    ["u_norm", "v_norm", "valid", "observation_age_sec"],
                    dtype=h5py.string_dtype("utf-8"),
                )
                h5.attrs["sparse_image_width"] = int(width)
                h5.attrs["sparse_image_height"] = int(height)
                h5.attrs["sparse_max_observation_age_sec"] = float(
                    max_observation_age_sec
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

            median_period = validate_policy_grid_metadata(h5, fps, temporary_path)

        os.replace(temporary_path, output_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise

    log(
        f"[INFO] wrote {output_path}: policy_rate_hz={int(fps)}, "
        f"median_policy_period_sec={median_period:.9f}"
    )


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
    bag_path, auto_raw_events_h5, _ = resolve_recording_dir(
        rec_dir,
        allow_missing_raw_events=True,
        logger=log,
    )
    if args.raw_events_h5 is not None:
        return bag_path, os.path.abspath(os.path.expanduser(args.raw_events_h5))

    if auto_raw_events_h5 is not None:
        return bag_path, auto_raw_events_h5
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
    parser.add_argument("--fps", type=float, choices=(30.0, 60.0), default=30.0)
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
        "--min-free-disk-gb",
        type=float,
        default=DEFAULT_MIN_FREE_DISK_GB,
        help="Stop when free space falls below this many GiB (default: 10)",
    )
    parser.add_argument(
        "--disk-check-path",
        default=DEFAULT_DISK_CHECK_PATH,
        help="Filesystem path monitored for free space (default: /)",
    )
    parser.add_argument(
        "--disk-check-interval-sec",
        type=float,
        default=DEFAULT_DISK_CHECK_INTERVAL_SEC,
        help="Seconds between disk-space checks (default: 5)",
    )

    parser.add_argument(
        "--raw_events_h5",
        default=None,
        help=(
            "Optional raw-event sidecar HDF5. When provided (or auto-resolved via "
            "--rec_dir), event tensors are rendered from raw packets."
        ),
    )
    parser.add_argument("--event-tracker-config", "--event_tracker_config")
    parser.add_argument("--sparse-only", "--sparse_only", action="store_true")
    parser.add_argument("--topic-profile", "--topic_profile",
                        choices=("auto", "legacy_rgb_primary", "event_native"),
                        default="auto")
    parser.add_argument("--event-tracker-source", "--event_tracker_source",
                        choices=("auto", "recorded", "offline"), default="auto")
    parser.add_argument("--event-update-topic", "--event_update_topic",
                        default=DEFAULT_EVENT_UPDATE_TOPIC)
    parser.add_argument("--rgb-camera-info-topic", "--rgb_camera_info_topic",
                        default=DEFAULT_RGB_CAMERA_INFO_TOPIC)
    parser.add_argument("--event-camera-info-topic", "--event_camera_info_topic",
                        default=DEFAULT_EVENT_CAMERA_INFO_TOPIC)
    parser.add_argument("--rgb-width", type=int)
    parser.add_argument("--rgb-height", type=int)
    parser.add_argument("--event-width", type=int)
    parser.add_argument("--event-height", type=int)
    parser.add_argument("--openmv-cam-root", "--openmv_cam_root")
    parser.add_argument("--sparse-source", "--sparse_source",
                        choices=("rgb", "event"), default=None)
    parser.add_argument("--max-observation-age-sec", "--max_observation_age_sec",
                        type=float, default=0.10)
    parser.add_argument("--save-event-tracker-cache", "--save_event_tracker_cache")
    parser.add_argument("--reuse-event-tracker-cache", "--reuse_event_tracker_cache")
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
    if not np.isfinite(args.min_free_disk_gb) or args.min_free_disk_gb <= 0.0:
        parser.error("--min-free-disk-gb must be finite and positive")
    if (
        not np.isfinite(args.disk_check_interval_sec)
        or args.disk_check_interval_sec <= 0.0
    ):
        parser.error("--disk-check-interval-sec must be finite and positive")
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
    if args.max_observation_age_sec <= 0.0:
        parser.error("--max-observation-age-sec must be positive")
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


def run_conversion(args: argparse.Namespace, conversion_start_wall: float) -> None:
    selected_rate = validate_policy_rate(args.fps)
    log(
        f"[INFO] selected policy grid: fps={selected_rate}, "
        f"policy_period_sec={1.0 / selected_rate:.9f}, "
        f"policy_period_ns={policy_period_ns(selected_rate)}"
    )
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
        ("event_tracker_config", None),
        ("openmv_cam_root", None),
        ("sparse_source", None),
        ("max_observation_age_sec", 0.10),
        ("save_event_tracker_cache", None),
        ("reuse_event_tracker_cache", None),
        ("min_free_disk_gb", DEFAULT_MIN_FREE_DISK_GB),
        ("disk_check_path", DEFAULT_DISK_CHECK_PATH),
        ("disk_check_interval_sec", DEFAULT_DISK_CHECK_INTERVAL_SEC),
        ("sparse_only", False), ("topic_profile", "auto"),
        ("event_tracker_source", "auto"),
        ("event_update_topic", DEFAULT_EVENT_UPDATE_TOPIC),
        ("rgb_camera_info_topic", DEFAULT_RGB_CAMERA_INFO_TOPIC),
        ("event_camera_info_topic", DEFAULT_EVENT_CAMERA_INFO_TOPIC),
        ("rgb_width", None), ("rgb_height", None),
        ("event_width", None), ("event_height", None),
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
        log("[WARNING] raw events H5 unavailable: sparse event fields will not be generated")
    log(f"[INFO] storage: {args.storage_id}")

    marker_pass_start_wall = time.perf_counter()
    marker_reader = open_reader(bag_path, args.storage_id)
    types = topic_type_map(marker_reader)
    profile = resolve_topic_profile(types, args.topic_profile)
    selected_rgb_topic = (None if args.sparse_only
                          else resolve_rgb_topic(types, args.rgb_topic))
    rgb_2d_topic = args.rgb_2d_topic
    if args.rgb_2d_topic == DEFAULT_RGB_2D_TOPIC and profile == "event_native":
        rgb_2d_topic = EVENT_NATIVE_RGB_2D_TOPIC
    recorded_available = types.get(args.event_update_topic) == EVENT_TRACKER_UPDATE_TYPE
    use_recorded = (args.event_tracker_source == "recorded" or
                    (args.event_tracker_source == "auto" and recorded_available))
    if args.event_tracker_source == "recorded" and not recorded_available:
        raise RuntimeError(f"Recorded event tracker requires {args.event_update_topic} :: {EVENT_TRACKER_UPDATE_TYPE}")
    if args.sparse_only and use_recorded:
        raw_events_h5 = None

    topics = Topics(
        rgb=selected_rgb_topic,
        rgb_2d=rgb_2d_topic,
        joint=args.joint_topic,
        episode=args.episode_topic,
        current_tcp_s=args.current_tcp_s_topic,
        goto_s=args.goto_s_topic,
        goto_s_target_base=args.goto_s_target_base_topic,
        # Custom message schemas are decoded from the MCAP schema by rosbags;
        # this avoids requiring the recording machine's ROS interface package.
        event_update=None,
        rgb_camera_info=(args.rgb_camera_info_topic if args.sparse_only and
                         types.get(args.rgb_camera_info_topic) == CAMERA_INFO_TYPE else None),
        event_camera_info=(args.event_camera_info_topic if use_recorded and
                           types.get(args.event_camera_info_topic) == CAMERA_INFO_TYPE else None),
    )
    if args.sparse_only and topics.rgb_camera_info is None and not (args.rgb_width and args.rgb_height):
        raise RuntimeError("Sparse-only conversion requires RGB CameraInfo or --rgb-width and --rgb-height")
    if use_recorded and topics.event_camera_info is None and not (args.event_width and args.event_height):
        raise RuntimeError("Recorded event conversion requires event CameraInfo or --event-width and --event-height")

    collect_goto_s_debug, collect_target_base_debug = validate_topics(
        types,
        topics,
        collect_goto_s_debug=True,
        collect_target_base_debug=not args.no_target_base,
        collect_rgb_2d=not args.no_rgb_2d,
        require_rgb=not args.sparse_only,
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

    tracker_updates_by_episode = {}
    event_tracker_aligner = None
    event_tracker_metadata = None
    if raw_events_h5 is not None and not use_recorded:
        openmv = import_openmv_tracker_api(args.openmv_cam_root)
        bag_name = Path(bag_path).name
        recording_name = bag_name[:-4] if bag_name.endswith("_bag") else bag_name
        tracker_config, tracker_config_json, pre_roll_ms, config_source = (
            resolve_event_tracker_config(args, recording_name, openmv)
        )
        config_hash = hashlib.sha256(tracker_config_json.encode("utf-8")).hexdigest()
        openmv_episodes = [
            openmv.Episode(
                Path(f"episode_{episode.output_idx}.hdf5"),
                f"episode_{episode.output_idx}", episode.output_idx,
                episode.source_idx, episode.start, episode.end,
                np.empty(0, dtype=np.float64),
            )
            for episode in episodes
        ]
        episode_names = [episode.name for episode in openmv_episodes]
        if args.reuse_event_tracker_cache:
            tracker_updates_by_episode = load_event_tracker_cache(
                args.reuse_event_tracker_cache, raw_events_h5, config_hash,
                episode_names, openmv.TrackerUpdate,
            )
            log(f"[INFO] reused event tracker cache: {args.reuse_event_tracker_cache}")
        else:
            tracker_updates_by_episode = openmv.run_tracker_updates(
                raw_events_h5, openmv_episodes, tracker_config,
                pre_roll_ms=pre_roll_ms,
            )
        if any(name not in tracker_updates_by_episode for name in episode_names):
            raise RuntimeError("OpenMV tracker did not return every selected episode")
        if args.save_event_tracker_cache:
            save_event_tracker_cache(
                args.save_event_tracker_cache, raw_events_h5, config_hash,
                tracker_updates_by_episode,
            )
            log(f"[INFO] saved event tracker cache: {args.save_event_tracker_cache}")
        event_tracker_aligner = openmv.align_tracker_updates_to_policy_grid
        event_tracker_metadata = {
            "schema_version": EVENT_TRACKER_PROCESSING_SCHEMA,
            "raw_events_h5": str(Path(raw_events_h5).resolve()),
            "tracker_config_json": tracker_config_json,
            "tracker_config_hash": config_hash,
            "tracker_config_source": config_source,
            "tracker_code_version": "openmv_cam@5336fa880f28a861f2ff79d4480d081a4a48a819",
            "availability_timestamp_domain": "packet_ros_t_ns",
            "sensor_timestamp_domain": "genx320_microseconds",
            "sensor_width": int(tracker_config.get("width", 320)),
            "sensor_height": int(tracker_config.get("height", 320)),
            "pre_roll_ms": float(pre_roll_ms),
            "max_observation_age_sec": float(args.max_observation_age_sec),
        }
        update_count = sum(len(rows) for rows in tracker_updates_by_episode.values())
        log(f"[INFO] OpenMV tracker updates: {update_count} across {len(episodes)} episodes")
    elif args.sparse_source == "event" and not use_recorded:
        raise RuntimeError("--sparse-source event requires a raw-event HDF5 sidecar")
    elif use_recorded:
        tracker_updates_by_episode = read_recorded_event_updates(
            bag_path, args.event_update_topic, episodes)
        event_tracker_metadata = {
            "schema_version": "recorded_event_tracker_updates_v1",
            "sensor_width": int(args.event_width or 320),
            "sensor_height": int(args.event_height or 320),
            "availability_timestamp_domain": "EventTrackerUpdate.availability_timestamp_ns",
            "receipt_timestamp_domain": "MCAP bag record time",
            "origin": "recorded_ros_topic",
            "source_topic": args.event_update_topic,
            "max_observation_age_sec": float(args.max_observation_age_sec),
        }

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
    if raw_events_h5 is not None and not args.sparse_only:
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
                        event_tracker_updates=tracker_updates_by_episode.get(
                            f"episode_{episode.output_idx}"
                        ) if raw_events_h5 is not None else None,
                        event_tracker_aligner=event_tracker_aligner,
                        event_tracker_metadata=event_tracker_metadata,
                        sparse_source=args.sparse_source,
                        max_observation_age_sec=args.max_observation_age_sec,
                        sparse_only=args.sparse_only,
                        rgb_sparse_size=((args.rgb_width, args.rgb_height)
                                         if args.rgb_width and args.rgb_height else None),
                        event_sparse_size=((args.event_width, args.event_height)
                                           if args.event_width and args.event_height else None),
                        topic_profile=profile,
                        recorded_event_rows=tracker_updates_by_episode.get(
                            f"episode_{episode.output_idx}") if use_recorded else None,
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
                    event_tracker_updates=tracker_updates_by_episode.get(
                        f"episode_{episode.output_idx}"
                    ) if raw_events_h5 is not None else None,
                    event_tracker_aligner=event_tracker_aligner,
                    event_tracker_metadata=event_tracker_metadata,
                    sparse_source=args.sparse_source,
                    max_observation_age_sec=args.max_observation_age_sec,
                    sparse_only=args.sparse_only,
                    rgb_sparse_size=((args.rgb_width, args.rgb_height)
                                     if args.rgb_width and args.rgb_height else None),
                    event_sparse_size=((args.event_width, args.event_height)
                                       if args.event_width and args.event_height else None),
                    topic_profile=profile,
                    recorded_event_rows=tracker_updates_by_episode.get(
                        f"episode_{episode.output_idx}") if use_recorded else None,
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


def main() -> None:
    conversion_start_wall = time.perf_counter()
    args = parse_args()
    for name, default in (
        ("min_free_disk_gb", DEFAULT_MIN_FREE_DISK_GB),
        ("disk_check_path", DEFAULT_DISK_CHECK_PATH),
        ("disk_check_interval_sec", DEFAULT_DISK_CHECK_INTERVAL_SEC),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)

    try:
        with DiskSpaceMonitor(
            path=args.disk_check_path,
            minimum_gib=args.min_free_disk_gb,
            interval_sec=args.disk_check_interval_sec,
        ):
            run_conversion(args, conversion_start_wall)
    except LowDiskSpaceError as error:
        log(f"[ERROR] {error}")
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
