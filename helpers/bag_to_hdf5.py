#!/usr/bin/env python3

import os
import gc
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
import cv2
import h5py
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPIC_RGB = "/camera/camera/color/image_raw"
TOPIC_EVENT = "/openmv_cam/image"
TOPIC_JOINT = "/joint_states"
TOPIC_GRIPPER_STATE = "/teleop/gripper_state_cmd"
TOPIC_TWIST = "/cartesian_cmd/twist"
TOPIC_WRENCH = "/right_franka/external_wrenches"
TOPIC_EPISODE = "/episode/control"
MIN_DURATION = 4.0  # seconds



FPS = 30.0
DT = 1.0 / FPS


@dataclass
class EpisodeWindow:
    idx: int
    start: float
    end: float


def log(msg: str):
    print(msg, flush=True)


def bag_timestamp_to_sec(ns: int) -> float:
    return float(ns) * 1e-9


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def bag_top_level_name(bag_path: str) -> str:
    """
    Derive a stable output directory name from the bag path.
    - If bag_path is a directory: use the directory name.
    - If bag_path is a file (e.g. *.mcap or *.db3): use filename without extension.
    """
    norm = os.path.normpath(bag_path)
    if os.path.isdir(norm):
        return os.path.basename(norm)
    return os.path.splitext(os.path.basename(norm))[0]


def resolve_recording_dir(recording_dir: str) -> Tuple[str, Optional[str], str]:
    """
    Resolve a recording directory into (bag_path, raw_events_h5_path, recording_name).

    Directory layout expected:
        recording_dir/
            <recording_name>_bag/       <- ROS2 bag
            <recording_name>_raw_events.h5  <- optional raw event HDF5
    """
    recording_dir = os.path.abspath(recording_dir)
    if not os.path.isdir(recording_dir):
        raise RuntimeError(f"recording_dir does not exist or is not a directory: {recording_dir}")

    recording_name = os.path.basename(recording_dir)

    # ---- find bag directory ----
    exact_bag = os.path.join(recording_dir, f"{recording_name}_bag")
    bag_candidates = []
    for entry in os.listdir(recording_dir):
        full = os.path.join(recording_dir, entry)
        if not os.path.isdir(full):
            continue
        if entry.endswith("_bag") or os.path.exists(os.path.join(full, "metadata.yaml")):
            bag_candidates.append(full)

    if len(bag_candidates) == 0:
        raise RuntimeError(
            "No ROS2 bag directory found in recording_dir. "
            "Expected a *_bag directory or a directory containing metadata.yaml."
        )
    elif os.path.isdir(exact_bag):
        bag_path = exact_bag
    elif len(bag_candidates) == 1:
        bag_path = bag_candidates[0]
    else:
        raise RuntimeError(
            f"Multiple bag candidates found in recording_dir and no exact match "
            f"for '{recording_name}_bag'. Candidates: {bag_candidates}"
        )

    # ---- find raw events HDF5 ----
    exact_h5 = os.path.join(recording_dir, f"{recording_name}_raw_events.h5")
    h5_candidates = [
        os.path.join(recording_dir, entry)
        for entry in os.listdir(recording_dir)
        if entry.endswith("_raw_events.h5") and os.path.isfile(os.path.join(recording_dir, entry))
    ]

    if len(h5_candidates) == 0:
        log("[WARNING] no *_raw_events.h5 found in recording_dir; falling back to /openmv_cam/image")
        raw_events_h5_path: Optional[str] = None
    elif os.path.isfile(exact_h5):
        raw_events_h5_path = exact_h5
    elif len(h5_candidates) == 1:
        raw_events_h5_path = h5_candidates[0]
    else:
        raise RuntimeError(
            f"Multiple *_raw_events.h5 candidates found in recording_dir and no exact match "
            f"for '{recording_name}_raw_events.h5'. Candidates: {h5_candidates}"
        )

    return os.path.abspath(bag_path), raw_events_h5_path, recording_name


def open_reader(bag_path: str, storage_id: str = "mcap"):
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id=storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def get_topic_type_map(reader) -> Dict[str, str]:
    topic_types = reader.get_all_topics_and_types()
    return {x.name: x.type for x in topic_types}


def get_type_class_map(
    topic_type_map: Dict[str, str],
    only_topics: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if only_topics is None:
        items = topic_type_map.items()
    else:
        items = [(t, topic_type_map[t]) for t in only_topics if t in topic_type_map]
    return {topic: get_message(msg_type) for topic, msg_type in items}


def check_required_topics(
    topic_type_map: Dict[str, str],
    require_event_topic: bool = True,
):
    required = {
        TOPIC_RGB,
        TOPIC_JOINT,
        TOPIC_GRIPPER_STATE,
        TOPIC_TWIST,
        TOPIC_WRENCH,
        TOPIC_EPISODE,
    }
    if require_event_topic:
        required.add(TOPIC_EVENT)
    present = set(topic_type_map.keys())
    missing = required - present
    if missing:
        raise RuntimeError(f"Missing required topics: {sorted(missing)}")

    log("[INFO] Required topics found.")
    for t in sorted(required):
        log(f"       {t}  ::  {topic_type_map[t]}")


def image_msg_to_numpy(msg) -> np.ndarray:
    """
    Convert sensor_msgs/msg/Image to numpy.
    Returns RGB for color images, single-channel for mono images.
    """
    h = msg.height
    w = msg.width
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        img = data.reshape((h, w, 3))
        return img

    if enc == "bgr8":
        img = data.reshape((h, w, 3))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if enc in ("mono8", "8uc1"):
        img = data.reshape((h, w))
        return img

    if enc == "rgba8":
        img = data.reshape((h, w, 4))
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if enc == "bgra8":
        img = data.reshape((h, w, 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def render_event_frame_from_raw_arrays(
    event_type: np.ndarray,
    event_x: np.ndarray,
    event_y: np.ndarray,
    width: int = 320,
    height: int = 320,
    contrast: float = 4.0,
    step: float = 1.0,
) -> np.ndarray:
    """
    Render an event frame from raw event arrays into a single-channel uint8 image.
    Neutral gray is 128. Positive events (type == 1) brighten, others darken.
    """
    if event_type.size == 0:
        return np.full((height, width), 128, dtype=np.uint8)

    xs = np.asarray(event_x, dtype=np.int64)
    ys = np.asarray(event_y, dtype=np.int64)
    et = np.asarray(event_type)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not np.any(valid):
        return np.full((height, width), 128, dtype=np.uint8)

    xs = xs[valid]
    ys = ys[valid]
    et = et[valid]

    acc = np.zeros((height, width), dtype=np.float32)
    values = np.where(et == 1, float(step), -float(step)).astype(np.float32)
    np.add.at(acc, (ys, xs), values)

    max_abs = float(np.max(np.abs(acc)))
    if max_abs > 0.0:
        acc = acc / max_abs

    img = 128.0 + (acc * float(contrast) * 127.0)
    img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


class RawEventStore:
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.f = h5py.File(h5_path, "r")

        required = [
            "events/type",
            "events/x",
            "events/y",
            "events/t_us",
            "packets/ros_t_ns",
            "packets/start_event_idx",
            "packets/end_event_idx",
        ]
        missing = [k for k in required if k not in self.f]
        if missing:
            self.f.close()
            raise RuntimeError(
                f"Raw event HDF5 missing required datasets: {missing}"
            )

        self.events_type_ds = self.f["events/type"]
        self.events_x_ds = self.f["events/x"]
        self.events_y_ds = self.f["events/y"]
        self.events_t_us_ds = self.f["events/t_us"]
        self.packets_ros_t_ns_ds = self.f["packets/ros_t_ns"]
        self.packets_start_event_idx_ds = self.f["packets/start_event_idx"]
        self.packets_end_event_idx_ds = self.f["packets/end_event_idx"]

        self.packet_ros_t_ns = np.asarray(self.packets_ros_t_ns_ds[:], dtype=np.int64)
        self.packet_start_event_idx = np.asarray(
            self.packets_start_event_idx_ds[:], dtype=np.int64
        )
        self.packet_end_event_idx = np.asarray(
            self.packets_end_event_idx_ds[:], dtype=np.int64
        )

        self.width = int(self.f.attrs.get("width", 320))
        self.height = int(self.f.attrs.get("height", 320))

        if self.packet_ros_t_ns.size > 0:
            self.packet_ros_start_ns = int(self.packet_ros_t_ns[0])
            self.packet_ros_end_ns = int(self.packet_ros_t_ns[-1])
            log(
                "[INFO] raw events packet ROS time range (sec): "
                f"{self.packet_ros_start_ns * 1e-9:.6f} .. {self.packet_ros_end_ns * 1e-9:.6f}"
            )
        else:
            self.packet_ros_start_ns = None
            self.packet_ros_end_ns = None
            log("[WARNING] raw events HDF5 has no packets in /packets/ros_t_ns")

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    def frame_3chef_at_bag_time(
        self,
        bag_t_sec: float,
        windows_ms: Tuple[float, float, float] = (50.0, 250.0, 1000.0),
        mode: str = "cumulative",
        contrast: float = 4.0,
        step: float = 1.0,
        packet_margin_ms: float = 50.0,
    ) -> np.ndarray:
        if len(windows_ms) != 3:
            raise RuntimeError(f"windows_ms must have length 3, got {windows_ms}")
        w0_ms, w1_ms, w2_ms = [float(w) for w in windows_ms]
        if not (w0_ms > 0.0 and w1_ms > 0.0 and w2_ms > 0.0):
            raise RuntimeError(f"windows_ms must be positive, got {windows_ms}")
        if not (w0_ms <= w1_ms <= w2_ms):
            raise RuntimeError(
                "windows_ms must be non-decreasing (small to large), "
                f"got {windows_ms}"
            )
        if mode not in ("cumulative", "shifted"):
            raise RuntimeError(f"Unsupported event frame mode: {mode}")

        neutral = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        if self.packet_ros_t_ns.size == 0:
            return neutral

        now_ros_ns = int(bag_t_sec * 1e9)
        max_window_ns = int(w2_ms * 1e6)
        margin_ns = int(float(packet_margin_ms) * 1e6)
        start_ros_ns = now_ros_ns - max_window_ns - margin_ns
        end_ros_ns = now_ros_ns

        p0 = int(np.searchsorted(self.packet_ros_t_ns, start_ros_ns, side="left"))
        p1 = int(np.searchsorted(self.packet_ros_t_ns, end_ros_ns, side="right"))
        if p0 >= p1:
            return neutral

        ev_start = int(self.packet_start_event_idx[p0])
        ev_end = int(self.packet_end_event_idx[p1 - 1])
        if ev_end <= ev_start:
            return neutral

        ev_type = np.asarray(self.events_type_ds[ev_start:ev_end])
        ev_x = np.asarray(self.events_x_ds[ev_start:ev_end])
        ev_y = np.asarray(self.events_y_ds[ev_start:ev_end])
        ev_t_us = np.asarray(self.events_t_us_ds[ev_start:ev_end], dtype=np.int64)

        if ev_t_us.size == 0:
            return neutral

        now_event_t_us = int(np.max(ev_t_us))
        w0_us = int(w0_ms * 1e3)
        w1_us = int(w1_ms * 1e3)
        w2_us = int(w2_ms * 1e3)

        if mode == "shifted":
            ranges = [
                (now_event_t_us - w0_us, now_event_t_us),
                (now_event_t_us - w1_us, now_event_t_us - w0_us),
                (now_event_t_us - w2_us, now_event_t_us - w1_us),
            ]
        else:
            ranges = [
                (now_event_t_us - w0_us, now_event_t_us),
                (now_event_t_us - w1_us, now_event_t_us),
                (now_event_t_us - w2_us, now_event_t_us),
            ]

        channels = []
        for lo_us, hi_us in ranges:
            if hi_us == now_event_t_us:
                mask = (ev_t_us >= lo_us) & (ev_t_us <= hi_us)
            else:
                mask = (ev_t_us >= lo_us) & (ev_t_us < hi_us)
            if not np.any(mask):
                channels.append(np.full((self.height, self.width), 128, dtype=np.uint8))
                continue

            frame = render_event_frame_from_raw_arrays(
                event_type=ev_type[mask],
                event_x=ev_x[mask],
                event_y=ev_y[mask],
                width=self.width,
                height=self.height,
                contrast=contrast,
                step=step,
            )
            channels.append(frame)

        return np.stack(channels, axis=2).astype(np.uint8)


def twist_msg_to_vec(msg) -> np.ndarray:
    return np.array([
        msg.twist.linear.x,
        msg.twist.linear.y,
        msg.twist.linear.z,
        msg.twist.angular.x,
        msg.twist.angular.y,
        msg.twist.angular.z,
    ], dtype=np.float32)


def wrench_msg_to_vec(msg) -> np.ndarray:
    arr = np.asarray(msg.data, dtype=np.float32)
    if arr.shape[0] != 12:
        raise RuntimeError(
            f"Expected /right_franka/external_wrenches to have 12 floats, got shape {arr.shape}"
        )
    return arr


def build_qpos_from_joint_state(joint_names: List[str], joint_pos: np.ndarray) -> np.ndarray:
    arm_joint_names = [
        "right_fr3_joint1",
        "right_fr3_joint2",
        "right_fr3_joint3",
        "right_fr3_joint4",
        "right_fr3_joint5",
        "right_fr3_joint6",
        "right_fr3_joint7",
    ]
    finger_joint_names = ["right_fr3_finger_joint1", "right_fr3_finger_joint2"]
    required_names = arm_joint_names + finger_joint_names

    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    missing = [name for name in required_names if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing required Franka joint names in /joint_states: {missing}")

    qpos_8d = np.empty(8, dtype=np.float32)
    for i, joint_name in enumerate(arm_joint_names):
        qpos_8d[i] = np.float32(joint_pos[name_to_idx[joint_name]])

    gripper_width = np.float32(
        joint_pos[name_to_idx["right_fr3_finger_joint1"]] +
        joint_pos[name_to_idx["right_fr3_finger_joint2"]]
    )
    qpos_8d[7] = gripper_width
    return qpos_8d


def infer_initial_gripper_width(joint_names: List[str], joint_pos: np.ndarray) -> np.float32:
    finger_joint_names = ["right_fr3_finger_joint1", "right_fr3_finger_joint2"]
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    missing = [name for name in finger_joint_names if name not in name_to_idx]
    if missing:
        raise RuntimeError(
            f"Missing finger joints for initial gripper width inference: {missing}"
        )

    return np.float32(
        joint_pos[name_to_idx["right_fr3_finger_joint1"]] +
        joint_pos[name_to_idx["right_fr3_finger_joint2"]]
    )


def extract_episode_windows(bag_path: str, storage_id: str = "mcap") -> List[EpisodeWindow]:
    log("[INFO] Pass 1/2: scanning /episode/control for episode boundaries...")
    reader = open_reader(bag_path, storage_id=storage_id)
    topic_type_map = get_topic_type_map(reader)
    msg_types = get_type_class_map(topic_type_map, only_topics={TOPIC_EPISODE})

    current_start: Optional[float] = None
    committed_windows: List[EpisodeWindow] = []

    n_total = 0
    n_episode_msgs = 0
    n_start = 0
    n_stop = 0
    n_cancel_current = 0
    n_cancel_last = 0
    n_ignored = 0

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        n_total += 1

        if n_total % 50000 == 0:
            log(f"[DEBUG] pass1 read {n_total} messages so far")

        if topic != TOPIC_EPISODE:
            continue

        n_episode_msgs += 1
        msg = deserialize_message(raw, msg_types[topic])
        t = bag_timestamp_to_sec(t_ns)

        log(f"[DEBUG] /episode/control at {t:.6f}s -> {msg.data}")

        if msg.data == 1:
            n_start += 1
            if current_start is not None:
                log(
                    "[WARNING] start received while already recording; "
                    "ignoring duplicate start"
                )
            else:
                current_start = t
                log(f"[INFO] start recording candidate episode at {t:.6f}")

        elif msg.data == 2:
            n_stop += 1
            if current_start is None:
                log("[WARNING] stop received while not recording; ignoring")
            else:
                ep = EpisodeWindow(
                    idx=len(committed_windows),
                    start=current_start,
                    end=t,
                )
                committed_windows.append(ep)
                log(
                    f"[INFO] committed candidate episode {ep.idx}: "
                    f"start={ep.start:.6f}, end={ep.end:.6f}, dur={ep.end - ep.start:.3f}s"
                )
                current_start = None

        elif msg.data == 3:
            n_cancel_current += 1
            if current_start is None:
                log("[WARNING] cancel_current received while not recording; ignoring")
            else:
                log(
                    f"[INFO] cancelled current candidate episode: "
                    f"start={current_start:.6f}, cancel_time={t:.6f}"
                )
                current_start = None

        elif msg.data == 4:
            n_cancel_last += 1
            if current_start is not None:
                log(
                    "[WARNING] cancel_last received while currently recording; "
                    "ignoring"
                )
            elif committed_windows:
                popped = committed_windows.pop()
                log(
                    f"[INFO] removed last committed candidate episode: "
                    f"idx={popped.idx}, start={popped.start:.6f}, end={popped.end:.6f}"
                )
                for i, ep in enumerate(committed_windows):
                    ep.idx = i
            else:
                log("[WARNING] cancel_last received but no committed episode exists")

        else:
            n_ignored += 1
            log(f"[WARNING] unknown /episode/control value {msg.data}; ignoring")

    log(f"[INFO] pass1 done. total messages read: {n_total}")
    log(f"[INFO] episode control messages: {n_episode_msgs}")
    log(f"[INFO] marker counts: start={n_start}, stop={n_stop}, cancel_current={n_cancel_current}, cancel_last={n_cancel_last}, ignored={n_ignored}")

    if current_start is not None:
        log(
            f"[WARNING] bag ended while recording candidate episode from "
            f"{current_start:.6f}; discarding unfinished candidate"
        )

    if not committed_windows:
        raise RuntimeError("No valid episode windows found after cancellation handling.")

    log("[INFO] final committed candidate windows before duration filtering:")
    for ep in committed_windows:
        log(
            f"       episode {ep.idx}: start={ep.start:.6f}, "
            f"end={ep.end:.6f}, dur={ep.end - ep.start:.3f}s"
        )

    return committed_windows


def first_index_ge(times: List[float], t: float, start_idx: int = 0) -> int:
    i = start_idx
    n = len(times)
    while i < n and times[i] < t:
        i += 1
    return min(i, n - 1)


def last_index_le(times: List[float], t: float, start_idx: int = 0) -> int:
    i = start_idx
    n = len(times)
    if n == 0:
        return -1

    while i < n and times[i] <= t:
        i += 1
    return i - 1


def collect_single_episode_data(
    bag_path: str,
    ep: EpisodeWindow,
    require_event_topic: bool = True,
    storage_id: str = "mcap",
) -> Dict[str, Any]:
    """
    Stream only one episode into memory.
    This is the key RAM fix.
    """
    log("")
    log(f"[INFO] Pass 2/2: collecting episode {ep.idx}")
    log(f"[INFO] time window: [{ep.start:.6f}, {ep.end:.6f}]  dur={ep.end - ep.start:.3f}s")

    reader = open_reader(bag_path, storage_id=storage_id)
    topic_type_map = get_topic_type_map(reader)

    # We include only messages within this episode window.
    data = {
        "rgb_t": [],
        "rgb_msg": [],
        "event_t": [],
        "event_msg": [],
        "joint_t": [],
        "joint_pos": [],
        "gripper_state_t": [],
        "gripper_state": [],
        "twist_t": [],
        "twist": [],
        "wrench_t": [],
        "wrench": [],
        "joint_names": None,
    }

    msg_count = 0
    kept_count = 0

    # Track whether topics already passed episode end in time-ordered bag.
    seen_after_end = {
        TOPIC_RGB: False,
        TOPIC_JOINT: False,
        TOPIC_GRIPPER_STATE: False,
        TOPIC_TWIST: False,
        TOPIC_WRENCH: False,
    }
    if require_event_topic:
        seen_after_end[TOPIC_EVENT] = False

    tracked_topics = set(seen_after_end.keys())
    msg_types = get_type_class_map(topic_type_map, only_topics=tracked_topics)

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        msg_count += 1
        t = bag_timestamp_to_sec(t_ns)

        if msg_count % 50000 == 0:
            log(f"[DEBUG] episode {ep.idx}: scanned {msg_count} bag messages, kept {kept_count}")

        if topic not in tracked_topics:
            continue

        if t < ep.start:
            continue

        if t > ep.end:
            seen_after_end[topic] = True
            if all(seen_after_end.values()):
                log(f"[DEBUG] episode {ep.idx}: all tracked topics have passed episode end, stopping bag scan early")
                break
            continue

        msg = deserialize_message(raw, msg_types[topic])
        kept_count += 1

        if topic == TOPIC_RGB:
            data["rgb_t"].append(t)
            data["rgb_msg"].append(msg)

        elif topic == TOPIC_EVENT:
            data["event_t"].append(t)
            data["event_msg"].append(msg)

        elif topic == TOPIC_JOINT:
            names = list(msg.name)
            pos = np.array(msg.position, dtype=np.float32)

            if data["joint_names"] is None:
                data["joint_names"] = names
                log(f"[INFO] episode {ep.idx}: canonical joint order set from first /joint_states message")
                log(f"[INFO] joint names: {data['joint_names']}")

            if names != data["joint_names"]:
                name_to_idx = {n: i for i, n in enumerate(names)}
                reordered = np.empty(len(data["joint_names"]), dtype=np.float32)
                for i, n in enumerate(data["joint_names"]):
                    if n not in name_to_idx:
                        raise RuntimeError(
                            f"Episode {ep.idx}: joint '{n}' missing in later /joint_states message."
                        )
                    reordered[i] = pos[name_to_idx[n]]
                pos = reordered

            data["joint_t"].append(t)
            data["joint_pos"].append(pos)

        elif topic == TOPIC_GRIPPER_STATE:
            data["gripper_state_t"].append(t)
            data["gripper_state"].append(np.array([1.0 if msg.data else 0.0], dtype=np.float32))

        elif topic == TOPIC_TWIST:
            data["twist_t"].append(t)
            data["twist"].append(twist_msg_to_vec(msg))

        elif topic == TOPIC_WRENCH:
            data["wrench_t"].append(t)
            data["wrench"].append(wrench_msg_to_vec(msg))

    log(f"[INFO] episode {ep.idx}: scan finished")
    log(f"       scanned bag messages: {msg_count}")
    log(f"       kept messages:        {kept_count}")
    log(f"       rgb frames:           {len(data['rgb_t'])}")
    if require_event_topic:
        log(f"       event frames:         {len(data['event_t'])}")
    else:
        log("       event frames:         not collected; using raw_events_h5")
    log(f"       joint msgs:           {len(data['joint_t'])}")
    log(f"       gripper state msgs:   {len(data['gripper_state_t'])}")
    log(f"       twist msgs:           {len(data['twist_t'])}")
    log(f"       wrench msgs:          {len(data['wrench_t'])}")

    required_data_keys = ["rgb_t", "joint_t", "twist_t", "wrench_t"]
    if require_event_topic:
        required_data_keys.append("event_t")

    for k in required_data_keys:
        if len(data[k]) == 0:
            raise RuntimeError(f"Episode {ep.idx}: no data collected for {k}")

    return data


def sample_episode_to_arrays(
    data: Dict[str, Any],
    ep: EpisodeWindow,
    raw_event_store: Optional[RawEventStore] = None,
    event_frame_windows_ms: Tuple[float, float, float] = (50.0, 250.0, 1000.0),
    event_frame_mode: str = "cumulative",
    event_frame_contrast: float = 4.0,
    event_frame_step: float = 1.0,
    event_packet_margin_ms: float = 50.0,
    initial_delay_steps: int = 0,
) -> Dict[str, np.ndarray]:
    log(f"[INFO] episode {ep.idx}: sampling onto {FPS:.1f} Hz grid using next-available datapoint")

    grid = np.arange(ep.start, ep.end + 1e-9, DT, dtype=np.float64)
    log(f"[INFO] episode {ep.idx}: grid has {len(grid)} steps")

    rgb_t = data["rgb_t"]
    joint_t = data["joint_t"]
    gripper_state_t = data["gripper_state_t"]
    twist_t = data["twist_t"]
    wrench_t = data["wrench_t"]

    rgb_msg = data["rgb_msg"]
    joint_pos = data["joint_pos"]
    gripper_state = data["gripper_state"]
    twist = data["twist"]
    wrench = data["wrench"]
    joint_names = data["joint_names"]

    if raw_event_store is None:
        event_t = data["event_t"]
        event_msg = data["event_msg"]
        event_idx = first_index_ge(event_t, ep.start)
    else:
        event_t = None
        event_msg = None
        event_idx = -1

    if joint_names is None:
        raise RuntimeError(f"Episode {ep.idx}: joint_names not set from /joint_states")

    rgb_idx = first_index_ge(rgb_t, ep.start)
    joint_idx = first_index_ge(joint_t, ep.start)
    gripper_state_idx = last_index_le(gripper_state_t, ep.start)
    twist_idx = first_index_ge(twist_t, ep.start)
    wrench_idx = first_index_ge(wrench_t, ep.start)

    initial_gripper_state = np.float32(0.0)  # open
    log(f"[INFO] episode {ep.idx}: sparse gripper state event count = {len(gripper_state_t)}")
    if raw_event_store is not None:
        log(
            f"[INFO] episode {ep.idx}: event source = raw H5 3chef "
            f"(mode={event_frame_mode}, windows_ms={event_frame_windows_ms})"
        )
    else:
        log(f"[INFO] episode {ep.idx}: event source = fallback /openmv_cam/image repeated to 3ch")

    rgb_frames = []
    event_frames = []
    qpos_seq = []
    gripper_seq = []
    twist_seq = []
    wrench_seq = []
    for i, t in enumerate(grid):
        rgb_idx = first_index_ge(rgb_t, t, rgb_idx)
        if raw_event_store is None:
            event_idx = first_index_ge(event_t, t, event_idx)
        joint_idx = first_index_ge(joint_t, t, joint_idx)
        gripper_state_idx = last_index_le(gripper_state_t, t, max(0, gripper_state_idx + 1))
        twist_idx = first_index_ge(twist_t, t, twist_idx)
        wrench_idx = first_index_ge(wrench_t, t, wrench_idx)

        if i % 100 == 0:
            if raw_event_store is None:
                event_debug = f"event_idx={event_idx}"
            else:
                event_debug = "event_idx=raw_h5"
            log(
                f"[DEBUG] episode {ep.idx}: sample {i:04d}/{len(grid)} "
                f"| rgb_idx={rgb_idx} {event_debug} joint_idx={joint_idx} "
                f"gripper_state_idx={gripper_state_idx} twist_idx={twist_idx} "
                f"wrench_idx={wrench_idx}"
            )

        rgb_np = image_msg_to_numpy(rgb_msg[rgb_idx])
        if raw_event_store is not None:
            event_np = raw_event_store.frame_3chef_at_bag_time(
                t,
                windows_ms=event_frame_windows_ms,
                mode=event_frame_mode,
                contrast=event_frame_contrast,
                step=event_frame_step,
                packet_margin_ms=event_packet_margin_ms,
            )
        else:
            event_np = image_msg_to_numpy(event_msg[event_idx])
            if event_np.ndim == 2:
                event_np = np.repeat(event_np[:, :, None], 3, axis=2)
            elif event_np.ndim == 3 and event_np.shape[2] == 3:
                pass
            else:
                raise RuntimeError(f"Unsupported fallback event image shape: {event_np.shape}")

        rgb_frames.append(rgb_np)
        event_frames.append(event_np)
        qpos_seq.append(build_qpos_from_joint_state(joint_names, joint_pos[joint_idx]))
        if gripper_state_idx >= 0:
            gripper_value = np.float32(gripper_state[gripper_state_idx][0])
        else:
            gripper_value = initial_gripper_state

        gripper_seq.append(np.array([gripper_value], dtype=np.float32))
        twist_seq.append(twist[twist_idx])
        wrench_seq.append(wrench[wrench_idx])

    rgb_frames = np.stack(rgb_frames, axis=0)
    event_frames = np.stack(event_frames, axis=0)
    qpos_seq = np.stack(qpos_seq, axis=0).astype(np.float32)
    gripper_seq = np.stack(gripper_seq, axis=0).astype(np.float32)
    twist_seq = np.stack(twist_seq, axis=0).astype(np.float32)
    wrench_seq = np.stack(wrench_seq, axis=0).astype(np.float32)
    action_combined = np.concatenate([twist_seq, gripper_seq], axis=1)
    timestamps = grid.astype(np.float64)

    # --- apply initial delay ---
    if initial_delay_steps > 0:
        n_before = len(timestamps)
        if initial_delay_steps >= n_before:
            raise RuntimeError(
                f"Episode {ep.idx}: initial_delay_steps={initial_delay_steps} "
                f"would remove all {n_before} sampled timesteps."
            )

        log(
            f"[INFO] episode {ep.idx}: applying initial delay: "
            f"dropping first {initial_delay_steps}/{n_before} sampled timesteps "
            f"({initial_delay_steps / FPS:.3f}s at {FPS:.1f} Hz)"
        )

        timestamps = timestamps[initial_delay_steps:]
        rgb_frames = rgb_frames[initial_delay_steps:]
        event_frames = event_frames[initial_delay_steps:]
        qpos_seq = qpos_seq[initial_delay_steps:]
        gripper_seq = gripper_seq[initial_delay_steps:]
        twist_seq = twist_seq[initial_delay_steps:]
        wrench_seq = wrench_seq[initial_delay_steps:]
        action_combined = action_combined[initial_delay_steps:]

    return {
        "timestamps": timestamps,
        "rgb": rgb_frames,
        "event": event_frames,
        "qpos": qpos_seq,
        "twist": twist_seq,
        "wrench": wrench_seq,
        "wrench_o_f_ext_hat_k": wrench_seq[:, 0:6],
        "wrench_k_f_ext_hat_k": wrench_seq[:, 6:12],
        "gripper": gripper_seq,
        "combined": action_combined,
    }


def write_episode_hdf5(
    out_path: str,
    arrays: Dict[str, np.ndarray],
    joint_names: List[str],
    ep: EpisodeWindow,
    event_representation: str = "mono_or_3ch_fallback",
    event_frame_windows_ms=None,
    event_frame_mode=None,
    event_frame_contrast=None,
    event_frame_step=None,
    raw_events_h5=None,
    initial_delay_steps: int = 0,
):
    log(f"[INFO] episode {ep.idx}: writing HDF5 -> {out_path}")

    with h5py.File(out_path, "w") as f:
        f.attrs["sim"] = False
        f.attrs["fps"] = FPS
        f.attrs["episode_index"] = ep.idx
        f.attrs["episode_start"] = ep.start
        f.attrs["episode_end"] = ep.end
        f.attrs["joint_names"] = np.array(joint_names, dtype=h5py.string_dtype("utf-8"))
        if event_frame_windows_ms is None:
            event_frame_windows_ms = [50.0, 250.0, 1000.0]
        if event_frame_mode is None:
            event_frame_mode = "cumulative"
        if event_frame_contrast is None:
            event_frame_contrast = 4.0
        if event_frame_step is None:
            event_frame_step = 1.0

        f.attrs["event_representation"] = event_representation
        f.attrs["event_frame_windows_ms"] = np.array(event_frame_windows_ms, dtype=np.float32)
        f.attrs["event_frame_mode"] = event_frame_mode
        f.attrs["event_frame_contrast"] = float(event_frame_contrast)
        f.attrs["event_frame_step"] = float(event_frame_step)
        if raw_events_h5:
            f.attrs["raw_events_h5"] = str(raw_events_h5)

        f.attrs["initial_delay_steps"] = int(initial_delay_steps)
        f.attrs["initial_delay_seconds"] = float(initial_delay_steps) / float(FPS)

        if len(arrays["timestamps"]) > 0:
            f.attrs["effective_episode_start"] = float(arrays["timestamps"][0])
            f.attrs["effective_episode_end"] = float(arrays["timestamps"][-1])

        obs = f.create_group("observations")
        img = obs.create_group("images")

        obs.create_dataset("timestamps", data=arrays["timestamps"], dtype=np.float64)
        obs.create_dataset("qpos", data=arrays["qpos"], dtype=np.float32)

        img.create_dataset(
            "rgb",
            data=arrays["rgb"],
            dtype=np.uint8,
            chunks=(1, *arrays["rgb"].shape[1:]),
        )
        img.create_dataset(
            "event",
            data=arrays["event"],
            dtype=np.uint8,
            chunks=(1, *arrays["event"].shape[1:]),
        )

        wrench_group = obs.create_group("wrenches")
        wrench_group.attrs["source_topic"] = TOPIC_WRENCH
        wrench_group.attrs["combined_layout"] = np.array(
            [
                "o_Fx", "o_Fy", "o_Fz", "o_Tx", "o_Ty", "o_Tz",
                "k_Fx", "k_Fy", "k_Fz", "k_Tx", "k_Ty", "k_Tz",
            ],
            dtype=h5py.string_dtype("utf-8"),
        )
        wrench_group.create_dataset(
            "combined",
            data=arrays["wrench"],
            dtype=np.float32,
        )
        wrench_group.create_dataset(
            "o_f_ext_hat_k",
            data=arrays["wrench_o_f_ext_hat_k"],
            dtype=np.float32,
        )
        wrench_group.create_dataset(
            "k_f_ext_hat_k",
            data=arrays["wrench_k_f_ext_hat_k"],
            dtype=np.float32,
        )

        f.create_dataset("action", data=arrays["combined"], dtype=np.float32)

    log(f"[INFO] episode {ep.idx}: HDF5 write done")


def resolve_recording_collection_dir(collection_dir: str) -> List[Tuple[str, Optional[str], str]]:
    collection_dir = os.path.abspath(os.path.expanduser(collection_dir))

    if not os.path.isdir(collection_dir):
        raise RuntimeError(
            f"recording_collection_dir does not exist or is not a directory: {collection_dir}"
        )

    child_dirs = [
        os.path.join(collection_dir, name)
        for name in sorted(os.listdir(collection_dir))
        if os.path.isdir(os.path.join(collection_dir, name))
    ]

    if not child_dirs:
        raise RuntimeError(
            f"No recording directories found in recording_collection_dir: {collection_dir}"
        )

    resolved = []
    for child in child_dirs:
        try:
            resolved.append(resolve_recording_dir(child))
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve child recording directory '{child}': {e}"
            ) from e

    return resolved


def process_one_recording(
    bag_path: str,
    storage_id: str,
    raw_events_h5: Optional[str],
    output_parent_dir: str,
    output_name: str,
    max_episodes: Optional[int],
    event_frame_mode: str,
    event_frame_windows_ms,
    event_frame_contrast: float,
    event_frame_step: float,
    event_packet_margin_ms: float,
    initial_delay_steps: int,
) -> List[int]:
    out_dir = os.path.join(output_parent_dir, output_name)
    ensure_dir(out_dir)
    log(f"[INFO] Output directory: {out_dir}")
    log(f"[INFO] rosbag2 storage_id: {storage_id}")
    log(f"[INFO] initial_delay_steps: {initial_delay_steps} ({initial_delay_steps / FPS:.3f}s)")
    log(f"[INFO] event_frame_windows_ms/time_shifts: {event_frame_windows_ms}")

    log("[INFO] Opening bag for metadata check...")
    reader = open_reader(bag_path, storage_id=storage_id)
    topic_type_map = get_topic_type_map(reader)
    check_required_topics(
        topic_type_map,
        require_event_topic=(raw_events_h5 is None),
    )

    windows = extract_episode_windows(bag_path, storage_id=storage_id)
    filtered_windows = []
    for ep in windows:
        duration = ep.end - ep.start
        if duration < MIN_DURATION:
            log(f"[INFO] dropping episode {ep.idx}: too short ({duration:.3f}s)")
        else:
            filtered_windows.append(EpisodeWindow(
                idx=len(filtered_windows),
                start=ep.start,
                end=ep.end,
            ))

    log(f"[INFO] kept {len(filtered_windows)} / {len(windows)} episodes after filtering")
    windows = filtered_windows

    if max_episodes is not None:
        windows = windows[:max_episodes]
        log(f"[INFO] max_episodes applied -> processing first {len(windows)} episodes")

    raw_event_store = RawEventStore(raw_events_h5) if raw_events_h5 else None

    if raw_event_store is not None and windows:
        selected_start = min(ep.start for ep in windows)
        selected_end = max(ep.end for ep in windows)
        log(
            "[INFO] selected episode time range (sec): "
            f"{selected_start:.6f} .. {selected_end:.6f}"
        )
        if (
            raw_event_store.packet_ros_start_ns is None
            or raw_event_store.packet_ros_end_ns is None
        ):
            raise RuntimeError(
                "Raw event HDF5 has no packet ROS timestamps. Cannot construct 3chef."
            )

        raw_start_sec = raw_event_store.packet_ros_start_ns * 1e-9
        raw_end_sec = raw_event_store.packet_ros_end_ns * 1e-9
        if selected_end < raw_start_sec or selected_start > raw_end_sec:
            raise RuntimeError(
                "Raw event HDF5 packet ROS timestamps do not overlap selected "
                "episode windows. Cannot construct 3chef."
            )

    lengths: List[int] = []

    try:
        for ep in windows:
            try:
                data = collect_single_episode_data(
                    bag_path,
                    ep,
                    require_event_topic=(raw_event_store is None),
                    storage_id=storage_id,
                )
                arrays = sample_episode_to_arrays(
                    data,
                    ep,
                    raw_event_store=raw_event_store,
                    event_frame_windows_ms=tuple(event_frame_windows_ms),
                    event_frame_mode=event_frame_mode,
                    event_frame_contrast=event_frame_contrast,
                    event_frame_step=event_frame_step,
                    event_packet_margin_ms=event_packet_margin_ms,
                    initial_delay_steps=initial_delay_steps,
                )

                out_path = os.path.join(out_dir, f"episode_{ep.idx}.hdf5")
                event_representation = (
                    "3chef_raw_events" if raw_event_store is not None
                    else "fallback_repeated_mono_event_image"
                )

                write_episode_hdf5(
                    out_path,
                    arrays,
                    data["joint_names"],
                    ep,
                    event_representation=event_representation,
                    event_frame_windows_ms=event_frame_windows_ms,
                    event_frame_mode=event_frame_mode,
                    event_frame_contrast=event_frame_contrast,
                    event_frame_step=event_frame_step,
                    raw_events_h5=raw_events_h5,
                    initial_delay_steps=initial_delay_steps,
                )

                lengths.append(len(arrays["timestamps"]))

                del data
                del arrays
                gc.collect()

                log(f"[INFO] episode {ep.idx}: finished successfully")
                log("")

            except Exception as e:
                log(f"[ERROR] episode {ep.idx} failed: {repr(e)}")
                raise
    finally:
        if raw_event_store is not None:
            raw_event_store.close()

    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording_collection_dir",
        type=str,
        default=None,
        help="Optional parent directory whose direct children are recording directories. Processes all child recordings.",
    )
    parser.add_argument(
        "--recording_dir",
        type=str,
        default=None,
        help="Optional recording directory containing one *_bag directory and one *_raw_events.h5 file.",
    )
    parser.add_argument(
        "--bag",
        type=str,
        default=None,
        help=(
            "Path to rosbag2 bag directory. Direct .mcap or .db3 file paths may also work "
            "depending on rosbag2 storage backend. Required unless --recording_dir is provided."
        ),
    )
    parser.add_argument(
        "--storage_id",
        type=str,
        default="mcap",
        choices=["mcap", "sqlite3"],
        help="rosbag2 storage backend to use. Default: mcap. Use sqlite3 for legacy .db3 bags.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Parent directory to write HDF5 episodes into (a bag-named subdirectory is created)",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional limit for debugging",
    )
    parser.add_argument(
        "--initial_delay_steps",
        type=int,
        default=0,
        help=(
            "Number of sampled timesteps/frames to drop from the beginning of each episode "
            "after sampling. At 30 Hz, 30 steps = 1 second."
        ),
    )
    parser.add_argument(
        "--raw_events_h5",
        type=str,
        default=None,
        help="Optional OpenMV raw event HDF5. If provided, construct 3chef from raw events instead of using /openmv_cam/image.",
    )
    parser.add_argument(
        "--event_frame_mode",
        type=str,
        default="cumulative",
        choices=["cumulative", "shifted"],
        help="3chef construction mode. Default cumulative: ch0=[t-50ms,t], ch1=[t-250ms,t], ch2=[t-1000ms,t].",
    )
    parser.add_argument(
        "--event_frame_windows_ms",
        type=float,
        nargs=3,
        default=[50.0, 250.0, 1000.0],
        help="Three time windows in ms for 3chef channels.",
    )
    parser.add_argument(
        "--time_shifts",
        type=float,
        nargs=3,
        default=None,
        metavar=("CH0_MS", "CH1_MS", "CH2_MS"),
        help=(
            "Alias for --event_frame_windows_ms. Three 3chef cumulative/shifted horizons in ms, "
            "e.g. --time_shifts 50 250 1500."
        ),
    )
    parser.add_argument(
        "--event_frame_contrast",
        type=float,
        default=4.0,
        help="Contrast multiplier for event frame rendering.",
    )
    parser.add_argument(
        "--event_frame_step",
        type=float,
        default=1.0,
        help="Per-event accumulation step.",
    )
    parser.add_argument(
        "--event_packet_margin_ms",
        type=float,
        default=50.0,
        help="Extra raw-event packet margin before the largest window.",
    )
    args = parser.parse_args()

    # --- validate and resolve initial_delay_steps ---
    if args.initial_delay_steps < 0:
        parser.error("--initial_delay_steps must be >= 0")

    # --- resolve event frame windows: --time_shifts overrides --event_frame_windows_ms ---
    if args.time_shifts is not None:
        args.event_frame_windows_ms = args.time_shifts

    if len(args.event_frame_windows_ms) != 3:
        parser.error("event frame windows/time shifts must contain exactly 3 values")

    if any(w <= 0 for w in args.event_frame_windows_ms):
        parser.error("event frame windows/time shifts must all be positive")

    if any(args.event_frame_windows_ms[i] > args.event_frame_windows_ms[i + 1] for i in range(2)):
        parser.error("event frame windows/time shifts must be non-decreasing, e.g. 50 250 1500")

    # --- mutually exclusive input mode validation ---
    input_modes = [
        args.bag is not None,
        args.recording_dir is not None,
        args.recording_collection_dir is not None,
    ]
    if sum(input_modes) != 1:
        parser.error(
            "Exactly one input mode must be used: --bag, --recording_dir, or --recording_collection_dir."
        )

    if args.recording_dir is not None and args.raw_events_h5 is not None:
        parser.error(
            "--raw_events_h5 cannot be used with --recording_dir; it is auto-detected."
        )
    if args.recording_collection_dir is not None and args.raw_events_h5 is not None:
        parser.error(
            "--raw_events_h5 cannot be used with --recording_collection_dir; "
            "raw event H5 files are auto-detected per recording."
        )

    # --- build recordings list ---
    if args.bag is not None:
        recordings = [{
            "bag_path": args.bag,
            "raw_events_h5": args.raw_events_h5,
            "output_name": bag_top_level_name(args.bag),
        }]

    elif args.recording_dir is not None:
        bag_path, raw_events_h5, recording_name = resolve_recording_dir(args.recording_dir)
        log(f"[INFO] recording_dir: {args.recording_dir}")
        log(f"[INFO] resolved bag path: {bag_path}")
        if raw_events_h5:
            log(f"[INFO] resolved raw events H5: {raw_events_h5}")
        else:
            log("[WARNING] no raw events H5 resolved; using fallback event topic")
        log(f"[INFO] output recording name: {recording_name}")
        recordings = [{
            "bag_path": bag_path,
            "raw_events_h5": raw_events_h5,
            "output_name": recording_name,
        }]

    else:  # args.recording_collection_dir is not None
        resolved = resolve_recording_collection_dir(args.recording_collection_dir)
        recordings = [
            {
                "bag_path": bp,
                "raw_events_h5": reh5,
                "output_name": rname,
            }
            for bp, reh5, rname in resolved
        ]

    # --- process each recording ---
    log(f"[INFO] rosbag2 storage_id: {args.storage_id}")
    all_lengths: Dict[str, List[int]] = {}
    for i, rec in enumerate(recordings):
        log("")
        log("=" * 80)
        log(f"[INFO] Processing recording {i + 1}/{len(recordings)}: {rec['output_name']}")
        log(f"[INFO] bag_path: {rec['bag_path']}")
        log(f"[INFO] storage_id: {args.storage_id}")
        if rec["raw_events_h5"]:
            log(f"[INFO] raw_events_h5: {rec['raw_events_h5']}")
        else:
            log("[WARNING] raw_events_h5: none; using fallback /openmv_cam/image")
        log("=" * 80)

        try:
            lengths = process_one_recording(
                bag_path=rec["bag_path"],
                storage_id=args.storage_id,
                raw_events_h5=rec["raw_events_h5"],
                output_parent_dir=args.out_dir,
                output_name=rec["output_name"],
                max_episodes=args.max_episodes,
                event_frame_mode=args.event_frame_mode,
                event_frame_windows_ms=args.event_frame_windows_ms,
                event_frame_contrast=args.event_frame_contrast,
                event_frame_step=args.event_frame_step,
                event_packet_margin_ms=args.event_packet_margin_ms,
                initial_delay_steps=args.initial_delay_steps,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed while processing recording '{rec['output_name']}': {e}"
            ) from e

        all_lengths[rec["output_name"]] = lengths

    # --- collection summary ---
    log("")
    log("[INFO] Processing summary")
    total_episodes = 0
    for name, lengths in all_lengths.items():
        total_episodes += len(lengths)
        if lengths:
            log(
                f"       {name}: episodes={len(lengths)}, "
                f"min={min(lengths)}, mean={np.mean(lengths):.2f}, max={max(lengths)}"
            )
        else:
            log(f"       {name}: episodes=0")
    log(f"[INFO] total recordings: {len(all_lengths)}")
    log(f"[INFO] total episodes:   {total_episodes}")


if __name__ == "__main__":
    main()