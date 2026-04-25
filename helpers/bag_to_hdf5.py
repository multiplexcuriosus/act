#!/usr/bin/env python3

import os
import gc
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
import cv2
import h5py
import numpy as np
from tqdm import tqdm

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPIC_RGB = "/camera/camera/color/image_raw"
TOPIC_EVENT = "/openmv_cam/image"
TOPIC_JOINT = "/joint_states"
TOPIC_GRIPPER_STATE = "/teleop/gripper_state_cmd"
TOPIC_TWIST = "/cartesian_cmd/twist"
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
    - If bag_path is a file (e.g. *.db3): use filename without extension.
    """
    norm = os.path.normpath(bag_path)
    if os.path.isdir(norm):
        return os.path.basename(norm)
    return os.path.splitext(os.path.basename(norm))[0]


def open_reader(bag_path: str):
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3",
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


def get_type_class_map(topic_type_map: Dict[str, str]) -> Dict[str, Any]:
    return {topic: get_message(msg_type) for topic, msg_type in topic_type_map.items()}


def check_required_topics(topic_type_map: Dict[str, str]):
    required = {
        TOPIC_RGB,
        TOPIC_EVENT,
        TOPIC_JOINT,
        TOPIC_GRIPPER_STATE,
        TOPIC_TWIST,
        TOPIC_EPISODE,
    }
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


def twist_msg_to_vec(msg) -> np.ndarray:
    return np.array([
        msg.twist.linear.x,
        msg.twist.linear.y,
        msg.twist.linear.z,
        msg.twist.angular.x,
        msg.twist.angular.y,
        msg.twist.angular.z,
    ], dtype=np.float32)


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


def extract_episode_windows(bag_path: str) -> List[EpisodeWindow]:
    log("[INFO] Pass 1/2: scanning /episode/control for episode boundaries...")
    reader = open_reader(bag_path)
    topic_type_map = get_topic_type_map(reader)
    msg_types = get_type_class_map(topic_type_map)

    starts = []
    ends = []

    n_total = 0
    n_episode_msgs = 0

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
            starts.append(t)
        elif msg.data == 2:
            ends.append(t)

    log(f"[INFO] pass1 done. total messages read: {n_total}")
    log(f"[INFO] episode control messages: {n_episode_msgs}")
    log(f"[INFO] starts found: {len(starts)}")
    log(f"[INFO] ends found:   {len(ends)}")

    # Pair each start with the next end after it.
    windows = []
    end_idx = 0
    last_end = -np.inf

    for i, s in enumerate(starts):
        if s <= last_end:
            log(f"[WARNING] skipping start {i} because it is before previous end")
            continue

        while end_idx < len(ends) and ends[end_idx] <= s:
            end_idx += 1

        if end_idx >= len(ends):
            log(f"[WARNING] no matching end found for start at {s:.6f}")
            break

        e = ends[end_idx]
        windows.append(EpisodeWindow(idx=len(windows), start=s, end=e))
        log(f"[INFO] episode {len(windows)-1}: start={s:.6f}, end={e:.6f}, dur={e-s:.3f}s")
        last_end = e
        end_idx += 1

    if not windows:
        raise RuntimeError("No valid episode windows found.")

    return windows


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
) -> Dict[str, Any]:
    """
    Stream only one episode into memory.
    This is the key RAM fix.
    """
    log("")
    log(f"[INFO] Pass 2/2: collecting episode {ep.idx}")
    log(f"[INFO] time window: [{ep.start:.6f}, {ep.end:.6f}]  dur={ep.end - ep.start:.3f}s")

    reader = open_reader(bag_path)
    topic_type_map = get_topic_type_map(reader)
    msg_types = get_type_class_map(topic_type_map)

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
        "joint_names": None,
    }

    msg_count = 0
    kept_count = 0

    # Track whether topics already passed episode end in time-ordered bag.
    seen_after_end = {
        TOPIC_RGB: False,
        TOPIC_EVENT: False,
        TOPIC_JOINT: False,
        TOPIC_GRIPPER_STATE: False,
        TOPIC_TWIST: False,
    }

    tracked_topics = set(seen_after_end.keys())

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

    log(f"[INFO] episode {ep.idx}: scan finished")
    log(f"       scanned bag messages: {msg_count}")
    log(f"       kept messages:        {kept_count}")
    log(f"       rgb frames:           {len(data['rgb_t'])}")
    log(f"       event frames:         {len(data['event_t'])}")
    log(f"       joint msgs:           {len(data['joint_t'])}")
    log(f"       gripper state msgs:   {len(data['gripper_state_t'])}")
    log(f"       twist msgs:           {len(data['twist_t'])}")

    for k in ["rgb_t", "event_t", "joint_t", "twist_t"]:
        if len(data[k]) == 0:
            raise RuntimeError(f"Episode {ep.idx}: no data collected for {k}")

    return data


def sample_episode_to_arrays(data: Dict[str, Any], ep: EpisodeWindow) -> Dict[str, np.ndarray]:
    log(f"[INFO] episode {ep.idx}: sampling onto {FPS:.1f} Hz grid using next-available datapoint")

    grid = np.arange(ep.start, ep.end + 1e-9, DT, dtype=np.float64)
    log(f"[INFO] episode {ep.idx}: grid has {len(grid)} steps")

    rgb_t = data["rgb_t"]
    event_t = data["event_t"]
    joint_t = data["joint_t"]
    gripper_state_t = data["gripper_state_t"]
    twist_t = data["twist_t"]

    rgb_msg = data["rgb_msg"]
    event_msg = data["event_msg"]
    joint_pos = data["joint_pos"]
    gripper_state = data["gripper_state"]
    twist = data["twist"]
    joint_names = data["joint_names"]

    if joint_names is None:
        raise RuntimeError(f"Episode {ep.idx}: joint_names not set from /joint_states")

    rgb_idx = first_index_ge(rgb_t, ep.start)
    event_idx = first_index_ge(event_t, ep.start)
    joint_idx = first_index_ge(joint_t, ep.start)
    gripper_state_idx = last_index_le(gripper_state_t, ep.start)
    twist_idx = first_index_ge(twist_t, ep.start)

    initial_gripper_state = np.float32(0.0)  # open
    log(f"[INFO] episode {ep.idx}: sparse gripper state event count = {len(gripper_state_t)}")

    rgb_frames = []
    event_frames = []
    qpos_seq = []
    gripper_seq = []
    twist_seq = []
    for i, t in enumerate(grid):
        rgb_idx = first_index_ge(rgb_t, t, rgb_idx)
        event_idx = first_index_ge(event_t, t, event_idx)
        joint_idx = first_index_ge(joint_t, t, joint_idx)
        gripper_state_idx = last_index_le(gripper_state_t, t, max(0, gripper_state_idx + 1))
        twist_idx = first_index_ge(twist_t, t, twist_idx)

        if i % 100 == 0:
            log(
                f"[DEBUG] episode {ep.idx}: sample {i:04d}/{len(grid)} "
                f"| rgb_idx={rgb_idx} event_idx={event_idx} joint_idx={joint_idx} "
                f"gripper_state_idx={gripper_state_idx} twist_idx={twist_idx}"
            )

        rgb_np = image_msg_to_numpy(rgb_msg[rgb_idx])
        event_np = image_msg_to_numpy(event_msg[event_idx])

        rgb_frames.append(rgb_np)
        event_frames.append(event_np)
        qpos_seq.append(build_qpos_from_joint_state(joint_names, joint_pos[joint_idx]))
        if gripper_state_idx >= 0:
            gripper_value = np.float32(gripper_state[gripper_state_idx][0])
        else:
            gripper_value = initial_gripper_state

        gripper_seq.append(np.array([gripper_value], dtype=np.float32))
        twist_seq.append(twist[twist_idx])

    rgb_frames = np.stack(rgb_frames, axis=0)
    event_frames = np.stack(event_frames, axis=0)
    qpos_seq = np.stack(qpos_seq, axis=0).astype(np.float32)
    gripper_seq = np.stack(gripper_seq, axis=0).astype(np.float32)
    twist_seq = np.stack(twist_seq, axis=0).astype(np.float32)
    action_combined = np.concatenate([twist_seq, gripper_seq], axis=1)

    log(f"[INFO] episode {ep.idx}: sampling done")
    log(f"       rgb shape:        {rgb_frames.shape}, dtype={rgb_frames.dtype}")
    log(f"       event shape:      {event_frames.shape}, dtype={event_frames.dtype}")
    log(f"       qpos shape:       {qpos_seq.shape}, dtype={qpos_seq.dtype}")
    log(f"       twist shape:      {twist_seq.shape}, dtype={twist_seq.dtype}")
    log(f"       gripper shape:    {gripper_seq.shape}, dtype={gripper_seq.dtype}")
    preview_n = min(5, len(gripper_seq))
    log(f"       dense gripper first {preview_n} values: {gripper_seq[:preview_n, 0].tolist()}")
    unique_vals, unique_counts = np.unique(gripper_seq[:, 0], return_counts=True)
    counts_str = ", ".join([f"{float(v):.1f}:{int(c)}" for v, c in zip(unique_vals, unique_counts)])
    log(f"       dense gripper value counts: {counts_str}")
    log(f"       combined action:  {action_combined.shape}, dtype={action_combined.dtype}")
    log(f"       expected qpos dim/action dim: 8/7")

    return {
        "timestamps": grid.astype(np.float64),
        "rgb": rgb_frames,
        "event": event_frames,
        "qpos": qpos_seq,
        "twist": twist_seq,
        "gripper": gripper_seq,
        "combined": action_combined,
    }


def write_episode_hdf5(
    out_path: str,
    arrays: Dict[str, np.ndarray],
    joint_names: List[str],
    ep: EpisodeWindow,
):
    log(f"[INFO] episode {ep.idx}: writing HDF5 -> {out_path}")

    with h5py.File(out_path, "w") as f:
        f.attrs["sim"] = False
        f.attrs["fps"] = FPS
        f.attrs["episode_index"] = ep.idx
        f.attrs["episode_start"] = ep.start
        f.attrs["episode_end"] = ep.end
        f.attrs["joint_names"] = np.array(joint_names, dtype=h5py.string_dtype("utf-8"))

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

        f.create_dataset("action", data=arrays["combined"], dtype=np.float32)

    log(f"[INFO] episode {ep.idx}: HDF5 write done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag",
        type=str,
        required=True,
        help="Path to rosbag2 bag directory OR sqlite file path",
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
    args = parser.parse_args()

    bag_path = args.bag
    out_dir = os.path.join(args.out_dir, bag_top_level_name(bag_path))

    ensure_dir(out_dir)
    log(f"[INFO] Output directory: {out_dir}")

    log("[INFO] Opening bag for metadata check...")
    reader = open_reader(bag_path)
    topic_type_map = get_topic_type_map(reader)
    check_required_topics(topic_type_map)

    windows = extract_episode_windows(bag_path)
    filtered_windows = []
    for ep in windows:
        duration = ep.end - ep.start
        if duration < MIN_DURATION:
            log(f"[INFO] dropping episode {ep.idx}: too short ({duration:.3f}s)")
        else:
            filtered_windows.append(ep)

    log(f"[INFO] kept {len(filtered_windows)} / {len(windows)} episodes after filtering")

    windows = filtered_windows


    if args.max_episodes is not None:
        windows = windows[:args.max_episodes]
        log(f"[INFO] max_episodes applied -> processing first {len(windows)} episodes")

    lengths = []

    for ep in windows:
        try:
            data = collect_single_episode_data(bag_path, ep)
            arrays = sample_episode_to_arrays(data, ep)

            out_path = os.path.join(out_dir, f"episode_{ep.idx}.hdf5")
            write_episode_hdf5(out_path, arrays, data["joint_names"], ep)

            lengths.append(len(arrays["timestamps"]))

            # free memory aggressively
            del data
            del arrays
            gc.collect()

            log(f"[INFO] episode {ep.idx}: finished successfully")
            log("")

        except Exception as e:
            log(f"[ERROR] episode {ep.idx} failed: {repr(e)}")
            raise

    if lengths:
        log("[INFO] All requested episodes done")
        log(f"[INFO] num episodes: {len(lengths)}")
        log(f"[INFO] min length:   {min(lengths)}")
        log(f"[INFO] mean length:  {np.mean(lengths):.2f}")
        log(f"[INFO] max length:   {max(lengths)}")


if __name__ == "__main__":
    main()