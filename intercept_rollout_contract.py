import bisect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

INTERCEPT_HISTORY_OFFSETS = (-6, -3, 0)
ARM_JOINT_NAMES = (
    "right_fr3_joint1",
    "right_fr3_joint2",
    "right_fr3_joint3",
    "right_fr3_joint4",
    "right_fr3_joint5",
    "right_fr3_joint6",
    "right_fr3_joint7",
)


EXPECTED_INTERCEPT_METADATA = {
    "data_mode": "intercept",
    "raw_qpos_dim": 7,
    "state_dim": 21,
    "action_dim": 1,
    "rgb_history_frames": 3,
    "rgb_history_offsets": list(INTERCEPT_HISTORY_OFFSETS),
    "qpos_history_offsets": list(INTERCEPT_HISTORY_OFFSETS),
    "rgb_frame_order": "oldest_to_newest",
    "qpos_flatten_order": "oldest_to_newest",
    "image_channels": 9,
    "action_type": "measured_tcp_s_delta",
    "action_representation": "future_delta_relative_to_anchor",
    "action_anchor_offset": 0,
    "action_first_target_offset": 1,
    "action_positive_direction": "robot_base_positive_x",
    "action_units": "m",
}


@dataclass(frozen=True)
class SyncSelection:
    history_indices: Tuple[int, int, int]
    rgb_timestamps: Tuple[float, float, float]
    qpos_timestamps: Tuple[float, float, float]
    anchor_tcp_s: float
    anchor_tcp_s_timestamp: float
    qpos_history: np.ndarray


class TemporalAbsoluteAggregator:
    """Aggregate per-step absolute predictions from chunked receding-horizon outputs."""

    def __init__(self, chunk_size: int, decay: float = 0.01) -> None:
        self.chunk_size = int(chunk_size)
        self.decay = float(decay)
        self._history: List[Tuple[int, np.ndarray]] = []

    def reset(self) -> None:
        self._history.clear()

    def add_prediction(self, step_index: int, absolute_chunk: np.ndarray) -> None:
        chunk = np.asarray(absolute_chunk, dtype=np.float32).reshape(-1)
        if chunk.shape != (self.chunk_size,):
            raise ValueError(
                f"absolute_chunk must have shape ({self.chunk_size},), got {chunk.shape}"
            )
        self._history.append((int(step_index), chunk))

        min_step = int(step_index) - self.chunk_size
        self._history = [item for item in self._history if item[0] >= min_step]

    def value_for_step(self, current_step: int) -> Optional[float]:
        values: List[float] = []
        for source_step, chunk in self._history:
            token_index = int(current_step) - int(source_step)
            if 0 <= token_index < self.chunk_size:
                values.append(float(chunk[token_index]))

        if not values:
            return None

        weights = np.exp(-self.decay * np.arange(len(values), dtype=np.float64))
        weights = weights / np.sum(weights)
        return float(np.sum(np.asarray(values, dtype=np.float64) * weights))


def compute_history_indices(anchor_index: int, history_offsets: Sequence[int]) -> List[int]:
    anchor_index = int(anchor_index)
    indices = [anchor_index + int(offset) for offset in history_offsets]
    if any(index < 0 for index in indices):
        raise ValueError(
            f"Not enough history for anchor_index={anchor_index} and offsets={tuple(history_offsets)}"
        )
    return indices


def select_latest_index_at_or_before(
    timestamps: Sequence[float],
    target_timestamp: float,
) -> Optional[int]:
    if not timestamps:
        return None
    idx = bisect.bisect_right(timestamps, float(target_timestamp)) - 1
    if idx < 0:
        return None
    return idx


def extract_arm_qpos(joint_names: Sequence[str], joint_positions: Sequence[float]) -> np.ndarray:
    name_to_idx = {name: index for index, name in enumerate(joint_names)}
    missing = [name for name in ARM_JOINT_NAMES if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing required FR3 arm joints: {missing}")

    return np.asarray(
        [joint_positions[name_to_idx[name]] for name in ARM_JOINT_NAMES],
        dtype=np.float32,
    )


def build_qpos_history(qpos_samples: Sequence[np.ndarray]) -> np.ndarray:
    if len(qpos_samples) != 3:
        raise ValueError(f"Expected exactly 3 qpos samples, got {len(qpos_samples)}")

    flattened = []
    for sample in qpos_samples:
        qpos = np.asarray(sample, dtype=np.float32).reshape(-1)
        if qpos.shape != (7,):
            raise ValueError(f"Each qpos sample must be shape (7,), got {qpos.shape}")
        flattened.append(qpos)

    out = np.concatenate(flattened, axis=0)
    if out.shape != (21,):
        raise AssertionError(f"qpos history shape mismatch: expected (21,), got {out.shape}")
    return out


def build_rgb_history_tensor(rgb_frames: Sequence[np.ndarray], image_size: int) -> np.ndarray:
    if len(rgb_frames) != 3:
        raise ValueError(f"Expected 3 RGB frames for temporal history, got {len(rgb_frames)}")

    target = int(image_size)
    if target <= 0:
        raise ValueError(f"image_size must be positive, got {target}")

    processed = []
    for frame in rgb_frames:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Each RGB frame must be HWC3, got {arr.shape}")
        resized = cv2.resize(arr, (target, target), interpolation=cv2.INTER_AREA)
        processed.append(resized)

    image_hwc = np.concatenate(processed, axis=2)
    if image_hwc.shape != (target, target, 9):
        raise AssertionError(
            f"Temporal RGB concat mismatch: expected {(target, target, 9)}, got {image_hwc.shape}"
        )

    image = np.transpose(image_hwc, (2, 0, 1))[None, None, ...].astype(np.float32) / 255.0
    if image.shape != (1, 1, 9, target, target):
        raise AssertionError(
            "Temporal RGB tensor shape mismatch: "
            f"expected {(1, 1, 9, target, target)}, got {image.shape}"
        )
    return image


def validate_intercept_stats_and_config(
    stats: Dict[str, object],
    policy_config: Dict[str, object],
    expected_chunk_size: int,
) -> Dict[str, np.ndarray]:
    for key, expected in EXPECTED_INTERCEPT_METADATA.items():
        if key not in stats:
            raise ValueError(f"Missing interception checkpoint metadata in dataset_stats.pkl: {key}")
        if stats[key] != expected:
            raise ValueError(
                f"Interception checkpoint metadata mismatch for {key}: "
                f"expected {expected!r}, found {stats[key]!r}"
            )

    required_config = {
        "state_dim": 21,
        "action_dim": 1,
        "num_queries": int(expected_chunk_size),
        "rgb_history_frames": 3,
        "image_channels": 9,
    }
    for key, expected in required_config.items():
        if int(policy_config.get(key, -1)) != expected:
            raise ValueError(
                f"Interception policy config mismatch for {key}: "
                f"expected {expected}, found {policy_config.get(key)}"
            )

    if bool(policy_config.get("use_bce_last_action_dim", False)):
        raise ValueError("Interception rollout requires use_bce_last_action_dim=false")

    arrays: Dict[str, np.ndarray] = {}
    for stat_key, expected_shape in (
        ("qpos_mean", (21,)),
        ("qpos_std", (21,)),
        ("action_mean", (1,)),
        ("action_std", (1,)),
    ):
        if stat_key not in stats:
            raise ValueError(f"Missing required stats key: {stat_key}")
        arr = np.asarray(stats[stat_key], dtype=np.float32).reshape(-1)
        if arr.shape != expected_shape:
            raise ValueError(
                f"Stats shape mismatch for {stat_key}: expected {expected_shape}, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Stats {stat_key} contains non-finite values")
        arrays[stat_key] = arr

    if np.any(arrays["qpos_std"] <= 0.0):
        raise ValueError("qpos_std must be strictly positive")
    if np.any(arrays["action_std"] <= 0.0):
        raise ValueError("action_std must be strictly positive")

    return arrays


def denormalize_delta_chunk(
    normalized_chunk: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    chunk = np.asarray(normalized_chunk, dtype=np.float32).reshape(-1)
    mean = np.asarray(action_mean, dtype=np.float32).reshape(-1)
    std = np.asarray(action_std, dtype=np.float32).reshape(-1)

    if mean.shape != (1,) or std.shape != (1,):
        raise ValueError(
            f"Expected scalar action stats shaped (1,), got mean={mean.shape}, std={std.shape}"
        )

    delta = chunk * std[0] + mean[0]
    if not np.all(np.isfinite(delta)):
        raise ValueError("Denormalized delta chunk contains non-finite values")
    return delta.astype(np.float32)


def absolute_s_from_anchor(anchor_s: float, delta_chunk: np.ndarray) -> np.ndarray:
    anchor = float(anchor_s)
    if not math.isfinite(anchor):
        raise ValueError(f"anchor_s must be finite, got {anchor_s}")

    delta = np.asarray(delta_chunk, dtype=np.float32).reshape(-1)
    out = anchor + delta
    if not np.all(np.isfinite(out)):
        raise ValueError("Absolute-s chunk contains non-finite values")
    return out.astype(np.float32)


def select_sync_observation(
    rgb_timestamps: Sequence[float],
    joint_timestamps: Sequence[float],
    joint_qpos_samples: Sequence[np.ndarray],
    tcp_timestamps: Sequence[float],
    tcp_values: Sequence[float],
    history_offsets: Sequence[int] = INTERCEPT_HISTORY_OFFSETS,
) -> SyncSelection:
    if not rgb_timestamps:
        raise ValueError("No RGB samples available")
    if len(joint_timestamps) != len(joint_qpos_samples):
        raise ValueError("joint_timestamps and joint_qpos_samples length mismatch")
    if len(tcp_timestamps) != len(tcp_values):
        raise ValueError("tcp_timestamps and tcp_values length mismatch")

    anchor_index = len(rgb_timestamps) - 1
    history_indices = compute_history_indices(anchor_index, history_offsets)

    selected_rgb_ts = [float(rgb_timestamps[idx]) for idx in history_indices]

    selected_qpos = []
    selected_qpos_ts = []
    for rgb_ts in selected_rgb_ts:
        joint_idx = select_latest_index_at_or_before(joint_timestamps, rgb_ts)
        if joint_idx is None:
            raise ValueError(
                "No causal JointState sample available at or before RGB timestamp "
                f"{rgb_ts:.6f}"
            )
        selected_qpos_ts.append(float(joint_timestamps[joint_idx]))
        selected_qpos.append(joint_qpos_samples[joint_idx])

    qpos_history = build_qpos_history(selected_qpos)

    anchor_ts = selected_rgb_ts[-1]
    tcp_idx = select_latest_index_at_or_before(tcp_timestamps, anchor_ts)
    if tcp_idx is None:
        raise ValueError(
            "No causal current_tcp_s sample available at or before RGB anchor timestamp "
            f"{anchor_ts:.6f}"
        )

    anchor_tcp_s = float(tcp_values[tcp_idx])
    anchor_tcp_ts = float(tcp_timestamps[tcp_idx])

    if not math.isfinite(anchor_tcp_s):
        raise ValueError(f"Anchor current_tcp_s is non-finite: {anchor_tcp_s}")

    return SyncSelection(
        history_indices=tuple(history_indices),
        rgb_timestamps=tuple(selected_rgb_ts),
        qpos_timestamps=tuple(selected_qpos_ts),
        anchor_tcp_s=anchor_tcp_s,
        anchor_tcp_s_timestamp=anchor_tcp_ts,
        qpos_history=qpos_history,
    )


def validate_anchor_freshness(
    anchor_timestamp: float,
    observation_timestamp: float,
    now_timestamp: float,
    max_anchor_age_sec: float,
    max_observation_age_sec: float,
) -> None:
    anchor_age = float(observation_timestamp) - float(anchor_timestamp)
    obs_age = float(now_timestamp) - float(observation_timestamp)

    if anchor_age < 0.0:
        raise ValueError(
            "Anchor timestamp is newer than observation timestamp; causal sync violated"
        )
    if max_anchor_age_sec > 0.0 and anchor_age > float(max_anchor_age_sec):
        raise ValueError(
            f"Anchor current_tcp_s is stale: age={anchor_age:.6f}s exceeds max {max_anchor_age_sec:.6f}s"
        )
    if max_observation_age_sec > 0.0 and obs_age > float(max_observation_age_sec):
        raise ValueError(
            f"Observation is stale: age={obs_age:.6f}s exceeds max {max_observation_age_sec:.6f}s"
        )
