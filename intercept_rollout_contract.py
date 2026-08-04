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


EXPECTED_INTERCEPT_COMMON_METADATA = {
    "data_mode": "intercept",
    "raw_qpos_dim": 7,
    "state_dim": 21,
    "action_dim": 1,
    "qpos_history_offsets": list(INTERCEPT_HISTORY_OFFSETS),
    "qpos_flatten_order": "oldest_to_newest",
    "image_channels": 9,
    "action_type": "measured_tcp_s_delta",
    "action_representation": "future_delta_relative_to_anchor",
    "action_anchor_offset": 0,
    "action_first_target_offset": 1,
    "action_positive_direction": "robot_base_positive_x",
    "action_units": "m",
}

EXPECTED_INTERCEPT_RGB_METADATA = {
    **EXPECTED_INTERCEPT_COMMON_METADATA,
    "input_modality": "rgb",
    "camera_names": ["rgb"],
    "visual_history_frames": 3,
    "visual_history_offsets": list(INTERCEPT_HISTORY_OFFSETS),
    "channels_per_visual_frame": 3,
    "visual_frame_order": "oldest_to_newest",
    "image_normalization": "imagenet",
    "rgb_history_frames": 3,
    "rgb_history_offsets": list(INTERCEPT_HISTORY_OFFSETS),
    "rgb_frame_order": "oldest_to_newest",
}

EXPECTED_INTERCEPT_EVENT_METADATA = {
    **EXPECTED_INTERCEPT_COMMON_METADATA,
    "input_modality": "event",
    "camera_names": ["event"],
    "visual_history_frames": 3,
    "visual_history_offsets": list(INTERCEPT_HISTORY_OFFSETS),
    "channels_per_visual_frame": 3,
    "visual_frame_order": "oldest_to_newest",
    "image_normalization": "shifted_3chef_centered",
    "rgb_history_frames": 3,
    "rgb_history_offsets": list(INTERCEPT_HISTORY_OFFSETS),
    "rgb_frame_order": "oldest_to_newest",
    "event_representation": "shifted_3chef_signed",
    "event_frame_mode": "shifted",
    "event_frame_windows_ms": [50.0, 100.0, 200.0],
    "event_channel_order": "recent_to_oldest",
    "event_scaling": "signed_log1p_fixed_clip",
    "event_clip_count": 16.0,
    "event_neutral_u8": 128,
    "event_sampling_policy": "latest_packet_at_or_before_grid_time",
}

EXPECTED_INTERCEPT_XYT_METADATA = {
    **EXPECTED_INTERCEPT_COMMON_METADATA,
    "input_modality": "event",
    "camera_names": ["event"],
    "visual_history_frames": 1,
    "visual_history_offsets": [0],
    "channels_per_visual_frame": 9,
    "visual_frame_order": "oldest_to_newest",
    "rgb_history_frames": 1,
    "rgb_history_offsets": [0],
    "rgb_frame_order": "oldest_to_newest",
    "image_normalization": "signed_event_u8_centered",
    "event_representation": "xyt_signed_voxel_v1",
    "event_horizon_ms": 200.0,
    "event_temporal_bins": 9,
    "event_bin_width_ms": 200.0 / 9.0,
    "event_spatial_height": 320,
    "event_spatial_width": 320,
    "event_channel_order": "oldest_to_newest",
    "event_polarity_encoding": "signed",
    "event_scaling": "signed_log1p_fixed_clip",
    "event_clip_count": 16.0,
    "event_neutral_u8": 128,
    "event_sampling_policy": "latest_packet_at_or_before_grid_time",
    "qpos_history_frames": 3,
}


@dataclass(frozen=True)
class InterceptVisualContract:
    representation: str
    input_modality: str
    visual_history_offsets: Tuple[int, ...]
    qpos_history_offsets: Tuple[int, int, int]
    channels_per_visual_frame: int
    image_channels: int
    image_normalization: str
    expected_encoding: str
    expected_height: int
    expected_width: int


def resolve_intercept_visual_contract(
    stats: Dict[str, object],
    *,
    cli_camera_name: Optional[str] = None,
    cli_event_representation: Optional[str] = None,
    cli_visual_history_frames: Optional[int] = None,
) -> InterceptVisualContract:
    """Resolve the live visual layout from checkpoint metadata and CLI assertions."""
    checkpoint_modality = _stats_modality(stats)
    representation = stats.get("event_representation")

    if checkpoint_modality in (None, "rgb") and representation is None:
        checkpoint_modality = "rgb"
        resolved_representation = "rgb_history"
        expected = EXPECTED_INTERCEPT_RGB_METADATA
        expected_encoding = "rgb8"
        expected_height = int(stats.get("image_size", 320))
        expected_width = expected_height
    elif checkpoint_modality == "event" and representation == "shifted_3chef_signed":
        resolved_representation = "shifted_3chef_signed"
        expected = EXPECTED_INTERCEPT_EVENT_METADATA
        expected_encoding = "8UC3/bgr8/rgb8"
        expected_height = int(stats.get("image_size", 320))
        expected_width = expected_height
    elif checkpoint_modality == "event" and representation == "xyt_signed_voxel_v1":
        resolved_representation = "xyt_signed_voxel_v1"
        expected = EXPECTED_INTERCEPT_XYT_METADATA
        expected_encoding = "8UC9"
        expected_height = 320
        expected_width = 320
    else:
        raise ValueError(
            "Unsupported or incomplete checkpoint visual representation: "
            f"input_modality={checkpoint_modality!r}, "
            f"event_representation={representation!r}"
        )

    expected_camera = str(expected["input_modality"])
    if cli_camera_name is not None and str(cli_camera_name) != expected_camera:
        raise ValueError(
            "Checkpoint/CLI camera mismatch: "
            f"checkpoint={expected_camera!r}, CLI={cli_camera_name!r}"
        )
    if cli_event_representation is not None:
        asserted = str(cli_event_representation)
        if asserted != resolved_representation:
            raise ValueError(
                "Checkpoint/CLI representation mismatch: "
                f"checkpoint={resolved_representation!r}, CLI={asserted!r}"
            )

    visual_offsets = tuple(int(value) for value in expected["visual_history_offsets"])
    if (
        cli_visual_history_frames is not None
        and int(cli_visual_history_frames) != len(visual_offsets)
    ):
        raise ValueError(
            "Checkpoint/CLI visual history mismatch: "
            f"checkpoint={len(visual_offsets)}, CLI={cli_visual_history_frames}"
        )

    qpos_offsets = tuple(int(value) for value in expected["qpos_history_offsets"])
    if len(qpos_offsets) != 3:
        raise ValueError(f"Expected three qpos offsets, got {qpos_offsets}")

    return InterceptVisualContract(
        representation=resolved_representation,
        input_modality=expected_camera,
        visual_history_offsets=visual_offsets,
        qpos_history_offsets=qpos_offsets,
        channels_per_visual_frame=int(expected["channels_per_visual_frame"]),
        image_channels=int(expected["image_channels"]),
        image_normalization=str(expected["image_normalization"]),
        expected_encoding=expected_encoding,
        expected_height=expected_height,
        expected_width=expected_width,
    )


@dataclass(frozen=True)
class SyncSelection:
    history_indices: Tuple[int, ...]
    visual_timestamps: Tuple[float, ...]
    qpos_target_timestamps: Tuple[float, float, float]
    qpos_timestamps: Tuple[float, float, float]
    anchor_tcp_s: float
    anchor_tcp_s_timestamp: float
    qpos_history: np.ndarray

    @property
    def rgb_timestamps(self) -> Tuple[float, ...]:
        # Backward-compatible alias used by existing rollout code/tests.
        return self.visual_timestamps


@dataclass(frozen=True)
class AggregationSelection:
    value: float
    contributor_count: int
    effective_age_frames: float


class TemporalAbsoluteAggregator:
    """Aggregate per-step absolute predictions from chunked receding-horizon outputs."""

    def __init__(
        self,
        chunk_size: int,
        mode: str = "full",
        decay: float = 0.01,
        recent_window: int = 5,
        recent_half_life: float = 1.0,
        lookahead_steps: int = 0,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self.mode = str(mode)
        self.decay = float(decay)
        self.recent_window = int(recent_window)
        self.recent_half_life = float(recent_half_life)
        self.lookahead_steps = int(lookahead_steps)

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")
        if self.mode not in {"full", "latest", "recent"}:
            raise ValueError(f"mode must be one of full/latest/recent, got {self.mode!r}")
        if not math.isfinite(self.decay) or self.decay < 0.0:
            raise ValueError(f"decay must be finite and >= 0, got {self.decay}")
        if self.recent_window <= 0 or self.recent_window > self.chunk_size:
            raise ValueError(
                "recent_window must be a positive integer not exceeding chunk_size, "
                f"got recent_window={self.recent_window}, chunk_size={self.chunk_size}"
            )
        if not math.isfinite(self.recent_half_life) or self.recent_half_life <= 0.0:
            raise ValueError(
                "recent_half_life must be finite and > 0, "
                f"got {self.recent_half_life}"
            )
        if self.lookahead_steps < 0:
            raise ValueError(
                f"lookahead_steps must be >= 0, got {self.lookahead_steps}"
            )
        if self.lookahead_steps >= self.chunk_size:
            raise ValueError(
                f"lookahead_steps must be < chunk_size, got lookahead_steps={self.lookahead_steps}, "
                f"chunk_size={self.chunk_size}"
            )
        if self.mode == "recent" and self.recent_window > (self.chunk_size - self.lookahead_steps):
            raise ValueError(
                "recent_window must be <= chunk_size - lookahead_steps for mode=recent, "
                f"got recent_window={self.recent_window}, chunk_size={self.chunk_size}, "
                f"lookahead_steps={self.lookahead_steps}"
            )

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

    def _valid_contributions_for_step(self, current_step: int) -> List[Tuple[int, float, int]]:
        contributions: List[Tuple[int, float, int]] = []
        for source_step, chunk in self._history:
            source_age = int(current_step) - int(source_step)
            if source_age < 0:
                continue
            token_index = source_age + self.lookahead_steps
            if token_index < self.chunk_size:
                contributions.append((int(source_step), float(chunk[token_index]), int(token_index)))
        return contributions

    def selection_for_step(self, current_step: int) -> Optional[AggregationSelection]:
        contributions = self._valid_contributions_for_step(current_step)
        if not contributions:
            return None

        current_step_i = int(current_step)

        if self.mode == "latest":
            source_step, value, _token_index = max(contributions, key=lambda item: item[0])
            age = float(current_step_i - source_step)
            return AggregationSelection(
                value=float(value),
                contributor_count=1,
                effective_age_frames=age,
            )

        if self.mode == "recent":
            newest = sorted(contributions, key=lambda item: item[0], reverse=True)[: self.recent_window]
            newest_sorted = sorted(newest, key=lambda item: item[0])
            values = np.asarray([item[1] for item in newest_sorted], dtype=np.float64)
            ages = np.asarray(
                [float(current_step_i - item[0]) for item in newest_sorted],
                dtype=np.float64,
            )
            weights = np.power(0.5, ages / self.recent_half_life)
            weights = weights / np.sum(weights)
            return AggregationSelection(
                value=float(np.sum(values * weights)),
                contributor_count=int(values.shape[0]),
                effective_age_frames=float(np.sum(ages * weights)),
            )

        # Legacy full mode behavior: keep oldest-first values and exp(-decay * arange)
        values = np.asarray([item[1] for item in contributions], dtype=np.float64)
        ages = np.asarray(
            [float(current_step_i - item[0]) for item in contributions],
            dtype=np.float64,
        )
        weights = np.exp(-self.decay * np.arange(len(values), dtype=np.float64))
        weights = weights / np.sum(weights)
        return AggregationSelection(
            value=float(np.sum(values * weights)),
            contributor_count=int(values.shape[0]),
            effective_age_frames=float(np.sum(ages * weights)),
        )

    def value_for_step(self, current_step: int) -> Optional[float]:
        selection = self.selection_for_step(current_step)
        if selection is None:
            return None
        return float(selection.value)


def resolve_temporal_agg_mode(
    temporal_agg_mode: Optional[str],
    temporal_agg_legacy: Optional[bool],
) -> str:
    if temporal_agg_mode is not None and temporal_agg_legacy is not None:
        raise ValueError(
            "Cannot combine legacy temporal-aggregation flags with --temporal-agg-mode"
        )

    if temporal_agg_mode is not None:
        if temporal_agg_mode not in {"full", "latest", "recent"}:
            raise ValueError(
                f"Invalid temporal aggregation mode {temporal_agg_mode!r}; "
                "expected one of full/latest/recent"
            )
        return temporal_agg_mode

    if temporal_agg_legacy is None:
        return "full"
    if temporal_agg_legacy:
        return "full"
    return "latest"


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


def build_visual_history_tensor(
    visual_frames: Sequence[np.ndarray],
    image_size: int,
    modality: str,
    expected_channels: int = 9,
) -> np.ndarray:
    if len(visual_frames) <= 0:
        raise ValueError("Expected at least one visual frame")

    target = int(image_size)
    if target <= 0:
        raise ValueError(f"image_size must be positive, got {target}")

    modality = str(modality).strip().lower()
    if modality not in ("rgb", "event"):
        raise ValueError(f"Unsupported visual modality: {modality!r}")

    processed = []
    total_channels = 0
    for frame in visual_frames:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] <= 0:
            raise ValueError(f"Each visual frame must be HWC, got {arr.shape}")
        if modality == "event" and arr.dtype != np.uint8:
            raise ValueError(
                f"Event frame dtype must be uint8 before policy preprocessing, got {arr.dtype}"
            )
        if arr.shape[:2] == (target, target):
            resized = arr
        else:
            resized = np.stack(
                [
                    cv2.resize(
                        arr[..., channel],
                        (target, target),
                        interpolation=cv2.INTER_AREA,
                    )
                    for channel in range(arr.shape[2])
                ],
                axis=-1,
            )
        processed.append(resized)
        total_channels += int(resized.shape[2])

    channels = int(expected_channels)
    if channels <= 0:
        raise ValueError(f"expected_channels must be positive, got {channels}")

    image_hwc = np.concatenate(processed, axis=2)
    if total_channels != channels or image_hwc.shape != (target, target, channels):
        raise AssertionError(
            "Temporal visual concat mismatch: "
            f"expected {(target, target, channels)}, got {image_hwc.shape}"
        )

    image = np.transpose(image_hwc, (2, 0, 1))[None, None, ...].astype(np.float32) / 255.0
    if image.shape != (1, 1, channels, target, target):
        raise AssertionError(
            "Temporal visual tensor shape mismatch: "
            f"expected {(1, 1, channels, target, target)}, got {image.shape}"
        )
    return image


def build_rgb_history_tensor(rgb_frames: Sequence[np.ndarray], image_size: int) -> np.ndarray:
    return build_visual_history_tensor(rgb_frames, image_size=image_size, modality="rgb")


def decode_uint8_hwc_image_message(
    message,
    *,
    expected_encoding: str,
    expected_height: int,
    expected_width: int,
    expected_channels: int,
) -> np.ndarray:
    """Decode a uint8 HWC ROS image, including messages with padded rows."""
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    channels = int(expected_channels)
    if height != int(expected_height) or width != int(expected_width):
        raise ValueError(
            "Image dimensions mismatch: "
            f"expected {(expected_height, expected_width)}, got {(height, width)}"
        )
    if str(message.encoding) != str(expected_encoding):
        raise ValueError(
            f"Image encoding mismatch: expected {expected_encoding!r}, "
            f"got {message.encoding!r}"
        )

    row_bytes = width * channels
    if step < row_bytes:
        raise ValueError(
            f"Image step is too small: expected at least {row_bytes}, got {step}"
        )
    expected_data_length = height * step
    if len(message.data) != expected_data_length:
        raise ValueError(
            "Image data length mismatch: "
            f"expected {expected_data_length}, got {len(message.data)}"
        )

    raw = np.frombuffer(memoryview(message.data), dtype=np.uint8)
    rows = raw.reshape(height, step)
    packed = rows[:, :row_bytes].reshape(height, width, channels)
    return np.ascontiguousarray(packed)


def validate_xyt_orientation_parity(
    live_rotation_degrees: Optional[int],
    *,
    parity_verified: bool,
) -> str:
    """Validate that live XYT orientation matches unrotated offline conversion."""
    if live_rotation_degrees is None:
        raise ValueError(
            "Cannot verify XYT orientation because the OpenMV rotation parameter "
            "is unavailable"
        )
    rotation = int(live_rotation_degrees)
    if rotation not in (-90, 0, 90, 180):
        raise ValueError(f"Unsupported live XYT rotation: {rotation}")
    if rotation == 0:
        return "zero_rotation"
    if not parity_verified:
        raise ValueError(
            "Live XYT rotation differs from unrotated offline conversion: "
            f"rotation={rotation}. Provide verified synthetic/recorded orientation "
            "parity before rollout."
        )
    return "comparison_verified"


def _resolve_policy_modality(policy_config: Dict[str, object]) -> str:
    camera_names = policy_config.get("camera_names")
    if not isinstance(camera_names, (list, tuple)) or len(camera_names) != 1:
        raise ValueError(
            "Interception policy config must set camera_names to exactly one value: ['rgb'] or ['event']"
        )
    camera_name = str(camera_names[0])
    if camera_name not in ("rgb", "event"):
        raise ValueError(
            f"Interception camera_names must be ['rgb'] or ['event'], got {camera_names}"
        )

    input_modality = policy_config.get("input_modality")
    if input_modality is None:
        # Legacy fallback allowed only for RGB checkpoints.
        if camera_name == "rgb":
            return "rgb"
        raise ValueError(
            "Event rollout requires explicit input_modality='event' in policy config"
        )

    input_modality = str(input_modality)
    if input_modality not in ("rgb", "event"):
        raise ValueError(f"Unsupported policy input_modality: {input_modality!r}")
    if input_modality != camera_name:
        raise ValueError(
            "Interception modality mismatch in policy config: "
            f"camera_names={camera_names}, input_modality={input_modality!r}"
        )
    return input_modality


def _stats_modality(stats: Dict[str, object]) -> Optional[str]:
    modality = stats.get("input_modality")
    if modality is not None:
        modality = str(modality)
        if modality not in ("rgb", "event"):
            raise ValueError(f"Unsupported checkpoint input_modality in stats: {modality!r}")
        return modality

    camera_names = stats.get("camera_names")
    if camera_names == ["rgb"]:
        return "rgb"
    if camera_names == ["event"]:
        return "event"
    return None


def validate_intercept_stats_and_config(
    stats: Dict[str, object],
    policy_config: Dict[str, object],
    expected_chunk_size: int,
) -> Dict[str, np.ndarray]:
    rollout_modality = _resolve_policy_modality(policy_config)
    checkpoint_modality = _stats_modality(stats)

    if checkpoint_modality is None:
        if rollout_modality != "rgb":
            raise ValueError(
                "Checkpoint does not declare event modality metadata; refusing event rollout"
            )
        checkpoint_modality = "rgb"

    if checkpoint_modality != rollout_modality:
        raise ValueError(
            "Checkpoint/rollout modality mismatch: "
            f"checkpoint={checkpoint_modality}, rollout={rollout_modality}"
        )

    visual_contract = resolve_intercept_visual_contract(
        stats,
        cli_camera_name=str(policy_config.get("camera_names", [""])[0]),
        cli_event_representation=policy_config.get("event_representation"),
        cli_visual_history_frames=policy_config.get("visual_history_frames"),
    )
    is_xyt = visual_contract.representation == "xyt_signed_voxel_v1"
    expected_metadata = (
        EXPECTED_INTERCEPT_XYT_METADATA
        if is_xyt
        else (
            EXPECTED_INTERCEPT_EVENT_METADATA
            if rollout_modality == "event"
            else EXPECTED_INTERCEPT_RGB_METADATA
        )
    )

    for key, expected in expected_metadata.items():
        if key not in stats:
            if rollout_modality == "rgb" and key in ("input_modality", "camera_names", "visual_history_frames", "visual_history_offsets", "channels_per_visual_frame", "visual_frame_order", "image_normalization"):
                # Legacy RGB checkpoints are allowed to miss newer metadata keys.
                continue
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
        "rgb_history_frames": 1 if is_xyt else 3,
        "visual_history_frames": 1 if is_xyt else 3,
        "channels_per_visual_frame": 9 if is_xyt else 3,
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

    expected_norm = visual_contract.image_normalization
    actual_norm = str(policy_config.get("image_normalization", ""))
    if actual_norm != expected_norm:
        raise ValueError(
            "Interception policy config mismatch for image_normalization: "
            f"expected {expected_norm!r}, found {policy_config.get('image_normalization')!r}"
        )

    expected_offsets = [0] if is_xyt else list(INTERCEPT_HISTORY_OFFSETS)
    if list(policy_config.get("visual_history_offsets", expected_offsets)) != expected_offsets:
        raise ValueError(
            "Interception policy config mismatch for visual_history_offsets: "
            f"expected {expected_offsets}, found {policy_config.get('visual_history_offsets')}"
        )

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
    qpos_history_offsets: Sequence[int] = INTERCEPT_HISTORY_OFFSETS,
    frame_period_sec: Optional[float] = None,
    qpos_relative_to_anchor: bool = False,
    max_qpos_age_sec: Optional[float] = None,
) -> SyncSelection:
    if not rgb_timestamps:
        raise ValueError("No visual samples available")
    if len(joint_timestamps) != len(joint_qpos_samples):
        raise ValueError("joint_timestamps and joint_qpos_samples length mismatch")
    if len(tcp_timestamps) != len(tcp_values):
        raise ValueError("tcp_timestamps and tcp_values length mismatch")

    anchor_index = len(rgb_timestamps) - 1
    history_indices = compute_history_indices(anchor_index, history_offsets)

    selected_rgb_ts = [float(rgb_timestamps[idx]) for idx in history_indices]

    anchor_ts = selected_rgb_ts[-1]
    if qpos_relative_to_anchor:
        if frame_period_sec is None or float(frame_period_sec) <= 0.0:
            raise ValueError(
                "frame_period_sec must be positive for anchor-relative qpos history"
            )
        qpos_targets = [
            anchor_ts + int(offset) * float(frame_period_sec)
            for offset in qpos_history_offsets
        ]
    else:
        qpos_targets = list(selected_rgb_ts)
    if len(qpos_targets) != 3:
        raise ValueError(f"Expected exactly three qpos target times, got {qpos_targets}")

    selected_qpos = []
    selected_qpos_ts = []
    for target_ts in qpos_targets:
        joint_idx = select_latest_index_at_or_before(joint_timestamps, target_ts)
        if joint_idx is None:
            raise ValueError(
                "No causal JointState sample available at or before qpos target "
                f"timestamp {target_ts:.6f}"
            )
        selected_timestamp = float(joint_timestamps[joint_idx])
        age = float(target_ts) - selected_timestamp
        if age < 0.0:
            raise ValueError("Causal qpos synchronization selected a future sample")
        if (
            max_qpos_age_sec is not None
            and float(max_qpos_age_sec) > 0.0
            and age > float(max_qpos_age_sec)
        ):
            raise ValueError(
                "JointState sample is stale for qpos target: "
                f"age={age:.6f}s exceeds max {float(max_qpos_age_sec):.6f}s"
            )
        selected_qpos_ts.append(selected_timestamp)
        selected_qpos.append(joint_qpos_samples[joint_idx])

    qpos_history = build_qpos_history(selected_qpos)

    tcp_idx = select_latest_index_at_or_before(tcp_timestamps, anchor_ts)
    if tcp_idx is None:
        raise ValueError(
            "No causal current_tcp_s sample available at or before visual anchor timestamp "
            f"{anchor_ts:.6f}"
        )

    anchor_tcp_s = float(tcp_values[tcp_idx])
    anchor_tcp_ts = float(tcp_timestamps[tcp_idx])

    if not math.isfinite(anchor_tcp_s):
        raise ValueError(f"Anchor current_tcp_s is non-finite: {anchor_tcp_s}")

    return SyncSelection(
        history_indices=tuple(history_indices),
        visual_timestamps=tuple(selected_rgb_ts),
        qpos_target_timestamps=tuple(qpos_targets),
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
