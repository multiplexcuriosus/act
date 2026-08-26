import bisect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from image_preprocessing import mask_and_center_crop_square_rotated_event_image

INTERCEPT_HISTORY_OFFSETS = (-6, -3, 0)
RAW_EVENT_FRAME_SHAPE = (320, 320, 3)
ARM_JOINT_NAMES = (
    "right_fr3_joint1",
    "right_fr3_joint2",
    "right_fr3_joint3",
    "right_fr3_joint4",
    "right_fr3_joint5",
    "right_fr3_joint6",
    "right_fr3_joint7",
)


@dataclass(frozen=True)
class EventSpatialPreprocessingConfig:
    mask_x: Tuple[int, int]
    crop_square: int
    fill_value: int = 128


def add_event_spatial_preprocessing_arguments(parser) -> None:
    """Add opt-in live-event spatial preprocessing arguments to a CLI parser."""
    parser.add_argument(
        "--event-mask-x",
        nargs=2,
        type=int,
        metavar=("XTOP", "XBOTTOM"),
        default=None,
    )
    parser.add_argument(
        "--event-crop-square",
        type=int,
        metavar="SIDE",
        default=None,
    )
    parser.add_argument(
        "--event-mask-fill-value",
        type=int,
        metavar="VALUE",
        default=128,
    )


def resolve_event_spatial_preprocessing(
    *,
    modality: str,
    mask_x: Optional[Sequence[int]],
    crop_square: Optional[int],
    fill_value: int = 128,
) -> Optional[EventSpatialPreprocessingConfig]:
    """Validate and resolve the optional live-event spatial transform."""
    resolved_fill = int(fill_value)
    if not 0 <= resolved_fill <= 255:
        raise ValueError(
            "--event-mask-fill-value must satisfy 0 <= VALUE <= 255, "
            f"got {resolved_fill}"
        )

    has_mask = mask_x is not None
    has_crop = crop_square is not None
    if has_mask != has_crop:
        raise ValueError("--event-mask-x and --event-crop-square must be supplied together")
    if not has_mask:
        return None

    if str(modality).strip().lower() != "event":
        raise ValueError("Event spatial preprocessing is only valid with --camera_name event")
    if len(mask_x) != 2:
        raise ValueError(f"--event-mask-x requires XTOP XBOTTOM, got {mask_x!r}")

    resolved_mask = (int(mask_x[0]), int(mask_x[1]))
    for name, value in zip(("XTOP", "XBOTTOM"), resolved_mask):
        if not 0 <= value < RAW_EVENT_FRAME_SHAPE[1]:
            raise ValueError(
                f"--event-mask-x {name} must satisfy 0 <= value < 320, got {value}"
            )

    resolved_crop = int(crop_square)
    if not 0 < resolved_crop <= RAW_EVENT_FRAME_SHAPE[1]:
        raise ValueError(
            "--event-crop-square must satisfy 0 < SIDE <= 320, "
            f"got {resolved_crop}"
        )

    return EventSpatialPreprocessingConfig(
        mask_x=resolved_mask,
        crop_square=resolved_crop,
        fill_value=resolved_fill,
    )


def preprocess_event_history_frames(
    visual_frames: Sequence[np.ndarray],
    config: Optional[EventSpatialPreprocessingConfig],
) -> List[np.ndarray]:
    """Apply the configured training spatial transform to selected event frames."""
    if config is None:
        return list(visual_frames)

    transformed = []
    for index, frame in enumerate(visual_frames):
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            raise ValueError(
                "Event spatial preprocessing requires raw frame dtype uint8; "
                f"history frame {index} has dtype {array.dtype}"
            )
        if array.shape != RAW_EVENT_FRAME_SHAPE:
            raise ValueError(
                "Event spatial preprocessing requires raw frame shape (320, 320, 3); "
                f"history frame {index} has shape {array.shape}. The image topic may "
                "already be cropped or otherwise incompatible."
            )
        transformed.append(
            mask_and_center_crop_square_rotated_event_image(
                array,
                mask_x=config.mask_x,
                square_side=config.crop_square,
                fill_value=config.fill_value,
            )
        )
    return transformed


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
class SyncSelection:
    history_indices: Tuple[int, int, int]
    visual_timestamps: Tuple[float, float, float]
    qpos_timestamps: Tuple[float, float, float]
    anchor_tcp_s: float
    anchor_tcp_s_timestamp: float
    qpos_history: np.ndarray

    @property
    def rgb_timestamps(self) -> Tuple[float, float, float]:
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


def select_qpos_history_at_targets(
    timestamps: Sequence[float],
    qpos_samples: Sequence[np.ndarray],
    target_timestamps: Sequence[float],
) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Select the latest causal qpos sample for every policy-history target."""
    if len(timestamps) != len(qpos_samples):
        raise ValueError("qpos timestamps and samples must have matching lengths")
    selected = []
    selected_timestamps = []
    for target in target_timestamps:
        index = select_latest_index_at_or_before(timestamps, target)
        if index is None:
            raise ValueError(f"No causal qpos sample at history target {target:.9f}")
        selected.append(qpos_samples[index])
        selected_timestamps.append(float(timestamps[index]))
    return build_qpos_history(selected), tuple(selected_timestamps)


def skip_duplicate_source_frame(input_modality: str) -> bool:
    """Dense visual policies are frame-driven; sparse policies are clock-driven."""
    return str(input_modality) != "sparse_ball"


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
    if not qpos_samples:
        raise ValueError("Expected at least one qpos sample")

    flattened = []
    for sample in qpos_samples:
        qpos = np.asarray(sample, dtype=np.float32).reshape(-1)
        if qpos.shape != (7,):
            raise ValueError(f"Each qpos sample must be shape (7,), got {qpos.shape}")
        flattened.append(qpos)

    out = np.concatenate(flattened, axis=0)
    expected_shape = (7 * len(qpos_samples),)
    if out.shape != expected_shape:
        raise AssertionError(
            f"qpos history shape mismatch: expected {expected_shape}, got {out.shape}")
    return out


def build_visual_history_tensor(
    visual_frames: Sequence[np.ndarray],
    image_size: int,
    modality: str,
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

    image_hwc = np.concatenate(processed, axis=2)
    if total_channels != 9 or image_hwc.shape != (target, target, 9):
        raise AssertionError(
            f"Temporal visual concat mismatch: expected {(target, target, 9)}, got {image_hwc.shape}"
        )

    image = np.transpose(image_hwc, (2, 0, 1))[None, None, ...].astype(np.float32) / 255.0
    if image.shape != (1, 1, 9, target, target):
        raise AssertionError(
            "Temporal visual tensor shape mismatch: "
            f"expected {(1, 1, 9, target, target)}, got {image.shape}"
        )
    return image


def build_rgb_history_tensor(rgb_frames: Sequence[np.ndarray], image_size: int) -> np.ndarray:
    return build_visual_history_tensor(rgb_frames, image_size=image_size, modality="rgb")


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


def validate_normalization_stats(
    stats: Dict[str, object],
    *,
    include_sparse: bool = False,
    qpos_dim: int = 21,
) -> Dict[str, np.ndarray]:
    """Validate normalization arrays shared by dense and sparse rollouts."""
    expected = [
        ("qpos_mean", (int(qpos_dim),), False),
        ("qpos_std", (int(qpos_dim),), True),
        ("action_mean", (1,), False),
        ("action_std", (1,), True),
    ]
    if include_sparse:
        expected.extend((
            ("sparse_mean", (4,), False),
            ("sparse_std", (4,), True),
        ))

    arrays: Dict[str, np.ndarray] = {}
    for stat_key, expected_shape, strictly_positive in expected:
        if stat_key not in stats:
            raise ValueError(f"Missing required stats key: {stat_key}")
        arr = np.asarray(stats[stat_key], dtype=np.float32)
        if arr.shape != expected_shape:
            raise ValueError(
                f"Stats shape mismatch for {stat_key}: "
                f"expected {expected_shape}, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Stats {stat_key} contains non-finite values")
        if strictly_positive and np.any(arr <= 0.0):
            raise ValueError(f"{stat_key} must be strictly positive")
        arrays[stat_key] = arr
    return arrays


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

    is_xyt = stats.get("event_representation") == "xyt_signed_voxel_v1"
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
        if (
            key == "image_normalization"
            and rollout_modality == "event"
            and stats[key] in ("signed_event_u8_centered", "shifted_3chef_centered")
        ):
            continue
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

    expected_norm = "signed_event_u8_centered" if rollout_modality == "event" else "imagenet"
    actual_norm = str(policy_config.get("image_normalization", ""))
    accepted_norms = {expected_norm}
    if rollout_modality == "event" and not is_xyt:
        accepted_norms.add("shifted_3chef_centered")
    if actual_norm not in accepted_norms:
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

    return validate_normalization_stats(stats)


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
        raise ValueError("No visual samples available")
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
                "No causal JointState sample available at or before visual timestamp "
                f"{rgb_ts:.6f}"
            )
        selected_qpos_ts.append(float(joint_timestamps[joint_idx]))
        selected_qpos.append(joint_qpos_samples[joint_idx])

    qpos_history = build_qpos_history(selected_qpos)

    anchor_ts = selected_rgb_ts[-1]
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
