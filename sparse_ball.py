"""Shared four-feature sparse-ball construction for training and rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import warnings

import numpy as np


SPARSE_FEATURE_DIM = 4
SUPPORTED_POLICY_RATES_HZ = (30, 60)
SPARSE_HISTORY_OFFSETS_SEC = (-0.2, -0.1, 0.0)
SPARSE_HISTORY_OFFSETS = (-6, -3, 0)
SPARSE_HISTORY_LENGTH = 3
SPARSE_HISTORY_MODE_LEGACY = "legacy"
SPARSE_HISTORY_MODE_M_WINDOW = "m_window"
DEFAULT_HISTORY_MS = 200.0
DEFAULT_SPARSE_HISTORY_CAPACITY = 32
SPARSE_FEATURE_NAMES = ("u_norm", "v_norm", "valid", "observation_age_sec")
SPARSE_SOURCE_TIMESTAMP_POLICY = "point_stamped_header_latest_at_or_before_policy_time"
DEFAULT_MAX_OBSERVATION_AGE_SEC = 0.10
DEFAULT_RGB_SPARSE_TOPIC = "/ball_tracker2/ball_2d_px"
DEFAULT_EVENT_SPARSE_TOPIC = "/openmv_cam/event_tracker/ball_2d_px"


def validate_policy_rate(rate_hz) -> int:
    """Return a supported integral dataset/runtime policy rate."""
    rate = int(rate_hz)
    if float(rate_hz) != rate or rate not in SUPPORTED_POLICY_RATES_HZ:
        raise ValueError(
            f"policy rate must be one of {SUPPORTED_POLICY_RATES_HZ}, got {rate_hz!r}"
        )
    return rate


def policy_period_ns(rate_hz) -> int:
    """Rounded integer nanoseconds used to describe an HDF5 policy grid."""
    rate = validate_policy_rate(rate_hz)
    return int(round(1_000_000_000 / rate))


def policy_period_sec(rate_hz) -> float:
    """Floating-point seconds used for runtime timing."""
    return 1.0 / validate_policy_rate(rate_hz)


def sparse_history_offsets_frames(rate_hz):
    """Return the fixed-time sparse history offsets at the selected rate."""
    rate = validate_policy_rate(rate_hz)
    return tuple(int(round(offset * rate)) for offset in SPARSE_HISTORY_OFFSETS_SEC)


def qpos_history_offsets_for_window(rate_hz, history_ms=DEFAULT_HISTORY_MS):
    """Return every policy-grid offset in an inclusive causal time window."""
    rate = validate_policy_rate(rate_hz)
    history_ms = float(history_ms)
    if not np.isfinite(history_ms) or history_ms < 0:
        raise ValueError("history_ms must be finite and non-negative")
    frames = int(round(history_ms * rate / 1000.0))
    return tuple(range(-frames, 1))


def resolve_history_config(mode=SPARSE_HISTORY_MODE_LEGACY, policy_rate_hz=30,
                           history_ms=DEFAULT_HISTORY_MS,
                           sparse_history_capacity=DEFAULT_SPARSE_HISTORY_CAPACITY):
    """Resolve tensor dimensions while retaining the historical default."""
    mode = str(mode)
    if mode not in (SPARSE_HISTORY_MODE_LEGACY, SPARSE_HISTORY_MODE_M_WINDOW):
        raise ValueError("sparse_history_mode must be 'legacy' or 'm_window'")
    if mode == SPARSE_HISTORY_MODE_LEGACY:
        offsets = sparse_history_offsets_frames(policy_rate_hz)
        capacity = SPARSE_HISTORY_LENGTH
        horizon_ms = 200.0
    else:
        offsets = qpos_history_offsets_for_window(policy_rate_hz, history_ms)
        capacity = int(sparse_history_capacity)
        horizon_ms = float(history_ms)
        if capacity <= 0:
            raise ValueError("sparse_history_capacity must be positive")
    return {
        "history_mode": mode,
        "history_horizon_ms": horizon_ms,
        "history_horizon_sec": horizon_ms / 1000.0,
        "sparse_history_capacity": capacity,
        "sparse_feature_dim": SPARSE_FEATURE_DIM,
        "qpos_history_offsets": offsets,
        "qpos_history_length": len(offsets),
        "state_dim": 7 * len(offsets),
        "qpos_flatten_order": "oldest_to_newest",
        "causal_sampling_policy": "source_timestamp_at_or_before_policy_anchor_within_horizon",
    }


def default_sparse_topic(source: str) -> str:
    """Return the repository default PointStamped topic for a sparse source."""
    if source == "rgb":
        return DEFAULT_RGB_SPARSE_TOPIC
    if source == "event":
        return DEFAULT_EVENT_SPARSE_TOPIC
    raise ValueError(f"sparse_source must be 'rgb' or 'event', got {source!r}")


def sparse_dataset_paths(source: str):
    """Return canonical raw HDF5 paths for a sparse source."""
    if source not in ("rgb", "event"):
        raise ValueError(f"sparse_source must be 'rgb' or 'event', got {source!r}")
    prefix = f"/observations/sparse_tracking/{source}"
    return f"{prefix}_2d_px", f"{prefix}_valid", f"{prefix}_source_timestamps"


def raw_sparse_dataset_paths(source: str):
    """Return canonical literal (not policy-grid aligned) sparse stream paths."""
    if source not in ("rgb", "event"):
        raise ValueError(f"sparse_source must be 'rgb' or 'event', got {source!r}")
    prefix = "/observations/sparse_tracking"
    return (f"{prefix}/raw_{source}_timestamps",
            f"{prefix}/raw_{source}_2d_px",
            f"{prefix}/raw_{source}_valid")


def construct_sparse_features(
    observation_timestamps,
    coordinates,
    raw_valid,
    source_timestamps,
    image_width: int,
    image_height: int,
    max_observation_age_sec: float = DEFAULT_MAX_OBSERVATION_AGE_SEC,
):
    """Construct ``[u_norm, v_norm, valid, age]`` without future leakage."""
    times = np.asarray(observation_timestamps, dtype=np.float64)
    points = np.asarray(coordinates, dtype=np.float64)
    valid = np.asarray(raw_valid).reshape(-1)
    source_times = np.asarray(source_timestamps, dtype=np.float64)
    max_age = float(max_observation_age_sec)
    if times.ndim != 1 or source_times.shape != times.shape:
        raise ValueError("observation and source timestamps must have matching shape (T,)")
    if points.shape != (len(times), 2) or valid.shape != times.shape:
        raise ValueError("sparse coordinates must be (T,2) and validity must be (T,)")
    if len(times) and (not np.isfinite(times).all() or np.any(np.diff(times) < 0)):
        raise ValueError("observation timestamps must be finite and monotonic")
    if image_width <= 1 or image_height <= 1:
        raise ValueError("sparse image dimensions must both exceed one pixel")
    if not np.isfinite(max_age) or max_age <= 0:
        raise ValueError("max_observation_age_sec must be positive and finite")

    output = np.zeros((len(times), SPARSE_FEATURE_DIM), dtype=np.float32)
    output[:, 3] = max_age
    ages = times - source_times
    usable = (
        np.isfinite(source_times)
        & np.isfinite(points).all(axis=1)
        & (valid != 0)
        & (source_times <= times)
        & (ages >= 0.0)
        & (ages <= max_age)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] < image_width)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < image_height)
    )
    output[usable, 0] = 2.0 * points[usable, 0] / (image_width - 1) - 1.0
    output[usable, 1] = 2.0 * points[usable, 1] / (image_height - 1) - 1.0
    output[usable, 2] = 1.0
    output[usable, 3] = np.clip(ages[usable], 0.0, max_age)
    return output


@dataclass(frozen=True)
class SparsePoint:
    """A timestamped sparse point used by causal rollout history."""

    source_timestamp: float
    u: float
    v: float
    valid: int = 1
    availability_timestamp: float | None = None


def construct_causal_sparse_window(
    points: Iterable[SparsePoint], policy_timestamp: float, history_ms: float,
    capacity: int, image_width: int, image_height: int, *, return_info=False,
):
    """Build a front-padded, oldest-to-newest literal sparse observation window."""
    anchor = float(policy_timestamp)
    horizon = float(history_ms) / 1000.0
    capacity = int(capacity)
    if not np.isfinite(anchor) or not np.isfinite(horizon) or horizon < 0:
        raise ValueError("policy_timestamp and history_ms must be finite; history_ms >= 0")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if image_width <= 1 or image_height <= 1:
        raise ValueError("sparse image dimensions must both exceed one pixel")
    ordered = sorted(points, key=lambda item: item.source_timestamp)
    stamps = np.asarray([p.source_timestamp for p in ordered], dtype=np.float64)
    if stamps.size and (not np.isfinite(stamps).all() or np.any(np.diff(stamps) < 0)):
        raise ValueError("sparse source timestamps must be finite and monotonic")
    selected = [
        p for p in ordered
        if anchor - horizon <= p.source_timestamp <= anchor
        and (p.availability_timestamp is None
             or float(p.availability_timestamp) <= anchor)
    ]
    overflow_count = max(0, len(selected) - capacity)
    if overflow_count:
        selected = selected[-capacity:]
    result = np.zeros((capacity, SPARSE_FEATURE_DIM), dtype=np.float32)
    start = capacity - len(selected)
    for row, point in enumerate(selected, start=start):
        age = anchor - float(point.source_timestamp)
        in_bounds = (np.isfinite(point.u) and np.isfinite(point.v)
                     and 0 <= point.u < image_width and 0 <= point.v < image_height)
        result[row, 3] = age
        if point.valid and in_bounds:
            result[row, 0] = 2.0 * float(point.u) / (image_width - 1) - 1.0
            result[row, 1] = 2.0 * float(point.v) / (image_height - 1) - 1.0
            result[row, 2] = 1.0
    info = {
        "overflow": bool(overflow_count),
        "overflow_count": overflow_count,
        "selected_count": len(selected),
        "selected_timestamps": np.asarray(
            [p.source_timestamp for p in selected], dtype=np.float64
        ),
        "selected_availability_timestamps": np.asarray(
            [p.source_timestamp if p.availability_timestamp is None
             else p.availability_timestamp for p in selected], dtype=np.float64
        ),
    }
    return (result, info) if return_info else result


def construct_causal_sparse_history(
    points: Iterable[SparsePoint],
    policy_timestamp: float,
    history_seconds: Sequence[float],
    image_width: int,
    image_height: int,
    max_observation_age_sec: float = DEFAULT_MAX_OBSERVATION_AGE_SEC,
):
    """Select the latest header-stamped point at or before each history time."""
    ordered = sorted(points, key=lambda item: item.source_timestamp)
    targets = np.asarray(
        [float(policy_timestamp) + float(offset) for offset in history_seconds],
        dtype=np.float64,
    )
    source = np.full(len(targets), np.nan, dtype=np.float64)
    coordinates = np.zeros((len(targets), 2), dtype=np.float64)
    valid = np.zeros(len(targets), dtype=np.uint8)
    if ordered:
        stamps = np.asarray([item.source_timestamp for item in ordered], dtype=np.float64)
        indices = np.searchsorted(stamps, targets, side="right") - 1
        present = indices >= 0
        for row, index in zip(np.flatnonzero(present), indices[present]):
            item = ordered[int(index)]
            source[row] = item.source_timestamp
            coordinates[row] = (item.u, item.v)
            valid[row] = item.valid
    return construct_sparse_features(
        targets, coordinates, valid, source, image_width, image_height,
        max_observation_age_sec,
    )


def validate_sparse_checkpoint_contract(
    stats, source, image_width, image_height, max_observation_age_sec
):
    """Reject incompatible dense or sparse checkpoint metadata."""
    required = {
        "input_modality": "sparse_ball",
        "sparse_source": source,
        "sparse_feature_dim": SPARSE_FEATURE_DIM,
        "sparse_history_length": SPARSE_HISTORY_LENGTH,
        "sparse_feature_names": list(SPARSE_FEATURE_NAMES),
        "sparse_image_width": int(image_width),
        "sparse_image_height": int(image_height),
    }
    for key, expected in required.items():
        if key not in stats:
            raise ValueError(f"Sparse checkpoint is missing {key}")
        if stats[key] != expected:
            raise ValueError(
                f"Sparse checkpoint contract mismatch for {key}: "
                f"configured={expected!r}, saved={stats[key]!r}"
            )
    saved_age = float(stats.get("sparse_max_observation_age_sec", np.nan))
    if not np.isclose(saved_age, float(max_observation_age_sec), atol=1e-12):
        raise ValueError(
            "Sparse checkpoint contract mismatch for max observation age: "
            f"configured={max_observation_age_sec}, saved={saved_age}"
        )


def resolve_sparse_checkpoint_contract(
    stats, requested_policy_rate_hz, requested_chunk_size,
    requested_sparse_source, warning_callback=None,
):
    """Resolve strict modern metadata or the legacy 30 Hz sparse contract."""
    rate = validate_policy_rate(requested_policy_rate_hz)
    chunk_size = int(requested_chunk_size)
    expected_offsets = sparse_history_offsets_frames(rate)
    if chunk_size != rate:
        raise ValueError(
            f"Sparse rollout at {rate} Hz requires chunk_size={rate}, got {chunk_size}"
        )

    expected = {
        "policy_rate_hz": rate,
        "qpos_history_offsets": list(expected_offsets),
        "chunk_size": chunk_size,
        "sparse_history_offsets_frames": list(expected_offsets),
        "sparse_source": requested_sparse_source,
        "sparse_feature_dim": SPARSE_FEATURE_DIM,
        "sparse_history_length": SPARSE_HISTORY_LENGTH,
    }
    legacy_fields = (
        "policy_rate_hz", "qpos_history_offsets", "chunk_size",
        "sparse_history_offsets_frames",
    )
    missing_legacy = [key for key in legacy_fields if key not in stats]
    if missing_legacy and rate != 30:
        raise ValueError(
            "Sparse checkpoint requested at 60 Hz is missing rate/history metadata "
            f"{missing_legacy}. Legacy metadata is inferred only at 30 Hz; retrain the "
            "checkpoint or regenerate its metadata for 60 Hz."
        )

    resolved = {}
    for key, expected_value in expected.items():
        if key not in stats:
            if key in legacy_fields and rate == 30:
                resolved[key] = expected_value
                continue
            raise ValueError(f"Sparse checkpoint is missing {key}")
        saved = stats[key]
        if isinstance(saved, np.ndarray):
            saved = saved.tolist()
        if key in ("qpos_history_offsets", "sparse_history_offsets_frames"):
            saved = list(saved)
        elif key in ("policy_rate_hz", "chunk_size", "sparse_feature_dim",
                     "sparse_history_length"):
            saved = int(saved)
        if saved != expected_value:
            raise ValueError(
                f"Sparse checkpoint/rollout mismatch for {key}: "
                f"rollout={expected_value!r}, saved={saved!r}"
            )
        resolved[key] = saved

    resolved["legacy_inferred_fields"] = missing_legacy
    resolved["legacy_inferred"] = bool(missing_legacy)
    if missing_legacy:
        message = (
            f"[WARN] Legacy sparse checkpoint metadata missing {missing_legacy}; "
            "assuming the legacy 30 Hz sparse contract."
        )
        if warning_callback is None:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            warning_callback(message)
    return resolved
