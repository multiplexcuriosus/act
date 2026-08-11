"""Shared four-feature sparse-ball construction for training and rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


SPARSE_FEATURE_DIM = 4
SPARSE_HISTORY_OFFSETS = (-6, -3, 0)
SPARSE_HISTORY_LENGTH = 3
SPARSE_FEATURE_NAMES = ("u", "v", "valid", "observation_age")
SPARSE_SOURCE_TIMESTAMP_POLICY = "point_stamped_header_latest_at_or_before_policy_time"
DEFAULT_MAX_OBSERVATION_AGE_SEC = 0.10
DEFAULT_RGB_SPARSE_TOPIC = "/ball_tracker2/ball_2d_px"
DEFAULT_EVENT_SPARSE_TOPIC = "/openmv_cam/event_tracker/ball_2d_px"


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
