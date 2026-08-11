"""Causal four-feature sparse-ball preprocessing for live ACT rollout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import bisect
import numpy as np

SPARSE_BALL_FEATURE_NAMES = ("u_norm", "v_norm", "valid", "observation_age")
SPARSE_BALL_FEATURE_DIM = 4
SPARSE_BALL_HISTORY_OFFSETS_SEC = (-0.2, -0.1, 0.0)
SPARSE_BALL_COORDINATE_CONVENTION = "pixel_divided_by_image_dimension"
SPARSE_BALL_TIMESTAMP_POLICY = "point_stamped_header_timestamp_causal_at_or_before_target_time"


@dataclass(frozen=True, order=True)
class BallObservation:
    timestamp: float
    u: float
    v: float
    valid: bool = True


@dataclass(frozen=True)
class SparseSelection:
    feature: np.ndarray
    source_timestamp: Optional[float]
    observation_age: float
    valid: bool


def normalize_pixel(u: float, v: float, width: int, height: int) -> Tuple[float, float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    return float(u) / float(width), float(v) / float(height)


def history_target_times(policy_time: float,
                         offsets_sec: Sequence[float] = SPARSE_BALL_HISTORY_OFFSETS_SEC
                         ) -> Tuple[float, ...]:
    if len(offsets_sec) != 3 or tuple(float(x) for x in offsets_sec) != SPARSE_BALL_HISTORY_OFFSETS_SEC:
        raise ValueError("canonical sparse history offsets must be (-0.2, -0.1, 0.0) seconds")
    return tuple(float(policy_time) + float(offset) for offset in offsets_sec)


def _ordered(observations: Iterable[BallObservation]) -> Tuple[BallObservation, ...]:
    result = tuple(sorted(observations, key=lambda item: item.timestamp))
    for obs in result:
        if not np.isfinite(obs.timestamp):
            raise ValueError("ball observation timestamps must be finite")
        if obs.valid and not np.isfinite((obs.u, obs.v)).all():
            raise ValueError("valid ball observations must contain finite coordinates")
    return result


def select_sparse_observation(observations: Sequence[BallObservation], target_time: float,
                              width: int, height: int,
                              max_observation_age: float) -> SparseSelection:
    if max_observation_age <= 0.0 or not np.isfinite(max_observation_age):
        raise ValueError("max_observation_age must be finite and positive")
    ordered = _ordered(observations)
    stamps = [item.timestamp for item in ordered]
    idx = bisect.bisect_right(stamps, float(target_time)) - 1
    invalid = np.asarray([0.0, 0.0, 0.0, max_observation_age], dtype=np.float32)
    if idx < 0:
        return SparseSelection(invalid, None, max_observation_age, False)
    selected = ordered[idx]
    age = float(target_time) - selected.timestamp
    usable = bool(selected.valid and 0.0 <= age <= max_observation_age)
    if not usable:
        return SparseSelection(invalid, selected.timestamp, max_observation_age, False)
    u_norm, v_norm = normalize_pixel(selected.u, selected.v, width, height)
    feature = np.asarray([u_norm, v_norm, 1.0, age], dtype=np.float32)
    return SparseSelection(feature, selected.timestamp, age, True)


def sparse_feature_at_time(observations: Sequence[BallObservation], target_time: float,
                           width: int, height: int, max_observation_age: float) -> np.ndarray:
    return select_sparse_observation(
        observations, target_time, width, height, max_observation_age
    ).feature


def build_sparse_history(observations: Sequence[BallObservation], target_times: Sequence[float],
                         width: int, height: int, max_observation_age: float) -> np.ndarray:
    if len(target_times) != 3:
        raise ValueError(f"sparse history requires exactly 3 target times, got {len(target_times)}")
    result = np.stack([
        sparse_feature_at_time(observations, target, width, height, max_observation_age)
        for target in target_times
    ]).astype(np.float32)
    if result.shape != (3, 4):
        raise AssertionError(f"sparse history must have shape (3, 4), got {result.shape}")
    return result


def normalize_sparse_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(4)
    std = np.asarray(std, dtype=np.float32).reshape(4)
    if features.shape[-1] != 4 or np.any(std <= 0) or not np.isfinite(features).all():
        raise ValueError("invalid sparse features or normalization statistics")
    return (features - mean) / std


def sparse_metadata(width: int, height: int, max_observation_age: float, source_topic: str,
                    sparse_source: str) -> dict:
    return {
        "input_modality": "sparse_ball",
        "sparse_source": str(sparse_source),
        "sparse_feature_dim": 4,
        "sparse_feature_names": list(SPARSE_BALL_FEATURE_NAMES),
        "sparse_history_length": 3,
        "sparse_history_offsets_sec": list(SPARSE_BALL_HISTORY_OFFSETS_SEC),
        "image_width": int(width),
        "image_height": int(height),
        "coordinate_convention": SPARSE_BALL_COORDINATE_CONVENTION,
        "max_observation_age_sec": float(max_observation_age),
        "sparse_topic": str(source_topic),
        "source_timestamp_policy": SPARSE_BALL_TIMESTAMP_POLICY,
        "missing_observation_policy": "zeros_valid_zero_age_capped_at_max",
    }
