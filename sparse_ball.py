"""Shared causal sparse-ball preprocessing for training and live rollout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple
import bisect
import numpy as np

SPARSE_BALL_FEATURE_NAMES = ("u", "v", "du_dt", "dv_dt", "valid", "observation_age")
SPARSE_BALL_FEATURE_DIM = 6
SPARSE_BALL_HISTORY_OFFSETS = (-6, -3, 0)
SPARSE_BALL_COORDINATE_CONVENTION = "normalized_image_coordinates_minus1_to_plus1"
SPARSE_BALL_VELOCITY_CONVENTION = "normalized_image_coordinates_per_second"
SPARSE_BALL_TIMESTAMP_POLICY = "source_header_timestamp_causal_at_or_before_policy_time"


@dataclass(frozen=True, order=True)
class BallObservation:
    timestamp: float
    u: float
    v: float


def normalize_pixel(u: float, v: float, width: int, height: int) -> Tuple[float, float]:
    if width <= 1 or height <= 1:
        raise ValueError(f"image dimensions must exceed one pixel, got {width}x{height}")
    return 2.0 * float(u) / float(width - 1) - 1.0, 2.0 * float(v) / float(height - 1) - 1.0


def _ordered(observations: Iterable[BallObservation]) -> Tuple[BallObservation, ...]:
    result = tuple(sorted(observations, key=lambda item: item.timestamp))
    if any(not np.isfinite((o.timestamp, o.u, o.v)).all() for o in result):
        raise ValueError("ball observations must contain only finite timestamps and coordinates")
    return result


def sparse_feature_at_time(observations: Sequence[BallObservation], policy_time: float,
                           width: int, height: int, max_observation_age: float) -> np.ndarray:
    """Construct one feature using source timestamps only; future samples are invisible."""
    obs = _ordered(observations)
    stamps = [item.timestamp for item in obs]
    idx = bisect.bisect_right(stamps, float(policy_time)) - 1
    if idx < 0:
        return np.asarray([0, 0, 0, 0, 0, max_observation_age], dtype=np.float32)
    latest = obs[idx]
    u, v = normalize_pixel(latest.u, latest.v, width, height)
    age = float(np.clip(float(policy_time) - latest.timestamp, 0.0, max_observation_age))
    du_dt = dv_dt = 0.0
    if idx > 0:
        previous = obs[idx - 1]
        dt = latest.timestamp - previous.timestamp
        if dt > 0.0:
            pu, pv = normalize_pixel(previous.u, previous.v, width, height)
            du_dt, dv_dt = (u - pu) / dt, (v - pv) / dt
    valid = float(float(policy_time) - latest.timestamp <= max_observation_age)
    return np.asarray([u, v, du_dt, dv_dt, valid, age], dtype=np.float32)


def build_sparse_history(observations: Sequence[BallObservation], grid_times: Sequence[float],
                         anchor_index: int, width: int, height: int,
                         max_observation_age: float,
                         history_offsets: Sequence[int] = SPARSE_BALL_HISTORY_OFFSETS) -> np.ndarray:
    indices = [max(0, int(anchor_index) + int(offset)) for offset in history_offsets]
    result = np.stack([sparse_feature_at_time(observations, grid_times[i], width, height,
                                              max_observation_age) for i in indices])
    expected = (len(history_offsets), SPARSE_BALL_FEATURE_DIM)
    if result.shape != expected:
        raise ValueError(f"sparse history must have shape {expected}, got {result.shape}")
    return result.astype(np.float32)


def normalize_sparse_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(SPARSE_BALL_FEATURE_DIM)
    std = np.asarray(std, dtype=np.float32).reshape(SPARSE_BALL_FEATURE_DIM)
    if features.shape[-1] != SPARSE_BALL_FEATURE_DIM or np.any(std <= 0) or not np.isfinite(features).all():
        raise ValueError("invalid sparse features or normalization statistics")
    return (features - mean) / std


def sparse_metadata(width: int, height: int, max_observation_age: float, source_topic: str,
                    history_offsets: Sequence[int] = SPARSE_BALL_HISTORY_OFFSETS) -> dict:
    return {
        "input_modality": "sparse_ball", "sparse_feature_dim": SPARSE_BALL_FEATURE_DIM,
        "sparse_feature_names": list(SPARSE_BALL_FEATURE_NAMES),
        "sparse_history_offsets": list(history_offsets), "image_width": int(width),
        "image_height": int(height), "coordinate_convention": SPARSE_BALL_COORDINATE_CONVENTION,
        "velocity_convention": SPARSE_BALL_VELOCITY_CONVENTION,
        "max_observation_age_sec": float(max_observation_age), "ball_source_topic": str(source_topic),
        "source_timestamp_policy": SPARSE_BALL_TIMESTAMP_POLICY,
        "missing_observation_policy": "hold_last_position_zero_velocity_valid_zero_when_stale_zero_before_first",
    }
