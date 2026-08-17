"""Focused tests for the canonical four-feature sparse-ball contract."""

import numpy as np
import pytest

from sparse_ball import (
    SPARSE_FEATURE_NAMES, policy_period_ns, policy_period_sec,
    SparsePoint, construct_causal_sparse_history, construct_sparse_features,
    default_sparse_topic, sparse_dataset_paths, sparse_history_offsets_frames,
    validate_policy_rate, validate_sparse_checkpoint_contract,
)


@pytest.mark.parametrize("width,height", [(1280, 720), (320, 320)])
def test_sparse_features_normalize_and_use_source_age(width, height):
    times = np.asarray([1.0, 1.1, 1.2])
    points = np.asarray([[0, 0], [(width - 1) / 2, (height - 1) / 2],
                         [width - 1, height - 1]])
    features = construct_sparse_features(
        times, points, [1, 1, 1], [0.95, 1.08, 1.2], width, height, 0.10
    )
    assert features.shape == (3, 4)
    np.testing.assert_allclose(features[:, :2], [[-1, -1], [0, 0], [1, 1]])
    np.testing.assert_allclose(features[:, 2], 1)
    np.testing.assert_allclose(features[:, 3], [0.05, 0.02, 0], atol=1e-6)


def test_invalid_stale_future_and_nonfinite_rows_use_canonical_fill():
    features = construct_sparse_features(
        [1, 1, 1, 1], [[1, 2], [1, 2], [np.nan, 2], [1, 2]],
        [0, 1, 1, 1], [0.99, 0.8, 0.99, 1.01], 320, 320, 0.10,
    )
    np.testing.assert_array_equal(
        features, np.tile(np.asarray([0, 0, 0, 0.10], "f4"), (4, 1))
    )


def test_causal_history_never_selects_future_point():
    history = construct_causal_sparse_history(
        [SparsePoint(0.79, 10, 20), SparsePoint(0.95, 30, 40),
         SparsePoint(1.01, 300, 300)],
        1.0, (-0.2, -0.1, 0.0), 320, 320, 0.20,
    )
    assert history.shape == (3, 4)
    expected_u = 2 * np.asarray([10, 10, 30]) / 319 - 1
    np.testing.assert_allclose(history[:, 0], expected_u)
    assert np.all(history[:, 2] == 1)


def test_source_paths_and_checkpoint_contract_are_source_specific():
    assert default_sparse_topic("rgb") == "/ball_tracker2/ball_2d_px"
    assert default_sparse_topic("event") == "/openmv_cam/event_tracker/ball_2d_px"
    assert sparse_dataset_paths("rgb") == (
        "/observations/sparse_tracking/rgb_2d_px",
        "/observations/sparse_tracking/rgb_valid",
        "/observations/sparse_tracking/rgb_source_timestamps",
    )
    assert sparse_dataset_paths("event")[2].endswith("event_source_timestamps")
    stats = {
        "input_modality": "sparse_ball", "sparse_source": "rgb",
        "sparse_feature_dim": 4, "sparse_history_length": 3,
        "sparse_feature_names": list(SPARSE_FEATURE_NAMES),
        "sparse_image_width": 1280, "sparse_image_height": 720,
        "sparse_max_observation_age_sec": 0.10,
    }
    validate_sparse_checkpoint_contract(stats, "rgb", 1280, 720, 0.10)
    with pytest.raises(ValueError, match="sparse_source"):
        validate_sparse_checkpoint_contract(stats, "event", 1280, 720, 0.10)
    with pytest.raises(ValueError, match="sparse_feature_dim"):
        validate_sparse_checkpoint_contract(
            {**stats, "sparse_feature_dim": 6}, "rgb", 1280, 720, 0.10
        )


@pytest.mark.parametrize(
    "rate,period_ns,offsets", [(30, 33333333, (-6, -3, 0)),
                               (60, 16666667, (-12, -6, 0))],
)
def test_explicit_30_60_hz_contract(rate, period_ns, offsets):
    assert policy_period_ns(rate) == period_ns
    assert policy_period_sec(rate) == pytest.approx(1.0 / rate)
    assert sparse_history_offsets_frames(rate) == offsets
    assert SPARSE_FEATURE_NAMES == (
        "u_norm", "v_norm", "valid", "observation_age_sec"
    )


@pytest.mark.parametrize("rate", [0, 29, 31, 120, 30.5])
def test_unsupported_policy_rates_are_rejected(rate):
    with pytest.raises(ValueError, match="policy rate"):
        validate_policy_rate(rate)
