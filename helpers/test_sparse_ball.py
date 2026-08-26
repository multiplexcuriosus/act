"""Focused tests for the canonical four-feature sparse-ball contract."""

import pathlib
import pickle

import numpy as np
import pytest

from sparse_ball import (
    SPARSE_FEATURE_NAMES, policy_period_ns, policy_period_sec,
    SparsePoint, construct_causal_sparse_history, construct_causal_sparse_window,
    construct_sparse_features, qpos_history_offsets_for_window,
    default_sparse_topic, sparse_dataset_paths, sparse_history_offsets_frames,
    resolve_sparse_checkpoint_contract, resolve_sparse_topic, validate_policy_rate,
    validate_sparse_checkpoint_contract,
)
from intercept_rollout_contract import validate_normalization_stats


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


def test_repeated_policy_ticks_reuse_source_and_increase_age():
    points = [SparsePoint(0.95, 30, 40)]
    first = construct_causal_sparse_history(
        points, 1.00, (0.0,), 320, 320, 0.20,
    )
    second = construct_causal_sparse_history(
        points, 1.05, (0.0,), 320, 320, 0.20,
    )
    assert first[0, 2] == second[0, 2] == 1
    assert first[0, 3] == pytest.approx(0.05)
    assert second[0, 3] == pytest.approx(0.10)
    np.testing.assert_allclose(first[0, :2], second[0, :2])


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


def test_sparse_topic_defaults_mismatches_and_custom_override():
    assert resolve_sparse_topic("rgb", None) == "/ball_tracker2/ball_2d_px"
    assert resolve_sparse_topic("event", None) == "/openmv_cam/event_tracker/ball_2d_px"
    with pytest.raises(ValueError, match=r"source='rgb'.*event_tracker"):
        resolve_sparse_topic("rgb", "/openmv_cam/event_tracker/ball_2d_px")
    with pytest.raises(ValueError, match=r"source='event'.*ball_tracker2"):
        resolve_sparse_topic("event", "/ball_tracker2/ball_2d_px")
    assert resolve_sparse_topic("rgb", "/custom/sparse_point") == "/custom/sparse_point"


def _valid_normalization_stats():
    return {
        "qpos_mean": np.zeros(21, dtype=np.float32),
        "qpos_std": np.ones(21, dtype=np.float32),
        "action_mean": np.zeros(1, dtype=np.float32),
        "action_std": np.ones(1, dtype=np.float32),
        "sparse_mean": np.zeros(4, dtype=np.float32),
        "sparse_std": np.ones(4, dtype=np.float32),
    }


def test_valid_sparse_normalization_statistics():
    arrays = validate_normalization_stats(
        _valid_normalization_stats(), include_sparse=True,
    )
    assert arrays["qpos_mean"].shape == (21,)
    assert arrays["action_std"].shape == (1,)
    assert arrays["sparse_mean"].shape == arrays["sparse_std"].shape == (4,)


@pytest.mark.parametrize(
    "key", ["qpos_mean", "qpos_std", "action_mean", "action_std",
            "sparse_mean", "sparse_std"],
)
def test_sparse_normalization_statistics_require_every_array(key):
    stats = _valid_normalization_stats()
    stats.pop(key)
    with pytest.raises(ValueError, match=key):
        validate_normalization_stats(stats, include_sparse=True)


@pytest.mark.parametrize(
    "key,bad_shape",
    [("qpos_mean", (20,)), ("qpos_std", (1, 21)),
     ("action_mean", (2,)), ("action_std", (1, 1)),
     ("sparse_mean", (3,)), ("sparse_std", (1, 4))],
)
def test_sparse_normalization_statistics_reject_wrong_shapes(key, bad_shape):
    stats = _valid_normalization_stats()
    stats[key] = np.ones(bad_shape, dtype=np.float32)
    with pytest.raises(ValueError, match=key):
        validate_normalization_stats(stats, include_sparse=True)


@pytest.mark.parametrize(
    "key", ["qpos_mean", "qpos_std", "action_mean", "action_std",
            "sparse_mean", "sparse_std"],
)
def test_sparse_normalization_statistics_reject_nonfinite_values(key):
    stats = _valid_normalization_stats()
    stats[key] = stats[key].copy()
    stats[key].flat[0] = np.nan
    with pytest.raises(ValueError, match=key):
        validate_normalization_stats(stats, include_sparse=True)


@pytest.mark.parametrize("key,bad", [("qpos_std", 0.0), ("qpos_std", -1.0),
                                      ("action_std", 0.0), ("action_std", -1.0),
                                      ("sparse_std", 0.0), ("sparse_std", -1.0)])
def test_sparse_normalization_statistics_require_positive_std(key, bad):
    stats = _valid_normalization_stats()
    stats[key] = stats[key].copy()
    stats[key].flat[0] = bad
    with pytest.raises(ValueError, match=key):
        validate_normalization_stats(stats, include_sparse=True)


@pytest.mark.parametrize(
    "rate,period_ns,offsets,chunk_size",
    [(30, 33333333, (-6, -3, 0), 30),
     (60, 16666667, (-12, -6, 0), 60)],
)
def test_explicit_30_60_hz_contract(rate, period_ns, offsets, chunk_size):
    assert policy_period_ns(rate) == period_ns
    assert policy_period_sec(rate) == pytest.approx(1.0 / rate)
    assert sparse_history_offsets_frames(rate) == offsets
    assert SPARSE_FEATURE_NAMES == (
        "u_norm", "v_norm", "valid", "observation_age_sec"
    )
    assert chunk_size == rate  # one policy-second action chunk


@pytest.mark.parametrize("rate", [0, 29, 31, 120, 30.5])
def test_unsupported_policy_rates_are_rejected(rate):
    with pytest.raises(ValueError, match="policy rate"):
        validate_policy_rate(rate)


def _checkpoint_contract(rate=30):
    offsets = [-6, -3, 0] if rate == 30 else [-12, -6, 0]
    return {
        "input_modality": "sparse_ball", "state_dim": 21, "action_dim": 1,
        "policy_rate_hz": np.int64(rate),
        "chunk_size": np.asarray(rate),
        "qpos_history_offsets": tuple(offsets),
        "sparse_history_offsets_frames": np.asarray(offsets),
        "sparse_source": "rgb",
        "sparse_feature_dim": np.int64(4),
        "sparse_history_length": np.asarray(3),
    }


def test_m_window_boundaries_padding_overflow_invalid_and_future_exclusion():
    points = [
        SparsePoint(0.899999, 1, 1),
        SparsePoint(0.900000, 10, 20),
        SparsePoint(0.950000, float("nan"), float("nan"), valid=0),
        SparsePoint(1.000000, 30, 40),
        SparsePoint(1.000001, 50, 60),
    ]
    window, info = construct_causal_sparse_window(
        points, 1.0, 100.0, 32, 320, 320, return_info=True)
    assert window.shape == (32, 4)
    assert info["selected_count"] == 3
    assert info["padding_count"] == 29
    np.testing.assert_array_equal(window[-3:, 2], [1, 0, 1])
    np.testing.assert_allclose(window[-3:, 3], [0.1, 0.05, 0.0], atol=1e-7)

    overflow, overflow_info = construct_causal_sparse_window(
        [SparsePoint(0.9 + i * 0.001, i, i) for i in range(40)],
        1.0, 100.0, 32, 320, 320, return_info=True)
    assert overflow.shape == (32, 4)
    assert overflow_info["overflow_count"] == 8
    assert overflow_info["selected_timestamps"][0] == pytest.approx(0.908)


def _m_window_contract(**overrides):
    stats = {
        "input_modality": "sparse_ball", "sparse_source": "event",
        "sparse_feature_dim": 4, "sparse_history_length": 32,
        "sparse_history_mode": "m_window", "history_horizon_ms": 100.0,
        "sparse_history_capacity": 32, "state_dim": 49, "action_dim": 1,
        "qpos_history_offsets": list(range(-6, 1)),
        "policy_rate_hz": 60, "chunk_size": 60,
    }
    stats.update(overrides)
    return stats


def test_m_window_checkpoint_contract_accepts_exact_contract_and_rejects_legacy_shapes():
    assert qpos_history_offsets_for_window(60, 100) == tuple(range(-6, 1))
    resolved = resolve_sparse_checkpoint_contract(
        _m_window_contract(), 60, 60, "event", "m_window", 100, 32, 49)
    assert resolved["sparse_history_length"] == 32
    for bad in ({"state_dim": 21}, {"sparse_history_length": 3}):
        with pytest.raises(ValueError):
            resolve_sparse_checkpoint_contract(
                _m_window_contract(**bad), 60, 60, "event",
                "m_window", 100, 32, 49)


@pytest.mark.parametrize("rate", [30, 60])
def test_modern_sparse_checkpoint_contract_succeeds(rate):
    stats = _checkpoint_contract(rate)
    resolved = resolve_sparse_checkpoint_contract(stats, rate, rate, "rgb")
    assert resolved["policy_rate_hz"] == rate
    assert resolved["chunk_size"] == rate
    assert resolved["qpos_history_offsets"] == (
        [-6, -3, 0] if rate == 30 else [-12, -6, 0]
    )
    assert resolved["inferred_legacy_fields"] == []
    assert isinstance(stats["sparse_history_offsets_frames"], np.ndarray)


def test_legacy_30hz_contract_infers_missing_and_none_fields_once():
    stats = _checkpoint_contract()
    missing = ["policy_rate_hz", "chunk_size", "qpos_history_offsets",
               "sparse_history_offsets_frames"]
    for key in missing:
        if key == "chunk_size":
            stats[key] = None
        else:
            stats.pop(key)
    with pytest.warns(RuntimeWarning, match="Legacy 30 Hz.*metadata missing") as caught:
        resolved = resolve_sparse_checkpoint_contract(stats, 30, 30, "rgb")
    assert len(caught) == 1
    assert resolved["policy_rate_hz"] == resolved["chunk_size"] == 30
    assert resolved["qpos_history_offsets"] == [-6, -3, 0]
    assert resolved["sparse_history_offsets_frames"] == [-6, -3, 0]
    assert resolved["inferred_legacy_fields"] == missing
    assert stats["chunk_size"] is None


def test_legacy_contract_is_not_inferred_at_60hz():
    stats = _checkpoint_contract(60)
    for key in ("policy_rate_hz", "chunk_size", "qpos_history_offsets",
                "sparse_history_offsets_frames"):
        stats.pop(key)
    with pytest.raises(ValueError, match="lacks explicit 60 Hz sparse contract metadata"):
        resolve_sparse_checkpoint_contract(stats, 60, 60, "rgb")


@pytest.mark.parametrize(
    "rate,key,bad",
    [
        (30, "policy_rate_hz", 60), (60, "policy_rate_hz", 30),
        (30, "chunk_size", 60), (60, "chunk_size", 30),
        (30, "qpos_history_offsets", [-12, -6, 0]),
        (60, "qpos_history_offsets", [-6, -3, 0]),
        (30, "sparse_history_offsets_frames", [-12, -6, 0]),
        (60, "sparse_history_offsets_frames", [-6, -3, 0]),
        (30, "sparse_source", "event"),
        (30, "sparse_feature_dim", 6),
        (30, "sparse_history_length", 2),
    ],
)
def test_sparse_checkpoint_contract_rejects_mismatches(rate, key, bad):
    stats = _checkpoint_contract(rate)
    stats[key] = bad
    with pytest.raises(ValueError, match=key):
        resolve_sparse_checkpoint_contract(stats, rate, rate, "rgb")


def test_supplied_legacy_checkpoint_contract_smoke():
    path = pathlib.Path(
        "/home/jau/dyros/data/ckpts/intercept_sparse_grid_20260814_174207/"
        "rgb/sparse_rgb_uvnorm_valid_age_hist3_30hz_lr1e-5_bs8_kl1/"
        "dataset_stats.pkl"
    )
    with path.open("rb") as stream:
        stats = pickle.load(stream)
    with pytest.warns(RuntimeWarning, match="chunk_size"):
        resolved = resolve_sparse_checkpoint_contract(stats, 30, 30, "rgb")
    assert resolved["policy_rate_hz"] == 30
    assert resolved["chunk_size"] == 30
    assert resolved["qpos_history_offsets"] == [-6, -3, 0]
    assert resolved["sparse_history_offsets_frames"] == [-6, -3, 0]
    assert resolved["sparse_source"] == "rgb"
    assert resolved["inferred_legacy_fields"] == ["chunk_size"]
