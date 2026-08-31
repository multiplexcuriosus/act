"""Focused tests for the canonical four-feature sparse-ball contract."""

import numpy as np
import pytest

from sparse_ball import (
    SPARSE_FEATURE_NAMES, policy_period_ns, policy_period_sec,
    SparsePoint, construct_causal_sparse_history, construct_causal_sparse_window,
    construct_sparse_features, qpos_history_offsets_for_window,
    default_sparse_topic, sparse_dataset_paths, sparse_history_offsets_frames,
    resolve_sparse_checkpoint_contract, validate_policy_rate,
    validate_sparse_checkpoint_contract,
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


def test_m_window_is_causal_front_padded_and_uses_anchor_age():
    rows, info = construct_causal_sparse_window(
        [SparsePoint(.79, 1, 2), SparsePoint(.81, 10, 20),
         SparsePoint(.95, 30, 40, 0), SparsePoint(1.01, 50, 60)],
        1.0, 200, 4, 320, 320, return_info=True,
    )
    np.testing.assert_array_equal(rows[:2], np.zeros((2, 4), dtype=np.float32))
    assert rows[2, 2] == 1 and rows[2, 3] == pytest.approx(.19)
    assert rows[3, 2] == 0 and rows[3, 3] == pytest.approx(.05)
    np.testing.assert_allclose(info['selected_timestamps'], [.81, .95])
    assert not info['overflow']


def test_m_window_overflow_retains_newest_deterministically():
    points = [SparsePoint(.8 + i * .01, i, i) for i in range(21)]
    _, info = construct_causal_sparse_window(
        points, 1.0, 200, 3, 320, 320, return_info=True,
    )
    assert info['overflow'] and info['overflow_count'] == 18
    np.testing.assert_allclose(info['selected_timestamps'], [.98, .99, 1.0])


@pytest.mark.parametrize('rate,length,state_dim', [(30, 7, 49), (60, 13, 91)])
def test_m_window_qpos_offsets(rate, length, state_dim):
    offsets = qpos_history_offsets_for_window(rate, 200)
    assert offsets == tuple(range(-(length - 1), 1))
    assert 7 * len(offsets) == state_dim


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


def _checkpoint_rate_contract(rate):
    offsets = list(sparse_history_offsets_frames(rate))
    return {
        "policy_rate_hz": rate,
        "qpos_history_offsets": offsets,
        "chunk_size": rate,
        "sparse_history_offsets_frames": offsets,
        "sparse_source": "rgb",
        "sparse_feature_dim": 4,
        "sparse_history_length": 3,
    }


@pytest.mark.parametrize("rate", [30, 60])
def test_modern_sparse_checkpoint_rate_contract(rate):
    resolved = resolve_sparse_checkpoint_contract(
        _checkpoint_rate_contract(rate), rate, rate, "rgb"
    )
    assert resolved["legacy_inferred"] is False
    assert resolved["qpos_history_offsets"] == list(
        sparse_history_offsets_frames(rate)
    )


def test_legacy_30_hz_checkpoint_contract_is_inferred():
    stats = _checkpoint_rate_contract(30)
    missing = [
        "policy_rate_hz", "qpos_history_offsets", "chunk_size",
        "sparse_history_offsets_frames",
    ]
    for key in missing:
        stats.pop(key)
    with pytest.warns(RuntimeWarning, match="legacy 30 Hz sparse contract"):
        resolved = resolve_sparse_checkpoint_contract(stats, 30, 30, "rgb")
    assert resolved["legacy_inferred"] is True
    assert resolved["legacy_inferred_fields"] == missing
    assert resolved["qpos_history_offsets"] == [-6, -3, 0]
    assert resolved["chunk_size"] == 30


def test_legacy_checkpoint_metadata_is_never_inferred_at_60_hz():
    stats = _checkpoint_rate_contract(30)
    stats.pop("policy_rate_hz")
    stats.pop("qpos_history_offsets")
    with pytest.raises(ValueError, match="inferred only at 30 Hz.*retrain"):
        resolve_sparse_checkpoint_contract(stats, 60, 60, "rgb")


@pytest.mark.parametrize(
    "requested_rate,saved_rate", [(30, 60), (60, 30)]
)
def test_sparse_checkpoint_rate_mismatch_is_rejected(requested_rate, saved_rate):
    with pytest.raises(ValueError, match="mismatch for policy_rate_hz"):
        resolve_sparse_checkpoint_contract(
            _checkpoint_rate_contract(saved_rate), requested_rate,
            requested_rate, "rgb",
        )


def test_sparse_checkpoint_history_and_chunk_mismatches_are_rejected():
    stats = _checkpoint_rate_contract(60)
    stats["qpos_history_offsets"] = [-6, -3, 0]
    with pytest.raises(ValueError, match="mismatch for qpos_history_offsets"):
        resolve_sparse_checkpoint_contract(stats, 60, 60, "rgb")
    stats = _checkpoint_rate_contract(60)
    stats["chunk_size"] = 30
    with pytest.raises(ValueError, match="mismatch for chunk_size"):
        resolve_sparse_checkpoint_contract(stats, 60, 60, "rgb")


def test_m_window_rejects_source_early_but_receipt_late():
    points = [SparsePoint(0.90, 10, 20, 1, availability_timestamp=1.01)]
    window, info = construct_causal_sparse_window(
        points, 1.0, 200, 4, 1280, 720, return_info=True)
    assert info["selected_count"] == 0
    assert not window[:, 2].any()
    later, info = construct_causal_sparse_window(
        points, 1.02, 200, 4, 1280, 720, return_info=True)
    assert info["selected_count"] == 1
    assert later[-1, 2] == 1
    assert later[-1, 3] == pytest.approx(0.12)
