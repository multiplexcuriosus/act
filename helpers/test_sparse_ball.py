"""Focused tests for the canonical four-feature sparse-ball runtime contract."""

import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "detr"))

from intercept_rollout_contract import (EXPECTED_INTERCEPT_METADATA,
                                        select_latest_index_at_or_before,
                                        select_qpos_history_at_targets,
                                        validate_intercept_stats_and_config)
from policy import ACTPolicy
from sparse_ball import (SPARSE_FEATURE_NAMES, SparsePoint,
                         construct_causal_sparse_history, construct_sparse_features,
                         rate_contract)


@pytest.mark.parametrize("width,height", [(1280, 720), (320, 320)])
def test_rgb_and_event_pixel_centers_normalize_to_minus_one_plus_one(width, height):
    points = np.asarray([[0, 0], [(width - 1) / 2, (height - 1) / 2],
                         [width - 1, height - 1]])
    features = construct_sparse_features(
        [1.0, 1.1, 1.2], points, [1, 1, 1], [.95, 1.08, 1.2],
        width, height, .1,
    )
    assert SPARSE_FEATURE_NAMES == ("u", "v", "valid", "observation_age")
    assert features.shape == (3, 4)
    np.testing.assert_allclose(features[:, :2], [[-1, -1], [0, 0], [1, 1]])
    np.testing.assert_allclose(features[:, 2:], [[1, .05], [1, .02], [1, 0]], atol=1e-6)


def test_exact_targets_latest_before_future_rejection_and_timestamp_selection():
    points = [SparsePoint(.50, 1, 1), SparsePoint(.79, 2, 2),
              SparsePoint(.795, 3, 3), SparsePoint(.90, 4, 4),
              SparsePoint(1.01, 319, 319)]
    history = construct_causal_sparse_history(
        points, 1.0, (-.2, -.1, 0.), 320, 320, .21,
    )
    expected_u = 2 * np.asarray([3, 4, 4]) / 319 - 1
    np.testing.assert_allclose(history[:, 0], expected_u)
    np.testing.assert_allclose(history[:, 3], [.005, 0., .1], atol=1e-6)


def test_invalid_missing_stale_nonfinite_out_of_range_use_exact_sentinel():
    result = construct_sparse_features(
        [1, 1, 1, 1, 1], [[1, 2], [1, 2], [np.nan, 2], [-1, 2], [320, 2]],
        [0, 1, 1, 1, 1], [.99, .8, .99, .99, .99], 320, 320, .1,
    )
    sentinel = np.asarray([0., 0., 0., .1], np.float32)
    np.testing.assert_array_equal(result, np.tile(sentinel, (5, 1)))
    missing = construct_causal_sparse_history([], 1., (-.2, -.1, 0.), 320, 320, .1)
    np.testing.assert_array_equal(missing, np.tile(sentinel, (3, 1)))


def test_reused_detection_age_increases_on_later_policy_ticks_and_stales():
    points = [SparsePoint(1., 100, 100)]
    first = construct_causal_sparse_history(points, 1.05, (-.2, -.1, 0.), 320, 320, .1)
    later = construct_causal_sparse_history(points, 1.10, (-.2, -.1, 0.), 320, 320, .1)
    stale = construct_causal_sparse_history(points, 1.11, (-.2, -.1, 0.), 320, 320, .1)
    assert first[-1, 3] == pytest.approx(.05)
    assert later[-1, 3] == pytest.approx(.10)
    np.testing.assert_array_equal(stale[-1], np.asarray([0., 0., 0., .1], np.float32))


def test_qpos_and_tcp_selection_are_independently_causal():
    stamps = [.79, .795, .89, .91, .99, 1.01]
    samples = [np.full(7, i, np.float32) for i in range(len(stamps))]
    qpos, selected = select_qpos_history_at_targets(stamps, samples, (.8, .9, 1.0))
    assert selected == (.795, .89, .99)
    np.testing.assert_array_equal(qpos.reshape(3, 7)[:, 0], [1, 2, 4])
    assert select_latest_index_at_or_before([.9, .99, 1.01], 1.) == 1
    with pytest.raises(ValueError, match="target_timestamp"):
        select_qpos_history_at_targets([.9], [samples[0]], (.8, .9, 1.0))


def _model_config(feature_dim=4):
    return dict(lr=1e-4, num_queries=30, kl_weight=1, hidden_dim=32,
                dim_feedforward=64, lr_backbone=1e-5, backbone='resnet18',
                enc_layers=1, dec_layers=1, nheads=4, camera_names=['sparse_ball'],
                state_dim=21, action_dim=1, use_bce_last_action_dim=False,
                input_modality='sparse_ball', sparse_feature_dim=feature_dim,
                sparse_history_length=3, sparse_mean=[0] * feature_dim,
                sparse_std=[1] * feature_dim, device='cpu')


def _sparse_stats(source="rgb", width=1280, height=720):
    stats = dict(EXPECTED_INTERCEPT_METADATA)
    for key in ("rgb_history_frames", "rgb_history_offsets", "rgb_frame_order", "image_channels"):
        stats.pop(key)
    stats.update(input_modality="sparse_ball", sparse_source=source,
                 sparse_feature_dim=4, sparse_feature_names=list(SPARSE_FEATURE_NAMES),
                 sparse_history_length=3, sparse_history_offsets_sec=[-.2, -.1, 0.],
                 sparse_max_observation_age_sec=.1,
                 sparse_image_width=width, sparse_image_height=height, chunk_size=30,
                 qpos_mean=np.zeros(21), qpos_std=np.ones(21),
                 action_mean=np.zeros(1), action_std=np.ones(1),
                 sparse_mean=np.zeros(4), sparse_std=np.ones(4))
    return stats


def test_four_feature_checkpoint_loads_and_six_feature_or_source_mismatch_rejected():
    config = _model_config()
    runtime = {"sparse_source": "rgb", "max_observation_age_sec": .1,
               "image_width": 1280, "image_height": 720}
    arrays = validate_intercept_stats_and_config(_sparse_stats(), config, 30, runtime)
    assert arrays["sparse_mean"].shape == (4,)
    stats_without_redundant_chunk_size = _sparse_stats()
    stats_without_redundant_chunk_size.pop("chunk_size")
    validate_intercept_stats_and_config(stats_without_redundant_chunk_size, config, 30, runtime)
    policy = ACTPolicy(config)
    assert policy.model.backbones is None
    assert policy(torch.zeros(1, 21), torch.zeros(1, 3, 4)).shape == (1, 30, 1)
    bad = _sparse_stats()
    bad["sparse_feature_dim"] = 6
    with pytest.raises(ValueError, match="sparse_feature_dim"):
        validate_intercept_stats_and_config(bad, config, 30, runtime)
    with pytest.raises(ValueError, match="sparse_source"):
        validate_intercept_stats_and_config(_sparse_stats("event"), config, 30, runtime)


def test_rate_contract_and_checkpoint_rate_mismatch():
    assert rate_contract(30)[:2] == ((-6, -3, 0), 30)
    assert rate_contract(60)[:2] == ((-12, -6, 0), 60)
    stats = _sparse_stats()
    stats.update(policy_rate_hz=30, sparse_history_offsets_frames=[-6, -3, 0])
    runtime = {"sparse_source": "rgb", "max_observation_age_sec": .1,
               "image_width": 1280, "image_height": 720,
               "policy_rate_hz": 60}
    with pytest.raises(ValueError, match="policy rate|policy_rate_hz|chunk_size"):
        validate_intercept_stats_and_config(stats, _model_config(), 60, runtime)
