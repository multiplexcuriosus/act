import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "detr"))

from intercept_rollout_contract import (EXPECTED_INTERCEPT_METADATA,
                                        select_qpos_history_at_targets,
                                        validate_intercept_stats_and_config)
from policy import ACTPolicy
from sparse_ball import (BallObservation, SPARSE_BALL_FEATURE_NAMES, build_sparse_history,
                         history_target_times, select_sparse_observation)


def test_sparse_contract_and_exact_history_targets():
    assert SPARSE_BALL_FEATURE_NAMES == ("u_norm", "v_norm", "valid", "observation_age")
    assert history_target_times(10.0) == (9.8, 9.9, 10.0)


@pytest.mark.parametrize("width,height", [(1280, 720), (320, 240)])
def test_rgb_and_event_build_three_by_four_with_modality_dimensions(width, height):
    targets = history_target_times(1.0)
    observations = [BallObservation(t, width / 2, height / 4) for t in targets]
    result = build_sparse_history(observations, targets, width, height, 0.10)
    assert result.shape == (3, 4)
    np.testing.assert_allclose(result[:, :2], [[.5, .25]] * 3)
    np.testing.assert_allclose(result[:, 2:], [[1., 0.]] * 3, atol=1e-7)


def test_invalid_missing_stale_and_future_samples_use_exact_sentinel():
    sentinel = np.asarray([0., 0., 0., .1], np.float32)
    assert np.array_equal(select_sparse_observation([], 1., 100, 100, .1).feature, sentinel)
    stale = select_sparse_observation([BallObservation(.8, 10, 20)], 1., 100, 100, .1)
    assert np.array_equal(stale.feature, sentinel)
    invalid = select_sparse_observation([BallObservation(1., 10, 20, False)], 1., 100, 100, .1)
    assert np.array_equal(invalid.feature, sentinel)
    future = select_sparse_observation([BallObservation(1.01, 10, 20)], 1., 100, 100, .1)
    assert future.source_timestamp is None
    assert np.array_equal(future.feature, sentinel)


def test_header_time_reuse_increases_age_and_later_sample_does_not_leak_backward():
    observations = [BallObservation(.85, 10, 10), BallObservation(.99, 90, 90)]
    earlier = select_sparse_observation(observations, .90, 100, 100, .1)
    current = select_sparse_observation(observations, 1.00, 100, 100, .1)
    later_tick = select_sparse_observation(observations, 1.05, 100, 100, .1)
    assert earlier.source_timestamp == .85
    assert current.source_timestamp == .99
    assert later_tick.observation_age > current.observation_age
    assert earlier.feature[0] == pytest.approx(.1)


def test_sparse_selection_uses_timestamps_not_message_indices():
    observations = [BallObservation(.50, 1, 1), BallObservation(.79, 2, 2),
                    BallObservation(.795, 3, 3), BallObservation(.90, 4, 4)]
    result = build_sparse_history(observations, history_target_times(1.0), 10, 10, .1)
    assert result[0, 0] == pytest.approx(.3)
    assert result[1, 0] == pytest.approx(.4)
    assert result[2, 2] == 1.0


def test_qpos_history_is_independently_causal_and_missing_is_safe():
    stamps = [.79, .795, .89, .91, .99]
    samples = [np.full(7, i, np.float32) for i in range(len(stamps))]
    qpos, selected = select_qpos_history_at_targets(stamps, samples, (.8, .9, 1.0))
    assert selected == (.795, .89, .99)
    np.testing.assert_array_equal(qpos.reshape(3, 7)[:, 0], [1, 2, 4])
    with pytest.raises(ValueError, match="index=0"):
        select_qpos_history_at_targets([.9], [samples[0]], (.8, .9, 1.0))


def _model_config(feature_dim=4):
    return dict(lr=1e-4, num_queries=3, kl_weight=1, hidden_dim=32,
                dim_feedforward=64, lr_backbone=1e-5, backbone='resnet18',
                enc_layers=1, dec_layers=1, nheads=4, camera_names=['sparse_ball'],
                state_dim=21, action_dim=1, use_bce_last_action_dim=False,
                input_modality='sparse_ball', sparse_feature_dim=feature_dim,
                sparse_history_length=3, sparse_mean=[0] * feature_dim,
                sparse_std=[1] * feature_dim, device='cpu')


def test_sparse_forward_accepts_b_three_four_without_visual_backbone():
    policy = ACTPolicy(_model_config())
    assert policy.model.backbones is None
    output = policy(torch.zeros(2, 21), torch.zeros(2, 3, 4))
    assert output.shape == (2, 3, 1)


def test_six_feature_configuration_is_rejected():
    with pytest.raises(ValueError, match="features=4"):
        ACTPolicy(_model_config(6))


def _sparse_stats(source="rgb", width=1280, height=720):
    stats = dict(EXPECTED_INTERCEPT_METADATA)
    for key in ("rgb_history_frames", "rgb_history_offsets", "rgb_frame_order", "image_channels"):
        stats.pop(key)
    stats.update(input_modality="sparse_ball", sparse_source=source,
                 sparse_feature_dim=4,
                 sparse_feature_names=list(SPARSE_BALL_FEATURE_NAMES),
                 sparse_history_length=3, sparse_history_offsets_sec=[-.2, -.1, 0.],
                 max_observation_age_sec=.1, image_width=width, image_height=height,
                 chunk_size=30, qpos_mean=np.zeros(21), qpos_std=np.ones(21),
                 action_mean=np.zeros(1), action_std=np.ones(1),
                 sparse_mean=np.zeros(4), sparse_std=np.ones(4))
    return stats


def test_checkpoint_contract_accepts_four_features_and_rejects_source_or_six_features():
    config = {**_model_config(), "num_queries": 30}
    arrays = validate_intercept_stats_and_config(
        _sparse_stats(), config, 30,
        {"sparse_source": "rgb", "max_observation_age_sec": .1,
         "image_width": 1280, "image_height": 720},
    )
    assert arrays["sparse_mean"].shape == (4,)
    with pytest.raises(ValueError, match="sparse_source"):
        validate_intercept_stats_and_config(
            _sparse_stats("rgb"), config, 30, {"sparse_source": "event"}
        )
    bad = _sparse_stats()
    bad["sparse_feature_dim"] = 6
    with pytest.raises(ValueError, match="sparse_feature_dim"):
        validate_intercept_stats_and_config(bad, config, 30)


def test_sparse_checkpoint_state_dict_roundtrip():
    original = ACTPolicy(_model_config())
    restored = ACTPolicy(_model_config())
    status = restored.load_state_dict(original.state_dict())
    assert not status.missing_keys and not status.unexpected_keys
