import numpy as np
import pytest
import torch
import h5py
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detr"))
if "roma.mappings" not in sys.modules:
    roma = types.ModuleType("roma")
    mappings = types.ModuleType("roma.mappings")
    mappings.special_gramschmidt = lambda value: value
    roma.mappings = mappings
    sys.modules["roma"] = roma
    sys.modules["roma.mappings"] = mappings

from sparse_ball import (BallObservation, SPARSE_BALL_FEATURE_NAMES, build_sparse_history,
                         normalize_pixel, sparse_feature_at_time)
from intercept_rollout_contract import validate_intercept_stats_and_config
from policy import ACTPolicy
from utils import EpisodicInterceptDataset, get_intercept_norm_stats


def test_coordinate_feature_order_and_causal_velocity():
    assert SPARSE_BALL_FEATURE_NAMES == ("u", "v", "du_dt", "dv_dt", "valid", "observation_age")
    assert normalize_pixel(0, 0, 641, 481) == (-1.0, -1.0)
    obs = [BallObservation(1.0, 0, 0), BallObservation(2.0, 640, 480),
           BallObservation(3.0, 123, 234)]
    feature = sparse_feature_at_time(obs, 2.5, 641, 481, 1.0)
    np.testing.assert_allclose(feature, [1, 1, 2, 2, 1, .5])
    # The future observation at 3.0 must not affect position or velocity.
    np.testing.assert_allclose(feature, sparse_feature_at_time(obs[:2], 2.5, 641, 481, 1.0))


def test_missing_stale_and_history_offsets():
    obs = [BallObservation(1.0, 320, 240)]
    missing = sparse_feature_at_time(obs, .5, 641, 481, .2)
    np.testing.assert_allclose(missing, [0, 0, 0, 0, 0, .2])
    stale = sparse_feature_at_time(obs, 2.0, 641, 481, .2)
    assert stale[4] == 0 and stale[5] == pytest.approx(.2)
    grid = np.arange(10, dtype=float)
    history = build_sparse_history(obs, grid, 8, 641, 481, .2)
    assert history.shape == (3, 6)
    # offsets [-6,-3,0] select policy times 2,5,8.
    np.testing.assert_allclose(history[:, 5], [.2, .2, .2])


def _config():
    return dict(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_queries=30,
                kl_weight=1, hidden_dim=32, dim_feedforward=64, enc_layers=1, dec_layers=1,
                nheads=4, camera_names=['sparse_ball'], state_dim=21, action_dim=1,
                use_bce_last_action_dim=False, input_modality='sparse_ball',
                sparse_feature_dim=6, sparse_history_length=3, sparse_mean=[0]*6,
                sparse_std=[1]*6, device='cpu')


def test_sparse_model_has_no_backbone_and_output_contract():
    policy = ACTPolicy(_config())
    assert policy.model.backbones is None
    assert not any('backbone' in name for name, _ in policy.model.named_parameters())
    output = policy(torch.zeros(2, 21), torch.zeros(2, 3, 6))
    assert output.shape == (2, 30, 1)
    with pytest.raises(ValueError, match=r'\[B,3,6\]'):
        policy(torch.zeros(1, 21), torch.zeros(1, 1, 9, 32, 32))


def test_policy_wrapper_applies_sparse_normalization():
    config = {**_config(), 'sparse_mean': [1,2,3,4,0,.1],
              'sparse_std': [2,2,2,2,1,.1]}
    policy = ACTPolicy(config)
    raw = torch.tensor([[[3.,4.,5.,6.,1.,.2]]]).repeat(1,3,1)
    expected = torch.tensor([[[1.,1.,1.,1.,1.,1.]]]).repeat(1,3,1)
    torch.testing.assert_close(policy.preprocess_image(raw), expected)


def test_checkpoint_modality_contract_and_sparse_stats():
    stats = dict(data_mode='intercept', raw_qpos_dim=7, state_dim=21, action_dim=1,
                 qpos_history_offsets=[-6,-3,0], qpos_flatten_order='oldest_to_newest',
                 action_type='measured_tcp_s_delta', action_representation='future_delta_relative_to_anchor',
                 action_anchor_offset=0, action_first_target_offset=1,
                 action_positive_direction='robot_base_positive_x', action_units='m',
                 input_modality='sparse_ball', sparse_feature_dim=6,
                 sparse_feature_names=list(SPARSE_BALL_FEATURE_NAMES), sparse_history_offsets=[-6,-3,0],
                 image_width=641, image_height=481,
                 coordinate_convention='normalized_image_coordinates_minus1_to_plus1',
                 velocity_convention='normalized_image_coordinates_per_second', max_observation_age_sec=.2,
                 ball_source_topic='/ball_tracker2/ball_2d_px',
                 source_timestamp_policy='source_header_timestamp_causal_at_or_before_policy_time',
                 missing_observation_policy='hold_last_position_zero_velocity_valid_zero_when_stale_zero_before_first',
                 qpos_mean=np.zeros(21), qpos_std=np.ones(21), action_mean=np.zeros(1),
                 action_std=np.ones(1), sparse_mean=np.zeros(6), sparse_std=np.ones(6))
    arrays = validate_intercept_stats_and_config(stats, _config(), 30)
    assert arrays['sparse_mean'].shape == (6,)
    with pytest.raises(ValueError, match='modality'):
        validate_intercept_stats_and_config(stats, {**_config(), 'input_modality': 'wat'}, 30)


def test_synthetic_sparse_dataset_sample(tmp_path):
    path = tmp_path / 'episode_0.hdf5'
    with h5py.File(path, 'w') as root:
        root.attrs.update(action_type='measured_tcp_s_absolute', action_representation='absolute',
                          action_positive_direction='robot_base_positive_x', input_modality='sparse_ball',
                          sparse_history_offsets=np.asarray([-6,-3,0]), image_width=641, image_height=481,
                          coordinate_convention='normalized_image_coordinates_minus1_to_plus1',
                          velocity_convention='normalized_image_coordinates_per_second',
                          max_observation_age_sec=.2, ball_source_topic='/ball_tracker2/ball_2d_px',
                          source_timestamp_policy='source_header_timestamp_causal_at_or_before_policy_time')
        root.attrs['missing_observation_policy'] = 'hold_last_position_zero_velocity_valid_zero_when_stale_zero_before_first'
        root.attrs['sparse_feature_names'] = np.asarray(SPARSE_BALL_FEATURE_NAMES, dtype=h5py.string_dtype())
        root.create_dataset('/action', data=np.arange(10, dtype=np.float32)[:,None])
        root.create_dataset('/observations/qpos', data=np.zeros((10,7), np.float32))
        root.create_dataset('/observations/timestamps', data=np.arange(10)/30.)
        root.create_dataset('/observations/sparse_ball', data=np.zeros((10,6), np.float32))
    stats = get_intercept_norm_stats([str(path)], 30, input_modality='sparse_ball')
    sample = EpisodicInterceptDataset([str(path)], ['sparse_ball'], 30, stats,
                                      input_modality='sparse_ball', image_size=32)[0]
    assert sample[0].shape == (3,6)
    assert sample[1].shape == (21,)
    assert sample[2].shape == (30,1)
