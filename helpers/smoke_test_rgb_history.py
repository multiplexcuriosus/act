import os
import sys
import tempfile
from collections import deque

import h5py
import numpy as np
import torch
import torchvision


HELPERS_DIR = os.path.dirname(__file__)
ACT_DIR = os.path.dirname(HELPERS_DIR)
if ACT_DIR not in sys.path:
    sys.path.insert(0, ACT_DIR)

from detr.models.backbone import adapt_resnet_input_channels
import detr.models.backbone as backbone_mod
from imitate_episodes import get_image
from policy import ACTPolicy, _build_image_normalizer
from utils import EpisodicJointDataset, _prepare_rgb_history_frames


def make_episode(path, episode_len=3, image_shape=(12, 12, 3)):
    qpos = np.stack([np.array([step, step + 100], dtype=np.float32) for step in range(episode_len)], axis=0)
    action = np.stack([np.array([step, step + 200], dtype=np.float32) for step in range(episode_len)], axis=0)
    rgb = np.stack([
        np.full(image_shape, fill_value=step * 20, dtype=np.uint8)
        for step in range(episode_len)
    ], axis=0)
    event = np.stack([
        np.full(image_shape[:2], fill_value=step * 30, dtype=np.uint8)
        for step in range(episode_len)
    ], axis=0)

    with h5py.File(path, 'w') as root:
        root.attrs['sim'] = False
        root.create_dataset('/observations/qpos', data=qpos)
        root.create_dataset('/action', data=action)
        root.create_dataset('/observations/images/rgb', data=rgb)
        root.create_dataset('/observations/images/event', data=event)


def make_norm_stats(qpos_dim=2, action_dim=2):
    return {
        'action_mean': np.zeros(action_dim, dtype=np.float32),
        'action_std': np.ones(action_dim, dtype=np.float32),
        'qpos_mean': np.zeros(qpos_dim, dtype=np.float32),
        'qpos_std': np.ones(qpos_dim, dtype=np.float32),
    }


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f'{message}: expected {expected}, got {actual}')


def assert_allclose(actual, expected, message, atol=1e-6):
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(f'{message}: expected {expected}, got {actual}')


def assert_tensor_shape(tensor, expected_shape, message):
    if tuple(tensor.shape) != tuple(expected_shape):
        raise AssertionError(f'{message}: expected {expected_shape}, got {tuple(tensor.shape)}')


def test_dataset_shapes_and_order(tmpdir):
    episode_path = os.path.join(tmpdir, 'episode_0.hdf5')
    make_episode(episode_path)

    stats = make_norm_stats()
    dataset_default = EpisodicJointDataset(
        [episode_path],
        ['rgb'],
        chunk_size=2,
        norm_stats=stats,
        image_size=16,
    )
    image_data, qpos_data, action_data, is_pad = dataset_default[0]
    assert_tensor_shape(image_data, (1, 3, 16, 16), 'Default RGB sample shape')

    dataset_event = EpisodicJointDataset(
        [episode_path],
        ['event'],
        chunk_size=2,
        norm_stats=stats,
        image_size=16,
    )
    event_image_data, _, _, _ = dataset_event[0]
    assert_tensor_shape(event_image_data, (1, 3, 16, 16), 'Event sample shape remains unchanged')

    dataset_rgb_event = EpisodicJointDataset(
        [episode_path],
        ['rgb', 'event'],
        chunk_size=2,
        norm_stats=stats,
        image_size=16,
    )
    rgb_event_image_data, _, _, _ = dataset_rgb_event[0]
    assert_tensor_shape(rgb_event_image_data, (2, 3, 16, 16), 'RGB+event sample shape remains unchanged')

    dataset_temporal = EpisodicJointDataset(
        [episode_path],
        ['rgb'],
        chunk_size=2,
        norm_stats=stats,
        image_size=16,
        rgb_history_frames=3,
    )
    temporal_image_data, temporal_qpos, temporal_action, temporal_is_pad = dataset_temporal[0]
    assert_tensor_shape(temporal_image_data, (1, 9, 16, 16), 'Temporal RGB sample shape')
    grouped_means = temporal_image_data[0].reshape(3, 3, 16, 16).mean(dim=(1, 2, 3)).numpy() * 255.0
    assert_allclose(grouped_means, np.array([0.0, 20.0, 40.0], dtype=np.float32), 'Temporal RGB ordering oldest->newest', atol=0.5)
    assert_allclose(temporal_qpos.numpy(), np.array([2.0, 102.0], dtype=np.float32), 'Temporal qpos alignment')
    assert_allclose(temporal_action[0].numpy(), np.array([2.0, 202.0], dtype=np.float32), 'Temporal action alignment')
    assert_equal(bool(temporal_is_pad[0].item()), False, 'Temporal action first step not padded')


def test_shared_augmentation():
    frame = np.full((12, 12, 3), fill_value=127, dtype=np.uint8)
    np.random.seed(0)
    processed_frames = _prepare_rgb_history_frames(
        [frame.copy(), frame.copy(), frame.copy()],
        ['rgb'],
        image_size=(16, 16),
        photometric_aug=True,
        spatial_aug=True,
    )
    if not (np.array_equal(processed_frames[0], processed_frames[1]) and np.array_equal(processed_frames[1], processed_frames[2])):
        raise AssertionError('Shared augmentation must keep identical frames identical.')


def test_normalizer_lengths():
    for channels in (1, 3, 6, 9):
        normalize = _build_image_normalizer(channels)
        assert_equal(len(normalize.mean), channels, f'Normalizer mean length for {channels} channels')
        assert_equal(len(normalize.std), channels, f'Normalizer std length for {channels} channels')


def test_backbone_channel_adaptation():
    old_weight = torch.arange(64 * 3 * 7 * 7, dtype=torch.float32).reshape(64, 3, 7, 7)

    resnet6 = torchvision.models.resnet18(weights=None)
    with torch.no_grad():
        resnet6.conv1.weight.copy_(old_weight)
    adapt_resnet_input_channels(resnet6, 6)
    assert_equal(resnet6.conv1.in_channels, 6, '6-channel conv1 width')
    expected6 = old_weight.repeat(1, 2, 1, 1) / 2.0
    if not torch.allclose(resnet6.conv1.weight, expected6):
        raise AssertionError('6-channel conv1 initialization must repeat pretrained filters / 2.')

    resnet9 = torchvision.models.resnet18(weights=None)
    with torch.no_grad():
        resnet9.conv1.weight.copy_(old_weight)
    adapt_resnet_input_channels(resnet9, 9)
    assert_equal(resnet9.conv1.in_channels, 9, '9-channel conv1 width')
    expected9 = old_weight.repeat(1, 3, 1, 1) / 3.0
    if not torch.allclose(resnet9.conv1.weight, expected9):
        raise AssertionError('9-channel conv1 initialization must repeat pretrained filters / 3.')


def make_policy_config(image_channels):
    return {
        'lr': 1e-4,
        'num_queries': 4,
        'kl_weight': 1,
        'hidden_dim': 64,
        'dim_feedforward': 128,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 1,
        'dec_layers': 1,
        'nheads': 4,
        'camera_names': ['rgb'],
        'state_dim': 2,
        'action_dim': 2,
        'use_bce_last_action_dim': False,
        'device': 'cpu',
        'image_size': 64,
        'event_channel_selection': None,
        'event_channel_indices': None,
        'rgb_history_frames': image_channels // 3,
        'image_channels': image_channels,
    }


def test_act_forward_smoke():
    backbone_mod.is_main_process = lambda: False
    for image_channels in (6, 9):
        policy = ACTPolicy(make_policy_config(image_channels))
        batch_size = 2
        qpos = torch.zeros((batch_size, 2), dtype=torch.float32)
        image = torch.zeros((batch_size, 1, image_channels, 64, 64), dtype=torch.float32)
        actions = torch.zeros((batch_size, 4, 2), dtype=torch.float32)
        is_pad = torch.zeros((batch_size, 4), dtype=torch.bool)
        loss_dict = policy(qpos, image, actions, is_pad)
        if 'loss' not in loss_dict:
            raise AssertionError('ACT forward training smoke test must produce a loss dict.')
        predicted_actions = policy(qpos, image)
        assert_tensor_shape(predicted_actions, (batch_size, 4, 2), f'ACT inference output shape for image_channels={image_channels}')


def test_eval_temporal_image_path():
    class DummyTs:
        def __init__(self, value):
            self.observation = {
                'images': {
                    'rgb': np.full((8, 8, 3), fill_value=value, dtype=np.uint8)
                }
            }

    buffer = deque(maxlen=3)
    warning_state = {'printed': False}
    get_image(DummyTs(10), ['rgb'], torch.device('cpu'), image_size=8, rgb_history_frames=3, rgb_history_buffer=buffer, padding_notice_state=warning_state)
    get_image(DummyTs(20), ['rgb'], torch.device('cpu'), image_size=8, rgb_history_frames=3, rgb_history_buffer=buffer, padding_notice_state=warning_state)
    image = get_image(DummyTs(30), ['rgb'], torch.device('cpu'), image_size=8, rgb_history_frames=3, rgb_history_buffer=buffer, padding_notice_state=warning_state)
    assert_tensor_shape(image, (1, 1, 9, 8, 8), 'Eval temporal image tensor shape')
    grouped_means = image[0, 0].reshape(3, 3, 8, 8).mean(dim=(1, 2, 3)).numpy() * 255.0
    assert_allclose(grouped_means, np.array([10.0, 20.0, 30.0], dtype=np.float32), 'Eval temporal RGB ordering', atol=0.5)


def test_invalid_combinations(tmpdir):
    episode_path = os.path.join(tmpdir, 'episode_1.hdf5')
    make_episode(episode_path)
    stats = make_norm_stats()
    try:
        EpisodicJointDataset(
            [episode_path],
            ['event'],
            chunk_size=2,
            norm_stats=stats,
            image_size=16,
            rgb_history_frames=2,
        )
    except ValueError as exc:
        if 'camera_names' not in str(exc):
            raise AssertionError(f'Unexpected invalid-combination error: {exc}')
    else:
        raise AssertionError('Temporal RGB should reject event-only mode.')


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dataset_shapes_and_order(tmpdir)
        test_invalid_combinations(tmpdir)
    test_shared_augmentation()
    test_normalizer_lengths()
    test_backbone_channel_adaptation()
    test_act_forward_smoke()
    test_eval_temporal_image_path()
    print('RGB history smoke tests passed.')


if __name__ == '__main__':
    main()