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
from utils import (
    EpisodicJointDataset,
    EpisodicInterceptDataset,
    _prepare_rgb_history_frames,
    compute_history_indices,
    get_intercept_norm_stats,
    load_intercept_data,
)


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


def make_intercept_episode(
    path,
    qpos,
    action_abs_s,
    rgb_values,
    action_positive_direction='robot_base_positive_x',
    include_metadata=True,
):
    qpos = np.asarray(qpos, dtype=np.float32)
    action_abs_s = np.asarray(action_abs_s, dtype=np.float32).reshape(-1, 1)
    rgb_values = np.asarray(rgb_values, dtype=np.uint8)

    if qpos.ndim != 2 or qpos.shape[1] != 7:
        raise ValueError(f'qpos must be (T,7), got {qpos.shape}')
    if action_abs_s.ndim != 2 or action_abs_s.shape[1] != 1:
        raise ValueError(f'action must be (T,1), got {action_abs_s.shape}')
    if rgb_values.ndim != 1 or rgb_values.shape[0] != qpos.shape[0]:
        raise ValueError('rgb_values must be length T')

    T = qpos.shape[0]
    rgb = np.stack([
        np.full((12, 12, 3), fill_value=int(value), dtype=np.uint8)
        for value in rgb_values
    ], axis=0)

    with h5py.File(path, 'w') as root:
        root.attrs['sim'] = False
        if include_metadata:
            root.attrs['action_type'] = 'measured_tcp_s_absolute'
            root.attrs['action_representation'] = 'absolute'
            root.attrs['action_positive_direction'] = action_positive_direction
        root.create_dataset('/observations/qpos', data=qpos)
        root.create_dataset('/action', data=action_abs_s)
        root.create_dataset('/observations/images/rgb', data=rgb)


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


def test_intercept_history_indices():
    assert_equal(compute_history_indices(0, (-6, -3, 0)), [0, 0, 0], 't=0 history indices')
    assert_equal(compute_history_indices(3, (-6, -3, 0)), [0, 0, 3], 't=3 history indices')
    assert_equal(compute_history_indices(8, (-6, -3, 0)), [2, 5, 8], 't>=6 history indices')


def test_intercept_qpos_rgb_alignment_and_order(tmpdir):
    T = 10
    qpos = np.stack([
        np.array([step + d * 100 for d in range(7)], dtype=np.float32)
        for step in range(T)
    ], axis=0)
    action_abs = np.linspace(0.0, -0.09, T, dtype=np.float32)
    rgb_values = np.arange(T, dtype=np.uint8) * 10

    episode_path = os.path.join(tmpdir, 'episode_intercept_align.hdf5')
    make_intercept_episode(episode_path, qpos, action_abs, rgb_values)

    stats = {
        'action_mean': np.zeros(1, dtype=np.float32),
        'action_std': np.ones(1, dtype=np.float32),
        'qpos_mean': np.zeros(21, dtype=np.float32),
        'qpos_std': np.ones(21, dtype=np.float32),
    }
    dataset = EpisodicInterceptDataset(
        [episode_path],
        ['rgb'],
        chunk_size=4,
        norm_stats=stats,
        history_offsets=(-6, -3, 0),
        photometric_aug=False,
        spatial_aug=False,
        image_size=16,
    )

    original_randint = np.random.randint
    try:
        np.random.randint = lambda low, high=None, size=None, dtype=None: 8
        image_data, qpos_data, action_data, is_pad = dataset[0]
    finally:
        np.random.randint = original_randint

    assert_tensor_shape(image_data, (1, 9, 16, 16), 'Interception RGB history shape')
    assert_tensor_shape(qpos_data, (21,), 'Interception qpos flattened history shape')
    assert_tensor_shape(action_data, (4, 1), 'Interception action chunk shape')
    assert_tensor_shape(is_pad, (4,), 'Interception is_pad shape')

    expected_indices = [2, 5, 8]
    expected_qpos_flat = qpos[expected_indices].reshape(-1)
    assert_allclose(qpos_data.numpy(), expected_qpos_flat, 'Qpos history uses same indices and oldest-to-newest flattening')

    grouped_means = image_data[0].reshape(3, 3, 16, 16).mean(dim=(1, 2, 3)).numpy() * 255.0
    expected_rgb = rgb_values[expected_indices].astype(np.float32)
    assert_allclose(grouped_means, expected_rgb, 'RGB history uses same indices and oldest-to-newest stacking', atol=0.5)


def test_intercept_first_token_and_delta_chunk(tmpdir):
    # Known absolute sequence: [0.0, -0.01, -0.03, -0.03, -0.07]
    # At anchor t=1, expected deltas are:
    # k=0: s(2)-s(1) = -0.02
    # k=1: s(3)-s(1) = -0.02
    # k=2: s(4)-s(1) = -0.06
    qpos = np.stack([
        np.array([step + d for d in range(7)], dtype=np.float32)
        for step in range(5)
    ], axis=0)
    action_abs = np.array([0.0, -0.01, -0.03, -0.03, -0.07], dtype=np.float32)
    rgb_values = np.array([10, 20, 30, 40, 50], dtype=np.uint8)
    episode_path = os.path.join(tmpdir, 'episode_intercept_delta.hdf5')
    make_intercept_episode(episode_path, qpos, action_abs, rgb_values)

    stats = {
        'action_mean': np.zeros(1, dtype=np.float32),
        'action_std': np.ones(1, dtype=np.float32),
        'qpos_mean': np.zeros(21, dtype=np.float32),
        'qpos_std': np.ones(21, dtype=np.float32),
    }
    dataset = EpisodicInterceptDataset(
        [episode_path],
        ['rgb'],
        chunk_size=4,
        norm_stats=stats,
        history_offsets=(-6, -3, 0),
        photometric_aug=False,
        spatial_aug=False,
        image_size=16,
    )

    original_randint = np.random.randint
    try:
        np.random.randint = lambda low, high=None, size=None, dtype=None: 1
        _, _, action_data, is_pad = dataset[0]
    finally:
        np.random.randint = original_randint

    expected = np.array([[-0.02], [-0.02], [-0.06], [0.0]], dtype=np.float32)
    assert_allclose(action_data.numpy(), expected, 'Delta chunk must use s(t+k+1)-s(t) and zero-pad tail')
    assert_allclose(is_pad.numpy().astype(np.float32), np.array([0, 0, 0, 1], dtype=np.float32), 'Tail padding mask')
    if abs(float(action_data[0, 0].item())) < 1e-8:
        raise AssertionError('First action token must not be zero by construction.')


def test_intercept_stats_exclude_padding_and_not_absolute(tmpdir):
    qpos_a = np.stack([
        np.array([step + d for d in range(7)], dtype=np.float32)
        for step in range(4)
    ], axis=0)
    qpos_b = np.stack([
        np.array([10 + step + d for d in range(7)], dtype=np.float32)
        for step in range(5)
    ], axis=0)

    # Include both stationary and moving segments.
    action_a = np.array([0.0, 0.0, -0.01, -0.03], dtype=np.float32)
    action_b = np.array([-0.10, -0.10, -0.10, -0.12, -0.12], dtype=np.float32)

    ep_a = os.path.join(tmpdir, 'episode_0.hdf5')
    ep_b = os.path.join(tmpdir, 'episode_1.hdf5')
    make_intercept_episode(ep_a, qpos_a, action_a, np.array([5, 6, 7, 8], dtype=np.uint8))
    make_intercept_episode(ep_b, qpos_b, action_b, np.array([9, 10, 11, 12, 13], dtype=np.uint8))

    stats = get_intercept_norm_stats([ep_a, ep_b], chunk_size=3, history_offsets=(-6, -3, 0))
    assert_tensor_shape(torch.from_numpy(stats['action_mean']), (1,), 'Interception action_mean shape')
    assert_tensor_shape(torch.from_numpy(stats['action_std']), (1,), 'Interception action_std shape')
    assert_tensor_shape(torch.from_numpy(stats['qpos_mean']), (21,), 'Interception qpos_mean shape')
    assert_tensor_shape(torch.from_numpy(stats['qpos_std']), (21,), 'Interception qpos_std shape')

    abs_mean = np.mean(np.concatenate([action_a, action_b]))
    if np.isclose(float(stats['action_mean'][0]), float(abs_mean)):
        raise AssertionError('Interception action stats must be computed from derived deltas, not absolute s values.')


def test_intercept_no_action_is_commanded_requirement_and_metadata_rejection(tmpdir):
    dataset_dir = os.path.join(tmpdir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    qpos = np.stack([
        np.array([step + d for d in range(7)], dtype=np.float32)
        for step in range(6)
    ], axis=0)
    action_abs = np.array([0.0, -0.01, -0.02, -0.02, -0.03, -0.04], dtype=np.float32)
    rgb_values = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)

    make_intercept_episode(
        os.path.join(dataset_dir, 'episode_0.hdf5'),
        qpos,
        action_abs,
        rgb_values,
        action_positive_direction='robot_base_positive_x',
        include_metadata=True,
    )
    make_intercept_episode(
        os.path.join(dataset_dir, 'episode_1.hdf5'),
        qpos,
        action_abs,
        rgb_values,
        action_positive_direction='robot_base_positive_x',
        include_metadata=True,
    )

    # Should load without /action_is_commanded.
    train_loader, val_loader, stats, _ = load_intercept_data(
        dataset_dirs=dataset_dir,
        camera_names=['rgb'],
        chunk_size=3,
        batch_size_train=1,
        batch_size_val=1,
        raw_qpos_dim=7,
        state_dim=21,
        action_dim=1,
        rgb_history_frames=3,
        history_offsets=(-6, -3, 0),
        image_size=16,
    )
    _ = next(iter(train_loader))
    _ = next(iter(val_loader))
    if stats['action_dim'] != 1:
        raise AssertionError('Interception action_dim in stats must be 1.')

    # Reject table-frame-positive metadata.
    bad_dir = os.path.join(tmpdir, 'dataset_bad')
    os.makedirs(bad_dir, exist_ok=True)
    make_intercept_episode(
        os.path.join(bad_dir, 'episode_0.hdf5'),
        qpos,
        action_abs,
        rgb_values,
        action_positive_direction='table_frame_positive_s',
        include_metadata=True,
    )
    make_intercept_episode(
        os.path.join(bad_dir, 'episode_1.hdf5'),
        qpos,
        action_abs,
        rgb_values,
        action_positive_direction='table_frame_positive_s',
        include_metadata=True,
    )
    try:
        load_intercept_data(
            dataset_dirs=bad_dir,
            camera_names=['rgb'],
            chunk_size=3,
            batch_size_train=1,
            batch_size_val=1,
            raw_qpos_dim=7,
            state_dim=21,
            action_dim=1,
            rgb_history_frames=3,
            history_offsets=(-6, -3, 0),
            image_size=16,
        )
    except ValueError as exc:
        if 'table_frame_positive_s' not in str(exc):
            raise AssertionError(f'Expected table-frame metadata rejection, got: {exc}')
    else:
        raise AssertionError('Expected table-frame-positive metadata to be rejected.')

    # Reject missing converter metadata.
    missing_meta_dir = os.path.join(tmpdir, 'dataset_missing_meta')
    os.makedirs(missing_meta_dir, exist_ok=True)
    make_intercept_episode(
        os.path.join(missing_meta_dir, 'episode_0.hdf5'),
        qpos,
        action_abs,
        rgb_values,
        include_metadata=False,
    )
    make_intercept_episode(
        os.path.join(missing_meta_dir, 'episode_1.hdf5'),
        qpos,
        action_abs,
        rgb_values,
        include_metadata=False,
    )
    try:
        load_intercept_data(
            dataset_dirs=missing_meta_dir,
            camera_names=['rgb'],
            chunk_size=3,
            batch_size_train=1,
            batch_size_val=1,
            raw_qpos_dim=7,
            state_dim=21,
            action_dim=1,
            rgb_history_frames=3,
            history_offsets=(-6, -3, 0),
            image_size=16,
        )
    except ValueError as exc:
        if 'Missing interception metadata' not in str(exc):
            raise AssertionError(f'Expected missing-metadata rejection, got: {exc}')
    else:
        raise AssertionError('Expected missing interception metadata to be rejected.')


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dataset_shapes_and_order(tmpdir)
        test_invalid_combinations(tmpdir)
        test_intercept_qpos_rgb_alignment_and_order(tmpdir)
        test_intercept_first_token_and_delta_chunk(tmpdir)
        test_intercept_stats_exclude_padding_and_not_absolute(tmpdir)
        test_intercept_no_action_is_commanded_requirement_and_metadata_rejection(tmpdir)
    test_intercept_history_indices()
    test_shared_augmentation()
    test_normalizer_lengths()
    test_backbone_channel_adaptation()
    test_act_forward_smoke()
    test_eval_temporal_image_path()
    print('RGB history smoke tests passed.')


if __name__ == '__main__':
    main()