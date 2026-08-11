import argparse
from collections import deque
import cv2
import datetime
import json
import os
import pickle
import platform
import socket
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
#import wandb
import time

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import infer_intercept_visual_config, load_intercept_data, load_joint_data, load_pose_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from utils import INTERCEPT_HISTORY_OFFSETS_DEFAULT
from policy import ACTPolicy, ACTTaskPolicy, CNNMLPPolicy
from sparse_ball import validate_sparse_checkpoint_contract
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed


def resolve_device(device_override=None):
    if device_override is not None:
        return torch.device(device_override)

    try:
        if torch.cuda.is_available():
            torch.zeros(1, device='cuda')
            return torch.device('cuda')
    except Exception as exc:
        print(f'[WARN] CUDA unavailable, falling back to CPU: {exc}')

    return torch.device('cpu')


def validate_camera_names(camera_names):
    allowed_single = [["rgb"], ["event"], ["sparse_ball"]]
    allowed_dual = ["rgb", "event"]

    if camera_names in allowed_single:
        return camera_names

    if camera_names == allowed_dual:
        return camera_names

    if set(camera_names) == {"rgb", "event"}:
        raise ValueError(
            "RGB+event training only supports '--camera_names rgb event'. "
            "Do not use '--camera_names event rgb' because camera order changes the model input slots."
        )

    raise ValueError(
        f"Unsupported camera_names={camera_names}. "
        "Allowed: ['rgb'], ['event'], ['sparse_ball'], or ['rgb', 'event']."
    )


def validate_visual_history_settings(
    camera_names,
    data_mode,
    visual_history_frames,
    rgb_history_frames_alias,
    event_channel_selection,
    expected_intercept_frames=3,
):
    if visual_history_frames is not None and rgb_history_frames_alias is not None:
        if int(visual_history_frames) != int(rgb_history_frames_alias):
            raise ValueError(
                "Conflicting history settings: --visual_history_frames and --rgb_history_frames "
                f"must match when both are provided, got {visual_history_frames} vs {rgb_history_frames_alias}"
            )

    if visual_history_frames is None:
        if rgb_history_frames_alias is not None:
            visual_history_frames = int(rgb_history_frames_alias)
        else:
            visual_history_frames = int(expected_intercept_frames) if data_mode == 'intercept' else 1

    visual_history_frames = int(visual_history_frames)
    if data_mode == 'intercept' and visual_history_frames != int(expected_intercept_frames):
        raise ValueError(
            "Interception visual history does not match dataset metadata: "
            f"expected {expected_intercept_frames}, got {visual_history_frames}"
        )
    if data_mode == 'intercept':
        if camera_names not in (['rgb'], ['event'], ['sparse_ball']):
            raise ValueError(
                f"Interception mode currently supports only a single camera modality, got {camera_names}"
            )
        if event_channel_selection is not None:
            raise ValueError('Interception mode does not support --event_channel_selection.')
        return visual_history_frames
    if visual_history_frames == 1:
        return visual_history_frames
    if camera_names != ['rgb']:
        raise ValueError(
            f"visual_history_frames={visual_history_frames} currently supports only --camera_names rgb outside interception mode, got {camera_names}"
        )
    if event_channel_selection is not None:
        raise ValueError('visual_history_frames > 1 does not support --event_channel_selection.')
    if data_mode not in ('joint', 'intercept'):
        raise ValueError(
            f"visual_history_frames={visual_history_frames} is only supported for joint/intercept mode, got data_mode={data_mode}"
        )
    return visual_history_frames


def _infer_legacy_image_channels(stats, fallback_camera_names=None, fallback_event_channel_selection=None):
    if 'image_channels' in stats:
        return int(stats['image_channels'])
    if stats.get('event_channel_selection', fallback_event_channel_selection) is not None:
        return 1
    saved_rgb_history_frames = int(stats.get('rgb_history_frames', 1))
    saved_camera_names = stats.get('camera_names', fallback_camera_names)
    if saved_rgb_history_frames > 1 and saved_camera_names == ['rgb']:
        return 3 * saved_rgb_history_frames
    return 3


def _infer_input_modality_from_stats(stats, fallback_camera_names=None):
    if 'input_modality' in stats:
        return str(stats['input_modality'])
    saved_camera_names = stats.get('camera_names', fallback_camera_names)
    if saved_camera_names == ['rgb']:
        return 'rgb'
    return None


def _validate_eval_image_config(config, stats):
    policy_config = config['policy_config']
    if policy_config.get('input_modality') == 'sparse_ball':
        validate_sparse_checkpoint_contract(
            stats, policy_config.get('sparse_source'),
            stats.get('sparse_image_width'), stats.get('sparse_image_height'),
            policy_config.get('max_observation_age_sec', 0.10),
        )
        policy_config['sparse_mean'] = stats['sparse_mean']
        policy_config['sparse_std'] = stats['sparse_std']
        return
    configured_visual_history_frames = int(
        policy_config.get(
            'visual_history_frames',
            config.get('visual_history_frames', policy_config.get('rgb_history_frames', config.get('rgb_history_frames', 1))),
        )
    )
    configured_image_channels = int(
        policy_config.get(
            'image_channels',
            config.get('image_channels', 3 * configured_visual_history_frames),
        )
    )
    configured_input_modality = str(
        policy_config.get('input_modality', config.get('input_modality', 'rgb'))
    )

    saved_visual_history_frames = int(
        stats.get('visual_history_frames', stats.get('rgb_history_frames', 1))
    )
    saved_image_channels = _infer_legacy_image_channels(
        stats,
        fallback_camera_names=config.get('camera_names'),
        fallback_event_channel_selection=config.get('event_channel_selection'),
    )
    saved_input_modality = _infer_input_modality_from_stats(
        stats,
        fallback_camera_names=config.get('camera_names'),
    )

    if saved_input_modality is None:
        raise ValueError(
            'Unable to infer input_modality from dataset_stats.pkl. '
            'Legacy inference is only allowed for RGB checkpoints with camera_names=[\'rgb\'].'
        )

    if (
        configured_visual_history_frames != saved_visual_history_frames
        or configured_image_channels != saved_image_channels
        or configured_input_modality != saved_input_modality
    ):
        raise ValueError(
            'Configured image stack does not match dataset_stats.pkl: '
            f'configured input_modality={configured_input_modality}, '
            f'configured visual_history_frames={configured_visual_history_frames}, '
            f'configured image_channels={configured_image_channels}, '
            f'saved input_modality={saved_input_modality}, '
            f'saved visual_history_frames={saved_visual_history_frames}, '
            f'saved image_channels={saved_image_channels}. '
            'Use matching settings or retrain; RGB and event interception checkpoints are not interchangeable.'
        )

    policy_config['input_modality'] = configured_input_modality
    policy_config['visual_history_frames'] = configured_visual_history_frames
    policy_config['visual_history_offsets'] = list(
        stats.get('visual_history_offsets', INTERCEPT_HISTORY_OFFSETS_DEFAULT)
    )
    policy_config['visual_frame_order'] = 'oldest_to_newest'
    policy_config['channels_per_visual_frame'] = int(
        stats.get('channels_per_visual_frame', 3)
    )
    policy_config['image_channels'] = configured_image_channels
    policy_config['image_normalization'] = str(
        stats.get(
            'image_normalization',
            'signed_event_u8_centered' if configured_input_modality == 'event' else 'imagenet',
        )
    )


def _validate_intercept_checkpoint_metadata(config, stats):
    configured_camera_names = config.get('camera_names')
    configured_input_modality = str(
        config['policy_config'].get('input_modality', config.get('input_modality', 'rgb'))
    )
    if configured_input_modality == 'sparse_ball':
        policy_config = config['policy_config']
        validate_sparse_checkpoint_contract(
            stats, policy_config.get('sparse_source'),
            stats.get('sparse_image_width'), stats.get('sparse_image_height'),
            policy_config.get('max_observation_age_sec', 0.10),
        )
        if configured_camera_names != ['sparse_ball']:
            raise ValueError("Sparse checkpoints require camera_names=['sparse_ball']")
        return

    saved_input_modality = _infer_input_modality_from_stats(
        stats,
        fallback_camera_names=configured_camera_names,
    )
    if saved_input_modality is None:
        raise ValueError(
            'Cannot infer input_modality from checkpoint metadata. '
            'Legacy fallback is only supported when camera_names == [\'rgb\'].'
        )

    is_xyt = stats.get('event_representation') == 'xyt_signed_voxel_v1'
    expected_visual_frames = 1 if is_xyt else 3
    expected_visual_offsets = [0] if is_xyt else list(INTERCEPT_HISTORY_OFFSETS_DEFAULT)
    expected_channels_per_frame = 9 if is_xyt else 3
    required_equal = {
        'data_mode': 'intercept',
        'raw_qpos_dim': 7,
        'state_dim': 21,
        'action_dim': 1,
        'visual_history_frames': expected_visual_frames,
        'visual_history_offsets': expected_visual_offsets,
        'channels_per_visual_frame': expected_channels_per_frame,
        'visual_frame_order': 'oldest_to_newest',
        'qpos_history_offsets': list(INTERCEPT_HISTORY_OFFSETS_DEFAULT),
        'qpos_flatten_order': 'oldest_to_newest',
        'image_channels': 9,
        'action_type': 'measured_tcp_s_delta',
        'action_representation': 'future_delta_relative_to_anchor',
        'action_anchor_offset': 0,
        'action_first_target_offset': 1,
        'action_positive_direction': 'robot_base_positive_x',
        'action_units': 'm',
    }

    legacy_key_mapping = {
        'visual_history_frames': 'rgb_history_frames',
        'visual_history_offsets': 'rgb_history_offsets',
        'visual_frame_order': 'rgb_frame_order',
    }

    for key, expected in required_equal.items():
        value = None
        if key in stats:
            value = stats[key]
        elif key in legacy_key_mapping and legacy_key_mapping[key] in stats:
            value = stats[legacy_key_mapping[key]]

        if value is None:
            raise ValueError(f"Missing interception checkpoint metadata in dataset_stats.pkl: {key}")
        if value != expected:
            raise ValueError(
                f"Interception checkpoint metadata mismatch for {key}: expected {expected!r}, found {value!r}"
            )

    if saved_input_modality not in ('rgb', 'event'):
        raise ValueError(f"Unsupported saved interception modality: {saved_input_modality!r}")
    if configured_input_modality not in ('rgb', 'event'):
        raise ValueError(f"Unsupported configured interception modality: {configured_input_modality!r}")
    if configured_input_modality != saved_input_modality:
        raise ValueError(
            'Configured interception modality does not match checkpoint metadata: '
            f'configured={configured_input_modality}, saved={saved_input_modality}'
        )

    expected_camera_names = ['event'] if configured_input_modality == 'event' else ['rgb']
    if configured_camera_names != expected_camera_names:
        raise ValueError(
            'Configured camera_names do not match interception modality: '
            f'camera_names={configured_camera_names}, modality={configured_input_modality}'
        )

    expected_image_norm = 'signed_event_u8_centered' if configured_input_modality == 'event' else 'imagenet'
    saved_image_norm = stats.get('image_normalization', expected_image_norm)
    accepted_norms = {expected_image_norm}
    if configured_input_modality == 'event':
        accepted_norms.add('shifted_3chef_centered')
    if saved_image_norm not in accepted_norms:
        raise ValueError(
            f"Interception checkpoint image_normalization mismatch: expected {expected_image_norm!r}, found {saved_image_norm!r}"
        )

    if configured_input_modality == 'event':
        if is_xyt:
            required_event_meta = {
                'event_representation': 'xyt_signed_voxel_v1',
                'event_horizon_ms': 200.0,
                'event_temporal_bins': 9,
                'event_spatial_height': 320,
                'event_spatial_width': 320,
                'event_channel_order': 'oldest_to_newest',
                'event_polarity_encoding': 'signed',
                'event_scaling': 'signed_log1p_fixed_clip',
                'event_neutral_u8': 128,
                'event_sampling_policy': 'latest_packet_at_or_before_grid_time',
            }
        else:
            required_event_meta = {
                'event_representation': 'shifted_3chef_signed',
                'event_frame_mode': 'shifted',
                'event_frame_windows_ms': [50.0, 100.0, 200.0],
                'event_channel_order': 'recent_to_oldest',
                'event_scaling': 'signed_log1p_fixed_clip',
                'event_neutral_u8': 128,
                'event_sampling_policy': 'latest_packet_at_or_before_grid_time',
            }
        for key, expected in required_event_meta.items():
            if key not in stats:
                raise ValueError(f"Missing interception event metadata in dataset_stats.pkl: {key}")
            if stats[key] != expected:
                raise ValueError(
                    f"Interception event metadata mismatch for {key}: expected {expected!r}, found {stats[key]!r}"
                )
        if 'event_clip_count' not in stats:
            raise ValueError('Missing interception event metadata in dataset_stats.pkl: event_clip_count')
        if float(stats['event_clip_count']) <= 0.0:
            raise ValueError(
                f"Interception event metadata event_clip_count must be positive, got {stats['event_clip_count']}"
            )

    policy_config = config['policy_config']
    if int(policy_config.get('state_dim')) != 21:
        raise ValueError(f"Interception policy state_dim must be 21, got {policy_config.get('state_dim')}")
    if int(policy_config.get('action_dim')) != 1:
        raise ValueError(f"Interception policy action_dim must be 1, got {policy_config.get('action_dim')}")
    if int(policy_config.get('image_channels')) != 9:
        raise ValueError(f"Interception policy image_channels must be 9, got {policy_config.get('image_channels')}")
    if int(policy_config.get('visual_history_frames', expected_visual_frames)) != expected_visual_frames:
        raise ValueError(
            f"Interception policy visual_history_frames must be {expected_visual_frames}, got {policy_config.get('visual_history_frames')}"
        )
    if bool(policy_config.get('use_bce_last_action_dim')):
        raise ValueError('Interception checkpoints must not enable use_bce_last_action_dim.')

def main(args):
    set_seed(args['seed'])
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    use_waypoint = args['use_waypoint']
    constant_waypoint = args['constant_waypoint']
    dataset_dirs = args['dataset_dirs']
    if dataset_dirs is not None:
        dataset_source = dataset_dirs
    else:
        dataset_source = args['dataset_dir']
        if dataset_source is None:
            raise ValueError("Either --dataset_dir or --dataset_dirs must be provided.")
    camera_names = args['camera_names']
    camera_names = validate_camera_names(camera_names)
    data_mode = args['data_mode']
    intercept_visual_config = None
    if data_mode == 'intercept':
        intercept_visual_config = infer_intercept_visual_config(
            dataset_source, camera_names
        )
    visual_history_frames = validate_visual_history_settings(
        camera_names,
        data_mode,
        args.get('visual_history_frames'),
        args['rgb_history_frames'],
        args['event_channel_selection'],
        expected_intercept_frames=(
            intercept_visual_config['visual_history_frames']
            if intercept_visual_config is not None
            else 3
        ),
    )
    rgb_history_frames = visual_history_frames  # legacy alias kept for compatibility
    event_channel_selection = args['event_channel_selection']
    inferred_input_modality = (
        'sparse_ball' if camera_names == ['sparse_ball']
        else ('event' if camera_names == ['event'] else 'rgb')
    )
    input_modality = args.get('input_modality') or inferred_input_modality
    if input_modality != inferred_input_modality:
        raise ValueError(
            f"--input_modality {input_modality!r} does not match --camera_names {camera_names}"
        )
    sparse_source = args.get('sparse_source')
    sparse_feature_dim = int(args.get('sparse_feature_dim', 4))
    sparse_history_length = int(args.get('sparse_history_length', 3))
    max_observation_age_sec = float(args.get('max_observation_age_sec', 0.10))
    if input_modality == 'sparse_ball':
        if sparse_source not in ('rgb', 'event'):
            raise ValueError('--sparse_source rgb|event is required for sparse_ball')
        if sparse_feature_dim != 4 or sparse_history_length != 3:
            raise ValueError('Sparse ACT requires feature_dim=4 and history_length=3')
    visual_history_offsets = (
        list(intercept_visual_config['visual_history_offsets'])
        if intercept_visual_config is not None
        else (list(INTERCEPT_HISTORY_OFFSETS_DEFAULT) if visual_history_frames == 3 else [0])
    )
    visual_frame_order = 'oldest_to_newest'
    channels_per_visual_frame = (
        int(intercept_visual_config['channels_per_visual_frame'])
        if intercept_visual_config is not None
        else 3
    )
    image_normalization = (
        str(intercept_visual_config['image_normalization'])
        if intercept_visual_config is not None
        else ('signed_event_u8_centered' if input_modality == 'event' else 'imagenet')
    )
    if event_channel_selection is not None:
        if camera_names != ['event']:
            raise NotImplementedError(
                '--event_channel_selection is currently only implemented for --camera_names event'
            )
        event_channel_indices = [event_channel_selection - 1]
        image_channels = 1
    elif input_modality == 'rgb':
        event_channel_indices = None
        image_channels = channels_per_visual_frame * visual_history_frames
    else:
        event_channel_indices = None
        image_channels = channels_per_visual_frame * visual_history_frames
    img_aug = args['img_aug']
    photometric_aug = args['photometric_aug']
    spatial_aug = args['spatial_aug']
    if img_aug:
        print('[WARN] --img_aug is deprecated and will be removed in a future release. Use --photometric_aug and/or --spatial_aug.')
        photometric_aug = True
        spatial_aug = True
    print(f'[INFO] Augmentation flags: photometric_aug={photometric_aug}, spatial_aug={spatial_aug}')
    episode_len = args['episode_len']
    action_dim = args['action_dim']
    
    image_size = int(args['image_size'])
    if image_size <= 0:
        raise ValueError(f"--image_size must be a positive integer, got {image_size}")
    if image_size % 32 != 0:
        print(f"[WARN] image_size={image_size} is not a multiple of 32; this may affect model performance.")

    if use_waypoint:
        print('Using waypoint')
    if constant_waypoint is not None:
        print(f'Constant waypoint: {constant_waypoint}')

    # task name may still be used for eval env selection, but does not control dataset structure
    is_sim = task_name[:4] == 'sim_'
    if data_mode == 'joint':
        if args['state_dim'] is None:
            raise ValueError("--state_dim is required when --data_mode joint")
        if action_dim is None:
            raise ValueError("--action_dim is required when --data_mode joint")
        state_dim = args['state_dim']
    elif data_mode == 'intercept':
        if args['state_dim'] is None:
            state_dim = 21
        else:
            state_dim = int(args['state_dim'])
            if state_dim != 21:
                raise ValueError(f"--state_dim must be 21 when --data_mode intercept, got {state_dim}")

        if action_dim is None:
            action_dim = 1
        else:
            action_dim = int(action_dim)
            if action_dim != 1:
                raise ValueError(f"--action_dim must be 1 when --data_mode intercept, got {action_dim}")

        if camera_names not in (['rgb'], ['event'], ['sparse_ball']):
            raise ValueError(f"--data_mode intercept requires --camera_names rgb or event, got {camera_names}")
        if event_channel_selection is not None:
            raise ValueError('--event_channel_selection is not supported for interception mode.')
        if image_channels != int(intercept_visual_config['image_channels']):
            raise ValueError(
                "Interception image_channels mismatch: "
                f"derived={image_channels}, metadata={intercept_visual_config['image_channels']}"
            )
    elif data_mode == 'pose':
        # Keep existing ACTTask pose default while allowing override.
        state_dim = args['state_dim'] if args['state_dim'] is not None else 10
        # Keep previous behavior for pose mode unless explicitly overridden.
        action_dim = action_dim if action_dim is not None else state_dim
    else:
        raise ValueError(f"Unsupported data_mode: {data_mode}")

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # fixed parameters
    lr_backbone = 1e-5
    backbone = 'resnet18'
    device = resolve_device()
    use_bce_last_action_dim = (
        policy_class.startswith('ACT') and
        args['use_bce_last_action_dim']
        and (
            (data_mode == 'joint' and action_dim == 7)
        )
    )
    if data_mode == 'intercept' and args['use_bce_last_action_dim']:
        print('[INFO] Interception mode ignores BCE-on-last-dim; use_bce_last_action_dim is forced to False.')
    if policy_class.startswith('ACT'):
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         'action_dim': action_dim,
                         'use_bce_last_action_dim': use_bce_last_action_dim,
                         'device': device.type,
                         'image_size': image_size,
                         'event_channel_selection': event_channel_selection,
                         'event_channel_indices': event_channel_indices,
                         'input_modality': input_modality,
                         'visual_history_frames': visual_history_frames,
                         'visual_history_offsets': list(visual_history_offsets),
                         'visual_frame_order': visual_frame_order,
                         'channels_per_visual_frame': channels_per_visual_frame,
                         'image_normalization': image_normalization,
                         'rgb_history_frames': rgb_history_frames,
                         'rgb_history_offsets': list(visual_history_offsets),
                         'image_channels': image_channels,
                         'sparse_source': sparse_source,
                         'sparse_feature_dim': sparse_feature_dim,
                         'sparse_history_length': sparse_history_length,
                         'max_observation_age_sec': max_observation_age_sec,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'state_dim': state_dim, 'action_dim': action_dim,
                         'device': device.type, 'image_size': image_size,
                         'event_channel_selection': event_channel_selection,
                         'event_channel_indices': event_channel_indices,
                         'input_modality': input_modality,
                         'visual_history_frames': visual_history_frames,
                         'visual_history_offsets': list(visual_history_offsets),
                         'visual_frame_order': visual_frame_order,
                         'channels_per_visual_frame': channels_per_visual_frame,
                         'image_normalization': image_normalization,
                         'rgb_history_frames': rgb_history_frames,
                         'rgb_history_offsets': list(visual_history_offsets),
                         'image_channels': image_channels}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'device': device.type,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'run_id': run_id,
        'dataset_source': dataset_source,
        'batch_size_train': batch_size_train,
        'batch_size_val': batch_size_val,
        'img_aug': img_aug,
        'photometric_aug': photometric_aug,
        'spatial_aug': spatial_aug,
        'data_mode': data_mode,
        'use_waypoint': use_waypoint,
        'constant_waypoint': constant_waypoint,
        'image_size': image_size,
        'event_channel_selection': event_channel_selection,
        'event_channel_indices': event_channel_indices,
        'input_modality': input_modality,
        'visual_history_frames': visual_history_frames,
        'visual_history_offsets': list(visual_history_offsets),
        'visual_frame_order': visual_frame_order,
        'channels_per_visual_frame': channels_per_visual_frame,
        'image_normalization': image_normalization,
        'rgb_history_frames': rgb_history_frames,
        'rgb_history_offsets': list(visual_history_offsets),
        'image_channels': image_channels,
        'profile_memory': args['profile_memory'],
        'memory_profile_num_epochs': args['memory_profile_num_epochs'],
        'json_log_interval_epochs': 500,
        'save_extra_checkpoints': args['save_extra_checkpoints'],
        'checkpoint_interval': args['checkpoint_interval'],
        'log_path': os.path.join(ckpt_dir, f'run_metrics_{run_id}.json')
    }
    '''
    wandb.login(key = '7afe8f0cb860fa959ee2daf0f8ba40575f703063')
    wandb.init(
        project=task_name,
        config={
            "dataset": dataset_source,
            "camera names": camera_names,
            "model dof": state_dim,
            "ckpt dir": ckpt_dir,
            "chunk size": args['chunk_size'],
            "batch size": batch_size_train,
            "epochs": num_epochs,
            "lr": args['lr'],
            "seed": args['seed']
        },
        name=time.strftime('%Y%m%d_%H%M%S')
    )
    '''
    
    if is_eval:
        ckpt_names = [f'policy_val_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    if args['data_mode'] == 'pose':
        train_dataloader, val_dataloader, stats, _ = load_pose_data(
            dataset_source,
            camera_names,
            args['chunk_size'],
            batch_size_train,
            batch_size_val,
            photometric_aug=photometric_aug,
            spatial_aug=spatial_aug,
            image_size=image_size,
            event_channel_indices=event_channel_indices,
        )
    elif args['data_mode'] == 'joint':
        train_dataloader, val_dataloader, stats, _ = load_joint_data(
            dataset_source,
            camera_names,
            args['chunk_size'],
            batch_size_train,
            batch_size_val,
            model_dof=None,
            photometric_aug=photometric_aug,
            spatial_aug=spatial_aug,
            qpos_dim=state_dim,
            action_dim=action_dim,
            action_key='/action',
            image_size=image_size,
            event_channel_indices=event_channel_indices,
            rgb_history_frames=rgb_history_frames,
        )
        if use_bce_last_action_dim:
            # Keep binary gripper-state target (last action dim) in raw 0/1 space.
            stats['action_mean'][-1] = 0.0
            stats['action_std'][-1] = 1.0
            train_dataloader.dataset.norm_stats['action_mean'][-1] = 0.0
            train_dataloader.dataset.norm_stats['action_std'][-1] = 1.0
            val_dataloader.dataset.norm_stats['action_mean'][-1] = 0.0
            val_dataloader.dataset.norm_stats['action_std'][-1] = 1.0
            print('[INFO] Using BCE on last action dim; forcing action_mean[-1]=0 and action_std[-1]=1')
    elif args['data_mode'] == 'intercept':
        train_dataloader, val_dataloader, stats, _ = load_intercept_data(
            dataset_source,
            camera_names,
            args['chunk_size'],
            batch_size_train,
            batch_size_val,
            photometric_aug=photometric_aug,
            spatial_aug=spatial_aug,
            raw_qpos_dim=7,
            state_dim=state_dim,
            action_dim=action_dim,
            image_size=image_size,
            rgb_history_frames=rgb_history_frames,
            visual_history_frames=visual_history_frames,
            history_offsets=INTERCEPT_HISTORY_OFFSETS_DEFAULT,
            input_modality=input_modality,
            sparse_source=sparse_source,
            sparse_feature_dim=sparse_feature_dim,
            sparse_history_length=sparse_history_length,
            max_observation_age_sec=max_observation_age_sec,
        )
    else:
        raise ValueError(f"Unsupported data_mode: {args['data_mode']}")

    # Interception shape and config checks before the first optimization step.
    if args['data_mode'] == 'intercept':
        batch_image, batch_qpos, batch_action, batch_is_pad = next(iter(train_dataloader))
        print(
            '[DEBUG] intercept batch shapes: '
            f'image={tuple(batch_image.shape)}, '
            f'qpos={tuple(batch_qpos.shape)}, '
            f'action={tuple(batch_action.shape)}, '
            f'is_pad={tuple(batch_is_pad.shape)}'
        )
        if input_modality == 'sparse_ball':
            assert batch_image.shape[1:] == (3, 4), batch_image.shape
        else:
            assert batch_image.ndim == 5 and batch_image.shape[1] == 1, batch_image.shape
        assert batch_qpos.ndim == 2 and batch_qpos.shape[1] == 21, batch_qpos.shape
        assert batch_action.ndim == 3 and batch_action.shape[2] == 1, batch_action.shape
        assert batch_action.shape[1] == args['chunk_size'], batch_action.shape
        assert batch_is_pad.shape == (batch_action.shape[0], args['chunk_size']), batch_is_pad.shape

        assert tuple(stats['action_mean'].shape) == (1,), stats['action_mean'].shape
        assert tuple(stats['action_std'].shape) == (1,), stats['action_std'].shape
        assert tuple(stats['qpos_mean'].shape) == (21,), stats['qpos_mean'].shape
        assert tuple(stats['qpos_std'].shape) == (21,), stats['qpos_std'].shape

        assert int(policy_config['state_dim']) == 21, policy_config['state_dim']
        assert int(policy_config['action_dim']) == 1, policy_config['action_dim']
        if input_modality == 'sparse_ball':
            policy_config['sparse_mean'] = stats['sparse_mean']
            policy_config['sparse_std'] = stats['sparse_std']
        assert bool(policy_config['use_bce_last_action_dim']) is False, policy_config['use_bce_last_action_dim']

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats['image_size'] = image_size
    stats['event_channel_selection'] = event_channel_selection
    stats['event_channel_indices'] = event_channel_indices
    stats['input_modality'] = input_modality
    stats['visual_history_frames'] = visual_history_frames
    stats['visual_history_offsets'] = list(visual_history_offsets)
    stats['visual_frame_order'] = visual_frame_order
    stats['channels_per_visual_frame'] = channels_per_visual_frame
    stats['image_normalization'] = image_normalization
    stats['rgb_history_frames'] = rgb_history_frames
    stats['rgb_history_offsets'] = list(visual_history_offsets)
    stats['rgb_frame_order'] = visual_frame_order
    stats['image_channels'] = image_channels
    stats['camera_names'] = camera_names
    stats['data_mode'] = data_mode
    if data_mode == 'intercept':
        stats['raw_qpos_dim'] = 7
        stats['state_dim'] = 21
        stats['action_dim'] = 1
        stats['qpos_history_frames'] = 3
        stats['qpos_history_offsets'] = list(INTERCEPT_HISTORY_OFFSETS_DEFAULT)
        stats['qpos_flatten_order'] = 'oldest_to_newest'
        stats['action_type'] = 'measured_tcp_s_delta'
        stats['action_representation'] = 'future_delta_relative_to_anchor'
        stats['action_anchor_offset'] = 0
        stats['action_first_target_offset'] = 1
        stats['action_positive_direction'] = 'robot_base_positive_x'
        stats['action_units'] = 'm'
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    train_best_ckpt_info, best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)

    # save best validation checkpoint always
    if best_ckpt_info is not None:
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        best_ckpt_path = os.path.join(ckpt_dir, f'policy_val_best.ckpt')
        torch.save(best_state_dict, best_ckpt_path)

    # optionally save train-best checkpoint
    if args['save_extra_checkpoints'] and train_best_ckpt_info is not None:
        train_best_epoch, min_train_loss, train_best_state_dict = train_best_ckpt_info
        train_best_ckpt_path = os.path.join(ckpt_dir, f'policy_train_best.ckpt')
        torch.save(train_best_state_dict, train_best_ckpt_path)


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'ACTTask':
        policy = ACTTaskPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'ACTTask':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_profile_epochs(num_epochs, num_profile_epochs):
    if num_profile_epochs <= 0:
        return set()
    if num_epochs <= 0:
        return set()
    if num_profile_epochs == 1:
        return {0}
    return set(int(round(x)) for x in np.linspace(0, num_epochs - 1, num_profile_epochs))


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        if obj.size <= 32:
            return obj.tolist()
        return {'shape': list(obj.shape), 'dtype': str(obj.dtype)}
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return {'shape': list(obj.shape), 'dtype': str(obj.dtype), 'device': str(obj.device)}
    if hasattr(obj, 'as_posix'):
        try:
            return str(obj)
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def tensor_to_float_dict(d):
    result = {}
    for key, value in d.items():
        if torch.is_tensor(value) and value.numel() == 1:
            result[key] = float(value.detach().cpu().item())
        elif isinstance(value, np.generic):
            result[key] = float(value.item())
        elif isinstance(value, (int, float)):
            result[key] = float(value)
        else:
            result[key] = make_json_safe(value)
    return result


def write_json_atomic(path, obj):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(make_json_safe(obj), f, indent=2)
    os.replace(tmp_path, path)


def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def get_gpu_info():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        return {
            'cuda_available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': current_device,
            'device_name': torch.cuda.get_device_name(current_device),
            'cuda_version': torch.version.cuda,
        }
    return {'cuda_available': False}


def count_parameters(model):
    return {
        'total': sum(p.numel() for p in model.parameters()),
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def summarize_memory_samples(samples, phase):
    phase_samples = [sample for sample in samples if sample['phase'] == phase]
    if not phase_samples:
        return None
    return {
        'num_samples': len(phase_samples),
        'peak_allocated_mb_max': max(sample['peak_allocated_mb'] for sample in phase_samples),
        'peak_reserved_mb_max': max(sample['peak_reserved_mb'] for sample in phase_samples),
        'current_allocated_mb_last': phase_samples[-1]['current_allocated_mb'],
        'current_reserved_mb_last': phase_samples[-1]['current_reserved_mb'],
        'step_time_s_mean': float(np.mean([sample['step_time_s'] for sample in phase_samples])),
        'step_time_s_max': float(np.max([sample['step_time_s'] for sample in phase_samples])),
    }


def get_image(ts, camera_names, device, image_size=320, event_channel_indices=None, rgb_history_frames=1, rgb_history_buffer=None, padding_notice_state=None):
    rgb_history_frames = int(rgb_history_frames)
    if rgb_history_frames > 1:
        if camera_names != ['rgb']:
            raise ValueError(
                f"rgb_history_frames={rgb_history_frames} currently supports only camera_names=['rgb'], got {camera_names}"
            )
        if event_channel_indices is not None:
            raise ValueError('rgb_history_frames > 1 does not support event_channel_selection in eval.')
        if rgb_history_buffer is None:
            raise ValueError('rgb_history_buffer is required when rgb_history_frames > 1.')

        curr_image = np.asarray(ts.observation['images']['rgb'])
        if curr_image.ndim == 2:
            curr_image = curr_image[..., None]
        if curr_image.ndim != 3 or curr_image.shape[-1] != 3:
            raise ValueError(f"Temporal RGB eval expects HWC3 rgb images, got shape {curr_image.shape}")

        rgb_history_buffer.append(curr_image)
        temporal_frames = list(rgb_history_buffer)
        if len(temporal_frames) < rgb_history_frames:
            pad_count = rgb_history_frames - len(temporal_frames)
            temporal_frames = [temporal_frames[0]] * pad_count + temporal_frames
            if padding_notice_state is not None and not padding_notice_state.get('printed', False):
                print(
                    f"[WARN] Eval RGB history deque is shorter than rgb_history_frames={rgb_history_frames}; "
                    'left-padding with the earliest available frame for initial rollout steps.'
                )
                padding_notice_state['printed'] = True

        processed_frames = []
        for frame in temporal_frames:
            resized_frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            processed_frames.append(rearrange(resized_frame, 'h w c -> c h w'))

        curr_image = np.concatenate(processed_frames, axis=0)
        curr_image = np.expand_dims(curr_image, axis=0)
        return torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)

    curr_images = []
    expected_channels = None
    for cam_name in camera_names:
        curr_image = np.asarray(ts.observation['images'][cam_name])
        if curr_image.ndim == 2:
            curr_image = curr_image[..., None]
        if cam_name == 'event' and event_channel_indices is not None:
            curr_image = curr_image[..., event_channel_indices]
        if curr_image.shape[-1] not in (1, 3):
            raise ValueError(
                f"Unsupported number of channels for camera '{cam_name}': {curr_image.shape[-1]}"
            )
        if expected_channels is None:
            expected_channels = curr_image.shape[-1]
        elif curr_image.shape[-1] != expected_channels:
            raise ValueError(
                f"All cameras must have matching channels for stacking. Expected {expected_channels}, got {curr_image.shape[-1]} for {cam_name}."
            )
        curr_image = cv2.resize(curr_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        curr_image = rearrange(curr_image, 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(config['seed'])
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    action_dim = config['action_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    device = resolve_device(config.get('device'))
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load stats before policy to catch image-stack mismatches clearly.
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    _validate_eval_image_config(config, stats)
    if stats.get('data_mode') == 'intercept' or config.get('data_mode') == 'intercept':
        _validate_intercept_checkpoint_metadata(config, stats)
        raise RuntimeError(
            'Interception evaluation is disabled in this legacy eval_bc path. '
            'This evaluator cannot reconstruct absolute cont_tracker targets from measured current_tcp_s, '
            'and predicted delta-s tokens must not be interpreted as joint targets.'
        )

    # load policy and checkpoint
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(loading_status)
    policy.to(device)
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    eval_image_size = int(stats.get('image_size', config.get('image_size', 320)))
    event_channel_indices = stats.get('event_channel_indices', config.get('event_channel_indices', None))
    rgb_history_frames = int(
        stats.get(
            'visual_history_frames',
            stats.get('rgb_history_frames', policy_config.get('visual_history_frames', policy_config.get('rgb_history_frames', 1))),
        )
    )
    if event_channel_indices is not None and camera_names != ['event']:
        raise NotImplementedError(
            '--event_channel_selection is currently only implemented for --camera_names event'
        )
    print(f"[INFO] Eval image_size={eval_image_size}")

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    _printed_eval_shape = False
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        rgb_history_buffer = deque(maxlen=rgb_history_frames) if rgb_history_frames > 1 else None
        rgb_padding_notice_state = {'printed': False}
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, action_dim], device=device)

        qpos_history = torch.zeros((1, max_timesteps, state_dim), device=device)
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(
                    ts,
                    camera_names,
                    device,
                    image_size=eval_image_size,
                    event_channel_indices=event_channel_indices,
                    rgb_history_frames=rgb_history_frames,
                    rgb_history_buffer=rgb_history_buffer,
                    padding_notice_state=rgb_padding_notice_state,
                )

                if not _printed_eval_shape:
                    print(f"[DEBUG] eval curr_image.shape={tuple(curr_image.shape)}")
                    if event_channel_indices is not None:
                        assert curr_image.ndim == 5, curr_image.shape
                        assert curr_image.shape[1] == 1, curr_image.shape
                        assert curr_image.shape[2] == 1, curr_image.shape
                    elif rgb_history_frames > 1:
                        assert curr_image.shape == (1, 1, 3 * rgb_history_frames, eval_image_size, eval_image_size), curr_image.shape
                    _printed_eval_shape = True

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).to(device).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    device = next(policy.parameters()).device
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    checkpoint_interval = int(config.get('checkpoint_interval', 1000))
    save_extra_checkpoints = bool(config.get('save_extra_checkpoints', False))
    profile_memory_enabled = bool(config.get('profile_memory', False))

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    latest_idx = 0

    policy.to(resolve_device(policy_config.get('device')))
    optimizer = make_optimizer(policy_class, policy)

    profile_epochs = get_profile_epochs(
        num_epochs,
        int(config.get('memory_profile_num_epochs', 3))
    )

    run_log = {
        'run_id': config.get('run_id'),
        'created_at': datetime.datetime.now().isoformat(),
        'finished_at': None,
        'status': 'running',
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'git_commit': get_git_commit(),
        'gpu': get_gpu_info(),
        'config': make_json_safe(config),
        'architecture': {
            'policy_class': policy_class,
            'hidden_dim': policy_config.get('hidden_dim'),
            'dim_feedforward': policy_config.get('dim_feedforward'),
            'num_queries': policy_config.get('num_queries'),
            'chunk_size': policy_config.get('num_queries'),
            'enc_layers': policy_config.get('enc_layers'),
            'dec_layers': policy_config.get('dec_layers'),
            'nheads': policy_config.get('nheads'),
            'backbone': policy_config.get('backbone'),
            'camera_names': policy_config.get('camera_names'),
            'state_dim': policy_config.get('state_dim'),
            'action_dim': policy_config.get('action_dim'),
            'use_bce_last_action_dim': policy_config.get('use_bce_last_action_dim'),
            'temporal_agg': config.get('temporal_agg'),
            'batch_size_train': config.get('batch_size_train'),
            'batch_size_val': config.get('batch_size_val'),
            'image_size': policy_config.get('image_size'),
            'photometric_aug': config.get('photometric_aug'),
            'spatial_aug': config.get('spatial_aug'),
            'event_channel_selection': policy_config.get('event_channel_selection'),
            'event_channel_indices': policy_config.get('event_channel_indices'),
            'input_modality': policy_config.get('input_modality', 'rgb'),
            'visual_history_frames': policy_config.get('visual_history_frames', policy_config.get('rgb_history_frames', 1)),
            'visual_history_offsets': policy_config.get('visual_history_offsets', policy_config.get('rgb_history_offsets', [0])),
            'visual_frame_order': policy_config.get('visual_frame_order', policy_config.get('rgb_frame_order', 'oldest_to_newest')),
            'channels_per_visual_frame': policy_config.get('channels_per_visual_frame', 3),
            'image_normalization': policy_config.get('image_normalization', 'imagenet'),
            'rgb_history_frames': policy_config.get('rgb_history_frames', 1),
            'image_channels': policy_config.get('image_channels'),
        },
        'model': count_parameters(policy),
        'profile_memory': profile_memory_enabled,
        'profile_epochs': sorted(list(profile_epochs)),
        'epochs': [],
        'memory_profiles': [],
        'best': {
            'train': None,
            'val': None
        }
    }
    write_json_atomic(config['log_path'], run_log)

    train_history = []
    validation_history = []
    min_train_loss = [np.inf, np.inf]
    min_val_loss = [np.inf, np.inf]
    train_best_ckpt_info = None
    best_ckpt_info = None
    train_best_epoch = None
    best_epoch = None
    epoch_val_loss = None
    epoch_train_loss = None
    printed_train_image_shape = False
    for epoch in tqdm(range(latest_idx, num_epochs)):
        wandb_log = {}
        epoch_start_time = time.perf_counter()
        epoch_memory_profile = []
        should_profile_epoch = (
            bool(config.get('profile_memory', False))
            and epoch in profile_epochs
            and torch.cuda.is_available()
        )

        # training
        policy.train()
        optimizer.zero_grad()
        train_epoch_dicts = []
        for batch_idx, data in enumerate(train_dataloader):
            if not printed_train_image_shape:
                image_data = data[0]
                print(f"[DEBUG] train image_data.shape={tuple(image_data.shape)}")
                if policy_config.get('input_modality') == 'sparse_ball':
                    expected_history = int(policy_config.get('sparse_history_length', 3))
                    expected_features = int(policy_config.get('sparse_feature_dim', 4))
                    assert image_data.ndim == 3, image_data.shape
                    assert image_data.shape[1:] == (
                        expected_history, expected_features
                    ), image_data.shape
                elif policy_config.get('event_channel_indices') is not None:
                    assert image_data.ndim == 5, image_data.shape
                    assert image_data.shape[1] == 1, image_data.shape
                    assert image_data.shape[2] == 1, image_data.shape
                else:
                    visual_history_frames = int(policy_config.get('visual_history_frames', policy_config.get('rgb_history_frames', 1)))
                    channels_per_visual_frame = int(policy_config.get('channels_per_visual_frame', 3))
                    if visual_history_frames > 1:
                        expected_channels = channels_per_visual_frame * visual_history_frames
                        assert image_data.shape[1] == 1, image_data.shape
                        assert image_data.shape[2] == expected_channels, image_data.shape
                    elif int(policy_config.get('image_channels', 3)) in (1, 3):
                        assert image_data.shape[1] == len(policy_config.get('camera_names', [])), image_data.shape
                        assert image_data.shape[2] == int(policy_config.get('image_channels', 3)), image_data.shape
                    else:
                        assert image_data.shape[1] == 1, image_data.shape
                        assert image_data.shape[2] == int(policy_config.get('image_channels')), image_data.shape
                printed_train_image_shape = True

            if should_profile_epoch:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                step_start_time = time.perf_counter()

            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            detached_forward_dict = detach_dict(forward_dict)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_epoch_dicts.append(detached_forward_dict)

            if should_profile_epoch:
                torch.cuda.synchronize()
                step_time_s = time.perf_counter() - step_start_time
                mem_sample = {
                    'phase': 'train',
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'wall_time_s': time.time(),
                    'step_time_s': step_time_s,
                    'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                    'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
                    'current_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                    'current_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                    'losses': tensor_to_float_dict(detached_forward_dict)
                }
                epoch_memory_profile.append(mem_sample)

        train_epoch_summary = compute_dict_mean(train_epoch_dicts)
        train_history.append(train_epoch_summary)
        epoch_train_loss = train_epoch_summary['loss']
        train_summary_string = '    '
        for k, v in train_epoch_summary.items():
            train_summary_string += f'{k}: {v.item():.3f} '
            wandb_log[f'Train {k}'] = v.item()

        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                if should_profile_epoch:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    val_start_time = time.perf_counter()

                forward_dict = forward_pass(data, policy)
                detached_forward_dict = detach_dict(forward_dict)
                epoch_dicts.append(detached_forward_dict)

                if should_profile_epoch:
                    torch.cuda.synchronize()
                    val_step_time_s = time.perf_counter() - val_start_time
                    val_mem_sample = {
                        'phase': 'val',
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'wall_time_s': time.time(),
                        'step_time_s': val_step_time_s,
                        'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                        'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
                        'current_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                        'current_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                        'losses': tensor_to_float_dict(detached_forward_dict)
                    }
                    epoch_memory_profile.append(val_mem_sample)

            val_epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(val_epoch_summary)
            epoch_val_loss = val_epoch_summary['loss']
        val_summary_string = '    '
        for k, v in val_epoch_summary.items():
            val_summary_string += f'{k}: {v.item():.3f} '
            wandb_log[f'Val {k}'] = v.item()

        epoch_duration_s = time.perf_counter() - epoch_start_time

        train_memory_summary = summarize_memory_samples(epoch_memory_profile, 'train')
        val_memory_summary = summarize_memory_samples(epoch_memory_profile, 'val')

        if epoch_train_loss < min_train_loss[0]:
            min_train_loss = [epoch_train_loss, epoch_val_loss]
            train_best_epoch = epoch
            train_best_ckpt_info = (epoch, min_train_loss, deepcopy(policy.state_dict()))
            run_log['best']['train'] = {
                'epoch': int(train_best_epoch),
                'train_loss': float(epoch_train_loss.detach().cpu().item() if torch.is_tensor(epoch_train_loss) else epoch_train_loss),
                'val_loss': float(epoch_val_loss.detach().cpu().item() if torch.is_tensor(epoch_val_loss) else epoch_val_loss)
            }
        if epoch_val_loss < min_val_loss[1]:
            min_val_loss = [epoch_train_loss, epoch_val_loss]
            best_epoch = epoch
            best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            run_log['best']['val'] = {
                'epoch': int(best_epoch),
                'train_loss': float(epoch_train_loss.detach().cpu().item() if torch.is_tensor(epoch_train_loss) else epoch_train_loss),
                'val_loss': float(epoch_val_loss.detach().cpu().item() if torch.is_tensor(epoch_val_loss) else epoch_val_loss)
            }

        if should_profile_epoch:
            run_log['epochs'].append({
                'epoch': epoch,
                'wall_time_s': time.time(),
                'epoch_duration_s': epoch_duration_s,
                'profiled_memory': True,
                'train': {
                    'loss': float(epoch_train_loss.detach().cpu().item() if torch.is_tensor(epoch_train_loss) else epoch_train_loss),
                    'losses': tensor_to_float_dict(train_epoch_summary),
                    'memory_summary': train_memory_summary
                },
                'val': {
                    'loss': float(epoch_val_loss.detach().cpu().item() if torch.is_tensor(epoch_val_loss) else epoch_val_loss),
                    'losses': tensor_to_float_dict(val_epoch_summary),
                    'memory_summary': val_memory_summary
                }
            })
            run_log['memory_profiles'].append({
                'epoch': epoch,
                'samples': epoch_memory_profile
            })

        should_write_log = (
            epoch == 0 or
            epoch == num_epochs - 1 or
            should_profile_epoch or
            (profile_memory_enabled and epoch % 10 == 0)
        )
        if should_write_log:
            write_json_atomic(config['log_path'], run_log)
        
        # wandb.log(wandb_log)
        
        if epoch % 100 == 0:
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

        if save_extra_checkpoints and checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            if train_best_ckpt_info is not None:
                train_best_epoch, min_train_loss, train_best_state_dict = train_best_ckpt_info
                train_best_ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{train_best_epoch}_seed_{seed}.ckpt')
                torch.save(train_best_state_dict, train_best_ckpt_path)

            if best_ckpt_info is not None:
                best_epoch, min_val_loss, best_state_dict = best_ckpt_info
                best_ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
                torch.save(best_state_dict, best_ckpt_path)

            print(f'\nTraining result till Epoch {epoch}:')
            print(f'  Val loss:   {epoch_val_loss:.6f}')
            print(val_summary_string)
            print(f'  Train loss: {epoch_train_loss:.6f}')
            print(train_summary_string)
            print(f'  Best train loss at epoch {train_best_epoch}')
            print(f'    Train loss:  {min_train_loss[0]:.6f} Val loss: {min_train_loss[1]:.6f}')
            print(f'  Best val loss at epoch {best_epoch}')
            print(f'    Train loss: {min_val_loss[0]:.6f} Val loss: {min_val_loss[1]:.6f}')
            if should_profile_epoch:
                print(f'  Memory profile epoch: {epoch}')
                print(f'  Train peak allocated: {train_memory_summary["peak_allocated_mb_max"]:.1f} MB')
                print(f'  Train peak reserved:  {train_memory_summary["peak_reserved_mb_max"]:.1f} MB')
                print(f'  Val peak allocated:   {val_memory_summary["peak_allocated_mb_max"]:.1f} MB')
                print(f'  Val peak reserved:    {val_memory_summary["peak_reserved_mb_max"]:.1f} MB')
            print(f'  JSON log: {config["log_path"]}')

    if save_extra_checkpoints:
        ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
        torch.save(policy.state_dict(), ckpt_path)

        if train_best_ckpt_info is not None:
            train_best_epoch, min_train_loss, train_best_state_dict = train_best_ckpt_info
            train_best_ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{train_best_epoch}_seed_{seed}.ckpt')
            torch.save(train_best_state_dict, train_best_ckpt_path)

        if best_ckpt_info is not None:
            best_epoch, min_val_loss, best_state_dict = best_ckpt_info
            best_ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
            torch.save(best_state_dict, best_ckpt_path)

    run_log['finished_at'] = datetime.datetime.now().isoformat()
    run_log['status'] = 'finished'
    write_json_atomic(config['log_path'], run_log)

    print(f'\nTraining finished:')
    print(f'  Val loss:   {epoch_val_loss:.5f}')
    print(val_summary_string)
    print(f'  Train loss: {epoch_train_loss:.5f}')
    print(train_summary_string)
    print(f'  Best train loss at epoch {train_best_epoch}')
    print(f'    train loss:  {min_train_loss[0]:.5f} val loss: {min_train_loss[1]:.5f}')
    print(f'  Best val loss at epoch {best_epoch}')
    print(f'    train loss: {min_val_loss[0]:.5f} val loss: {min_val_loss[1]:.5f}')
    print(f'  JSON log: {config["log_path"]}')
    
    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return train_best_ckpt_info, best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    # print(f'Saved plots to {ckpt_dir}')

'''
python imitate_episodes.py \
--policy_class ACTTask --kl_weight 10 --chunk_size 30 --hidden_dim 512 \
--batch_size 8 --dim_feedforward 3200 --num_epochs 5000 --lr 1e-5 --seed 0 \
--task_name real_tocabi_pick_n_place --ckpt_dir /media/lyh/SSD2TB/act/ckpt/tocabi/real_tocabi_pick_n_place/ee_global
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', required=False)
    parser.add_argument('--dataset_dirs', nargs='+', type=str, required=False, default=None, help='dataset directories')
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--episode_len', action='store', type=int, help='episode_len for eval rollouts', required=False, default=400)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--camera_names', nargs='+', required=True, help='camera names to load from dataset')
    parser.add_argument('--input_modality', choices=['rgb', 'event', 'sparse_ball'])
    parser.add_argument('--sparse_source', choices=['rgb', 'event'])
    parser.add_argument('--sparse_feature_dim', type=int, default=4)
    parser.add_argument('--sparse_history_length', type=int, default=3)
    parser.add_argument('--max_observation_age_sec', type=float, default=0.10)
    parser.add_argument(
        '--event_channel_selection',
        type=int,
        choices=[1, 2, 3],
        default=None,
        help='For event camera only: select one event channel to train on. Example: --event_channel_selection 3'
    )
    parser.add_argument(
        '--visual_history_frames',
        type=int,
        choices=[1, 2, 3],
        default=None,
        help='Visual history frame count. Defaults by mode: intercept=3 using spaced offsets [-6,-3,0], other modes=1.'
    )
    parser.add_argument(
        '--rgb_history_frames',
        type=int,
        choices=[1, 2, 3],
        default=None,
        help='[LEGACY ALIAS] Same as --visual_history_frames. If both are provided, they must match.'
    )
    parser.add_argument('--data_mode', choices=['joint', 'intercept', 'pose'], required=True, help='dataset mode')
    parser.add_argument('--state_dim', action='store', type=int, required=False, default=None, help='state dimension (joint requires explicit value; intercept defaults to 21 and rejects other values; optional override for pose mode)')
    parser.add_argument('--action_dim', action='store', type=int, required=False, default=None, help='action dimension (joint requires explicit value; intercept defaults to 1 and rejects other values; optional override for pose mode)')
    parser.add_argument('--photometric_aug', action='store_true', help='enable photometric augmentation (ColorJitter on non-event images only)')
    parser.add_argument('--spatial_aug', action='store_true', help='enable spatial augmentation (rotate/crop transform)')
    parser.add_argument('--img_aug', action='store_true', help='[DEPRECATED] enable both --photometric_aug and --spatial_aug')
    parser.add_argument('--use_bce_last_action_dim', action='store_true', help='use BCEWithLogits on final action dim')
    parser.add_argument('--no_use_bce_last_action_dim', action='store_false', dest='use_bce_last_action_dim', help='disable BCEWithLogits on final action dim')
    parser.add_argument('--profile_memory', action='store_true', help='enable sparse CUDA memory profiling at first/mid/last epoch')
    parser.add_argument('--memory_profile_num_epochs', type=int, default=3, help='number of epochs to profile across training; default 3 gives first/mid/last')
    parser.add_argument('--save_extra_checkpoints', action='store_true', help='enable extra checkpoint files (policy_epoch_*, policy_last.ckpt, policy_train_best.ckpt). Disabled by default.')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Save intermediate best checkpoints every N epochs. Use 0 to disable intermediate checkpointing.')

    parser.add_argument('--image_size', type=int, default=320, help='Canonical square image size, e.g. 320, 224, or 160')

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    # for waypoints
    parser.add_argument('--use_waypoint', action='store_true')
    parser.add_argument('--constant_waypoint', action='store', type=int, help='constant_waypoint', required=False)

    parser.set_defaults(use_bce_last_action_dim=True)

    main(vars(parser.parse_args()))
