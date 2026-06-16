import argparse
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
import cv2
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
#import wandb
import time

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_joint_data, load_pose_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, ACTTaskPolicy, CNNMLPPolicy
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
    allowed_single = [["rgb"], ["event"]]
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
        "Allowed: ['rgb'], ['event'], or ['rgb', 'event']."
    )

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
    img_aug = args['img_aug']
    episode_len = args['episode_len']
    action_dim = args['action_dim']
    image_size = int(args['image_size'])
    event_input_channels = int(args['event_input_channels'])
    event_channel_index = int(args['event_channel_index'])

    if image_size <= 0:
        raise ValueError(f"--image_size must be positive, got {image_size}")
    if event_input_channels not in (1, 3):
        raise ValueError(f"--event_input_channels must be 1 or 3, got {event_input_channels}")
    if event_input_channels == 1 and camera_names != ["event"]:
        raise ValueError(
            "--event_input_channels 1 is currently supported only for --camera_names event. "
            "Use --event_input_channels 3 for RGB or RGB+event training."
        )
    if image_size % 32 != 0:
        print(
            f"[WARN] --image_size={image_size} is not divisible by 32. "
            "ResNet feature-map sizes are typically cleaner with multiples of 32."
        )
    
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
        policy_class == 'ACT' and
        data_mode == 'joint' and
        action_dim == 7 and
        args['use_bce_last_action_dim']
    )
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
                         'event_input_channels': event_input_channels,
                         'image_size': image_size,
                         'state_dim': state_dim,
                         'action_dim': action_dim,
                         'use_bce_last_action_dim': use_bce_last_action_dim,
                         'device': device.type
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'image_size': image_size, 'state_dim': state_dim, 'action_dim': action_dim,
                         'device': device.type}
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
        'event_input_channels': event_input_channels,
        'event_channel_index': event_channel_index,
        'image_size': image_size,
        'real_robot': not is_sim,
        'run_id': run_id,
        'dataset_source': dataset_source,
        'batch_size_train': batch_size_train,
        'batch_size_val': batch_size_val,
        'img_aug': img_aug,
        'data_mode': data_mode,
        'use_waypoint': use_waypoint,
        'constant_waypoint': constant_waypoint,
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
            img_aug=img_aug,
            image_size=image_size,
            event_input_channels=event_input_channels,
            event_channel_index=event_channel_index,
        )
    elif args['data_mode'] == 'joint':
        train_dataloader, val_dataloader, stats, _ = load_joint_data(
            dataset_source,
            camera_names,
            args['chunk_size'],
            batch_size_train,
            batch_size_val,
            model_dof=None,
            img_aug=img_aug,
            qpos_dim=state_dim,
            action_dim=action_dim,
            action_key='/action',
            image_size=image_size,
            event_input_channels=event_input_channels,
            event_channel_index=event_channel_index,
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
    else:
        raise ValueError(f"Unsupported data_mode: {args['data_mode']}")

    stats['camera_names'] = list(camera_names)
    stats['image_size'] = int(image_size)
    stats['event_input_channels'] = int(event_input_channels)
    stats['event_channel_index'] = int(event_channel_index)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
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


def get_image(ts, camera_names, device, image_size=None):
    target_size = int(image_size) if image_size is not None else 320
    curr_images = []
    for cam_name in camera_names:
        curr_image = np.asarray(ts.observation['images'][cam_name])
        if curr_image.ndim == 2:
            curr_image = curr_image[..., None]
        if curr_image.ndim != 3:
            raise ValueError(f"Unsupported eval image shape for camera '{cam_name}': {curr_image.shape}")
        if curr_image.shape[-1] == 1:
            curr_image = np.repeat(curr_image, 3, axis=-1)
        elif curr_image.shape[-1] != 3:
            raise ValueError(f"Unsupported eval image channels for camera '{cam_name}': {curr_image.shape}")

        curr_image = cv2.resize(curr_image, (target_size, target_size), interpolation=cv2.INTER_AREA)
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

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(loading_status)
    policy.to(device)
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

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
    printed_eval_image_shape = False
    for rollout_id in range(num_rollouts):
        rollout_id += 0
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
                curr_image = get_image(ts, camera_names, device, image_size=config.get('image_size', 320))
                if not printed_eval_image_shape:
                    print(f"[DEBUG] eval curr_image.shape={tuple(curr_image.shape)}")
                    printed_eval_image_shape = True

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
            'batch_size_val': config.get('batch_size_val')
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
    parser.add_argument('--data_mode', choices=['joint', 'pose'], required=True, help='dataset mode')
    parser.add_argument('--state_dim', action='store', type=int, required=False, default=None, help='state dimension (required for joint mode; optional override for pose mode)')
    parser.add_argument('--action_dim', action='store', type=int, required=False, default=None, help='action dimension (required for joint mode; optional override for pose mode)')
    parser.add_argument('--img_aug', action='store_true', help='enable image augmentation (disabled by default)')
    parser.add_argument('--use_bce_last_action_dim', action='store_true', help='use BCEWithLogits on final action dim')
    parser.add_argument('--no_use_bce_last_action_dim', action='store_false', dest='use_bce_last_action_dim', help='disable BCEWithLogits on final action dim')
    parser.add_argument('--profile_memory', action='store_true', help='enable sparse CUDA memory profiling at first/mid/last epoch')
    parser.add_argument('--memory_profile_num_epochs', type=int, default=3, help='number of epochs to profile across training; default 3 gives first/mid/last')
    parser.add_argument('--save_extra_checkpoints', action='store_true', help='enable extra checkpoint files (policy_epoch_*, policy_last.ckpt, policy_train_best.ckpt). Disabled by default.')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Save intermediate best checkpoints every N epochs. Use 0 to disable intermediate checkpointing.')
    parser.add_argument('--image_size', type=int, default=320, help='Canonical square image size, e.g. 320 or 160')
    parser.add_argument('--event_input_channels', type=int, choices=[1, 3], default=3, help='Event image channels: 3 for fake-RGB (default), 1 for true single-channel event-only training')
    parser.add_argument('--event_channel_index', type=int, default=2, help='When loading 3-channel event frames as 1-channel, select this channel index')

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
