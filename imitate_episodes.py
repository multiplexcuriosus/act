import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
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
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    camera_names = args['camera_names']
    data_mode = args['data_mode']
    img_aug = args['img_aug']
    episode_len = args['episode_len']
    
    if use_waypoint:
        print('Using waypoint')
    if constant_waypoint is not None:
        print(f'Constant waypoint: {constant_waypoint}')

    # task name may still be used for eval env selection, but does not control dataset structure
    is_sim = task_name[:4] == 'sim_'
    if data_mode == 'joint':
        if args['state_dim'] is None:
            raise ValueError("--state_dim is required when --data_mode joint")
        state_dim = args['state_dim']
    elif data_mode == 'pose':
        # Keep existing ACTTask pose default while allowing override.
        state_dim = args['state_dim'] if args['state_dim'] is not None else 10
    else:
        raise ValueError(f"Unsupported data_mode: {data_mode}")

    # fixed parameters
    lr_backbone = 1e-5
    backbone = 'resnet18'
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
                         'state_dim': state_dim
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'state_dim': state_dim}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }
    '''
    wandb.login(key = '7afe8f0cb860fa959ee2daf0f8ba40575f703063')
    wandb.init(
        project=task_name,
        config={
            "dataset": dataset_dir,
            "camera names": camera_names,
            "model dof": state_dim,
            "num episodes": num_episodes,
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
            dataset_dir,
            num_episodes,
            camera_names,
            args['chunk_size'],
            batch_size_train,
            batch_size_val,
            img_aug=img_aug,
        )
    elif args['data_mode'] == 'joint':
        train_dataloader, val_dataloader, stats, _ = load_joint_data(
            dataset_dir,
            num_episodes,
            camera_names,
            args['chunk_size'],
            batch_size_train,
            batch_size_val,
            state_dim,
            img_aug=img_aug,
        )
    else:
        raise ValueError(f"Unsupported data_mode: {args['data_mode']}")

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    train_best_ckpt_info, best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)

    # save best checkpoint
    if train_best_ckpt_info is not None:
        train_best_epoch, min_train_loss, train_best_state_dict = train_best_ckpt_info
        train_best_ckpt_path = os.path.join(ckpt_dir, f'policy_train_best.ckpt')
        torch.save(train_best_state_dict, train_best_ckpt_path)

    if best_ckpt_info is not None:
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        best_ckpt_path = os.path.join(ckpt_dir, f'policy_val_best.ckpt')
        torch.save(best_state_dict, best_ckpt_path)


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


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(config['seed'])
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
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
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
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
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

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
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
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
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    latest_idx = 0

    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_train_loss = [np.inf, np.inf]
    min_val_loss = [np.inf, np.inf]
    train_best_ckpt_info = None
    best_ckpt_info = None
    for epoch in tqdm(range(latest_idx, num_epochs)):
        wandb_log = {}

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        e = epoch - latest_idx
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*e:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        train_summary_string = '    '
        for k, v in epoch_summary.items():
            train_summary_string += f'{k}: {v.item():.3f} '
            wandb_log[f'Train {k}'] = v.item()

        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            epoch_val_loss = epoch_summary['loss']
        val_summary_string = '    '
        for k, v in epoch_summary.items():
            val_summary_string += f'{k}: {v.item():.3f} '
            wandb_log[f'Val {k}'] = v.item()
        
        # wandb.log(wandb_log)

        if epoch_train_loss < min_train_loss[0]:
            min_train_loss = [epoch_train_loss, epoch_val_loss]
            train_best_ckpt_info = (epoch, min_train_loss, deepcopy(policy.state_dict()))
        if epoch_val_loss < min_val_loss[1]:
            min_val_loss = [epoch_train_loss, epoch_val_loss]
            best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        
        if epoch % 100 == 0:
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

        if epoch % 1000 == 0:
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

    print(f'\Training finished:')
    print(f'  Val loss:   {epoch_val_loss:.5f}')
    print(val_summary_string)
    print(f'  Train loss: {epoch_train_loss:.5f}')
    print(train_summary_string)
    print(f'  Best train loss at epoch {train_best_epoch}')
    print(f'    train loss:  {min_train_loss[0]:.5f} val loss: {min_train_loss[1]:.5f}')
    print(f'  Best val loss at epoch {best_epoch}')
    print(f'    train loss: {min_val_loss[0]:.5f} val loss: {min_val_loss[1]:.5f}')
    
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
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--episode_len', action='store', type=int, help='episode_len for eval rollouts', required=False, default=400)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--camera_names', nargs='+', required=True, help='camera names to load from dataset')
    parser.add_argument('--data_mode', choices=['joint', 'pose'], required=True, help='dataset mode')
    parser.add_argument('--state_dim', action='store', type=int, required=False, default=None, help='state dimension (required for joint mode; optional override for pose mode)')
    parser.add_argument('--img_aug', action='store_true', help='enable image augmentation (disabled by default)')

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    # for waypoints
    parser.add_argument('--use_waypoint', action='store_true')
    parser.add_argument('--constant_waypoint', action='store', type=int, help='constant_waypoint', required=False)

    main(vars(parser.parse_args()))
