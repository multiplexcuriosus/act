#!/usr/bin/env python3
import os
import time
import argparse
import pickle
from typing import Optional

import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from einops import rearrange

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from policy import ACTPolicy

def ros_img_to_torch_img(msg: Image, bridge: CvBridge) -> torch.Tensor:
    img_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    #img_rgb = cv2.resize(img_rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
    img_float = torch.from_numpy(img_rgb / 255.0).float()
    return img_float    


class ToyEnvPolicyPlayerNode(Node):
    def __init__(self, args):
        super().__init__('toy_env_policy_player_node')

        self.bridge = CvBridge()

        self.image_topic = args.image_topic
        self.state_topic = args.state_topic
        self.action_topic = args.action_topic
        self.episode_cmd_topic = args.episode_cmd_topic

        self.rate_hz = args.fps
        self.temporal_agg = args.temporal_agg
        self.max_timesteps = args.max_timesteps
        self.auto_reset = args.auto_reset
        self.reset_on_start = args.reset_on_start

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        self.latest_image_msg: Optional[Image] = None
        self.latest_state: Optional[np.ndarray] = None

        self.t = 0
        self.running = False
        self.all_actions = None

        policy_config = {
            'lr': args.lr,
            'num_queries': args.chunk_size,
            'kl_weight': args.kl_weight,
            'hidden_dim': args.hidden_dim,
            'dim_feedforward': args.dim_feedforward,
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': args.enc_layers,
            'dec_layers': args.dec_layers,
            'nheads': args.nheads,
            'camera_names': [args.camera_name],
            'state_dim': args.state_dim,
        }

        self.num_queries = policy_config['num_queries']
        self.query_frequency = 1 if self.temporal_agg else self.num_queries

        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
        stats_path = os.path.join(args.ckpt_dir, args.stats_name)

        self.policy = ACTPolicy(policy_config)
        loading_status = self.policy.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
        )
        self.get_logger().info(f'Checkpoint load status: {loading_status}')
        self.policy.to(self.device)
        self.policy.eval()
        self.get_logger().info(f'Loaded checkpoint: {ckpt_path}')

        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        required_keys = ['qpos_mean', 'qpos_std', 'action_mean', 'action_std']
        for k in required_keys:
            self.get_logger().info(f'{k}: {stats[k]}')

            if k not in stats:
                raise KeyError(f'Missing key "{k}" in dataset stats file: {stats_path}')

        self.qpos_mean = np.asarray(stats['qpos_mean'], dtype=np.float32)
        self.qpos_std = np.asarray(stats['qpos_std'], dtype=np.float32)
        self.action_mean = np.asarray(stats['action_mean'], dtype=np.float32)
        self.action_std = np.asarray(stats['action_std'], dtype=np.float32)

        self.pre_process = lambda s: (s - self.qpos_mean) / np.clip(self.qpos_std, 1e-6, None)
        self.post_process = lambda a: a * self.action_std + self.action_mean

        if self.temporal_agg:
            action_dim = len(self.action_mean)
            self.all_time_actions = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.num_queries, action_dim],
                device=self.device
            )

        self.action_pub = self.create_publisher(Vector3, self.action_topic, 10)
        self.episode_cmd_pub = self.create_publisher(String, self.episode_cmd_topic, 10)

        self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.create_subscription(Float32MultiArray, self.state_topic, self.state_cb, 10)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_cb)

        self.running = True
        if self.reset_on_start:
            self.send_reset()

        self.debug_static_first_action_only = args.debug_static_first_action_only
        if self.debug_static_first_action_only:
            self.get_logger().info('Debug mode enabled: always using first action of freshly queried chunk.')

        self.get_logger().info('Toy env policy player started.')

    def image_cb(self, msg: Image) -> None:
        self.latest_image_msg = msg

    def state_cb(self, msg: Float32MultiArray) -> None:
        self.latest_state = np.asarray(msg.data, dtype=np.float32)

    def send_reset(self) -> None:
        msg = String()
        msg.data = 'reset'
        self.episode_cmd_pub.publish(msg)
        self.get_logger().info('Published reset command.')

    def ready(self) -> bool:
        if self.latest_image_msg is None:
            return False
        if self.latest_state is None:
            return False
        return True
    

    def build_policy_inputs(self):
        full_state = np.asarray(self.latest_state, dtype=np.float32)
        blue_x, blue_y = full_state[0:2]
        qpos_numpy = np.array([blue_x, blue_y], dtype=np.float32)
        qpos_norm = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)

        image = ros_img_to_torch_img(self.latest_image_msg, self.bridge)

        curr_image = rearrange(image, 'h w c -> c h w').unsqueeze(0).unsqueeze(0)
        curr_image = curr_image.to(self.device)

        return qpos_numpy, qpos, curr_image

    def run_policy(self):
        with torch.inference_mode():
            t0 = time.time()

            qpos_numpy, qpos, curr_image = self.build_policy_inputs()

            if self.debug_static_first_action_only:
                # For frozen-scene debugging:
                # always re-query on the current observation
                # and always take the first predicted action in the chunk.
                self.all_actions = self.policy(qpos, curr_image)
                raw_action = self.all_actions[:, 0]

            else:
                if self.t % self.query_frequency == 0:
                    self.all_actions = self.policy(qpos, curr_image)

                if self.temporal_agg:
                    self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = self.all_actions
                    actions_for_curr_step = self.all_time_actions[:, self.t]
                    actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]

                    if len(actions_for_curr_step) == 0:
                        raw_action = self.all_actions[:, 0]
                    else:
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).float().to(self.device).unsqueeze(1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = self.all_actions[:, self.t % self.query_frequency]

            raw_action = raw_action.squeeze(0).detach().cpu().numpy()
            action = self.post_process(raw_action)

            msg = Vector3()
            msg.x = float(action[0])
            msg.y = float(action[1])
            msg.z = 0.0
            self.action_pub.publish(msg)

            dt_ms = (time.time() - t0) * 1000.0
            self.get_logger().info(
                f't={self.t:04d} state={qpos_numpy.tolist()} action={[float(action[0]), float(action[1])]} raw_action={[float(raw_action[0]), float(raw_action[1])]}'
                f'inference_ms={dt_ms:.2f}'
            )

            self.t += 1

            if self.t >= self.max_timesteps:
                self.get_logger().info('Reached max_timesteps.')
                self.t = 0
                if self.temporal_agg:
                    self.all_time_actions.zero_()
                if self.auto_reset:
                    self.send_reset()

    def timer_cb(self) -> None:
        if not self.running:
            return

        if not self.ready():
            return

        try:
            self.run_policy()
        except Exception as e:
            self.get_logger().error(f'Policy step failed: {e}')
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default='policy_val_best.ckpt')
    parser.add_argument('--stats_name', type=str, default='dataset_stats.pkl')
    parser.add_argument('--debug_static_first_action_only', action='store_true')

    parser.add_argument('--state_dim', type=int, default=2)
    parser.add_argument('--camera_name', type=str, default='toy')

    parser.add_argument('--image_topic', type=str, default='/toy_il/image')
    parser.add_argument('--state_topic', type=str, default='/toy_il/state')
    parser.add_argument('--action_topic', type=str, default='/toy_il/action')
    parser.add_argument('--episode_cmd_topic', type=str, default='/toy_il/episode_cmd')

    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--max_timesteps', type=int, default=200)
    parser.add_argument('--auto_reset', action='store_true')
    parser.add_argument('--reset_on_start', action='store_true')

    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--kl_weight', type=int, required=True)
    parser.add_argument('--chunk_size', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--dim_feedforward', type=int, required=True)
    parser.add_argument('--temporal_agg', action='store_true')

    parser.add_argument('--enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', type=int, default=7)
    parser.add_argument('--nheads', type=int, default=8)

    args = parser.parse_args()

    rclpy.init()
    node = ToyEnvPolicyPlayerNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()