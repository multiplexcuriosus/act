#!/usr/bin/env python3
import os
import time
import argparse
import pickle
from typing import Optional, Tuple, List

import numpy as np
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64

import rclpy
from rclpy.node import Node

from policy import ACTPolicy


def ros_img_to_torch_img(msg: Image, bridge: CvBridge) -> torch.Tensor:
    img_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    img_float = torch.from_numpy(img_rgb / 255.0).float()
    return img_float


def build_qpos_from_joint_state(
    joint_names: List[str],
    joint_pos: np.ndarray,
    state_dim: int,
) -> np.ndarray:
    if state_dim != 8:
        raise ValueError(f"This rollout expects state_dim=8, got {state_dim}")

    arm_joint_names = [
        "right_fr3_joint1",
        "right_fr3_joint2",
        "right_fr3_joint3",
        "right_fr3_joint4",
        "right_fr3_joint5",
        "right_fr3_joint6",
        "right_fr3_joint7",
    ]
    finger_joint_names = ["right_fr3_finger_joint1", "right_fr3_finger_joint2"]

    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    required = arm_joint_names + finger_joint_names
    missing = [name for name in required if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing required joints in JointState: {missing}")

    qpos = np.empty(8, dtype=np.float32)
    for i, name in enumerate(arm_joint_names):
        qpos[i] = np.float32(joint_pos[name_to_idx[name]])

    gripper_width = np.float32(
        joint_pos[name_to_idx["right_fr3_finger_joint1"]] +
        joint_pos[name_to_idx["right_fr3_finger_joint2"]]
    )
    qpos[7] = gripper_width
    return qpos


class FrankaActRolloutNode(Node):
    def __init__(self, args):
        super().__init__("franka_act_rollout")

        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.image_topic = args.image_topic
        self.joint_topic = args.joint_topic
        self.twist_topic = args.twist_topic
        self.gripper_topic = args.gripper_topic
        self.camera_name = args.camera_name

        self.fps = args.fps
        self.max_timesteps = args.max_timesteps
        self.temporal_agg = args.temporal_agg
        self.state_dim = args.state_dim
        self.action_dim_cfg = args.action_dim

        self.latest_image_msg: Optional[Image] = None
        self.latest_joint_msg: Optional[JointState] = None

        self.t = 0
        self.running = args.start_immediately

        policy_config = {
            "lr": args.lr,
            "num_queries": args.chunk_size,
            "kl_weight": args.kl_weight,
            "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": args.enc_layers,
            "dec_layers": args.dec_layers,
            "nheads": args.nheads,
            "camera_names": [self.camera_name],
            "state_dim": self.state_dim,
            "action_dim": self.action_dim_cfg,
        }

        self.num_queries = policy_config["num_queries"]
        self.query_frequency = 1 if self.temporal_agg else self.num_queries
        self.all_actions = None

        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
        stats_path = os.path.join(args.ckpt_dir, args.stats_name)

        self.policy = ACTPolicy(policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.get_logger().info(f"Checkpoint load status: {loading_status}")
        self.policy.to(self.device)
        self.policy.eval()
        self.get_logger().info(f"Loaded checkpoint: {ckpt_path}")

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        for k in ["qpos_mean", "qpos_std", "action_mean", "action_std"]:
            if k not in stats:
                raise KeyError(f"Missing key '{k}' in dataset stats: {stats_path}")

        self.qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
        self.qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)

        if len(self.qpos_mean) != self.state_dim:
            raise ValueError(
                f"state_dim ({self.state_dim}) does not match stats qpos dim ({len(self.qpos_mean)})"
            )

        self.action_dim = len(self.action_mean)
        if self.action_dim_cfg is not None and self.action_dim_cfg != self.action_dim:
            raise ValueError(
                f"action_dim arg ({self.action_dim_cfg}) does not match stats action dim ({self.action_dim})"
            )

        if self.action_dim != 7:
            self.get_logger().warn(
                f"Expected action_dim=7 for twist+gripper, got {self.action_dim}. "
                "Will still run, but publishing uses action[0:6] + action[6]."
            )

        self.pre_process = lambda s: (s - self.qpos_mean) / np.clip(self.qpos_std, 1e-6, None)
        self.post_process = lambda a: a * self.action_std + self.action_mean

        if self.temporal_agg:
            self.all_time_actions = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.num_queries, self.action_dim],
                device=self.device,
            )

        self.twist_pub = self.create_publisher(TwistStamped, self.twist_topic, 10)
        self.gripper_pub = self.create_publisher(Float64, self.gripper_topic, 10)

        self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.create_subscription(JointState, self.joint_topic, self.joint_cb, 10)

        self.timer = self.create_timer(1.0 / self.fps, self.timer_cb)

        self.get_logger().info(f"image_topic={self.image_topic}")
        self.get_logger().info(f"joint_topic={self.joint_topic}")
        self.get_logger().info(f"twist_topic={self.twist_topic}")
        self.get_logger().info(f"gripper_topic={self.gripper_topic}")
        self.get_logger().info(
            f"state_dim={self.state_dim} action_dim={self.action_dim} temporal_agg={self.temporal_agg} fps={self.fps}"
        )
        if self.running:
            self.get_logger().info("Rollout starts immediately.")
        else:
            self.get_logger().info("Waiting for first valid observation, then rollout will start.")

    def image_cb(self, msg: Image) -> None:
        self.latest_image_msg = msg

    def joint_cb(self, msg: JointState) -> None:
        self.latest_joint_msg = msg

    def ready(self) -> bool:
        return self.latest_image_msg is not None and self.latest_joint_msg is not None

    def build_policy_inputs(self) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        if self.latest_joint_msg is None:
            raise RuntimeError("JointState not received")
        if self.latest_image_msg is None:
            raise RuntimeError("Image not received")

        joint_msg = self.latest_joint_msg
        qpos_numpy = build_qpos_from_joint_state(
            joint_names=list(joint_msg.name),
            joint_pos=np.asarray(joint_msg.position, dtype=np.float32),
            state_dim=self.state_dim,
        )

        qpos_norm = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)

        image = ros_img_to_torch_img(self.latest_image_msg, self.bridge)
        curr_image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(self.device)

        return qpos_numpy, qpos, curr_image

    def infer_action(self, qpos: torch.Tensor, curr_image: torch.Tensor) -> np.ndarray:
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
        return self.post_process(raw_action)

    def publish_action(self, action: np.ndarray) -> None:
        if action.shape[0] < 7:
            raise RuntimeError(f"Predicted action has dim {action.shape[0]}, expected at least 7")

        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"
        twist_msg.twist.linear.x = float(action[0])
        twist_msg.twist.linear.y = float(action[1])
        twist_msg.twist.linear.z = float(action[2])
        twist_msg.twist.angular.x = float(action[3])
        twist_msg.twist.angular.y = float(action[4])
        twist_msg.twist.angular.z = float(action[5])
        self.twist_pub.publish(twist_msg)

        gripper_msg = Float64()
        gripper_msg.data = float(action[6])
        #self.gripper_pub.publish(gripper_msg)

    def run_policy_step(self) -> None:
        with torch.inference_mode():
            t0 = time.time()
            qpos_numpy, qpos, curr_image = self.build_policy_inputs()
            action = self.infer_action(qpos, curr_image)
            action[2] *= 0.1
            self.publish_action(action)

            dt_ms = (time.time() - t0) * 1000.0
            self.get_logger().info(
                f"t={self.t:04d} qpos={qpos_numpy.tolist()} "
                f"action={action.tolist()} gripper_target={float(action[6]):.6f} "
                f"inference_ms={dt_ms:.2f}"
            )

            self.t += 1
            if self.t >= self.max_timesteps:
                self.get_logger().info("Reached max_timesteps, wrapping timestep counter.")
                self.t = 0
                if self.temporal_agg:
                    self.all_time_actions.zero_()

    def timer_cb(self) -> None:
        if not self.ready():
            return

        if not self.running:
            self.running = True
            self.get_logger().info("Received first valid image and joint state. Starting rollout.")

        try:
            self.run_policy_step()
        except Exception as e:
            self.get_logger().error(f"Policy step failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="policy_val_best.ckpt")
    parser.add_argument("--stats_name", type=str, default="dataset_stats.pkl")

    parser.add_argument("--image_topic", type=str, default="/camera/camera/color/image_raw")
    parser.add_argument("--joint_topic", type=str, default="/joint_states")
    parser.add_argument("--twist_topic", type=str, default="/cartesian_cmd/twist")
    parser.add_argument("--gripper_topic", type=str, default="/teleop/gripper_cmd")

    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max_timesteps", type=int, default=100000)

    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--camera_name", type=str, default="rgb")

    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--kl_weight", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--dim_feedforward", type=int, required=True)
    parser.add_argument("--temporal_agg", action="store_true")

    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)

    parser.add_argument("--start_immediately", action="store_true")

    args = parser.parse_args()

    rclpy.init()
    node = FrankaActRolloutNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
