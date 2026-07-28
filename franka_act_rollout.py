#!/usr/bin/env python3
import argparse
import os
import pickle
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32MultiArray

import policy as policy_module


ARM_JOINT_NAMES = [
    "right_fr3_joint1",
    "right_fr3_joint2",
    "right_fr3_joint3",
    "right_fr3_joint4",
    "right_fr3_joint5",
    "right_fr3_joint6",
    "right_fr3_joint7",
]
FINGER_JOINT_NAMES = ["right_fr3_finger_joint1", "right_fr3_finger_joint2"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageSequenceGate:
    def __init__(self, required_history: int):
        self.required_history = int(required_history)
        self.latest_sequence = 0
        self.last_processed_sequence = 0

    def note_image_received(self) -> int:
        self.latest_sequence += 1
        return self.latest_sequence

    def claim_latest_ready_sequence(self, buffer_len: int) -> Optional[int]:
        if buffer_len < self.required_history:
            return None
        if self.latest_sequence == self.last_processed_sequence:
            return None
        self.last_processed_sequence = self.latest_sequence
        return self.latest_sequence


def ros_img_to_torch_img(msg: Image, bridge: CvBridge) -> torch.Tensor:
    img_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    img_float = torch.from_numpy(img_rgb / 255.0).float()
    return img_float


def ensure_policy_image_normalizer_support() -> None:
    if getattr(policy_module, "_act_rollout_history_patch", False):
        return

    original_builder = policy_module._build_image_normalizer

    def _build_image_normalizer(image_channels: int):
        if image_channels in (1, 3):
            return original_builder(image_channels)
        if image_channels in (6, 9):
            repeats = image_channels // 3
            return policy_module.transforms.Normalize(
                mean=IMAGENET_MEAN * repeats,
                std=IMAGENET_STD * repeats,
            )
        raise ValueError(f"Unsupported image_channels={image_channels}")

    policy_module._build_image_normalizer = _build_image_normalizer
    policy_module._act_rollout_history_patch = True


def resolve_default_image_topic(
    image_topic: Optional[str],
    rollout_mode: str,
    camera_name: str,
) -> str:
    if image_topic is not None:
        return image_topic
    if camera_name == "event":
        return "/openmv_cam/event_frame_3ch"
    if rollout_mode == "intercept":
        return "/top_cam/camera/color/image_raw"
    if camera_name == "rgb":
        return "/camera/camera/color/image_raw"
    raise ValueError(f"Unsupported camera_name='{camera_name}' for automatic image topic selection")


def coerce_square_image_size(image_size_value: Any) -> int:
    if isinstance(image_size_value, (tuple, list)):
        if len(image_size_value) != 2 or int(image_size_value[0]) != int(image_size_value[1]):
            raise ValueError(f"Expected square image_size metadata, got {image_size_value}")
        return int(image_size_value[0])
    return int(image_size_value)


def rgb_frame_order_is_oldest_to_newest(rgb_frame_order: Any) -> bool:
    if isinstance(rgb_frame_order, str):
        normalized = rgb_frame_order.strip().lower().replace("-", "_")
        return normalized in {
            "oldest_to_newest",
            "oldest_first",
            "chronological",
            "chronological_oldest_to_newest",
        }
    return False


def load_rollout_stats(stats_path: str) -> Dict[str, Any]:
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    for key in ["qpos_mean", "qpos_std", "action_mean", "action_std"]:
        if key not in stats:
            raise KeyError(f"Missing key '{key}' in dataset stats: {stats_path}")
    return stats


def resolve_rollout_configuration(args, stats: Dict[str, Any]) -> Dict[str, Any]:
    qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
    qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats["action_std"], dtype=np.float32)

    stats_state_dim = int(len(qpos_mean))
    stats_action_dim = int(len(action_mean))

    if args.state_dim is not None and int(args.state_dim) != stats_state_dim:
        raise ValueError(
            f"state_dim arg ({args.state_dim}) does not match stats qpos dim ({stats_state_dim})"
        )
    if args.action_dim is not None and int(args.action_dim) != stats_action_dim:
        raise ValueError(
            f"action_dim arg ({args.action_dim}) does not match stats action dim ({stats_action_dim})"
        )

    rgb_history_frames = int(stats.get("rgb_history_frames", 1))
    image_channels = int(stats.get("image_channels", 3 * rgb_history_frames))
    image_size = coerce_square_image_size(stats.get("image_size", 320))
    camera_names = list(stats.get("camera_names", [args.camera_name]))
    rgb_frame_order = stats.get("rgb_frame_order", "oldest_to_newest")

    if len(camera_names) != 1:
        raise ValueError(
            f"This rollout only supports a single camera input, got camera_names={camera_names}"
        )

    if args.rollout_mode == "intercept":
        raise ValueError(
            "This script's legacy intercept mode expects "
            "[absolute_s, execute_logit] and is incompatible with the new "
            "30-step delta-s policy. Use franka_act_intercept_rollout.py."
        )

    return {
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "state_dim": stats_state_dim,
        "action_dim": stats_action_dim,
        "rgb_history_frames": rgb_history_frames,
        "image_channels": image_channels,
        "image_size": image_size,
        "camera_names": camera_names,
        "rgb_frame_order": rgb_frame_order,
    }


def rgb_image_msg_to_numpy(msg: Image, bridge: CvBridge) -> np.ndarray:
    image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
    return image


def preprocess_rgb_history_images(
    rgb_frames: Sequence[np.ndarray],
    image_size: int,
) -> torch.Tensor:
    if len(rgb_frames) not in (1, 2, 3):
        raise ValueError(f"Expected 1-3 RGB frames, got {len(rgb_frames)}")

    resized_frames = []
    for frame in rgb_frames:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}")
        resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        resized_frames.append(resized.astype(np.float32) / 255.0)

    stacked_hwc = np.concatenate(resized_frames, axis=2)
    expected_channels = 3 * len(resized_frames)
    assert stacked_hwc.shape == (image_size, image_size, expected_channels), stacked_hwc.shape

    stacked_chw = np.transpose(stacked_hwc, (2, 0, 1))
    curr_image = torch.from_numpy(stacked_chw).float().unsqueeze(0).unsqueeze(0)
    assert tuple(curr_image.shape) == (1, 1, expected_channels, image_size, image_size), curr_image.shape
    return curr_image


def select_current_intercept_action(
    action_chunk: torch.Tensor,
    num_queries: int,
) -> torch.Tensor:
    if action_chunk.ndim != 3:
        raise ValueError(f"Expected action_chunk.ndim == 3, got {action_chunk.ndim}")
    if action_chunk.shape[0] != 1:
        raise ValueError(f"Expected action_chunk.shape[0] == 1, got {action_chunk.shape[0]}")
    if action_chunk.shape[1] != num_queries:
        raise ValueError(
            f"Expected action_chunk.shape[1] == num_queries ({num_queries}), got {action_chunk.shape[1]}"
        )
    if action_chunk.shape[2] != 2:
        raise ValueError(f"Expected action_chunk.shape[2] == 2, got {action_chunk.shape[2]}")
    return action_chunk[0, 0]


def convert_intercept_action(
    raw_action: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> Tuple[float, float, float]:
    raw_action = np.asarray(raw_action, dtype=np.float32)
    if raw_action.shape != (2,):
        raise ValueError(f"Expected raw intercept action shape (2,), got {raw_action.shape}")

    target_s_m = float(raw_action[0] * action_std[0] + action_mean[0])
    execute_logit = float(raw_action[1])
    execute_probability = float(torch.sigmoid(torch.tensor(execute_logit, dtype=torch.float32)).item())

    if not np.isfinite(target_s_m):
        raise ValueError(f"Non-finite target_s_m={target_s_m}")
    if not np.isfinite(execute_logit):
        raise ValueError(f"Non-finite execute_logit={execute_logit}")
    if not np.isfinite(execute_probability):
        raise ValueError(f"Non-finite execute_probability={execute_probability}")
    if not 0.0 <= execute_probability <= 1.0:
        raise ValueError(f"execute_probability out of range: {execute_probability}")

    return target_s_m, execute_logit, execute_probability


def build_intercept_prediction_data(
    target_s_m: float,
    execute_probability: float,
) -> List[float]:
    return [float(target_s_m), float(execute_probability)]


def build_qpos_from_joint_state(
    joint_names: List[str],
    joint_pos: np.ndarray,
    state_dim: int,
) -> np.ndarray:
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    required = list(ARM_JOINT_NAMES)
    if state_dim == 8:
        required += FINGER_JOINT_NAMES
    missing = [name for name in required if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing required joints in JointState: {missing}")

    if state_dim not in (7, 8):
        raise ValueError(f"Unsupported state_dim={state_dim}; expected 7 or 8")

    qpos = np.empty(state_dim, dtype=np.float32)
    for i, name in enumerate(ARM_JOINT_NAMES):
        qpos[i] = np.float32(joint_pos[name_to_idx[name]])

    if state_dim == 8:
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
        self.rollout_mode = args.rollout_mode
        self.joint_topic = args.joint_topic
        self.twist_topic = args.twist_topic
        self.gripper_state_topic = args.gripper_state_topic
        self.prediction_topic = args.intercept_prediction_topic
        self.prediction_log_interval = max(1, int(args.prediction_log_interval))
        self.fps = args.fps
        self.max_timesteps = args.max_timesteps
        self.temporal_agg = args.temporal_agg

        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
        stats_path = os.path.join(args.ckpt_dir, args.stats_name)
        stats = load_rollout_stats(stats_path)
        rollout_cfg = resolve_rollout_configuration(args, stats)

        self.qpos_mean = rollout_cfg["qpos_mean"]
        self.qpos_std = rollout_cfg["qpos_std"]
        self.action_mean = rollout_cfg["action_mean"]
        self.action_std = rollout_cfg["action_std"]
        self.state_dim = rollout_cfg["state_dim"]
        self.action_dim = rollout_cfg["action_dim"]
        self.rgb_history_frames = rollout_cfg["rgb_history_frames"]
        self.image_channels = rollout_cfg["image_channels"]
        self.image_size = rollout_cfg["image_size"]
        self.camera_names = rollout_cfg["camera_names"]
        self.rgb_frame_order = rollout_cfg["rgb_frame_order"]
        self.camera_name = self.camera_names[0]
        self.image_topic = resolve_default_image_topic(
            args.image_topic,
            self.rollout_mode,
            self.camera_name,
        )

        if self.rollout_mode == "intercept":
            ensure_policy_image_normalizer_support()

        self.latest_image_msg: Optional[Image] = None
        self.latest_joint_msg: Optional[JointState] = None
        self.rgb_buffer = deque(maxlen=self.rgb_history_frames)
        self.image_sequence_gate = ImageSequenceGate(self.rgb_history_frames)

        self.t = 0
        self.prediction_counter = 0
        self.running = args.start_immediately
        self.all_actions = None

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
            "action_dim": self.action_dim,
            "image_channels": self.image_channels,
            "image_size": self.image_size,
            "use_bce_last_action_dim": args.use_bce_last_action_dim,
        }

        self.num_queries = policy_config["num_queries"]
        self.query_frequency = 1 if self.temporal_agg else self.num_queries

        self.policy = policy_module.ACTPolicy(policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.policy.to(self.device)
        self.policy.eval()

        if self.rollout_mode == "twist_gripper" and self.action_dim != 7:
            self.get_logger().warn(
                f"Expected action_dim=7 for twist+gripper, got {self.action_dim}. "
                "Will still run, but publishing uses action[0:6] + action[6]."
            )

        self.pre_process = lambda s: (s - self.qpos_mean) / np.clip(self.qpos_std, 1e-6, None)

        if self.rollout_mode == "twist_gripper":
            self.close_threshold = 0.7
            self.open_threshold = 0.3
            self.gripper_closed_state = False
            self.last_gripper_state_publish_time = 0.0
            self.min_gripper_publish_interval = 0.5
            self.sent_initial_gripper_state = False

            if self.temporal_agg:
                self.all_time_actions = torch.zeros(
                    [self.max_timesteps, self.max_timesteps + self.num_queries, self.action_dim],
                    device=self.device,
                )

            self.twist_pub = self.create_publisher(TwistStamped, self.twist_topic, 10)
            self.gripper_state_pub = self.create_publisher(Bool, self.gripper_state_topic, 10)
        else:
            # Prediction contract: msg.data == [target_s_m, execute_probability].
            self.prediction_pub = self.create_publisher(
                Float32MultiArray,
                self.prediction_topic,
                1,
            )

        image_qos = qos_profile_sensor_data if self.rollout_mode == "intercept" else 10
        self.create_subscription(Image, self.image_topic, self.image_cb, image_qos)
        self.create_subscription(JointState, self.joint_topic, self.joint_cb, 10)
        self.timer = self.create_timer(1.0 / self.fps, self.timer_cb)

        self.get_logger().info(f"Using device: {self.device}")
        self.get_logger().info(f"Checkpoint load status: {loading_status}")
        self.get_logger().info(f"rollout_mode={self.rollout_mode}")
        self.get_logger().info(f"checkpoint_path={ckpt_path}")
        self.get_logger().info(f"stats_path={stats_path}")
        self.get_logger().info(f"resolved_state_dim={self.state_dim}")
        self.get_logger().info(f"resolved_action_dim={self.action_dim}")
        self.get_logger().info(f"image_topic={self.image_topic}")
        self.get_logger().info(f"prediction_topic={self.prediction_topic}")
        self.get_logger().info(f"joint_topic={self.joint_topic}")
        self.get_logger().info(f"twist_topic={self.twist_topic}")
        self.get_logger().info(f"gripper_state_topic={self.gripper_state_topic}")
        self.get_logger().info(f"image_size={self.image_size}")
        self.get_logger().info(f"rgb_history_frames={self.rgb_history_frames}")
        self.get_logger().info(f"image_channels={self.image_channels}")
        self.get_logger().info(f"camera_names={self.camera_names}")
        self.get_logger().info(f"rgb_frame_order={self.rgb_frame_order}")
        self.get_logger().info(f"chunk_size={self.num_queries}")
        self.get_logger().info(f"temporal_agg={self.temporal_agg}")
        if self.running:
            self.get_logger().info("Rollout starts immediately.")
        else:
            self.get_logger().info("Waiting for first valid observation, then rollout will start.")

    def image_cb(self, msg: Image) -> None:
        if self.rollout_mode == "intercept":
            self.rgb_buffer.append(msg)
            self.image_sequence_gate.note_image_received()
        self.latest_image_msg = msg

    def joint_cb(self, msg: JointState) -> None:
        self.latest_joint_msg = msg

    def ready(self) -> bool:
        if self.latest_joint_msg is None:
            return False
        if self.rollout_mode == "intercept":
            return len(self.rgb_buffer) == self.rgb_history_frames
        return self.latest_image_msg is not None

    def build_policy_inputs_from_image(
        self,
        image_msg: Image,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        if self.latest_joint_msg is None:
            raise RuntimeError("JointState not received")

        joint_msg = self.latest_joint_msg
        qpos_numpy = build_qpos_from_joint_state(
            joint_names=list(joint_msg.name),
            joint_pos=np.asarray(joint_msg.position, dtype=np.float32),
            state_dim=self.state_dim,
        )

        qpos_norm = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)

        image = ros_img_to_torch_img(image_msg, self.bridge)
        curr_image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(self.device)
        return qpos_numpy, qpos, curr_image

    def build_legacy_policy_inputs(self) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        if self.latest_image_msg is None:
            raise RuntimeError("Image not received")
        return self.build_policy_inputs_from_image(self.latest_image_msg)

    def build_intercept_policy_inputs(self) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        if self.latest_joint_msg is None:
            raise RuntimeError("JointState not received")
        if len(self.rgb_buffer) != self.rgb_history_frames:
            raise RuntimeError(
                f"RGB history incomplete: have {len(self.rgb_buffer)}, need {self.rgb_history_frames}"
            )

        joint_msg = self.latest_joint_msg
        qpos_numpy = build_qpos_from_joint_state(
            joint_names=list(joint_msg.name),
            joint_pos=np.asarray(joint_msg.position, dtype=np.float32),
            state_dim=self.state_dim,
        )
        qpos_norm = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)

        rgb_frames = [rgb_image_msg_to_numpy(msg, self.bridge) for msg in self.rgb_buffer]
        curr_image = preprocess_rgb_history_images(rgb_frames, self.image_size).to(self.device)
        assert tuple(curr_image.shape) == (
            1,
            1,
            self.image_channels,
            self.image_size,
            self.image_size,
        ), curr_image.shape
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
        return raw_action

    def infer_intercept_action(
        self,
        qpos: torch.Tensor,
        curr_image: torch.Tensor,
    ) -> np.ndarray:
        action_chunk = self.policy(qpos, curr_image)
        raw_action = select_current_intercept_action(action_chunk, self.num_queries)
        raw_action = raw_action.detach().cpu().numpy()
        if raw_action.shape != (2,):
            raise RuntimeError(f"Expected intercept raw_action shape (2,), got {raw_action.shape}")
        return raw_action

    def publish_initial_gripper_open(self) -> None:
        msg = Bool()
        msg.data = False
        self.gripper_state_pub.publish(msg)
        self.gripper_closed_state = False
        self.last_gripper_state_publish_time = time.time()
        self.sent_initial_gripper_state = True
        self.get_logger().info("Published initial gripper state OPEN")

    def publish_action(self, raw_action: np.ndarray) -> Tuple[float, bool, bool]:
        if raw_action.shape[0] < 7:
            raise RuntimeError(f"Predicted action has dim {raw_action.shape[0]}, expected at least 7")

        twist = raw_action[:6] * self.action_std[:6] + self.action_mean[:6]

        clip_mag = 0.05
        twist[0] = np.clip(twist[0], -clip_mag, clip_mag)
        twist[1] = np.clip(twist[1], -clip_mag, clip_mag)
        twist[2] = np.clip(twist[2], -clip_mag, clip_mag)
        twist[3] = np.clip(twist[3], -0.10, 0.10)
        twist[4] = np.clip(twist[4], -0.10, 0.10)
        twist[5] = np.clip(twist[5], -0.10, 0.10)

        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"
        twist_msg.twist.linear.x = float(twist[0])
        twist_msg.twist.linear.y = float(twist[1])
        twist_msg.twist.linear.z = float(twist[2])
        twist_msg.twist.angular.x = float(twist[3])
        twist_msg.twist.angular.y = float(twist[4])
        twist_msg.twist.angular.z = float(twist[5])
        self.twist_pub.publish(twist_msg)

        grip_logit = float(raw_action[-1])
        grip_prob = 1.0 / (1.0 + np.exp(-grip_logit))

        new_state = self.gripper_closed_state
        if (not self.gripper_closed_state) and (grip_prob > self.close_threshold):
            new_state = True
        elif self.gripper_closed_state and (grip_prob < self.open_threshold):
            new_state = False

        did_publish = False
        now_sec = time.time()
        if new_state != self.gripper_closed_state:
            if (now_sec - self.last_gripper_state_publish_time) > self.min_gripper_publish_interval:
                gripper_msg = Bool()
                gripper_msg.data = new_state
                self.gripper_state_pub.publish(gripper_msg)
                self.gripper_closed_state = new_state
                self.last_gripper_state_publish_time = now_sec
                did_publish = True

        return grip_logit, grip_prob, did_publish

    def publish_intercept_prediction(self, raw_action: np.ndarray) -> Tuple[float, float, float]:
        target_s_m, execute_logit, execute_probability = convert_intercept_action(
            raw_action=raw_action,
            action_mean=self.action_mean,
            action_std=self.action_std,
        )

        msg = Float32MultiArray()
        msg.data = build_intercept_prediction_data(target_s_m, execute_probability)
        self.prediction_pub.publish(msg)
        return target_s_m, execute_logit, execute_probability

    def run_twist_gripper_policy_step(self) -> None:
        with torch.inference_mode():
            t0 = time.time()
            qpos_numpy, qpos, curr_image = self.build_legacy_policy_inputs()
            raw_action = self.infer_action(qpos, curr_image)
            grip_logit, grip_prob, did_publish = self.publish_action(raw_action)

            dt_ms = (time.time() - t0) * 1000.0
            self.get_logger().info(
                f"t={self.t:04d} qpos={qpos_numpy.tolist()} "
                f"raw_action={raw_action.tolist()} grip_logit={grip_logit:.3f} grip_prob={grip_prob:.3f} "
                f"gripper_state={'CLOSED' if self.gripper_closed_state else 'OPEN'} published={did_publish} "
                f"inference_ms={dt_ms:.2f}"
            )

            self.t += 1
            if self.t >= self.max_timesteps:
                self.get_logger().info("Reached max_timesteps, wrapping timestep counter.")
                self.t = 0
                if self.temporal_agg:
                    self.all_time_actions.zero_()

    def run_intercept_policy_step(self, image_sequence: int) -> None:
        with torch.inference_mode():
            t0 = time.time()
            _qpos_numpy, qpos, curr_image = self.build_intercept_policy_inputs()
            raw_action = self.infer_intercept_action(qpos, curr_image)

            try:
                target_s_m, execute_logit, execute_probability = self.publish_intercept_prediction(raw_action)
            except ValueError as exc:
                self.get_logger().warn(
                    f"Skipping invalid intercept prediction for image_seq={image_sequence}: {exc}"
                )
                return

            self.prediction_counter += 1
            dt_ms = (time.time() - t0) * 1000.0
            if self.prediction_counter % self.prediction_log_interval == 0:
                newest_msg = self.rgb_buffer[-1]
                stamp_text = "n/a"
                if hasattr(newest_msg, "header") and hasattr(newest_msg.header, "stamp"):
                    stamp = newest_msg.header.stamp
                    stamp_text = f"{stamp.sec}.{stamp.nanosec:09d}"

                self.get_logger().info(
                    f"prediction={self.prediction_counter:05d} image_seq={image_sequence} image_stamp={stamp_text} "
                    f"target_s_m={target_s_m:.6f} execute_logit={execute_logit:.6f} "
                    f"execute_probability={execute_probability:.6f} inference_ms={dt_ms:.2f} "
                    f"input_shape={tuple(curr_image.shape)}"
                )

    def run_policy_step(self) -> None:
        if self.rollout_mode == "intercept":
            image_sequence = self.image_sequence_gate.claim_latest_ready_sequence(len(self.rgb_buffer))
            if image_sequence is None:
                return
            self.run_intercept_policy_step(image_sequence)
            return
        self.run_twist_gripper_policy_step()

    def timer_cb(self) -> None:
        if not self.ready():
            return

        if self.rollout_mode == "twist_gripper" and not self.sent_initial_gripper_state:
            self.publish_initial_gripper_open()

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
    parser.add_argument(
        "--rollout_mode",
        type=str,
        choices=["twist_gripper", "intercept"],
        default="twist_gripper",
    )

    parser.add_argument("--image_topic", type=str)
    parser.add_argument("--joint_topic", type=str, default="/joint_states")
    parser.add_argument("--twist_topic", type=str, default="/cartesian_cmd/twist")
    parser.add_argument("--gripper_state_topic", type=str, default="/teleop/gripper_state_cmd")
    parser.add_argument(
        "--intercept_prediction_topic",
        type=str,
        default="/act/intercept_prediction",
    )

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_timesteps", type=int, default=100000)

    parser.add_argument("--state_dim", type=int, default=None)
    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--camera_name", type=str, default="rgb")
    parser.add_argument("--prediction_log_interval", type=int, default=10)

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
    parser.add_argument("--use_bce_last_action_dim", action="store_true")
    parser.add_argument("--no_use_bce_last_action_dim", action="store_false", dest="use_bce_last_action_dim")
    parser.set_defaults(use_bce_last_action_dim=True)

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
