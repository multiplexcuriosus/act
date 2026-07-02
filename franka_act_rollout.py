#!/usr/bin/env python3
import os
import sys
import time
import argparse
import pickle
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

import rclpy
from rclpy.node import Node

from policy import ACTPolicy


def validate_camera_names(camera_names: List[str]) -> List[str]:
    if camera_names == ["rgb"]:
        return camera_names
    if camera_names == ["event"]:
        return camera_names
    if camera_names == ["rgb", "event"]:
        return camera_names
    if set(camera_names) == {"rgb", "event"}:
        raise ValueError(
            "RGB+event rollout only supports '--camera_names rgb event'. "
            "Do not use '--camera_names event rgb' because this changes the model input slots."
        )
    raise ValueError(
        f"Unsupported camera_names={camera_names}. "
        "Allowed: ['rgb'], ['event'], or ['rgb', 'event']."
    )


def default_image_topic_for_camera(cam_name: str) -> str:
    if cam_name == "rgb":
        return "/camera/camera/color/image_raw"
    if cam_name == "event":
        return "/openmv_cam/event_frame_3ch"
    raise ValueError(f"Unsupported camera name: {cam_name}")


def ensure_hwc3(image: np.ndarray, cam_name: str) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"{cam_name}: expected image shape [H,W], [H,W,1], or [H,W,3], got {tuple(image.shape)}")


def extract_event_single_channel(
    image: np.ndarray,
    event_channel_index: int,
    cam_name: str,
) -> np.ndarray:
    image = np.asarray(image)

    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] >= 3:
        if not (0 <= event_channel_index < image.shape[2]):
            raise ValueError(
                f"Invalid --event_channel_index={event_channel_index} for {cam_name} image shape {tuple(image.shape)}"
            )
        return image[:, :, event_channel_index]

    raise ValueError(
        f"{cam_name}: expected event image shape [H,W], [H,W,1], or [H,W,>=3], got {tuple(image.shape)}"
    )


def ensure_hw_or_hwc1(image: np.ndarray, cam_name: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image
    raise ValueError(f"{cam_name}: expected image shape [H,W] or [H,W,1], got {tuple(image.shape)}")


def maybe_resize_for_rgb_event(image: np.ndarray, cam_name: str, camera_names: List[str]) -> np.ndarray:
    if camera_names != ["rgb", "event"]:
        return image

    if cam_name == "rgb":
        if image.shape[:2] != (320, 320):
            image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)
        return image

    if cam_name == "event":
        if image.shape[:2] != (320, 320):
            raise ValueError(
                f"event image must be 320x320 in RGB+event rollout, got {tuple(image.shape)}"
            )
        return image

    raise ValueError(f"Unsupported camera '{cam_name}' in camera_names={camera_names}")


def resize_image_np(
    img_rgb: np.ndarray,
    target_size: Tuple[int, int] = (320, 320),
    resize_mode: str = "warp",
) -> np.ndarray:
    target_h, target_w = target_size

    if resize_mode == "none":
        return img_rgb

    if resize_mode == "warp":
        return cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if resize_mode == "letterbox":
        src_h, src_w = img_rgb.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(img_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target_h, target_w, 3), dtype=img_rgb.dtype)
        top = (target_h - resized_h) // 2
        left = (target_w - resized_w) // 2
        canvas[top:top + resized_h, left:left + resized_w] = resized
        return canvas

    raise ValueError(f"Unsupported resize_mode: {resize_mode}")


def resize_single_channel_image_np(
    img: np.ndarray,
    target_size: Tuple[int, int] = (320, 320),
    resize_mode: str = "warp",
) -> np.ndarray:
    target_h, target_w = target_size

    img = ensure_hw_or_hwc1(img, "event")
    if img.ndim == 3:
        img = img[:, :, 0]

    if resize_mode == "none":
        return img

    if resize_mode == "warp":
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if resize_mode == "letterbox":
        src_h, src_w = img.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target_h, target_w), dtype=img.dtype)
        top = (target_h - resized_h) // 2
        left = (target_w - resized_w) // 2
        canvas[top:top + resized_h, left:left + resized_w] = resized
        return canvas

    raise ValueError(f"Unsupported resize_mode: {resize_mode}")


def infer_event_input_channels_from_state_dict(state_dict: dict) -> Optional[int]:
    conv1_keys = [
        "model.backbones.0.0.body.conv1.weight",
        "backbones.0.0.body.conv1.weight",
    ]
    for key in conv1_keys:
        weight = state_dict.get(key)
        if weight is not None:
            if weight.ndim != 4:
                raise ValueError(f"Unexpected conv1 weight shape at '{key}': {tuple(weight.shape)}")
            in_channels = int(weight.shape[1])
            if in_channels not in (1, 3):
                raise ValueError(
                    f"Unsupported conv1 input channels inferred from checkpoint key '{key}': {in_channels}"
                )
            return in_channels
    return None


def ros_img_to_torch_img(
    msg: Image,
    bridge: CvBridge,
    target_size: Tuple[int, int] = (320, 320),
    resize_mode: str = "warp",
) -> Tuple[torch.Tensor, Tuple[int, ...], Tuple[int, ...]]:
    img_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    original_shape = tuple(img_rgb.shape)

    if resize_mode != "none":
        img_rgb = resize_image_np(img_rgb, target_size=target_size, resize_mode=resize_mode)

    processed_shape = tuple(img_rgb.shape)
    img_float = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).to(torch.float32)
    return img_float, original_shape, processed_shape


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

        if args.camera_names is not None and args.camera_name is not None:
            raise ValueError("Use either --camera_names or --camera_name, not both.")

        raw_camera_names = args.camera_names if args.camera_names is not None else [args.camera_name or "rgb"]
        self.camera_names = validate_camera_names(list(raw_camera_names))

        self.image_topic = args.image_topic
        self.joint_topic = args.joint_topic
        self.twist_topic = args.twist_topic
        self.gripper_state_topic = args.gripper_state_topic
        self.camera_name = self.camera_names[0]
        self.image_size = tuple(args.image_size)
        self.resize_mode = args.resize_mode
        self.event_channel_index = int(args.event_channel_index)
        self.freeze_startup_image = args.freeze_startup_image
        self.dryrun = args.dryrun
        self.logged_first_image_stats = False

        if args.event_input_channels not in (1, 3):
            raise ValueError(f"--event_input_channels must be 1 or 3, got {args.event_input_channels}")
        self.event_input_channels = int(args.event_input_channels)
        self.event_input_channels_explicit = bool(getattr(args, "event_input_channels_explicit", False))

        if self.event_input_channels == 1 and self.camera_names != ["event"]:
            raise ValueError(
                "--event_input_channels 1 is currently supported only for --camera_names event. "
                "Use --event_input_channels 3 for RGB or RGB+event rollout."
            )

        if len(self.camera_names) > 1 and self.image_topic is not None:
            raise ValueError(
                "--image_topic is only supported for single-camera rollout. "
                "For multi-camera rollout, use default topics based on --camera_names order."
            )

        self.image_topics = {
            cam: (self.image_topic if len(self.camera_names) == 1 and self.image_topic is not None
                  else default_image_topic_for_camera(cam))
            for cam in self.camera_names
        }



        self.fps = args.fps
        self.max_timesteps = args.max_timesteps
        self.temporal_agg = args.temporal_agg
        self.state_dim = args.state_dim
        self.action_dim_cfg = args.action_dim

        self.latest_image_msg: Optional[Image] = None
        self.frozen_image_msg: Optional[Image] = None
        self.latest_image_msgs = {cam: None for cam in self.camera_names}
        self.frozen_image_msgs = {cam: None for cam in self.camera_names}
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
            "camera_names": self.camera_names,
            "event_input_channels": self.event_input_channels,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim_cfg,
            "use_bce_last_action_dim": args.use_bce_last_action_dim,
        }

        self.num_queries = policy_config["num_queries"]
        self.query_frequency = 1 if self.temporal_agg else self.num_queries
        self.all_actions = None

        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
        stats_path = os.path.join(args.ckpt_dir, args.stats_name)

        ckpt_obj = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj

        if self.camera_names == ["event"]:
            inferred_event_input_channels = infer_event_input_channels_from_state_dict(state_dict)
            if inferred_event_input_channels is not None:
                if self.event_input_channels_explicit and self.event_input_channels != inferred_event_input_channels:
                    raise ValueError(
                        f"Checkpoint expects event_input_channels={inferred_event_input_channels} from conv1, "
                        f"but CLI requested --event_input_channels {self.event_input_channels}."
                    )
                if not self.event_input_channels_explicit:
                    self.event_input_channels = inferred_event_input_channels
                    self.get_logger().info(
                        f"Using event_input_channels={self.event_input_channels} inferred from checkpoint conv1."
                    )
                policy_config["event_input_channels"] = self.event_input_channels

        trained_camera_names = None
        if isinstance(ckpt_obj, dict):
            for cfg_key in ["policy_config", "config"]:
                cfg = ckpt_obj.get(cfg_key)
                if isinstance(cfg, dict) and "camera_names" in cfg:
                    trained_camera_names = cfg["camera_names"]
                    break

        self.policy = ACTPolicy(policy_config)
        loading_status = self.policy.load_state_dict(state_dict)
        self.get_logger().info(f"Checkpoint load status: {loading_status}")
        self.policy.to(self.device)
        self.policy.eval()
        self.get_logger().info(f"Loaded checkpoint: {ckpt_path}")

        if self.camera_names == ["event"]:
            conv1_in_channels = int(self.policy.model.backbones[0][0].body.conv1.in_channels)
            if conv1_in_channels != self.event_input_channels:
                raise RuntimeError(
                    "Sanity check failed: model conv1.in_channels does not match event_input_channels "
                    f"({conv1_in_channels} vs {self.event_input_channels})."
                )
            self.get_logger().info(
                f"Sanity check: event mode conv1.in_channels={conv1_in_channels}"
            )

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        if self.camera_names == ["event"] and "event_input_channels" in stats:
            stats_event_channels = int(stats["event_input_channels"])
            if stats_event_channels != self.event_input_channels:
                raise ValueError(
                    f"dataset_stats event_input_channels={stats_event_channels} but rollout uses event_input_channels={self.event_input_channels}."
                )

        for k in ["qpos_mean", "qpos_std", "action_mean", "action_std"]:
            if k not in stats:
                raise KeyError(f"Missing key '{k}' in dataset stats: {stats_path}")

        if trained_camera_names is None and "camera_names" in stats:
            trained_camera_names = stats["camera_names"]

        if trained_camera_names is None:
            self.get_logger().warn(
                "Checkpoint/stats do not include camera_names. Cannot verify rollout camera order against training config."
            )
        else:
            trained_camera_names = list(trained_camera_names)
            if trained_camera_names != self.camera_names:
                raise ValueError(
                    f"Checkpoint camera_names={trained_camera_names} but rollout camera_names={self.camera_names}. "
                    "Camera order must match training exactly."
                )

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

        # Binary gripper-state postprocessing from BCE logits.
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

        for cam_name in self.camera_names:
            topic = self.image_topics[cam_name]
            self.create_subscription(
                Image,
                topic,
                lambda msg, cam=cam_name: self.image_cb(msg, cam),
                10,
            )
        self.create_subscription(JointState, self.joint_topic, self.joint_cb, 10)

        self.timer = self.create_timer(1.0 / self.fps, self.timer_cb)

        self.get_logger().info(f"camera_names={self.camera_names}")
        self.get_logger().info(f"image_topics={self.image_topics}")
        self.get_logger().info(f"joint_topic={self.joint_topic}")
        self.get_logger().info(f"twist_topic={self.twist_topic}")
        self.get_logger().info(f"gripper_state_topic={self.gripper_state_topic}")
        self.get_logger().info(f"dryrun={self.dryrun} (twist publish disabled when true)")
        self.get_logger().info(
            f"state_dim={self.state_dim} action_dim={self.action_dim} temporal_agg={self.temporal_agg} fps={self.fps}"
        )
        self.get_logger().info(
            f"image_preprocess resize_mode={self.resize_mode} target_size={self.image_size} freeze_startup_image={self.freeze_startup_image}"
        )
        self.get_logger().info(
            f"event_input_channels={self.event_input_channels} event_channel_index={self.event_channel_index}"
        )
        if self.resize_mode != "warp":
            self.get_logger().warn(
                "Current checkpoints were trained with warp resize to 320x320. Use other resize modes only with matching retrained checkpoints."
            )
        if self.resize_mode == "letterbox":
            self.get_logger().warn(
                "Letterbox resize selected. Use this only with policies trained using letterbox preprocessing."
            )
        if self.freeze_startup_image:
            self.get_logger().warn(
                "freeze_startup_image is enabled. This is an ablation/debug mode and does not use normal live visual feedback."
            )
        if self.running:
            self.get_logger().info("Rollout starts immediately.")
        else:
            self.get_logger().info("Waiting for first valid observation, then rollout will start.")

        self.logged_camera_stack_debug = False

    def image_cb(self, msg: Image, cam_name: str) -> None:
        self.latest_image_msgs[cam_name] = msg
        if cam_name == self.camera_names[0]:
            self.latest_image_msg = msg

        if self.freeze_startup_image and self.frozen_image_msgs[cam_name] is None:
            self.frozen_image_msgs[cam_name] = msg
            if cam_name == self.camera_names[0]:
                self.frozen_image_msg = msg
            self.get_logger().info(f"Captured startup image for frozen-image rollout. camera={cam_name}")

    def joint_cb(self, msg: JointState) -> None:
        self.latest_joint_msg = msg

    def ready(self) -> bool:
        if self.freeze_startup_image:
            return all(self.frozen_image_msgs[cam] is not None for cam in self.camera_names) and self.latest_joint_msg is not None
        return all(self.latest_image_msgs[cam] is not None for cam in self.camera_names) and self.latest_joint_msg is not None

    def log_image_preprocess_once(
        self,
        original_shape: Tuple[int, ...],
        processed_shape: Tuple[int, ...],
        curr_image: torch.Tensor,
    ) -> None:
        if self.logged_first_image_stats:
            return

        image_cpu = curr_image.detach().cpu()
        self.get_logger().info(
            "First image preprocess: "
            f"original_shape={original_shape} processed_shape={processed_shape} "
            f"policy_tensor_shape={tuple(curr_image.shape)} resize_mode={self.resize_mode} target_size={self.image_size}"
        )
        self.get_logger().info(
            "First image stats after normalization: "
            f"min={float(image_cpu.min()):.6f} max={float(image_cpu.max()):.6f} "
            f"mean={float(image_cpu.mean()):.6f} std={float(image_cpu.std()):.6f}"
        )
        self.logged_first_image_stats = True

    def build_policy_inputs(self) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
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

        image_chw_list = []
        original_shapes = []
        processed_shapes = []
        for cam_name in self.camera_names:
            image_msg_for_policy = self.frozen_image_msgs[cam_name] if self.freeze_startup_image else self.latest_image_msgs[cam_name]
            if image_msg_for_policy is None:
                raise RuntimeError(f"Image not received for camera '{cam_name}'")

            if self.camera_names == ["event"] and self.event_input_channels == 1:
                image_np = self.bridge.imgmsg_to_cv2(image_msg_for_policy, desired_encoding='passthrough')
                image_np = extract_event_single_channel(
                    np.asarray(image_np),
                    event_channel_index=self.event_channel_index,
                    cam_name=cam_name,
                )
                original_shapes.append(tuple(image_np.shape))

                if self.resize_mode != "none":
                    image_np = resize_single_channel_image_np(
                        image_np,
                        target_size=self.image_size,
                        resize_mode=self.resize_mode,
                    )
                image_np = ensure_hw_or_hwc1(image_np, cam_name)
                if image_np.ndim == 2:
                    image_np = image_np[:, :, None]
            else:
                image_np = self.bridge.imgmsg_to_cv2(image_msg_for_policy, desired_encoding='rgb8')
                original_shapes.append(tuple(image_np.shape))

                image_np = ensure_hwc3(np.asarray(image_np), cam_name)

                if self.camera_names == ["rgb", "event"]:
                    image_np = maybe_resize_for_rgb_event(image_np, cam_name, self.camera_names)
                elif self.resize_mode != "none":
                    image_np = resize_image_np(image_np, target_size=self.image_size, resize_mode=self.resize_mode)

            processed_shapes.append(tuple(image_np.shape))
            image_chw_list.append(np.transpose(image_np, (2, 0, 1)))

        image_stack = np.stack(image_chw_list, axis=0)
        curr_image = torch.from_numpy(image_stack).to(torch.float32).div(255.0).unsqueeze(0).to(self.device)

        if self.camera_names == ["rgb", "event"]:
            expected_shape = (1, 2, 3, 320, 320)
            if tuple(curr_image.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected policy image shape for rgb+event rollout: got {tuple(curr_image.shape)}, expected {expected_shape}"
                )
        elif self.camera_names == ["event"] and self.resize_mode == "warp" and self.image_size == (320, 320):
            expected_channels = self.event_input_channels
            expected_shape = (1, 1, expected_channels, 320, 320)
            if tuple(curr_image.shape) != expected_shape:
                raise RuntimeError(
                    "Unexpected policy image shape for event rollout: "
                    f"got {tuple(curr_image.shape)}, expected {expected_shape}"
                )
        elif self.resize_mode == "warp" and self.image_size == (320, 320) and len(self.camera_names) == 1:
            expected_shape = (1, 1, 3, 320, 320)
            if tuple(curr_image.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected policy image shape for default rollout: got {tuple(curr_image.shape)}, expected {expected_shape}"
                )

        if self.camera_names == ["event"]:
            if int(curr_image.shape[2]) != self.event_input_channels:
                raise RuntimeError(
                    "Sanity check failed: curr_image channels do not match event_input_channels "
                    f"({int(curr_image.shape[2])} vs {self.event_input_channels})."
                )

        if not self.logged_camera_stack_debug:
            self.get_logger().info(
                f"[DEBUG] rollout camera_names={self.camera_names}, curr_image.shape={curr_image.shape}"
            )
            self.logged_camera_stack_debug = True

        self.log_image_preprocess_once(tuple(original_shapes), tuple(processed_shapes), curr_image)

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

        # Denormalize only continuous twist dimensions.
        twist = raw_action[:6] * self.action_std[:6] + self.action_mean[:6]

        CLIP_MAG = 0.05

        twist[0] = np.clip(twist[0], -CLIP_MAG, CLIP_MAG)
        twist[1] = np.clip(twist[1], -CLIP_MAG, CLIP_MAG)
        twist[2] = np.clip(twist[2], -CLIP_MAG, CLIP_MAG)

        twist[3] = np.clip(twist[3], -0.10, 0.10)
        twist[4] = np.clip(twist[4], -0.10, 0.10)
        twist[5] = np.clip(twist[5], -0.10, 0.10)

        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"
        twist_msg.twist.linear.x = float(twist[0])
        twist_msg.twist.linear.y = float(twist[1]) #debug
        twist_msg.twist.linear.z = float(twist[2]) #debug
        twist_msg.twist.angular.x = float(twist[3])
        twist_msg.twist.angular.y = float(twist[4])
        twist_msg.twist.angular.z = float(twist[5])
        if not self.dryrun:
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

    def run_policy_step(self) -> None:
        with torch.inference_mode():
            t0 = time.time()
            qpos_numpy, qpos, curr_image = self.build_policy_inputs()
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

            # logging
            denorm_action = raw_action * self.action_std + self.action_mean

            print(
                f"raw={raw_action} "
                f"denorm={denorm_action} "
            )

    def timer_cb(self) -> None:
        if not self.ready():
            return

        if not self.sent_initial_gripper_state:
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

    parser.add_argument("--image_topic", type=str)
    parser.add_argument("--joint_topic", type=str, default="/joint_states")
    parser.add_argument("--twist_topic", type=str, default="/cartesian_cmd/twist")
    parser.add_argument("--gripper_state_topic", type=str, default="/teleop/gripper_state_cmd")
    parser.add_argument("--image_size", type=int, nargs=2, default=[320, 320], metavar=("H", "W"))
    parser.add_argument("--resize_mode", type=str, choices=["warp", "letterbox", "none"], default="warp")
    parser.add_argument("--event_input_channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--event_channel_index", type=int, default=2)
    parser.add_argument("--freeze_startup_image", action="store_true", default=False)
    parser.add_argument("--dryrun", action="store_true", default=False)

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_timesteps", type=int, default=100000)

    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--camera_names", type=str, nargs='+')
    parser.add_argument("--camera_name", type=str, choices=["rgb", "event"], default=None)

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
    args.event_input_channels_explicit = "--event_input_channels" in sys.argv

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
