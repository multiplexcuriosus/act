#!/usr/bin/env python3
import os
import time
import argparse
import pickle
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, Float64MultiArray

import rclpy
from rclpy.node import Node

from intercept_rollout_contract import (
    TemporalAbsoluteAggregator,
    absolute_s_from_anchor,
    build_rgb_history_tensor,
    denormalize_delta_chunk,
    extract_arm_qpos,
    select_sync_observation,
    validate_anchor_freshness,
    validate_intercept_stats_and_config,
)
from policy import ACTPolicy


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _first_last(values: np.ndarray, n: int = 3) -> str:
    flat = np.asarray(values).reshape(-1)
    if flat.size <= 2 * n:
        return np.array2string(flat, precision=5, separator=", ")
    first = np.array2string(flat[:n], precision=5, separator=", ")
    last = np.array2string(flat[-n:], precision=5, separator=", ")
    return f"{first} ... {last}"


class FrankaActRolloutNode(Node):
    def __init__(self, args):
        super().__init__("franka_act_rollout_intercept")

        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.image_topic = args.image_topic
        self.joint_topic = args.joint_topic
        self.current_tcp_s_topic = args.current_tcp_s_topic
        self.prediction_topic = args.prediction_topic
        self.prediction_current_topic = args.prediction_current_topic
        self.publish_current_scalar = bool(args.publish_current_scalar)

        self.fps = args.fps
        self.temporal_agg = args.temporal_agg
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.chunk_size = args.chunk_size
        self.image_size = int(args.image_size)
        self.max_source_buffer = int(args.max_source_buffer)
        self.max_observation_age_sec = float(args.max_observation_age_sec)
        self.max_anchor_age_sec = float(args.max_anchor_age_sec)
        self.diag_log_period_sec = float(args.diag_log_period_sec)
        self.reject_log_period_sec = float(args.reject_log_period_sec)

        self.rgb_buffer: Deque[Tuple[float, Image]] = deque(maxlen=self.max_source_buffer)
        self.joint_buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=self.max_source_buffer)
        self.tcp_buffer: Deque[Tuple[float, float]] = deque(maxlen=self.max_source_buffer)

        self.step_index = 0
        self.running = args.start_immediately
        self._last_diag_log_sec = 0.0
        self._last_reject_log_sec = 0.0
        self._last_reject_reason = ""

        policy_config = {
            "lr": args.lr,
            "num_queries": self.chunk_size,
            "kl_weight": args.kl_weight,
            "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": args.enc_layers,
            "dec_layers": args.dec_layers,
            "nheads": args.nheads,
            "camera_names": [args.camera_name],
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "use_bce_last_action_dim": args.use_bce_last_action_dim,
            "rgb_history_frames": args.rgb_history_frames,
            "image_channels": 9,
            "image_size": self.image_size,
        }

        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
        stats_path = os.path.join(args.ckpt_dir, args.stats_name)

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        stats_arrays = validate_intercept_stats_and_config(
            stats=stats,
            policy_config=policy_config,
            expected_chunk_size=self.chunk_size,
        )

        self.qpos_mean = stats_arrays["qpos_mean"]
        self.qpos_std = stats_arrays["qpos_std"]
        self.action_mean = stats_arrays["action_mean"]
        self.action_std = stats_arrays["action_std"]

        self.policy = ACTPolicy(policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.get_logger().info(f"Checkpoint load status: {loading_status}")
        self.policy.to(self.device)
        self.policy.eval()
        self.get_logger().info(f"Loaded checkpoint: {ckpt_path}")

        self.pre_process = lambda s: (s - self.qpos_mean) / self.qpos_std
        self.aggregator = TemporalAbsoluteAggregator(self.chunk_size) if self.temporal_agg else None

        self.prediction_pub = self.create_publisher(Float64MultiArray, self.prediction_topic, 10)
        self.prediction_current_pub = None
        if self.publish_current_scalar:
            self.prediction_current_pub = self.create_publisher(
                Float64,
                self.prediction_current_topic,
                10,
            )

        self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.create_subscription(JointState, self.joint_topic, self.joint_cb, 10)
        self.create_subscription(Float64, self.current_tcp_s_topic, self.current_tcp_s_cb, 10)

        self.timer = self.create_timer(1.0 / self.fps, self.timer_cb)

        self.get_logger().info(f"image_topic={self.image_topic}")
        self.get_logger().info(f"joint_topic={self.joint_topic}")
        self.get_logger().info(f"current_tcp_s_topic={self.current_tcp_s_topic}")
        self.get_logger().info(f"prediction_topic={self.prediction_topic}")
        if self.publish_current_scalar:
            self.get_logger().info(f"prediction_current_topic={self.prediction_current_topic}")
        self.get_logger().info(
            "This rollout node publishes interception predictions only; it does not publish robot motion commands."
        )
        self.get_logger().info(
            "Interception contract: "
            f"state_dim={self.state_dim} action_dim={self.action_dim} "
            f"chunk_size={self.chunk_size} rgb_history_frames={args.rgb_history_frames} "
            f"image_channels=9 use_bce_last_action_dim={args.use_bce_last_action_dim}"
        )
        if self.running:
            self.get_logger().info("Rollout starts immediately.")
        else:
            self.get_logger().info("Waiting for complete history and synchronized inputs before rollout starts.")

    def _append_monotonic(self, name: str, queue, timestamp_sec: float, value) -> None:
        if queue and timestamp_sec < queue[-1][0]:
            self.get_logger().warn(
                f"Ignoring out-of-order {name} sample: {timestamp_sec:.6f} < {queue[-1][0]:.6f}"
            )
            return
        queue.append((timestamp_sec, value))

    def image_cb(self, msg: Image) -> None:
        try:
            t = stamp_to_sec(msg.header.stamp)
        except Exception:
            t = self.get_clock().now().nanoseconds * 1e-9
        self._append_monotonic("RGB", self.rgb_buffer, t, msg)

    def joint_cb(self, msg: JointState) -> None:
        try:
            qpos = extract_arm_qpos(msg.name, msg.position)
        except Exception as exc:
            self.get_logger().warn(f"Dropping JointState sample: {exc}")
            return
        t = stamp_to_sec(msg.header.stamp)
        self._append_monotonic("JointState", self.joint_buffer, t, qpos)

    def current_tcp_s_cb(self, msg: Float64) -> None:
        value = float(msg.data)
        if not np.isfinite(value):
            self.get_logger().warn("Dropping non-finite /middle_line/current_tcp_s sample")
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        self._append_monotonic("current_tcp_s", self.tcp_buffer, t, value)

    def ready(self) -> bool:
        return bool(self.rgb_buffer) and bool(self.joint_buffer) and bool(self.tcp_buffer)

    def _reject(self, reason: str) -> None:
        now = time.time()
        if (
            reason != self._last_reject_reason
            or (now - self._last_reject_log_sec) >= self.reject_log_period_sec
        ):
            self.get_logger().warn(f"Inference rejected: {reason}")
            self._last_reject_log_sec = now
            self._last_reject_reason = reason

    def build_policy_inputs(self):
        rgb_timestamps = [item[0] for item in self.rgb_buffer]
        rgb_messages = [item[1] for item in self.rgb_buffer]
        joint_timestamps = [item[0] for item in self.joint_buffer]
        joint_qpos = [item[1] for item in self.joint_buffer]
        tcp_timestamps = [item[0] for item in self.tcp_buffer]
        tcp_values = [item[1] for item in self.tcp_buffer]

        sync = select_sync_observation(
            rgb_timestamps=rgb_timestamps,
            joint_timestamps=joint_timestamps,
            joint_qpos_samples=joint_qpos,
            tcp_timestamps=tcp_timestamps,
            tcp_values=tcp_values,
        )

        selected_rgb_msgs = [rgb_messages[index] for index in sync.history_indices]
        rgb_frames = [
            self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            for msg in selected_rgb_msgs
        ]
        image_np = build_rgb_history_tensor(rgb_frames, self.image_size)

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        anchor_observation_timestamp = sync.rgb_timestamps[-1]
        validate_anchor_freshness(
            anchor_timestamp=sync.anchor_tcp_s_timestamp,
            observation_timestamp=anchor_observation_timestamp,
            now_timestamp=now_sec,
            max_anchor_age_sec=self.max_anchor_age_sec,
            max_observation_age_sec=self.max_observation_age_sec,
        )

        qpos_norm = self.pre_process(sync.qpos_history)
        if not np.all(np.isfinite(qpos_norm)):
            raise ValueError("Normalized qpos contains non-finite values")

        qpos = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)
        image = torch.from_numpy(image_np).float().to(self.device)
        return sync, qpos, image, now_sec

    def publish_predictions(self, absolute_chunk: np.ndarray, current_value: Optional[float]) -> None:
        chunk_msg = Float64MultiArray()
        chunk_msg.data = [float(value) for value in absolute_chunk.tolist()]
        self.prediction_pub.publish(chunk_msg)

        if self.prediction_current_pub is not None and current_value is not None:
            current_msg = Float64()
            current_msg.data = float(current_value)
            self.prediction_current_pub.publish(current_msg)

    def run_policy_step(self) -> None:
        with torch.inference_mode():
            t0 = time.time()
            sync, qpos, curr_image, now_sec = self.build_policy_inputs()

            raw_output = self.policy(qpos, curr_image)
            expected_shape = (1, self.chunk_size, 1)
            if tuple(raw_output.shape) != expected_shape:
                raise RuntimeError(
                    f"Policy output shape mismatch: expected {expected_shape}, got {tuple(raw_output.shape)}"
                )
            if not torch.isfinite(raw_output).all():
                raise RuntimeError("Policy output contains non-finite values")

            normalized_delta = raw_output[0, :, 0].detach().cpu().numpy()
            delta_s = denormalize_delta_chunk(normalized_delta, self.action_mean, self.action_std)
            absolute_s = absolute_s_from_anchor(sync.anchor_tcp_s, delta_s)

            current_value = float(absolute_s[0])
            if self.temporal_agg:
                self.aggregator.add_prediction(self.step_index, absolute_s)
                aggregated = self.aggregator.value_for_step(self.step_index)
                if aggregated is not None:
                    current_value = float(aggregated)

            self.publish_predictions(absolute_s, current_value)

            dt_ms = (time.time() - t0) * 1000.0
            max_input_age = now_sec - sync.rgb_timestamps[-1]
            max_sync_error = max(
                max(abs(rgb_ts - qpos_ts) for rgb_ts, qpos_ts in zip(sync.rgb_timestamps, sync.qpos_timestamps)),
                sync.rgb_timestamps[-1] - sync.anchor_tcp_s_timestamp,
            )

            now_wall = time.time()
            if (now_wall - self._last_diag_log_sec) >= self.diag_log_period_sec:
                self.get_logger().info(
                    "Inference diagnostics: "
                    f"step={self.step_index} "
                    f"history_indices={list(sync.history_indices)} "
                    f"rgb_ts={[round(ts, 6) for ts in sync.rgb_timestamps]} "
                    f"qpos_ts={[round(ts, 6) for ts in sync.qpos_timestamps]} "
                    f"anchor_s={sync.anchor_tcp_s:+.6f} anchor_ts={sync.anchor_tcp_s_timestamp:.6f} "
                    f"qpos_shape={tuple(sync.qpos_history.shape)} image_shape={tuple(curr_image.shape)} "
                    f"raw_shape={tuple(raw_output.shape)} delta_shape={tuple(delta_s.shape)} abs_shape={tuple(absolute_s.shape)} "
                    f"delta_s={_first_last(delta_s)} abs_s={_first_last(absolute_s)} "
                    f"current_abs={current_value:+.6f} max_input_age={max_input_age:.4f}s "
                    f"max_sync_error={max_sync_error:.4f}s inference_ms={dt_ms:.2f}"
                )
                self._last_diag_log_sec = now_wall

            self.step_index += 1

    def timer_cb(self) -> None:
        if not self.ready():
            self._reject("waiting for RGB, JointState, and current_tcp_s streams")
            return

        if not self.running:
            self.running = True
            self.get_logger().info("Received first valid synchronized streams. Starting interception rollout.")

        try:
            self.run_policy_step()
        except ValueError as exc:
            self._reject(str(exc))
        except RuntimeError as exc:
            self._reject(str(exc))
        except Exception as exc:
            self.get_logger().error(f"Policy step failed: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="policy_val_best.ckpt")
    parser.add_argument("--stats_name", type=str, default="dataset_stats.pkl")

    parser.add_argument("--image_topic", type=str, default="/top_cam/camera/color/image_raw")
    parser.add_argument("--joint_topic", type=str, default="/joint_states")
    parser.add_argument("--current_tcp_s_topic", type=str, default="/middle_line/current_tcp_s")
    parser.add_argument("--prediction_topic", type=str, default="/act/intercept_prediction")
    parser.add_argument(
        "--prediction_current_topic",
        type=str,
        default="/act/intercept_prediction_current",
    )
    parser.add_argument("--publish_current_scalar", action="store_true")
    parser.add_argument("--no_publish_current_scalar", action="store_false", dest="publish_current_scalar")
    parser.set_defaults(publish_current_scalar=True)

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_source_buffer", type=int, default=256)
    parser.add_argument("--max_observation_age_sec", type=float, default=0.20)
    parser.add_argument("--max_anchor_age_sec", type=float, default=0.10)
    parser.add_argument("--diag_log_period_sec", type=float, default=1.0)
    parser.add_argument("--reject_log_period_sec", type=float, default=1.0)

    parser.add_argument("--state_dim", type=int, default=21)
    parser.add_argument("--action_dim", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--rgb_history_frames", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--camera_name", type=str, default="rgb", choices=["rgb"])

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--temporal_agg", action="store_true")

    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)

    parser.add_argument("--start_immediately", action="store_true")
    parser.add_argument(
        "--use_bce_last_action_dim",
        action="store_true",
        help="Deprecated for interception rollout; must remain disabled.",
    )
    parser.add_argument("--no_use_bce_last_action_dim", action="store_false", dest="use_bce_last_action_dim")
    parser.set_defaults(use_bce_last_action_dim=False)

    args = parser.parse_args()

    if int(args.state_dim) != 21:
        raise ValueError(f"Interception rollout requires --state_dim 21, got {args.state_dim}")
    if int(args.action_dim) != 1:
        raise ValueError(f"Interception rollout requires --action_dim 1, got {args.action_dim}")
    if int(args.chunk_size) != 30:
        raise ValueError(f"Interception rollout requires --chunk_size 30, got {args.chunk_size}")
    if int(args.rgb_history_frames) != 3:
        raise ValueError(
            f"Interception rollout requires --rgb_history_frames 3, got {args.rgb_history_frames}"
        )
    if int(args.image_size) != 320:
        raise ValueError(f"Interception rollout requires --image_size 320, got {args.image_size}")
    if args.use_bce_last_action_dim:
        raise ValueError("Interception rollout forbids --use_bce_last_action_dim")

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
