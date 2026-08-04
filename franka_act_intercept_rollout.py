#!/usr/bin/env python3
import os
import time
import argparse
import pickle
import threading
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from intercept_rollout_contract import (
    TemporalAbsoluteAggregator,
    absolute_s_from_anchor,
    build_visual_history_tensor,
    decode_uint8_hwc_image_message,
    denormalize_delta_chunk,
    extract_arm_qpos,
    resolve_intercept_visual_contract,
    resolve_temporal_agg_mode,
    select_sync_observation,
    validate_anchor_freshness,
    validate_intercept_stats_and_config,
    validate_xyt_orientation_parity,
)


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
        self.temporal_agg_reset_service = args.temporal_agg_reset_service
        self.publish_current_scalar = bool(args.publish_current_scalar)

        self.fps = args.fps
        if self.fps <= 0.0:
            raise ValueError(f"fps must be > 0, got {self.fps}")
        self.temporal_agg_mode = str(args.temporal_agg_mode)
        self.recent_agg_window = int(args.recent_agg_window)
        self.recent_agg_half_life = float(args.recent_agg_half_life)
        self.action_lookahead_steps = int(args.action_lookahead_steps)
        self.selected_target_offset_frames = int(self.action_lookahead_steps + 1)
        self.selected_target_offset_ms = 1000.0 * float(self.selected_target_offset_frames) / float(self.fps)
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.chunk_size = args.chunk_size
        self.image_size = int(args.image_size)
        self.max_source_buffer = int(args.max_source_buffer)
        self.max_observation_age_sec = float(args.max_observation_age_sec)
        self.max_anchor_age_sec = float(args.max_anchor_age_sec)
        self.diag_log_period_sec = float(args.diag_log_period_sec)
        self.reject_log_period_sec = float(args.reject_log_period_sec)

        self.visual_buffer: Deque[Tuple[float, Image]] = deque(maxlen=self.max_source_buffer)
        self.joint_buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=self.max_source_buffer)
        self.tcp_buffer: Deque[Tuple[float, float]] = deque(maxlen=self.max_source_buffer)

        self.step_index = 0
        self.accepted_prediction_count = 0
        self.post_inference_stale_count = 0
        self.duplicate_timer_tick_skip_count = 0
        self.accepted_distinct_anchor_count = 0
        self.aggregation_reset_gap_count = 0
        self.backward_anchor_count = 0
        self.failed_or_stale_inference_gap_count = 0
        self.running = args.start_immediately
        self._last_diag_log_sec = 0.0
        self._last_reject_log_sec = 0.0
        self._last_reject_reason = ""
        self._last_attempted_anchor_ns: Optional[int] = None
        self._last_anchor_discontinuity_reset_reason = ""
        self._last_throttled_log_sec: Dict[str, float] = {}
        self.rollout_epoch = 0
        self.reset_anchor_floor_ns: Optional[int] = None
        self._aggregation_lock = threading.Lock()

        self.expected_step_ns = int(round(1e9 / self.fps))
        self.anchor_gap_tolerance_ns = max(int(round(0.35 * self.expected_step_ns)), 5_000_000)
        self.anchor_gap_reset_threshold_ns = self.expected_step_ns + self.anchor_gap_tolerance_ns

        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
        stats_path = os.path.join(args.ckpt_dir, args.stats_name)
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        self.visual_contract = resolve_intercept_visual_contract(
            stats,
            cli_camera_name=args.camera_name,
            cli_event_representation=args.event_representation,
            cli_visual_history_frames=args.rgb_history_frames,
        )
        self.event_representation = self.visual_contract.representation
        self.input_modality = self.visual_contract.input_modality
        self.image_channels = self.visual_contract.image_channels
        self.visual_history_offsets = self.visual_contract.visual_history_offsets
        self.qpos_history_offsets = self.visual_contract.qpos_history_offsets
        self.image_normalization = self.visual_contract.image_normalization

        self.live_event_rotation_degrees = None
        self.orientation_parity_status = "not_applicable"
        if self.event_representation == "xyt_signed_voxel_v1":
            self.live_event_rotation_degrees = self._query_openmv_rotation_degrees()
            self.orientation_parity_status = validate_xyt_orientation_parity(
                self.live_event_rotation_degrees,
                parity_verified=bool(args.xyt_orientation_parity_verified),
            )

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
            "camera_names": [self.input_modality],
            "input_modality": self.input_modality,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "use_bce_last_action_dim": args.use_bce_last_action_dim,
            "rgb_history_frames": len(self.visual_history_offsets),
            "visual_history_frames": len(self.visual_history_offsets),
            "visual_history_offsets": list(self.visual_history_offsets),
            "channels_per_visual_frame": self.visual_contract.channels_per_visual_frame,
            "visual_frame_order": "oldest_to_newest",
            "image_normalization": self.image_normalization,
            "image_channels": self.image_channels,
            "image_size": self.image_size,
            "event_representation": (
                self.event_representation if self.input_modality == "event" else None
            ),
        }

        stats_arrays = validate_intercept_stats_and_config(
            stats=stats,
            policy_config=policy_config,
            expected_chunk_size=self.chunk_size,
        )

        self.qpos_mean = stats_arrays["qpos_mean"]
        self.qpos_std = stats_arrays["qpos_std"]
        self.action_mean = stats_arrays["action_mean"]
        self.action_std = stats_arrays["action_std"]

        from policy import ACTPolicy

        self.policy = ACTPolicy(policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.get_logger().info(f"Checkpoint load status: {loading_status}")
        self.policy.to(self.device)
        self.policy.eval()
        self._warm_up_model_once()
        self.get_logger().info(f"Loaded checkpoint: {ckpt_path}")

        self.pre_process = lambda s: (s - self.qpos_mean) / self.qpos_std
        self.aggregator = TemporalAbsoluteAggregator(
            chunk_size=self.chunk_size,
            mode=self.temporal_agg_mode,
            recent_window=self.recent_agg_window,
            recent_half_life=self.recent_agg_half_life,
            lookahead_steps=self.action_lookahead_steps,
        )

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
        current_tcp_s_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Float64,
            self.current_tcp_s_topic,
            self.current_tcp_s_cb,
            current_tcp_s_qos,
        )
        self.reset_temporal_agg_srv = self.create_service(
            Trigger,
            self.temporal_agg_reset_service,
            self._handle_reset_temporal_aggregation,
        )

        self.timer = self.create_timer(1.0 / self.fps, self.timer_cb)

        self.get_logger().info(f"image_topic={self.image_topic}")
        self.get_logger().info(f"joint_topic={self.joint_topic}")
        self.get_logger().info(f"current_tcp_s_topic={self.current_tcp_s_topic}")
        self.get_logger().info(f"temporal_agg_reset_service={self.temporal_agg_reset_service}")
        self.get_logger().info(f"temporal_agg_mode={self.temporal_agg_mode}")
        self.get_logger().info(f"recent_agg_window={self.recent_agg_window}")
        self.get_logger().info(f"recent_agg_half_life={self.recent_agg_half_life}")
        self.get_logger().info(f"action_lookahead_steps={self.action_lookahead_steps}")
        self.get_logger().info(f"selected_target_offset_frames={self.selected_target_offset_frames}")
        self.get_logger().info(f"selected_target_offset_ms={self.selected_target_offset_ms:.2f}")
        self.get_logger().info(f"prediction_chunk_topic={self.prediction_topic}")
        self.get_logger().info("prediction_chunk_msg_type=std_msgs/msg/Float64MultiArray")
        self.get_logger().info(f"prediction_current_topic={self.prediction_current_topic}")
        self.get_logger().info("prediction_current_msg_type=std_msgs/msg/Float64")
        self.get_logger().info(f"publish_current_scalar={self.publish_current_scalar}")
        self.get_logger().info(f"checkpoint_path={ckpt_path}")
        self.get_logger().info(f"stats_path={stats_path}")
        self.get_logger().info(
            "Visual contract: "
            f"representation={self.event_representation} "
            f"image_topic={self.image_topic} "
            f"expected_encoding={self.visual_contract.expected_encoding} "
            f"expected_dimensions=({self.visual_contract.expected_height},"
            f"{self.visual_contract.expected_width}) "
            f"visual_offsets={list(self.visual_history_offsets)} "
            f"qpos_offsets={list(self.qpos_history_offsets)} "
            f"image_channels={self.image_channels} "
            f"normalization={self.image_normalization}"
        )
        if self.event_representation == "xyt_signed_voxel_v1":
            self.get_logger().info(
                "XYT orientation parity: "
                "offline_rotation_degrees=0 "
                f"live_rotation_degrees={self.live_event_rotation_degrees} "
                f"status={self.orientation_parity_status}"
            )
            self.get_logger().warn(
                "Do not call OpenMV rotation services during XYT rollout: those services "
                "change effective output rotation without updating the ROS parameter."
            )
        self.get_logger().info(
            "This rollout node publishes interception predictions only; it does not publish robot motion commands."
        )
        self.get_logger().info(
            "Interception contract: "
            f"state_dim={self.state_dim} action_dim={self.action_dim} "
            f"chunk_size={self.chunk_size} "
            f"visual_history_frames={len(self.visual_history_offsets)} "
            f"image_channels={self.image_channels} "
            f"use_bce_last_action_dim={args.use_bce_last_action_dim}"
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
        self._append_monotonic("visual", self.visual_buffer, t, msg)

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
        return bool(self.visual_buffer) and bool(self.joint_buffer) and bool(self.tcp_buffer)

    def _reject(self, reason: str) -> None:
        now = time.time()
        if (
            reason != self._last_reject_reason
            or (now - self._last_reject_log_sec) >= self.reject_log_period_sec
        ):
            self.get_logger().warn(f"Inference rejected: {reason}")
            self._last_reject_log_sec = now
            self._last_reject_reason = reason

    def _throttled_warn(self, key: str, message: str) -> None:
        now = time.time()
        last = self._last_throttled_log_sec.get(key)
        if last is not None and (now - last) < self.reject_log_period_sec:
            return
        self._last_throttled_log_sec[key] = now
        self.get_logger().warn(message)

    def _run_policy_forward(self, qpos: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            raw_output = self.policy(qpos, image)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.time() - start) * 1000.0
        return raw_output, inference_ms

    def _query_openmv_rotation_degrees(self) -> Optional[int]:
        client = self.create_client(GetParameters, "/openmv_cam/get_parameters")
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                "OpenMV parameter service is unavailable; cannot verify XYT orientation"
            )
            return None
        request = GetParameters.Request()
        request.names = ["event_frame_rotation_degrees"]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            self.get_logger().error(
                "Timed out querying /openmv_cam event_frame_rotation_degrees"
            )
            return None
        values = future.result().values
        if len(values) != 1:
            self.get_logger().error("OpenMV rotation query returned no parameter value")
            return None
        if values[0].type != ParameterType.PARAMETER_INTEGER:
            self.get_logger().error(
                "OpenMV event_frame_rotation_degrees is unavailable or not an integer"
            )
            return None
        return int(values[0].integer_value)

    def _warm_up_model_once(self) -> None:
        warmup_qpos = torch.zeros(
            (1, self.state_dim),
            dtype=torch.float32,
            device=self.device,
        )
        warmup_image = torch.zeros(
            (
                1,                    # batch
                1,                    # one logical camera
                self.image_channels,
                self.image_size,
                self.image_size,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        _warmup_output, _warmup_ms = self._run_policy_forward(
            warmup_qpos,
            warmup_image,
        )
        self.get_logger().info("Model warm-up inference complete.")

    def _anchor_timestamp_ns_from_observation(
            self,
            visual_msg: Image,
            fallback_sec: float,
        ) -> int:
            try:
                stamp = visual_msg.header.stamp
                return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            except Exception:
                return int(round(float(fallback_sec) * 1e9))

    def _reset_temporal_aggregation(self, reason: str) -> int:
        discarded_entries = len(getattr(self.aggregator, "_history", []))
        self.aggregator.reset()
        self.step_index = 0
        self._last_anchor_discontinuity_reset_reason = reason
        return discarded_entries

    def _handle_reset_temporal_aggregation(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request
        with self._aggregation_lock:
            floor_anchor_ns: Optional[int] = None
            if self.visual_buffer:
                floor_sec, floor_msg = self.visual_buffer[-1]
                floor_anchor_ns = self._anchor_timestamp_ns_from_observation(floor_msg, floor_sec)

            discarded_entries = self._reset_temporal_aggregation("service_reset")
            self.rollout_epoch += 1
            self.reset_anchor_floor_ns = floor_anchor_ns
            if floor_anchor_ns is not None:
                self._last_attempted_anchor_ns = floor_anchor_ns

            response.success = True
            response.message = f"rollout_epoch={self.rollout_epoch}"

        floor_text = "none" if floor_anchor_ns is None else str(floor_anchor_ns)
        self.get_logger().info(
            "Temporal aggregation reset acknowledged: "
            f"epoch={self.rollout_epoch} reason=service_reset "
            f"visual_anchor_floor_ns={floor_text} discarded_entries={discarded_entries}"
        )
        return response

    def _warn_post_inference_stale(
        self,
        visual_age_sec: float,
        tcp_age_sec: float,
        inference_ms: float,
    ) -> None:
        self._throttled_warn(
            "post_inference_stale",
            "Rejecting stale post-inference rollout result: "
            f"visual_age={visual_age_sec:.4f}s (limit={self.max_observation_age_sec:.4f}s), "
            f"tcp_age={tcp_age_sec:.4f}s (limit={self.max_anchor_age_sec:.4f}s), "
            f"inference_ms={inference_ms:.2f}"
        )

    def build_policy_inputs(self):
        visual_timestamps = [item[0] for item in self.visual_buffer]
        visual_messages = [item[1] for item in self.visual_buffer]
        joint_timestamps = [item[0] for item in self.joint_buffer]
        joint_qpos = [item[1] for item in self.joint_buffer]
        tcp_timestamps = [item[0] for item in self.tcp_buffer]
        tcp_values = [item[1] for item in self.tcp_buffer]

        sync = select_sync_observation(
            rgb_timestamps=visual_timestamps,
            joint_timestamps=joint_timestamps,
            joint_qpos_samples=joint_qpos,
            tcp_timestamps=tcp_timestamps,
            tcp_values=tcp_values,
            history_offsets=self.visual_history_offsets,
            qpos_history_offsets=self.qpos_history_offsets,
            frame_period_sec=1.0 / self.fps,
            qpos_relative_to_anchor=(
                self.event_representation == "xyt_signed_voxel_v1"
            ),
            max_qpos_age_sec=(
                self.max_anchor_age_sec
                if self.event_representation == "xyt_signed_voxel_v1"
                else None
            ),
        )

        selected_visual_msgs = [visual_messages[index] for index in sync.history_indices]
        if self.event_representation == "xyt_signed_voxel_v1":
            visual_frames = [
                decode_uint8_hwc_image_message(
                    selected_visual_msgs[0],
                    expected_encoding="8UC9",
                    expected_height=self.visual_contract.expected_height,
                    expected_width=self.visual_contract.expected_width,
                    expected_channels=self.image_channels,
                )
            ]
        else:
            desired_encoding = "passthrough" if self.input_modality == "event" else "rgb8"
            visual_frames = [
                self.bridge.imgmsg_to_cv2(msg, desired_encoding=desired_encoding)
                for msg in selected_visual_msgs
            ]
        if self.input_modality == "event" and self.event_representation != "xyt_signed_voxel_v1":
            for frame in visual_frames:
                if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(
                        "Event image decoding contract violated: expected uint8 HxWx3 from /openmv_cam/event_frame_3ch"
                    )
        image_np = build_visual_history_tensor(
            visual_frames,
            self.image_size,
            modality=self.input_modality,
            expected_channels=self.image_channels,
        )

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        anchor_observation_timestamp = sync.visual_timestamps[-1]
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
        anchor_visual_msg = selected_visual_msgs[-1]
        anchor_timestamp_ns = self._anchor_timestamp_ns_from_observation(
            anchor_visual_msg,
            anchor_observation_timestamp,
        )
        return sync, qpos, image, anchor_timestamp_ns

    def publish_predictions(self, absolute_chunk: np.ndarray, current_value: Optional[float]) -> None:
        chunk_msg = Float64MultiArray()
        chunk_msg.data = [float(value) for value in absolute_chunk.tolist()]
        self.prediction_pub.publish(chunk_msg)

        if self.prediction_current_pub is not None and current_value is not None:
            current_msg = Float64()
            current_msg.data = float(current_value)
            self.prediction_current_pub.publish(current_msg)

    def run_policy_step(self) -> None:
        step_start_wall = time.time()
        sync, qpos, curr_image, anchor_timestamp_ns = self.build_policy_inputs()

        with self._aggregation_lock:
            if self.reset_anchor_floor_ns is not None and anchor_timestamp_ns <= self.reset_anchor_floor_ns:
                self.duplicate_timer_tick_skip_count += 1
                self._last_attempted_anchor_ns = anchor_timestamp_ns
                return

            if self._last_attempted_anchor_ns is not None:
                delta_ns = anchor_timestamp_ns - self._last_attempted_anchor_ns
                if delta_ns == 0:
                    self.duplicate_timer_tick_skip_count += 1
                    return
                if delta_ns < 0:
                    self.backward_anchor_count += 1
                    self._reset_temporal_aggregation("backward_visual_anchor_timestamp")
                    self._throttled_warn(
                        "backward_anchor",
                        "Detected backward visual anchor timestamp; resetting temporal aggregation. "
                        f"anchor_ns={anchor_timestamp_ns} delta_ns={delta_ns}",
                    )
                elif delta_ns > self.anchor_gap_reset_threshold_ns:
                    self.aggregation_reset_gap_count += 1
                    self._reset_temporal_aggregation("visual_anchor_gap")
                    self._throttled_warn(
                        "anchor_gap",
                        "Detected visual anchor timestamp gap; resetting temporal aggregation. "
                        f"delta_ms={delta_ns / 1e6:.2f} expected_ms={self.expected_step_ns / 1e6:.2f} "
                        f"tolerance_ms={self.anchor_gap_tolerance_ns / 1e6:.2f}",
                    )

            # Mark this anchor as attempted before running inference so duplicate timer ticks
            # cannot trigger repeated attempts on the same cached frame.
            self._last_attempted_anchor_ns = anchor_timestamp_ns

        try:
            raw_output, inference_ms = self._run_policy_forward(qpos, curr_image)
        except Exception:
            with self._aggregation_lock:
                self.failed_or_stale_inference_gap_count += 1
                self._reset_temporal_aggregation("inference_forward_failed")
            raise

        expected_shape = (1, self.chunk_size, 1)
        if tuple(raw_output.shape) != expected_shape:
            with self._aggregation_lock:
                self.failed_or_stale_inference_gap_count += 1
                self._reset_temporal_aggregation("inference_output_shape_failed")
            raise RuntimeError(
                f"Policy output shape mismatch: expected {expected_shape}, got {tuple(raw_output.shape)}"
            )
        if not torch.isfinite(raw_output).all():
            with self._aggregation_lock:
                self.failed_or_stale_inference_gap_count += 1
                self._reset_temporal_aggregation("inference_output_non_finite")
            raise RuntimeError("Policy output contains non-finite values")

        post_now_sec = self.get_clock().now().nanoseconds * 1e-9
        try:
            validate_anchor_freshness(
                anchor_timestamp=sync.anchor_tcp_s_timestamp,
                observation_timestamp=sync.visual_timestamps[-1],
                now_timestamp=post_now_sec,
                max_anchor_age_sec=self.max_anchor_age_sec,
                max_observation_age_sec=self.max_observation_age_sec,
            )
        except ValueError:
            with self._aggregation_lock:
                self.post_inference_stale_count += 1
                self.failed_or_stale_inference_gap_count += 1
            visual_age_sec = post_now_sec - sync.visual_timestamps[-1]
            tcp_age_sec = sync.visual_timestamps[-1] - sync.anchor_tcp_s_timestamp
            self._warn_post_inference_stale(
                visual_age_sec=visual_age_sec,
                tcp_age_sec=tcp_age_sec,
                inference_ms=inference_ms,
            )
            with self._aggregation_lock:
                self._reset_temporal_aggregation("post_inference_stale")
            return

        try:
            normalized_delta = raw_output[0, :, 0].detach().cpu().numpy()
            delta_s = denormalize_delta_chunk(normalized_delta, self.action_mean, self.action_std)
            absolute_s = absolute_s_from_anchor(sync.anchor_tcp_s, delta_s)
            if not np.all(np.isfinite(absolute_s)):
                raise RuntimeError("Absolute-s chunk contains non-finite values")

            with self._aggregation_lock:
                self.aggregator.add_prediction(self.step_index, absolute_s)
                selection = self.aggregator.selection_for_step(self.step_index)
            if selection is None:
                raise RuntimeError("Temporal aggregator produced no current value")
            if not np.isfinite(selection.value):
                raise ValueError("Temporal aggregator produced non-finite current absolute-s")
            if not np.isfinite(selection.effective_age_frames):
                raise ValueError("Temporal aggregator produced non-finite effective age")
            current_value = float(selection.value)
            agg_contributors = int(selection.contributor_count)
            agg_effective_age_frames = float(selection.effective_age_frames)
        except Exception:
            with self._aggregation_lock:
                self.failed_or_stale_inference_gap_count += 1
                self._reset_temporal_aggregation("prediction_validation_failed")
            raise

        with self._aggregation_lock:
            if self.reset_anchor_floor_ns is not None and anchor_timestamp_ns > self.reset_anchor_floor_ns:
                self.reset_anchor_floor_ns = None

        self.publish_predictions(absolute_s, current_value)
        with self._aggregation_lock:
            self.accepted_prediction_count += 1
            self.accepted_distinct_anchor_count += 1

        total_step_ms = (time.time() - step_start_wall) * 1000.0
        max_input_age = post_now_sec - sync.visual_timestamps[-1]
        max_sync_error = max(
            max(
                abs(target_ts - qpos_ts)
                for target_ts, qpos_ts in zip(
                    sync.qpos_target_timestamps,
                    sync.qpos_timestamps,
                )
            ),
            sync.visual_timestamps[-1] - sync.anchor_tcp_s_timestamp,
        )

        now_wall = time.time()
        if (now_wall - self._last_diag_log_sec) >= self.diag_log_period_sec:
            agg_effective_age_ms = 1000.0 * agg_effective_age_frames / self.fps
            self.get_logger().info(
                "Inference diagnostics: "
                f"step={self.step_index} "
                f"accepted={self.accepted_prediction_count} stale_post_inference={self.post_inference_stale_count} "
                f"duplicate_tick_skips={self.duplicate_timer_tick_skip_count} "
                f"accepted_anchors={self.accepted_distinct_anchor_count} "
                f"agg_resets_gap={self.aggregation_reset_gap_count} "
                f"backward_timestamps={self.backward_anchor_count} "
                f"failed_or_stale_gaps={self.failed_or_stale_inference_gap_count} "
                f"rollout_epoch={self.rollout_epoch} "
                f"agg_mode={self.temporal_agg_mode} "
                f"agg_contributors={agg_contributors} "
                f"agg_effective_age_frames={agg_effective_age_frames:.4f} "
                f"agg_effective_age_ms={agg_effective_age_ms:.2f} "
                f"action_lookahead_steps={self.action_lookahead_steps} "
                f"selected_target_offset_frames={self.selected_target_offset_frames} "
                f"selected_target_offset_ms={self.selected_target_offset_ms:.2f} "
                f"history_indices={list(sync.history_indices)} "
                f"visual_ts={[round(ts, 6) for ts in sync.visual_timestamps]} "
                f"qpos_target_ts={[round(ts, 6) for ts in sync.qpos_target_timestamps]} "
                f"qpos_ts={[round(ts, 6) for ts in sync.qpos_timestamps]} "
                f"anchor_s={sync.anchor_tcp_s:+.6f} anchor_ts={sync.anchor_tcp_s_timestamp:.6f} "
                f"qpos_shape={tuple(sync.qpos_history.shape)} image_shape={tuple(curr_image.shape)} "
                f"raw_shape={tuple(raw_output.shape)} delta_shape={tuple(delta_s.shape)} abs_shape={tuple(absolute_s.shape)} "
                f"delta_s={_first_last(delta_s)} abs_s={_first_last(absolute_s)} "
                f"current_abs={current_value:+.6f} max_input_age={max_input_age:.4f}s "
                f"max_sync_error={max_sync_error:.4f}s inference_ms={inference_ms:.2f} step_ms={total_step_ms:.2f}"
            )
            self._last_diag_log_sec = now_wall

        with self._aggregation_lock:
            self.step_index += 1

    def timer_cb(self) -> None:
        if not self.ready():
            self._reject("waiting for visual image, JointState, and current_tcp_s streams")
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
    parser.add_argument("--prediction_topic", type=str, default="/act/intercept_prediction_chunk_abs_s")
    parser.add_argument(
        "--prediction_current_topic",
        type=str,
        default="/act/intercept_prediction_current_abs_s",
    )
    parser.add_argument(
        "--temporal_agg_reset_service",
        type=str,
        default="/act/reset_temporal_aggregation",
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
    parser.add_argument("--rgb_history_frames", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument(
        "--camera_name",
        type=str,
        default=None,
        choices=["rgb", "event"],
    )
    parser.add_argument(
        "--event_representation",
        type=str,
        default=None,
        choices=[
            "rgb_history",
            "shifted_3chef_signed",
            "xyt_signed_voxel_v1",
        ],
    )
    parser.add_argument(
        "--xyt_orientation_parity_verified",
        action="store_true",
        help=(
            "Allow nonzero live OpenMV rotation only after a synthetic or recorded "
            "comparison proves parity with unrotated offline XYT tensors."
        ),
    )

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument(
        "--temporal-agg-mode",
        choices=["full", "latest", "recent"],
        default=None,
    )
    parser.add_argument(
        "--recent-agg-window",
        type=int,
        choices=[3, 4, 5],
        default=5,
    )
    parser.add_argument(
        "--recent-agg-half-life",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--action-lookahead-steps",
        type=int,
        default=0,
    )
    legacy_temporal_agg_group = parser.add_mutually_exclusive_group()
    legacy_temporal_agg_group.add_argument(
        "--temporal_agg",
        "--temporal-agg",
        dest="temporal_agg_legacy",
        action="store_const",
        const=True,
    )
    legacy_temporal_agg_group.add_argument(
        "--no-temporal-agg",
        dest="temporal_agg_legacy",
        action="store_const",
        const=False,
    )
    parser.set_defaults(temporal_agg_legacy=None)

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
    try:
        args.temporal_agg_mode = resolve_temporal_agg_mode(
            temporal_agg_mode=args.temporal_agg_mode,
            temporal_agg_legacy=args.temporal_agg_legacy,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if int(args.state_dim) != 21:
        raise ValueError(f"Interception rollout requires --state_dim 21, got {args.state_dim}")
    if int(args.action_dim) != 1:
        raise ValueError(f"Interception rollout requires --action_dim 1, got {args.action_dim}")
    if int(args.chunk_size) != 30:
        raise ValueError(f"Interception rollout requires --chunk_size 30, got {args.chunk_size}")
    if args.rgb_history_frames is not None and int(args.rgb_history_frames) <= 0:
        raise ValueError(
            "--rgb_history_frames must be positive when explicitly supplied, "
            f"got {args.rgb_history_frames}"
        )
    if int(args.image_size) != 320:
        raise ValueError(f"Interception rollout requires --image_size 320, got {args.image_size}")
    if args.use_bce_last_action_dim:
        raise ValueError("Interception rollout forbids --use_bce_last_action_dim")
    if int(args.action_lookahead_steps) < 0 or int(args.action_lookahead_steps) >= int(args.chunk_size):
        parser.error(
            "--action-lookahead-steps must satisfy 0 <= value < chunk_size; "
            f"got value={args.action_lookahead_steps}, chunk_size={args.chunk_size}"
        )
    if (
        str(args.temporal_agg_mode) == "recent"
        and int(args.recent_agg_window) > int(args.chunk_size) - int(args.action_lookahead_steps)
    ):
        parser.error(
            "For --temporal-agg-mode recent, --recent-agg-window must be <= chunk_size - "
            "action_lookahead_steps; "
            f"got recent_agg_window={args.recent_agg_window}, "
            f"chunk_size={args.chunk_size}, action_lookahead_steps={args.action_lookahead_steps}"
        )

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
