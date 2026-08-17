#!/usr/bin/env python3
import os
import time
import argparse
import pickle
import threading
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from intercept_rollout_contract import (
    TemporalAbsoluteAggregator,
    add_event_spatial_preprocessing_arguments,
    absolute_s_from_anchor,
    build_visual_history_tensor,
    denormalize_delta_chunk,
    extract_arm_qpos,
    preprocess_event_history_frames,
    resolve_event_spatial_preprocessing,
    resolve_temporal_agg_mode,
    select_sync_observation,
    validate_anchor_freshness,
    validate_intercept_stats_and_config,
)
from policy import ACTPolicy
from rollout_latency_trace import RolloutLatencyTracer, resolve_latency_trace_cuda_sync
from sparse_ball import (
    SparsePoint, construct_causal_sparse_history,
    SPARSE_HISTORY_OFFSETS_SEC, default_sparse_topic,
    resolve_sparse_checkpoint_contract, sparse_history_offsets_frames, validate_policy_rate,
    validate_sparse_checkpoint_contract,
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
        self.latency_trace_cuda_sync = bool(args.latency_trace_cuda_sync)

        self.policy_rate_hz = validate_policy_rate(args.policy_rate_hz)
        self.fps = float(args.fps)
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
        self.input_modality = str(args.input_modality)
        self.qpos_history_offsets = (
            sparse_history_offsets_frames(self.policy_rate_hz)
            if self.input_modality == "sparse_ball" else (-6, -3, 0)
        )
        self.sparse_source = args.sparse_source
        self.sparse_topic = args.sparse_topic or default_sparse_topic(self.sparse_source)
        self.event_spatial_preprocessing = args.event_spatial_preprocessing
        self._event_spatial_shape_trace_logged = False
        self.image_channels = 0 if self.input_modality == "sparse_ball" else 9
        self.sparse_image_width = 1280 if self.sparse_source == "rgb" else 320
        self.sparse_image_height = 720 if self.sparse_source == "rgb" else 320
        self.image_size = int(args.image_size)
        self.max_source_buffer = int(args.max_source_buffer)
        self.max_observation_age_sec = float(args.max_observation_age_sec)
        self.max_anchor_age_sec = float(args.max_anchor_age_sec)
        self.diag_log_period_sec = float(args.diag_log_period_sec)
        self.reject_log_period_sec = float(args.reject_log_period_sec)

        self.visual_buffer: Deque[Tuple[float, Any]] = deque(maxlen=self.max_source_buffer)
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
            "camera_names": (["sparse_ball"] if self.input_modality == "sparse_ball"
                             else [args.camera_name]),
            "input_modality": self.input_modality,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "use_bce_last_action_dim": args.use_bce_last_action_dim,
            "rgb_history_frames": args.rgb_history_frames,
            "visual_history_frames": args.rgb_history_frames,
            "visual_history_offsets": list(
                self.qpos_history_offsets
                if self.input_modality == "sparse_ball" else (-6, -3, 0)
            ),
            "qpos_history_offsets": list(self.qpos_history_offsets),
            "channels_per_visual_frame": 0 if self.input_modality == "sparse_ball" else 3,
            "visual_frame_order": "oldest_to_newest",
            "image_normalization": (
                "sparse_train_split_standardization"
                if self.input_modality == "sparse_ball"
                else ("shifted_3chef_centered" if self.input_modality == "event" else "imagenet")
            ),
            "image_channels": self.image_channels,
            "image_size": self.image_size,
            "sparse_source": self.sparse_source,
            "sparse_feature_dim": 4,
            "sparse_history_length": 3,
            "max_observation_age_sec": self.max_observation_age_sec,
            "policy_rate_hz": self.policy_rate_hz,
        }

        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
        stats_path = os.path.join(args.ckpt_dir, args.stats_name)

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        if self.input_modality == "sparse_ball":
            self.sparse_image_width = int(stats.get("sparse_image_width", self.sparse_image_width))
            self.sparse_image_height = int(stats.get("sparse_image_height", self.sparse_image_height))
            validate_sparse_checkpoint_contract(
                stats, self.sparse_source, self.sparse_image_width,
                self.sparse_image_height, self.max_observation_age_sec,
            )
            sparse_runtime_contract = resolve_sparse_checkpoint_contract(
                stats, self.policy_rate_hz, self.chunk_size, self.sparse_source,
            )
            self.qpos_history_offsets = tuple(
                sparse_runtime_contract["qpos_history_offsets"]
            )
            inferred = sparse_runtime_contract["inferred_legacy_fields"]
            if inferred:
                self.get_logger().warn(
                    "Legacy 30 Hz sparse checkpoint metadata inferred: "
                    f"{inferred}"
                )
            stats_arrays = {key: np.asarray(stats[key]) for key in (
                "qpos_mean", "qpos_std", "action_mean", "action_std")}
            policy_config["sparse_mean"] = stats["sparse_mean"]
            policy_config["sparse_std"] = stats["sparse_std"]
        else:
            stats_arrays = validate_intercept_stats_and_config(
                stats=stats, policy_config=policy_config,
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

        self.latency_tracer = RolloutLatencyTracer(
            self,
            enabled=bool(args.enable_latency_trace),
            topic=str(args.latency_trace_topic),
            run_id=str(args.latency_run_id),
            modality=self.input_modality,
        )
        self._active_latency_trace = None

        if self.input_modality == "sparse_ball":
            self.create_subscription(
                PointStamped,
                self.sparse_topic,
                self.sparse_cb,
                10,
            )
        else:
            self.create_subscription(
                Image,
                self.image_topic,
                self.image_cb,
                10,
            )
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

        if self.input_modality == "sparse_ball":
            self.get_logger().info(
                f"sparse_source={self.sparse_source} sparse_topic={self.sparse_topic}"
            )
        else:
            self.get_logger().info(f"image_topic={self.image_topic}")
        if self.event_spatial_preprocessing is None:
            self.get_logger().info("event spatial preprocessing: disabled")
        else:
            event_config = self.event_spatial_preprocessing
            self.get_logger().info(
                "event spatial preprocessing: enabled "
                f"mask_x={event_config.mask_x} crop_square={event_config.crop_square} "
                f"fill_value={event_config.fill_value}"
            )
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
        self._append_monotonic("visual", self.visual_buffer, t, msg)

    def sparse_cb(self, msg: PointStamped) -> None:
        """Buffer sparse detections exclusively by PointStamped header time."""
        t = stamp_to_sec(msg.header.stamp)
        if not np.isfinite(t):
            self.get_logger().warn("Dropping sparse point with non-finite header stamp")
            return
        self._append_monotonic("sparse PointStamped", self.visual_buffer, t, msg)

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

    def _trace_mark(self, stage: str, **detail) -> None:
        tracer = getattr(self, "latency_tracer", None)
        if tracer is not None:
            tracer.mark(getattr(self, "_active_latency_trace", None), stage, **detail)

    def _run_policy_forward(self, qpos: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, float]:
        with torch.inference_mode():
            normalized_image = self.policy.preprocess_image(image)
        self._trace_mark(
            "tensor_construction_and_preprocessing",
            qpos_shape=list(qpos.shape),
            image_shape=list(image.shape),
        )
        cuda_timing = self.device.type == "cuda" and self.latency_trace_cuda_sync
        if cuda_timing:
            sync_start = time.perf_counter()
            torch.cuda.synchronize()
            self._trace_mark(
                "cuda_sync_before_forward",
                cuda_sync_before_ms=(time.perf_counter() - sync_start) * 1000.0,
            )
        start = time.perf_counter()
        with torch.inference_mode():
            raw_output = self.policy.forward_inference(qpos, normalized_image)
        if cuda_timing:
            sync_start = time.perf_counter()
            torch.cuda.synchronize()
            self._trace_mark(
                "cuda_sync_after_forward",
                cuda_sync_after_ms=(time.perf_counter() - sync_start) * 1000.0,
            )
        inference_ms = (time.perf_counter() - start) * 1000.0
        self._trace_mark(
            "model_forward_pass",
            model_forward_ms=inference_ms,
            cuda_synchronized=bool(cuda_timing),
        )
        return raw_output, inference_ms

    def _warm_up_model_once(self) -> None:
        warmup_qpos = torch.zeros(
            (1, self.state_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if self.input_modality == "sparse_ball":
            warmup_image = torch.zeros((1, 3, 4), dtype=torch.float32, device=self.device)
        else:
            warmup_image = torch.zeros((
                1,                    # batch
                1,                    # one logical camera
                self.image_channels,  # 9 for 3 RGB history frames
                self.image_size,
                self.image_size,
            ), dtype=torch.float32, device=self.device)

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
            history_offsets=self.qpos_history_offsets,
        )
        self._trace_mark(
            "input_history_selection",
            selected_visual_timestamps_ns=[
                int(round(ts * 1e9)) for ts in sync.visual_timestamps
            ],
            selected_visual_anchor_timestamp_ns=int(
                round(sync.visual_timestamps[-1] * 1e9)
            ),
            selected_qpos_timestamps_ns=[int(round(ts * 1e9)) for ts in sync.qpos_timestamps],
        )

        selected_visual_msgs = [visual_messages[index] for index in sync.history_indices]
        if self.input_modality == "sparse_ball":
            sparse_points = [
                SparsePoint(stamp_to_sec(msg.header.stamp), msg.point.x, msg.point.y)
                for msg in visual_messages
            ]
            image_np = construct_causal_sparse_history(
                sparse_points, sync.visual_timestamps[-1], SPARSE_HISTORY_OFFSETS_SEC,
                self.sparse_image_width, self.sparse_image_height,
                self.max_observation_age_sec,
            )[None, ...]
            visual_frames = []
        else:
            desired_encoding = "passthrough" if self.input_modality == "event" else "rgb8"
            visual_frames = [
                self.bridge.imgmsg_to_cv2(msg, desired_encoding=desired_encoding)
                for msg in selected_visual_msgs
            ]
        if self.input_modality == "event":
            for frame in visual_frames:
                if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(
                        "Event image decoding contract violated: expected uint8 HxWx3 from /openmv_cam/event_frame_3ch"
                    )
        raw_event_shape = visual_frames[0].shape if visual_frames and self.event_spatial_preprocessing is not None else None
        if self.input_modality != "sparse_ball":
            visual_frames = preprocess_event_history_frames(
                visual_frames, self.event_spatial_preprocessing,
            )
            image_np = build_visual_history_tensor(
                visual_frames, self.image_size, modality=self.input_modality
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
        self._trace_mark("complete_observation_frame_acceptance")

        qpos_norm = self.pre_process(sync.qpos_history)
        if not np.all(np.isfinite(qpos_norm)):
            raise ValueError("Normalized qpos contains non-finite values")

        qpos = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)
        image = torch.from_numpy(image_np).float().to(self.device)
        if (
            self.input_modality != "sparse_ball"
            and self.event_spatial_preprocessing is not None
            and not self._event_spatial_shape_trace_logged
        ):
            crop_shape = visual_frames[0].shape
            self.get_logger().info(
                "event spatial preprocessing: "
                f"raw {raw_event_shape[0]}x{raw_event_shape[1]}x{raw_event_shape[2]} -> "
                f"crop {crop_shape[0]}x{crop_shape[1]}x{crop_shape[2]} -> "
                f"policy tensor {image_np.shape[-2]}x{image_np.shape[-1]}x{image_np.shape[2]}"
            )
            self._event_spatial_shape_trace_logged = True
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

    def run_policy_step(self) -> Tuple[bool, str]:
        step_start_wall = time.time()
        sync, qpos, curr_image, anchor_timestamp_ns = self.build_policy_inputs()

        with self._aggregation_lock:
            if self.reset_anchor_floor_ns is not None and anchor_timestamp_ns <= self.reset_anchor_floor_ns:
                self.duplicate_timer_tick_skip_count += 1
                self._last_attempted_anchor_ns = anchor_timestamp_ns
                return False, "frame_at_or_before_reset_floor"

            if self._last_attempted_anchor_ns is not None:
                delta_ns = anchor_timestamp_ns - self._last_attempted_anchor_ns
                if delta_ns == 0:
                    self.duplicate_timer_tick_skip_count += 1
                    return False, "duplicate_source_frame"
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
            return False, "post_inference_stale"

        try:
            normalized_delta = raw_output[0, :, 0].detach().cpu().numpy()
            delta_s = denormalize_delta_chunk(normalized_delta, self.action_mean, self.action_std)
            absolute_s = absolute_s_from_anchor(sync.anchor_tcp_s, delta_s)
            self._trace_mark("denormalization_and_absolute_target_conversion")
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
            self._trace_mark(
                "temporal_aggregation_lookahead_selection",
                chunk_size=self.chunk_size,
                lookahead_index=self.action_lookahead_steps,
                selected_target=current_value,
            )
        except Exception:
            with self._aggregation_lock:
                self.failed_or_stale_inference_gap_count += 1
                self._reset_temporal_aggregation("prediction_validation_failed")
            raise

        with self._aggregation_lock:
            if self.reset_anchor_floor_ns is not None and anchor_timestamp_ns > self.reset_anchor_floor_ns:
                self.reset_anchor_floor_ns = None

        self.publish_predictions(absolute_s, current_value)
        self._trace_mark("prediction_publication_completion")
        with self._aggregation_lock:
            self.accepted_prediction_count += 1
            self.accepted_distinct_anchor_count += 1

        total_step_ms = (time.time() - step_start_wall) * 1000.0
        max_input_age = post_now_sec - sync.visual_timestamps[-1]
        max_sync_error = max(
            max(abs(visual_ts - qpos_ts) for visual_ts, qpos_ts in zip(sync.visual_timestamps, sync.qpos_timestamps)),
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
        return True, ""

    def timer_cb(self) -> None:
        source_stamp_ns = 0
        if self.visual_buffer:
            source_stamp_ns = int(self.visual_buffer[-1][0] * 1e9)
        trace = self.latency_tracer.begin(source_stamp_ns)
        self._active_latency_trace = trace
        valid = False
        rejection_reason = ""
        if not self.ready():
            rejection_reason = "waiting for visual input, JointState, and current_tcp_s streams"
            self._reject(rejection_reason)
            self.latency_tracer.finish(trace, valid=False, rejection_reason=rejection_reason)
            self._active_latency_trace = None
            return

        if not self.running:
            self.running = True
            self.get_logger().info("Received first valid synchronized streams. Starting interception rollout.")

        try:
            valid, rejection_reason = self.run_policy_step()
        except ValueError as exc:
            rejection_reason = str(exc)
            self._reject(rejection_reason)
        except RuntimeError as exc:
            rejection_reason = str(exc)
            self._reject(rejection_reason)
        except Exception as exc:
            self.get_logger().error(f"Policy step failed: {exc}")
            rejection_reason = str(exc)
            raise
        finally:
            self.latency_tracer.finish(trace, valid=valid, rejection_reason=rejection_reason)
            self._active_latency_trace = None


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

    parser.add_argument("--policy_rate_hz", type=int, choices=[30, 60], default=30)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_source_buffer", type=int, default=256)
    parser.add_argument("--max_observation_age_sec", type=float, default=None)
    parser.add_argument("--max_anchor_age_sec", type=float, default=0.10)
    parser.add_argument("--diag_log_period_sec", type=float, default=1.0)
    parser.add_argument("--reject_log_period_sec", type=float, default=1.0)
    parser.add_argument("--enable_latency_trace", action="store_true", default=False)
    latency_cuda_sync_group = parser.add_mutually_exclusive_group()
    latency_cuda_sync_group.add_argument(
        "--latency_trace_cuda_sync",
        dest="latency_trace_cuda_sync",
        action="store_true",
        help="Synchronize CUDA around traced model inference (default when latency tracing is enabled).",
    )
    latency_cuda_sync_group.add_argument(
        "--no_latency_trace_cuda_sync",
        dest="latency_trace_cuda_sync",
        action="store_false",
        help="Disable CUDA synchronization while latency tracing is enabled.",
    )
    parser.set_defaults(latency_trace_cuda_sync=None)
    parser.add_argument("--latency_trace_topic", type=str, default="/intercept_trace/act_rollout")
    parser.add_argument("--latency_run_id", type=str, default="")

    parser.add_argument("--state_dim", type=int, default=21)
    parser.add_argument("--action_dim", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--rgb_history_frames", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--camera_name", type=str, default="rgb", choices=["rgb", "event"])
    parser.add_argument("--input_modality", choices=["rgb", "event", "sparse_ball"])
    parser.add_argument("--sparse_source", choices=["rgb", "event"])
    parser.add_argument("--sparse_topic", type=str)
    add_event_spatial_preprocessing_arguments(parser)

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
    args.latency_trace_cuda_sync = resolve_latency_trace_cuda_sync(
        args.enable_latency_trace,
        args.latency_trace_cuda_sync,
    )
    args.policy_rate_hz = validate_policy_rate(args.policy_rate_hz)
    if args.fps is None:
        args.fps = float(args.policy_rate_hz)
    if not np.isclose(float(args.fps), float(args.policy_rate_hz), atol=1e-9):
        parser.error(
            f"--fps {args.fps} disagrees with --policy_rate_hz {args.policy_rate_hz}; "
            "only matched 30 Hz and 60 Hz configurations are supported"
        )
    if args.chunk_size is None:
        args.chunk_size = args.policy_rate_hz
    if args.input_modality is None:
        args.input_modality = args.camera_name
    if args.max_observation_age_sec is None:
        args.max_observation_age_sec = (
            0.10 if args.input_modality == "sparse_ball" else 0.20
        )
    if args.input_modality == "sparse_ball" and args.sparse_source is None:
        parser.error("--sparse_source rgb|event is required with sparse_ball")
    if args.input_modality in ("rgb", "event") and args.input_modality != args.camera_name:
        parser.error("Dense --input_modality must match --camera_name")
    try:
        args.temporal_agg_mode = resolve_temporal_agg_mode(
            temporal_agg_mode=args.temporal_agg_mode,
            temporal_agg_legacy=args.temporal_agg_legacy,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        args.event_spatial_preprocessing = resolve_event_spatial_preprocessing(
            modality=("rgb" if args.input_modality == "sparse_ball" else args.camera_name),
            mask_x=args.event_mask_x,
            crop_square=args.event_crop_square,
            fill_value=args.event_mask_fill_value,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if int(args.state_dim) != 21:
        raise ValueError(f"Interception rollout requires --state_dim 21, got {args.state_dim}")
    if int(args.action_dim) != 1:
        raise ValueError(f"Interception rollout requires --action_dim 1, got {args.action_dim}")
    if int(args.chunk_size) != int(args.policy_rate_hz):
        raise ValueError(
            f"Interception rollout at {args.policy_rate_hz} Hz requires "
            f"--chunk_size {args.policy_rate_hz}, got {args.chunk_size}"
        )
    if int(args.rgb_history_frames) != 3:
        raise ValueError(
            f"Interception rollout requires --rgb_history_frames 3, got {args.rgb_history_frames}"
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
