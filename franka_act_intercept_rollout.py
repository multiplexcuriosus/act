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
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from intercept_rollout_contract import (
    TemporalAbsoluteAggregator,
    absolute_s_from_anchor,
    build_rgb_history_tensor,
    denormalize_delta_chunk,
    extract_arm_qpos,
    resolve_temporal_agg_mode,
    select_latest_index_at_or_before,
    select_qpos_history_at_targets,
    select_sync_observation,
    validate_anchor_freshness,
    validate_intercept_stats_and_config,
)
from policy import ACTPolicy
from rollout_latency_trace import RolloutLatencyTracer, image_source_stamp_ns
from sparse_ball import (BallObservation, build_sparse_history, history_target_times,
                         select_sparse_observation)


SPARSE_TOPICS = {
    "rgb": "/ball_tracker2/ball_2d_px",
    "event": "/openmv_cam/event_tracker/ball_2d_px",
}
SPARSE_SOURCE_DIMENSIONS = {"rgb": (1280, 720), "event": (320, 320)}


def resolve_sparse_topic(source: str, explicit_topic: Optional[str],
                         allow_mismatch: bool = False) -> str:
    source = str(source)
    if source not in SPARSE_TOPICS:
        raise ValueError(f"sparse_source must be rgb or event, got {source!r}")
    topic = explicit_topic or SPARSE_TOPICS[source]
    other_source = "event" if source == "rgb" else "rgb"
    if topic == SPARSE_TOPICS[other_source] and not allow_mismatch:
        raise ValueError(
            f"sparse_source={source} conflicts with sparse_topic={topic}; "
            "use --allow_sparse_topic_source_mismatch only for an intentional override"
        )
    return topic


def rollout_subscription_types(input_modality: str) -> Tuple[str, ...]:
    if input_modality == "sparse_ball":
        return ("geometry_msgs/msg/PointStamped", "sensor_msgs/msg/JointState", "std_msgs/msg/Float64")
    return ("sensor_msgs/msg/Image", "sensor_msgs/msg/JointState", "std_msgs/msg/Float64")


def skip_duplicate_source_frame(input_modality: str) -> bool:
    """Dense legacy rollout is frame-driven; sparse rollout is policy-clock driven."""
    return input_modality != "sparse_ball"


def create_visual_subscription(node, input_modality: str, image_topic: str,
                               sparse_topic: str):
    """Create exactly one visual-input subscription for the selected modality."""
    if input_modality == "sparse_ball":
        return node.create_subscription(PointStamped, sparse_topic, node.ball_cb, 10)
    return node.create_subscription(Image, image_topic, node.image_cb, 10)


def clear_sparse_temporal_history(sparse_buffer, joint_buffer, tcp_buffer) -> None:
    sparse_buffer.clear()
    joint_buffer.clear()
    tcp_buffer.clear()


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

        self.bridge = None
        if args.input_modality != "sparse_ball":
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.image_topic = args.image_topic
        self.input_modality = str(args.input_modality)
        self.sparse_source = str(args.sparse_source)
        self.ball_topic = args.sparse_topic
        self.joint_topic = args.joint_topic
        self.current_tcp_s_topic = args.current_tcp_s_topic
        self.prediction_topic = args.prediction_topic
        self.prediction_current_topic = args.prediction_current_topic
        self.temporal_agg_reset_service = args.temporal_agg_reset_service
        self.publish_current_scalar = bool(args.publish_current_scalar)
        self.latency_trace_cuda_sync = bool(args.latency_trace_cuda_sync)

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
        self.image_channels = 0 if self.input_modality == "sparse_ball" else 9
        self.image_size = int(args.image_size)
        self.max_source_buffer = int(args.max_source_buffer)
        self.max_observation_age_sec = float(args.max_observation_age_sec)
        self.max_anchor_age_sec = float(args.max_anchor_age_sec)
        self.diag_log_period_sec = float(args.diag_log_period_sec)
        self.reject_log_period_sec = float(args.reject_log_period_sec)

        self.rgb_buffer: Deque[Tuple[float, Image]] = deque(maxlen=self.max_source_buffer)
        self.sparse_buffer: Deque[Tuple[float, PointStamped]] = deque(maxlen=self.max_source_buffer)
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
        self._policy_step_ids_by_step: Dict[int, str] = {}

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
            "camera_names": [args.camera_name],
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "use_bce_last_action_dim": args.use_bce_last_action_dim,
            "rgb_history_frames": args.rgb_history_frames,
            "image_channels": self.image_channels,
            "image_size": self.image_size,
            "input_modality": self.input_modality,
            "sparse_feature_dim": args.sparse_feature_dim,
            "sparse_history_length": args.sparse_history_length,
        }

        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
        stats_path = os.path.join(args.ckpt_dir, args.stats_name)

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        stats_arrays = validate_intercept_stats_and_config(
            stats=stats,
            policy_config=policy_config,
            expected_chunk_size=self.chunk_size,
            sparse_runtime={
                "sparse_source": self.sparse_source,
                "max_observation_age_sec": self.max_observation_age_sec,
                "image_width": SPARSE_SOURCE_DIMENSIONS[self.sparse_source][0],
                "image_height": SPARSE_SOURCE_DIMENSIONS[self.sparse_source][1],
            } if self.input_modality == "sparse_ball" else None,
        )

        self.qpos_mean = stats_arrays["qpos_mean"]
        self.qpos_std = stats_arrays["qpos_std"]
        self.action_mean = stats_arrays["action_mean"]
        self.action_std = stats_arrays["action_std"]
        if self.input_modality == "sparse_ball":
            policy_config["sparse_mean"] = stats_arrays["sparse_mean"]
            policy_config["sparse_std"] = stats_arrays["sparse_std"]
            self.ball_image_width = int(stats["image_width"])
            self.ball_image_height = int(stats["image_height"])
            self.sparse_max_observation_age = self.max_observation_age_sec
            validated = stats_arrays.get("_validated_metadata", np.asarray([], dtype=object)).tolist()
            unavailable = stats_arrays.get("_unavailable_metadata", np.asarray([], dtype=object)).tolist()
            self.get_logger().info(
                f"Sparse checkpoint metadata validated={validated} unavailable={unavailable}"
            )

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

        self.visual_subscription = create_visual_subscription(
            self, self.input_modality, self.image_topic, self.ball_topic
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

        self.get_logger().info(f"input_modality={self.input_modality}")
        if self.input_modality == "sparse_ball":
            self.get_logger().info(f"sparse_source={self.sparse_source} sparse_topic={self.ball_topic}")
        self.get_logger().info(f"source_topic={self.ball_topic if self.input_modality == 'sparse_ball' else self.image_topic}")
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
        self._append_monotonic("RGB", self.rgb_buffer, t, msg)

    def ball_cb(self, msg: PointStamped) -> None:
        t = stamp_to_sec(msg.header.stamp)
        self._append_monotonic("sparse_ball", self.sparse_buffer, t, msg)

    def joint_cb(self, msg: JointState) -> None:
        try:
            qpos = extract_arm_qpos(msg.name, msg.position)
        except Exception as exc:
            self.get_logger().warn(f"Dropping JointState sample: {exc}")
            return
        t = stamp_to_sec(msg.header.stamp)
        if t <= 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
            self._throttled_warn(
                "jointstate_missing_header_stamp",
                "JointState header timestamp unavailable; using ROS receipt time for qpos only.",
            )
        self._append_monotonic("JointState", self.joint_buffer, t, qpos)

    def current_tcp_s_cb(self, msg: Float64) -> None:
        value = float(msg.data)
        if not np.isfinite(value):
            self.get_logger().warn("Dropping non-finite /middle_line/current_tcp_s sample")
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        self._append_monotonic("current_tcp_s", self.tcp_buffer, t, value)

    def ready(self) -> bool:
        visual_ready = bool(self.sparse_buffer) if self.input_modality == "sparse_ball" else bool(self.rgb_buffer)
        return visual_ready and bool(self.joint_buffer) and bool(self.tcp_buffer)

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

    def _trace_event(self, event: str, **detail) -> None:
        tracer = getattr(self, "latency_tracer", None)
        if tracer is not None:
            tracer.emit(getattr(self, "_active_latency_trace", None), event, **detail)

    def _run_policy_forward(self, qpos: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, float]:
        with torch.inference_mode():
            normalized_image = self.policy.preprocess_image(image)
        self._trace_mark(
            "tensor_construction_and_preprocessing",
            qpos_shape=list(qpos.shape),
            image_shape=list(image.shape),
        )
        # A traced CUDA completion must include a synchronization boundary;
        # otherwise Python return only proves that kernels were enqueued.
        tracer = getattr(self, "latency_tracer", None)
        cuda_timing = self.device.type == "cuda" and (
            self.latency_trace_cuda_sync or (tracer is not None and tracer.enabled)
        )
        if cuda_timing:
            sync_start = time.perf_counter()
            torch.cuda.synchronize()
            self._trace_mark(
                "cuda_sync_before_forward",
                cuda_sync_before_ms=(time.perf_counter() - sync_start) * 1000.0,
            )
        start = time.perf_counter()
        self._trace_event("inference_started")
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
        self._trace_event(
            "inference_completed",
            cuda_synchronized=bool(cuda_timing),
        )
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
        warmup_shape = ((1, 3, 4) if self.input_modality == "sparse_ball" else (
                1,                    # batch
                1,                    # one logical camera
                self.image_channels,  # 9 for 3 RGB history frames
                self.image_size,
                self.image_size,
            ))
        warmup_image = torch.zeros(
            warmup_shape,
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
            rgb_msg: Image,
            fallback_sec: float,
        ) -> int:
            try:
                stamp = rgb_msg.header.stamp
                return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            except Exception:
                return int(round(float(fallback_sec) * 1e9))

    def _reset_temporal_aggregation(self, reason: str) -> int:
        discarded_entries = len(getattr(self.aggregator, "_history", []))
        self.aggregator.reset()
        self.step_index = 0
        self._policy_step_ids_by_step.clear()
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
            source_buffer = self.sparse_buffer if self.input_modality == "sparse_ball" else self.rgb_buffer
            if source_buffer:
                floor_sec, floor_msg = source_buffer[-1]
                floor_anchor_ns = self._anchor_timestamp_ns_from_observation(floor_msg, floor_sec)

            discarded_entries = self._reset_temporal_aggregation("service_reset")
            clear_sparse_temporal_history(self.sparse_buffer, self.joint_buffer, self.tcp_buffer)
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
            f"rgb_anchor_floor_ns={floor_text} discarded_entries={discarded_entries}"
        )
        return response

    def _warn_post_inference_stale(
        self,
        rgb_age_sec: float,
        tcp_age_sec: float,
        inference_ms: float,
    ) -> None:
        self._throttled_warn(
            "post_inference_stale",
            "Rejecting stale post-inference rollout result: "
            f"rgb_age={rgb_age_sec:.4f}s (limit={self.max_observation_age_sec:.4f}s), "
            f"tcp_age={tcp_age_sec:.4f}s (limit={self.max_anchor_age_sec:.4f}s), "
            f"inference_ms={inference_ms:.2f}"
        )

    def build_policy_inputs(self, policy_time: Optional[float] = None):
        if self.input_modality == "sparse_ball":
            return self._build_sparse_policy_inputs(float(policy_time))
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
        self._trace_mark(
            "input_history_selection",
            selected_history_timestamps_ns=[int(round(ts * 1e9)) for ts in sync.rgb_timestamps],
            selected_qpos_timestamps_ns=[int(round(ts * 1e9)) for ts in sync.qpos_timestamps],
        )

        selected_rgb_msgs = [rgb_messages[index] for index in sync.history_indices]
        history_source_stamp_ns = [image_source_stamp_ns(msg) for msg in selected_rgb_msgs]
        if self.input_modality == "sparse_ball":
            observations = [BallObservation(ts, float(msg.point.x), float(msg.point.y))
                            for ts, msg in self.rgb_buffer]
            image_np = build_sparse_history(
                observations, rgb_timestamps, len(rgb_timestamps) - 1,
                self.ball_image_width, self.ball_image_height, self.sparse_max_observation_age,
            )[None, ...]
            self._trace_mark("sparse_observation_selection_and_construction",
                             ball_source_timestamp_ns=history_source_stamp_ns[-1])
        else:
            rgb_frames = [self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8") for msg in selected_rgb_msgs]
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
        self._trace_mark("complete_observation_frame_acceptance")

        qpos_norm = self.pre_process(sync.qpos_history)
        if not np.all(np.isfinite(qpos_norm)):
            raise ValueError("Normalized qpos contains non-finite values")

        qpos = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)
        image = torch.from_numpy(image_np).float().to(self.device)
        anchor_rgb_msg = selected_rgb_msgs[-1]
        anchor_timestamp_ns = self._anchor_timestamp_ns_from_observation(
            anchor_rgb_msg,
            anchor_observation_timestamp,
        )
        return sync, qpos, image, anchor_timestamp_ns, history_source_stamp_ns

    def _build_sparse_policy_inputs(self, policy_time: float):
        targets = history_target_times(policy_time)
        observations = [
            BallObservation(ts, float(msg.point.x), float(msg.point.y),
                            bool(np.isfinite((msg.point.x, msg.point.y)).all()
                                 and 0.0 <= float(msg.point.x) < self.ball_image_width
                                 and 0.0 <= float(msg.point.y) < self.ball_image_height))
            for ts, msg in self.sparse_buffer
        ]
        selections = [
            select_sparse_observation(observations, target, self.ball_image_width,
                                      self.ball_image_height, self.sparse_max_observation_age)
            for target in targets
        ]
        image_np = build_sparse_history(
            observations, targets, self.ball_image_width, self.ball_image_height,
            self.sparse_max_observation_age,
        )[None, ...]
        joint_timestamps = [item[0] for item in self.joint_buffer]
        qpos_samples = [item[1] for item in self.joint_buffer]
        qpos_history, qpos_timestamps = select_qpos_history_at_targets(
            joint_timestamps, qpos_samples, targets
        )
        tcp_timestamps = [item[0] for item in self.tcp_buffer]
        tcp_index = select_latest_index_at_or_before(tcp_timestamps, policy_time)
        if tcp_index is None:
            raise ValueError(f"No causal current_tcp_s at policy timestamp {policy_time:.9f}")
        tcp_timestamp, anchor_tcp_s = self.tcp_buffer[tcp_index]
        sync = type("SparseSync", (), {
            "history_indices": (-1, -1, -1),
            "rgb_timestamps": targets,
            "qpos_timestamps": qpos_timestamps,
            "anchor_tcp_s": float(anchor_tcp_s),
            "anchor_tcp_s_timestamp": float(tcp_timestamp),
            "qpos_history": qpos_history,
        })()
        qpos_norm = self.pre_process(qpos_history)
        if not np.all(np.isfinite(qpos_norm)):
            raise ValueError("Normalized qpos contains non-finite values")
        qpos = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)
        image = torch.from_numpy(image_np).float().to(self.device)
        source_stamps = [0 if item.source_timestamp is None else int(round(item.source_timestamp * 1e9))
                         for item in selections]
        latest_stamp = source_stamps[-1]
        no_new = bool(latest_stamp and latest_stamp == self._last_attempted_anchor_ns)
        self._trace_mark(
            "sparse_observation_selection_and_construction",
            sparse_source=self.sparse_source,
            sparse_topic=self.ball_topic,
            policy_tick_timestamp=policy_time,
            history_target_timestamps=list(targets),
            sparse_source_timestamps=[item.source_timestamp for item in selections],
            sparse_observation_ages=[item.observation_age for item in selections],
            sparse_valid_flags=[item.valid for item in selections],
            selected_qpos_timestamps=list(qpos_timestamps),
            inference_without_new_sparse_message=no_new,
        )
        return sync, qpos, image, latest_stamp, source_stamps

    def publish_predictions(
        self,
        absolute_chunk: np.ndarray,
        current_value: Optional[float],
        *,
        contributing_policy_step_ids,
    ) -> None:
        chunk_msg = Float64MultiArray()
        chunk_msg.data = [float(value) for value in absolute_chunk.tolist()]
        self.prediction_pub.publish(chunk_msg)
        self.latency_tracer.emit(
            self._active_latency_trace, "action_published",
            action_chunk_index=int(self.action_lookahead_steps),
            contributing_policy_step_ids=list(contributing_policy_step_ids),
            output_topic=self.prediction_topic,
        )

        if self.prediction_current_pub is not None and current_value is not None:
            current_msg = Float64()
            current_msg.data = float(current_value)
            self.prediction_current_pub.publish(current_msg)
            self.latency_tracer.target_s_published(
                self._active_latency_trace,
                float(current_value),
                action_chunk_index=int(self.action_lookahead_steps),
                contributing_policy_step_ids=list(contributing_policy_step_ids),
                output_topic=self.prediction_current_topic,
                downstream_tracker_causality="unavailable_float64_interface",
            )

    def run_policy_step(self) -> Tuple[bool, str]:
        step_start_wall = time.perf_counter()
        policy_time = self.get_clock().now().nanoseconds * 1e-9
        sync, qpos, curr_image, anchor_timestamp_ns, history_source_stamp_ns = self.build_policy_inputs(policy_time)

        with self._aggregation_lock:
            if self.input_modality != "sparse_ball" and self.reset_anchor_floor_ns is not None and anchor_timestamp_ns <= self.reset_anchor_floor_ns:
                self.duplicate_timer_tick_skip_count += 1
                self._last_attempted_anchor_ns = anchor_timestamp_ns
                return False, "frame_at_or_before_reset_floor"

            if skip_duplicate_source_frame(self.input_modality) and self._last_attempted_anchor_ns is not None:
                delta_ns = anchor_timestamp_ns - self._last_attempted_anchor_ns
                if delta_ns == 0:
                    self.duplicate_timer_tick_skip_count += 1
                    return False, "duplicate_source_frame"
                if delta_ns < 0:
                    self.backward_anchor_count += 1
                    self._reset_temporal_aggregation("backward_rgb_anchor_timestamp")
                    self._throttled_warn(
                        "backward_anchor",
                        "Detected backward RGB anchor timestamp; resetting temporal aggregation. "
                        f"anchor_ns={anchor_timestamp_ns} delta_ns={delta_ns}",
                    )
                elif delta_ns > self.anchor_gap_reset_threshold_ns:
                    self.aggregation_reset_gap_count += 1
                    self._reset_temporal_aggregation("rgb_anchor_gap")
                    self._throttled_warn(
                        "anchor_gap",
                        "Detected RGB anchor timestamp gap; resetting temporal aggregation. "
                        f"delta_ms={delta_ns / 1e6:.2f} expected_ms={self.expected_step_ns / 1e6:.2f} "
                        f"tolerance_ms={self.anchor_gap_tolerance_ns / 1e6:.2f}",
                    )

            # Mark this anchor as attempted before running inference so duplicate timer ticks
            # cannot trigger repeated attempts on the same cached frame.
            self._last_attempted_anchor_ns = anchor_timestamp_ns

        policy_step_id = self.latency_tracer.policy_input_accepted(
            self._active_latency_trace, history_source_stamp_ns
        )

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
            if self.input_modality != "sparse_ball":
                validate_anchor_freshness(
                    anchor_timestamp=sync.anchor_tcp_s_timestamp,
                    observation_timestamp=sync.rgb_timestamps[-1],
                    now_timestamp=post_now_sec,
                    max_anchor_age_sec=self.max_anchor_age_sec,
                    max_observation_age_sec=self.max_observation_age_sec,
                )
        except ValueError:
            with self._aggregation_lock:
                self.post_inference_stale_count += 1
                self.failed_or_stale_inference_gap_count += 1
            rgb_age_sec = post_now_sec - sync.rgb_timestamps[-1]
            tcp_age_sec = sync.rgb_timestamps[-1] - sync.anchor_tcp_s_timestamp
            self._warn_post_inference_stale(
                rgb_age_sec=rgb_age_sec,
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
                if policy_step_id:
                    self._policy_step_ids_by_step[self.step_index] = policy_step_id
                minimum_retained_step = self.step_index - self.chunk_size
                self._policy_step_ids_by_step = {
                    step: identifier for step, identifier in self._policy_step_ids_by_step.items()
                    if step >= minimum_retained_step
                }
                selection = self.aggregator.selection_for_step(self.step_index)
                contributor_steps = self.aggregator.contributing_steps_for_step(self.step_index)
            if selection is None:
                raise RuntimeError("Temporal aggregator produced no current value")
            if not np.isfinite(selection.value):
                raise ValueError("Temporal aggregator produced non-finite current absolute-s")
            if not np.isfinite(selection.effective_age_frames):
                raise ValueError("Temporal aggregator produced non-finite effective age")
            current_value = float(selection.value)
            agg_contributors = int(selection.contributor_count)
            agg_effective_age_frames = float(selection.effective_age_frames)
            contributing_policy_step_ids = [
                self._policy_step_ids_by_step[step]
                for step in contributor_steps if step in self._policy_step_ids_by_step
            ]
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

        self.publish_predictions(
            absolute_s,
            current_value,
            contributing_policy_step_ids=contributing_policy_step_ids,
        )
        self._trace_mark("prediction_publication_completion")
        with self._aggregation_lock:
            self.accepted_prediction_count += 1
            self.accepted_distinct_anchor_count += 1

        total_step_ms = (time.perf_counter() - step_start_wall) * 1000.0
        self._trace_mark("rollout_step_complete", policy_forward_ms=inference_ms,
                         total_rollout_step_ms=total_step_ms)
        max_input_age = post_now_sec - sync.rgb_timestamps[-1]
        max_sync_error = max(
            max(abs(rgb_ts - qpos_ts) for rgb_ts, qpos_ts in zip(sync.rgb_timestamps, sync.qpos_timestamps)),
            sync.rgb_timestamps[-1] - sync.anchor_tcp_s_timestamp,
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
                f"rgb_ts={[round(ts, 6) for ts in sync.rgb_timestamps]} "
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
        source_buffer = self.sparse_buffer if self.input_modality == "sparse_ball" else self.rgb_buffer
        if source_buffer:
            source_stamp_ns = image_source_stamp_ns(source_buffer[-1][1])
        trace = self.latency_tracer.begin(
            source_stamp_ns,
            source_modality=(f"sparse_ball/{self.sparse_source}" if self.input_modality == "sparse_ball" else self.input_modality),
        )
        self._active_latency_trace = trace
        valid = False
        rejection_reason = ""
        if not self.ready():
            rejection_reason = f"waiting for {self.input_modality}, JointState, and current_tcp_s streams"
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
    parser.add_argument("--input_modality", choices=["rgb", "sparse_ball"], default="rgb")
    parser.add_argument("--sparse_source", choices=["rgb", "event"], default="rgb")
    parser.add_argument("--sparse_topic", type=str, default=None)
    parser.add_argument("--ball_topic", dest="sparse_topic", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--allow_sparse_topic_source_mismatch", action="store_true")
    parser.add_argument("--sparse_feature_dim", type=int, default=4)
    parser.add_argument("--sparse_history_length", type=int, default=3)
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

    parser.add_argument("--policy_fps", "--fps", dest="fps", type=float, default=30.0)
    parser.add_argument("--max_source_buffer", type=int, default=256)
    parser.add_argument(
        "--max_observation_age_sec", type=float, default=None,
        help="Sparse default: 0.10 s; legacy dense default: 0.20 s.",
    )
    parser.add_argument("--max_anchor_age_sec", type=float, default=0.10)
    parser.add_argument("--diag_log_period_sec", type=float, default=1.0)
    parser.add_argument("--reject_log_period_sec", type=float, default=1.0)
    parser.add_argument("--enable_latency_trace", action="store_true", default=False)
    parser.add_argument("--latency_trace_cuda_sync", action="store_true", default=False)
    parser.add_argument("--latency_trace_topic", type=str, default="/intercept_trace/act_rollout")
    parser.add_argument("--latency_run_id", type=str, default="")

    parser.add_argument("--state_dim", type=int, default=21)
    parser.add_argument("--action_dim", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--rgb_history_frames", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--camera_name", type=str, default="rgb", choices=["rgb", "sparse_ball"])

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
    args.camera_name = args.input_modality
    if args.max_observation_age_sec is None:
        args.max_observation_age_sec = 0.10 if args.input_modality == "sparse_ball" else 0.20
    try:
        args.sparse_topic = resolve_sparse_topic(
            args.sparse_source, args.sparse_topic, args.allow_sparse_topic_source_mismatch
        )
    except ValueError as exc:
        parser.error(str(exc))
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
    if int(args.sparse_feature_dim) != 4:
        parser.error(f"--sparse_feature_dim must be 4, got {args.sparse_feature_dim}")
    if int(args.sparse_history_length) != 3:
        parser.error(f"--sparse_history_length must be 3, got {args.sparse_history_length}")
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
