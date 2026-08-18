"""Runtime-level regressions for sparse policy-clock scheduling."""

import importlib
import sys
import threading
import time
import types
from collections import deque

import numpy as np
import torch

from intercept_rollout_contract import (
    select_qpos_history_at_targets,
    skip_duplicate_source_frame,
)
from sparse_ball import SparsePoint, construct_causal_sparse_history


def _load_rollout_module(monkeypatch):
    """Import the rollout class with only its unavailable ROS surface stubbed."""
    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = type("CvBridge", (), {})
    monkeypatch.setitem(sys.modules, "cv_bridge", cv_bridge)

    for package, names in (
        ("sensor_msgs.msg", ("Image", "JointState")),
        ("geometry_msgs.msg", ("PointStamped",)),
        ("std_msgs.msg", ("Float64", "Float64MultiArray")),
    ):
        module = types.ModuleType(package)
        for name in names:
            setattr(module, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, package, module)

    std_srvs = types.ModuleType("std_srvs.srv")
    std_srvs.Trigger = type(
        "Trigger", (),
        {"Request": type("Request", (), {}), "Response": type("Response", (), {})},
    )
    monkeypatch.setitem(sys.modules, "std_srvs.srv", std_srvs)

    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = object
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_qos.QoSProfile = type("QoSProfile", (), {})
    rclpy_qos.ReliabilityPolicy = type(
        "ReliabilityPolicy", (), {"BEST_EFFORT": object()},
    )
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", rclpy_node)
    monkeypatch.setitem(sys.modules, "rclpy.qos", rclpy_qos)

    policy = types.ModuleType("policy")
    policy.ACTPolicy = type("ACTPolicy", (), {})
    monkeypatch.setitem(sys.modules, "policy", policy)
    sys.modules.pop("franka_act_intercept_rollout", None)
    return importlib.import_module("franka_act_intercept_rollout")


class _Clock:
    def __init__(self, seconds):
        self._nanoseconds = iter(int(round(value * 1e9)) for value in seconds)

    def now(self):
        return types.SimpleNamespace(nanoseconds=next(self._nanoseconds))


class _Logger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass


def _base_runtime_node(module, modality, clock_seconds):
    node = object.__new__(module.FrankaActRolloutNode)
    node.input_modality = modality
    node._clock = _Clock(clock_seconds)
    node.get_clock = lambda: node._clock
    node.get_logger = lambda: _Logger()
    node._aggregation_lock = threading.Lock()
    node._last_attempted_anchor_ns = None
    node.reset_anchor_floor_ns = None
    node.duplicate_timer_tick_skip_count = 0
    node.backward_anchor_count = 0
    node.aggregation_reset_gap_count = 0
    node.failed_or_stale_inference_gap_count = 0
    node.post_inference_stale_count = 0
    node.accepted_prediction_count = 0
    node.accepted_distinct_anchor_count = 0
    node.anchor_gap_reset_threshold_ns = 100_000_000
    node.expected_step_ns = 33_333_333
    node.anchor_gap_tolerance_ns = 10_000_000
    node.chunk_size = 3
    node.action_lookahead_steps = 0
    node.aggregator = module.TemporalAbsoluteAggregator(
        chunk_size=3, mode="latest", recent_window=3,
    )
    node.step_index = 0
    node.action_mean = np.zeros(1, dtype=np.float32)
    node.action_std = np.ones(1, dtype=np.float32)
    node.max_anchor_age_sec = 1.0
    node.max_observation_age_sec = 0.5
    node._last_policy_output_timestamp = None
    node._last_diag_log_sec = time.time()
    node.diag_log_period_sec = 10_000.0
    node.fps = 30.0
    node.temporal_agg_mode = "latest"
    node.selected_target_offset_frames = 1
    node.selected_target_offset_ms = 1000.0 / 30.0
    node.rollout_epoch = 0
    node._last_anchor_discontinuity_reset_reason = ""
    node._last_throttled_log_sec = {}
    node.reject_log_period_sec = 1.0
    node._active_latency_trace = None
    node._trace_mark = lambda *_args, **_kwargs: None
    node.publish_predictions = lambda *_args, **_kwargs: None
    node._forward_calls = 0

    def forward(_qpos, _image):
        node._forward_calls += 1
        return torch.zeros((1, 3, 1), dtype=torch.float32), 0.1

    node._run_policy_forward = forward
    return node


def test_unchanged_sparse_source_timestamp_does_not_skip_policy_ticks():
    source_stamp_ns = 950_000_000
    attempted_anchor_ns = source_stamp_ns
    decisions = []
    for _policy_stamp_ns in (1_000_000_000, 1_033_333_333):
        duplicate = source_stamp_ns == attempted_anchor_ns
        decisions.append(duplicate and skip_duplicate_source_frame("sparse_ball"))
    assert decisions == [False, False]


def test_dense_rgb_and_event_still_skip_unchanged_source_frames():
    for modality in ("rgb", "event"):
        assert skip_duplicate_source_frame(modality)
        assert (123 == 123) and skip_duplicate_source_frame(modality)


def test_repeated_sparse_ticks_are_causal_and_age_the_cached_observation():
    points = [SparsePoint(0.79, 10, 20), SparsePoint(0.95, 30, 40)]
    first = construct_causal_sparse_history(
        points, 1.00, (-0.2, -0.1, 0.0), 320, 320, 0.20,
    )
    second = construct_causal_sparse_history(
        points, 1.05, (-0.2, -0.1, 0.0), 320, 320, 0.20,
    )
    assert second[-1, 3] > first[-1, 3]
    assert second[-1, 3] == np.float32(0.10)
    assert np.all(second[:, 2] == 1)


def test_sparse_qpos_selection_is_policy_time_causal():
    stamps = [0.79, 0.81, 0.89, 0.91, 0.99, 1.01]
    samples = [np.full(7, index, dtype=np.float32)
               for index in range(len(stamps))]
    history, selected = select_qpos_history_at_targets(
        stamps, samples, (0.8, 0.9, 1.0),
    )
    assert selected == (0.79, 0.89, 0.99)
    np.testing.assert_array_equal(history.reshape(3, 7)[:, 0], [0, 2, 4])


def test_real_run_policy_step_accepts_repeated_sparse_source_ticks(monkeypatch):
    module = _load_rollout_module(monkeypatch)
    node = _base_runtime_node(module, "sparse_ball", [1.0, 1.0, 1.1, 1.1])
    point = types.SimpleNamespace(
        header=types.SimpleNamespace(
            stamp=types.SimpleNamespace(sec=0, nanosec=750_000_000),
        ),
        point=types.SimpleNamespace(x=30.0, y=40.0),
    )
    node.visual_buffer = deque([(0.75, point)])
    node.joint_buffer = deque(
        (stamp, np.full(7, stamp, dtype=np.float32))
        for stamp in (0.7, 0.8, 0.9, 1.0, 1.1)
    )
    node.tcp_buffer = deque([(1.0, 0.2), (1.1, 0.2)])
    node.sparse_image_width = 320
    node.sparse_image_height = 320
    node.sparse_source = "event"
    node.sparse_topic = "/openmv_cam/event_tracker/ball_2d_px"
    node.pre_process = lambda value: value
    node.device = torch.device("cpu")
    node._last_sparse_message_tick_ns = None
    node._last_sparse_diagnostics = {}

    first = module.FrankaActRolloutNode.run_policy_step(node)
    first_diagnostics = dict(node._last_sparse_diagnostics)
    second = module.FrankaActRolloutNode.run_policy_step(node)

    assert first == (True, "")
    assert second == (True, "")
    assert node._forward_calls == 2
    assert node.accepted_prediction_count == 2
    assert node.duplicate_timer_tick_skip_count == 0
    assert first_diagnostics["policy_timestamp"] == 1.0
    assert node._last_sparse_diagnostics["policy_timestamp"] == 1.1
    assert first_diagnostics["sparse_source_timestamps"][-1] == 0.75
    assert node._last_sparse_diagnostics["sparse_source_timestamps"][-1] == 0.75
    assert node._last_sparse_diagnostics["inference_without_new_sparse_message"]


def test_real_run_policy_step_rejects_dense_duplicate_anchor(monkeypatch):
    module = _load_rollout_module(monkeypatch)
    node = _base_runtime_node(module, "event", [1.0, 1.0, 1.1])
    sync = types.SimpleNamespace(
        visual_timestamps=(0.8, 0.9, 1.0),
        qpos_timestamps=(0.8, 0.9, 1.0),
        anchor_tcp_s=0.2,
        anchor_tcp_s_timestamp=1.0,
        history_indices=(0, 1, 2),
        qpos_history=np.zeros(21, dtype=np.float32),
    )
    node.build_policy_inputs = lambda _policy_time: (
        sync, torch.zeros((1, 21)), torch.zeros((1, 1, 9, 2, 2)), 1_000_000_000,
    )

    assert module.FrankaActRolloutNode.run_policy_step(node) == (True, "")
    assert module.FrankaActRolloutNode.run_policy_step(node) == (
        False, "duplicate_source_frame",
    )
    assert node._forward_calls == 1
    assert node.accepted_prediction_count == 1
    assert node.duplicate_timer_tick_skip_count == 1


def test_sparse_epoch_reset_clears_diagnostics_without_enabling_duplicates(monkeypatch):
    module = _load_rollout_module(monkeypatch)
    node = _base_runtime_node(module, "sparse_ball", [1.2, 1.2])
    point = types.SimpleNamespace(
        header=types.SimpleNamespace(
            stamp=types.SimpleNamespace(sec=0, nanosec=750_000_000),
        ),
        point=types.SimpleNamespace(x=30.0, y=40.0),
    )
    node.visual_buffer = deque([(0.75, point)])
    node.joint_buffer = deque(
        (stamp, np.zeros(7, dtype=np.float32))
        for stamp in (0.7, 0.8, 0.9, 1.0, 1.2)
    )
    node.tcp_buffer = deque([(1.2, 0.2)])
    node.sparse_image_width = node.sparse_image_height = 320
    node.sparse_source = "event"
    node.sparse_topic = "/openmv_cam/event_tracker/ball_2d_px"
    node.pre_process = lambda value: value
    node.device = torch.device("cpu")
    node._last_sparse_message_tick_ns = 750_000_000
    node._last_sparse_diagnostics = {"stale": True}
    node._last_policy_output_timestamp = 1.0

    response = types.SimpleNamespace(success=False, message="")
    module.FrankaActRolloutNode._handle_reset_temporal_aggregation(
        node, object(), response,
    )
    assert response.success
    assert node.rollout_epoch == 1
    assert node._last_sparse_message_tick_ns is None
    assert node._last_sparse_diagnostics == {}
    assert node._last_policy_output_timestamp is None
    assert not skip_duplicate_source_frame("sparse_ball")

    assert module.FrankaActRolloutNode.run_policy_step(node) == (True, "")
    assert node._forward_calls == 1
    assert node.duplicate_timer_tick_skip_count == 0
