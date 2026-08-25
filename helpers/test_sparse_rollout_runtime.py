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


def _sparse_builder_node(module, points, max_age=0.2):
    node = _base_runtime_node(module, "sparse_ball", [])
    node.visual_buffer = deque(points)
    node.joint_buffer = deque(
        (stamp, np.full(7, stamp, dtype=np.float32))
        for stamp in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
    )
    node.tcp_buffer = deque([(1.0, 0.2), (1.1, 0.2)])
    node.sparse_image_width = node.sparse_image_height = 320
    node.sparse_source = "event"
    node.sparse_topic = "/openmv_cam/event_tracker/ball_2d_px"
    node.max_observation_age_sec = max_age
    node.pre_process = lambda value: value
    node.device = torch.device("cpu")
    node._last_sparse_message_tick_ns = None
    node._last_sparse_diagnostics = {}
    node.event_update_buffer = deque()
    node._trace_stages = []
    node._trace_mark = lambda stage, **detail: node._trace_stages.append((stage, detail))
    return node


def _point(stamp, u=30.0, v=40.0):
    msg = types.SimpleNamespace(point=types.SimpleNamespace(x=u, y=v))
    return stamp, msg


def _typed(module, update_id, stamp, *, valid=True, u=30.0, v=40.0,
           receipt=None, packet_id=100, reason=""):
    return module.BufferedEventObservation(
        tracker_update_id=update_id, tracker_update_id_valid=True,
        source_packet_id=packet_id,
        source_packet_id_valid=True, source_timestamp_ns=int(stamp * 1e9),
        receipt_timestamp_ns=int((stamp if receipt is None else receipt) * 1e9),
        sensor_window_start_us=1, sensor_window_end_us=2, x_px=u, y_px=v,
        vx_px_s=3.0, vy_px_s=4.0, valid=valid, rejection_reason=reason,
        candidate_count=2, window_event_count=50, confidence=0.9,
        velocity_valid=True,
    )


def _set_typed_updates(node, updates):
    node.event_update_buffer = deque(updates)
    node.visual_buffer = deque(
        (update.source_timestamp, update) for update in updates if update.valid
    )


def test_event_provenance_sequence_unchanged_then_changed_timestamp(monkeypatch):
    module = _load_rollout_module(monkeypatch)
    node = _sparse_builder_node(module, [_point(0.95)])
    module.FrankaActRolloutNode._build_sparse_policy_inputs(node, 1.0)
    first = dict(node._last_sparse_diagnostics["event_provenance"])
    module.FrankaActRolloutNode._build_sparse_policy_inputs(node, 1.1)
    unchanged = dict(node._last_sparse_diagnostics["event_provenance"])
    node.visual_buffer.append(_point(1.08, 31.0, 41.0))
    module.FrankaActRolloutNode._build_sparse_policy_inputs(node, 1.1)
    changed = node._last_sparse_diagnostics["event_provenance"]
    assert first["event_observation_sequence"] == 1
    assert unchanged["event_observation_sequence"] == 1
    assert not unchanged["event_input_changed"]
    assert changed["event_observation_sequence"] == 2
    assert changed["event_input_changed"]


def test_event_provenance_invalid_and_stale(monkeypatch):
    module = _load_rollout_module(monkeypatch)
    invalid = _sparse_builder_node(module, [_point(0.95, -1.0, 40.0)])
    module.FrankaActRolloutNode._build_sparse_policy_inputs(invalid, 1.0)
    assert not invalid._last_sparse_diagnostics["event_provenance"]["event_valid"]

    stale = _sparse_builder_node(module, [_point(0.5)], max_age=0.2)
    module.FrankaActRolloutNode._build_sparse_policy_inputs(stale, 1.0)
    provenance = stale._last_sparse_diagnostics["event_provenance"]
    assert not provenance["event_valid"]
    assert provenance["event_age_sec"] == 0.5


def test_event_selection_precedes_sparse_build_and_model_forward(monkeypatch):
    module = _load_rollout_module(monkeypatch)
    node = _sparse_builder_node(module, [_point(0.95)])
    module.FrankaActRolloutNode._build_sparse_policy_inputs(node, 1.0)
    node._trace_stages.append(("model_forward_pass", {}))
    names = [stage for stage, _detail in node._trace_stages]
    assert names.index("event_observation_selected") < names.index("sparse_input_built")
    assert names.index("sparse_input_built") < names.index("model_forward_pass")


def test_typed_valid_update_and_complete_history_provenance(monkeypatch):
    module = _load_rollout_module(monkeypatch)
    node = _sparse_builder_node(module, [])
    updates = [_typed(module, i, stamp) for i, stamp in enumerate((0.79, 0.89, 0.99), 7)]
    _set_typed_updates(node, updates)
    module.FrankaActRolloutNode._build_sparse_policy_inputs(node, 1.0)
    p = node._last_sparse_diagnostics["event_provenance"]
    assert p["tracker_latest_update_id"] == 9
    assert p["policy_selected_update_id"] == 9
    assert p["policy_selected_valid"]
    assert p["selected_event_update_ids"] == [7, 8, 9]
    assert p["selected_event_packet_ids"] == [100, 100, 100]
    assert p["selected_event_valid_flags"] == [True, True, True]


def test_invalid_latest_holds_prior_valid_and_records_receipt_delay(monkeypatch):
    module = _load_rollout_module(monkeypatch)
    node = _sparse_builder_node(module, [])
    valid = _typed(module, 10, 0.95, receipt=0.97)
    invalid = _typed(module, 11, 0.99, valid=False, receipt=1.00,
                     reason="no_candidate", u=float("nan"), v=float("nan"))
    _set_typed_updates(node, [valid, invalid])
    module.FrankaActRolloutNode._build_sparse_policy_inputs(node, 1.0)
    p = node._last_sparse_diagnostics["event_provenance"]
    assert p["tracker_latest_update_id"] == 11
    assert not p["tracker_latest_valid"]
    assert p["tracker_latest_rejection_reason"] == "no_candidate"
    assert p["policy_selected_update_id"] == 10
    assert p["policy_selected_valid"] and p["policy_selected_is_held"]
    assert p["policy_selected_receipt_timestamp_ns"] == 970_000_000


def test_repeated_and_changed_update_ids_with_unchanged_coordinates(monkeypatch):
    module = _load_rollout_module(monkeypatch)
    node = _sparse_builder_node(module, [])
    first = _typed(module, 12, 0.95)
    _set_typed_updates(node, [first])
    module.FrankaActRolloutNode._build_sparse_policy_inputs(node, 1.0)
    module.FrankaActRolloutNode._build_sparse_policy_inputs(node, 1.01)
    assert not node._last_sparse_diagnostics["event_provenance"]["event_policy_observation_changed"]
    changed_id = _typed(module, 13, 1.005, u=30.0, v=40.0)
    _set_typed_updates(node, [first, changed_id])
    module.FrankaActRolloutNode._build_sparse_policy_inputs(node, 1.01)
    p = node._last_sparse_diagnostics["event_provenance"]
    assert p["event_policy_observation_changed"]
    assert p["policy_selected_u"] == 30.0


def test_out_of_order_and_future_updates_do_not_break_causal_selection(monkeypatch):
    module = _load_rollout_module(monkeypatch)
    node = _sparse_builder_node(module, [])
    causal = _typed(module, 20, 0.95)
    future = _typed(module, 22, 1.05)
    out_of_order_latest_received = _typed(module, 21, 0.90)
    _set_typed_updates(node, [causal, future, out_of_order_latest_received])
    module.FrankaActRolloutNode._build_sparse_policy_inputs(node, 1.0)
    p = node._last_sparse_diagnostics["event_provenance"]
    assert p["tracker_latest_update_id"] == 21
    assert p["policy_selected_update_id"] == 20


def test_typed_callback_retains_invalid_and_legacy_fallback(monkeypatch):
    module = _load_rollout_module(monkeypatch)
    node = _sparse_builder_node(module, [])
    node._clock = _Clock([1.02, 1.03])
    msg = types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=1, nanosec=0)),
        availability_timestamp_ns=1_000_000_000,
        tracker_update_id=5, tracker_update_id_valid=True,
        source_packet_id=8, source_packet_id_valid=True,
        sensor_window_start_us=10, sensor_window_end_us=20,
        x_px=0.0, y_px=0.0, vx_px_s=0.0, vy_px_s=0.0,
        valid=False, rejection_reason="empty", candidate_count=0,
        window_event_count=0, confidence=0.0, velocity_valid=False,
    )
    module.FrankaActRolloutNode.event_tracker_update_cb(node, msg)
    assert len(node.event_update_buffer) == 1
    assert not node.visual_buffer
    legacy = types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=1, nanosec=10)),
        point=types.SimpleNamespace(x=4.0, y=5.0),
    )
    module.FrankaActRolloutNode.sparse_cb(node, legacy)
    assert node.event_update_buffer[-1].legacy_pointstamped
    assert node.event_update_buffer[-1].tracker_update_id is None


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
