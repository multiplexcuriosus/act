import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("rclpy")
from franka_act_intercept_rollout import (SPARSE_TOPICS, clear_sparse_temporal_history,
                                          create_visual_subscription,
                                          policy_period_sec,
                                          resolve_sparse_topic,
                                          rollout_subscription_types,
                                          skip_duplicate_source_frame,
                                          validate_sparse_checkpoint_state_dict)
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image


def test_default_rgb_and_event_topic_selection():
    assert resolve_sparse_topic("rgb", None) == "/ball_tracker2/ball_2d_px"
    assert resolve_sparse_topic("event", None) == "/openmv_cam/event_tracker/ball_2d_px"


def test_explicit_topic_override_and_mismatch_validation():
    assert resolve_sparse_topic("rgb", "/custom/ball") == "/custom/ball"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_sparse_topic("rgb", SPARSE_TOPICS["event"])
    assert resolve_sparse_topic("rgb", SPARSE_TOPICS["event"], True) == SPARSE_TOPICS["event"]


def test_sparse_subscription_plan_contains_no_image():
    sparse_types = rollout_subscription_types("sparse_ball")
    assert "sensor_msgs/msg/Image" not in sparse_types
    assert "geometry_msgs/msg/PointStamped" in sparse_types
    assert "sensor_msgs/msg/Image" in rollout_subscription_types("rgb")


def test_sparse_mode_actually_creates_point_subscription_not_image():
    class FakeNode:
        ball_cb = object()
        image_cb = object()
        def __init__(self):
            self.created = []
        def create_subscription(self, msg_type, topic, callback, qos):
            self.created.append((msg_type, topic, callback, qos))
            return self.created[-1]

    node = FakeNode()
    create_visual_subscription(node, "sparse_ball", "/rgb/image", "/sparse/ball")
    assert len(node.created) == 1
    assert node.created[0][0] is PointStamped
    assert node.created[0][0] is not Image


def test_sparse_fixed_clock_never_skips_a_duplicate_detection_timestamp():
    assert skip_duplicate_source_frame("sparse_ball") is False
    assert skip_duplicate_source_frame("rgb") is True
    assert policy_period_sec(30) == pytest.approx(1 / 30)
    assert policy_period_sec(60) == pytest.approx(1 / 60)
    with pytest.raises(ValueError, match="policy_rate_hz"):
        policy_period_sec(20)


def test_sparse_readiness_does_not_require_a_detection():
    from franka_act_intercept_rollout import FrankaActRolloutNode

    node = object.__new__(FrankaActRolloutNode)
    node.input_modality = "sparse_ball"
    node.sparse_buffer = []
    node.rgb_buffer = []
    node.joint_buffer = [(1.0, object())]
    node.tcp_buffer = [(1.0, 0.0)]
    assert FrankaActRolloutNode.ready(node)


def test_two_sparse_timer_ticks_infer_without_a_new_message():
    from franka_act_intercept_rollout import FrankaActRolloutNode

    class FakeTracer:
        def begin(self, *_args, **_kwargs):
            return object()
        def finish(self, *_args, **_kwargs):
            pass

    class FakeNode:
        input_modality = "sparse_ball"
        sparse_source = "rgb"
        # The same cached sparse message remains present for both timer ticks.
        sparse_buffer = [(1.0, object())]
        rgb_buffer = []
        running = True
        latency_tracer = FakeTracer()
        _active_latency_trace = None
        calls = 0

        def ready(self):
            return True
        def run_policy_step(self):
            self.calls += 1
            return True, ""
        def _reject(self, _reason):
            raise AssertionError("timer tick unexpectedly rejected")

    node = FakeNode()
    FrankaActRolloutNode.timer_cb(node)
    FrankaActRolloutNode.timer_cb(node)
    assert node.calls == 2


def test_prediction_interface_constants_remain_unchanged():
    source = (ROOT / "franka_act_intercept_rollout.py").read_text()
    assert 'default="/act/intercept_prediction_chunk_abs_s"' in source
    assert 'default="/act/intercept_prediction_current_abs_s"' in source
    assert "Float64MultiArray" in source and "Float64" in source


def test_reset_clears_sparse_qpos_and_tcp_temporal_history():
    sparse, qpos, tcp = [1], [2], [3]
    clear_sparse_temporal_history(sparse, qpos, tcp)
    assert sparse == [] and qpos == [] and tcp == []


def test_sparse_resources_and_warmup_have_no_dense_allocation():
    source = (ROOT / "franka_act_intercept_rollout.py").read_text()
    assert 'if args.input_modality != "sparse_ball":\n            from cv_bridge import CvBridge' in source
    assert 'warmup_shape = ((1, 3, 4) if self.input_modality == "sparse_ball"' in source
    assert 'create_timer(policy_period_sec(self.fps), self.timer_cb)' in source


def test_checkpoint_tensor_shapes_validate_chunk_action_and_no_backbone():
    import torch

    valid = {
        "model.query_embed.weight": torch.zeros(30, 512),
        "model.action_head.weight": torch.zeros(1, 512),
        "sparse_mean": torch.zeros(4),
        "sparse_std": torch.ones(4),
    }
    validate_sparse_checkpoint_state_dict(valid, 30)
    with pytest.raises(ValueError, match="chunk_size"):
        validate_sparse_checkpoint_state_dict(
            {**valid, "model.query_embed.weight": torch.zeros(29, 512)}, 30
        )
    with pytest.raises(ValueError, match="visual backbone"):
        validate_sparse_checkpoint_state_dict({**valid, "model.backbones.0.x": torch.zeros(1)}, 30)
