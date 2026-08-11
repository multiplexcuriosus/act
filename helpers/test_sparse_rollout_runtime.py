import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("rclpy")
from franka_act_intercept_rollout import (SPARSE_TOPICS, clear_sparse_temporal_history,
                                          create_visual_subscription,
                                          resolve_sparse_topic,
                                          rollout_subscription_types,
                                          skip_duplicate_source_frame)
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


def test_prediction_interface_constants_remain_unchanged():
    source = (ROOT / "franka_act_intercept_rollout.py").read_text()
    assert 'default="/act/intercept_prediction_chunk_abs_s"' in source
    assert 'default="/act/intercept_prediction_current_abs_s"' in source
    assert "Float64MultiArray" in source and "Float64" in source


def test_reset_clears_sparse_qpos_and_tcp_temporal_history():
    sparse, qpos, tcp = [1], [2], [3]
    clear_sparse_temporal_history(sparse, qpos, tcp)
    assert sparse == [] and qpos == [] and tcp == []
