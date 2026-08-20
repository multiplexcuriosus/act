"""Focused contracts for sparse comparison rollout configuration."""

import argparse
import importlib
import sys
import types

import pytest

from sparse_ball import resolve_sparse_checkpoint_contract, resolve_sparse_topic


def _load_rollout(monkeypatch):
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

    services = types.ModuleType("std_srvs.srv")
    services.Trigger = type(
        "Trigger", (),
        {"Request": type("Request", (), {}), "Response": type("Response", (), {})},
    )
    monkeypatch.setitem(sys.modules, "std_srvs.srv", services)

    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = object
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_qos.QoSProfile = type("QoSProfile", (), {})
    rclpy_qos.ReliabilityPolicy = type("ReliabilityPolicy", (), {"BEST_EFFORT": object()})
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", rclpy_node)
    monkeypatch.setitem(sys.modules, "rclpy.qos", rclpy_qos)

    policy = types.ModuleType("policy")
    policy.ACTPolicy = type("ACTPolicy", (), {})
    monkeypatch.setitem(sys.modules, "policy", policy)
    sys.modules.pop("franka_act_intercept_rollout", None)
    return importlib.import_module("franka_act_intercept_rollout")


@pytest.fixture
def rollout(monkeypatch):
    return _load_rollout(monkeypatch)


def _dryrun_parser(rollout):
    parser = argparse.ArgumentParser()
    rollout.add_dryrun_argument(parser)
    return parser


def test_dryrun_argument_is_boolean(rollout):
    parser = _dryrun_parser(rollout)
    assert parser.parse_args([]).dryrun is False
    assert parser.parse_args(["--dryrun"]).dryrun is True


def test_dryrun_argument_rejects_a_value(rollout):
    with pytest.raises(SystemExit):
        _dryrun_parser(rollout).parse_args(["--dryrun", "rgb"])


@pytest.mark.parametrize(
    "source,topic",
    (
        ("rgb", "/ball_tracker2/ball_2d_px"),
        ("event", "/openmv_cam/event_tracker/ball_2d_px"),
    ),
)
def test_dryrun_keeps_selected_source_for_input_and_checkpoint(rollout, source, topic):
    runtime = rollout.resolve_rollout_runtime(source, True)
    assert runtime["sparse_source"] == source
    assert resolve_sparse_topic(runtime["sparse_source"], None) == topic
    checkpoint_contract = resolve_sparse_checkpoint_contract(
        {
            "sparse_source": source,
            "sparse_feature_dim": 4,
            "sparse_history_length": 3,
            "policy_rate_hz": 30,
            "chunk_size": 30,
            "qpos_history_offsets": [-6, -3, 0],
            "sparse_history_offsets_frames": [-6, -3, 0],
        },
        requested_policy_rate_hz=30,
        requested_chunk_size=30,
        requested_sparse_source=runtime["sparse_source"],
    )
    assert checkpoint_contract["sparse_source"] == source


@pytest.mark.parametrize("source", ("rgb", "event"))
def test_active_endpoints_are_unchanged(rollout, source):
    runtime = rollout.resolve_rollout_runtime(source, False)
    assert runtime["node_fqn"] == "/franka_act_rollout_intercept"
    assert runtime["prediction_topic"] == "/act/intercept_prediction_chunk_abs_s"
    assert runtime["prediction_current_topic"] == "/act/intercept_prediction_current_abs_s"
    assert runtime["reset_service"] == "/act/reset_temporal_aggregation"


@pytest.mark.parametrize("source", ("rgb", "event"))
def test_dryrun_endpoints_are_namespaced_and_never_production(rollout, source):
    runtime = rollout.resolve_rollout_runtime(source, True)
    prefix = f"/act_dryrun/{source}"
    assert runtime["namespace"] == prefix
    assert runtime["node_fqn"] == f"{prefix}/franka_act_rollout_intercept"
    assert runtime["prediction_topic"] == f"{prefix}/intercept_prediction_chunk_abs_s"
    assert runtime["prediction_current_topic"] == f"{prefix}/intercept_prediction_current_abs_s"
    assert runtime["reset_service"] == f"{prefix}/reset_temporal_aggregation"
    assert runtime["latency_trace_topic"] == f"{prefix}/intercept_trace/act_rollout"
    assert runtime["prediction_topic"] != rollout.PRODUCTION_PREDICTION_TOPIC
    assert runtime["prediction_current_topic"] != rollout.PRODUCTION_CURRENT_PREDICTION_TOPIC


def test_active_explicit_topics_are_preserved(rollout):
    runtime = rollout.resolve_rollout_runtime(
        "rgb",
        False,
        prediction_topic="/custom/chunk",
        prediction_current_topic="/custom/current",
        reset_service="/custom/reset",
        latency_trace_topic="/custom/trace",
    )
    assert runtime["prediction_topic"] == "/custom/chunk"
    assert runtime["prediction_current_topic"] == "/custom/current"
    assert runtime["reset_service"] == "/custom/reset"
    assert runtime["latency_trace_topic"] == "/custom/trace"


@pytest.mark.parametrize("source", ("rgb", "event"))
def test_no_opposite_source_validation_exists(rollout, source):
    runtime = rollout.resolve_rollout_runtime(source, True)
    assert runtime["sparse_source"] == source
