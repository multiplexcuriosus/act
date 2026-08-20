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


@pytest.mark.parametrize("source", ("rgb", "event"))
def test_dryrun_argument_parses_both_sources(rollout, source):
    assert _dryrun_parser(rollout).parse_args(["--dryrun", source]).dryrun == source


@pytest.mark.parametrize("argv", (["--dryrun"], ["--dryrun", "depth"]))
def test_dryrun_argument_rejects_missing_and_invalid_values(rollout, argv):
    with pytest.raises(SystemExit):
        _dryrun_parser(rollout).parse_args(argv)


@pytest.mark.parametrize(
    "enabled,dryrun,topic",
    (
        ("rgb", "event", "/openmv_cam/event_tracker/ball_2d_px"),
        ("event", "rgb", "/ball_tracker2/ball_2d_px"),
    ),
)
def test_opposite_sources_are_accepted_and_effective_for_sparse_input(
    rollout, enabled, dryrun, topic
):
    runtime = rollout.resolve_rollout_runtime(enabled, dryrun)
    assert runtime["enabled_source"] == enabled
    assert runtime["effective_source"] == dryrun
    assert resolve_sparse_topic(runtime["effective_source"], None) == topic
    checkpoint_contract = resolve_sparse_checkpoint_contract(
        {
            "sparse_source": dryrun,
            "sparse_feature_dim": 4,
            "sparse_history_length": 3,
            "policy_rate_hz": 30,
            "chunk_size": 30,
            "qpos_history_offsets": [-6, -3, 0],
            "sparse_history_offsets_frames": [-6, -3, 0],
        },
        requested_policy_rate_hz=30,
        requested_chunk_size=30,
        requested_sparse_source=runtime["effective_source"],
    )
    assert checkpoint_contract["sparse_source"] == dryrun


@pytest.mark.parametrize("source", ("rgb", "event"))
def test_same_source_is_rejected(rollout, source):
    with pytest.raises(ValueError, match="opposite"):
        rollout.resolve_rollout_runtime(source, source)


def test_active_endpoints_are_unchanged(rollout):
    runtime = rollout.resolve_rollout_runtime("rgb", None)
    assert runtime["node_fqn"] == "/franka_act_rollout_intercept"
    assert runtime["prediction_topic"] == "/act/intercept_prediction_chunk_abs_s"
    assert runtime["prediction_current_topic"] == "/act/intercept_prediction_current_abs_s"
    assert runtime["reset_service"] == "/act/reset_temporal_aggregation"


@pytest.mark.parametrize("source", ("rgb", "event"))
def test_dryrun_endpoints_are_namespaced_and_never_production(rollout, source):
    enabled = "event" if source == "rgb" else "rgb"
    runtime = rollout.resolve_rollout_runtime(enabled, source)
    prefix = f"/act_dryrun/{source}"
    assert runtime["namespace"] == prefix
    assert runtime["node_fqn"] == f"{prefix}/franka_act_rollout_intercept"
    assert runtime["prediction_topic"] == f"{prefix}/intercept_prediction_chunk_abs_s"
    assert runtime["prediction_current_topic"] == f"{prefix}/intercept_prediction_current_abs_s"
    assert runtime["reset_service"] == f"{prefix}/reset_temporal_aggregation"
    assert runtime["latency_trace_topic"] == f"{prefix}/latency_trace"
    assert runtime["prediction_topic"] != rollout.PRODUCTION_PREDICTION_TOPIC
    assert runtime["prediction_current_topic"] != rollout.PRODUCTION_CURRENT_PREDICTION_TOPIC
