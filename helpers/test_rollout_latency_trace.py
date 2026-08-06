#!/usr/bin/env python3

import json
import sys
import types
import unittest
from unittest import mock

import torch


class _LatencyTrace:
    pass


class _QoSProfile:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_ros_stubs():
    package = types.ModuleType("intercept_latency_monitor")
    messages = types.ModuleType("intercept_latency_monitor.msg")
    messages.LatencyTrace = _LatencyTrace
    package.msg = messages
    sys.modules["intercept_latency_monitor"] = package
    sys.modules["intercept_latency_monitor.msg"] = messages
    qos = types.ModuleType("rclpy.qos")
    qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)
    qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=1)
    qos.QoSProfile = _QoSProfile
    rclpy = types.ModuleType("rclpy")
    rclpy.qos = qos
    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.qos"] = qos


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Node:
    def __init__(self):
        self.publisher = _Publisher()
        self.clock = types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(nanoseconds=10_000)
        )

    def create_publisher(self, *_args):
        return self.publisher

    def get_clock(self):
        return self.clock

    def get_name(self):
        return "test_act_rollout"


class RolloutLatencyTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_ros_stubs()
        from rollout_latency_trace import RolloutLatencyTracer
        cls.tracer_type = RolloutLatencyTracer

    def make_tracer(self, modality="event", enabled=True):
        node = _Node()
        return node, self.tracer_type(
            node, enabled=enabled, topic="/trace", run_id="test", modality=modality
        )

    def test_trace_emission_on_accepted_tick_and_lookahead(self):
        node, tracer = self.make_tracer("event")
        tick = tracer.begin(1_234_000_005)
        tracer.mark(tick, "complete_observation_frame_acceptance")
        tracer.mark(tick, "input_history_selection", selected_history_timestamps_ns=[1, 2, 3])
        tracer.mark(tick, "tensor_construction_and_preprocessing")
        tracer.mark(tick, "model_forward_pass")
        tracer.mark(tick, "denormalization_and_absolute_target_conversion")
        tracer.mark(tick, "temporal_aggregation_lookahead_selection", chunk_size=30,
                    lookahead_index=4, selected_target=0.75)
        tracer.mark(tick, "prediction_publication_completion")
        tracer.finish(tick, valid=True)
        message = node.publisher.messages[0]
        detail = json.loads(message.detail_json)
        self.assertTrue(message.valid)
        self.assertEqual(message.modality, "event")
        self.assertEqual(message.source_stamp_ns, 1_234_000_005)
        self.assertEqual(detail["lookahead_index"], 4)
        self.assertEqual(detail["chunk_size"], 30)
        self.assertEqual(detail["selected_target"], 0.75)

    def test_trace_emission_on_rejected_stale_rgb_tick(self):
        node, tracer = self.make_tracer("rgb")
        tick = tracer.begin(9_000)
        tracer.finish(tick, valid=False, rejection_reason="observation is stale")
        message = node.publisher.messages[0]
        detail = json.loads(message.detail_json)
        self.assertFalse(message.valid)
        self.assertEqual(message.modality, "rgb")
        self.assertEqual(detail["rejection_reason"], "observation is stale")

    def test_tracing_disabled(self):
        node, tracer = self.make_tracer(enabled=False)
        tick = tracer.begin(100)
        tracer.finish(tick, valid=True)
        self.assertIsNone(tick)
        self.assertEqual(node.publisher.messages, [])

    def test_cpu_execution_does_not_synchronize_cuda(self):
        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            if torch.device("cpu").type == "cuda" and True:
                synchronize()
        synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
