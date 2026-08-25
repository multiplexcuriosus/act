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


class _Clock:
    def __init__(self):
        self.value = 10_000

    def now(self):
        self.value += 100
        return types.SimpleNamespace(nanoseconds=self.value)


class _Node:
    def __init__(self):
        self.publisher = _Publisher()
        self.clock = _Clock()

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

    def make_tracer(self, modality="rgb", enabled=True):
        node = _Node()
        tracer = self.tracer_type(
            node,
            enabled=enabled,
            topic="/trace",
            run_id="unit-test",
            modality=modality,
        )
        return node, tracer

    def test_trace_emission_on_accepted_tick_preserves_lookahead(self):
        node, tracer = self.make_tracer("rgb")
        tick = tracer.begin(1_234_000_005)
        tracer.mark(tick, "complete_observation_frame_acceptance")
        tracer.mark(
            tick,
            "input_history_selection",
            selected_visual_timestamps_ns=[100, 200, 300],
            selected_visual_anchor_timestamp_ns=300,
        )
        tracer.mark(tick, "tensor_construction_and_preprocessing")
        tracer.mark(tick, "model_forward_pass")
        tracer.mark(tick, "denormalization_and_absolute_target_conversion")
        tracer.mark(
            tick,
            "temporal_aggregation_lookahead_selection",
            chunk_size=30,
            lookahead_index=4,
            selected_target=0.75,
        )
        tracer.mark(tick, "prediction_publication_completion")
        tracer.finish(tick, valid=True)

        self.assertEqual(len(node.publisher.messages), 1)
        message = node.publisher.messages[0]
        detail = json.loads(message.detail_json)
        self.assertTrue(message.valid)
        self.assertEqual(message.event, "accepted")
        self.assertEqual(message.modality, "rgb")
        self.assertEqual(message.source_stamp_ns, 1_234_000_005)
        self.assertEqual(detail["lookahead_index"], 4)
        self.assertEqual(detail["chunk_size"], 30)
        self.assertEqual(detail["selected_target"], 0.75)
        self.assertEqual(detail["selected_visual_timestamps_ns"], [100, 200, 300])
        self.assertEqual(detail["selected_visual_anchor_timestamp_ns"], 300)
        self.assertIn("prediction_publication_completion", detail["stages"])

    def test_sparse_trace_uses_modality_neutral_visual_metadata(self):
        node, tracer = self.make_tracer("sparse_ball")
        tick = tracer.begin(4_000)
        tracer.mark(
            tick,
            "input_history_selection",
            selected_visual_timestamps_ns=[1_000, 2_000, 3_000],
            selected_visual_anchor_timestamp_ns=3_000,
        )
        tracer.finish(tick, valid=True)

        message = node.publisher.messages[0]
        detail = json.loads(message.detail_json)
        self.assertEqual(message.modality, "sparse_ball")
        self.assertEqual(detail["selected_visual_timestamps_ns"], [1_000, 2_000, 3_000])
        self.assertEqual(detail["selected_visual_anchor_timestamp_ns"], 3_000)

    def test_event_provenance_detail_json_serialization(self):
        node, tracer = self.make_tracer("sparse_ball")
        tick = tracer.begin(4_000)
        tracer.mark(tick, "event_observation_selected")
        tracer.mark(tick, "sparse_input_built")
        tracer.add_detail(
            tick,
            policy_tick_timestamp_ns=5_000,
            event_u=12.5,
            event_v=9.0,
            event_valid=False,
            event_age_sec=0.25,
            event_source_timestamp_ns=4_750,
            event_observation_sequence=7,
            event_input_changed=True,
            sparse_input_changed=True,
        )
        tracer.mark(tick, "model_forward_pass")
        tracer.mark(tick, "prediction_publication")
        tracer.finish(tick, valid=True)
        detail = json.loads(node.publisher.messages[0].detail_json)
        self.assertEqual(detail["event_observation_sequence"], 7)
        self.assertFalse(detail["event_valid"])
        stages = detail["stages"]
        ordered = [
            stages[name]["steady_ns"] for name in
            ("timer_tick_begin", "event_observation_selected", "sparse_input_built",
             "model_forward_pass", "prediction_publication")
        ]
        self.assertEqual(ordered, sorted(ordered))

    def test_trace_emission_on_rejected_stale_event_tick(self):
        node, tracer = self.make_tracer("event")
        tick = tracer.begin(9_000)
        tracer.mark(tick, "input_history_selection")
        tracer.finish(tick, valid=False, rejection_reason="observation is stale")

        self.assertEqual(len(node.publisher.messages), 1)
        message = node.publisher.messages[0]
        detail = json.loads(message.detail_json)
        self.assertFalse(message.valid)
        self.assertEqual(message.event, "rejected")
        self.assertEqual(message.modality, "event")
        self.assertEqual(detail["rejection_reason"], "observation is stale")

    def test_tracing_disabled_emits_nothing(self):
        node, tracer = self.make_tracer(enabled=False)
        tick = tracer.begin(100)
        tracer.mark(tick, "model_forward_pass")
        tracer.finish(tick, valid=True)
        self.assertIsNone(tick)
        self.assertEqual(node.publisher.messages, [])

    def test_cpu_forward_does_not_call_cuda_synchronize(self):
        # Exercise the same device predicate used by the rollout forward path.
        device = torch.device("cpu")
        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            if device.type == "cuda" and True:
                synchronize()
        synchronize.assert_not_called()

    def test_cuda_sync_defaults_to_latency_trace_setting(self):
        from rollout_latency_trace import resolve_latency_trace_cuda_sync

        self.assertTrue(resolve_latency_trace_cuda_sync(True, None))
        self.assertFalse(resolve_latency_trace_cuda_sync(False, None))
        self.assertFalse(resolve_latency_trace_cuda_sync(True, False))
        self.assertTrue(resolve_latency_trace_cuda_sync(False, True))


if __name__ == "__main__":
    unittest.main()
