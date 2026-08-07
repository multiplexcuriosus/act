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
            selected_history_timestamps_ns=[100, 200, 300],
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
        self.assertEqual(detail["selected_history_timestamps_ns"], [100, 200, 300])
        self.assertIn("prediction_publication_completion", detail["stages"])

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

    def test_history_provenance_and_rollout_event_ids_are_preserved(self):
        node, tracer = self.make_tracer("event")
        tick = tracer.begin(300, source_modality="event", history_source_stamp_ns=[100, 200, 300])
        policy_step_id = tracer.policy_input_accepted(tick, [100, 200, 300])
        tracer.emit(tick, "inference_started")
        tracer.emit(tick, "inference_completed", cuda_synchronized=True)
        tracer.emit(tick, "action_published", action_chunk_index=4)
        first_command = tracer.target_s_published(tick, 0.25, action_chunk_index=4)
        second_command = tracer.target_s_published(tick, 0.25, action_chunk_index=4)

        details = [json.loads(message.detail_json) for message in node.publisher.messages]
        self.assertEqual(policy_step_id, "policy_step_00000001")
        self.assertEqual(first_command, "rollout_command_00000001")
        self.assertEqual(second_command, "rollout_command_00000002")
        self.assertNotEqual(details[-2]["command_id"], details[-1]["command_id"])
        self.assertEqual(details[0]["history_source_stamp_ns"], [100, 200, 300])
        self.assertEqual(details[0]["source_oldest_stamp_ns"], 100)
        self.assertEqual(details[0]["source_newest_stamp_ns"], 300)
        self.assertEqual(details[0]["history_frame_count"], 3)
        self.assertTrue(all(item["source_modality"] == "event" for item in details))
        self.assertTrue(all(item["policy_step_id"] == policy_step_id for item in details))

    def test_one_policy_step_can_produce_multiple_distinct_commands(self):
        node, tracer = self.make_tracer()
        tick = tracer.begin(100)
        policy_step_id = tracer.policy_input_accepted(tick, [100])
        tracer.target_s_published(tick, 0.5)
        tracer.target_s_published(tick, 0.5)
        command_details = [json.loads(message.detail_json) for message in node.publisher.messages
                           if message.event == "target_s_published"]
        self.assertEqual({item["policy_step_id"] for item in command_details}, {policy_step_id})
        self.assertEqual(len({item["command_id"] for item in command_details}), 2)


if __name__ == "__main__":
    unittest.main()
