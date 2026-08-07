"""Opt-in ACT rollout tracing with aggregate ticks and causal milestone records."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


def image_source_stamp_ns(message: Any) -> int:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return 0
    try:
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return 0
    return value if value > 0 else 0


@dataclass
class TickTrace:
    sequence: int
    source_stamp_ns: int
    receipt_ros_stamp_ns: int
    start_ros_stamp_ns: int
    start_steady_ns: int
    last_steady_ns: int
    stages: Dict[str, Dict[str, int]] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)
    source_modality: str = "unknown"
    history_source_stamp_ns: List[int] = field(default_factory=list)
    policy_step_id: str = ""
    causal_id: str = ""
    last_event_id: str = ""


class RolloutLatencyTracer:
    """Collect tick milestones and publish exactly one LatencyTrace at completion."""

    def __init__(self, node: Any, *, enabled: bool, topic: str, run_id: str, modality: str) -> None:
        self._node = node
        self._run_id = str(run_id)
        self._modality = str(modality)
        self._sequence = 0
        self._record_sequence = 0
        self._policy_step_sequence = 0
        self._command_sequence = 0
        self._event_sequence = 0
        self._publisher = None
        if enabled:
            from intercept_latency_monitor.msg import LatencyTrace
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

            self._message_type = LatencyTrace
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self._publisher = node.create_publisher(LatencyTrace, topic, qos)

    @property
    def enabled(self) -> bool:
        return self._publisher is not None

    def begin(
        self,
        source_stamp_ns: int,
        *,
        source_modality: Optional[str] = None,
        history_source_stamp_ns: Optional[Sequence[int]] = None,
    ) -> Optional[TickTrace]:
        if self._publisher is None:
            return None
        self._sequence += 1
        ros_ns = int(self._node.get_clock().now().nanoseconds)
        steady_ns = time.monotonic_ns()
        history = [int(value) for value in (history_source_stamp_ns or [])]
        if not history and int(source_stamp_ns) > 0:
            history = [int(source_stamp_ns)]
        trace = TickTrace(
            sequence=self._sequence,
            source_stamp_ns=int(source_stamp_ns),
            receipt_ros_stamp_ns=ros_ns,
            start_ros_stamp_ns=ros_ns,
            start_steady_ns=steady_ns,
            last_steady_ns=steady_ns,
            source_modality=str(source_modality or self._modality),
            history_source_stamp_ns=history,
        )
        self.mark(trace, "timer_tick_begin")
        return trace

    def _new_event_id(self) -> str:
        self._event_sequence += 1
        return "rollout_event_%08d" % self._event_sequence

    def policy_input_accepted(
        self, trace: Optional[TickTrace], history_source_stamp_ns: Sequence[int]
    ) -> Optional[str]:
        if trace is None:
            return None
        history = [int(value) for value in history_source_stamp_ns]
        self._policy_step_sequence += 1
        trace.policy_step_id = "policy_step_%08d" % self._policy_step_sequence
        trace.causal_id = trace.policy_step_id
        trace.history_source_stamp_ns = history
        valid_history = [value for value in history if value > 0]
        if valid_history:
            trace.source_stamp_ns = valid_history[-1]
        self.emit(trace, "policy_input_accepted", execution_path="pcu_act_rollout")
        return trace.policy_step_id

    def emit(self, trace: Optional[TickTrace], event: str, **detail: Any) -> Optional[str]:
        if self._publisher is None or trace is None:
            return None
        ros_ns = int(self._node.get_clock().now().nanoseconds)
        steady_ns = time.monotonic_ns()
        event_id = self._new_event_id()
        history = list(trace.history_source_stamp_ns)
        valid_history = [value for value in history if value > 0]
        payload: Dict[str, Any] = {
            "event_id": event_id,
            "causal_id": trace.causal_id,
            "parent_event_id": trace.last_event_id,
            "source_modality": trace.source_modality,
            "source_oldest_stamp_ns": min(valid_history) if valid_history else 0,
            "source_newest_stamp_ns": max(valid_history) if valid_history else 0,
            "history_source_stamp_ns": history,
            "history_frame_count": len(history),
            "policy_step_id": trace.policy_step_id,
            "execution_path": "pcu_act_rollout",
            "output_ros_stamp_ns": ros_ns,
        }
        payload.update(detail)
        self._record_sequence += 1
        msg = self._message_type()
        msg.run_id = self._run_id
        msg.stage = "rollout"
        msg.event = str(event)
        msg.modality = trace.source_modality
        msg.node_name = self._node.get_name()
        msg.sequence = self._record_sequence
        msg.parent_sequence = 0
        msg.source_stamp_ns = payload["source_newest_stamp_ns"]
        msg.receipt_ros_stamp_ns = trace.receipt_ros_stamp_ns
        msg.start_ros_stamp_ns = ros_ns
        msg.end_ros_stamp_ns = ros_ns
        msg.start_steady_ns = steady_ns
        msg.end_steady_ns = steady_ns
        msg.valid = True
        msg.scalar_value = float(payload.get("target_s", 0.0) or 0.0)
        msg.detail_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self._publisher.publish(msg)
        trace.last_event_id = event_id
        return event_id

    def target_s_published(self, trace: Optional[TickTrace], target_s: float, **detail: Any) -> Optional[str]:
        if trace is None:
            return None
        self._command_sequence += 1
        command_id = "rollout_command_%08d" % self._command_sequence
        self.emit(trace, "target_s_published", command_id=command_id,
                  target_s=float(target_s), **detail)
        return command_id

    def mark(self, trace: Optional[TickTrace], stage: str, **detail: Any) -> None:
        if trace is None:
            return
        now_ns = time.monotonic_ns()
        trace.stages[str(stage)] = {
            "steady_ns": now_ns,
            "since_previous_ns": now_ns - trace.last_steady_ns,
            "since_tick_begin_ns": now_ns - trace.start_steady_ns,
        }
        trace.last_steady_ns = now_ns
        trace.detail.update(detail)

    def add_detail(self, trace: Optional[TickTrace], **detail: Any) -> None:
        if trace is not None:
            trace.detail.update(detail)

    def finish(self, trace: Optional[TickTrace], *, valid: bool, rejection_reason: str = "") -> None:
        if self._publisher is None or trace is None:
            return
        end_ros_ns = int(self._node.get_clock().now().nanoseconds)
        end_steady_ns = time.monotonic_ns()
        aggregate_event_id = self._new_event_id()
        valid_history = [value for value in trace.history_source_stamp_ns if value > 0]
        detail = dict(trace.detail)
        detail.update({
            "event_id": aggregate_event_id,
            "parent_event_id": trace.last_event_id,
            "tick_sequence": trace.sequence,
            "valid": bool(valid),
            "rejection_reason": str(rejection_reason),
            "stages": trace.stages,
            "source_modality": trace.source_modality,
            "history_source_stamp_ns": trace.history_source_stamp_ns,
            "source_oldest_stamp_ns": min(valid_history) if valid_history else 0,
            "source_newest_stamp_ns": max(valid_history) if valid_history else 0,
            "history_frame_count": len(trace.history_source_stamp_ns),
            "policy_step_id": trace.policy_step_id,
            "causal_id": trace.causal_id,
            "execution_path": "pcu_act_rollout",
            "output_ros_stamp_ns": end_ros_ns,
        })
        msg = self._message_type()
        msg.run_id = self._run_id
        msg.stage = "act_rollout"
        msg.event = "accepted" if valid else "rejected"
        msg.modality = self._modality
        msg.node_name = self._node.get_name()
        msg.sequence = trace.sequence
        msg.parent_sequence = 0
        msg.source_stamp_ns = trace.source_stamp_ns
        msg.receipt_ros_stamp_ns = trace.receipt_ros_stamp_ns
        msg.start_ros_stamp_ns = trace.start_ros_stamp_ns
        msg.end_ros_stamp_ns = end_ros_ns
        msg.start_steady_ns = trace.start_steady_ns
        msg.end_steady_ns = end_steady_ns
        msg.valid = bool(valid)
        selected_target = detail.get("selected_target")
        msg.scalar_value = float(selected_target) if selected_target is not None else 0.0
        msg.detail_json = json.dumps(detail, separators=(",", ":"), sort_keys=True)
        self._publisher.publish(msg)
