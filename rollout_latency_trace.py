"""Opt-in, one-record-per-tick latency tracing for ACT interception rollout."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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


class RolloutLatencyTracer:
    """Collect tick milestones and publish exactly one LatencyTrace at completion."""

    def __init__(self, node: Any, *, enabled: bool, topic: str, run_id: str, modality: str) -> None:
        self._node = node
        self._run_id = str(run_id)
        self._modality = str(modality)
        self._sequence = 0
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

    def begin(self, source_stamp_ns: int) -> Optional[TickTrace]:
        if self._publisher is None:
            return None
        self._sequence += 1
        ros_ns = int(self._node.get_clock().now().nanoseconds)
        steady_ns = time.monotonic_ns()
        trace = TickTrace(
            sequence=self._sequence,
            source_stamp_ns=int(source_stamp_ns),
            receipt_ros_stamp_ns=ros_ns,
            start_ros_stamp_ns=ros_ns,
            start_steady_ns=steady_ns,
            last_steady_ns=steady_ns,
        )
        self.mark(trace, "timer_tick_begin")
        return trace

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

    def finish(self, trace: Optional[TickTrace], *, valid: bool, rejection_reason: str = "") -> None:
        if self._publisher is None or trace is None:
            return
        end_ros_ns = int(self._node.get_clock().now().nanoseconds)
        end_steady_ns = time.monotonic_ns()
        detail = dict(trace.detail)
        detail.update({
            "tick_sequence": trace.sequence,
            "valid": bool(valid),
            "rejection_reason": str(rejection_reason),
            "stages": trace.stages,
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
