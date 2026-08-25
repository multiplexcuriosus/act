#!/usr/bin/env python3
"""Export and summarize one causal event-provenance row per ACT trace."""

import argparse
import csv
import json
import math
from typing import Any, Dict, Iterable, Iterator, List, TextIO


TRACE_FIELDS = (
    "event_observation_sequence", "event_source_timestamp_ns", "event_age_sec",
    "event_valid", "event_u", "event_v", "event_input_changed",
    "policy_tick_timestamp_ns", "observation_selection_timestamp_ns",
    "tracker_latest_update_id", "tracker_latest_source_packet_id",
    "tracker_latest_source_packet_id_valid", "tracker_latest_timestamp_ns",
    "tracker_latest_receipt_timestamp_ns", "tracker_latest_age_sec",
    "tracker_latest_u", "tracker_latest_v", "tracker_latest_valid",
    "tracker_latest_rejection_reason", "tracker_latest_window_event_count",
    "policy_selected_update_id", "policy_selected_source_packet_id",
    "policy_selected_source_packet_id_valid", "policy_selected_timestamp_ns",
    "policy_selected_receipt_timestamp_ns", "policy_selected_age_sec",
    "policy_selected_u", "policy_selected_v", "policy_selected_valid",
    "policy_selected_is_held", "policy_selected_hold_duration_sec",
    "policy_selected_rejection_reason", "selected_event_source_timestamps_ns",
    "selected_event_update_ids", "selected_event_ages_sec",
    "selected_event_valid_flags", "selected_event_packet_ids",
    "sparse_input_changed", "event_source_changed",
    "event_policy_observation_changed", "prediction_current_abs_s",
    "prediction_delta_abs_s", "prediction_changed_gt_0_1mm",
    "prediction_chunk_size", "temporal_aggregation_mode",
    "temporal_aggregation_contributor_count",
    "temporal_aggregation_effective_age_frames",
)
DERIVED_FIELDS = (
    "latest_tracker_update_changed", "selected_observation_changed",
    "selected_observation_was_held", "source_to_policy_age_ms",
    "callback_transport_delay_ms", "tracker_invalid_latest",
    "prediction_changed_given_selected_change",
)
FIELDS = TRACE_FIELDS + DERIVED_FIELDS


def _detail(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("detail_json", record)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("detail_json must decode to an object")
    return value


def _ms_delta(end, start):
    if end is None or start is None:
        return None
    return (float(end) - float(start)) / 1e6


def act_event_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """Decode stable trace fields and derived causal-analysis columns."""
    detail = _detail(record)
    row = {field: detail.get(field) for field in TRACE_FIELDS}
    # Preserve producer-computed change state; this is exact across the live run.
    row.update({
        "latest_tracker_update_changed": detail.get("latest_tracker_update_changed"),
        "selected_observation_changed": detail.get("event_policy_observation_changed"),
        "selected_observation_was_held": detail.get("policy_selected_is_held"),
        "source_to_policy_age_ms": _ms_delta(
            detail.get("policy_tick_timestamp_ns"),
            detail.get("policy_selected_timestamp_ns")),
        "callback_transport_delay_ms": _ms_delta(
            detail.get("policy_selected_receipt_timestamp_ns"),
            detail.get("policy_selected_timestamp_ns")),
        "tracker_invalid_latest": (
            None if detail.get("tracker_latest_valid") is None
            else not bool(detail.get("tracker_latest_valid"))),
        "prediction_changed_given_selected_change": (
            bool(detail.get("prediction_changed_gt_0_1mm"))
            if detail.get("event_policy_observation_changed") else None),
    })
    for field in ("selected_event_source_timestamps_ns", "selected_event_update_ids",
                  "selected_event_ages_sec", "selected_event_valid_flags",
                  "selected_event_packet_ids"):
        if isinstance(row[field], (list, dict)):
            row[field] = json.dumps(row[field], separators=(",", ":"))
    return row


def iter_json_records(stream: TextIO) -> Iterator[Dict[str, Any]]:
    text = stream.read().strip()
    if not text:
        return
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
        return
    yield from decoded if isinstance(decoded, list) else (decoded,)


def write_rows(records: Iterable[Dict[str, Any]], stream: TextIO) -> None:
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    for record in records:
        if record.get("stage", "act_rollout") == "act_rollout":
            writer.writerow(act_event_row(record))


def _probability(rows, outcome, condition):
    selected = [row for row in rows if bool(row.get(condition))]
    return None if not selected else sum(bool(row.get(outcome)) for row in selected) / len(selected)


def _percentiles(values: List[float]) -> Dict[str, Any]:
    values = sorted(float(value) for value in values
                    if value is not None and math.isfinite(float(value)))
    if not values:
        return {key: None for key in ("p50", "p90", "p95", "p99")}
    def percentile(q):
        index = (len(values) - 1) * q
        low, high = int(math.floor(index)), int(math.ceil(index))
        return values[low] if low == high else values[low] + (values[high] - values[low]) * (index - low)
    return {f"p{int(q * 100)}": percentile(q) for q in (.50, .90, .95, .99)}


def summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    count = len(rows)
    return {
        "tick_count": count,
        "p_selected_changed_given_latest_changed": _probability(
            rows, "selected_observation_changed", "latest_tracker_update_changed"),
        "p_sparse_changed_given_selected_changed": _probability(
            rows, "sparse_input_changed", "selected_observation_changed"),
        "p_prediction_changed_given_sparse_changed": _probability(
            rows, "prediction_changed_gt_0_1mm", "sparse_input_changed"),
        "p_prediction_changed_given_latest_invalid": _probability(
            rows, "prediction_changed_gt_0_1mm", "tracker_invalid_latest"),
        "held_observation_fraction": (None if not count else
                                      sum(bool(r.get("selected_observation_was_held")) for r in rows) / count),
        "invalid_latest_fraction": (None if not count else
                                    sum(bool(r.get("tracker_invalid_latest")) for r in rows) / count),
        "hold_duration_sec_percentiles": _percentiles(
            [r.get("policy_selected_hold_duration_sec") for r in rows]),
        "source_age_ms_percentiles": _percentiles(
            [r.get("source_to_policy_age_ms") for r in rows]),
        "transport_delay_ms_percentiles": _percentiles(
            [r.get("callback_transport_delay_ms") for r in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json")
    parser.add_argument("output_csv")
    parser.add_argument("--summary-json")
    args = parser.parse_args()
    with open(args.input_json, encoding="utf-8") as source:
        records = list(iter_json_records(source))
    rows = [act_event_row(record) for record in records
            if record.get("stage", "act_rollout") == "act_rollout"]
    with open(args.output_csv, "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as target:
            json.dump(summarize_rows(rows), target, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
