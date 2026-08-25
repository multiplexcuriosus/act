#!/usr/bin/env python3
"""Export one CSV row per ACT LatencyTrace from JSON/JSONL records.

Input records may be decoded LatencyTrace dictionaries containing ``detail_json``
or the decoded detail object itself.  This deliberately uses event source time,
not detector-stage latency, as the freshness authority.
"""

import argparse
import csv
import json
from typing import Any, Dict, Iterable, Iterator, TextIO


FIELDS = (
    "policy_tick_timestamp_ns",
    "event_source_timestamp_ns",
    "event_age_sec",
    "event_observation_sequence",
    "event_valid",
    "event_u",
    "event_v",
    "event_input_changed",
    "sparse_input_changed",
    "prediction_current_abs_s",
    "prediction_delta_abs_s",
    "prediction_changed_gt_0_1mm",
)


def act_event_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the stable event/ACT columns from one trace record."""
    detail = record.get("detail_json", record)
    if isinstance(detail, str):
        detail = json.loads(detail)
    if not isinstance(detail, dict):
        raise ValueError("detail_json must decode to an object")
    return {field: detail.get(field) for field in FIELDS}


def iter_json_records(stream: TextIO) -> Iterator[Dict[str, Any]]:
    """Accept a JSON array or newline-delimited JSON objects."""
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
    if isinstance(decoded, list):
        yield from decoded
    else:
        yield decoded


def write_rows(records: Iterable[Dict[str, Any]], stream: TextIO) -> None:
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    for record in records:
        if record.get("stage", "act_rollout") == "act_rollout":
            writer.writerow(act_event_row(record))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", help="LatencyTrace JSON array or JSONL file")
    parser.add_argument("output_csv", help="destination CSV")
    args = parser.parse_args()
    with open(args.input_json, "r", encoding="utf-8") as source:
        records = list(iter_json_records(source))
    with open(args.output_csv, "w", encoding="utf-8", newline="") as target:
        write_rows(records, target)


if __name__ == "__main__":
    main()
