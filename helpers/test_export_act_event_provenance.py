import io
import json

from helpers.export_act_event_provenance import act_event_row, summarize_rows, write_rows


def test_export_one_row_per_act_trace():
    detail = {
        "policy_tick_timestamp_ns": 100,
        "event_source_timestamp_ns": 90,
        "event_age_sec": 1e-8,
        "event_observation_sequence": 3,
        "event_valid": True,
        "event_u": 12.0,
        "event_v": 13.0,
        "event_input_changed": True,
        "sparse_input_changed": True,
        "prediction_current_abs_s": 0.4,
        "prediction_changed_gt_0_1mm": False,
    }
    row = act_event_row({"detail_json": json.dumps(detail)})
    assert row["event_observation_sequence"] == 3
    assert row["prediction_current_abs_s"] == 0.4
    output = io.StringIO()
    write_rows([{"stage": "act_rollout", "detail_json": json.dumps(detail)}], output)
    assert len(output.getvalue().splitlines()) == 2


def test_typed_csv_derived_fields_and_summary():
    detail = {
        "policy_tick_timestamp_ns": 1_000_000_000,
        "tracker_latest_timestamp_ns": 950_000_000,
        "tracker_latest_valid": False,
        "tracker_latest_receipt_timestamp_ns": 970_000_000,
        "policy_selected_timestamp_ns": 900_000_000,
        "policy_selected_receipt_timestamp_ns": 920_000_000,
        "policy_selected_is_held": True,
        "policy_selected_hold_duration_sec": 0.1,
        "latest_tracker_update_changed": True,
        "event_policy_observation_changed": True,
        "sparse_input_changed": True,
        "prediction_changed_gt_0_1mm": True,
        "selected_event_update_ids": [1, 2, 3],
    }
    row = act_event_row({"detail_json": json.dumps(detail)})
    assert row["source_to_policy_age_ms"] == 100.0
    assert row["callback_transport_delay_ms"] == 20.0
    assert row["tracker_invalid_latest"]
    assert row["selected_event_update_ids"] == "[1,2,3]"
    summary = summarize_rows([row])
    assert summary["p_selected_changed_given_latest_changed"] == 1.0
    assert summary["p_sparse_changed_given_selected_changed"] == 1.0
    assert summary["p_prediction_changed_given_sparse_changed"] == 1.0
    assert summary["p_prediction_changed_given_latest_invalid"] == 1.0
