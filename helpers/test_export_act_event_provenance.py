import io
import json

from helpers.export_act_event_provenance import act_event_row, write_rows


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
