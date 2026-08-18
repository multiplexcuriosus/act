"""Runtime-level regressions for sparse policy-clock scheduling."""

import numpy as np

from intercept_rollout_contract import (
    select_qpos_history_at_targets,
    skip_duplicate_source_frame,
)
from sparse_ball import SparsePoint, construct_causal_sparse_history


def test_unchanged_sparse_source_timestamp_does_not_skip_policy_ticks():
    source_stamp_ns = 950_000_000
    attempted_anchor_ns = source_stamp_ns
    decisions = []
    for _policy_stamp_ns in (1_000_000_000, 1_033_333_333):
        duplicate = source_stamp_ns == attempted_anchor_ns
        decisions.append(duplicate and skip_duplicate_source_frame("sparse_ball"))
    assert decisions == [False, False]


def test_dense_rgb_and_event_still_skip_unchanged_source_frames():
    for modality in ("rgb", "event"):
        assert skip_duplicate_source_frame(modality)
        assert (123 == 123) and skip_duplicate_source_frame(modality)


def test_repeated_sparse_ticks_are_causal_and_age_the_cached_observation():
    points = [SparsePoint(0.79, 10, 20), SparsePoint(0.95, 30, 40)]
    first = construct_causal_sparse_history(
        points, 1.00, (-0.2, -0.1, 0.0), 320, 320, 0.20,
    )
    second = construct_causal_sparse_history(
        points, 1.05, (-0.2, -0.1, 0.0), 320, 320, 0.20,
    )
    assert second[-1, 3] > first[-1, 3]
    assert second[-1, 3] == np.float32(0.10)
    assert np.all(second[:, 2] == 1)


def test_sparse_qpos_selection_is_policy_time_causal():
    stamps = [0.79, 0.81, 0.89, 0.91, 0.99, 1.01]
    samples = [np.full(7, index, dtype=np.float32)
               for index in range(len(stamps))]
    history, selected = select_qpos_history_at_targets(
        stamps, samples, (0.8, 0.9, 1.0),
    )
    assert selected == (0.79, 0.89, 0.99)
    np.testing.assert_array_equal(history.reshape(3, 7)[:, 0], [0, 2, 4])
