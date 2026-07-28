#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(__file__)
ACT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ACT_ROOT not in sys.path:
    sys.path.insert(0, ACT_ROOT)

from intercept_rollout_contract import (  # noqa: E402
    EXPECTED_INTERCEPT_METADATA,
    AggregationSelection,
    TemporalAbsoluteAggregator,
    absolute_s_from_anchor,
    build_qpos_history,
    build_rgb_history_tensor,
    compute_history_indices,
    denormalize_delta_chunk,
    resolve_temporal_agg_mode,
    select_sync_observation,
    validate_anchor_freshness,
    validate_intercept_stats_and_config,
)


class InterceptRolloutContractTests(unittest.TestCase):
    def make_stats(self):
        stats = dict(EXPECTED_INTERCEPT_METADATA)
        stats.update(
            {
                "qpos_mean": np.zeros(21, dtype=np.float32),
                "qpos_std": np.ones(21, dtype=np.float32),
                "action_mean": np.asarray([0.1], dtype=np.float32),
                "action_std": np.asarray([0.2], dtype=np.float32),
            }
        )
        return stats

    def make_policy_config(self):
        return {
            "state_dim": 21,
            "action_dim": 1,
            "num_queries": 30,
            "rgb_history_frames": 3,
            "image_channels": 9,
            "use_bce_last_action_dim": False,
        }

    def test_history_indices(self):
        self.assertEqual(compute_history_indices(10, (-6, -3, 0)), [4, 7, 10])

    def test_rgb_concat_order_and_shape(self):
        frame_a = np.full((4, 4, 3), 10, dtype=np.uint8)
        frame_b = np.full((4, 4, 3), 20, dtype=np.uint8)
        frame_c = np.full((4, 4, 3), 30, dtype=np.uint8)
        image = build_rgb_history_tensor([frame_a, frame_b, frame_c], image_size=4)

        self.assertEqual(image.shape, (1, 1, 9, 4, 4))
        channel_means = image[0, 0].reshape(3, 3, 4, 4).mean(axis=(1, 2, 3)) * 255.0
        np.testing.assert_allclose(channel_means, np.asarray([10.0, 20.0, 30.0]), atol=0.5)

    def test_qpos_history_order_and_shape(self):
        q0 = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
        q1 = np.asarray([8, 9, 10, 11, 12, 13, 14], dtype=np.float32)
        q2 = np.asarray([15, 16, 17, 18, 19, 20, 21], dtype=np.float32)
        q_hist = build_qpos_history([q0, q1, q2])

        self.assertEqual(q_hist.shape, (21,))
        np.testing.assert_allclose(q_hist[:7], q0)
        np.testing.assert_allclose(q_hist[7:14], q1)
        np.testing.assert_allclose(q_hist[14:], q2)

    def test_timestamp_selection_with_delays(self):
        rgb_ts = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        joint_ts = [0.95, 1.15, 1.35, 1.55]
        joint_q = [np.full(7, fill_value=i, dtype=np.float32) for i in range(len(joint_ts))]
        tcp_ts = [0.9, 1.0, 1.45, 1.59]
        tcp_vals = [0.0, 0.1, 0.2, 0.25]

        sync = select_sync_observation(
            rgb_timestamps=rgb_ts,
            joint_timestamps=joint_ts,
            joint_qpos_samples=joint_q,
            tcp_timestamps=tcp_ts,
            tcp_values=tcp_vals,
            history_offsets=(-6, -3, 0),
        )

        self.assertEqual(sync.history_indices, (0, 3, 6))
        self.assertEqual(sync.rgb_timestamps, (1.0, 1.3, 1.6))
        self.assertEqual(sync.qpos_timestamps, (0.95, 1.15, 1.55))
        self.assertAlmostEqual(sync.anchor_tcp_s, 0.25)
        self.assertAlmostEqual(sync.anchor_tcp_s_timestamp, 1.59)

    def test_chunk_denormalization(self):
        norm = np.asarray([0.0, 1.0, -1.0], dtype=np.float32)
        out = denormalize_delta_chunk(
            norm,
            action_mean=np.asarray([0.1], dtype=np.float32),
            action_std=np.asarray([0.2], dtype=np.float32),
        )
        np.testing.assert_allclose(out, np.asarray([0.1, 0.3, -0.1], dtype=np.float32), atol=1e-6)

    def test_anchor_plus_delta_conversion(self):
        delta = np.asarray([-0.2, 0.0, 0.3], dtype=np.float32)
        absolute = absolute_s_from_anchor(1.5, delta)
        np.testing.assert_allclose(absolute, np.asarray([1.3, 1.5, 1.8], dtype=np.float32), atol=1e-6)

    def test_no_bce_contract(self):
        stats = self.make_stats()
        config = self.make_policy_config()
        config["use_bce_last_action_dim"] = True
        with self.assertRaises(ValueError):
            validate_intercept_stats_and_config(stats, config, expected_chunk_size=30)

    def test_stale_anchor_rejected(self):
        with self.assertRaises(ValueError):
            validate_anchor_freshness(
                anchor_timestamp=1.0,
                observation_timestamp=1.3,
                now_timestamp=1.31,
                max_anchor_age_sec=0.1,
                max_observation_age_sec=1.0,
            )

    def test_legacy_stats_fail_fast(self):
        stats = self.make_stats()
        stats["action_dim"] = 2
        with self.assertRaises(ValueError):
            validate_intercept_stats_and_config(stats, self.make_policy_config(), expected_chunk_size=30)

    def test_temporal_aggregation_in_absolute_coordinates(self):
        agg = TemporalAbsoluteAggregator(chunk_size=5, decay=0.0)
        agg.add_prediction(0, np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
        agg.add_prediction(1, np.asarray([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float32))

        self.assertAlmostEqual(agg.value_for_step(1), (2.0 + 10.0) / 2.0)
        self.assertAlmostEqual(agg.value_for_step(2), (3.0 + 11.0) / 2.0)

    def test_default_constructor_selects_full_mode(self):
        agg = TemporalAbsoluteAggregator(chunk_size=5)
        self.assertEqual(agg.mode, "full")
        self.assertEqual(agg.lookahead_steps, 0)

    def test_lookahead_zero_preserves_existing_behavior_for_all_modes(self):
        chunk_old = np.asarray([100, 101, 102, 103, 104, 105], dtype=np.float32)
        chunk_new = np.asarray([200, 201, 202, 203, 204, 205], dtype=np.float32)
        current_step = 4

        full = TemporalAbsoluteAggregator(chunk_size=6, mode="full", decay=0.0, lookahead_steps=0)
        latest = TemporalAbsoluteAggregator(chunk_size=6, mode="latest", lookahead_steps=0)
        recent = TemporalAbsoluteAggregator(
            chunk_size=6,
            mode="recent",
            recent_window=2,
            recent_half_life=1.0,
            lookahead_steps=0,
        )

        full.add_prediction(3, chunk_old)
        full.add_prediction(4, chunk_new)
        latest.add_prediction(3, chunk_old)
        latest.add_prediction(4, chunk_new)
        recent.add_prediction(3, chunk_old)
        recent.add_prediction(4, chunk_new)

        # L=0 uses token index = current_step - source_step.
        self.assertAlmostEqual(latest.selection_for_step(current_step).value, 200.0)
        self.assertAlmostEqual(full.selection_for_step(current_step).value, (101.0 + 200.0) / 2.0)
        recent_expected = (101.0 * 0.5 + 200.0 * 1.0) / 1.5
        self.assertAlmostEqual(recent.selection_for_step(current_step).value, recent_expected)

    def test_full_mode_matches_legacy_weighted_result_exactly(self):
        agg = TemporalAbsoluteAggregator(chunk_size=8, mode="full", decay=0.25)
        chunk0 = np.asarray([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.float32)
        chunk1 = np.asarray([20, 21, 22, 23, 24, 25, 26, 27], dtype=np.float32)
        chunk2 = np.asarray([30, 31, 32, 33, 34, 35, 36, 37], dtype=np.float32)
        agg.add_prediction(0, chunk0)
        agg.add_prediction(1, chunk1)
        agg.add_prediction(2, chunk2)

        current_step = 3
        values_oldest_to_newest = np.asarray([13.0, 22.0, 31.0], dtype=np.float64)
        legacy_weights = np.exp(-0.25 * np.arange(3, dtype=np.float64))
        legacy_weights = legacy_weights / np.sum(legacy_weights)
        expected = float(np.sum(values_oldest_to_newest * legacy_weights))

        selection = agg.selection_for_step(current_step)
        self.assertIsInstance(selection, AggregationSelection)
        self.assertIsNotNone(selection)
        self.assertAlmostEqual(selection.value, expected, places=12)
        self.assertAlmostEqual(agg.value_for_step(current_step), expected, places=12)

    def test_full_mode_retains_oldest_first_weight_ordering(self):
        agg = TemporalAbsoluteAggregator(chunk_size=6, mode="full", decay=2.0)
        agg.add_prediction(0, np.asarray([1000, 1001, 1002, 1003, 1004, 1005], dtype=np.float32))
        agg.add_prediction(1, np.asarray([100, 101, 102, 103, 104, 105], dtype=np.float32))
        agg.add_prediction(2, np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32))

        selection = agg.selection_for_step(3)
        self.assertIsNotNone(selection)
        weights = np.exp(-2.0 * np.arange(3, dtype=np.float64))
        weights = weights / np.sum(weights)
        expected = float(np.sum(np.asarray([1003.0, 102.0, 2.0], dtype=np.float64) * weights))
        self.assertAlmostEqual(selection.value, expected, places=12)
        self.assertGreater(selection.value, 800.0)

    def test_latest_returns_only_newest_step_aligned_contribution(self):
        agg = TemporalAbsoluteAggregator(chunk_size=8, mode="latest")
        agg.add_prediction(5, np.asarray([50, 51, 52, 53, 54, 55, 56, 57], dtype=np.float32))
        agg.add_prediction(6, np.asarray([60, 61, 62, 63, 64, 65, 66, 67], dtype=np.float32))
        agg.add_prediction(7, np.asarray([70, 71, 72, 73, 74, 75, 76, 77], dtype=np.float32))

        selection = agg.selection_for_step(7)
        self.assertIsNotNone(selection)
        self.assertAlmostEqual(selection.value, 70.0)
        self.assertEqual(selection.contributor_count, 1)
        self.assertAlmostEqual(selection.effective_age_frames, 0.0)

    def test_latest_with_lookahead_five_uses_token_five_from_newest_chunk(self):
        agg = TemporalAbsoluteAggregator(chunk_size=12, mode="latest", lookahead_steps=5)
        agg.add_prediction(9, np.asarray([900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911], dtype=np.float32))
        agg.add_prediction(10, np.asarray([1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011], dtype=np.float32))

        selection = agg.selection_for_step(10)
        self.assertIsNotNone(selection)
        self.assertAlmostEqual(selection.value, 1005.0)
        self.assertEqual(selection.contributor_count, 1)
        self.assertAlmostEqual(selection.effective_age_frames, 0.0)

    def test_recent_with_lookahead_uses_step_shifted_tokens_and_weights_1_2_4(self):
        agg = TemporalAbsoluteAggregator(
            chunk_size=12,
            mode="recent",
            recent_window=3,
            recent_half_life=1.0,
            lookahead_steps=5,
        )
        # At current_step=10 and L=5, expected tokens are oldest->newest: 7, 6, 5.
        old_chunk = np.asarray([0, 0, 0, 0, 0, 0, 0, 307, 0, 0, 0, 0], dtype=np.float32)   # source 8 -> token 7
        mid_chunk = np.asarray([0, 0, 0, 0, 0, 0, 406, 0, 0, 0, 0, 0], dtype=np.float32)   # source 9 -> token 6
        new_chunk = np.asarray([0, 0, 0, 0, 0, 505, 0, 0, 0, 0, 0, 0], dtype=np.float32)   # source 10 -> token 5
        agg.add_prediction(8, old_chunk)
        agg.add_prediction(9, mid_chunk)
        agg.add_prediction(10, new_chunk)

        selection = agg.selection_for_step(10)
        self.assertIsNotNone(selection)
        expected = (307.0 * 1.0 + 406.0 * 2.0 + 505.0 * 4.0) / 7.0
        self.assertAlmostEqual(selection.value, expected, places=12)

    def test_all_modes_use_step_aligned_token_index(self):
        chunks = {
            10: np.asarray([1000, 1010, 1020, 1030, 1040, 1050], dtype=np.float32),
            11: np.asarray([2000, 2010, 2020, 2030, 2040, 2050], dtype=np.float32),
            12: np.asarray([3000, 3010, 3020, 3030, 3040, 3050], dtype=np.float32),
        }
        current_step = 13

        full = TemporalAbsoluteAggregator(chunk_size=6, mode="full", decay=0.0)
        latest = TemporalAbsoluteAggregator(chunk_size=6, mode="latest")
        recent = TemporalAbsoluteAggregator(chunk_size=6, mode="recent", recent_window=3, recent_half_life=1.0)
        for source_step, chunk in chunks.items():
            full.add_prediction(source_step, chunk)
            latest.add_prediction(source_step, chunk)
            recent.add_prediction(source_step, chunk)

        self.assertAlmostEqual(latest.selection_for_step(current_step).value, 3010.0)

        full_expected = (1030.0 + 2020.0 + 3010.0) / 3.0
        self.assertAlmostEqual(full.selection_for_step(current_step).value, full_expected)

        weights = np.asarray([0.5 ** 3, 0.5 ** 2, 0.5 ** 1], dtype=np.float64)
        weights = weights / np.sum(weights)
        recent_expected = float(np.sum(np.asarray([1030.0, 2020.0, 3010.0]) * weights))
        self.assertAlmostEqual(recent.selection_for_step(current_step).value, recent_expected)

    def test_recent_window_three_half_life_one_has_relative_weights_1_2_4(self):
        agg = TemporalAbsoluteAggregator(chunk_size=10, mode="recent", recent_window=3, recent_half_life=1.0)
        agg.add_prediction(0, np.asarray([0, 0, 0, 3, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        agg.add_prediction(1, np.asarray([0, 0, 4, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        agg.add_prediction(2, np.asarray([0, 5, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))

        selection = agg.selection_for_step(3)
        self.assertIsNotNone(selection)
        expected = (3.0 * 1.0 + 4.0 * 2.0 + 5.0 * 4.0) / 7.0
        self.assertAlmostEqual(selection.value, expected, places=12)

    def test_recent_window_five_half_life_one_has_relative_weights_1_2_4_8_16(self):
        agg = TemporalAbsoluteAggregator(chunk_size=12, mode="recent", recent_window=5, recent_half_life=1.0)
        for source_step, value in enumerate([10, 20, 30, 40, 50]):
            chunk = np.zeros((12,), dtype=np.float32)
            chunk[5 - source_step] = float(value)
            agg.add_prediction(source_step, chunk)

        selection = agg.selection_for_step(5)
        self.assertIsNotNone(selection)
        expected = (10 * 1 + 20 * 2 + 30 * 4 + 40 * 8 + 50 * 16) / 31.0
        self.assertAlmostEqual(selection.value, expected, places=12)

    def test_recent_uses_available_contributors_during_startup(self):
        agg = TemporalAbsoluteAggregator(chunk_size=8, mode="recent", recent_window=5, recent_half_life=1.0)
        agg.add_prediction(0, np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32))
        agg.add_prediction(1, np.asarray([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.float32))

        selection = agg.selection_for_step(1)
        self.assertIsNotNone(selection)
        self.assertEqual(selection.contributor_count, 2)
        expected = (2.0 * 0.5 + 10.0 * 1.0) / 1.5
        self.assertAlmostEqual(selection.value, expected, places=12)

    def test_recent_ignores_contributions_older_than_recent_window(self):
        agg = TemporalAbsoluteAggregator(chunk_size=20, mode="recent", recent_window=3, recent_half_life=1.0)
        for source_step in range(0, 7):
            chunk = np.full((20,), fill_value=float(source_step), dtype=np.float32)
            agg.add_prediction(source_step, chunk)

        selection = agg.selection_for_step(6)
        self.assertIsNotNone(selection)
        weights = np.asarray([0.25, 0.5, 1.0], dtype=np.float64)
        weights = weights / np.sum(weights)
        expected = float(np.sum(np.asarray([4.0, 5.0, 6.0], dtype=np.float64) * weights))
        self.assertEqual(selection.contributor_count, 3)
        self.assertAlmostEqual(selection.value, expected, places=12)

    def test_full_with_lookahead_uses_source_age_plus_lookahead_and_legacy_order(self):
        agg = TemporalAbsoluteAggregator(chunk_size=12, mode="full", decay=0.3, lookahead_steps=5)
        agg.add_prediction(8, np.asarray([800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811], dtype=np.float32))
        agg.add_prediction(9, np.asarray([900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911], dtype=np.float32))
        agg.add_prediction(10, np.asarray([1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011], dtype=np.float32))

        # current_step=10, L=5 => tokens oldest->newest: 7, 6, 5 => values 807, 906, 1005
        selection = agg.selection_for_step(10)
        self.assertIsNotNone(selection)
        values = np.asarray([807.0, 906.0, 1005.0], dtype=np.float64)
        weights = np.exp(-0.3 * np.arange(3, dtype=np.float64))
        weights = weights / np.sum(weights)
        expected = float(np.sum(values * weights))
        self.assertAlmostEqual(selection.value, expected, places=12)

    def test_all_contributions_align_to_same_future_timestep(self):
        chunk_size = 20
        current_step = 10
        lookahead = 5
        agg = TemporalAbsoluteAggregator(
            chunk_size=chunk_size,
            mode="full",
            decay=0.0,
            lookahead_steps=lookahead,
        )

        # chunk[k] = source_step + k + 1 => selected value should be current_step + L + 1 for all contributors.
        for source_step in (8, 9, 10):
            chunk = np.asarray([source_step + k + 1 for k in range(chunk_size)], dtype=np.float32)
            agg.add_prediction(source_step, chunk)

        expected_value = float(current_step + lookahead + 1)
        contributions = agg._valid_contributions_for_step(current_step)
        self.assertEqual(len(contributions), 3)
        for _source_step, value, _token_index in contributions:
            self.assertAlmostEqual(value, expected_value)

    def test_future_source_step_is_rejected_even_with_positive_lookahead(self):
        agg = TemporalAbsoluteAggregator(chunk_size=12, mode="full", lookahead_steps=5)
        agg.add_prediction(11, np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.float32))
        self.assertIsNone(agg.selection_for_step(10))

    def test_contributions_exceeding_chunk_after_lookahead_are_excluded(self):
        agg = TemporalAbsoluteAggregator(chunk_size=12, mode="full", lookahead_steps=5, decay=0.0)
        agg.add_prediction(0, np.asarray([10 + k for k in range(12)], dtype=np.float32))
        agg.add_prediction(8, np.asarray([80 + k for k in range(12)], dtype=np.float32))

        # current_step=10: source 0 => token 15 invalid; source 8 => token 7 valid
        selection = agg.selection_for_step(10)
        self.assertIsNotNone(selection)
        self.assertEqual(selection.contributor_count, 1)
        self.assertAlmostEqual(selection.value, 87.0)

    def test_latest_allows_maximum_valid_lookahead(self):
        agg = TemporalAbsoluteAggregator(chunk_size=30, mode="latest", lookahead_steps=29)
        newest = np.asarray([1000 + k for k in range(30)], dtype=np.float32)
        agg.add_prediction(42, newest)
        selection = agg.selection_for_step(42)
        self.assertIsNotNone(selection)
        self.assertAlmostEqual(selection.value, 1029.0)

    def test_large_lookahead_is_valid_for_latest_even_if_recent_default_window_would_not_fit(self):
        agg = TemporalAbsoluteAggregator(chunk_size=6, mode="latest", lookahead_steps=5)
        agg.add_prediction(3, np.asarray([30, 31, 32, 33, 34, 35], dtype=np.float32))
        self.assertAlmostEqual(agg.selection_for_step(3).value, 35.0)

    def test_contributor_count_and_effective_age_for_all_modes(self):
        chunk_size = 10
        current_step = 5
        source_steps = [2, 3, 4]

        full = TemporalAbsoluteAggregator(chunk_size=chunk_size, mode="full", decay=0.2)
        latest = TemporalAbsoluteAggregator(chunk_size=chunk_size, mode="latest")
        recent = TemporalAbsoluteAggregator(
            chunk_size=chunk_size,
            mode="recent",
            recent_window=3,
            recent_half_life=1.0,
        )

        for source_step in source_steps:
            chunk = np.zeros((chunk_size,), dtype=np.float32)
            chunk[current_step - source_step] = float(100 + source_step)
            full.add_prediction(source_step, chunk)
            latest.add_prediction(source_step, chunk)
            recent.add_prediction(source_step, chunk)

        full_sel = full.selection_for_step(current_step)
        latest_sel = latest.selection_for_step(current_step)
        recent_sel = recent.selection_for_step(current_step)

        self.assertEqual(full_sel.contributor_count, 3)
        self.assertEqual(latest_sel.contributor_count, 1)
        self.assertEqual(recent_sel.contributor_count, 3)

        ages = np.asarray([3.0, 2.0, 1.0], dtype=np.float64)
        full_weights = np.exp(-0.2 * np.arange(3, dtype=np.float64))
        full_weights /= np.sum(full_weights)
        self.assertAlmostEqual(full_sel.effective_age_frames, float(np.sum(ages * full_weights)), places=12)
        self.assertAlmostEqual(latest_sel.effective_age_frames, 1.0, places=12)

        recent_weights = np.asarray([0.125, 0.25, 0.5], dtype=np.float64)
        recent_weights /= np.sum(recent_weights)
        self.assertAlmostEqual(
            recent_sel.effective_age_frames,
            float(np.sum(ages * recent_weights)),
            places=12,
        )

    def test_reset_removes_all_contributors(self):
        agg = TemporalAbsoluteAggregator(chunk_size=5, mode="recent", recent_window=3, recent_half_life=1.0)
        agg.add_prediction(0, np.asarray([1, 2, 3, 4, 5], dtype=np.float32))
        self.assertIsNotNone(agg.selection_for_step(0))
        agg.reset()
        self.assertIsNone(agg.selection_for_step(0))
        self.assertIsNone(agg.value_for_step(0))

    def test_invalid_constructor_values_raise_value_error(self):
        with self.assertRaises(ValueError):
            TemporalAbsoluteAggregator(chunk_size=0)
        with self.assertRaises(ValueError):
            TemporalAbsoluteAggregator(chunk_size=5, mode="unknown")
        with self.assertRaises(ValueError):
            TemporalAbsoluteAggregator(chunk_size=5, decay=-0.01)
        with self.assertRaises(ValueError):
            TemporalAbsoluteAggregator(chunk_size=5, decay=float("inf"))
        with self.assertRaises(ValueError):
            TemporalAbsoluteAggregator(chunk_size=5, recent_window=0)
        with self.assertRaises(ValueError):
            TemporalAbsoluteAggregator(chunk_size=5, recent_window=6)
        with self.assertRaises(ValueError):
            TemporalAbsoluteAggregator(chunk_size=5, recent_half_life=0.0)
        with self.assertRaises(ValueError):
            TemporalAbsoluteAggregator(chunk_size=5, recent_half_life=float("nan"))
        with self.assertRaises(ValueError):
            TemporalAbsoluteAggregator(chunk_size=5, lookahead_steps=-1)
        with self.assertRaises(ValueError):
            TemporalAbsoluteAggregator(chunk_size=5, lookahead_steps=5)
        with self.assertRaises(ValueError):
            TemporalAbsoluteAggregator(
                chunk_size=10,
                mode="recent",
                recent_window=6,
                lookahead_steps=5,
            )

    def test_value_for_step_wrapper_remains_available(self):
        agg = TemporalAbsoluteAggregator(chunk_size=5, mode="latest")
        agg.add_prediction(10, np.asarray([1, 2, 3, 4, 5], dtype=np.float32))
        wrapped = agg.value_for_step(10)
        selected = agg.selection_for_step(10)
        self.assertIsNotNone(wrapped)
        self.assertIsNotNone(selected)
        self.assertAlmostEqual(wrapped, selected.value)

    def test_cli_mode_resolution_rules(self):
        self.assertEqual(resolve_temporal_agg_mode(None, None), "full")
        self.assertEqual(resolve_temporal_agg_mode(None, True), "full")
        self.assertEqual(resolve_temporal_agg_mode(None, False), "latest")
        self.assertEqual(resolve_temporal_agg_mode("full", None), "full")
        self.assertEqual(resolve_temporal_agg_mode("latest", None), "latest")
        self.assertEqual(resolve_temporal_agg_mode("recent", None), "recent")
        with self.assertRaises(ValueError):
            resolve_temporal_agg_mode("full", True)
        with self.assertRaises(ValueError):
            resolve_temporal_agg_mode("latest", False)

    def test_effective_age_frames_does_not_include_lookahead(self):
        agg = TemporalAbsoluteAggregator(
            chunk_size=12,
            mode="recent",
            recent_window=3,
            recent_half_life=1.0,
            lookahead_steps=5,
        )
        for source_step in (8, 9, 10):
            chunk = np.zeros((12,), dtype=np.float32)
            chunk[(10 - source_step) + 5] = float(100 + source_step)
            agg.add_prediction(source_step, chunk)

        selection = agg.selection_for_step(10)
        self.assertIsNotNone(selection)
        ages = np.asarray([2.0, 1.0, 0.0], dtype=np.float64)
        weights = np.asarray([0.25, 0.5, 1.0], dtype=np.float64)
        weights /= np.sum(weights)
        expected_age = float(np.sum(ages * weights))
        self.assertAlmostEqual(selection.effective_age_frames, expected_age, places=12)


if __name__ == "__main__":
    unittest.main()
