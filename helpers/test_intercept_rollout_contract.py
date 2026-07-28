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
    EXPECTED_INTERCEPT_METADATA_BY_TARGET,
    TemporalAbsoluteAggregator,
    absolute_s_from_anchor,
    build_absolute_prediction_chunk,
    build_qpos_history,
    build_rgb_history_tensor,
    compute_history_indices,
    denormalize_delta_chunk,
    select_sync_observation,
    validate_anchor_freshness,
    validate_intercept_stats_and_config,
)


class InterceptRolloutContractTests(unittest.TestCase):
    def make_stats(self, intercept_target="measured_tcp_trajectory"):
        stats = dict(EXPECTED_INTERCEPT_METADATA_BY_TARGET[intercept_target])
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

    def test_desired_goal_reconstruction(self):
        delta = np.asarray([0.11, 0.12, 0.13], dtype=np.float32)
        chunk, predicted_delta_goal, predicted_goal_abs_s = build_absolute_prediction_chunk(
            anchor_tcp_s=-0.02,
            delta_chunk=delta,
            intercept_target="desired_goal",
            chunk_size=30,
        )

        self.assertAlmostEqual(predicted_delta_goal, 0.12)
        self.assertAlmostEqual(predicted_goal_abs_s, 0.10)
        self.assertEqual(chunk.shape, (30,))
        self.assertTrue(np.allclose(chunk, np.full(30, 0.10, dtype=np.float32)))

    def test_measured_mode_reconstruction_remains_relative(self):
        delta = np.asarray([0.11, 0.12, 0.13], dtype=np.float32)
        chunk, predicted_delta_goal, predicted_goal_abs_s = build_absolute_prediction_chunk(
            anchor_tcp_s=-0.02,
            delta_chunk=delta,
            intercept_target="measured_tcp_trajectory",
            chunk_size=30,
        )

        self.assertAlmostEqual(predicted_delta_goal, 0.12)
        self.assertAlmostEqual(predicted_goal_abs_s, 0.10)
        np.testing.assert_allclose(chunk, np.asarray([0.09, 0.10, 0.11], dtype=np.float32), atol=1e-6)

    def test_no_bce_contract(self):
        stats = self.make_stats()
        config = self.make_policy_config()
        config["use_bce_last_action_dim"] = True
        with self.assertRaises(ValueError):
            validate_intercept_stats_and_config(stats, config, expected_chunk_size=30)

    def test_desired_goal_contract(self):
        stats = self.make_stats(intercept_target="desired_goal")
        config = self.make_policy_config()
        arrays = validate_intercept_stats_and_config(stats, config, expected_chunk_size=30)
        self.assertEqual(arrays["action_mean"].shape, (1,))
        self.assertEqual(arrays["action_std"].shape, (1,))

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

    def test_missing_intercept_target_rejected(self):
        stats = self.make_stats()
        del stats["intercept_target"]
        with self.assertRaises(ValueError):
            validate_intercept_stats_and_config(stats, self.make_policy_config(), expected_chunk_size=30)

    def test_temporal_aggregation_in_absolute_coordinates(self):
        agg = TemporalAbsoluteAggregator(chunk_size=4, decay=0.0)
        agg.add_prediction(0, np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        agg.add_prediction(1, np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float32))

        self.assertAlmostEqual(agg.value_for_step(1), (2.0 + 10.0) / 2.0)
        self.assertAlmostEqual(agg.value_for_step(2), (3.0 + 11.0) / 2.0)


if __name__ == "__main__":
    unittest.main()
