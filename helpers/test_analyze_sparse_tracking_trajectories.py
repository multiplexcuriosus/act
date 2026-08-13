#!/usr/bin/env python3
"""Synthetic tests for sparse spatial trajectory analysis."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_sparse_tracking_trajectories as analysis  # noqa: E402


def trajectory(points, timestamps=None, valid=None, ages=None, modality="rgb", max_gap=.15):
    points = np.asarray(points, dtype=float)
    n = len(points)
    timestamps = np.arange(n) * .02 if timestamps is None else np.asarray(timestamps, dtype=float)
    valid = np.ones(n) if valid is None else np.asarray(valid)
    ages = np.zeros(n) if ages is None else np.asarray(ages, dtype=float)
    return analysis.build_trajectory(modality, timestamps, points, valid, timestamps, ages, .1, 1, max_gap)


def write_episode(path, rgb_points, event_points=None, rgb_source=None, event_source=None,
                  rgb_valid=None, event_valid=None, rgb_ages=None, event_ages=None):
    rgb_points = np.asarray(rgb_points, dtype=float); n = len(rgb_points)
    times = np.arange(n, dtype=float) * .02
    with h5py.File(path, "w") as h5:
        h5.attrs["episode_index"] = 7
        obs = h5.require_group("observations")
        obs.create_dataset("timestamps", data=times)
        obs.create_dataset("timestamps_ns", data=np.asarray(times * 1e9, dtype=np.int64))
        sparse = obs.require_group("sparse_tracking")
        def put(prefix, points, source, valid, ages):
            sparse.create_dataset(f"{prefix}_2d_px", data=points)
            sparse.create_dataset(f"{prefix}_valid", data=np.ones(n) if valid is None else valid)
            sparse.create_dataset(f"{prefix}_source_timestamps", data=times if source is None else source)
            sparse.create_dataset(f"{prefix}_source_age_sec", data=np.zeros(n) if ages is None else ages)
        put("rgb", rgb_points, rgb_source, rgb_valid, rgb_ages)
        if event_points is not None:
            put("event", np.asarray(event_points, dtype=float), event_source, event_valid, event_ages)


class TrajectoryAnalysisTests(unittest.TestCase):
    def test_clear_forward_reverse_detects_turn(self):
        data, _ = trajectory([[0, 0], [10, 0], [20, 0], [30, 0], [20, 0], [10, 0], [0, 0]])
        result = analysis.infer_turning_point(data, half_window=2, min_displacement_px=5)
        self.assertIn(result["status"], ("detected", "ambiguous"))
        self.assertEqual(result["selected"]["policy_frame_index"], 3)
        self.assertAlmostEqual(result["selected"]["reversal_angle_deg"], 180)

    def test_monotonic_has_no_turn(self):
        data, _ = trajectory([[x, 0] for x in range(0, 70, 10)])
        self.assertEqual(analysis.infer_turning_point(data)["status"], "no_turn_detected")

    def test_stationary_jitter_has_no_false_turn(self):
        data, _ = trajectory([[10, 10], [11, 9], [9, 11], [10, 10], [11, 10], [10, 9]])
        self.assertEqual(analysis.infer_turning_point(data, min_displacement_px=5)["status"], "no_turn_detected")

    def test_held_timestamps_are_deduplicated(self):
        data, diagnostic = trajectory([[0, 0], [0, 0], [10, 0], [10, 0], [20, 0]], timestamps=[0, 0, .02, .02, .04])
        self.assertEqual(diagnostic["fresh_detection_count"], 5)
        self.assertEqual(diagnostic["unique_source_update_count"], 3)
        self.assertEqual(data.policy_frames.tolist(), [0, 2, 4])

    def test_invalid_stale_and_nonfinite_are_excluded(self):
        points = [[0, 0], [10, 0], [20, 0], [np.nan, 0], [40, 0]]
        data, diagnostic = trajectory(points, valid=[1, 0, 1, 1, 1], ages=[0, 0, .2, 0, .01])
        self.assertEqual(diagnostic["valid_count"], 3)
        self.assertEqual(data.policy_frames.tolist(), [0, 4])

    def test_large_gap_breaks_candidate_continuity(self):
        times = [0, .02, .04, .50, .52, .54]
        data, _ = trajectory([[0, 0], [10, 0], [20, 0], [10, 0], [0, 0], [-10, 0]], timestamps=times)
        self.assertEqual(len(np.unique(data.segment_ids)), 2)
        self.assertEqual(analysis.infer_turning_point(data, half_window=2)["status"], "no_turn_detected")

    def test_coordinates_remain_modality_specific(self):
        common = [[0, 0], [10, 0], [20, 0], [30, 0], [20, 0], [10, 0], [0, 0]]
        rgb_t, rgb_d = trajectory(np.asarray(common) + [500, 300], modality="rgb")
        event_t, event_d = trajectory(np.asarray(common) + [100, 100], modality="event")
        rgb_turn = analysis.infer_turning_point(rgb_t); event_turn = analysis.infer_turning_point(event_t)
        rgb = analysis.make_episode_metric("x", "0", "rgb", rgb_t, rgb_d, rgb_turn, 4)
        event = analysis.make_episode_metric("x", "0", "event", event_t, event_d, event_turn, 4)
        paired = analysis.compare_modalities(rgb, event)
        self.assertEqual(paired["rgb_turning_point_u"], 530)
        self.assertEqual(paired["event_turning_point_u"], 130)
        self.assertNotIn("spatial_error", paired)

    def test_paired_timestamp_comparison(self):
        points = [[0, 0], [10, 0], [20, 0], [30, 0], [20, 0], [10, 0], [0, 0]]
        rgb_t, rgb_d = trajectory(points)
        event_t, event_d = trajectory(points)
        event_t.policy_timestamps = event_t.policy_timestamps + .01
        rgb = analysis.make_episode_metric("x", "0", "rgb", rgb_t, rgb_d, analysis.infer_turning_point(rgb_t), 4)
        event = analysis.make_episode_metric("x", "0", "event", event_t, event_d, analysis.infer_turning_point(event_t), 4)
        paired = analysis.compare_modalities(rgb, event)
        self.assertAlmostEqual(paired["event_minus_rgb_timestamp_ms"], 10)
        self.assertEqual(paired["event_lead_lag"], "event_lags")

    def test_missing_event_warns_and_rgb_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); episode = root / "episode.hdf5"; output = root / "out"
            points = [[0, 0], [10, 0], [20, 0], [30, 0], [20, 0], [10, 0], [0, 0]]
            write_episode(episode, points)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = analysis.main([str(episode), "--output-dir", str(output), "--smoothing-window", "1"])
            self.assertEqual(result, 0)
            self.assertIn("missing event fields", stderr.getvalue())
            rows = (output / "episode_metrics.csv").read_text()
            self.assertIn("RGB native pixels", rows)

    def test_cli_creates_csv_json_and_plots(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); episode = root / "episode.hdf5"; output = root / "out"
            rgb = np.asarray([[500+x, 300] for x in (0, 10, 20, 30, 20, 10, 0)])
            event = np.asarray([[100+x, 100] for x in (0, 10, 20, 30, 20, 10, 0)])
            write_episode(episode, rgb, event, event_source=np.arange(7) * .02 + .005)
            result = analysis.main([str(episode), "--output-dir", str(output), "--plot", "--smoothing-window", "1"])
            self.assertEqual(result, 0)
            for name in ("episode_metrics.csv", "turning_point_comparison.csv", "summary.json"):
                self.assertTrue((output / name).is_file(), name)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["paired"]["both_turns"], 1)
            expected = (
                "episode_7_rgb_event_trajectory.png", "all_rgb_trajectories.png",
                "all_event_trajectories.png", "all_event_trajectories_turn_aligned.png",
                "turning_point_frame_comparison.png", "turning_point_time_difference.png",
                "pre_turn_detection_coverage.png", "source_age_distribution.png",
                "inter_detection_gap_distribution.png",
            )
            for name in expected:
                self.assertTrue((output / "plots" / name).is_file(), name)
            self.assertFalse((output / "plots" / "episode_7_rgb_trajectory.png").exists())
            self.assertFalse((output / "plots" / "episode_7_event_trajectory.png").exists())


if __name__ == "__main__":
    unittest.main()
