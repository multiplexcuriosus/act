#!/usr/bin/env python3
"""Synthetic tests for native sidecar event-detection analysis."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_event_detection_gaps as analysis  # noqa: E402


FIELDS = analysis.SIDECAR_CORE_DATASETS + analysis.SIDECAR_OPTIONAL_DATASETS


def arrays(times=(0.0, 0.02, 0.04, 0.20, 0.22), valid=(1, 1, 1, 1, 1), packet=None):
    n = len(times)
    packet = np.arange(n) if packet is None else np.asarray(packet)
    result = {
        "available_ros_t_ns": np.asarray(np.asarray(times) * 1e9, dtype=np.int64),
        "packet_id": packet,
        "sensor_window_start_us": np.arange(n) * 1000,
        "sensor_window_end_us": np.arange(n) * 1000 + 5000,
        "valid": np.asarray(valid),
    }
    for name in analysis.SIDECAR_OPTIONAL_DATASETS:
        if name == "rejection_reason":
            result[name] = np.asarray([b"none" if value else b"threshold" for value in valid])
        elif name not in result:
            result[name] = np.arange(n, dtype=float)
    return result


def write_group(path, name, values):
    with h5py.File(path, "a") as h5:
        group = h5.require_group(f"episodes/{name}")
        for key, value in values.items():
            group.create_dataset(key, data=value)


class SidecarAnalysisTests(unittest.TestCase):
    def test_long_inactive_margins_and_mixed_rows(self):
        data = arrays(times=(0, .01, .02, .04, .20, .22, .50), valid=(0, 0, 1, 0, 1, 1, 0))
        self.assertEqual(analysis.extract_active_interval(data), (2, 5))
        runs = analysis.sidecar_run_length_encode(data, "e")
        self.assertEqual([row["row_count"] for row in runs], [1, 1, 2])
        self.assertEqual(sum(row["row_count"] for row in runs), 4)

    def test_specified_valid_timestamps_and_phase_split(self):
        data = arrays()
        gaps, indices = analysis.calculate_valid_detection_gaps(data, "e")
        np.testing.assert_allclose([row["inter_detection_interval_ms"] for row in gaps], [20, 20, 160, 20])
        phases, _ = analysis.segment_dense_phases(data, "e", phase_gap_ms=100, min_phase_detections=3)
        self.assertEqual([row["valid_detection_count"] for row in phases], [3, 2])
        self.assertEqual([row["is_dense"] for row in phases], [True, False])
        windows = analysis.calculate_phase_window_rates(phases, data, rate_window_ms=25)
        self.assertEqual([row["valid_detection_count"] for row in windows], [2, 1])

    def test_duplicate_packets_are_not_double_counted(self):
        data = arrays(times=(0, .001, .02, .04), valid=(1, 1, 1, 1), packet=(7, 7, 8, 9))
        gaps, indices = analysis.calculate_valid_detection_gaps(data)
        self.assertEqual(indices.tolist(), [0, 2, 3])
        self.assertEqual(len(gaps), 2)
        metric, _, _ = analysis.summarize_sidecar_episode(data, "e", "s.h5", "/episodes/episode_0")
        self.assertEqual(metric["total_sidecar_rows"], 4)
        self.assertEqual(metric["valid_detection_count"], 3)

    def test_no_and_one_valid_detection(self):
        none = arrays(times=(0, .1), valid=(0, 0))
        metric, gaps, runs = analysis.summarize_sidecar_episode(none, "none", "s", "/g")
        self.assertIsNone(metric["active_interval_start"])
        self.assertIsNone(metric["median_inter_detection_interval_ms"])
        self.assertEqual((gaps, runs), ([], []))
        one = arrays(times=(0, .1, .2), valid=(0, 1, 0))
        metric, gaps, _ = analysis.summarize_sidecar_episode(one, "one", "s", "/g")
        self.assertEqual(metric["valid_detection_span_sec"], 0)
        self.assertIsNone(metric["sidecar_update_rate_hz"] if False else metric["median_inter_detection_interval_ms"])
        self.assertEqual(gaps, [])

    def test_validation_and_stable_sort(self):
        data = arrays(times=(.02, 0, .02), valid=(1, 1, 0), packet=(2, 1, 1))
        sorted_data, changed = analysis.stable_sort_sidecar_rows(data)
        self.assertTrue(changed)
        self.assertEqual(sorted_data["packet_id"].tolist(), [1, 1, 2])
        malformed = dict(data)
        malformed["available_ros_t_ns"] = np.asarray([0, np.nan, 2])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            analysis.validate_sidecar_arrays(malformed)
        missing = dict(data)
        del missing["valid"]
        with self.assertRaisesRegex(ValueError, "missing required"):
            analysis.validate_sidecar_arrays(missing)

    def test_missing_optional_fields_and_statistics(self):
        data = {key: value for key, value in arrays().items() if key in analysis.SIDECAR_CORE_DATASETS}
        analysis.validate_sidecar_arrays(data)
        stats = analysis.detection_window_statistics(data, "e")
        self.assertEqual({row["field"] for row in stats}, {"sensor_window_duration_ms", "valid"})
        self.assertEqual(stats[0]["median"], 5.0)

    def test_multiple_groups_loading_and_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "sidecar.hdf5"
            write_group(sidecar, "episode_0", arrays())
            write_group(sidecar, "episode_1", arrays(times=(0, .1), valid=(0, 1)))
            self.assertEqual(analysis.sidecar_group_names(sidecar),
                             ["/episodes/episode_0", "/episodes/episode_1"])
            groups = analysis._sidecar_group_metadata(sidecar)
            matched = analysis.match_episode_to_sidecar_group(Path(directory) / "episode_1.hdf5", {}, groups)
            self.assertEqual(matched, "/episodes/episode_1")
            loaded, _, _ = analysis.load_sidecar_group(sidecar, matched)
            self.assertEqual(len(loaded["valid"]), 2)

    def test_aggregate_summary(self):
        data = arrays()
        metric, gaps, _ = analysis.summarize_sidecar_episode(data, "e", "s", "/g")
        metric["_all_update_intervals_ms"] = (np.diff(data["available_ros_t_ns"]) * 1e-6).tolist()
        phases, _ = analysis.segment_dense_phases(data, "e", 100, 2)
        windows = analysis.calculate_phase_window_rates(phases, data, 25)
        stats = analysis.detection_window_statistics(data, "e")
        for row in stats:
            row["_finite_values"] = [5.0] if row["field"] == "sensor_window_duration_ms" else []
        summary = analysis.summarize_sidecar_aggregate([metric], gaps, phases, windows, stats, ["s"], 1, 0)
        self.assertEqual(summary["analysis_source"], "sidecar")
        self.assertEqual(summary["total_valid_detection_count"], 5)
        self.assertEqual(summary["median_sensor_window_duration_ms"], 5.0)

    def test_cli_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, sidecar, output = root / "episode_0.hdf5", root / "native.hdf5", root / "out"
            with h5py.File(episode, "w") as h5:
                h5.attrs["episode_index"] = 0
                h5.attrs["sparse_tracking_sidecar_h5"] = sidecar.name
                h5.attrs["sparse_tracking_tracker_config_json"] = json.dumps({"bin_ms": 1, "accumulation_window_ms": 5})
            write_group(sidecar, "episode_0", arrays())
            result = analysis.main([str(episode), "--analysis-source", "sidecar", "--output-dir", str(output), "--plot"])
            self.assertEqual(result, 0)
            for filename in ("sidecar_episode_metrics.csv", "sidecar_detection_intervals.csv",
                             "sidecar_rle_runs.csv", "sidecar_phases.csv", "sidecar_phase_rate_windows.csv",
                             "sidecar_window_statistics.csv", "sidecar_rejection_statistics.csv",
                             "sidecar_aggregate_summary.json"):
                self.assertTrue((output / filename).is_file(), filename)
            dashboard = output / "plots" / "sidecar_aggregate_summary.svg"
            self.assertTrue(dashboard.is_file())
            dashboard_text = dashboard.read_text(encoding="utf-8")
            self.assertIn("Pooled active-interval sidecar-row fraction", dashboard_text)
            self.assertIn("Recorded valid-output gap distribution", dashboard_text)
            self.assertIn("Per-episode active-interval valid-row fractions", dashboard_text)
            self.assertIn("episode mean", dashboard_text)
            self.assertIn('stroke-width="5"', dashboard_text)
            self.assertNotIn("missing sidecar rows", dashboard_text)
            self.assertNotIn("median/p90/p95/max", dashboard_text)
            window_plots = list((output / "plots").glob("episode_*_window_diagnostics.svg"))
            self.assertEqual(len(window_plots), 1)
            window_text = window_plots[0].read_text(encoding="utf-8")
            self.assertIn("equal-height bars", window_text)
            self.assertIn("width encodes observed duration", window_text)
            longest_run_plot = output / "plots" / "longest_valid_row_sequence_duration_range_mean.svg"
            self.assertTrue(longest_run_plot.is_file())
            longest_run_text = longest_run_plot.read_text(encoding="utf-8")
            self.assertIn("Longest valid-row-sequence duration per episode", longest_run_text)
            self.assertIn("min 220.00 ms", longest_run_text)
            self.assertIn("mean 220.00 ms", longest_run_text)
            self.assertIn("max 220.00 ms", longest_run_text)

    def test_default_episode_mode_remains_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, output = root / "episode.hdf5", root / "out"
            with h5py.File(episode, "w") as h5:
                observations = h5.require_group("observations")
                observations.create_dataset("timestamps", data=[0.0, 1 / 30, 2 / 30])
                sparse = observations.require_group("sparse_tracking")
                sparse.create_dataset("event_valid", data=[0, 1, 0])
                sparse.create_dataset("event_has_update", data=[0, 1, 0])
                sparse.create_dataset("event_source_timestamps", data=[0.0, 0.01, 0.01])
                sparse.create_dataset("event_source_age_sec", data=[np.nan, 0.0, 0.04])
            result = analysis.main([str(episode), "--output-dir", str(output)])
            self.assertEqual(result, 0)
            self.assertTrue((output / "per_episode_metrics.csv").is_file())
            self.assertTrue((output / "aggregate_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
