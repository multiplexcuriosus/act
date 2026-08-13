#!/usr/bin/env python3
"""Analyze policy-grid-visible event ball-detection presence in ACT HDF5 files.

The episode arrays are sampled on an approximately 30 Hz observation grid using
the latest tracker message at or before each grid time.  Consequently, ``valid``
can include held values.  This tool measures temporal presence as visible to the
policy grid; native event-tracker gaps faster than the observation grid require
analysis of the tracker sidecar.

With ``--analysis-source sidecar``, the tool instead measures recorded native
tracker-output rows.  That mode can describe output cadence and observed sensor
window spans, but cannot reconstruct empty 1 ms bins or internal CPU latency.

Only the Python standard library, NumPy, and h5py are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REQUIRED_DATASETS = (
    "observations/timestamps",
    "observations/sparse_tracking/event_valid",
    "observations/sparse_tracking/event_has_update",
    "observations/sparse_tracking/event_source_timestamps",
    "observations/sparse_tracking/event_source_age_sec",
)
GAP_THRESHOLDS_MS = (33.0, 50.0, 100.0, 200.0)
SIDECAR_GAP_THRESHOLDS_MS = (33.0, 50.0, 100.0, 200.0, 500.0)
ONE_SAMPLE_PERIOD_SEC = 1.0 / 30.0
SIDECAR_CORE_DATASETS = (
    "available_ros_t_ns", "packet_id", "sensor_window_start_us",
    "sensor_window_end_us", "valid",
)
SIDECAR_OPTIONAL_DATASETS = (
    "x_px", "y_px", "vx_px_s", "vy_px_s", "speed_px_s", "confidence",
    "velocity_valid", "window_event_count", "candidate_count", "blob_area_px",
    "blob_event_count", "blob_width_px", "blob_height_px", "circularity",
    "rejection_reason",
)
SIDECAR_NUMERIC_FIELDS = (
    "sensor_window_duration_ms", "window_event_count", "blob_event_count",
    "candidate_count", "blob_area_px", "blob_width_px", "blob_height_px",
    "circularity", "confidence", "speed_px_s", "vx_px_s", "vy_px_s",
    "velocity_valid", "valid",
)


def detect_new_updates(
    event_valid: Sequence[Any],
    source_timestamps: Sequence[float],
    timestamp_epsilon: float = 1e-9,
) -> np.ndarray:
    """Return valid samples whose finite source timestamp advanced from the prior sample.

    Sample zero is false because the episode contains no preceding sampled source
    timestamp with which to establish that a native update occurred.
    """
    valid = np.asarray(event_valid).reshape(-1) != 0
    source = np.asarray(source_timestamps, dtype=np.float64).reshape(-1)
    if valid.size != source.size:
        raise ValueError("event_valid and event_source_timestamps lengths differ")
    result = np.zeros(valid.size, dtype=bool)
    if valid.size > 1:
        result[1:] = (
            valid[1:]
            & np.isfinite(source[1:])
            & np.isfinite(source[:-1])
            & (source[1:] > source[:-1] + timestamp_epsilon)
        )
    return result


def construct_presence_signal(
    event_valid: Sequence[Any],
    event_source_age_sec: Sequence[float],
    event_source_timestamps: Sequence[float],
    mode: str = "fresh",
    max_age_sec: float = 0.050,
    timestamp_epsilon: float = 1e-9,
) -> np.ndarray:
    """Construct one of the ``valid``, ``fresh``, or ``new_update`` signals."""
    valid = np.asarray(event_valid).reshape(-1) != 0
    age = np.asarray(event_source_age_sec, dtype=np.float64).reshape(-1)
    source = np.asarray(event_source_timestamps, dtype=np.float64).reshape(-1)
    if not (valid.size == age.size == source.size):
        raise ValueError("presence input array lengths differ")
    if mode == "valid":
        return valid
    if mode == "fresh":
        return valid & np.isfinite(age) & (age <= max_age_sec)
    if mode == "new_update":
        return detect_new_updates(valid, source, timestamp_epsilon)
    raise ValueError(f"unknown presence mode: {mode}")


def timestamp_cell_edges(
    timestamps: Sequence[float], fallback_period_sec: float = ONE_SAMPLE_PERIOD_SEC
) -> np.ndarray:
    """Build sample-cell boundaries from timestamp midpoints."""
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("cannot construct edges for an empty episode")
    if times.size == 1:
        half = 0.5 * fallback_period_sec
        return np.array([times[0] - half, times[0] + half], dtype=np.float64)
    edges = np.empty(times.size + 1, dtype=np.float64)
    edges[0] = times[0] - 0.5 * (times[1] - times[0])
    edges[1:-1] = 0.5 * (times[:-1] + times[1:])
    edges[-1] = times[-1] + 0.5 * (times[-1] - times[-2])
    return edges


def run_length_encode(
    signal: Sequence[Any], timestamps: Sequence[float], episode_id: str = ""
) -> list[dict[str, Any]]:
    """Encode the complete binary signal, including present and absent runs."""
    present = np.asarray(signal, dtype=bool).reshape(-1)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if present.size != times.size:
        raise ValueError("signal and timestamp lengths differ")
    if present.size == 0:
        raise ValueError("cannot encode an empty signal")
    edges = timestamp_cell_edges(times)
    starts = np.r_[0, np.flatnonzero(present[1:] != present[:-1]) + 1]
    ends = np.r_[starts[1:], present.size]
    rows: list[dict[str, Any]] = []
    for run_index, (start, end) in enumerate(zip(starts, ends)):
        start_i, end_i = int(start), int(end)
        run_start, run_end = float(edges[start_i]), float(edges[end_i])
        rows.append(
            {
                "episode_id": episode_id,
                "run_index": run_index,
                "state": "present" if present[start_i] else "absent",
                "start_sample_index": start_i,
                "end_sample_index_exclusive": end_i,
                "sample_count": end_i - start_i,
                "start_timestamp": run_start,
                "end_timestamp": run_end,
                "start_offset_sec": run_start - float(edges[0]),
                "end_offset_sec": run_end - float(edges[0]),
                "duration_sec": run_end - run_start,
                "open_at_episode_start": start_i == 0,
                "open_at_episode_end": end_i == present.size,
            }
        )
    return rows


def extract_gaps(runs: Sequence[Mapping[str, Any]], interior_only: bool = False) -> list[dict[str, Any]]:
    """Extract absent runs; optionally omit runs touching either episode boundary."""
    gaps = [dict(run) for run in runs if run["state"] == "absent"]
    if interior_only:
        gaps = [
            gap
            for gap in gaps
            if not gap["open_at_episode_start"] and not gap["open_at_episode_end"]
        ]
    return gaps


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _duration_stats(durations: Sequence[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in durations]
    result: dict[str, float | int | None] = {
        "count": len(values),
        "total_sec": float(sum(values)),
        "min_sec": min(values) if values else None,
        "mean_sec": float(np.mean(values)) if values else None,
        "median_sec": _percentile(values, 50),
        "p90_sec": _percentile(values, 90),
        "p95_sec": _percentile(values, 95),
        "max_sec": max(values) if values else None,
    }
    for name in ("total", "min", "mean", "median", "p90", "p95", "max"):
        value = result[f"{name}_sec"]
        result[f"{name}_ms"] = None if value is None else float(value) * 1000.0
    return result


def _threshold_stats(gaps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    durations = np.asarray([gap["duration_sec"] for gap in gaps], dtype=np.float64)
    result: dict[str, Any] = {}
    for threshold_ms in GAP_THRESHOLDS_MS:
        label = str(int(threshold_ms))
        count = int(np.count_nonzero(durations > threshold_ms / 1000.0))
        result[f"gaps_gt_{label}ms_count"] = count
        result[f"gaps_gt_{label}ms_fraction"] = (
            float(count / durations.size) if durations.size else None
        )
    return result


def summarize_episode(
    episode_id: str,
    timestamps: Sequence[float],
    signal: Sequence[Any],
    runs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate time-aware presence and gap metrics for one validated episode."""
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    present = np.asarray(signal, dtype=bool).reshape(-1)
    encoded = list(runs) if runs is not None else run_length_encode(present, times, episode_id)
    edges = timestamp_cell_edges(times)
    duration = float(edges[-1] - edges[0])
    widths = np.diff(edges)
    all_gaps = extract_gaps(encoded)
    interior_gaps = extract_gaps(encoded, interior_only=True)
    gap_durations = [float(gap["duration_sec"]) for gap in all_gaps]
    interior_durations = [float(gap["duration_sec"]) for gap in interior_gaps]
    present_runs = [run for run in encoded if run["state"] == "present"]
    absent_time = float(sum(gap_durations))
    first_present = next((run for run in encoded if run["state"] == "present"), None)
    last_present = next((run for run in reversed(encoded) if run["state"] == "present"), None)
    gap_stats = _duration_stats(gap_durations)
    interior_stats = _duration_stats(interior_durations)
    median_period = float(np.median(np.diff(times))) if times.size > 1 else ONE_SAMPLE_PERIOD_SEC
    result: dict[str, Any] = {
        "episode_id": episode_id,
        "episode_duration_sec": duration,
        "number_of_samples": int(times.size),
        "nominal_grid_period_sec": ONE_SAMPLE_PERIOD_SEC,
        "nominal_grid_period_ms": ONE_SAMPLE_PERIOD_SEC * 1000.0,
        "median_grid_period_sec": median_period,
        "median_grid_period_ms": median_period * 1000.0,
        "number_of_present_samples": int(np.count_nonzero(present)),
        "number_of_absent_samples": int(times.size - np.count_nonzero(present)),
        "sample_presence_fraction": float(np.mean(present)),
        "time_weighted_presence_fraction": float(np.sum(widths[present]) / duration),
        "number_of_present_runs": len(present_runs),
        "number_of_absent_runs": len(all_gaps),
        "gap_count_including_boundary_gaps": len(all_gaps),
        "interior_gap_count_excluding_boundary_gaps": len(interior_gaps),
        "total_absent_time_sec": absent_time,
        "total_absent_time_ms": absent_time * 1000.0,
        "median_gap_duration_sec": gap_stats["median_sec"],
        "median_gap_duration_ms": None if gap_stats["median_sec"] is None else gap_stats["median_sec"] * 1000.0,
        "mean_gap_duration_sec": gap_stats["mean_sec"],
        "mean_gap_duration_ms": None if gap_stats["mean_sec"] is None else gap_stats["mean_sec"] * 1000.0,
        "p90_gap_duration_sec": gap_stats["p90_sec"],
        "p90_gap_duration_ms": None if gap_stats["p90_sec"] is None else gap_stats["p90_sec"] * 1000.0,
        "p95_gap_duration_sec": gap_stats["p95_sec"],
        "p95_gap_duration_ms": None if gap_stats["p95_sec"] is None else gap_stats["p95_sec"] * 1000.0,
        "maximum_gap_duration_sec": gap_stats["max_sec"],
        "maximum_gap_duration_ms": None if gap_stats["max_sec"] is None else gap_stats["max_sec"] * 1000.0,
        "minimum_gap_duration_sec": gap_stats["min_sec"],
        "minimum_gap_duration_ms": None if gap_stats["min_sec"] is None else gap_stats["min_sec"] * 1000.0,
        "gaps_per_second": len(all_gaps) / duration,
        "longest_continuous_present_duration_sec": max((float(r["duration_sec"]) for r in present_runs), default=0.0),
        "longest_continuous_absent_duration_sec": max(gap_durations, default=0.0),
        "time_to_first_present_detection_sec": (
            float(first_present["start_offset_sec"]) if first_present is not None else duration
        ),
        "time_since_last_present_detection_sec": (
            duration - float(last_present["end_offset_sec"]) if last_present is not None else duration
        ),
    }
    result.update({f"all_{key}": value for key, value in _threshold_stats(all_gaps).items()})
    result.update({f"interior_{key}": value for key, value in _threshold_stats(interior_gaps).items()})
    # Retain structured stats for aggregate use without placing them in the CSV.
    result["_all_gap_stats"] = gap_stats
    result["_interior_gap_stats"] = interior_stats
    return result


def summarize_aggregate(
    episode_metrics: Sequence[Mapping[str, Any]],
    all_runs: Sequence[Mapping[str, Any]],
    discovered_file_count: int,
    skipped_episode_count: int,
) -> dict[str, Any]:
    """Aggregate episode-level metrics without crossing episode boundaries."""
    metrics = list(episode_metrics)
    runs = list(all_runs)
    durations = np.asarray([m["episode_duration_sec"] for m in metrics], dtype=np.float64)
    weighted_present = sum(
        float(m["episode_duration_sec"]) * float(m["time_weighted_presence_fraction"])
        for m in metrics
    )
    total_duration = float(np.sum(durations))
    all_gaps = extract_gaps(runs)
    interior_gaps = extract_gaps(runs, interior_only=True)

    def metric_values(name: str) -> list[float]:
        return [float(m[name]) for m in metrics if m.get(name) is not None]

    def episode_exceedance(threshold_ms: float) -> float | None:
        if not metrics:
            return None
        key = f"all_gaps_gt_{int(threshold_ms)}ms_count"
        return float(np.mean([int(m[key]) > 0 for m in metrics]))

    max_gaps = metric_values("maximum_gap_duration_sec")
    median_gaps = metric_values("median_gap_duration_sec")
    all_stats = _duration_stats([float(g["duration_sec"]) for g in all_gaps])
    interior_stats = _duration_stats([float(g["duration_sec"]) for g in interior_gaps])
    return {
        "number_of_discovered_files": int(discovered_file_count),
        "number_of_analyzed_episodes": len(metrics),
        "number_of_skipped_episodes": int(skipped_episode_count),
        "total_observation_duration_sec": total_duration,
        "time_weighted_presence_fraction": weighted_present / total_duration if total_duration else None,
        "pooled_total_gap_count": len(all_gaps),
        "pooled_all_gaps": {**all_stats, **_threshold_stats(all_gaps)},
        "pooled_interior_gaps": {**interior_stats, **_threshold_stats(interior_gaps)},
        "median_episode_presence_fraction": _percentile(metric_values("time_weighted_presence_fraction"), 50),
        "p25_episode_presence_fraction": _percentile(metric_values("time_weighted_presence_fraction"), 25),
        "p75_episode_presence_fraction": _percentile(metric_values("time_weighted_presence_fraction"), 75),
        "median_episode_gap_count": _percentile(metric_values("gap_count_including_boundary_gaps"), 50),
        "median_episode_maximum_gap_sec": _percentile(max_gaps, 50),
        "p90_episode_maximum_gap_sec": _percentile(max_gaps, 90),
        "p95_episode_maximum_gap_sec": _percentile(max_gaps, 95),
        "median_episode_median_gap_sec": _percentile(median_gaps, 50),
        "p90_episode_median_gap_sec": _percentile(median_gaps, 90),
        "p95_episode_median_gap_sec": _percentile(median_gaps, 95),
        "fraction_of_episodes_with_gap_gt_50ms": episode_exceedance(50),
        "fraction_of_episodes_with_gap_gt_100ms": episode_exceedance(100),
        "fraction_of_episodes_with_gap_gt_200ms": episode_exceedance(200),
    }


def _episode_roots(h5: h5py.File) -> list[str]:
    roots: list[str] = []
    if "observations" in h5 and isinstance(h5["observations"], h5py.Group):
        roots.append("/")

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Group) and name.endswith("/observations"):
            roots.append("/" + name[: -len("/observations")])

    h5.visititems(visitor)
    return sorted(set(roots))


def _dataset_path(root: str, relative: str) -> str:
    return "/" + relative if root == "/" else root.rstrip("/") + "/" + relative


def _load_episode(h5: h5py.File, root: str) -> dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_DATASETS if _dataset_path(root, name) not in h5]
    if missing:
        raise ValueError("missing required datasets: " + ", ".join(missing))
    arrays = {
        name.rsplit("/", 1)[-1]: np.asarray(h5[_dataset_path(root, name)]).reshape(-1)
        for name in REQUIRED_DATASETS
    }
    lengths = {name: array.size for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"dataset lengths differ: {lengths}")
    times = np.asarray(arrays["timestamps"], dtype=np.float64)
    if times.size == 0:
        raise ValueError("episode has no observations")
    if not np.all(np.isfinite(times)):
        raise ValueError("observation timestamps contain non-finite values")
    if times.size > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("observation timestamps are not strictly monotonically increasing")
    return arrays


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [key for key in rows[0] if not key.startswith("_")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(row.get(key)) for key in fieldnames})


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _print_episode(metric: Mapping[str, Any]) -> None:
    print(
        f"{metric['episode_id']}: samples={metric['number_of_samples']}, "
        f"duration={metric['episode_duration_sec']:.3f}s, "
        f"presence={100.0 * metric['time_weighted_presence_fraction']:.2f}%, "
        f"gaps={metric['gap_count_including_boundary_gaps']} "
        f"(interior={metric['interior_gap_count_excluding_boundary_gaps']}), "
        f"median/max gap={_fmt(metric['median_gap_duration_ms'], 1)}/"
        f"{_fmt(metric['maximum_gap_duration_ms'], 1)}ms"
    )


def _safe_plot_name(episode_id: str, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", episode_id).strip("._")
    return f"episode_{index:04d}_{stem[-80:] or 'episode'}.svg"


def _write_episode_plot(
    path: Path,
    metric: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    presence_mode: str,
    max_age_ms: float,
) -> None:
    """Write a dependency-free SVG timeline using actual run-time offsets."""
    width, height = 1100, 300
    left, right, timeline_y, timeline_h = 80, 30, 108, 72
    plot_width = width - left - right
    duration = float(metric["episode_duration_sec"])
    title = escape(str(metric["episode_id"]))
    mode_note = f"mode={presence_mode}"
    if presence_mode == "fresh":
        mode_note += f", max age={max_age_ms:g} ms"
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.small{font-size:12px}.label{font-size:14px}.title{font-size:17px;font-weight:bold}</style>',
        f'<text x="{left}" y="28" class="title">Event-detection presence timeline</text>',
        f'<text x="{left}" y="51" class="small">{title}</text>',
        f'<text x="{left}" y="72" class="small">{escape(mode_note)}; actual timestamp-cell duration={duration:.3f} s</text>',
        f'<text x="{left - 10}" y="{timeline_y + timeline_h / 2 + 5}" text-anchor="end" class="label">state</text>',
    ]
    for run in runs:
        start = float(run["start_offset_sec"])
        run_duration = float(run["duration_sec"])
        x = left + plot_width * start / duration
        rect_width = max(0.8, plot_width * run_duration / duration)
        color = "#2e7d32" if run["state"] == "present" else "#d32f2f"
        tooltip = escape(
            f"{run['state']}: {start:.6f}-{start + run_duration:.6f} s "
            f"({run_duration * 1000.0:.2f} ms, {run['sample_count']} samples)"
        )
        elements.append(
            f'<rect x="{x:.3f}" y="{timeline_y}" width="{rect_width:.3f}" '
            f'height="{timeline_h}" fill="{color}"><title>{tooltip}</title></rect>'
        )
    elements.extend(
        [
            f'<rect x="{left}" y="{timeline_y}" width="{plot_width}" height="{timeline_h}" fill="none" stroke="#333"/>',
            f'<text x="{left}" y="{timeline_y + timeline_h + 22}" class="small">0 s</text>',
            f'<text x="{left + plot_width}" y="{timeline_y + timeline_h + 22}" text-anchor="end" class="small">{duration:.3f} s</text>',
            '<rect x="80" y="226" width="16" height="16" fill="#2e7d32"/><text x="103" y="239" class="small">present</text>',
            '<rect x="180" y="226" width="16" height="16" fill="#d32f2f"/><text x="203" y="239" class="small">absent / gap</text>',
            f'<text x="360" y="239" class="small">presence={100.0 * float(metric["time_weighted_presence_fraction"]):.2f}% · gaps={metric["gap_count_including_boundary_gaps"]} · max gap={_fmt(metric["maximum_gap_duration_ms"], 1)} ms</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_aggregate_plot(
    path: Path,
    metrics: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    presence_mode: str,
    max_age_ms: float,
) -> None:
    """Write an SVG dashboard of aggregate presence and dropout behavior."""
    width, height = 1200, 700
    presence = float(aggregate["time_weighted_presence_fraction"])
    gap_ms = np.asarray(
        [float(run["duration_sec"]) * 1000.0 for run in runs if run["state"] == "absent"],
        dtype=np.float64,
    )
    counts = np.array(
        [
            np.count_nonzero(gap_ms <= 33.0),
            np.count_nonzero((gap_ms > 33.0) & (gap_ms <= 50.0)),
            np.count_nonzero((gap_ms > 50.0) & (gap_ms <= 100.0)),
            np.count_nonzero((gap_ms > 100.0) & (gap_ms <= 200.0)),
            np.count_nonzero(gap_ms > 200.0),
        ],
        dtype=int,
    )
    labels = ("≤33", "33–50", "50–100", "100–200", ">200")
    mode_note = f"mode={presence_mode}" + (
        f", max age={max_age_ms:g} ms" if presence_mode == "fresh" else ""
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:bold}.section{font-size:16px;font-weight:bold}</style>',
        '<text x="50" y="34" class="title">Event-detection aggregate analysis</text>',
        f'<text x="50" y="58" class="label">{escape(mode_note)} · episodes={len(metrics)} · duration={float(aggregate["total_observation_duration_sec"]):.3f} s</text>',
        '<text x="50" y="98" class="section">Time-weighted presence</text>',
    ]
    # Stacked presence/absence bar is less ambiguous than a pie and preserves exact proportions.
    bar_x, bar_y, bar_w, bar_h = 50, 120, 500, 65
    present_w = bar_w * presence
    elements.extend(
        [
            f'<rect x="{bar_x}" y="{bar_y}" width="{present_w:.3f}" height="{bar_h}" fill="#2e7d32"/>',
            f'<rect x="{bar_x + present_w:.3f}" y="{bar_y}" width="{bar_w - present_w:.3f}" height="{bar_h}" fill="#d32f2f"/>',
            f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" fill="none" stroke="#333"/>',
            f'<text x="{bar_x + 8}" y="{bar_y + 40}" fill="white" style="fill:white;font-weight:bold">present {presence * 100:.2f}%</text>',
            f'<text x="{bar_x + bar_w - 8}" y="{bar_y + 40}" text-anchor="end" fill="white" style="fill:white;font-weight:bold">absent {(1-presence) * 100:.2f}%</text>',
            '<text x="650" y="98" class="section">Gap-duration distribution (ms)</text>',
        ]
    )
    hist_x, hist_y, hist_w, hist_h = 650, 120, 490, 170
    max_count = max(int(np.max(counts)), 1)
    each_w = hist_w / len(counts)
    for index, (label, count) in enumerate(zip(labels, counts)):
        h = hist_h * int(count) / max_count
        x = hist_x + index * each_w + 12
        y = hist_y + hist_h - h
        elements.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{each_w - 24:.2f}" height="{h:.2f}" fill="#ef6c00"><title>{escape(label)} ms: {int(count)} gaps</title></rect>',
                f'<text x="{x + (each_w - 24) / 2:.2f}" y="{hist_y + hist_h + 18}" text-anchor="middle" class="small">{escape(label)}</text>',
                f'<text x="{x + (each_w - 24) / 2:.2f}" y="{max(y - 5, hist_y + 12):.2f}" text-anchor="middle" class="small">{int(count)}</text>',
            ]
        )
    elements.append(f'<line x1="{hist_x}" y1="{hist_y + hist_h}" x2="{hist_x + hist_w}" y2="{hist_y + hist_h}" stroke="#333"/>')
    elements.append('<text x="50" y="245" class="section">Episode presence fractions</text>')
    chart_x, chart_y, chart_w, chart_h = 50, 270, 1090, 320
    n = len(metrics)
    slot = chart_w / max(n, 1)
    for index, metric in enumerate(metrics):
        value = float(metric["time_weighted_presence_fraction"])
        height_px = chart_h * value
        x = chart_x + index * slot + min(5.0, slot * 0.1)
        rect_w = max(1.0, slot - min(10.0, slot * 0.2))
        title = escape(f"{metric['episode_id']}: {value * 100:.2f}%")
        elements.append(
            f'<rect x="{x:.2f}" y="{chart_y + chart_h - height_px:.2f}" width="{rect_w:.2f}" '
            f'height="{height_px:.2f}" fill="#1565c0"><title>{title}</title></rect>'
        )
        if n <= 30:
            elements.append(
                f'<text x="{x + rect_w / 2:.2f}" y="{chart_y + chart_h + 17}" text-anchor="middle" class="small">{index + 1}</text>'
            )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = chart_y + chart_h * (1.0 - fraction)
        elements.extend(
            [
                f'<line x1="{chart_x}" y1="{y:.2f}" x2="{chart_x + chart_w}" y2="{y:.2f}" stroke="#bbb" stroke-dasharray="3 4"/>',
                f'<text x="{chart_x - 8}" y="{y + 4:.2f}" text-anchor="end" class="small">{fraction * 100:.0f}%</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="50" y="660" class="small">Pooled gaps: {aggregate["pooled_total_gap_count"]}; median/p90/p95/max: '
            f'{_fmt(aggregate["pooled_all_gaps"]["median_ms"], 1)} / {_fmt(aggregate["pooled_all_gaps"]["p90_ms"], 1)} / '
            f'{_fmt(aggregate["pooled_all_gaps"]["p95_ms"], 1)} / {_fmt(aggregate["pooled_all_gaps"]["max_ms"], 1)} ms</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_plots(
    plot_dir: Path,
    episode_metrics: Sequence[Mapping[str, Any]],
    all_runs: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    presence_mode: str,
    max_age_ms: float,
) -> list[str]:
    """Create per-episode timelines and an aggregate SVG dashboard."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, metric in enumerate(episode_metrics):
        episode_runs = [run for run in all_runs if run["episode_id"] == metric["episode_id"]]
        path = plot_dir / _safe_plot_name(str(metric["episode_id"]), index)
        _write_episode_plot(path, metric, episode_runs, presence_mode, max_age_ms)
        paths.append(str(path))
    aggregate_path = plot_dir / "aggregate_summary.svg"
    _write_aggregate_plot(
        aggregate_path, episode_metrics, all_runs, aggregate, presence_mode, max_age_ms
    )
    paths.append(str(aggregate_path))
    return paths


def stable_sort_sidecar_rows(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], bool]:
    """Stable-sort sidecar rows by ROS time, packet id, and original row index."""
    result = {key: np.asarray(value).reshape(-1) for key, value in arrays.items()}
    times = np.asarray(result["available_ros_t_ns"], dtype=np.float64)
    packet = np.asarray(result.get("packet_id", np.zeros(times.size)), dtype=np.float64)
    nonmonotonic = bool(times.size > 1 and np.any(np.diff(times) < 0))
    if nonmonotonic:
        packet_key = np.where(np.isfinite(packet), packet, np.inf)
        order = np.lexsort((np.arange(times.size), packet_key, times))
        result = {key: value[order] for key, value in result.items()}
    return result, nonmonotonic


def validate_sidecar_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """Validate one loaded sidecar group without changing row multiplicity."""
    missing = [name for name in SIDECAR_CORE_DATASETS if name not in arrays]
    if missing:
        raise ValueError("missing required sidecar datasets: " + ", ".join(missing))
    dimensions = {name: np.asarray(value).ndim for name, value in arrays.items()}
    wrong = [name for name, ndim in dimensions.items() if ndim != 1]
    if wrong:
        raise ValueError("sidecar datasets must be one-dimensional: " + ", ".join(wrong))
    lengths = {name: len(np.asarray(value)) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"sidecar dataset lengths differ: {lengths}")
    times = np.asarray(arrays["available_ros_t_ns"], dtype=np.float64)
    if not np.all(np.isfinite(times)):
        raise ValueError("available_ros_t_ns contains non-finite values")
    packet = np.asarray(arrays["packet_id"], dtype=np.float64)
    if not np.all(np.isfinite(packet)):
        raise ValueError("packet_id contains non-finite values")
    starts = np.asarray(arrays["sensor_window_start_us"], dtype=np.float64)
    ends = np.asarray(arrays["sensor_window_end_us"], dtype=np.float64)
    if np.any(np.isfinite(starts) & np.isfinite(ends) & (ends < starts)):
        raise ValueError("sensor window end is earlier than its start")


def load_sidecar_group(path: Path | str, group_name: str) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
    """Load, validate, and time-sort a sidecar episode group."""
    with h5py.File(path, "r") as h5:
        if group_name not in h5 or not isinstance(h5[group_name], h5py.Group):
            raise ValueError(f"sidecar group does not exist: {group_name}")
        group = h5[group_name]
        names = SIDECAR_CORE_DATASETS + SIDECAR_OPTIONAL_DATASETS
        arrays = {name: np.asarray(group[name]) for name in names if name in group}
        attrs = {str(key): _json_value(value) for key, value in group.attrs.items()}
        for key, value in h5.attrs.items():
            attrs.setdefault(str(key), _json_value(value))
    validate_sidecar_arrays(arrays)
    sorted_arrays, reordered = stable_sort_sidecar_rows(arrays)
    return sorted_arrays, attrs, reordered


def sidecar_group_names(path: Path | str) -> list[str]:
    """Return groups that look like native tracker episode groups."""
    with h5py.File(path, "r") as h5:
        if "episodes" not in h5 or not isinstance(h5["episodes"], h5py.Group):
            return []
        return [f"/episodes/{name}" for name in sorted(h5["episodes"])
                if isinstance(h5[f"/episodes/{name}"], h5py.Group)]


def match_episode_to_sidecar_group(
    episode_path: Path | str, episode_attrs: Mapping[str, Any],
    groups: Sequence[tuple[str, Mapping[str, Any]]],
) -> str | None:
    """Match an episode using indices, conventional group names, then filename."""
    candidates = []
    for key in ("episode_index", "source_episode_index"):
        if key in episode_attrs:
            try:
                candidates.append(int(episode_attrs[key]))
            except (TypeError, ValueError):
                pass
    for group_name, attrs in groups:
        for key in ("episode_index", "source_episode_index"):
            try:
                if key in attrs and int(attrs[key]) in candidates:
                    return group_name
            except (TypeError, ValueError):
                pass
    for index in candidates:
        wanted = f"episode_{index}"
        for group_name, _ in groups:
            if Path(group_name).name == wanted:
                return group_name
    stem = Path(episode_path).stem
    for group_name, attrs in groups:
        filename_values = [value for key, value in attrs.items()
                           if "file" in str(key).lower() or "path" in str(key).lower()]
        if Path(group_name).name == stem or any(Path(str(value)).stem == stem for value in filename_values):
            return group_name
    return None


def extract_active_interval(arrays: Mapping[str, np.ndarray]) -> tuple[int, int] | None:
    """Return inclusive row bounds from first through last valid row."""
    indices = np.flatnonzero(np.asarray(arrays["valid"]).reshape(-1) != 0)
    return None if indices.size == 0 else (int(indices[0]), int(indices[-1]))


def _unique_valid_indices(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    valid_indices = np.flatnonzero(np.asarray(arrays["valid"]).reshape(-1) != 0)
    if valid_indices.size == 0:
        return valid_indices
    key_name = "packet_id" if "packet_id" in arrays else "available_ros_t_ns"
    values = np.asarray(arrays[key_name])[valid_indices]
    seen: set[Any] = set()
    kept = []
    for index, value in zip(valid_indices, values):
        key = value.item() if isinstance(value, np.generic) else value
        if key not in seen:
            seen.add(key)
            kept.append(int(index))
    return np.asarray(kept, dtype=np.int64)


def calculate_valid_detection_gaps(
    arrays: Mapping[str, np.ndarray], episode_id: str = ""
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Calculate gaps between unique recorded valid tracker outputs."""
    indices = _unique_valid_indices(arrays)
    times = np.asarray(arrays["available_ros_t_ns"], dtype=np.float64) * 1e-9
    packets = np.asarray(arrays.get("packet_id", np.full(times.size, np.nan)))
    rows = []
    if indices.size:
        origin = float(times[indices[0]])
        for detection_index, (previous, following) in enumerate(zip(indices[:-1], indices[1:])):
            between = slice(int(previous) + 1, int(following))
            interval = float(times[following] - times[previous])
            middle_packets = packets[between]
            rows.append({
                "episode_id": episode_id, "detection_index": detection_index,
                "previous_detection_time": float(times[previous]),
                "next_detection_time": float(times[following]),
                "previous_detection_offset_sec": float(times[previous] - origin),
                "next_detection_offset_sec": float(times[following] - origin),
                "inter_detection_interval_sec": interval,
                "inter_detection_interval_ms": interval * 1000.0,
                "sidecar_rows_between_detections": int(following - previous - 1),
                "invalid_rows_between_detections": int(np.count_nonzero(np.asarray(arrays["valid"])[between] == 0)),
                "unique_packets_between_detections": int(np.unique(middle_packets).size),
            })
    return rows, indices


def sidecar_run_length_encode(arrays: Mapping[str, np.ndarray], episode_id: str = "") -> list[dict[str, Any]]:
    """RLE the original active-interval row sequence (not continuous time)."""
    bounds = extract_active_interval(arrays)
    if bounds is None:
        return []
    lo, hi = bounds
    valid = np.asarray(arrays["valid"]).reshape(-1) != 0
    times = np.asarray(arrays["available_ros_t_ns"], dtype=np.float64) * 1e-9
    packet = np.asarray(arrays.get("packet_id", np.full(valid.size, np.nan)))
    active = valid[lo:hi + 1]
    starts = np.r_[0, np.flatnonzero(active[1:] != active[:-1]) + 1]
    ends = np.r_[starts[1:], active.size]
    rows = []
    for run_index, (start, end) in enumerate(zip(starts, ends)):
        first, stop = lo + int(start), lo + int(end)
        span = float(times[stop - 1] - times[first])
        rows.append({
            "episode_id": episode_id, "signal_source": "sidecar_rows", "run_index": run_index,
            "state": "present" if valid[first] else "absent", "start_row_index": first,
            "end_row_index_exclusive": stop, "row_count": stop - first,
            "first_row_timestamp": float(times[first]), "last_row_timestamp": float(times[stop - 1]),
            "row_span_sec": span, "row_span_ms": span * 1000.0,
            "first_row_packet_id": _json_value(packet[first]), "last_row_packet_id": _json_value(packet[stop - 1]),
            "invalid_row_count": int(np.count_nonzero(~valid[first:stop])),
        })
    return rows


def segment_dense_phases(
    arrays: Mapping[str, np.ndarray], episode_id: str = "", phase_gap_ms: float = 250.0,
    min_phase_detections: int = 3,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Split unique valid detections at gaps larger than phase_gap_ms."""
    _, indices = calculate_valid_detection_gaps(arrays, episode_id)
    if indices.size == 0:
        return [], indices
    times = np.asarray(arrays["available_ros_t_ns"], dtype=np.float64) * 1e-9
    split = np.flatnonzero(np.diff(times[indices]) > phase_gap_ms / 1000.0) + 1
    chunks = np.split(indices, split)
    valid = np.asarray(arrays["valid"]).reshape(-1) != 0
    packet = np.asarray(arrays.get("packet_id", np.full(valid.size, np.nan)))
    origin = float(times[indices[0]])
    phases = []
    for phase_id, chunk in enumerate(chunks):
        first, last = int(chunk[0]), int(chunk[-1])
        phase_times = times[chunk]
        intervals_ms = np.diff(phase_times) * 1000.0
        span = float(phase_times[-1] - phase_times[0])
        count = int(chunk.size)
        rows = slice(first, last + 1)
        phases.append({
            "episode_id": episode_id, "phase_id": phase_id,
            "is_dense": count >= min_phase_detections,
            "first_detection_time": float(phase_times[0]), "last_detection_time": float(phase_times[-1]),
            "phase_start_offset_sec": float(phase_times[0] - origin),
            "phase_end_offset_sec": float(phase_times[-1] - origin), "phase_span_sec": span,
            "valid_detection_count": count, "unique_valid_packet_count": int(np.unique(packet[chunk]).size),
            "interval_count": max(count - 1, 0),
            "interval_rate_hz": (count - 1) / span if count >= 2 and span > 0 else None,
            "endpoint_count_rate_hz": count / span if span > 0 else None,
            "median_inter_detection_interval_ms": _percentile(intervals_ms.tolist(), 50),
            "p90_inter_detection_interval_ms": _percentile(intervals_ms.tolist(), 90),
            "maximum_inter_detection_interval_ms": float(np.max(intervals_ms)) if intervals_ms.size else None,
            "invalid_sidecar_rows_in_phase": int(np.count_nonzero(~valid[rows])),
            "sidecar_rows_in_phase": last - first + 1,
            "valid_row_fraction_in_phase": float(np.mean(valid[rows])),
            "_detection_indices": chunk,
        })
    return phases, indices


def calculate_phase_window_rates(
    phases: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray], rate_window_ms: float = 250.0
) -> list[dict[str, Any]]:
    """Partition dense phases into non-overlapping, phase-anchored rate windows."""
    times = np.asarray(arrays["available_ros_t_ns"], dtype=np.float64) * 1e-9
    packet = np.asarray(arrays.get("packet_id", np.full(times.size, np.nan)))
    width = rate_window_ms / 1000.0
    rows = []
    for phase in phases:
        if not phase["is_dense"]:
            continue
        indices = np.asarray(phase["_detection_indices"], dtype=np.int64)
        start, last = float(times[indices[0]]), float(times[indices[-1]])
        count_windows = max(1, int(math.floor(max(last - start, 0.0) / width)) + 1)
        for window_index in range(count_windows):
            window_start = start + window_index * width
            window_end = window_start + width
            mask = (times[indices] >= window_start) & (times[indices] < window_end)
            selected = indices[mask]
            rows.append({
                "episode_id": phase["episode_id"], "phase_id": phase["phase_id"],
                "window_index": window_index, "window_start_time": window_start,
                "window_end_time": window_end, "window_start_offset_sec": window_start - start,
                "window_end_offset_sec": window_end - start, "window_duration_sec": width,
                "valid_detection_count": int(selected.size),
                "unique_valid_packet_count": int(np.unique(packet[selected]).size),
                "detection_rate_hz": float(selected.size / width),
            })
    return rows


def _numeric_stats(values: Sequence[Any]) -> dict[str, Any]:
    raw = np.asarray(values).reshape(-1)
    try:
        numeric = np.asarray(raw, dtype=np.float64)
        finite = numeric[np.isfinite(numeric)]
    except (TypeError, ValueError):
        finite = np.asarray([], dtype=np.float64)
    return {
        "count": int(raw.size), "valid_finite_count": int(finite.size),
        "median": _percentile(finite.tolist(), 50),
        "mean": float(np.mean(finite)) if finite.size else None,
        "p90": _percentile(finite.tolist(), 90), "p95": _percentile(finite.tolist(), 95),
        "minimum": float(np.min(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def detection_window_statistics(arrays: Mapping[str, np.ndarray], episode_id: str = "") -> list[dict[str, Any]]:
    """Return tidy per-field statistics for observed sidecar detection windows."""
    derived = dict(arrays)
    derived["sensor_window_duration_ms"] = (
        np.asarray(arrays["sensor_window_end_us"], dtype=np.float64)
        - np.asarray(arrays["sensor_window_start_us"], dtype=np.float64)
    ) / 1000.0
    rows = []
    for field in SIDECAR_NUMERIC_FIELDS:
        if field in derived:
            rows.append({"episode_id": episode_id, "field": field, **_numeric_stats(derived[field])})
    return rows


def _interval_distribution(intervals_sec: Sequence[float], prefix: str) -> dict[str, Any]:
    values = np.asarray(intervals_sec, dtype=np.float64)
    stats = _numeric_stats(values)
    result = {f"{prefix}_interval_{key}_sec": value for key, value in stats.items()
              if key not in ("count", "valid_finite_count")}
    result[f"{prefix}_interval_count"] = int(values.size)
    median = stats["median"]
    result[f"{prefix}_observed_rate_hz"] = 1.0 / median if median is not None and median > 0 else None
    return result


def summarize_sidecar_episode(
    arrays: Mapping[str, np.ndarray], episode_id: str, sidecar_path: Path | str,
    sidecar_group: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize native tracker rows and gaps for one sidecar episode."""
    times = np.asarray(arrays["available_ros_t_ns"], dtype=np.float64) * 1e-9
    valid = np.asarray(arrays["valid"]).reshape(-1) != 0
    packets = np.asarray(arrays["packet_id"])
    gaps, unique_indices = calculate_valid_detection_gaps(arrays, episode_id)
    runs = sidecar_run_length_encode(arrays, episode_id)
    metric: dict[str, Any] = {
        "episode_id": episode_id, "sidecar_path": str(sidecar_path), "sidecar_group": sidecar_group,
        "total_sidecar_rows": int(times.size), "valid_row_count": int(np.count_nonzero(valid)),
        "invalid_row_count": int(np.count_nonzero(~valid)),
        "valid_row_fraction": float(np.mean(valid)) if times.size else None,
        "invalid_row_fraction": float(np.mean(~valid)) if times.size else None,
        "unique_packet_count": int(np.unique(packets).size),
        "unique_timestamp_count": int(np.unique(times).size),
        "first_sidecar_row_time": float(times[0]) if times.size else None,
        "last_sidecar_row_time": float(times[-1]) if times.size else None,
        "sidecar_row_span_sec": float(times[-1] - times[0]) if times.size else None,
        "sidecar_update_rate_hz": ((times.size - 1) / (times[-1] - times[0])
                                   if times.size > 1 and times[-1] > times[0] else None),
        "valid_detection_count": int(unique_indices.size),
        "valid_unique_packet_count": int(np.unique(packets[unique_indices]).size),
    }
    if unique_indices.size:
        first, last = float(times[unique_indices[0]]), float(times[unique_indices[-1]])
        metric.update({
            "first_valid_detection_time": first, "last_valid_detection_time": last,
            "first_valid_detection_offset_sec": first - float(times[0]),
            "last_valid_detection_offset_sec": last - float(times[0]),
            "valid_detection_span_sec": last - first,
            "active_interval_start": first, "active_interval_end": last,
            "active_interval_duration_sec": last - first,
        })
        lo, hi = extract_active_interval(arrays)  # type: ignore[misc]
        active_valid = valid[lo:hi + 1]
        metric.update({
            "active_interval_sidecar_row_count": int(active_valid.size),
            "active_interval_valid_row_count": int(np.count_nonzero(active_valid)),
            "active_interval_invalid_row_count": int(np.count_nonzero(~active_valid)),
            "active_interval_valid_row_fraction": float(np.mean(active_valid)),
            "number_of_present_runs": sum(run["state"] == "present" for run in runs),
            "number_of_absent_runs": sum(run["state"] == "absent" for run in runs),
            "longest_present_run_rows": max((run["row_count"] for run in runs if run["state"] == "present"), default=0),
            "longest_absent_run_rows": max((run["row_count"] for run in runs if run["state"] == "absent"), default=0),
            "longest_present_run_ms": max((run["row_span_ms"] for run in runs if run["state"] == "present"), default=0.0),
        })
    else:
        for key in ("first_valid_detection_time", "last_valid_detection_time",
                    "first_valid_detection_offset_sec", "last_valid_detection_offset_sec",
                    "valid_detection_span_sec", "active_interval_start", "active_interval_end",
                    "active_interval_duration_sec", "active_interval_sidecar_row_count",
                    "active_interval_valid_row_count", "active_interval_invalid_row_count",
                    "active_interval_valid_row_fraction", "number_of_present_runs",
                    "number_of_absent_runs", "longest_present_run_rows", "longest_absent_run_rows"):
            metric[key] = None
        metric["longest_present_run_ms"] = None
    gap_ms = [float(row["inter_detection_interval_ms"]) for row in gaps]
    gap_stats = _numeric_stats(gap_ms)
    metric.update({
        "inter_detection_count": len(gaps),
        "median_inter_detection_interval_ms": gap_stats["median"],
        "mean_inter_detection_interval_ms": gap_stats["mean"],
        "p90_inter_detection_interval_ms": gap_stats["p90"],
        "p95_inter_detection_interval_ms": gap_stats["p95"],
        "minimum_inter_detection_interval_ms": gap_stats["minimum"],
        "maximum_inter_detection_interval_ms": gap_stats["maximum"],
    })
    for threshold in SIDECAR_GAP_THRESHOLDS_MS:
        count = sum(value > threshold for value in gap_ms)
        metric[f"gaps_gt_{int(threshold)}ms_count"] = count
        metric[f"gaps_gt_{int(threshold)}ms_fraction"] = count / len(gap_ms) if gap_ms else None
    metric.update(_interval_distribution(np.diff(times), "all_sidecar_row"))
    metric.update(_interval_distribution(np.diff(times[valid]), "valid_sidecar_row"))
    metric.update(_interval_distribution(np.diff(times[~valid]), "invalid_sidecar_row"))
    return metric, gaps, runs


def summarize_sidecar_aggregate(
    metrics: Sequence[Mapping[str, Any]], intervals: Sequence[Mapping[str, Any]],
    phases: Sequence[Mapping[str, Any]], windows: Sequence[Mapping[str, Any]],
    window_stats: Sequence[Mapping[str, Any]], sidecar_paths: Sequence[str],
    discovered_count: int, skipped_count: int,
) -> dict[str, Any]:
    """Combine episode-weighted summaries and explicitly pooled observations."""
    def metric_values(name: str) -> list[float]:
        return [float(row[name]) for row in metrics if row.get(name) is not None]
    pooled_gaps = [float(row["inter_detection_interval_ms"]) for row in intervals]
    pooled_phase_rates = [float(row["interval_rate_hz"]) for row in phases
                          if row.get("is_dense") and row.get("interval_rate_hz") is not None]
    pooled_window_rates = [float(row["detection_rate_hz"]) for row in windows]
    all_update_ms = []
    for metric in metrics:
        # Reconstructing the pooled distribution is impossible from summaries, so callers
        # attach it transiently when exact row-level intervals are available.
        all_update_ms.extend(metric.get("_all_update_intervals_ms", []))

    def field_values(field: str) -> list[float]:
        return [float(value) for row in window_stats if row["field"] == field
                for value in row.get("_finite_values", [])]
    # _finite_values is optional; orchestration supplies exact pools.
    sensor = field_values("sensor_window_duration_ms")
    events = field_values("window_event_count")
    dense_counts = [sum(bool(p["is_dense"]) and p["episode_id"] == m["episode_id"] for p in phases)
                    for m in metrics]
    return {
        "analysis_source": "sidecar", "sidecar_paths": sorted(set(sidecar_paths)),
        "number_of_discovered_files": int(discovered_count),
        "number_of_analyzed_episodes": len(metrics), "number_of_skipped_episodes": int(skipped_count),
        "episodes_without_valid_detections": sum(int(m["valid_detection_count"]) == 0 for m in metrics),
        "total_sidecar_rows": sum(int(m["total_sidecar_rows"]) for m in metrics),
        "total_valid_detection_count": sum(int(m["valid_detection_count"]) for m in metrics),
        "median_valid_detection_count_per_episode": _percentile(metric_values("valid_detection_count"), 50),
        "median_active_span_sec": _percentile(metric_values("valid_detection_span_sec"), 50),
        "p25_active_span_sec": _percentile(metric_values("valid_detection_span_sec"), 25),
        "p75_active_span_sec": _percentile(metric_values("valid_detection_span_sec"), 75),
        "median_inter_detection_interval_ms": _percentile(pooled_gaps, 50),
        "p90_inter_detection_interval_ms": _percentile(pooled_gaps, 90),
        "p95_inter_detection_interval_ms": _percentile(pooled_gaps, 95),
        "maximum_inter_detection_interval_ms": max(pooled_gaps) if pooled_gaps else None,
        "median_sidecar_update_interval_ms": _percentile(all_update_ms, 50),
        "p90_sidecar_update_interval_ms": _percentile(all_update_ms, 90),
        "p95_sidecar_update_interval_ms": _percentile(all_update_ms, 95),
        "median_sensor_window_duration_ms": _percentile(sensor, 50),
        "p90_sensor_window_duration_ms": _percentile(sensor, 90),
        "median_window_event_count": _percentile(events, 50), "p90_window_event_count": _percentile(events, 90),
        "median_dense_phase_count_per_episode": _percentile(dense_counts, 50),
        "median_phase_interval_rate_hz": _percentile(pooled_phase_rates, 50),
        "p90_phase_interval_rate_hz": _percentile(pooled_phase_rates, 90),
        "pooled_inter_detection_intervals": _numeric_stats(pooled_gaps),
        "pooled_phase_rates": _numeric_stats(pooled_phase_rates),
        "pooled_phase_window_rates": _numeric_stats(pooled_window_rates),
        "weighting_note": "Per-episode medians are episode-weighted; pooled distributions are row/interval-level.",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify event ball-detection presence and dropouts in ACT HDF5 episodes. "
            "Episodes are sampled at approximately 30 Hz, so this measures policy-grid-visible "
            "presence. Native event-tracker gaps faster than the observation grid require "
            "analysis of the tracker sidecar."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path, help="single episode/container .hdf5 file")
    source.add_argument("--top-dir", type=Path, help="directory containing .hdf5 files")
    parser.add_argument("--recursive", action="store_true", help="recursively discover HDF5 files")
    parser.add_argument("--max-files", type=int, help="analyze at most M discovered files")
    parser.add_argument("--output-dir", type=Path, help="directory for CSV and JSON reports")
    parser.add_argument(
        "--analysis-source", choices=("episode", "sidecar", "auto"), default="episode",
        help="episode policy-grid analysis (default), native sidecar rows, or automatic sidecar fallback",
    )
    parser.add_argument("--sidecar", type=Path, help="tracker-output sidecar HDF5 path")
    parser.add_argument("--phase-gap-ms", type=float, default=250.0,
                        help="gap larger than this starts a new sidecar phase (default: 250)")
    parser.add_argument("--min-phase-detections", type=int, default=3,
                        help="minimum detections marking a phase dense (default: 3)")
    parser.add_argument("--rate-window-ms", type=float, default=250.0,
                        help="dense-phase rate window width (default: 250)")
    parser.add_argument(
        "--plot", action="store_true",
        help="create dependency-free SVG timelines and summary charts under OUTPUT_DIR/plots",
    )
    parser.add_argument(
        "--presence-mode", choices=("valid", "fresh", "new_update"), default="fresh",
        help="presence signal (default: fresh; valid may include held tracker values)",
    )
    parser.add_argument(
        "--max-age-ms", type=float, default=50.0,
        help="maximum finite source age for fresh mode in milliseconds (default: 50)",
    )
    parser.add_argument(
        "--timestamp-epsilon", type=float, default=1e-9,
        help="minimum source-timestamp advance for new_update mode in seconds (default: 1e-9)",
    )
    args = parser.parse_args(argv)
    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max-files must be positive")
    if args.max_age_ms < 0 or not math.isfinite(args.max_age_ms):
        parser.error("--max-age-ms must be finite and nonnegative")
    if args.timestamp_epsilon < 0 or not math.isfinite(args.timestamp_epsilon):
        parser.error("--timestamp-epsilon must be finite and nonnegative")
    if args.phase_gap_ms < 0 or not math.isfinite(args.phase_gap_ms):
        parser.error("--phase-gap-ms must be finite and nonnegative")
    if args.min_phase_detections <= 0:
        parser.error("--min-phase-detections must be positive")
    if args.rate_window_ms <= 0 or not math.isfinite(args.rate_window_ms):
        parser.error("--rate-window-ms must be finite and positive")
    if args.recursive and args.top_dir is None:
        parser.error("--recursive requires --top-dir")
    return args


def _episode_attributes(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        return {str(key): _json_value(value) for key, value in h5.attrs.items()}


def _resolve_sidecar_path(episode_path: Path, attrs: Mapping[str, Any], explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    value = attrs.get("sparse_tracking_sidecar_h5")
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = episode_path.parent / candidate
    return candidate.resolve()


def _sidecar_group_metadata(path: Path) -> list[tuple[str, dict[str, Any]]]:
    groups = []
    with h5py.File(path, "r") as h5:
        for name in sidecar_group_names(path):
            attrs = {str(key): _json_value(value) for key, value in h5[name].attrs.items()}
            groups.append((name, attrs))
    return groups


def _write_rows_even_if_empty(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(rows[0].keys()) if rows else list(fields)
        fieldnames = [name for name in fieldnames if not name.startswith("_")]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _json_value(row.get(name)) for name in fieldnames})


def _parse_tracker_config(attrs: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = attrs.get("sparse_tracking_tracker_config_json")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        config = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"parse_error": "invalid sparse_tracking_tracker_config_json"}

    def selected(mapping: Mapping[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in mapping.items():
            lower = str(key).lower()
            if key in ("bin_ms", "accumulation_window_ms") or any(
                token in lower for token in ("morph", "spatial", "velocity", "history", "threshold")
            ):
                result[str(key)] = value
            if isinstance(value, Mapping):
                nested = selected(value)
                if nested:
                    result.setdefault(str(key), {}).update(nested)
        return result
    return selected(config) if isinstance(config, Mapping) else {"value": config}


def _sidecar_rejection_rows(arrays: Mapping[str, np.ndarray], episode_id: str) -> list[dict[str, Any]]:
    if "rejection_reason" not in arrays:
        return []
    values = np.asarray(arrays["rejection_reason"]).reshape(-1)
    decoded = [value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
               for value in values]
    unique, counts = np.unique(decoded, return_counts=True)
    return [{"episode_id": episode_id, "rejection_reason": reason, "count": int(count),
             "fraction_of_sidecar_rows": float(count / len(decoded)) if decoded else None}
            for reason, count in zip(unique, counts)]


def _simple_sidecar_svg(path: Path, title: str, subtitle: str, values: Sequence[float] = ()) -> None:
    width, height = 1000, 260
    vals = np.asarray(values, dtype=np.float64)
    bars = []
    if vals.size:
        counts, _ = np.histogram(vals, bins=min(30, max(1, int(np.sqrt(vals.size)))))
        maximum = max(int(np.max(counts)), 1)
        for index, count in enumerate(counts):
            x = 70 + 880 * index / len(counts)
            w = max(1, 880 / len(counts) - 2)
            h = 140 * count / maximum
            bars.append(f'<rect x="{x:.2f}" y="{220-h:.2f}" width="{w:.2f}" height="{h:.2f}" fill="#1565c0"/>')
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="40" y="35" font-family="Arial" font-size="20" font-weight="bold">{escape(title)}</text>',
               f'<text x="40" y="60" font-family="Arial" font-size="12">{escape(subtitle)}</text>',
               *bars, '</svg>']
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def _sidecar_row_svg(
    path: Path, title: str, subtitle: str, arrays: Mapping[str, np.ndarray],
    episode_phases: Sequence[Mapping[str, Any]], row_order_axis: bool = False,
) -> None:
    """Draw actual valid/invalid sidecar markers and active/phase boundaries."""
    times = np.asarray(arrays["available_ros_t_ns"], dtype=np.float64) * 1e-9
    valid = np.asarray(arrays["valid"]).reshape(-1) != 0
    if row_order_axis:
        xvalues = np.arange(times.size, dtype=float)
        axis_note = "row index (duplicate rows preserved)"
    else:
        xvalues = times
        axis_note = "available ROS time"
    lo, hi = (float(xvalues[0]), float(xvalues[-1])) if xvalues.size else (0.0, 1.0)
    scale = max(hi - lo, 1.0)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="280">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="45" y="30" font-family="Arial" font-size="20" font-weight="bold">{escape(title)}</text>',
        f'<text x="45" y="53" font-family="Arial" font-size="12">{escape(subtitle)}; x={axis_note}</text>',
        '<line x1="65" y1="215" x2="1040" y2="215" stroke="#555"/>',
    ]
    for index, (xvalue, is_valid) in enumerate(zip(xvalues, valid)):
        x = 65 + 975 * (float(xvalue) - lo) / scale
        y = 105 if is_valid else 185
        color = "#2e7d32" if is_valid else "#d32f2f"
        tooltip = escape(f"row={index}, t={times[index]:.9f}, state={'present' if is_valid else 'absent'}")
        elements.append(f'<circle cx="{x:.2f}" cy="{y}" r="4" fill="{color}"><title>{tooltip}</title></circle>')
    active = extract_active_interval(arrays)
    if active is not None:
        for index in active:
            x = 65 + 975 * (float(xvalues[index]) - lo) / scale
            elements.append(f'<line x1="{x:.2f}" y1="75" x2="{x:.2f}" y2="220" stroke="#6a1b9a" stroke-dasharray="5 4"/>')
    if not row_order_axis:
        for phase in episode_phases:
            if phase["is_dense"]:
                x = 65 + 975 * (float(phase["first_detection_time"]) - lo) / scale
                elements.append(f'<line x1="{x:.2f}" y1="75" x2="{x:.2f}" y2="220" stroke="#ef6c00"/>')
    elements.extend([
        '<text x="15" y="110" font-family="Arial" font-size="12">valid</text>',
        '<text x="15" y="190" font-family="Arial" font-size="12">invalid</text>',
        '<text x="65" y="250" font-family="Arial" font-size="11">Markers are recorded rows, not continuous 1 ms bins.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _sidecar_sensor_window_svg(
    path: Path, episode_id: str, arrays: Mapping[str, np.ndarray],
) -> None:
    """Plot equal-height row bars whose widths encode observed sensor-window spans."""
    starts = np.asarray(arrays["sensor_window_start_us"], dtype=np.float64)
    ends = np.asarray(arrays["sensor_window_end_us"], dtype=np.float64)
    durations_ms = (ends - starts) / 1000.0
    valid = np.asarray(arrays["valid"]).reshape(-1) != 0
    finite = durations_ms[np.isfinite(durations_ms)]
    maximum = float(np.max(finite)) if finite.size and np.max(finite) > 0 else 1.0
    count = durations_ms.size
    slot = 990.0 / max(count, 1)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="280">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="45" y="30" font-family="Arial" font-size="20" font-weight="bold">Observed sensor-window duration</text>',
        f'<text x="45" y="53" font-family="Arial" font-size="12">{escape(episode_id)}; equal-height bars in sidecar-row order, width encodes observed duration</text>',
        '<line x1="65" y1="205" x2="1055" y2="205" stroke="#555"/>',
    ]
    for index, (duration, is_valid) in enumerate(zip(durations_ms, valid)):
        normalized = max(float(duration), 0.0) / maximum if np.isfinite(duration) else 0.0
        bar_width = max(1.0, normalized * max(slot * 0.9, 1.0))
        center = 65 + (index + 0.5) * slot
        x = center - bar_width / 2.0
        color = "#2e7d32" if is_valid else "#d32f2f"
        tooltip = escape(
            f"row={index}, observed sensor-window duration={duration:.3f} ms, "
            f"state={'valid' if is_valid else 'invalid'}"
        )
        elements.append(
            f'<rect x="{x:.2f}" y="105" width="{bar_width:.2f}" height="100" '
            f'fill="{color}"><title>{tooltip}</title></rect>'
        )
    elements.extend([
        '<rect x="65" y="105" width="990" height="100" fill="none" stroke="#333"/>',
        '<rect x="65" y="230" width="14" height="14" fill="#2e7d32"/><text x="85" y="242" font-family="Arial" font-size="11">valid row</text>',
        '<rect x="165" y="230" width="14" height="14" fill="#d32f2f"/><text x="185" y="242" font-family="Arial" font-size="11">invalid row</text>',
        f'<text x="310" y="242" font-family="Arial" font-size="11">maximum observed duration={maximum:.3f} ms; bar widths scaled within this episode</text>',
        '</svg>',
    ])
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_sidecar_aggregate_dashboard(
    path: Path, metrics: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
) -> None:
    """Write an aggregate row-sequence dashboard for native sidecar output."""
    width, height = 1200, 760
    active_valid = sum(int(metric.get("active_interval_valid_row_count") or 0)
                       for metric in metrics)
    active_invalid = sum(int(metric.get("active_interval_invalid_row_count") or 0)
                         for metric in metrics)
    active_rows = active_valid + active_invalid
    valid_fraction = active_valid / active_rows if active_rows else 0.0
    gap_ms = np.asarray(
        [float(row["inter_detection_interval_ms"]) for row in intervals],
        dtype=np.float64,
    )
    counts = np.array([
        np.count_nonzero(gap_ms <= 33.0),
        np.count_nonzero((gap_ms > 33.0) & (gap_ms <= 50.0)),
        np.count_nonzero((gap_ms > 50.0) & (gap_ms <= 100.0)),
        np.count_nonzero((gap_ms > 100.0) & (gap_ms <= 200.0)),
        np.count_nonzero((gap_ms > 200.0) & (gap_ms <= 500.0)),
        np.count_nonzero(gap_ms > 500.0),
    ], dtype=int)
    labels = ("≤33", "33–50", "50–100", "100–200", "200–500", ">500")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:bold}.section{font-size:16px;font-weight:bold}.caveat{font-size:12px;fill:#7f1d1d}</style>',
        '<text x="50" y="34" class="title">Native sidecar tracker-output aggregate analysis</text>',
        f'<text x="50" y="58" class="label">episodes={len(metrics)} · active-interval rows={active_rows} · recorded unique-output gaps={gap_ms.size}</text>',
        '<text x="50" y="98" class="section">Pooled active-interval sidecar-row fraction</text>',
    ]
    bar_x, bar_y, bar_w, bar_h = 50, 120, 500, 65
    valid_w = bar_w * valid_fraction
    elements.extend([
        f'<rect x="{bar_x}" y="{bar_y}" width="{valid_w:.3f}" height="{bar_h}" fill="#2e7d32"/>',
        f'<rect x="{bar_x + valid_w:.3f}" y="{bar_y}" width="{bar_w - valid_w:.3f}" height="{bar_h}" fill="#d32f2f"/>',
        f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" fill="none" stroke="#333"/>',
        f'<text x="{bar_x + 8}" y="{bar_y + 40}" style="fill:white;font-weight:bold">valid {valid_fraction * 100:.2f}% ({active_valid})</text>',
        f'<text x="{bar_x + bar_w - 8}" y="{bar_y + 40}" text-anchor="end" style="fill:white;font-weight:bold">invalid {(1-valid_fraction) * 100:.2f}% ({active_invalid})</text>',
        '<text x="650" y="98" class="section">Recorded valid-output gap distribution (ms)</text>',
    ])
    hist_x, hist_y, hist_w, hist_h = 650, 120, 490, 170
    max_count = max(int(np.max(counts)), 1)
    each_w = hist_w / len(counts)
    for index, (label, count) in enumerate(zip(labels, counts)):
        h = hist_h * int(count) / max_count
        x = hist_x + index * each_w + 8
        y = hist_y + hist_h - h
        elements.extend([
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{each_w - 16:.2f}" height="{h:.2f}" fill="#ef6c00"><title>{escape(label)} ms: {int(count)} gaps</title></rect>',
            f'<text x="{x + (each_w - 16) / 2:.2f}" y="{hist_y + hist_h + 18}" text-anchor="middle" class="small">{escape(label)}</text>',
            f'<text x="{x + (each_w - 16) / 2:.2f}" y="{max(y - 5, hist_y + 12):.2f}" text-anchor="middle" class="small">{int(count)}</text>',
        ])
    elements.extend([
        f'<line x1="{hist_x}" y1="{hist_y + hist_h}" x2="{hist_x + hist_w}" y2="{hist_y + hist_h}" stroke="#333"/>',
        '<text x="50" y="290" class="section">Per-episode active-interval valid-row fractions</text>',
    ])
    chart_x, chart_y, chart_w, chart_h = 50, 315, 1090, 370
    values = [metric.get("active_interval_valid_row_fraction") for metric in metrics]
    slot = chart_w / max(len(values), 1)
    for index, (metric, value) in enumerate(zip(metrics, values)):
        if value is None:
            continue
        fraction = float(value)
        height_px = chart_h * fraction
        x = chart_x + index * slot + min(5.0, slot * 0.1)
        rect_w = max(1.0, slot - min(10.0, slot * 0.2))
        tooltip = escape(f"{metric['episode_id']}: {fraction * 100:.2f}% active valid rows")
        elements.append(
            f'<rect x="{x:.2f}" y="{chart_y + chart_h - height_px:.2f}" width="{rect_w:.2f}" '
            f'height="{height_px:.2f}" fill="#1565c0"><title>{tooltip}</title></rect>'
        )
        if len(values) <= 30:
            elements.append(f'<text x="{x + rect_w / 2:.2f}" y="{chart_y + chart_h + 17}" text-anchor="middle" class="small">{index + 1}</text>')
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = chart_y + chart_h * (1.0 - fraction)
        elements.extend([
            f'<line x1="{chart_x}" y1="{y:.2f}" x2="{chart_x + chart_w}" y2="{y:.2f}" stroke="#bbb" stroke-dasharray="3 4"/>',
            f'<text x="{chart_x - 8}" y="{y + 4:.2f}" text-anchor="end" class="small">{fraction * 100:.0f}%</text>',
        ])
    finite_values = [float(value) for value in values if value is not None]
    if finite_values:
        average = float(np.mean(finite_values))
        average_y = chart_y + chart_h * (1.0 - average)
        elements.extend([
            f'<line x1="{chart_x}" y1="{average_y:.2f}" x2="{chart_x + chart_w}" y2="{average_y:.2f}" stroke="#b71c1c" stroke-width="5"/>',
            f'<text x="{chart_x + chart_w - 4}" y="{average_y - 8:.2f}" text-anchor="end" font-family="Arial" font-size="12" font-weight="bold" fill="#b71c1c" style="fill:#b71c1c">episode mean {average * 100:.2f}%</text>',
        ])
    elements.extend([
        '</svg>',
    ])
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_longest_valid_run_range_plot(
    path: Path, metrics: Sequence[Mapping[str, Any]],
) -> None:
    """Plot episode longest-valid-run timestamp spans in milliseconds."""
    values = np.asarray([
        float(metric["longest_present_run_ms"])
        for metric in metrics if metric.get("longest_present_run_ms") is not None
    ], dtype=np.float64)
    width, height = 1000, 280
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.small{font-size:12px}.title{font-size:20px;font-weight:bold}</style>',
        '<text x="45" y="34" class="title">Longest valid-row-sequence duration per episode</text>',
        '<text x="45" y="58" class="small">Range-and-mean summary; duration is last-row timestamp minus first-row timestamp within each valid run</text>',
    ]
    axis_x, axis_y, axis_w = 80.0, 160.0, 850.0
    if values.size:
        minimum, mean, maximum = float(np.min(values)), float(np.mean(values)), float(np.max(values))
        scale_max = max(maximum, 1.0)

        def x_position(value: float) -> float:
            return axis_x + axis_w * value / scale_max

        min_x, mean_x, max_x = (x_position(value) for value in (minimum, mean, maximum))
        elements.extend([
            f'<line x1="{axis_x}" y1="{axis_y}" x2="{axis_x + axis_w}" y2="{axis_y}" stroke="#555"/>',
            f'<line x1="{min_x:.2f}" y1="{axis_y}" x2="{max_x:.2f}" y2="{axis_y}" stroke="#1565c0" stroke-width="8"/>',
        ])
        for index, value in enumerate(values):
            x = x_position(float(value))
            jitter = ((index % 7) - 3) * 3.0
            elements.append(
                f'<circle cx="{x:.2f}" cy="{axis_y + jitter:.2f}" r="4" fill="#78909c" fill-opacity="0.45"><title>episode longest valid-run span={value:.3f} ms</title></circle>'
            )
        elements.extend([
            f'<line x1="{min_x:.2f}" y1="125" x2="{min_x:.2f}" y2="195" stroke="#0d47a1" stroke-width="3"/>',
            f'<line x1="{max_x:.2f}" y1="125" x2="{max_x:.2f}" y2="195" stroke="#0d47a1" stroke-width="3"/>',
            f'<circle cx="{mean_x:.2f}" cy="{axis_y}" r="11" fill="#b71c1c" stroke="white" stroke-width="2"/>',
            f'<text x="{min_x:.2f}" y="218" text-anchor="middle" class="small">min {minimum:.2f} ms</text>',
            f'<text x="{mean_x:.2f}" y="105" text-anchor="middle" class="small" style="fill:#b71c1c;font-weight:bold">mean {mean:.2f} ms</text>',
            f'<text x="{max_x:.2f}" y="218" text-anchor="middle" class="small">max {maximum:.2f} ms</text>',
            f'<text x="{axis_x + axis_w / 2}" y="255" text-anchor="middle" class="small">Longest valid-row-sequence timestamp span (ms); episodes={values.size}</text>',
        ])
    else:
        elements.append('<text x="500" y="150" text-anchor="middle" class="small">No episode has a valid active interval.</text>')
    elements.append('</svg>')
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_sidecar_plots(
    plot_dir: Path, episodes: Sequence[tuple[str, Mapping[str, np.ndarray], Sequence[Mapping[str, Any]]]],
    metrics: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]], phases: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]], statistics: Sequence[Mapping[str, Any]],
    rejections: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Write dependency-free SVG diagnostics, explicitly identified as row based."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (episode_id, arrays, episode_phases) in enumerate(episodes):
        stem = _safe_plot_name(episode_id, index).removesuffix(".svg")
        specs = (
            ("full_timeline", "Native tracker-output timing", "All valid/invalid sidecar rows; vertical extent is marker state."),
            ("active_zoom", "Valid detection gaps", "First-to-last recorded valid output with dense-phase boundaries."),
            ("binary_row_signal", "Sidecar row-sequence binary signal", "Present/absent rows in row order; not a continuous 1 ms timeline."),
            ("window_diagnostics", "Observed sensor-window duration", "Observed window span and event diagnostics; not CPU latency."),
        )
        for suffix, title, subtitle in specs:
            path = plot_dir / f"{stem}_{suffix}.svg"
            if suffix == "window_diagnostics":
                _sidecar_sensor_window_svg(path, episode_id, arrays)
            else:
                _sidecar_row_svg(path, title, f"{episode_id}: {subtitle}", arrays,
                                 episode_phases, row_order_axis=suffix == "binary_row_signal")
            paths.append(str(path))
    dashboard_path = plot_dir / "sidecar_aggregate_summary.svg"
    _write_sidecar_aggregate_dashboard(dashboard_path, metrics, intervals)
    paths.append(str(dashboard_path))
    longest_run_path = plot_dir / "longest_valid_row_sequence_duration_range_mean.svg"
    _write_longest_valid_run_range_plot(longest_run_path, metrics)
    paths.append(str(longest_run_path))
    update_intervals = [float(value) * 1e-6 for _, arrays, _ in episodes
                        for value in np.diff(np.asarray(arrays["available_ros_t_ns"], dtype=np.float64))]
    sensor_durations = [value for row in statistics if row["field"] == "sensor_window_duration_ms"
                        for value in row.get("_finite_values", [])]
    window_events = [value for row in statistics if row["field"] == "window_event_count"
                     for value in row.get("_finite_values", [])]
    aggregate_specs = (
        ("valid_detection_gap_distribution.svg", "Valid detection gaps", [r["inter_detection_interval_ms"] for r in intervals]),
        ("native_update_interval_distribution.svg", "Native tracker-output timing", update_intervals),
        ("dense_phase_rate_distribution.svg", "Dense-phase interval rate", [p["interval_rate_hz"] for p in phases if p.get("interval_rate_hz") is not None]),
        ("phase_window_rate_distribution.svg", "Phase-window detection rate", [w["detection_rate_hz"] for w in windows]),
        ("sensor_window_duration_distribution.svg", "Observed sensor-window duration", sensor_durations),
        ("window_event_count_distribution.svg", "Window event count", window_events),
        ("rejection_reason_counts.svg", "Rejection-reason counts", [r["count"] for r in rejections]),
    )
    for filename, title, values in aggregate_specs:
        path = plot_dir / filename
        _simple_sidecar_svg(path, title, "Sidecar-derived offline analysis; missing rows are not confirmed absence.", values)
        paths.append(str(path))
    return paths


def run_sidecar_analysis(args: argparse.Namespace, files: Sequence[Path], base_dir: Path) -> int:
    """Analyze tracker-output sidecars matched to episode files."""
    output_dir = (args.output_dir or (base_dir / "event_detection_analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    statistics: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    plot_episodes = []
    sidecar_paths = []
    configs = []
    for episode_path in files:
        try:
            attrs = _episode_attributes(episode_path)
            sidecar_path = _resolve_sidecar_path(episode_path, attrs, args.sidecar)
            if sidecar_path is None or not sidecar_path.is_file():
                reason = "no matching sidecar path exists"
                if args.analysis_source == "auto":
                    print(f"warning: {episode_path}: {reason}; falling back to episode mode", file=sys.stderr)
                    return -1
                raise ValueError(reason)
            group_meta = _sidecar_group_metadata(sidecar_path)
            group_name = match_episode_to_sidecar_group(episode_path, attrs, group_meta)
            if group_name is None and len(group_meta) == 1:
                group_name = group_meta[0][0]
            if group_name is None:
                raise ValueError("episode could not be matched to a sidecar group")
            arrays, side_attrs, reordered = load_sidecar_group(sidecar_path, group_name)
            missing_optional = [name for name in SIDECAR_OPTIONAL_DATASETS if name not in arrays]
            if missing_optional:
                print(f"warning: {sidecar_path}::{group_name}: optional diagnostics unavailable: "
                      + ", ".join(missing_optional), file=sys.stderr)
            if reordered:
                print(f"warning: {sidecar_path}::{group_name}: non-monotonic timestamps; rows stable-sorted by timestamp, packet, original index", file=sys.stderr)
            episode_id = str(episode_path)
            metric, episode_intervals, episode_runs = summarize_sidecar_episode(
                arrays, episode_id, sidecar_path, group_name)
            raw_times = np.asarray(arrays["available_ros_t_ns"], dtype=np.float64) * 1e-9
            metric["_all_update_intervals_ms"] = (np.diff(raw_times) * 1000.0).tolist()
            episode_phases, _ = segment_dense_phases(
                arrays, episode_id, args.phase_gap_ms, args.min_phase_detections)
            episode_windows = calculate_phase_window_rates(episode_phases, arrays, args.rate_window_ms)
            for phase in episode_phases:
                phase_windows = [w for w in episode_windows if w["phase_id"] == phase["phase_id"]]
                rates = [float(w["detection_rate_hz"]) for w in phase_windows]
                phase.update({
                    "minimum_window_rate_hz": min(rates) if rates else None,
                    "median_window_rate_hz": _percentile(rates, 50), "p90_window_rate_hz": _percentile(rates, 90),
                    "maximum_window_rate_hz": max(rates) if rates else None,
                    "maximum_window_detection_count": max((w["valid_detection_count"] for w in phase_windows), default=None),
                })
            episode_stats = detection_window_statistics(arrays, episode_id)
            for stat in episode_stats:
                field = stat["field"]
                if field == "sensor_window_duration_ms":
                    values = (np.asarray(arrays["sensor_window_end_us"], dtype=float) - np.asarray(arrays["sensor_window_start_us"], dtype=float)) / 1000.0
                else:
                    values = np.asarray(arrays.get(field, []))
                try:
                    stat["_finite_values"] = np.asarray(values, dtype=float)[np.isfinite(np.asarray(values, dtype=float))].tolist()
                except (TypeError, ValueError):
                    stat["_finite_values"] = []
            metrics.append(metric)
            intervals.extend(episode_intervals)
            runs.extend(episode_runs)
            phases.extend(episode_phases)
            windows.extend(episode_windows)
            statistics.extend(episode_stats)
            rejections.extend(_sidecar_rejection_rows(arrays, episode_id))
            plot_episodes.append((episode_id, arrays, episode_phases))
            sidecar_paths.append(str(sidecar_path))
            config = _parse_tracker_config({**attrs, **side_attrs})
            if config is not None:
                configs.append({"episode_id": episode_id, "configuration": config})
            print(f"{episode_id}: native sidecar tracker-output/update cadence, rows={metric['total_sidecar_rows']}, unique valid outputs={metric['valid_detection_count']}")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            skipped.append({"file": str(episode_path), "episode": "", "reason": str(exc)})
            print(f"warning: unmatched/skipped episode {episode_path}: {exc}", file=sys.stderr)
    aggregate = summarize_sidecar_aggregate(metrics, intervals, phases, windows, statistics,
                                            sidecar_paths, len(files), len(skipped))
    aggregate_statistics = []
    for field in SIDECAR_NUMERIC_FIELDS:
        pooled = [value for row in statistics if row["field"] == field
                  for value in row.get("_finite_values", [])]
        if pooled:
            aggregate_statistics.append({"episode_id": "__aggregate_pooled_rows__", "field": field,
                                         **_numeric_stats(pooled)})
    statistics.extend(aggregate_statistics)
    if rejections:
        pooled_rejections: dict[str, int] = {}
        for row in rejections:
            pooled_rejections[str(row["rejection_reason"])] = (
                pooled_rejections.get(str(row["rejection_reason"]), 0) + int(row["count"])
            )
        total_rejections = sum(pooled_rejections.values())
        rejections.extend({"episode_id": "__aggregate_pooled_rows__", "rejection_reason": reason,
                           "count": count, "fraction_of_sidecar_rows": count / total_rejections}
                          for reason, count in sorted(pooled_rejections.items()))
    outputs = (
        ("sidecar_episode_metrics.csv", metrics, ("episode_id",)),
        ("sidecar_detection_intervals.csv", intervals, ("episode_id", "detection_index")),
        ("sidecar_rle_runs.csv", runs, ("episode_id", "signal_source", "run_index")),
        ("sidecar_phases.csv", phases, ("episode_id", "phase_id", "is_dense")),
        ("sidecar_phase_rate_windows.csv", windows, ("episode_id", "phase_id", "window_index")),
        ("sidecar_window_statistics.csv", statistics, ("episode_id", "field", "count")),
        ("sidecar_rejection_statistics.csv", rejections, ("episode_id", "rejection_reason", "count")),
    )
    for filename, rows_out, fields in outputs:
        _write_rows_even_if_empty(output_dir / filename, rows_out, fields)
    plot_paths = _write_sidecar_plots(
        output_dir / "plots", plot_episodes, metrics, intervals, phases, windows, statistics,
        rejections,
    ) if args.plot else []
    report = {
        **aggregate,
        "analysis_configuration": {
            "phase_gap_ms": args.phase_gap_ms,
            "min_phase_detections": args.min_phase_detections,
            "rate_window_ms": args.rate_window_ms,
            "cadence_label": "native sidecar tracker-output/update cadence",
            "rle_note": "row-sequence RLE, not a continuous millisecond signal",
        },
        "tracker_configurations": configs, "skipped_episodes": skipped, "plot_files": plot_paths,
        "measurement_scope": {
            "measured": ["tracker-output timing", "observed sensor-window spans", "recorded valid tracker-output gaps"],
            "not_available": [
                "every processed 1 ms bin", "reliable empty 1 ms-bin counts",
                "map-building latency", "morphology latency", "blob-extraction latency",
                "velocity-fit latency", "total CPU latency", "true sensor-event-to-detection latency",
            ],
            "absence_caveat": "Missing sidecar rows and gaps between recorded valid outputs are not confirmed detector absence.",
            "latency_note": "True sensor-event-to-detection latency requires additional runtime timestamps.",
        },
    }
    (output_dir / "sidecar_aggregate_summary.json").write_text(json.dumps(_json_value(report), indent=2) + "\n", encoding="utf-8")
    if not metrics:
        print("error: no sidecar episode could be analyzed", file=sys.stderr)
        return 1
    print(f"Reports written to {output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.input_file is not None:
        files = [args.input_file.resolve()]
        base_dir = files[0].parent
    else:
        top_dir = args.top_dir.resolve()
        if not top_dir.is_dir():
            print(f"error: top directory does not exist: {top_dir}", file=sys.stderr)
            return 2
        pattern = "**/*.hdf5" if args.recursive else "*.hdf5"
        files = sorted(path.resolve() for path in top_dir.glob(pattern) if path.is_file())
        if args.max_files is not None:
            files = files[: args.max_files]
        base_dir = top_dir
    discovered_count = len(files)
    if args.analysis_source == "sidecar":
        return run_sidecar_analysis(args, files, base_dir)
    if args.analysis_source == "auto":
        sidecar_files = []
        episode_fallback_files = []
        for path in files:
            try:
                attrs = _episode_attributes(path)
                candidate = _resolve_sidecar_path(path, attrs, args.sidecar)
                groups = _sidecar_group_metadata(candidate) if candidate and candidate.is_file() else []
                matched = match_episode_to_sidecar_group(path, attrs, groups)
                if matched is not None or len(groups) == 1:
                    sidecar_files.append(path)
                else:
                    episode_fallback_files.append(path)
                    print(f"warning: {path}: no matching sidecar; falling back to episode mode",
                          file=sys.stderr)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                episode_fallback_files.append(path)
                print(f"warning: {path}: sidecar lookup failed ({exc}); falling back to episode mode",
                      file=sys.stderr)
        if sidecar_files:
            sidecar_result = run_sidecar_analysis(args, sidecar_files, base_dir)
            if sidecar_result != 0 and not episode_fallback_files:
                return sidecar_result
        if not episode_fallback_files:
            return 0
        files = episode_fallback_files
    output_dir = (args.output_dir or (base_dir / "event_detection_analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Avoid consuming outputs if the output directory happens to sit below top-dir.
    files = [path for path in files if output_dir not in path.parents]

    print(
        f"Analysis configuration: mode={args.presence_mode}, max_age={args.max_age_ms:g}ms, "
        f"timestamp_epsilon={args.timestamp_epsilon:g}s"
    )
    episode_metrics: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    max_age_sec = args.max_age_ms / 1000.0

    for file_path in files:
        if not file_path.is_file():
            skipped.append({"file": str(file_path), "episode": "", "reason": "file does not exist"})
            print(f"warning: skipping {file_path}: file does not exist", file=sys.stderr)
            continue
        try:
            with h5py.File(file_path, "r") as h5:
                roots = _episode_roots(h5)
                if not roots:
                    reason = "no root or grouped observations layout found"
                    skipped.append({"file": str(file_path), "episode": "", "reason": reason})
                    print(f"warning: skipping {file_path}: {reason}", file=sys.stderr)
                    continue
                for root in roots:
                    episode_id = str(file_path) if root == "/" else f"{file_path}::{root}"
                    try:
                        arrays = _load_episode(h5, root)
                        signal = construct_presence_signal(
                            arrays["event_valid"], arrays["event_source_age_sec"],
                            arrays["event_source_timestamps"], args.presence_mode,
                            max_age_sec, args.timestamp_epsilon,
                        )
                        runs = run_length_encode(signal, arrays["timestamps"], episode_id)
                        metric = summarize_episode(episode_id, arrays["timestamps"], signal, runs)
                        has_update = np.asarray(arrays["event_has_update"]).reshape(-1) != 0
                        derived_update = detect_new_updates(
                            arrays["event_valid"], arrays["event_source_timestamps"], args.timestamp_epsilon
                        )
                        metric["event_has_update_true_count"] = int(np.count_nonzero(has_update))
                        metric["derived_new_update_true_count"] = int(np.count_nonzero(derived_update))
                        # Sample zero has no preceding in-episode source timestamp, so it is
                        # intentionally excluded from this apples-to-apples consistency check.
                        metric["event_has_update_mismatch_count"] = int(
                            np.count_nonzero(has_update[1:] != derived_update[1:])
                        )
                        metric["event_has_update_at_first_sample"] = bool(has_update[0])
                        episode_metrics.append(metric)
                        all_runs.extend(runs)
                        _print_episode(metric)
                    except (ValueError, TypeError, KeyError) as exc:
                        skipped.append({"file": str(file_path), "episode": root, "reason": str(exc)})
                        print(f"warning: skipping {episode_id}: {exc}", file=sys.stderr)
        except (OSError, ValueError) as exc:
            skipped.append({"file": str(file_path), "episode": "", "reason": str(exc)})
            print(f"warning: skipping {file_path}: {exc}", file=sys.stderr)

    if not episode_metrics:
        summary = {
            "analysis_configuration": {
                "presence_mode": args.presence_mode,
                "max_age_ms": args.max_age_ms,
                "timestamp_epsilon_sec": args.timestamp_epsilon,
                "plots_enabled": args.plot,
                "policy_grid_note": "approximately 30 Hz policy-grid-visible presence",
            },
            "aggregate_metrics": None,
            "skipped_episodes": skipped,
            "analyzed_episode_identifiers": [],
        }
        (output_dir / "aggregate_summary.json").write_text(
            json.dumps(_json_value(summary), indent=2) + "\n", encoding="utf-8"
        )
        print("error: no valid episode could be analyzed", file=sys.stderr)
        return 1

    aggregate = summarize_aggregate(episode_metrics, all_runs, discovered_count, len(skipped))
    _write_csv(output_dir / "per_episode_metrics.csv", episode_metrics)
    _write_csv(output_dir / "rle_runs.csv", all_runs)
    plot_paths = (
        write_plots(
            output_dir / "plots", episode_metrics, all_runs, aggregate,
            args.presence_mode, args.max_age_ms,
        )
        if args.plot else []
    )
    summary = {
        "analysis_configuration": {
            "presence_mode": args.presence_mode,
            "max_age_ms": args.max_age_ms,
            "max_age_sec": max_age_sec,
            "timestamp_epsilon_sec": args.timestamp_epsilon,
            "plots_enabled": args.plot,
            "gap_thresholds_ms": list(GAP_THRESHOLDS_MS),
            "policy_grid_note": (
                "approximately 30 Hz policy-grid-visible presence; native tracker gaps faster "
                "than the grid require tracker-sidecar analysis"
            ),
        },
        "aggregate_metrics": aggregate,
        "skipped_episodes": skipped,
        "analyzed_episode_identifiers": [m["episode_id"] for m in episode_metrics],
        "plot_files": plot_paths,
    }
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(_json_value(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Aggregate: "
        f"files={aggregate['number_of_discovered_files']}, "
        f"episodes={aggregate['number_of_analyzed_episodes']}, "
        f"skipped={aggregate['number_of_skipped_episodes']}, "
        f"duration={aggregate['total_observation_duration_sec']:.3f}s, "
        f"presence={100.0 * aggregate['time_weighted_presence_fraction']:.2f}%, "
        f"gaps={aggregate['pooled_total_gap_count']}, "
        f"pooled median/p90/p95/max gap="
        f"{_fmt(aggregate['pooled_all_gaps']['median_sec'])}/"
        f"{_fmt(aggregate['pooled_all_gaps']['p90_sec'])}/"
        f"{_fmt(aggregate['pooled_all_gaps']['p95_sec'])}/"
        f"{_fmt(aggregate['pooled_all_gaps']['max_sec'])}s"
    )
    if plot_paths:
        print(f"Plots written to {output_dir / 'plots'}")
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
