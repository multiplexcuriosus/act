#!/usr/bin/env python3
"""Analyze policy-grid-visible event ball-detection presence in ACT HDF5 files.

The episode arrays are sampled on an approximately 30 Hz observation grid using
the latest tracker message at or before each grid time.  Consequently, ``valid``
can include held values.  This tool measures temporal presence as visible to the
policy grid; native event-tracker gaps faster than the observation grid require
analysis of the tracker sidecar.

Only the Python standard library, NumPy, and h5py are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
ONE_SAMPLE_PERIOD_SEC = 1.0 / 30.0


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
    if args.recursive and args.top_dir is None:
        parser.error("--recursive requires --top-dir")
    return args


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
    summary = {
        "analysis_configuration": {
            "presence_mode": args.presence_mode,
            "max_age_ms": args.max_age_ms,
            "max_age_sec": max_age_sec,
            "timestamp_epsilon_sec": args.timestamp_epsilon,
            "gap_thresholds_ms": list(GAP_THRESHOLDS_MS),
            "policy_grid_note": (
                "approximately 30 Hz policy-grid-visible presence; native tracker gaps faster "
                "than the grid require tracker-sidecar analysis"
            ),
        },
        "aggregate_metrics": aggregate,
        "skipped_episodes": skipped,
        "analyzed_episode_identifiers": [m["episode_id"] for m in episode_metrics],
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
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
