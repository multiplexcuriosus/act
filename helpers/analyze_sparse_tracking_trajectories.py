#!/usr/bin/env python3
"""Analyze sparse RGB/event trajectories and infer spatial turning points.

A turning candidate is an interior unique source update whose incoming and
outgoing displacement vectors, measured over ``turn_half_window`` observations,
both exceed ``min_displacement_px`` and form at least
``min_reversal_angle_deg``.  Gaps larger than ``max_gap_ms`` break a trajectory
and cannot be crossed by a candidate.  Candidate score is

    (1 - cos(reversal_angle)) * min(incoming_length, outgoing_length)

The highest score is selected.  Candidates within 90% of the highest score are
reported as similarly strong and make the result ambiguous.  Coordinates stay
in their native systems: RGB 1280x720 pixels and event-camera crop 320x320
pixels.  They are never subtracted or interpreted as spatial agreement because
no calibration or homography is available.

Inputs are opened read-only.  Held policy-grid values are removed by
deduplicating consecutive equal source timestamps while retaining the first
corresponding policy-grid frame index.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


COORDINATE_SYSTEMS = {
    "rgb": {"label": "RGB native pixels (1280x720)", "width": 1280, "height": 720},
    "event": {"label": "event-camera crop native pixels (320x320)", "width": 320, "height": 320},
}
AMBIGUOUS_SCORE_FRACTION = 0.90
MODALITY_FIELDS = {
    "rgb": ("rgb_2d_px", "rgb_valid", "rgb_source_timestamps", "rgb_source_age_sec"),
    "event": ("event_2d_px", "event_valid", "event_source_timestamps", "event_source_age_sec"),
}


@dataclass
class Trajectory:
    modality: str
    policy_frames: np.ndarray
    policy_timestamps: np.ndarray
    source_timestamps: np.ndarray
    source_ages_sec: np.ndarray
    points: np.ndarray
    smoothed_points: np.ndarray
    segment_ids: np.ndarray


def _number(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _number(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_number(item) for item in value]
    return value


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None, "min": None}
    return {
        "count": int(array.size), "mean": float(np.mean(array)),
        "median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)), "max": float(np.max(array)),
        "min": float(np.min(array)),
    }


def _episode_roots(h5: h5py.File) -> list[str]:
    roots: list[str] = []
    if "observations" in h5 and isinstance(h5["observations"], h5py.Group):
        roots.append("/")
    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Group) and name.endswith("/observations"):
            roots.append("/" + name[:-len("/observations")])
    h5.visititems(visitor)
    return sorted(set(roots))


def _path(root: str, relative: str) -> str:
    return "/" + relative if root == "/" else root.rstrip("/") + "/" + relative


def discover_files(patterns: Sequence[str], recursive: bool = False) -> list[Path]:
    """Expand files, directories, and glob patterns to HDF5 paths."""
    found: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=recursive)
        candidates = [Path(item) for item in matches] if matches else [Path(pattern)]
        for candidate in candidates:
            if candidate.is_dir():
                iterator = candidate.rglob("*") if recursive else candidate.glob("*")
                found.update(item.resolve() for item in iterator if item.is_file() and item.suffix.lower() in (".h5", ".hdf5"))
            elif candidate.suffix.lower() in (".h5", ".hdf5"):
                found.add(candidate.resolve())
    return sorted(found)


def robust_median_smooth(points: np.ndarray, window: int) -> np.ndarray:
    """Centered per-coordinate median smoothing with clipped edge windows."""
    points = np.asarray(points, dtype=np.float64)
    if window <= 1 or len(points) == 0:
        return points.copy()
    half = window // 2
    return np.asarray([np.median(points[max(0, i-half):min(len(points), i+half+1)], axis=0) for i in range(len(points))])


def build_trajectory(
    modality: str, policy_timestamps: Sequence[float], points: np.ndarray,
    valid: Sequence[Any], source_timestamps: Sequence[float], source_ages_sec: Sequence[float],
    max_age_sec: float, smoothing_window: int, max_gap_sec: float,
) -> tuple[Trajectory, dict[str, Any]]:
    """Filter fresh finite rows, stable-sort, and remove held source values."""
    times = np.asarray(policy_timestamps, dtype=np.float64).reshape(-1)
    xy = np.asarray(points, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"{modality}_2d_px must have shape (T, 2), got {xy.shape}")
    valid_array = np.asarray(valid).reshape(-1) != 0
    source = np.asarray(source_timestamps, dtype=np.float64).reshape(-1)
    ages = np.asarray(source_ages_sec, dtype=np.float64).reshape(-1)
    lengths = {len(times), len(xy), len(valid_array), len(source), len(ages)}
    if len(lengths) != 1:
        raise ValueError(f"{modality} dataset lengths differ")
    finite_valid = valid_array & np.isfinite(xy).all(axis=1) & np.isfinite(source) & np.isfinite(ages)
    fresh = finite_valid & (ages >= -1e-9) & (ages <= max_age_sec)
    indices = np.flatnonzero(fresh)
    order = np.argsort(source[indices], kind="stable")
    indices = indices[order]
    if indices.size:
        keep = np.r_[True, np.diff(source[indices]) != 0]
        indices = indices[keep]
    selected_source = source[indices]
    gaps = np.diff(selected_source)
    segment_ids = np.r_[0, np.cumsum(gaps > max_gap_sec)].astype(int) if indices.size else np.empty(0, dtype=int)
    trajectory = Trajectory(
        modality=modality, policy_frames=indices.astype(int), policy_timestamps=times[indices],
        source_timestamps=selected_source, source_ages_sec=ages[indices], points=xy[indices],
        smoothed_points=robust_median_smooth(xy[indices], smoothing_window), segment_ids=segment_ids,
    )
    diagnostics = {
        "policy_sample_count": int(len(times)), "valid_count": int(np.count_nonzero(finite_valid)),
        "fresh_detection_count": int(np.count_nonzero(fresh)), "unique_source_update_count": int(len(indices)),
        "fresh_coverage": float(np.count_nonzero(fresh) / len(times)) if len(times) else None,
        # Retained internally so pre-turn policy-grid coverage includes held-but-fresh
        # rows even though those rows are excluded from trajectory geometry.
        "_fresh_policy_frames": np.flatnonzero(fresh).astype(int).tolist(),
    }
    return trajectory, diagnostics


def infer_turning_point(
    trajectory: Trajectory, half_window: int = 2, min_displacement_px: float = 5.0,
    min_reversal_angle_deg: float = 120.0,
) -> dict[str, Any]:
    """Return selected and all passing reversal candidates for one trajectory."""
    pts = trajectory.smoothed_points
    candidates: list[dict[str, Any]] = []
    for i in range(half_window, len(pts) - half_window):
        left, right = i - half_window, i + half_window
        if trajectory.segment_ids[left] != trajectory.segment_ids[right]:
            continue
        incoming = pts[i] - pts[left]
        outgoing = pts[right] - pts[i]
        incoming_mag, outgoing_mag = float(np.linalg.norm(incoming)), float(np.linalg.norm(outgoing))
        if incoming_mag < min_displacement_px or outgoing_mag < min_displacement_px:
            continue
        cosine = float(np.clip(np.dot(incoming, outgoing) / (incoming_mag * outgoing_mag), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
        if angle < min_reversal_angle_deg:
            continue
        score = float((1.0 - cosine) * min(incoming_mag, outgoing_mag))
        candidates.append({
            "trajectory_index": i, "policy_frame_index": int(trajectory.policy_frames[i]),
            "policy_timestamp": float(trajectory.policy_timestamps[i]),
            "source_timestamp": float(trajectory.source_timestamps[i]),
            "source_age_ms": float(trajectory.source_ages_sec[i] * 1000.0),
            "u": float(trajectory.points[i, 0]), "v": float(trajectory.points[i, 1]),
            "incoming_displacement_px": incoming_mag, "outgoing_displacement_px": outgoing_mag,
            "reversal_angle_deg": angle, "score": score,
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not candidates:
        return {"status": "no_turn_detected", "selected": None, "candidates": [], "ambiguous_candidates": []}
    selected = candidates[0]
    similar = [item for item in candidates if item["score"] >= selected["score"] * AMBIGUOUS_SCORE_FRACTION]
    status = "ambiguous" if len(similar) > 1 else "detected"
    return {"status": status, "selected": selected, "candidates": candidates, "ambiguous_candidates": similar}


def _gap_stats_ms(trajectory: Trajectory) -> dict[str, Any]:
    return _stats(np.diff(trajectory.source_timestamps) * 1000.0)


def _path_length(points: np.ndarray, segments: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.sum(distances[segments[1:] == segments[:-1]]))


def make_episode_metric(
    episode_path: str, episode_index: str, modality: str, trajectory: Trajectory,
    diagnostics: Mapping[str, Any], turn: Mapping[str, Any], min_detections: int,
) -> dict[str, Any]:
    usable = len(trajectory.points) >= min_detections
    selected = turn["selected"] if usable else None
    turn_status = turn["status"] if usable else "not_evaluated"
    stop = int(selected["trajectory_index"]) if selected else len(trajectory.points) - 1
    pre_count = stop + 1 if stop >= 0 else 0
    pre_duration = (float(trajectory.policy_timestamps[stop] - trajectory.policy_timestamps[0]) if pre_count > 1 else 0.0 if pre_count == 1 else None)
    pre_last_frame = int(trajectory.policy_frames[stop]) if pre_count else -1
    pre_denominator = pre_last_frame + 1 if pre_count else diagnostics["policy_sample_count"]
    fresh_frames = np.asarray(diagnostics.get("_fresh_policy_frames", []), dtype=int)
    pre_fresh_count = int(np.count_nonzero(fresh_frames <= pre_last_frame)) if pre_count else 0
    pre_coverage = float(pre_fresh_count / pre_denominator) if pre_denominator else None
    pre_gaps = np.diff(trajectory.source_timestamps[:pre_count]) * 1000.0 if pre_count > 1 else np.empty(0)
    source_stats = _stats(trajectory.source_ages_sec * 1000.0)
    gap_stats = _gap_stats_ms(trajectory)
    row: dict[str, Any] = {
        "episode_path": episode_path, "episode_index": episode_index, "modality": modality,
        "coordinate_system": COORDINATE_SYSTEMS[modality]["label"],
        "trajectory_status": "usable" if usable else ("insufficient_detections" if len(trajectory.points) else "no_fresh_detections"),
        **diagnostics, "first_policy_frame": int(trajectory.policy_frames[0]) if len(trajectory.points) else None,
        "last_policy_frame": int(trajectory.policy_frames[-1]) if len(trajectory.points) else None,
        "first_source_timestamp": float(trajectory.source_timestamps[0]) if len(trajectory.points) else None,
        "last_source_timestamp": float(trajectory.source_timestamps[-1]) if len(trajectory.points) else None,
        "pre_turn_count": pre_count, "pre_turn_fresh_policy_row_count": pre_fresh_count,
        "pre_turn_duration_sec": pre_duration,
        "pre_turn_valid_fresh_coverage": pre_coverage,
        "pre_turn_path_length_px": _path_length(trajectory.points[:pre_count], trajectory.segment_ids[:pre_count]),
        "source_age_median_ms": source_stats["median"], "source_age_p90_ms": source_stats["p90"],
        "source_age_p95_ms": source_stats["p95"], "source_age_max_ms": source_stats["max"],
        "gap_median_ms": gap_stats["median"], "gap_p90_ms": gap_stats["p90"],
        "gap_p95_ms": gap_stats["p95"], "gap_max_ms": gap_stats["max"],
        "longest_pre_turn_gap_ms": float(np.max(pre_gaps)) if pre_gaps.size else None,
        "turning_point_status": turn_status, "turning_point_u": selected["u"] if selected else None,
        "turning_point_v": selected["v"] if selected else None,
        "turning_point_policy_frame_index": selected["policy_frame_index"] if selected else None,
        "turning_point_timestamp": selected["policy_timestamp"] if selected else None,
        "turning_point_source_timestamp": selected["source_timestamp"] if selected else None,
        "turning_point_source_age_ms": selected["source_age_ms"] if selected else None,
        "incoming_displacement_px": selected["incoming_displacement_px"] if selected else None,
        "outgoing_displacement_px": selected["outgoing_displacement_px"] if selected else None,
        "reversal_angle_deg": selected["reversal_angle_deg"] if selected else None,
        "candidate_count": len(turn["candidates"]) if usable else 0,
        "similarly_strong_candidate_count": len(turn["ambiguous_candidates"]) if usable else 0,
    }
    return row


def compare_modalities(rgb: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    rgb_turn = rgb["turning_point_status"] in ("detected", "ambiguous")
    event_turn = event["turning_point_status"] in ("detected", "ambiguous")
    if rgb_turn and event_turn:
        state = "both"
    elif rgb_turn:
        state = "rgb_only"
    elif event_turn:
        state = "event_only"
    else:
        state = "neither"
    frame_delta = event["turning_point_policy_frame_index"] - rgb["turning_point_policy_frame_index"] if rgb_turn and event_turn else None
    time_delta = 1000.0 * (event["turning_point_timestamp"] - rgb["turning_point_timestamp"]) if rgb_turn and event_turn else None
    lead_lag = None if time_delta is None else ("event_leads" if time_delta < 0 else "event_lags" if time_delta > 0 else "simultaneous")
    return {
        "episode_path": rgb["episode_path"], "episode_index": rgb["episode_index"], "turn_detection_pair": state,
        "rgb_coordinate_system": rgb["coordinate_system"], "event_coordinate_system": event["coordinate_system"],
        "rgb_turning_point_u": rgb["turning_point_u"], "rgb_turning_point_v": rgb["turning_point_v"],
        "event_turning_point_u": event["turning_point_u"], "event_turning_point_v": event["turning_point_v"],
        "rgb_turning_point_frame_index": rgb["turning_point_policy_frame_index"],
        "event_turning_point_frame_index": event["turning_point_policy_frame_index"],
        "event_minus_rgb_frame_index": frame_delta,
        "rgb_turning_point_timestamp": rgb["turning_point_timestamp"],
        "event_turning_point_timestamp": event["turning_point_timestamp"],
        "event_minus_rgb_timestamp_ms": time_delta, "event_lead_lag": lead_lag,
        "rgb_unique_source_update_count": rgb["unique_source_update_count"],
        "event_unique_source_update_count": event["unique_source_update_count"],
        "event_minus_rgb_detection_count": event["unique_source_update_count"] - rgb["unique_source_update_count"],
        "rgb_pre_turn_coverage": rgb["pre_turn_valid_fresh_coverage"],
        "event_pre_turn_coverage": event["pre_turn_valid_fresh_coverage"],
        "event_minus_rgb_pre_turn_coverage": (event["pre_turn_valid_fresh_coverage"] - rgb["pre_turn_valid_fresh_coverage"] if event["pre_turn_valid_fresh_coverage"] is not None and rgb["pre_turn_valid_fresh_coverage"] is not None else None),
    }


def _aggregate_modality(rows: Sequence[Mapping[str, Any]], total_episodes: int) -> dict[str, Any]:
    usable = [row for row in rows if row["trajectory_status"] == "usable"]
    turns = [row for row in usable if row["turning_point_status"] in ("detected", "ambiguous")]
    def values(field: str, source: Sequence[Mapping[str, Any]] = rows) -> list[float]:
        return [float(row[field]) for row in source if row.get(field) is not None]
    return {
        "total_episodes": total_episodes, "episodes_with_metrics": len(rows), "usable_trajectories": len(usable),
        "turn_detected": len(turns), "ambiguous_turns": sum(row["turning_point_status"] == "ambiguous" for row in usable),
        "no_turn_detected": sum(row["turning_point_status"] == "no_turn_detected" for row in usable),
        "detection_count": _stats(values("fresh_detection_count")),
        "unique_source_update_count": _stats(values("unique_source_update_count")),
        "turning_point_u": _stats(values("turning_point_u", turns)), "turning_point_v": _stats(values("turning_point_v", turns)),
        "turning_point_policy_frame_index": _stats(values("turning_point_policy_frame_index", turns)),
        "turning_point_timestamp": _stats(values("turning_point_timestamp", turns)),
        "pre_turn_detection_count": _stats(values("pre_turn_count", usable)),
        "pre_turn_duration_sec": _stats(values("pre_turn_duration_sec", usable)),
        "pre_turn_valid_fresh_coverage": _stats(values("pre_turn_valid_fresh_coverage", usable)),
        "source_age_ms": _stats([value for row in rows for value in row.get("_source_ages_ms", [])]),
        "inter_detection_gap_ms": _stats([value for row in rows for value in row.get("_gaps_ms", [])]),
        "longest_pre_turn_gap_ms": _stats(values("longest_pre_turn_gap_ms", usable)),
        "pre_turn_path_length_px": _stats(values("pre_turn_path_length_px", usable)),
        "incoming_displacement_px": _stats(values("incoming_displacement_px", turns)),
        "outgoing_displacement_px": _stats(values("outgoing_displacement_px", turns)),
        "reversal_angle_deg": _stats(values("reversal_angle_deg", turns)),
    }


def _aggregate_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["event_minus_rgb_timestamp_ms"]) for row in rows if row["event_minus_rgb_timestamp_ms"] is not None]
    return {
        "paired_episodes": len(rows), "both_turns": sum(row["turn_detection_pair"] == "both" for row in rows),
        "rgb_only_turn": sum(row["turn_detection_pair"] == "rgb_only" for row in rows),
        "event_only_turn": sum(row["turn_detection_pair"] == "event_only" for row in rows),
        "neither_turn": sum(row["turn_detection_pair"] == "neither" for row in rows),
        "event_minus_rgb_frame_index": _stats([row["event_minus_rgb_frame_index"] for row in rows if row["event_minus_rgb_frame_index"] is not None]),
        "event_minus_rgb_timestamp_ms": _stats(deltas),
        "event_lead_lag": {"event_leads": sum(v < 0 for v in deltas), "simultaneous": sum(v == 0 for v in deltas), "event_lags": sum(v > 0 for v in deltas)},
        "event_minus_rgb_detection_count": _stats([row["event_minus_rgb_detection_count"] for row in rows]),
        "event_minus_rgb_pre_turn_coverage": _stats([row["event_minus_rgb_pre_turn_coverage"] for row in rows if row["event_minus_rgb_pre_turn_coverage"] is not None]),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fallback_fields: Sequence[str]) -> None:
    fields = [key for key in rows[0] if not key.startswith("_")] if rows else list(fallback_fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _number(row.get(key)) for key in fields})


def _safe_id(index: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(index)).strip("_") or "unknown"


def _plot_segments(ax: Any, trajectory: Trajectory) -> None:
    for segment in np.unique(trajectory.segment_ids):
        mask = trajectory.segment_ids == segment
        if np.count_nonzero(mask) > 1:
            ax.plot(trajectory.points[mask, 0], trajectory.points[mask, 1], color="0.55", lw=1, alpha=.7)


def plot_episode(output: Path, episode_id: str, modality: str, trajectory: Trajectory,
                 metric: Mapping[str, Any], turn: Mapping[str, Any]) -> Path:
    import matplotlib.pyplot as plt
    spec = COORDINATE_SYSTEMS[modality]
    fig, ax = plt.subplots(figsize=(8, 8 * spec["height"] / spec["width"] + 1.5))
    _plot_segments(ax, trajectory)
    if len(trajectory.points):
        scatter = ax.scatter(trajectory.points[:, 0], trajectory.points[:, 1], c=trajectory.source_timestamps, s=24, cmap="viridis", zorder=3)
        fig.colorbar(scatter, ax=ax, label="native source timestamp (s)")
    selected_index = turn["selected"]["trajectory_index"] if turn["selected"] else None
    ambiguous_indices = {item["trajectory_index"] for item in turn["ambiguous_candidates"]}
    for candidate in turn["candidates"]:
        if candidate["trajectory_index"] == selected_index:
            marker, color, size = "*", "red", 180
        elif candidate["trajectory_index"] in ambiguous_indices:
            marker, color, size = "D", "purple", 65
        else:
            marker, color, size = "x", "orange", 65
        ax.scatter(candidate["u"], candidate["v"], marker=marker, color=color, s=size, zorder=5)
    ax.set(xlim=(0, spec["width"]), ylim=(spec["height"], 0), xlabel=f"u — {spec['label']}", ylabel=f"v — {spec['label']}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{episode_id} — {modality.upper()} trajectory\nunique={metric['unique_source_update_count']}, fresh rows={metric['fresh_detection_count']}, source age median/p95={metric['source_age_median_ms']!s}/{metric['source_age_p95_ms']!s} ms")
    ax.grid(alpha=.2)
    path = output / f"episode_{_safe_id(episode_id)}_{modality}_trajectory.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_paired_episode(output: Path, episode_id: str, data: Mapping[str, tuple[Trajectory, Mapping[str, Any], Mapping[str, Any]]]) -> Path:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, modality in zip(axes, ("rgb", "event")):
        trajectory, metric, turn = data[modality]
        spec = COORDINATE_SYSTEMS[modality]; _plot_segments(ax, trajectory)
        if len(trajectory.points):
            ax.scatter(trajectory.points[:, 0], trajectory.points[:, 1], c=trajectory.source_timestamps, s=20, cmap="viridis")
        if modality == "rgb" and turn["selected"]:
            ax.scatter(turn["selected"]["u"], turn["selected"]["v"], marker="*", color="red", s=170)
        elif modality == "event" and data["rgb"][2]["selected"] and len(trajectory.points):
            # Mark where the event trajectory was when the RGB turn occurred.
            # Event updates need not land on the same policy-grid frame, so use
            # the event sample nearest in policy time.
            rgb_turn_time = data["rgb"][2]["selected"]["policy_timestamp"]
            event_index = int(np.argmin(np.abs(trajectory.policy_timestamps - rgb_turn_time)))
            ax.scatter(
                trajectory.points[event_index, 0], trajectory.points[event_index, 1],
                marker="*", color="blue", s=170, zorder=5,
            )
        ax.set(xlim=(0, spec["width"]), ylim=(spec["height"], 0), xlabel=f"u — {spec['label']}", ylabel="v")
        ax.set_aspect("equal", adjustable="box"); ax.set_title(f"{modality.upper()}: {metric['turning_point_status']}")
    fig.suptitle(f"{episode_id}: separate native coordinate systems; no spatial comparison")
    path = output / f"episode_{_safe_id(episode_id)}_rgb_event_trajectory.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def write_aggregate_plots(output: Path, episode_data: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib.pyplot as plt
    paths: list[str] = []
    by_modality = {m: [entry for entry in episode_data if m in entry] for m in ("rgb", "event")}
    for modality in ("rgb", "event"):
        fig, ax = plt.subplots(figsize=(8, 6)); spec = COORDINATE_SYSTEMS[modality]
        for entry in by_modality[modality]:
            trajectory = entry[modality][0]
            for segment in np.unique(trajectory.segment_ids):
                pts = trajectory.points[trajectory.segment_ids == segment]
                if len(pts) > 1: ax.plot(pts[:, 0], pts[:, 1], alpha=.18)
        # Native image coordinates: the origin is in the upper-left and v grows downward.
        ax.set(xlim=(0, spec["width"]), ylim=(spec["height"], 0), xlabel=f"u — {spec['label']}", ylabel="v")
        ax.set_aspect("equal", adjustable="box"); ax.set_title(f"All {modality.upper()} trajectories (large gaps disconnected)")
        if modality == "event":
            # Event detections use image coordinates: (0, 0) is the upper-left
            # corner and v increases downward.
            ax.scatter([0], [0], marker="o", s=30, color="black", zorder=5, clip_on=False)
            ax.annotate(
                "(0, 0)", xy=(0, 0), xytext=(8, -8),
                textcoords="offset points", ha="left", va="bottom",
                annotation_clip=False,
            )
        target = output / f"all_{modality}_trajectories.png"; fig.tight_layout(); fig.savefig(target, dpi=150); plt.close(fig); paths.append(str(target))
    fig, ax = plt.subplots(figsize=(7, 7))
    for entry in by_modality["event"]:
        trajectory, _, turn = entry["event"]
        if turn["selected"]:
            origin = trajectory.points[turn["selected"]["trajectory_index"]]
            pts = trajectory.points - origin
            for segment in np.unique(trajectory.segment_ids):
                part = pts[trajectory.segment_ids == segment]
                if len(part) > 1: ax.plot(part[:, 0], part[:, 1], alpha=.25)
    ax.axhline(0, color="0.5", lw=.7); ax.axvline(0, color="0.5", lw=.7); ax.set_aspect("equal", adjustable="box")
    ax.set(title="Turn-aligned event trajectories — relative diagnostic", xlabel="relative event u (px)", ylabel="relative event v (px)")
    target = output / "all_event_trajectories_turn_aligned.png"; fig.tight_layout(); fig.savefig(target, dpi=150); plt.close(fig); paths.append(str(target))
    both = [row for row in comparisons if row["event_minus_rgb_timestamp_ms"] is not None]
    fig, ax = plt.subplots(); ax.scatter([r["rgb_turning_point_frame_index"] for r in both], [r["event_turning_point_frame_index"] for r in both])
    if both:
        vals = [r["rgb_turning_point_frame_index"] for r in both] + [r["event_turning_point_frame_index"] for r in both]; ax.plot([min(vals), max(vals)], [min(vals), max(vals)], "k--", alpha=.5)
    ax.set(xlabel="RGB turn policy frame", ylabel="event turn policy frame", title="Turning-point frame comparison")
    target = output / "turning_point_frame_comparison.png"; fig.tight_layout(); fig.savefig(target, dpi=150); plt.close(fig); paths.append(str(target))
    fig, ax = plt.subplots(); delta = [r["event_minus_rgb_timestamp_ms"] for r in both]; ax.hist(delta, bins=min(20, max(1, len(delta)))); ax.axvline(0, color="k", ls="--")
    ax.set(xlabel="event minus RGB turn timestamp (ms)", ylabel="episodes", title="Event lead (<0) / lag (>0)")
    target = output / "turning_point_time_difference.png"; fig.tight_layout(); fig.savefig(target, dpi=150); plt.close(fig); paths.append(str(target))
    for filename, field, xlabel in (("pre_turn_detection_coverage.png", "pre_turn_valid_fresh_coverage", "pre-turn valid/fresh coverage"), ("source_age_distribution.png", "_source_ages_ms", "source age (ms)"), ("inter_detection_gap_distribution.png", "_gaps_ms", "inter-detection gap (ms)")):
        fig, ax = plt.subplots()
        for modality in ("rgb", "event"):
            rows = [r for r in metrics if r["modality"] == modality]
            vals = ([r[field] for r in rows if r.get(field) is not None] if not field.startswith("_") else [x for r in rows for x in r.get(field, [])])
            if vals: ax.hist(vals, bins=min(30, max(5, int(np.sqrt(len(vals))))), alpha=.5, label=modality.upper())
        ax.set(xlabel=xlabel, ylabel="count", title=xlabel.capitalize()); ax.legend()
        target = output / filename; fig.tight_layout(); fig.savefig(target, dpi=150); plt.close(fig); paths.append(str(target))
    return paths


def _print_stats(label: str, value: Any) -> None:
    if isinstance(value, dict) and {"count", "median", "p90", "p95", "max"}.issubset(value):
        print(f"  {label}: count={value['count']}, median={value['median']}, p90={value['p90']}, p95={value['p95']}, max={value['max']}")
    else:
        print(f"  {label}: {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="HDF5 files, directories, or glob patterns")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-observation-age-ms", type=float, default=100.0)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--turn-half-window", type=int, default=2)
    parser.add_argument("--min-displacement-px", type=float, default=5.0, help="Minimum displacement in each half-window (default: 5 px)")
    parser.add_argument("--min-reversal-angle-deg", type=float, default=120.0)
    parser.add_argument("--min-detections", type=int, default=4)
    parser.add_argument("--max-gap-ms", type=float, default=150.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_observation_age_ms < 0 or args.min_displacement_px < 0 or args.max_gap_ms <= 0:
        raise ValueError("age/displacement must be nonnegative and max-gap-ms must be positive")
    if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
        raise ValueError("smoothing-window must be a positive odd integer")
    if args.turn_half_window < 1 or args.min_detections < 1:
        raise ValueError("turn-half-window and min-detections must be positive")
    if not 0 <= args.min_reversal_angle_deg <= 180:
        raise ValueError("min-reversal-angle-deg must be in [0, 180]")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try: _validate_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr); return 2
    files = discover_files(args.paths, args.recursive)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    files = [item for item in files if output not in item.parents]
    metrics: list[dict[str, Any]] = []; comparisons: list[dict[str, Any]] = []; skipped: list[dict[str, str]] = []
    episode_data: list[dict[str, Any]] = []; episode_count = 0
    for file_path in files:
        try:
            with h5py.File(file_path, "r") as h5:
                roots = _episode_roots(h5)
                if not roots:
                    skipped.append({"episode_path": str(file_path), "episode_index": "", "reason": "no observations group"}); print(f"warning: {file_path}: no observations group", file=sys.stderr); continue
                for root in roots:
                    episode_count += 1
                    attr_index = h5.attrs.get("episode_index") if root == "/" else None
                    episode_index = str(attr_index if attr_index is not None else (Path(file_path).stem if root == "/" else root.strip("/").replace("/", "_")))
                    identifier = episode_index; entry: dict[str, Any] = {}
                    timestamp_path = _path(root, "observations/timestamps")
                    if timestamp_path not in h5:
                        reason = "missing observations/timestamps"; skipped.append({"episode_path": str(file_path), "episode_index": episode_index, "reason": reason}); print(f"warning: {file_path}::{root}: {reason}", file=sys.stderr); continue
                    policy_times = np.asarray(h5[timestamp_path], dtype=np.float64).reshape(-1)
                    for modality, fields in MODALITY_FIELDS.items():
                        relative = [f"observations/sparse_tracking/{field}" for field in fields]
                        missing = [field for field, rel in zip(fields, relative) if _path(root, rel) not in h5]
                        if missing:
                            reason = f"missing {modality} fields: {', '.join(missing)}"; skipped.append({"episode_path": str(file_path), "episode_index": episode_index, "reason": reason}); print(f"warning: {file_path}::{root}: {reason}", file=sys.stderr); continue
                        try:
                            arrays = [np.asarray(h5[_path(root, rel)]) for rel in relative]
                            trajectory, diagnostics = build_trajectory(modality, policy_times, arrays[0], arrays[1], arrays[2], arrays[3], args.max_observation_age_ms / 1000.0, args.smoothing_window, args.max_gap_ms / 1000.0)
                            if len(trajectory.points) >= args.min_detections:
                                turn = infer_turning_point(trajectory, args.turn_half_window, args.min_displacement_px, args.min_reversal_angle_deg)
                            else:
                                turn = {"status": "not_evaluated", "selected": None, "candidates": [], "ambiguous_candidates": []}
                            metric = make_episode_metric(str(file_path), episode_index, modality, trajectory, diagnostics, turn, args.min_detections)
                            metric["_source_ages_ms"] = (trajectory.source_ages_sec * 1000.0).tolist()
                            metric["_gaps_ms"] = (np.diff(trajectory.source_timestamps) * 1000.0).tolist()
                            metrics.append(metric); entry[modality] = (trajectory, metric, turn)
                        except (ValueError, TypeError, KeyError) as exc:
                            reason = f"{modality}: {exc}"; skipped.append({"episode_path": str(file_path), "episode_index": episode_index, "reason": reason}); print(f"warning: {file_path}::{root}: {reason}", file=sys.stderr)
                    if "rgb" in entry and "event" in entry:
                        comparisons.append(compare_modalities(entry["rgb"][1], entry["event"][1]))
                    entry["episode_id"] = identifier; episode_data.append(entry)
        except OSError as exc:
            skipped.append({"episode_path": str(file_path), "episode_index": "", "reason": str(exc)}); print(f"warning: cannot open {file_path}: {exc}", file=sys.stderr)
    rgb_summary = _aggregate_modality([r for r in metrics if r["modality"] == "rgb"], episode_count)
    event_summary = _aggregate_modality([r for r in metrics if r["modality"] == "event"], episode_count)
    paired_summary = _aggregate_pairs(comparisons)
    summary = {
        "turning_point_definition": "Spatial reversal over equal half-windows; score=(1-cos(angle))*min(incoming,outgoing); candidates within 90% of best are ambiguous; gaps over max_gap_ms break continuity.",
        "coordinate_system_warning": "RGB and event coordinates are separate native pixel systems and are not spatially compared without calibration or a homography.",
        "configuration": vars(args) | {"output_dir": str(output), "paths": list(args.paths)},
        "files_discovered": len(files), "episodes_processed": episode_count,
        "modalities": {"rgb": rgb_summary, "event": event_summary}, "paired": paired_summary,
        "skipped": skipped,
        "candidate_details": [{"episode_path": metric["episode_path"], "episode_index": metric["episode_index"], "modality": modality, "coordinate_system": COORDINATE_SYSTEMS[modality]["label"], "candidates": _number(turn["candidates"])} for entry in episode_data for modality in ("rgb", "event") if modality in entry for metric, turn in [(entry[modality][1], entry[modality][2])]],
    }
    _write_csv(output / "episode_metrics.csv", metrics, ("episode_path", "episode_index", "modality"))
    _write_csv(output / "turning_point_comparison.csv", comparisons, ("episode_path", "episode_index", "turn_detection_pair"))
    plots: list[str] = []
    if args.plot:
        plot_dir = output / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
        for entry in episode_data:
            if "rgb" in entry and "event" in entry: plots.append(str(plot_paired_episode(plot_dir, entry["episode_id"], entry)))
        plots.extend(write_aggregate_plots(plot_dir, episode_data, metrics, comparisons))
    summary["plot_files"] = plots
    (output / "summary.json").write_text(json.dumps(_number(summary), indent=2) + "\n", encoding="utf-8")
    print(f"Episodes processed: {episode_count}")
    for modality, aggregate in (("RGB", rgb_summary), ("event", event_summary)):
        print(f"{modality}:")
        for key, value in aggregate.items(): _print_stats(key, value)
    print("Paired RGB/event:")
    for key, value in paired_summary.items(): _print_stats(key, value)
    print(f"Output directory: {output}")
    if skipped: print(f"Skipped/missing modality records: {len(skipped)} (see summary.json)")
    return 0 if metrics else 1


if __name__ == "__main__":
    raise SystemExit(main())
