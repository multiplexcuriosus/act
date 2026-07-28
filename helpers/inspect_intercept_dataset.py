#!/usr/bin/env python3

<<<<<<< Updated upstream
"""Inspect interception HDF5 datasets and the action chunks seen by ACT.

The current interception schema stores measured absolute TCP ``s`` in
``/action[:, 0]``.  Relative action chunks are constructed by the training
loader.  This inspector therefore reports both:

1. the stored scalar action values; and
2. loader-style chunks ``action[t:t+K] - action[t]``, padded at the episode
   tail by repeating the final action value.

Only NumPy and h5py are required.  The script does not load complete image
datasets unless ``--image-check-samples`` is explicitly enabled.
"""
=======
"""Inspect interception HDF5 datasets and summarize measured and desired-goal targets."""
>>>>>>> Stashed changes

import argparse
import csv
import glob
import os
<<<<<<< Updated upstream
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
=======
from collections import defaultdict
from typing import Iterable, List, Optional
>>>>>>> Stashed changes

import h5py
import numpy as np


<<<<<<< Updated upstream
CONSISTENCY_ATTRIBUTES = (
    "action_coordinate",
    "action_origin",
    "action_positive_direction",
    "action_representation",
    "action_sampling_policy",
    "action_source_topic",
    "action_type",
    "action_units",
    "delta_action_construction",
    "fps",
    "joint_source_topic",
    "rgb_source_topic",
)


def expand_paths(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []

=======
def expand_paths(patterns: List[str]) -> List[str]:
    paths: List[str] = []
>>>>>>> Stashed changes
    for pattern in patterns:
        matches = glob.glob(pattern)
        candidates = matches if matches else [pattern]
        for candidate in candidates:
            if os.path.isdir(candidate):
                paths.extend(glob.glob(os.path.join(candidate, "*.hdf5")))
                paths.extend(glob.glob(os.path.join(candidate, "*.h5")))
            else:
                paths.append(candidate)
    return sorted(set(paths))


def format_attribute_value(value) -> str:
<<<<<<< Updated upstream
    """Produce a readable representation of an HDF5 attribute."""
=======
>>>>>>> Stashed changes
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return np.array2string(value, threshold=20, edgeitems=5, separator=", ")
    return repr(value)


<<<<<<< Updated upstream
def normalized_attribute_value(value):
    """Convert an HDF5 attribute to a hashable, comparable value."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return tuple(normalized_attribute_value(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return tuple(normalized_attribute_value(item) for item in value)
    return value


=======
>>>>>>> Stashed changes
def print_attributes(obj, indent: str) -> None:
    for key, value in obj.attrs.items():
        print(f"{indent}@{key} = {format_attribute_value(value)}")


def print_hdf5_structure(path: str, hdf5_file: h5py.File) -> None:
<<<<<<< Updated upstream
    """Print structure and metadata without loading complete datasets."""
=======
>>>>>>> Stashed changes
    print("\n" + "=" * 100)
    print(f"HDF5 STRUCTURE: {path}")
    print("=" * 100)
    print("/  [File]")
    print_attributes(hdf5_file, indent="  ")

    def visitor(name, obj):
        depth = name.count("/") + 1
        indent = "  " * depth
        basename = name.rsplit("/", 1)[-1]
        if isinstance(obj, h5py.Group):
            print(f"{indent}{basename}/  [Group]")
            print_attributes(obj, indent + "  ")
            return
        if isinstance(obj, h5py.Dataset):
            details = [f"shape={obj.shape}", f"dtype={obj.dtype}"]
            if obj.maxshape != obj.shape:
                details.append(f"maxshape={obj.maxshape}")
            if obj.chunks is not None:
                details.append(f"chunks={obj.chunks}")
            if obj.compression is not None:
                details.append(f"compression={obj.compression}")
                if obj.compression_opts is not None:
                    details.append(f"compression_opts={obj.compression_opts}")
            if obj.shuffle:
                details.append("shuffle=True")
            if obj.fletcher32:
                details.append("fletcher32=True")
            print(f"{indent}{basename}  [Dataset: " + ", ".join(details) + "]")
            print_attributes(obj, indent + "  ")

    hdf5_file.visititems(visitor)
    print("=" * 100)


<<<<<<< Updated upstream
class WarningCollector:
    """Collect category-level warnings and print each category once."""

    def __init__(self) -> None:
        self._messages: Dict[str, List[str]] = defaultdict(list)

    def add(self, category: str, message: str) -> None:
        self._messages[category].append(message)

    def print_summary(self) -> None:
        if not self._messages:
            print("No audit warnings.")
            return

        for category in sorted(self._messages):
            messages = self._messages[category]
            print(f"[WARN] {category}: {len(messages)} occurrence(s)")
            for message in messages[:5]:
                print(f"       {message}")
            if len(messages) > 5:
                print(f"       ... and {len(messages) - 5} more")


def print_counts(name: str, values: np.ndarray, zero_tol: float) -> None:
    values = np.asarray(values, dtype=np.float64)
    negative = int(np.sum(values < -zero_tol))
    zero = int(np.sum(np.abs(values) <= zero_tol))
    positive = int(np.sum(values > zero_tol))
    total = len(values)
=======
def finite_summary(values: Iterable[float]) -> Optional[dict]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def print_metric_summary(label: str, values: Iterable[float], unit: str = "") -> None:
    summary = finite_summary(values)
    if summary is None:
        print(f"{label}: no finite values")
        return
    suffix = f" {unit}" if unit else ""
    print(
        f"{label}: n={summary['count']}, "
        f"min/p25/median/mean/p75/max="
        f"{summary['min']:.4f}/"
        f"{summary['p25']:.4f}/"
        f"{summary['median']:.4f}/"
        f"{summary['mean']:.4f}/"
        f"{summary['p75']:.4f}/"
        f"{summary['max']:.4f}{suffix}"
    )
>>>>>>> Stashed changes


def direction_label(direction: int) -> str:
    if direction > 0:
        return "+s"
    if direction < 0:
        return "-s"
    return "near-zero"


def _direction_from_scalar(value: float, zero_tol: float) -> int:
    if not np.isfinite(value):
        return 0
    if value > zero_tol:
        return 1
    if value < -zero_tol:
        return -1
    return 0


def finite_summary(values: Iterable[float]) -> Optional[Dict[str, float]]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None

    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def print_metric_summary(
    label: str,
    values: Iterable[float],
    scale: float = 1.0,
    unit: str = "",
) -> None:
    summary = finite_summary(values)
    if summary is None:
        print(f"{label}: no finite values")
        return

    suffix = f" {unit}" if unit else ""
    print(
        f"{label}: n={summary['count']}, "
        f"min/p25/median/mean/p75/max="
        f"{summary['min'] * scale:.4f}/"
        f"{summary['p25'] * scale:.4f}/"
        f"{summary['median'] * scale:.4f}/"
        f"{summary['mean'] * scale:.4f}/"
        f"{summary['p75'] * scale:.4f}/"
        f"{summary['max'] * scale:.4f}{suffix}"
    )


def safe_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]

    if x_array.size < 2:
        return float("nan")
    if np.std(x_array) == 0.0 or np.std(y_array) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def first_persistent_index(mask: np.ndarray, persistence: int) -> Optional[int]:
    if persistence <= 1:
        indices = np.flatnonzero(mask)
        return int(indices[0]) if indices.size else None

    if len(mask) < persistence:
        return None

    run = np.convolve(
        mask.astype(np.int32),
        np.ones(persistence, dtype=np.int32),
        mode="valid",
    )
    indices = np.flatnonzero(run >= persistence)
    return int(indices[0]) if indices.size else None


def elapsed_time(
    index: Optional[int],
    timestamps: np.ndarray,
    fallback_fps: float,
) -> float:
    if index is None:
        return float("nan")
    if (
        len(timestamps) > index
        and len(timestamps) > 0
        and np.isfinite(timestamps[0])
        and np.isfinite(timestamps[index])
    ):
        return float(timestamps[index] - timestamps[0])
    return float(index / fallback_fps)


def direction_label(direction: int) -> str:
    if direction > 0:
        return "+s"
    if direction < 0:
        return "-s"
    return "near-zero"


@dataclass
class EpisodeRecord:
    path: str
    source_index: int
    sequence_index: int
    samples: int
    fps: float
    representation: str
    start_s: float
    final_s: float
    displacement: float
    direction: int
    target_magnitude: float
    duration_sec: float
    median_dt_sec: float
    motion_onset_sec: float
    t50_sec: float
    t90_sec: float
    max_step_m: float
    clipped_samples: int
    observation_duplicate_timestamps: int
    observation_large_gaps: int
    source_duplicate_timestamps: int
    source_reverse_timestamps: int
    future_source_matches: int
    age_mismatch_samples: int
    action_age_mean_sec: float
    action_age_max_sec: float
    chunk_anchor_count: int
    chunk_padded_fraction: float
    command_count_attr: int
    goto_s_count: int
    goto_target_count: int
    image_shape: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class ChunkAccumulator:
    chunk_size: int
    raw_by_direction: Dict[int, List[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )
    completion_by_direction: Dict[int, List[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )
    real_mask_by_direction: Dict[int, List[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )
    early_raw_by_direction: Dict[int, List[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )
    early_completion_by_direction: Dict[int, List[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )
    early_real_mask_by_direction: Dict[int, List[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_episode(
        self,
        action: np.ndarray,
        direction: int,
        target_magnitude: float,
        early_anchor_index: int,
    ) -> Tuple[int, float]:
        if direction == 0 or not np.isfinite(target_magnitude):
            return 0, float("nan")
        if target_magnitude <= 0.0 or action.size == 0:
            return 0, float("nan")

        chunks = []
        completions = []
        masks = []
        padded_tokens = 0
        total_tokens = 0

        for anchor in range(len(action)):
            real_count = min(self.chunk_size, len(action) - anchor)
            chunk = np.empty(self.chunk_size, dtype=np.float64)
            chunk[:real_count] = action[anchor : anchor + real_count]
            if real_count < self.chunk_size:
                chunk[real_count:] = action[-1]

            delta = chunk - chunk[0]
            mask = np.arange(self.chunk_size) < real_count
            completion = direction * delta / target_magnitude

            chunks.append(delta)
            completions.append(completion)
            masks.append(mask)
            padded_tokens += self.chunk_size - real_count
            total_tokens += self.chunk_size

        chunk_array = np.stack(chunks)
        completion_array = np.stack(completions)
        mask_array = np.stack(masks)

        self.raw_by_direction[direction].append(chunk_array)
        self.completion_by_direction[direction].append(completion_array)
        self.real_mask_by_direction[direction].append(mask_array)

        early_anchor = min(max(0, early_anchor_index), len(action) - 1)
        self.early_raw_by_direction[direction].append(
            chunk_array[early_anchor : early_anchor + 1]
        )
        self.early_completion_by_direction[direction].append(
            completion_array[early_anchor : early_anchor + 1]
        )
        self.early_real_mask_by_direction[direction].append(
            mask_array[early_anchor : early_anchor + 1]
        )

        padded_fraction = (
            padded_tokens / total_tokens if total_tokens else float("nan")
        )
        return len(action), padded_fraction

    @staticmethod
    def concatenate(
        mapping: Dict[int, List[np.ndarray]],
        direction: int,
        chunk_size: int,
        dtype=np.float64,
    ) -> np.ndarray:
        arrays = mapping.get(direction, [])
        if not arrays:
            return np.empty((0, chunk_size), dtype=dtype)
        return np.concatenate(arrays, axis=0)


def resolve_timestamps(
    group: h5py.Group,
    expected_length: int,
    warnings: WarningCollector,
    path: str,
) -> np.ndarray:
    if "timestamps" not in group:
        warnings.add("missing observation timestamps", path)
        return np.full(expected_length, np.nan, dtype=np.float64)

    timestamps = np.asarray(group["timestamps"][:], dtype=np.float64).reshape(-1)
    if len(timestamps) != expected_length:
        warnings.add(
            "length mismatch",
            f"{path}: observations/timestamps={len(timestamps)}, action={expected_length}",
        )
        result = np.full(expected_length, np.nan, dtype=np.float64)
        copy_count = min(expected_length, len(timestamps))
        result[:copy_count] = timestamps[:copy_count]
        return result
    return timestamps


def count_large_gaps(
    timestamps: np.ndarray,
    gap_factor: float,
) -> Tuple[int, int, float]:
    if len(timestamps) < 2:
        return 0, 0, float("nan")

    differences = np.diff(timestamps)
    finite_positive = differences[
        np.isfinite(differences) & (differences > 0.0)
    ]
    median_dt = (
        float(np.median(finite_positive))
        if finite_positive.size
        else float("nan")
    )
    duplicates_or_reverse = int(
        np.sum(np.isfinite(differences) & (differences <= 0.0))
    )
    large_gaps = 0
    if np.isfinite(median_dt) and median_dt > 0.0:
        large_gaps = int(
            np.sum(
                np.isfinite(differences)
                & (differences > gap_factor * median_dt)
            )
        )
    return duplicates_or_reverse, large_gaps, median_dt


def parse_history_offsets(text: str) -> List[int]:
    try:
        offsets = [int(item.strip()) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "history offsets must be comma-separated integers"
        ) from exc

    if not offsets or any(offset < 0 for offset in offsets):
        raise argparse.ArgumentTypeError(
            "history offsets must contain non-negative integers"
        )
    if 0 not in offsets:
        raise argparse.ArgumentTypeError(
            "history offsets must include 0 for the current observation"
        )
    return sorted(set(offsets), reverse=True)


def read_scalar_dataset(
    hdf5_file: h5py.File,
    name: str,
) -> Optional[np.ndarray]:
    if name not in hdf5_file:
        return None
    return np.asarray(hdf5_file[name][:], dtype=np.float64).reshape(-1)


def command_dataset_count(
    hdf5_file: h5py.File,
    dataset_name: str,
) -> int:
    if dataset_name not in hdf5_file:
        return 0
    return int(hdf5_file[dataset_name].shape[0])


def audit_episode(
    path: str,
    file_index: int,
    hdf5_file: h5py.File,
    args,
    warnings: WarningCollector,
    chunks: ChunkAccumulator,
) -> Tuple[Optional[EpisodeRecord], Optional[np.ndarray], Optional[np.ndarray]]:
    if "action" not in hdf5_file:
        warnings.add("missing /action", path)
        return None, None, None

    action_dataset = hdf5_file["action"]
    if action_dataset.ndim != 2:
        warnings.add(
            "unsupported /action shape",
            f"{path}: {action_dataset.shape}, expected (T, D)",
        )
        return None, None, None

    if action_dataset.shape[0] == 0:
        warnings.add("empty /action", path)
        return None, None, None

    if action_dataset.shape[1] == 2:
        return None, None, np.asarray(action_dataset[:], dtype=np.float64)

    if action_dataset.shape[1] != 1:
        warnings.add(
            "unsupported /action shape",
            f"{path}: {action_dataset.shape}",
        )
        return None, None, None

    action = np.asarray(action_dataset[:, 0], dtype=np.float64)
    if not np.all(np.isfinite(action)):
        warnings.add("non-finite action", path)
        return None, None, None

    observations = hdf5_file.get("observations")
    if observations is None:
        warnings.add("missing observations group", path)
        timestamps = np.full(len(action), np.nan, dtype=np.float64)
    else:
        timestamps = resolve_timestamps(
            observations,
            len(action),
            warnings,
            path,
        )

    if len(timestamps) and not np.all(np.isfinite(timestamps)):
        warnings.add("non-finite observation timestamps", path)

    duplicate_ts, large_gaps, median_dt = count_large_gaps(
        timestamps,
        args.gap_factor,
    )
    if duplicate_ts:
        warnings.add(
            "non-increasing observation timestamps",
            f"{path}: {duplicate_ts} interval(s)",
        )
    if large_gaps:
        warnings.add(
            "large observation timestamp gaps",
            f"{path}: {large_gaps} interval(s)",
        )

    fps = float(hdf5_file.attrs.get("fps", args.fallback_fps))
    if not np.isfinite(fps) or fps <= 0.0:
        warnings.add("invalid fps", f"{path}: {fps!r}")
        fps = args.fallback_fps

    endpoint_window = min(args.endpoint_window, len(action))
    start_s = float(np.median(action[:endpoint_window]))
    final_s = float(np.median(action[-endpoint_window:]))
    displacement = final_s - start_s
    if displacement > args.direction_tol:
        direction = 1
    elif displacement < -args.direction_tol:
        direction = -1
    else:
        direction = 0
        warnings.add(
            "near-zero episode displacement",
            f"{path}: {displacement:+.6f} m",
        )

    target_magnitude = abs(displacement)
    relative = action - start_s
    motion_mask = np.abs(relative) > args.motion_threshold
    onset_index = first_persistent_index(
        motion_mask,
        args.motion_persistence,
    )

    if direction != 0 and target_magnitude > 0.0:
        signed_progress = direction * relative
        t50_index = first_persistent_index(
            signed_progress >= 0.50 * target_magnitude,
            args.motion_persistence,
        )
        t90_index = first_persistent_index(
            signed_progress >= 0.90 * target_magnitude,
            args.motion_persistence,
        )
    else:
        t50_index = None
        t90_index = None

    motion_onset_sec = elapsed_time(onset_index, timestamps, fps)
    t50_sec = elapsed_time(t50_index, timestamps, fps)
    t90_sec = elapsed_time(t90_index, timestamps, fps)

    if onset_index is None:
        warnings.add("no detected motion", path)
    if direction != 0 and t90_index is None:
        warnings.add("no 90% completion", path)

    if len(timestamps) >= 2 and np.all(np.isfinite(timestamps[[0, -1]])):
        duration_sec = float(timestamps[-1] - timestamps[0])
    else:
        duration_sec = float((len(action) - 1) / fps)

    max_step = (
        float(np.max(np.abs(np.diff(action))))
        if len(action) >= 2
        else 0.0
    )
    if max_step > args.jump_threshold:
        warnings.add(
            "large action jump",
            f"{path}: maximum step {max_step:.6f} m",
        )

    clipped_samples = int(
        np.sum(np.abs(action) >= args.s_limit - args.clip_tolerance)
    )
    if clipped_samples:
        warnings.add(
            "samples near configured s limit",
            f"{path}: {clipped_samples} sample(s)",
        )

    source_timestamps = read_scalar_dataset(
        hdf5_file,
        "action_source_timestamps",
    )
    stored_ages = read_scalar_dataset(
        hdf5_file,
        "action_source_age_sec",
    )

    source_duplicates = 0
    source_reversals = 0
    future_matches = 0
    age_mismatches = 0
    age_mean = float("nan")
    age_max = float("nan")

    if source_timestamps is None:
        warnings.add("missing action source timestamps", path)
    elif len(source_timestamps) != len(action):
        warnings.add(
            "length mismatch",
            f"{path}: action_source_timestamps={len(source_timestamps)}, action={len(action)}",
        )
    else:
        source_differences = np.diff(source_timestamps)
        source_duplicates = int(
            np.sum(np.isfinite(source_differences) & (source_differences == 0.0))
        )
        source_reversals = int(
            np.sum(np.isfinite(source_differences) & (source_differences < 0.0))
        )
        if source_reversals:
            warnings.add(
                "decreasing action-source timestamps",
                f"{path}: {source_reversals} interval(s)",
            )

        matched_ages = timestamps - source_timestamps
        finite_ages = matched_ages[np.isfinite(matched_ages)]
        if finite_ages.size:
            age_mean = float(np.mean(finite_ages))
            age_max = float(np.max(finite_ages))
        future_matches = int(
            np.sum(np.isfinite(matched_ages) & (matched_ages < -args.time_tol))
        )
        if future_matches:
            warnings.add(
                "future action-source match",
                f"{path}: {future_matches} sample(s)",
            )

        if stored_ages is None:
            warnings.add("missing action source ages", path)
        elif len(stored_ages) != len(action):
            warnings.add(
                "length mismatch",
                f"{path}: action_source_age_sec={len(stored_ages)}, action={len(action)}",
            )
        else:
            if not np.all(np.isfinite(stored_ages)):
                warnings.add("non-finite stored action source ages", path)
            valid = np.isfinite(matched_ages) & np.isfinite(stored_ages)
            age_mismatches = int(
                np.sum(
                    valid
                    & (
                        np.abs(matched_ages - stored_ages)
                        > args.age_consistency_tol
                    )
                )
            )
            if age_mismatches:
                warnings.add(
                    "stored action age mismatch",
                    f"{path}: {age_mismatches} sample(s)",
                )

    representation = str(
        normalized_attribute_value(
            hdf5_file.attrs.get("action_representation", "unknown")
        )
    )
    if representation not in ("absolute", "delta", "relative"):
        warnings.add(
            "unknown action representation",
            f"{path}: {representation!r}",
        )

    chunk_anchor_count, padded_fraction = chunks.add_episode(
        action=action,
        direction=direction,
        target_magnitude=target_magnitude,
        early_anchor_index=max(args.history_offsets),
    )

    image_shape: Tuple[int, ...] = ()
    if observations is not None and "images/rgb" in observations:
        image_shape = tuple(observations["images/rgb"].shape)
        if image_shape[0] != len(action):
            warnings.add(
                "length mismatch",
                f"{path}: images/rgb={image_shape[0]}, action={len(action)}",
            )
    else:
        warnings.add("missing RGB images", path)

    if observations is not None and "qpos" in observations:
        qpos = observations["qpos"]
        if qpos.shape[0] != len(action):
            warnings.add(
                "length mismatch",
                f"{path}: qpos={qpos.shape[0]}, action={len(action)}",
            )
        qpos_values = np.asarray(qpos[:], dtype=np.float64)
        if not np.all(np.isfinite(qpos_values)):
            warnings.add("non-finite qpos", path)
        if qpos_values.ndim != 2 or qpos_values.shape[1] != 7:
            warnings.add(
                "unexpected qpos shape",
                f"{path}: {qpos_values.shape}",
            )
    else:
        warnings.add("missing qpos", path)

    source_index = int(hdf5_file.attrs.get("source_episode_index", file_index))
    command_count_attr = int(hdf5_file.attrs.get("command_count", 0))
    goto_s_count = command_dataset_count(
        hdf5_file,
        "commands/goto_s/values",
    )
    goto_target_count = command_dataset_count(
        hdf5_file,
        "commands/goto_s_target_base/points",
    )

    if command_count_attr != max(goto_s_count, goto_target_count):
        warnings.add(
            "command count mismatch",
            f"{path}: attr={command_count_attr}, "
            f"goto_s={goto_s_count}, goto_target={goto_target_count}",
        )

    record = EpisodeRecord(
        path=path,
        source_index=source_index,
        sequence_index=file_index,
        samples=len(action),
        fps=fps,
        representation=representation,
        start_s=start_s,
        final_s=final_s,
        displacement=displacement,
        direction=direction,
        target_magnitude=target_magnitude,
        duration_sec=duration_sec,
        median_dt_sec=median_dt,
        motion_onset_sec=motion_onset_sec,
        t50_sec=t50_sec,
        t90_sec=t90_sec,
        max_step_m=max_step,
        clipped_samples=clipped_samples,
        observation_duplicate_timestamps=duplicate_ts,
        observation_large_gaps=large_gaps,
        source_duplicate_timestamps=source_duplicates,
        source_reverse_timestamps=source_reversals,
        future_source_matches=future_matches,
        age_mismatch_samples=age_mismatches,
        action_age_mean_sec=age_mean,
        action_age_max_sec=age_max,
        chunk_anchor_count=chunk_anchor_count,
        chunk_padded_fraction=padded_fraction,
        command_count_attr=command_count_attr,
        goto_s_count=goto_s_count,
        goto_target_count=goto_target_count,
        image_shape=image_shape,
    )
    return record, action, None


def print_attribute_consistency(
    attribute_values: Dict[str, Counter],
) -> None:
    print("\nSchema attribute consistency:")
    for attribute in CONSISTENCY_ATTRIBUTES:
        values = attribute_values.get(attribute, Counter())
        if not values:
            print(f"  {attribute}: MISSING IN ALL FILES")
            continue
        if len(values) == 1:
            value, count = next(iter(values.items()))
            print(f"  {attribute}: {value!r} ({count} file(s))")
        else:
            formatted = ", ".join(
                f"{value!r}: {count}" for value, count in values.items()
            )
            print(f"  {attribute}: MIXED {{{formatted}}}")


def records_by_direction(
    records: Sequence[EpisodeRecord],
    direction: int,
) -> List[EpisodeRecord]:
    return [record for record in records if record.direction == direction]


def print_direction_conditioned_episode_stats(
    records: Sequence[EpisodeRecord],
) -> None:
    print("\n" + "=" * 100)
    print("DIRECTION-CONDITIONED EPISODE STATISTICS")
    print("=" * 100)

    for direction in (-1, 1):
        subset = records_by_direction(records, direction)
        print(f"\n{direction_label(direction)} episodes: {len(subset)}")
        print_metric_summary(
            "  target |final-start|",
            (record.target_magnitude for record in subset),
            scale=100.0,
            unit="cm",
        )
        print_metric_summary(
            "  starting TCP s",
            (record.start_s for record in subset),
            scale=100.0,
            unit="cm",
        )
        print_metric_summary(
            "  final TCP s",
            (record.final_s for record in subset),
            scale=100.0,
            unit="cm",
        )
        print_metric_summary(
            "  samples",
            (record.samples for record in subset),
        )
        print_metric_summary(
            "  duration",
            (record.duration_sec for record in subset),
            unit="s",
        )
        print_metric_summary(
            "  motion onset",
            (record.motion_onset_sec for record in subset),
            unit="s",
        )
        print_metric_summary(
            "  50% completion time",
            (record.t50_sec for record in subset),
            unit="s",
        )
        print_metric_summary(
            "  90% completion time",
            (record.t90_sec for record in subset),
            unit="s",
        )
        print_metric_summary(
            "  action source age",
            (record.action_age_mean_sec for record in subset),
            scale=1000.0,
            unit="ms",
        )
        print_metric_summary(
            "  loader chunk padded fraction",
            (record.chunk_padded_fraction for record in subset),
            scale=100.0,
            unit="%",
        )


def token_arrays(
    chunks: ChunkAccumulator,
    early: bool,
    real_only: bool,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    if early:
        raw_mapping = chunks.early_raw_by_direction
        completion_mapping = chunks.early_completion_by_direction
        mask_mapping = chunks.early_real_mask_by_direction
    else:
        raw_mapping = chunks.raw_by_direction
        completion_mapping = chunks.completion_by_direction
        mask_mapping = chunks.real_mask_by_direction

    result = {}
    for direction in (-1, 1):
        raw = chunks.concatenate(
            raw_mapping,
            direction,
            chunks.chunk_size,
        )
        completion = chunks.concatenate(
            completion_mapping,
            direction,
            chunks.chunk_size,
        )
        mask = chunks.concatenate(
            mask_mapping,
            direction,
            chunks.chunk_size,
            dtype=bool,
        ).astype(bool)

        if real_only:
            raw = np.where(mask, raw, np.nan)
            completion = np.where(mask, completion, np.nan)
        result[direction] = (raw, completion)
    return result


def nan_stat(values: np.ndarray, function, default=float("nan")) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(function(finite))


def print_token_table(
    chunks: ChunkAccumulator,
    early: bool,
    real_only: bool,
    title: str,
) -> None:
    arrays = token_arrays(chunks, early=early, real_only=real_only)
    raw_negative, completion_negative = arrays[-1]
    raw_positive, completion_positive = arrays[1]

    if raw_negative.size == 0 and raw_positive.size == 0:
        print(f"\n{title}: no usable chunks")
        return

    if early:
        mask_mapping = chunks.early_real_mask_by_direction
    else:
        mask_mapping = chunks.real_mask_by_direction
    all_masks = []
    for direction in (-1, 1):
        mask = chunks.concatenate(
            mask_mapping,
            direction,
            chunks.chunk_size,
            dtype=bool,
        ).astype(bool)
        if mask.size:
            all_masks.append(mask)
    combined_mask = (
        np.concatenate(all_masks, axis=0)
        if all_masks
        else np.empty((0, chunks.chunk_size), dtype=bool)
    )

    print("\n" + title)
    print(
        "token  real%   +s mean   -s mirrored   symmetry gap   "
        "+s median completion   -s median completion"
    )
    print(
        "-----  -----  ---------  -----------  ------------  "
        "---------------------  ---------------------"
    )

    for token in range(chunks.chunk_size):
        positive_mean = (
            nan_stat(raw_positive[:, token], np.mean)
            if raw_positive.size
            else float("nan")
        )
        negative_mirrored = (
            -nan_stat(raw_negative[:, token], np.mean)
            if raw_negative.size
            else float("nan")
        )
        symmetry_gap = positive_mean - negative_mirrored
        positive_completion = (
            nan_stat(completion_positive[:, token], np.median)
            if completion_positive.size
            else float("nan")
        )
        negative_completion = (
            nan_stat(completion_negative[:, token], np.median)
            if completion_negative.size
            else float("nan")
        )
        real_fraction = (
            float(np.mean(combined_mask[:, token]))
            if combined_mask.size
            else float("nan")
        )

        print(
            f"{token:5d}  "
            f"{real_fraction:5.1%}  "
            f"{positive_mean * 1000:+8.2f} mm  "
            f"{negative_mirrored * 1000:+8.2f} mm  "
            f"{symmetry_gap * 1000:+9.2f} mm  "
            f"{positive_completion:20.1%}  "
            f"{negative_completion:20.1%}"
        )


def print_order_effects(records: Sequence[EpisodeRecord]) -> None:
    ordered = sorted(records, key=lambda record: record.source_index)
    directions = np.asarray(
        [record.direction for record in ordered],
        dtype=np.float64,
    )
    indices = np.asarray(
        [record.source_index for record in ordered],
        dtype=np.float64,
    )
    magnitudes = np.asarray(
        [record.target_magnitude for record in ordered],
        dtype=np.float64,
    )
    starts = np.asarray(
        [record.start_s for record in ordered],
        dtype=np.float64,
    )

    nonzero = directions != 0
    directions = directions[nonzero]
    indices = indices[nonzero]
    magnitudes = magnitudes[nonzero]
    starts = starts[nonzero]

    print("\n" + "=" * 100)
    print("ORDER, START-POSITION, AND SESSION-PROXY CHECKS")
    print("=" * 100)

    if len(directions) < 2:
        print("Not enough directional episodes.")
        return

    alternations = directions[1:] != directions[:-1]
    alternating_fraction = float(np.mean(alternations))

    longest_run = 1
    current_run = 1
    for changed in alternations:
        if changed:
            current_run = 1
        else:
            current_run += 1
            longest_run = max(longest_run, current_run)

    print(f"direction alternation fraction: {alternating_fraction:.1%}")
    print(f"longest same-direction run: {longest_run}")
    print(
        "correlation(direction sign, source episode index): "
        f"{safe_correlation(directions, indices):+.4f}"
    )
    print(
        "correlation(target magnitude, source episode index): "
        f"{safe_correlation(magnitudes, indices):+.4f}"
    )
    print(
        "correlation(starting TCP s, direction sign): "
        f"{safe_correlation(starts, directions):+.4f}"
    )

    split = len(magnitudes) // 2
    if split > 0:
        first_half = magnitudes[:split]
        second_half = magnitudes[split:]
        print(
            "first-half / second-half mean target magnitude: "
            f"{np.mean(first_half) * 100:.3f} / "
            f"{np.mean(second_half) * 100:.3f} cm"
        )


def write_episode_csv(path: str, records: Sequence[EpisodeRecord]) -> None:
    fieldnames = list(EpisodeRecord.__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                name: getattr(record, name)
                for name in fieldnames
            }
            row["image_shape"] = "x".join(
                str(value) for value in record.image_shape
            )
            writer.writerow(row)


def write_token_csv(path: str, chunks: ChunkAccumulator) -> None:
    with open(path, "w", newline="", encoding="utf-8") as output:
        fieldnames = [
            "scope",
            "token",
            "direction",
            "sample_count",
            "real_token_fraction",
            "raw_delta_mean_m",
            "raw_delta_median_m",
            "raw_delta_p25_m",
            "raw_delta_p75_m",
            "completion_mean",
            "completion_median",
            "completion_p25",
            "completion_p75",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for scope, early, real_only in (
            ("all_anchors_training_padded", False, False),
            ("all_anchors_real_only", False, True),
            ("early_anchor_training_padded", True, False),
        ):
            mapping = token_arrays(
                chunks,
                early=early,
                real_only=real_only,
            )
            mask_mapping = (
                chunks.early_real_mask_by_direction
                if early
                else chunks.real_mask_by_direction
            )
            for direction in (-1, 1):
                raw, completion = mapping[direction]
                real_mask = chunks.concatenate(
                    mask_mapping,
                    direction,
                    chunks.chunk_size,
                    dtype=bool,
                ).astype(bool)
                for token in range(chunks.chunk_size):
                    raw_values = (
                        raw[:, token]
                        if raw.size
                        else np.asarray([], dtype=np.float64)
                    )
                    completion_values = (
                        completion[:, token]
                        if completion.size
                        else np.asarray([], dtype=np.float64)
                    )
                    raw_summary = finite_summary(raw_values)
                    completion_summary = finite_summary(completion_values)
                    writer.writerow(
                        {
                            "scope": scope,
                            "token": token,
                            "direction": direction_label(direction),
                            "sample_count": (
                                raw_summary["count"] if raw_summary else 0
                            ),
                            "real_token_fraction": (
                                float(np.mean(real_mask[:, token]))
                                if real_mask.size
                                else float("nan")
                            ),
                            "raw_delta_mean_m": (
                                raw_summary["mean"]
                                if raw_summary
                                else float("nan")
                            ),
                            "raw_delta_median_m": (
                                raw_summary["median"]
                                if raw_summary
                                else float("nan")
                            ),
                            "raw_delta_p25_m": (
                                raw_summary["p25"]
                                if raw_summary
                                else float("nan")
                            ),
                            "raw_delta_p75_m": (
                                raw_summary["p75"]
                                if raw_summary
                                else float("nan")
                            ),
                            "completion_mean": (
                                completion_summary["mean"]
                                if completion_summary
                                else float("nan")
                            ),
                            "completion_median": (
                                completion_summary["median"]
                                if completion_summary
                                else float("nan")
                            ),
                            "completion_p25": (
                                completion_summary["p25"]
                                if completion_summary
                                else float("nan")
                            ),
                            "completion_p75": (
                                completion_summary["p75"]
                                if completion_summary
                                else float("nan")
                            ),
                        }
                    )


def run_optional_image_check(
    records: Sequence[EpisodeRecord],
    samples_per_direction: int,
    warnings: WarningCollector,
) -> None:
    print("\n" + "=" * 100)
    print("OPTIONAL RGB SAMPLE CHECK")
    print("=" * 100)

    if samples_per_direction <= 0:
        print(
            "Disabled. Use --image-check-samples N to read N adjacent "
            "frame pairs per direction."
        )
        return

    for direction in (-1, 1):
        subset = [
            record
            for record in records
            if record.direction == direction and record.image_shape
        ]
        if not subset:
            print(f"{direction_label(direction)}: no RGB datasets")
            continue

        pair_candidates: List[Tuple[str, int]] = []
        for record in subset:
            if record.samples < 2:
                continue
            indices = np.linspace(
                0,
                record.samples - 2,
                num=min(samples_per_direction, record.samples - 1),
                dtype=int,
            )
            pair_candidates.extend((record.path, int(index)) for index in indices)

        if len(pair_candidates) > samples_per_direction:
            selection = np.linspace(
                0,
                len(pair_candidates) - 1,
                num=samples_per_direction,
                dtype=int,
            )
            pair_candidates = [pair_candidates[index] for index in selection]

        means = []
        standard_deviations = []
        exact_duplicates = 0
        mean_absolute_differences = []

        for path, index in pair_candidates:
            try:
                with h5py.File(path, "r") as hdf5_file:
                    images = hdf5_file["observations/images/rgb"]
                    first = np.asarray(images[index], dtype=np.int16)
                    second = np.asarray(images[index + 1], dtype=np.int16)
            except (OSError, KeyError) as exc:
                warnings.add(
                    "RGB sample read failure",
                    f"{path}: {exc}",
                )
                continue

            means.extend((float(np.mean(first)), float(np.mean(second))))
            standard_deviations.extend(
                (float(np.std(first)), float(np.std(second)))
            )
            difference = np.abs(second - first)
            mean_absolute_differences.append(float(np.mean(difference)))
            exact_duplicates += int(np.array_equal(first, second))

        print(f"\n{direction_label(direction)}:")
        print(f"  adjacent pairs read: {len(mean_absolute_differences)}")
        print(f"  exact duplicate pairs: {exact_duplicates}")
        print_metric_summary("  pixel mean", means)
        print_metric_summary("  pixel standard deviation", standard_deviations)
        print_metric_summary(
            "  adjacent-frame mean absolute difference",
            mean_absolute_differences,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
<<<<<<< Updated upstream
            "Print HDF5 structure, inspect stored interception actions, "
            "and audit loader-style relative action chunks, direction "
            "symmetry, timing, padding, and integrity."
=======
            "Print HDF5 structure and inspect interception-action balance, "
            "including desired-goal target coverage when present."
>>>>>>> Stashed changes
        )
    )
    parser.add_argument("paths", nargs="+", help="HDF5 files, dataset directories, or glob patterns.")
    parser.add_argument(
        "--zero-tol",
        type=float,
        default=1e-6,
<<<<<<< Updated upstream
        help=(
            "Absolute values at or below this threshold are classified "
            "as near zero. Default: 1e-6."
        ),
=======
        help="Absolute delta-s values at or below this threshold are classified as near zero.",
>>>>>>> Stashed changes
    )
    parser.add_argument(
        "--structure",
        choices=("none", "first", "all"),
        default="first",
<<<<<<< Updated upstream
        help=(
            "Print no structures, the structure of the first file, or "
            "the structure of every file. Default: first."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Policy action chunk length. Default: 30.",
    )
    parser.add_argument(
        "--history-offsets",
        type=parse_history_offsets,
        default=parse_history_offsets("6,3,0"),
        help=(
            "Comma-separated observation-history offsets. The largest "
            "offset selects the early policy anchor. Default: 6,3,0."
        ),
    )
    parser.add_argument(
        "--endpoint-window",
        type=int,
        default=5,
        help=(
            "Samples median-filtered at each episode endpoint when "
            "estimating required displacement. Default: 5."
        ),
    )
    parser.add_argument(
        "--direction-tol",
        type=float,
        default=0.005,
        help=(
            "Episodes with |final-start| at or below this value are "
            "classified as near-zero. Default: 0.005 m."
        ),
=======
        help="Print no structures, the complete structure of the first file, or every file.",
>>>>>>> Stashed changes
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.002,
        help=(
            "Absolute displacement from the start used for motion-onset "
            "detection. Default: 0.002 m."
        ),
    )
    parser.add_argument(
        "--motion-persistence",
        type=int,
        default=3,
        help=(
            "Consecutive samples required for motion/completion events. "
            "Default: 3."
        ),
    )
    parser.add_argument(
        "--gap-factor",
        type=float,
        default=1.5,
        help=(
            "Timestamp intervals above this multiple of the median are "
            "reported as large gaps. Default: 1.5."
        ),
    )
    parser.add_argument(
        "--time-tol",
        type=float,
        default=1e-9,
        help=(
            "Tolerance for detecting future source-message matching. "
            "Default: 1e-9 s."
        ),
    )
    parser.add_argument(
        "--age-consistency-tol",
        type=float,
        default=1e-5,
        help=(
            "Tolerance between stored source age and timestamp-derived "
            "age. Default: 1e-5 s."
        ),
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=0.02,
        help=(
            "Single-sample stored-action changes above this magnitude are "
            "reported. Default: 0.02 m."
        ),
    )
    parser.add_argument(
        "--s-limit",
        type=float,
        default=0.15,
        help="Expected absolute s limit used for clipping checks. Default: 0.15 m.",
    )
    parser.add_argument(
        "--clip-tolerance",
        type=float,
        default=0.001,
        help=(
            "Distance from --s-limit classified as near clipping. "
            "Default: 0.001 m."
        ),
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=30.0,
        help="FPS used when the file attribute is absent. Default: 30.",
    )
    parser.add_argument(
        "--image-check-samples",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Read N sampled adjacent RGB frame pairs per direction and "
            "check basic variation/duplication. Default: 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--episode-csv",
        help="Optional output CSV containing one audit row per episode.",
    )
    parser.add_argument(
        "--token-csv",
        help="Optional output CSV containing direction-conditioned token statistics.",
    )
    return parser


def validate_arguments(args, parser: argparse.ArgumentParser) -> None:
    positive_integer_fields = (
        "chunk_size",
        "endpoint_window",
        "motion_persistence",
    )
    for field_name in positive_integer_fields:
        if getattr(args, field_name) <= 0:
            parser.error(f"--{field_name.replace('_', '-')} must be positive")

    if args.image_check_samples < 0:
        parser.error("--image-check-samples must be non-negative")
    if args.fallback_fps <= 0.0:
        parser.error("--fallback-fps must be positive")
    if args.gap_factor <= 1.0:
        parser.error("--gap-factor must be greater than 1")
    if args.s_limit <= 0.0:
        parser.error("--s-limit must be positive")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_arguments(args, parser)

    paths = expand_paths(args.paths)
    if not paths:
        raise RuntimeError("No HDF5 files found.")

    missing_paths = [path for path in paths if not os.path.isfile(path)]
    for path in missing_paths:
        print(f"[WARN] file not found: {path}")
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
    paths = [path for path in paths if os.path.isfile(path)]
    if not paths:
        raise RuntimeError("None of the resolved paths are valid files.")

<<<<<<< Updated upstream
    warnings = WarningCollector()
    chunks = ChunkAccumulator(chunk_size=args.chunk_size)
    records: List[EpisodeRecord] = []
    stored_scalar_values: List[np.ndarray] = []
    episode_stored_means: List[float] = []
    episode_displacements: List[float] = []
    legacy_flags: List[np.ndarray] = []
    attribute_values: Dict[str, Counter] = defaultdict(Counter)
=======
    delta_s_values = []
    episode_mean_delta_s = []
    episode_net_delta_s = []
    legacy_flags = []
    selected_goto_s_counts = []
    desired_goal_values = []
    desired_goal_deltas = []
    desired_goal_start_vs_goal = []
    desired_goal_by_direction = {"+s": [], "-s": [], "near-zero": []}
    missing_or_ambiguous_goal_episodes = []
>>>>>>> Stashed changes

    opened_files = 0
    skipped_files = 0

    for file_index, path in enumerate(paths):
        try:
<<<<<<< Updated upstream
            with h5py.File(path, "r") as hdf5_file:
                opened_files += 1
                if (
                    args.structure == "all"
                    or (args.structure == "first" and file_index == 0)
                ):
                    print_hdf5_structure(path, hdf5_file)

                for attribute in CONSISTENCY_ATTRIBUTES:
                    if attribute in hdf5_file.attrs:
                        value = normalized_attribute_value(
                            hdf5_file.attrs[attribute]
                        )
                        attribute_values[attribute][value] += 1

                record, action, legacy = audit_episode(
                    path,
                    file_index,
                    hdf5_file,
                    args,
                    warnings,
                    chunks,
                )
=======
            with h5py.File(path, "r") as f:
                if args.structure == "all" or (args.structure == "first" and file_index == 0):
                    print_hdf5_structure(path, f)

                if "action" not in f:
                    print(f"[WARN] skipping action analysis for {path}: missing /action")
                    skipped_files += 1
                    continue

                action = np.asarray(f["action"][:])
                if action.ndim != 2:
                    print(f"[WARN] skipping action analysis for {path}: /action shape is {action.shape}, expected (T, D)")
                    skipped_files += 1
                    continue
                if action.shape[0] == 0:
                    print(f"[WARN] skipping {path}: /action is empty")
                    skipped_files += 1
                    continue

                if action.shape[1] == 1:
                    episode_delta_s = action[:, 0]
                    if not np.all(np.isfinite(episode_delta_s)):
                        print(f"[WARN] skipping {path}: non-finite delta_s values")
                        skipped_files += 1
                        continue

                    delta_s_values.append(episode_delta_s)
                    episode_mean_delta_s.append(float(np.mean(episode_delta_s)))
                    episode_net_delta_s.append(float(np.sum(episode_delta_s)))
                    current_files += 1

                    selected_values = None
                    if (
                        "/commands/selected_goto_s/timestamps" in f
                        and "/commands/selected_goto_s/values" in f
                        and "/targets/desired_intercept_s" in f
                    ):
                        selected_values = np.asarray(f["/commands/selected_goto_s/values"][:], dtype=np.float32).reshape(-1)
                        selected_count = int(np.sum(np.isfinite(selected_values)))
                        selected_goto_s_counts.append(selected_count)
                        if selected_count != 1:
                            missing_or_ambiguous_goal_episodes.append((path, selected_count))
                        else:
                            desired_goal = float(np.asarray(f["/targets/desired_intercept_s"][:], dtype=np.float32).reshape(-1)[0])
                            if not np.isfinite(desired_goal):
                                missing_or_ambiguous_goal_episodes.append((path, selected_count))
                            else:
                                desired_goal_values.append(desired_goal)
                                delta_goal = desired_goal - episode_delta_s
                                desired_goal_deltas.extend(delta_goal.tolist())
                                desired_goal_start_vs_goal.append((float(episode_delta_s[0]), desired_goal))
                                bucket = direction_label(_direction_from_scalar(desired_goal, args.zero_tol))
                                desired_goal_by_direction[bucket].append(desired_goal)

                elif action.shape[1] == 2:
                    flags = action[:, 1]
                    if not np.all(np.isfinite(flags)):
                        print(f"[WARN] skipping {path}: non-finite commanded flags")
                        skipped_files += 1
                        continue
                    legacy_flags.append(flags)
                    legacy_files += 1
                else:
                    print(f"[WARN] skipping action analysis for {path}: unsupported /action shape {action.shape}")
                    skipped_files += 1
>>>>>>> Stashed changes
        except OSError as exc:
            warnings.add("unable to open file", f"{path}: {exc}")
            skipped_files += 1
<<<<<<< Updated upstream
            continue

        if legacy is not None:
            flags = legacy[:, 1]
            if not np.all(np.isfinite(flags)):
                warnings.add("non-finite legacy flags", path)
                skipped_files += 1
                continue
            legacy_flags.append(flags)
            continue

        if record is None or action is None:
            skipped_files += 1
            continue

        records.append(record)
        stored_scalar_values.append(action)
        episode_stored_means.append(float(np.mean(action)))
        episode_displacements.append(record.displacement)
=======
>>>>>>> Stashed changes

    print("\n" + "=" * 100)
    print("DATASET SUMMARY")
    print("=" * 100)
<<<<<<< Updated upstream
    print(f"files resolved:                 {len(paths)}")
    print(f"files opened:                   {opened_files}")
    print(f"current scalar-action files:    {len(records)}")
    print(f"legacy commanded-flag files:    {len(legacy_flags)}")
    print(f"skipped/unsupported files:      {skipped_files}")

    if stored_scalar_values:
        stored = np.concatenate(stored_scalar_values)
        episode_means = np.asarray(episode_stored_means)
        displacements = np.asarray(episode_displacements)

        print("\nStored current schema: /action shape (T, 1)")
        print(
            "Important: these are stored scalar values. For files with "
            "action_representation='absolute', they are absolute TCP s, "
            "not delta_s."
        )
        print(f"stored action samples: {len(stored)}")
        print(
            "stored action min/mean/max: "
            f"{stored.min():+.8f} / "
            f"{stored.mean():+.8f} / "
            f"{stored.max():+.8f}"
        )
        print(
            "|stored action| min/mean/max: "
            f"{np.abs(stored).min():.8f} / "
            f"{np.abs(stored).mean():.8f} / "
            f"{np.abs(stored).max():.8f}"
        )
        print_counts(
            "Per-sample stored action balance",
            stored,
            args.zero_tol,
        )
        print_counts(
            "Per-episode mean stored action balance",
            episode_means,
            args.zero_tol,
        )
        print_counts(
            "Per-episode endpoint displacement balance (final-start)",
            displacements,
            args.direction_tol,
        )
=======
    print(f"files found:                  {len(paths)}")
    print(f"current delta-s files:        {current_files}")
    print(f"legacy commanded-flag files: {legacy_files}")
    print(f"skipped/unsupported files:    {skipped_files}")

    if delta_s_values:
        delta_s = np.concatenate(delta_s_values)
        episode_means = np.asarray(episode_mean_delta_s, dtype=np.float64)
        episode_nets = np.asarray(episode_net_delta_s, dtype=np.float64)

        print("\nCurrent schema: /action shape (T, 1)")
        print_metric_summary("delta_s", delta_s)
        print_metric_summary("|delta_s|", np.abs(delta_s))
        print_metric_summary("Per-episode mean delta_s", episode_means)
        print_metric_summary("Per-episode summed delta_s", episode_nets)

        if selected_goto_s_counts:
            print("\nSelected goal event counts:")
            print_metric_summary("selected_goto_s events per episode", selected_goto_s_counts)

        if desired_goal_values:
            print("\nDesired goal summary by direction:")
            for label in ("+s", "-s", "near-zero"):
                values = desired_goal_by_direction[label]
                if values:
                    print_metric_summary(f"desired_intercept_s {label}", values, unit="m")
            print_metric_summary("desired_intercept_s (all)", desired_goal_values, unit="m")
            print_metric_summary("delta_goal over all anchors", desired_goal_deltas, unit="m")
            print_metric_summary("starting TCP s", [pair[0] for pair in desired_goal_start_vs_goal], unit="m")
            print_metric_summary("goal s", [pair[1] for pair in desired_goal_start_vs_goal], unit="m")

        if missing_or_ambiguous_goal_episodes:
            print("\nEpisodes missing or containing multiple goal events:")
            for path, selected_count in missing_or_ambiguous_goal_episodes[:20]:
                print(f"  {path}: selected_goto_s finite_count={selected_count}")
            if len(missing_or_ambiguous_goal_episodes) > 20:
                print(f"  ... and {len(missing_or_ambiguous_goal_episodes) - 20} more")
>>>>>>> Stashed changes

    if legacy_flags:
        flags = np.concatenate(legacy_flags)
        print("\nLegacy schema: /action shape (T, 2)")
<<<<<<< Updated upstream
        print("unique commanded flags:", np.unique(flags))
        print("uncommanded:", int(np.sum(zero_flags)))
        print("commanded:", int(np.sum(one_flags)))
        print("commanded fraction:", float(np.mean(one_flags)))
        if not np.all(valid_binary):
            warnings.add(
                "non-binary legacy commanded flags",
                f"{int(np.sum(~valid_binary))} sample(s)",
            )
=======
        print(f"unique commanded flags: {np.unique(flags)}")
        print(f"uncommanded: {int(np.sum(np.isclose(flags, 0.0)))}")
        print(f"commanded: {int(np.sum(np.isclose(flags, 1.0)))}")
>>>>>>> Stashed changes

    print_attribute_consistency(attribute_values)

    if records:
        print_direction_conditioned_episode_stats(records)

        print("\n" + "=" * 100)
        print("LOADER-STYLE RELATIVE ACTION CHUNKS")
        print("=" * 100)
        print(
            f"construction: chunk = action[t:t+{args.chunk_size}] - action[t]; "
            "tail padding repeats the episode's final stored action"
        )
        print(
            f"early anchor: t=max(history offsets {args.history_offsets})="
            f"{max(args.history_offsets)}"
        )
        print(
            "Completion is direction * relative_delta / "
            "|median(final window)-median(start window)|."
        )

        print_token_table(
            chunks,
            early=True,
            real_only=False,
            title=(
                "Early-anchor token targets (one anchor per episode, "
                "including loader-style tail padding)"
            ),
        )
        print_token_table(
            chunks,
            early=False,
            real_only=False,
            title=(
                "All-anchor token targets (the training distribution, "
                "including loader-style tail padding)"
            ),
        )
        print_token_table(
            chunks,
            early=False,
            real_only=True,
            title=(
                "All-anchor real-only token targets "
                "(diagnostic view excluding padded token values)"
            ),
        )

        print_order_effects(records)

        print("\n" + "=" * 100)
        print("INTEGRITY TOTALS")
        print("=" * 100)
        print(
            "non-increasing observation timestamp intervals: "
            f"{sum(r.observation_duplicate_timestamps for r in records)}"
        )
        print(
            "large observation timestamp gaps: "
            f"{sum(r.observation_large_gaps for r in records)}"
        )
        print(
            "repeated action-source timestamps (allowed for as-of sampling): "
            f"{sum(r.source_duplicate_timestamps for r in records)}"
        )
        print(
            "decreasing action-source timestamp intervals: "
            f"{sum(r.source_reverse_timestamps for r in records)}"
        )
        print(
            "future action-source matches: "
            f"{sum(r.future_source_matches for r in records)}"
        )
        print(
            "stored/derived action-age mismatches: "
            f"{sum(r.age_mismatch_samples for r in records)}"
        )
        print(
            "samples near configured s limit: "
            f"{sum(r.clipped_samples for r in records)}"
        )
        print(
            "episodes with no recorded GOTO_S command: "
            f"{sum(r.goto_s_count == 0 for r in records)} / {len(records)}"
        )
        print(
            "episodes with no recorded GOTO_S target point: "
            f"{sum(r.goto_target_count == 0 for r in records)} / {len(records)}"
        )

        run_optional_image_check(
            records,
            args.image_check_samples,
            warnings,
        )

        if args.episode_csv:
            write_episode_csv(args.episode_csv, records)
            print(f"\nWrote per-episode audit CSV: {args.episode_csv}")
        if args.token_csv:
            write_token_csv(args.token_csv, chunks)
            print(f"Wrote token audit CSV: {args.token_csv}")

        print("\n" + "=" * 100)
        print("NOT AVAILABLE FROM THIS HDF5 SCHEMA")
        print("=" * 100)
        print(
            "- Ball initial position, velocity, approach angle, and observed "
            "intercept s: no ball/scene datasets are stored."
        )
        print(
            "- Train/validation split balance: no split assignment is stored "
            "in individual episode files."
        )
        print(
            "- Collection-session, lighting, and middle-line-recapture effects: "
            "no explicit session/condition identifiers are stored."
        )
        print(
            "- Visual verification of ball visibility and frame order still "
            "requires viewing sampled images; --image-check-samples only checks "
            "basic numeric variation and exact duplicates."
        )

    print("\n" + "=" * 100)
    print("AUDIT WARNINGS")
    print("=" * 100)
    warnings.print_summary()

    if not records and not legacy_flags:
        raise RuntimeError("No compatible /action datasets were found.")


if __name__ == "__main__":
    main()
