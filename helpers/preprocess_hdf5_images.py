#!/usr/bin/env python3
"""Safely mask and left-crop image datasets in interception HDF5 files."""

from __future__ import annotations

import argparse
import heapq
import math
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


ACT_ROOT = Path(__file__).resolve().parent.parent
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from image_preprocessing import (  # noqa: E402
    mask_and_center_crop_square_rotated_event_image,
    mask_and_left_crop_image,
    mask_and_left_crop_rotated_event_image,
    validate_image_transform,
)


TOOL_VERSION = "act.preprocess_hdf5_images/1"
METADATA_PREFIX = "act.preprocess_hdf5_images."
DATASET_KEYS = {
    "rgb": "/observations/images/rgb",
    "event": "/observations/images/event",
}
FILL_VALUES = {"rgb": 0, "event": 128}
TARGET_BATCH_BYTES = 64 * 1024 * 1024
REFERENCE_SCAN_BYTES = 16 * 1024 * 1024
SIMULATION_DIR = Path(__file__).resolve().parent / "crop_simulations"


@dataclass(frozen=True)
class DatasetPlan:
    modality: str
    key: str
    original_shape: Tuple[int, ...]
    result_shape: Tuple[int, ...]
    dtype: np.dtype
    logical_input_bytes: int
    logical_output_bytes: int


@dataclass(frozen=True)
class FilePlan:
    source: Path
    output: Path
    datasets: Tuple[DatasetPlan, ...]
    source_bytes: int


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _selected_modalities(modality: str) -> Tuple[str, ...]:
    return ("rgb", "event") if modality == "both" else (modality,)


def _validate_suffix(path: Path, label: str) -> None:
    if path.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError(f"{label} must end in .h5 or .hdf5: {path}")


def discover_inputs(top_dir: Path, output_dir: Path) -> List[Path]:
    top_resolved = top_dir.resolve()
    output_resolved = output_dir.resolve()
    if top_resolved == output_resolved:
        raise ValueError("--out-dir must not resolve to the same directory as --top-dir")
    discovered = []
    for path in top_resolved.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".h5", ".hdf5"):
            continue
        resolved = path.resolve()
        if _path_is_within(resolved, output_resolved):
            continue
        discovered.append(resolved)
    discovered.sort(key=lambda item: item.relative_to(top_resolved).as_posix())
    if not discovered:
        raise ValueError(f"No .h5 or .hdf5 files found under {top_resolved}")
    return discovered


def resolve_paths(args: argparse.Namespace) -> Tuple[str, Path, List[Tuple[Path, Path]]]:
    if args.input is not None:
        if args.out_dir is not None:
            raise ValueError("--out-dir is only valid with --top-dir")
        source = Path(args.input).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Input file does not exist: {source}")
        _validate_suffix(source, "--input")
        output = (
            Path(args.output).expanduser().resolve()
            if args.output is not None
            else source.with_name(f"{source.stem}_preprocessed.hdf5")
        )
        _validate_suffix(output, "--output")
        return "single file", source.parent, [(source, output)]

    if args.output is not None:
        raise ValueError("--output is only valid with --input")
    top_dir = Path(args.top_dir).expanduser().resolve()
    if not top_dir.is_dir():
        raise ValueError(f"Top directory does not exist: {top_dir}")
    output_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else top_dir.parent / f"{top_dir.name}_preprocessed"
    )
    sources = discover_inputs(top_dir, output_dir)
    mappings = [(source, output_dir / source.relative_to(top_dir)) for source in sources]
    return "directory", output_dir, mappings


def _object_address(obj: h5py.HLObject) -> int:
    return int(h5py.h5o.get_info(obj.id).addr)


def _dtype_contains_reference(dtype: np.dtype) -> bool:
    dtype = np.dtype(dtype)
    if h5py.check_dtype(ref=dtype) is not None:
        return True
    if dtype.fields:
        return any(_dtype_contains_reference(field[0]) for field in dtype.fields.values())
    if dtype.subdtype:
        return _dtype_contains_reference(dtype.subdtype[0])
    return False


def _references_in_value(value: object) -> Iterator[h5py.Reference]:
    if isinstance(value, h5py.Reference):
        yield value
        return
    if isinstance(value, np.void) and value.dtype.fields:
        for name in value.dtype.names or ():
            yield from _references_in_value(value[name])
        return
    if isinstance(value, np.ndarray):
        if not _dtype_contains_reference(value.dtype) and value.dtype.kind != "O":
            return
        for item in value.flat:
            yield from _references_in_value(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _references_in_value(item)


def _iter_dataset_batches(dataset: h5py.Dataset, target_bytes: int) -> Iterator[object]:
    if dataset.ndim == 0:
        yield dataset[()]
        return
    if dataset.shape[0] == 0:
        return
    bytes_per_first = max(1, int(dataset.dtype.itemsize) * int(np.prod(dataset.shape[1:])))
    batch = max(1, target_bytes // bytes_per_first)
    for start in range(0, int(dataset.shape[0]), batch):
        yield dataset[start : min(int(dataset.shape[0]), start + batch)]


def _reference_points_to_targets(
    h5: h5py.File,
    reference: h5py.Reference,
    target_addresses: set,
) -> bool:
    if not reference:
        return False
    try:
        return _object_address(h5[reference]) in target_addresses
    except (KeyError, RuntimeError, ValueError):
        return False


def reject_references_to_targets(h5: h5py.File, target_addresses: set) -> None:
    def check_attrs(obj: h5py.HLObject, path: str) -> None:
        for name, value in obj.attrs.items():
            for reference in _references_in_value(value):
                if _reference_points_to_targets(h5, reference, target_addresses):
                    raise ValueError(
                        f"HDF5 attribute {path or '/'}:{name} references a selected image dataset; "
                        "object and region references to recreated datasets are unsupported"
                    )

    check_attrs(h5, "")

    def visitor(name: str, obj: h5py.HLObject) -> None:
        path = f"/{name}"
        check_attrs(obj, path)
        if isinstance(obj, h5py.Dataset) and _dtype_contains_reference(obj.dtype):
            for values in _iter_dataset_batches(obj, REFERENCE_SCAN_BYTES):
                for reference in _references_in_value(np.asarray(values)):
                    if _reference_points_to_targets(h5, reference, target_addresses):
                        raise ValueError(
                            f"HDF5 dataset {path} contains a reference to a selected image dataset; "
                            "object and region references to recreated datasets are unsupported"
                        )

    h5.visititems(visitor)


def _expected_event_channels(root: h5py.File, dataset: h5py.Dataset) -> int:
    channels = int(dataset.shape[-1])
    if channels == 3:
        return 3
    representation = root.attrs.get("event_representation")
    if isinstance(representation, bytes):
        representation = representation.decode("utf-8")
    if representation != "xyt_signed_voxel_v1":
        raise ValueError(
            f"Rank-4 event data must have C=3 unless event_representation is "
            f"'xyt_signed_voxel_v1'; got C={channels}"
        )
    required = ("event_temporal_bins", "image_channels", "channels_per_visual_frame")
    missing = [name for name in required if name not in root.attrs]
    if missing:
        raise ValueError(f"XYT event dataset is missing channel metadata attributes: {missing}")
    declared = {name: int(root.attrs[name]) for name in required}
    if any(value != channels for value in declared.values()):
        raise ValueError(
            f"XYT event channel count C={channels} disagrees with metadata {declared}"
        )
    return channels


def validate_dataset(
    root: h5py.File,
    source: Path,
    modality: str,
    mask_x: Optional[Tuple[int, int]],
    crop_x: int,
    crop_square: Optional[int],
) -> DatasetPlan:
    key = DATASET_KEYS[modality]
    if key not in root:
        raise ValueError(f"Missing required dataset {key} in {source}")
    link = root.get(key, getlink=True)
    if not isinstance(link, h5py.HardLink):
        raise ValueError(f"Selected key {key} in {source} must be a regular hard-linked dataset")
    dataset = root[key]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Selected key {key} in {source} is not a dataset")
    if dataset.is_virtual or dataset.external:
        raise ValueError(f"Virtual or external selected dataset {key} in {source} is unsupported")
    if int(h5py.h5o.get_info(dataset.id).rc) != 1:
        raise ValueError(f"Selected dataset {key} in {source} has multiple hard links")
    if dataset.dtype != np.dtype(np.uint8):
        raise ValueError(f"Selected dataset {key} in {source} must be uint8, got {dataset.dtype}")
    if dataset.ndim not in (3, 4):
        raise ValueError(
            f"Selected dataset {key} in {source} must have shape [T,H,W] or [T,H,W,C], "
            f"got {dataset.shape}"
        )
    if any(int(size) <= 0 for size in dataset.shape):
        raise ValueError(f"Selected dataset {key} in {source} has an empty dimension: {dataset.shape}")
    if dataset.ndim == 4:
        if modality == "rgb" and int(dataset.shape[-1]) != 3:
            raise ValueError(f"Rank-4 RGB dataset {key} must have C=3, got {dataset.shape}")
        if modality == "event":
            _expected_event_channels(root, dataset)
    for attr_name in dataset.attrs:
        if str(attr_name).startswith(METADATA_PREFIX):
            raise ValueError(
                f"Dataset {key} in {source} was already processed by this tool; reprocessing is unsupported"
            )

    stored_height, stored_width = int(dataset.shape[1]), int(dataset.shape[2])
    if modality == "event":
        # Event transform coordinates are defined in the 90-degree-CCW debug view.
        transform_height, transform_width = stored_width, stored_height
    else:
        transform_height, transform_width = stored_height, stored_width
    result_shape = list(dataset.shape)
    if crop_square is not None:
        if modality != "event":
            raise ValueError("--crop-square is supported only for event datasets")
        if transform_height != transform_width:
            raise ValueError(
                f"--crop-square requires a square event image, got "
                f"{(stored_height, stored_width)} in {source}"
            )
        if not 0 < crop_square <= transform_width:
            raise ValueError(
                f"--crop-square must satisfy 0 < N <= image side {transform_width}, "
                f"got {crop_square} in {source}"
            )
        if crop_square == transform_width and mask_x is None:
            raise ValueError(
                f"--crop-square {crop_square} is a no-op for {source}; add a mask or "
                "choose a smaller square"
            )
        validate_image_transform(
            transform_height, transform_width, mask_x=mask_x, crop_x=0
        )
        result_shape[1] = crop_square
        result_shape[2] = crop_square
    else:
        validate_image_transform(
            transform_height, transform_width, mask_x=mask_x, crop_x=crop_x
        )
        crop_axis = 1 if modality == "event" else 2
        result_shape[crop_axis] = int(result_shape[crop_axis]) - crop_x
    frame_count = int(dataset.shape[0])
    logical_input = int(dataset.size) * int(dataset.dtype.itemsize)
    logical_output = frame_count * int(np.prod(result_shape[1:])) * int(dataset.dtype.itemsize)
    return DatasetPlan(
        modality=modality,
        key=key,
        original_shape=tuple(int(value) for value in dataset.shape),
        result_shape=tuple(int(value) for value in result_shape),
        dtype=np.dtype(dataset.dtype),
        logical_input_bytes=logical_input,
        logical_output_bytes=logical_output,
    )


def build_file_plans(
    mappings: Sequence[Tuple[Path, Path]],
    modalities: Sequence[str],
    mask_x: Optional[Tuple[int, int]],
    crop_x: int,
    overwrite: bool,
    crop_square: Optional[int] = None,
) -> List[FilePlan]:
    plans = []
    seen_outputs = set()
    all_sources = {source.resolve() for source, _ in mappings}
    for source, output in mappings:
        source = source.resolve()
        output = output.resolve()
        if source == output:
            raise ValueError(f"Output must differ from source: {source}")
        if output in all_sources:
            raise ValueError(f"Output path would overwrite a discovered input file: {output}")
        if output in seen_outputs:
            raise ValueError(f"Multiple inputs map to the same output: {output}")
        seen_outputs.add(output)
        if output.exists() and not overwrite:
            raise ValueError(
                f"Output already exists: {output}. Pass --overwrite-outputs to replace it."
            )
        with h5py.File(source, "r") as root:
            datasets = tuple(
                validate_dataset(root, source, modality, mask_x, crop_x, crop_square)
                for modality in modalities
            )
            target_addresses = {_object_address(root[item.key]) for item in datasets}
            reject_references_to_targets(root, target_addresses)
        plans.append(
            FilePlan(
                source=source,
                output=output,
                datasets=datasets,
                source_bytes=int(source.stat().st_size),
            )
        )
    return plans


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise ValueError(f"Could not find an existing parent for output path {path}")
        current = current.parent
    return current


def print_summary(
    input_mode: str,
    plans: Sequence[FilePlan],
    modality: str,
    mask_x: Optional[Tuple[int, int]],
    crop_x: int,
    crop_square: Optional[int],
    output_root: Path,
) -> None:
    source_bytes = sum(plan.source_bytes for plan in plans)
    logical_input = sum(item.logical_input_bytes for plan in plans for item in plan.datasets)
    logical_output = sum(item.logical_output_bytes for plan in plans for item in plan.datasets)
    conservative_outputs = sum(
        plan.source_bytes + sum(item.logical_output_bytes for item in plan.datasets)
        for plan in plans
    )
    disk = shutil.disk_usage(_existing_ancestor(output_root))
    unique_shapes: Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], int] = {}
    for plan in plans:
        for item in plan.datasets:
            marker = (item.modality, item.original_shape, item.result_shape)
            unique_shapes[marker] = unique_shapes.get(marker, 0) + 1

    print("\n=== HDF5 image preprocessing summary ===")
    print(f"Input mode: {input_mode}")
    print(f"File count: {len(plans)}")
    print(f"Selected modality: {modality}")
    print("Dataset keys: " + ", ".join(DATASET_KEYS[name] for name in _selected_modalities(modality)))
    print(f"Mask XTOP/XBOTTOM: {mask_x if mask_x is not None else 'none'}")
    print(f"Left crop: {crop_x} px")
    print(f"Centered event square crop: {crop_square if crop_square is not None else 'none'}")
    print(
        "Coordinate frame: RGB uses stored image coordinates; event uses the 90-degree-CCW "
        "debug-view coordinates"
    )
    print("RGB operation order: mask, then left crop in stored orientation")
    if crop_square is not None:
        print(
            "Event operation order: rotate 90 degrees CCW, mask, centered square crop, "
            "then rotate 90 degrees CW back to stored orientation"
        )
    else:
        print(
            "Event operation order: rotate 90 degrees CCW, mask, left crop, then rotate "
            "90 degrees CW back to stored orientation"
        )
    print("Fill values: RGB=0, event=128")
    print("Shapes:")
    for (name, original, result), count in sorted(unique_shapes.items()):
        print(f"  {name}: {original} -> {result} ({count} dataset(s))")
    print(f"Source files on disk: {_human_bytes(source_bytes)}")
    print(
        "Selected image logical size: "
        f"{_human_bytes(logical_input)} -> {_human_bytes(logical_output)}"
    )
    print(
        "Conservative output/temp budget: "
        f"{_human_bytes(conservative_outputs)} "
        "(full source copies plus possibly newly allocated transformed datasets)"
    )
    print(
        "Note: final files can remain near their original size after cropping because HDF5 may "
        "retain deleted dataset space internally."
    )
    print(f"Available disk space: {_human_bytes(int(disk.free))} at {_existing_ancestor(output_root)}")
    if int(disk.free) < conservative_outputs:
        print("WARNING: available space is below the conservative output/temp estimate.")
    print("Output paths:")
    for plan in plans:
        print(f"  {plan.source} -> {plan.output}")
    print(
        "Ball-pixel containment is not automatically verifiable: interception HDF5 files do "
        "not store /ball_tracker2/ball_2d_px. Compare the proposed boundaries against that "
        "topic in the source recording before processing the full dataset."
    )


def representative_file_indices(file_count: int, preview_files: int) -> List[int]:
    if file_count <= 0:
        return []
    count = min(int(preview_files), int(file_count))
    if count == 1:
        return [0]
    raw = np.linspace(0, file_count - 1, num=count)
    return sorted({int(round(value)) for value in raw} | {0, file_count - 1})


def _highest_left_half_event_activity_frames(
    dataset: h5py.Dataset,
    count: int = 16,
) -> List[Tuple[int, int]]:
    """Return ``(frame_index, activity)`` ranked by rotated left-half activity."""
    frame_bytes = max(1, int(np.prod(dataset.shape[1:])) * int(dataset.dtype.itemsize))
    batch_size = max(1, TARGET_BATCH_BYTES // frame_bytes)
    best: List[Tuple[int, int]] = []
    for start in range(0, int(dataset.shape[0]), batch_size):
        frames = dataset[start : min(int(dataset.shape[0]), start + batch_size)]
        rotated = np.rot90(frames, k=1, axes=(1, 2))
        left_width = max(1, int(rotated.shape[2]) // 2)
        left_half = rotated[:, :, :left_width, ...]
        activity = (
            np.abs(left_half.astype(np.int16) - 128)
            .reshape(left_half.shape[0], -1)
            .sum(axis=1, dtype=np.int64)
        )
        for local_index, value in enumerate(activity.tolist()):
            frame_index = start + local_index
            candidate = (int(value), -frame_index)
            if len(best) < count:
                heapq.heappush(best, candidate)
            elif candidate > best[0]:
                heapq.heapreplace(best, candidate)
    ranked = sorted(best, reverse=True)
    return [(-negative_index, activity) for activity, negative_index in ranked]


def _transform_frame(
    frame: np.ndarray,
    modality: str,
    mask_x: Optional[Tuple[int, int]],
    crop_x: int,
    crop_square: Optional[int],
) -> np.ndarray:
    if crop_square is not None:
        if modality != "event":
            raise ValueError("Centered square crop is supported only for event images")
        return mask_and_center_crop_square_rotated_event_image(
            frame,
            mask_x=mask_x,
            square_side=crop_square,
            fill_value=FILL_VALUES[modality],
        )
    if modality == "event":
        return mask_and_left_crop_rotated_event_image(
            frame,
            mask_x=mask_x,
            crop_x=crop_x,
            fill_value=FILL_VALUES[modality],
        )
    return mask_and_left_crop_image(
        frame,
        mask_x=mask_x,
        crop_x=crop_x,
        fill_value=FILL_VALUES[modality],
    )


def _displayable(frame: np.ndarray, modality: str) -> Tuple[np.ndarray, dict]:
    if frame.ndim == 2:
        return frame, {"cmap": "gray", "vmin": 0, "vmax": 255}
    if frame.shape[2] == 1:
        return frame[..., 0], {"cmap": "gray", "vmin": 0, "vmax": 255}
    if frame.shape[2] == 3:
        return frame, {}
    # XYT channels are temporal planes, not RGB. Show the most active channel.
    activity = np.abs(frame.astype(np.int16) - 128).reshape(-1, frame.shape[2]).sum(axis=0)
    channel = int(np.argmax(activity))
    return frame[..., channel], {"cmap": "gray", "vmin": 0, "vmax": 255}


def preview_plans(
    plans: Sequence[FilePlan],
    preview_files: int,
    mask_x: Optional[Tuple[int, int]],
    crop_x: int,
    crop_square: Optional[int] = None,
    mask_line_width: float = 0.1,
) -> Path:
    indices = representative_file_indices(len(plans), preview_files)
    session_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    session_dir = SIMULATION_DIR / f"{session_name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    session_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving representative crop simulations to: {session_dir}")
    for index in indices:
        print(f"  [{index + 1}/{len(plans)}] {plans[index].source}")
    if len(plans) > 1:
        print("The validated transform will be applied to every discovered file.")

    for plan_index in indices:
        plan = plans[plan_index]
        with h5py.File(plan.source, "r") as root:
            for dataset_plan in plan.datasets:
                dataset = root[dataset_plan.key]
                if dataset_plan.modality == "event":
                    ranked = _highest_left_half_event_activity_frames(dataset, count=16)
                    frame_specs = [
                        (frame_index, f"activity rank {rank} (score={activity})")
                        for rank, (frame_index, activity) in enumerate(ranked, start=1)
                    ]
                else:
                    frame_count = int(dataset.shape[0])
                    last = frame_count - 1
                    frame_specs = [(0, "first"), (frame_count // 2, "middle"), (last, "last")]
                rows = len(frame_specs)
                figure, axes = plt.subplots(rows, 2, figsize=(12, max(3, 3 * rows)), squeeze=False)
                figure.suptitle(f"{plan.source}\n{dataset_plan.modality}: {dataset_plan.key}")
                for row, (frame_index, label) in enumerate(frame_specs):
                    original = np.asarray(dataset[frame_index])
                    transformed = _transform_frame(
                        original, dataset_plan.modality, mask_x, crop_x, crop_square
                    )
                    if dataset_plan.modality == "event":
                        preview_original = np.rot90(original, k=1, axes=(0, 1))
                        preview_transformed = np.rot90(transformed, k=1, axes=(0, 1))
                    else:
                        preview_original = original
                        preview_transformed = transformed
                    shown_original, original_kwargs = _displayable(
                        preview_original, dataset_plan.modality
                    )
                    shown_transformed, transformed_kwargs = _displayable(
                        preview_transformed, dataset_plan.modality
                    )
                    axes[row, 0].imshow(shown_original, **original_kwargs)
                    axes[row, 1].imshow(shown_transformed, **transformed_kwargs)
                    height, width = preview_original.shape[:2]
                    if mask_x is not None:
                        axes[row, 0].plot(
                            [mask_x[0], mask_x[1]],
                            [0, height - 1],
                            color="yellow",
                            linewidth=mask_line_width,
                            label="mask boundary",
                        )
                    if crop_x > 0:
                        axes[row, 0].axvline(
                            crop_x,
                            color="cyan",
                            linewidth=1.5,
                            label="crop boundary",
                        )
                    if crop_square is not None:
                        start = (width - crop_square) // 2
                        end = start + crop_square - 1
                        axes[row, 0].plot(
                            [start, end, end, start, start],
                            [start, start, end, end, start],
                            color="cyan",
                            linewidth=1.5,
                            label="square crop boundary",
                        )
                    if mask_x is not None or crop_x > 0 or crop_square is not None:
                        axes[row, 0].legend(loc="lower right", fontsize="small")
                    axes[row, 0].set_title(f"{label} #{frame_index} original {width}x{height}")
                    axes[row, 1].set_title(
                        f"{label} #{frame_index} transformed "
                        f"{preview_transformed.shape[1]}x{preview_transformed.shape[0]}"
                    )
                    for axis in axes[row]:
                        axis.axis("off")
                figure.tight_layout()
                safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", plan.source.stem)
                simulation_path = session_dir / (
                    f"file_{plan_index + 1:04d}_{safe_stem}_{dataset_plan.modality}.png"
                )
                figure.savefig(simulation_path, dpi=150, bbox_inches="tight")
                plt.close(figure)
                print(f"  saved: {simulation_path}")
    return session_dir


def _freeze_attribute_value(h5: h5py.File, value: object) -> object:
    references = list(_references_in_value(value))
    if references:
        frozen = []
        for reference in references:
            if not reference:
                frozen.append(("null-reference",))
            else:
                obj = h5[reference]
                marker = ["reference", obj.name]
                if isinstance(reference, h5py.RegionReference):
                    region = h5py.h5r.get_region(reference, h5.id)
                    marker.extend([region.get_select_type(), region.get_select_npoints()])
                frozen.append(tuple(marker))
        return tuple(frozen)
    array = np.asarray(value)
    if array.dtype.kind in ("O", "U", "S"):
        return (array.shape, tuple(repr(item) for item in array.reshape(-1).tolist()))
    return (array.dtype.str, array.shape, array.tobytes())


def _attrs_manifest(h5: h5py.File, obj: h5py.HLObject) -> Tuple[Tuple[str, object], ...]:
    return tuple(
        sorted((str(name), _freeze_attribute_value(h5, value)) for name, value in obj.attrs.items())
    )


def unrelated_manifest(h5: h5py.File, selected_keys: set) -> Dict[str, object]:
    manifest: Dict[str, object] = {"/": ("group", _attrs_manifest(h5, h5))}

    def walk(group: h5py.Group, prefix: str, visited_groups: set) -> None:
        for name in sorted(group.keys()):
            path = f"{prefix}/{name}" if prefix else f"/{name}"
            if path in selected_keys:
                continue
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.SoftLink):
                manifest[path] = ("soft-link", link.path)
                continue
            if isinstance(link, h5py.ExternalLink):
                manifest[path] = ("external-link", link.filename, link.path)
                continue
            obj = group[name]
            if isinstance(obj, h5py.Dataset):
                manifest[path] = (
                    "dataset",
                    tuple(int(value) for value in obj.shape),
                    obj.dtype.str,
                    _attrs_manifest(h5, obj),
                )
            elif isinstance(obj, h5py.Group):
                manifest[path] = ("group", _attrs_manifest(h5, obj))
                address = _object_address(obj)
                if address not in visited_groups:
                    visited_groups.add(address)
                    walk(obj, path, visited_groups)
            else:
                manifest[path] = (type(obj).__name__, _attrs_manifest(h5, obj))

    walk(h5, "", {_object_address(h5)})
    return manifest


def _creation_kwargs(
    dataset: h5py.Dataset,
    result_shape: Tuple[int, ...],
) -> dict:
    kwargs = {}
    if dataset.chunks is not None:
        kwargs["chunks"] = tuple(
            min(int(chunk), int(size)) for chunk, size in zip(dataset.chunks, result_shape)
        )
    if dataset.compression is not None:
        kwargs["compression"] = dataset.compression
        kwargs["compression_opts"] = dataset.compression_opts
    if dataset.shuffle:
        kwargs["shuffle"] = True
    if dataset.fletcher32:
        kwargs["fletcher32"] = True
    if dataset.scaleoffset is not None:
        kwargs["scaleoffset"] = dataset.scaleoffset
    if dataset.fillvalue is not None:
        kwargs["fillvalue"] = dataset.fillvalue
    if dataset.chunks is not None and dataset.maxshape is not None:
        maxshape = list(dataset.maxshape)
        for axis, (old_size, new_size) in enumerate(zip(dataset.shape, result_shape)):
            if maxshape[axis] is not None and int(new_size) != int(old_size):
                reduction = int(old_size) - int(new_size)
                maxshape[axis] = max(int(new_size), int(maxshape[axis]) - reduction)
        kwargs["maxshape"] = tuple(maxshape)
    return kwargs


def _creation_signature(dataset: h5py.Dataset) -> dict:
    return {
        "chunks": dataset.chunks,
        "compression": dataset.compression,
        "compression_opts": dataset.compression_opts,
        "shuffle": bool(dataset.shuffle),
        "fletcher32": bool(dataset.fletcher32),
        "scaleoffset": dataset.scaleoffset,
        "fillvalue": dataset.fillvalue,
        "maxshape": dataset.maxshape,
    }


def _sample_indices(frame_count: int) -> List[int]:
    return sorted({0, frame_count // 2, frame_count - 1})


def transform_temporary_file(
    temporary: Path,
    plan: FilePlan,
    mask_x: Optional[Tuple[int, int]],
    crop_x: int,
    crop_square: Optional[int],
) -> None:
    with h5py.File(temporary, "r+") as root:
        for item in plan.datasets:
            old = root[item.key]
            attributes = [(name, old.attrs[name], old.attrs.get_id(name).dtype) for name in old.attrs]
            kwargs = _creation_kwargs(old, item.result_shape)
            parent_path, dataset_name = item.key.rsplit("/", 1)
            parent = root[parent_path or "/"]
            del parent[dataset_name]
            new = parent.create_dataset(dataset_name, shape=item.result_shape, dtype=item.dtype, **kwargs)
            for name, value, dtype in attributes:
                new.attrs.create(name, value, dtype=dtype)
            frame_bytes = max(1, int(np.prod(item.original_shape[1:])) * item.dtype.itemsize)
            batch_size = max(1, TARGET_BATCH_BYTES // frame_bytes)
            with h5py.File(plan.source, "r") as source_root:
                source_dataset = source_root[item.key]
                for start in range(0, item.original_shape[0], batch_size):
                    end = min(item.original_shape[0], start + batch_size)
                    source_frames = source_dataset[start:end]
                    transformed = np.stack(
                        [
                            _transform_frame(
                                frame, item.modality, mask_x, crop_x, crop_square
                            )
                            for frame in source_frames
                        ],
                        axis=0,
                    )
                    new[start:end] = transformed
            new.attrs[f"{METADATA_PREFIX}original_shape"] = np.asarray(
                item.original_shape, dtype=np.int64
            )
            new.attrs[f"{METADATA_PREFIX}mask_x"] = (
                "none" if mask_x is None else f"{mask_x[0]},{mask_x[1]}"
            )
            new.attrs[f"{METADATA_PREFIX}left_crop_px"] = int(crop_x)
            new.attrs[f"{METADATA_PREFIX}center_square_side_px"] = (
                "none" if crop_square is None else int(crop_square)
            )
            new.attrs[f"{METADATA_PREFIX}fill_value"] = int(FILL_VALUES[item.modality])
            new.attrs[f"{METADATA_PREFIX}operation_order"] = (
                "rotate_90ccw_then_mask_then_center_square_crop_then_rotate_90cw"
                if crop_square is not None
                else (
                    "rotate_90ccw_then_mask_then_left_crop_then_rotate_90cw"
                    if item.modality == "event"
                    else "mask_then_left_crop"
                )
            )
            new.attrs[f"{METADATA_PREFIX}transform_coordinate_frame"] = (
                "rotated_90deg_ccw_debug_view"
                if item.modality == "event"
                else "stored_image_top_left_origin"
            )
            new.attrs[f"{METADATA_PREFIX}stored_orientation_mapping"] = (
                (
                    "rotate_90deg_ccw_then_mask_and_center_square_crop_then_rotate_90deg_cw"
                    if crop_square is not None
                    else "rotate_90deg_ccw_then_mask_and_left_crop_then_rotate_90deg_cw"
                )
                if item.modality == "event"
                else "identity"
            )
            new.attrs[f"{METADATA_PREFIX}tool_version"] = TOOL_VERSION
        root.flush()


def verify_temporary_file(
    temporary: Path,
    plan: FilePlan,
    source_manifest: Dict[str, object],
    mask_x: Optional[Tuple[int, int]],
    crop_x: int,
    crop_square: Optional[int],
) -> None:
    selected_keys = {item.key for item in plan.datasets}
    with h5py.File(plan.source, "r") as source, h5py.File(temporary, "r") as output:
        if unrelated_manifest(output, selected_keys) != source_manifest:
            raise RuntimeError("Unrelated HDF5 object/link/attribute manifest changed")
        for item in plan.datasets:
            if item.key not in output:
                raise RuntimeError(f"Verified output is missing {item.key}")
            source_dataset = source[item.key]
            output_dataset = output[item.key]
            if tuple(output_dataset.shape) != item.result_shape:
                raise RuntimeError(
                    f"Output shape mismatch for {item.key}: {output_dataset.shape} != {item.result_shape}"
                )
            if output_dataset.dtype != source_dataset.dtype:
                raise RuntimeError(
                    f"Output dtype mismatch for {item.key}: {output_dataset.dtype} != {source_dataset.dtype}"
                )
            if int(output_dataset.shape[0]) != int(source_dataset.shape[0]):
                raise RuntimeError(f"Frame count changed for {item.key}")
            expected_creation = _creation_kwargs(source_dataset, item.result_shape)
            expected_signature = {
                "chunks": expected_creation.get("chunks"),
                "compression": expected_creation.get("compression"),
                "compression_opts": expected_creation.get("compression_opts"),
                "shuffle": bool(expected_creation.get("shuffle", False)),
                "fletcher32": bool(expected_creation.get("fletcher32", False)),
                "scaleoffset": expected_creation.get("scaleoffset"),
                "fillvalue": expected_creation.get("fillvalue", output_dataset.fillvalue),
                "maxshape": expected_creation.get("maxshape", item.result_shape),
            }
            if _creation_signature(output_dataset) != expected_signature:
                raise RuntimeError(
                    f"Dataset creation properties changed unexpectedly for {item.key}: "
                    f"{_creation_signature(output_dataset)} != {expected_signature}"
                )
            for name, value in source_dataset.attrs.items():
                if name not in output_dataset.attrs or _freeze_attribute_value(
                    source, value
                ) != _freeze_attribute_value(output, output_dataset.attrs[name]):
                    raise RuntimeError(f"Dataset attribute {item.key}:{name} was not preserved")
            for frame_index in _sample_indices(int(source_dataset.shape[0])):
                expected = _transform_frame(
                    source_dataset[frame_index],
                    item.modality,
                    mask_x,
                    crop_x,
                    crop_square,
                )
                if not np.array_equal(output_dataset[frame_index], expected):
                    raise RuntimeError(
                        f"Sample verification failed for {item.key} frame {frame_index}"
                    )
            if output_dataset.attrs.get(f"{METADATA_PREFIX}tool_version") != TOOL_VERSION:
                raise RuntimeError(f"Missing preprocessing metadata on {item.key}")


def process_file(
    plan: FilePlan,
    mask_x: Optional[Tuple[int, int]],
    crop_x: int,
    crop_square: Optional[int] = None,
) -> None:
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan.output.parent / f".{plan.output.name}.{uuid.uuid4().hex}.tmp"
    selected_keys = {item.key for item in plan.datasets}
    with h5py.File(plan.source, "r") as source:
        source_manifest = unrelated_manifest(source, selected_keys)
    try:
        shutil.copy2(plan.source, temporary)
        transform_temporary_file(temporary, plan, mask_x, crop_x, crop_square)
        verify_temporary_file(
            temporary, plan, source_manifest, mask_x, crop_x, crop_square
        )
        os.replace(temporary, plan.output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def process_all(
    plans: Sequence[FilePlan],
    mask_x: Optional[Tuple[int, int]],
    crop_x: int,
    crop_square: Optional[int] = None,
) -> None:
    completed: List[Path] = []
    failed: Optional[Tuple[Path, Exception]] = None
    for plan in plans:
        print(f"[WRITE] {plan.source} -> {plan.output}")
        try:
            process_file(plan, mask_x, crop_x, crop_square)
            completed.append(plan.output)
            print(f"[OK] {plan.output}")
        except Exception as exc:  # final report must include partial directory progress
            failed = (plan.output, exc)
            break

    print("\n=== Final processing summary ===")
    print(f"Completed: {len(completed)}/{len(plans)}")
    for path in completed:
        print(f"  completed: {path}")
    if failed is not None:
        failed_path, error = failed
        print(f"  failed: {failed_path}: {error}")
        attempted = len(completed) + 1
        for plan in plans[attempted:]:
            print(f"  not attempted: {plan.output}")
        raise RuntimeError(f"Processing failed for {failed_path}: {error}") from error
    print("Status: success")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    examples = """examples:
  python3 helpers/preprocess_hdf5_images.py --input episode_0.hdf5 --modality event --mask-x 220 80 --crop-x 60
  python3 helpers/preprocess_hdf5_images.py --input episode_0.hdf5 --modality event --mask-x 220 80 --crop-square 256
  python3 helpers/preprocess_hdf5_images.py --top-dir dataset --out-dir dataset_cropped --modality rgb --mask-x 260 110 --crop-x 80
  python3 helpers/preprocess_hdf5_images.py --input episode_0.hdf5 --modality both --mask-x 220 80 --crop-x 60 --preview-only
"""
    parser = argparse.ArgumentParser(
        description="Safely mask and crop interception HDF5 image datasets.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="One .h5 or .hdf5 input file")
    inputs.add_argument("--top-dir", help="Recursively process .h5/.hdf5 files")
    parser.add_argument("--modality", required=True, choices=("rgb", "event", "both"))
    parser.add_argument("--mask-x", nargs=2, type=int, metavar=("XTOP", "XBOTTOM"))
    crops = parser.add_mutually_exclusive_group()
    crops.add_argument("--crop-x", type=int, default=None, metavar="L")
    crops.add_argument(
        "--crop-square",
        type=int,
        default=None,
        metavar="N_PIX",
        help="Event only: keep a concentric N_PIX x N_PIX square",
    )
    parser.add_argument("--output", help="Output file for --input mode")
    parser.add_argument("--out-dir", help="Output root for --top-dir mode")
    parser.add_argument("--overwrite-outputs", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip terminal confirmation")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not save crop simulation images",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Validate, summarize, and save simulations without writing HDF5 outputs",
    )
    parser.add_argument(
        "--preview-files",
        type=int,
        default=3,
        metavar="N",
        help="Directory files to preview, evenly spaced including endpoints (default: 3)",
    )
    parser.add_argument(
        "--mask-line-width",
        type=float,
        default=0.1,
        metavar="WIDTH",
        help="Yellow mask-boundary line width in saved simulations (default: 0.1)",
    )
    args = parser.parse_args(argv)
    if args.preview_files <= 0:
        parser.error("--preview-files must be positive")
    if not math.isfinite(args.mask_line_width) or args.mask_line_width <= 0.0:
        parser.error("--mask-line-width must be positive and finite")
    if args.mask_x is None and args.crop_square is None and (
        args.crop_x is None or args.crop_x == 0
    ):
        parser.error(
            "At least one effective operation is required; --crop-x 0 alone is a no-op"
        )
    if args.crop_x is not None and args.crop_x < 0:
        parser.error("--crop-x must be non-negative")
    if args.crop_square is not None and args.crop_square <= 0:
        parser.error("--crop-square must be positive")
    if args.crop_square is not None and args.modality != "event":
        parser.error("--crop-square requires --modality event")
    args.crop_x = 0 if args.crop_x is None else int(args.crop_x)
    args.mask_x = None if args.mask_x is None else tuple(int(value) for value in args.mask_x)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_mode, output_root, mappings = resolve_paths(args)
    plans = build_file_plans(
        mappings,
        _selected_modalities(args.modality),
        args.mask_x,
        args.crop_x,
        args.overwrite_outputs,
        crop_square=args.crop_square,
    )
    print_summary(
        input_mode,
        plans,
        args.modality,
        args.mask_x,
        args.crop_x,
        args.crop_square,
        output_root,
    )

    if not args.no_preview:
        preview_plans(
            plans,
            args.preview_files,
            args.mask_x,
            args.crop_x,
            args.crop_square,
            args.mask_line_width,
        )

    if args.preview_only:
        print("Preview-only complete: no HDF5 outputs were written.")
        return 0
    if not args.yes:
        response = input("Type exactly 'yes' to write the output files: ")
        if response != "yes":
            print("Aborted: confirmation was not exactly 'yes'. No HDF5 outputs were written.")
            return 1

    process_all(plans, args.mask_x, args.crop_x, args.crop_square)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
