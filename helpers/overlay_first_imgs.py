#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np


DEFAULT_IMAGE_KEY = "observations/images/event"


def is_event_key(image_key: str) -> bool:
    """Return True if the selected key targets event frames."""
    normalized_key = image_key.rstrip("/")
    return normalized_key.endswith("event") or Path(normalized_key).name == "event"


def load_first_frame(hdf5_path: Path, image_key: str) -> np.ndarray:
    """Load the first frame from image_key in an HDF5 file."""
    with h5py.File(hdf5_path, "r") as h5_file:
        if image_key not in h5_file:
            raise KeyError(f"Missing dataset: {image_key}")
        dataset = h5_file[image_key]
        if dataset.shape[0] == 0:
            raise ValueError(f"Dataset is empty: {image_key}")
        frame = dataset[0]

    return np.asarray(frame)


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert an array to uint8 using safe event/RGB visualization rules."""
    if array.dtype == np.uint8:
        return array

    array_float = np.asarray(array, dtype=np.float32)
    if array_float.size == 0:
        raise ValueError("Cannot convert empty array to uint8")

    max_value = float(np.nanmax(array_float))
    if max_value <= 1.0:
        array_float = array_float * 255.0
    else:
        array_float = np.clip(array_float, 0.0, 255.0)

    return np.clip(array_float, 0.0, 255.0).astype(np.uint8)


def first_frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a first frame to RGB uint8, preserving existing RGB behavior."""
    image = to_uint8(frame)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return image


def first_event_frame_to_3chef(frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Convert first event frame to RGB composite and three grayscale channels."""
    if frame.ndim != 3:
        raise ValueError(f"Expected event tensor shape (H, W, 3) or (3, H, W), got {frame.shape}")

    if frame.shape[2] == 3:
        composite_rgb = frame
    elif frame.shape[0] == 3:
        composite_rgb = np.transpose(frame, (1, 2, 0))
    else:
        raise ValueError(f"Expected event tensor shape (H, W, 3) or (3, H, W), got {frame.shape}")

    composite_rgb_u8 = to_uint8(composite_rgb)
    channels = [composite_rgb_u8[..., idx] for idx in range(3)]
    return composite_rgb_u8, channels


def first_event_frame_to_channels(frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return an HWC event tensor and its ordered grayscale channels."""
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"Expected an HWC or CHW event tensor, got {arr.shape}")
    if arr.shape[-1] in (3, 9):
        hwc = arr
    elif arr.shape[0] in (3, 9):
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(
            f"Expected 3Chef (3 channels) or XYT (9 channels), got {arr.shape}"
        )
    hwc_u8 = to_uint8(hwc)
    return hwc_u8, [hwc_u8[..., idx] for idx in range(hwc_u8.shape[-1])]


def collect_hdf5_files(top_dirs: List[Path]) -> List[Path]:
    """Recursively collect HDF5 files under multiple top-level directories."""
    files: set[Path] = set()

    for top_dir in top_dirs:
        if not top_dir.exists() or not top_dir.is_dir():
            raise ValueError(f"Input path does not exist or is not a directory: {top_dir}")

        files.update(top_dir.rglob("*.hdf5"))
        files.update(top_dir.rglob("*.h5"))

    return sorted(files)


def dedup_name(name: str, seen: Dict[str, int]) -> str:
    """Deduplicate repeated names by appending _1, _2, ..."""
    count = seen.get(name, 0)
    seen[name] = count + 1
    if count == 0:
        return name
    return f"{name}_{count}"


def save_first_images(
    hdf5_files: List[Path],
    image_key: str,
    output_dir: Path,
) -> Tuple[int, int, int, int]:
    """Save first frames in RGB or event-3chef mode.

    Returns:
        (saved_rgb_images, saved_event_channel_images, saved_event_composites, skipped_files)
    """
    event_mode = is_event_key(image_key)
    seen_stems: Dict[str, int] = {}

    saved_rgb_images = 0
    saved_event_channel_images = 0
    saved_event_composites = 0
    skipped_files = 0

    if event_mode:
        mode_output_root = output_dir / "event_channels"
    else:
        mode_output_root = output_dir

    mode_output_root.mkdir(parents=True, exist_ok=True)

    for file_path in hdf5_files:
        try:
            frame = load_first_frame(file_path, image_key)
        except Exception as exc:
            print(f"Skipping file={file_path}, key={image_key}: {exc}")
            skipped_files += 1
            continue

        unique_stem = dedup_name(file_path.stem, seen_stems)

        if event_mode:
            try:
                event_hwc, channels = first_event_frame_to_channels(frame)
            except Exception as exc:
                shape = tuple(np.asarray(frame).shape)
                print(
                    f"Skipping file={file_path}, key={image_key}, shape={shape}: {exc}",
                )
                skipped_files += 1
                continue

            sample_output_dir = mode_output_root / unique_stem
            sample_output_dir.mkdir(parents=True, exist_ok=True)

            wrote_ok = True
            for idx, channel in enumerate(channels):
                channel_path = sample_output_dir / f"bin_{idx:02d}.png"
                if not cv2.imwrite(str(channel_path), channel):
                    print(
                        f"Skipping file={file_path}, key={image_key}: "
                        f"failed to write {channel_path}",
                    )
                    wrote_ok = False
                    break

            if not wrote_ok:
                skipped_files += 1
                continue

            if len(channels) == 3:
                composite_rgb = event_hwc
            else:
                rows = []
                for row_start in range(0, len(channels), 3):
                    row = channels[row_start:row_start + 3]
                    while len(row) < 3:
                        row.append(np.full_like(channels[0], 128))
                    rows.append(np.concatenate(row, axis=1))
                montage = np.concatenate(rows, axis=0)
                composite_rgb = cv2.cvtColor(montage, cv2.COLOR_GRAY2RGB)
            composite_path = sample_output_dir / "ordered_channels.png"
            composite_bgr = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(composite_path), composite_bgr):
                print(
                    f"Skipping file={file_path}, key={image_key}: "
                    f"failed to write {composite_path}",
                )
                skipped_files += 1
                continue

            saved_event_channel_images += len(channels)
            saved_event_composites += 1
            continue

        try:
            rgb_image = first_frame_to_rgb(frame)
        except Exception as exc:
            shape = tuple(np.asarray(frame).shape)
            print(f"Skipping file={file_path}, key={image_key}, shape={shape}: {exc}")
            skipped_files += 1
            continue

        output_path = mode_output_root / f"{unique_stem}_first_image.png"
        image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), image_bgr):
            print(f"Skipping file={file_path}, key={image_key}: failed to write {output_path}")
            skipped_files += 1
            continue

        saved_rgb_images += 1

    return saved_rgb_images, saved_event_channel_images, saved_event_composites, skipped_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save first frames from HDF5 files (RGB, 3Chef, or XYT event mode).",
    )

    parser.add_argument(
        "top_dirs",
        type=Path,
        nargs="+",
        help="One or more top-level directories containing HDF5 files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("first_imgs"),
        help="Output directory root. Default: first_imgs",
    )

    parser.add_argument(
        "--image-key",
        type=str,
        default=DEFAULT_IMAGE_KEY,
        help=f"HDF5 image dataset key. Default: {DEFAULT_IMAGE_KEY}",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Deprecated compatibility flag (ignored).",
    )

    parser.add_argument(
        "--white-threshold",
        type=int,
        default=245,
        help="Deprecated compatibility flag (ignored).",
    )

    args = parser.parse_args()

    if args.gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {args.gamma}")

    if not 0 <= args.white_threshold <= 255:
        raise ValueError(f"white-threshold must be in [0, 255], got {args.white_threshold}")

    hdf5_files = collect_hdf5_files(args.top_dirs)

    if not hdf5_files:
        raise RuntimeError("No .hdf5/.h5 files found in provided folders.")

    args.output.mkdir(parents=True, exist_ok=True)

    saved_rgb_images, saved_event_channel_images, saved_event_composites, skipped_files = save_first_images(
        hdf5_files=hdf5_files,
        image_key=args.image_key,
        output_dir=args.output,
    )

    print(f"Input folders: {len(args.top_dirs)}")
    print(f"Found HDF5 files: {len(hdf5_files)}")
    print(f"Event mode: {is_event_key(args.image_key)}")
    print(f"Saved RGB images: {saved_rgb_images}")
    print(f"Saved event channel images: {saved_event_channel_images}")
    print(f"Saved event composite RGB images: {saved_event_composites}")
    print(f"Skipped files: {skipped_files}")
    print(f"Output folder: {args.output}")


if __name__ == "__main__":
    main()
