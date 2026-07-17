#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List
import cv2
import h5py
import numpy as np


DEFAULT_IMAGE_KEY = "observations/images/event"


def load_frame_at_index(hdf5_path: Path, image_key: str, frame_index: int) -> np.ndarray:
    """Load one image frame by index from image_key in an HDF5 file."""
    with h5py.File(hdf5_path, "r") as h5_file:
        if image_key not in h5_file:
            raise KeyError(f"Missing dataset: {image_key}")
        dataset = h5_file[image_key]
        if dataset.shape[0] <= frame_index:
            raise IndexError(
                f"Frame index {frame_index} out of range for dataset length {dataset.shape[0]}",
            )
        image = dataset[frame_index]

    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return image


def collect_hdf5_files(top_dirs: List[Path]) -> List[Path]:
    """Recursively collect HDF5 files under multiple top-level directories."""
    files: set[Path] = set()

    for top_dir in top_dirs:
        if not top_dir.exists() or not top_dir.is_dir():
            raise ValueError(f"Input path does not exist or is not a directory: {top_dir}")

        files.update(top_dir.rglob("*.hdf5"))
        files.update(top_dir.rglob("*.h5"))

    return sorted(files)


def save_images_at_index(
    hdf5_files: List[Path],
    image_key: str,
    output_dir: Path,
    frame_index: int,
) -> int:
    """Save each file's selected frame as <hdf5_stem>_frame_<idx>.png."""
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    used_output_names: set[str] = set()

    for file_path in hdf5_files:
        try:
            image = load_frame_at_index(file_path, image_key, frame_index)
        except Exception as exc:
            print(f"Skipping {file_path}: {exc}")
            continue

        base_name = f"{file_path.stem}_frame_{frame_index}"
        out_name = f"{base_name}.png"

        # Keep the requested naming pattern while avoiding accidental overwrite
        # when multiple directories contain identical HDF5 file names.
        dedup_idx = 1
        while out_name in used_output_names or (output_dir / out_name).exists():
            out_name = f"{base_name}_{dedup_idx}.png"
            dedup_idx += 1

        used_output_names.add(out_name)
        output_path = output_dir / out_name

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), image_bgr):
            print(f"Failed to save: {output_path}")
            continue

        saved_count += 1

    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a selected frame from each HDF5 file under one or more folders.",
    )

    parser.add_argument(
        "top_dirs",
        type=Path,
        nargs="+",
        help="One or more top-level directories containing HDF5 files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "first_imgs",
        help="Output folder for saved first images. Default: sibling folder named first_imgs under act/.",
    )

    parser.add_argument(
        "--image-key",
        type=str,
        default=DEFAULT_IMAGE_KEY,
        help=f"HDF5 image dataset key. Default: {DEFAULT_IMAGE_KEY}",
    )

    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="0-based frame index to save from each dataset. Default: 0 (first frame).",
    )

    args = parser.parse_args()

    if args.frame_index < 0:
        raise ValueError(f"frame-index must be >= 0, got {args.frame_index}")

    hdf5_files = collect_hdf5_files(args.top_dirs)

    if not hdf5_files:
        raise RuntimeError("No .hdf5/.h5 files found in provided folders.")

    saved_count = save_images_at_index(
        hdf5_files=hdf5_files,
        image_key=args.image_key,
        output_dir=args.output_dir,
        frame_index=args.frame_index,
    )

    print(f"Input folders: {len(args.top_dirs)}")
    print(f"Found HDF5 files: {len(hdf5_files)}")
    print(f"Frame index: {args.frame_index}")
    print(f"Saved images: {saved_count}")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
