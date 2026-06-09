#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List,Tuple
import cv2
import h5py
import numpy as np


DEFAULT_IMAGE_KEY = "observations/images/rgb"


def load_first_image(hdf5_path: Path, image_key: str) -> np.ndarray:
    """Load the first image from image_key in an HDF5 file."""
    with h5py.File(hdf5_path, "r") as h5_file:
        if image_key not in h5_file:
            raise KeyError(f"Missing dataset: {image_key}")
        image = h5_file[image_key][0]

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


def overlay_images(
    hdf5_files: List[Path],
    image_key: str,
    gamma: float,
    white_threshold: int,
) -> Tuple[np.ndarray, int]:
    """Create one equal-weight overlay from all first images."""
    overlay_shape = None
    image_sum = None
    pixel_count = None
    loaded_count = 0

    for file_path in hdf5_files:
        try:
            image = load_first_image(file_path, image_key)
        except Exception as exc:
            print(f"Skipping {file_path}: {exc}")
            continue

        if overlay_shape is None:
            overlay_shape = image.shape
            image_sum = np.zeros(overlay_shape, dtype=np.float32)
            pixel_count = np.zeros(overlay_shape[:2], dtype=np.float32)

        if image.shape[:2] != overlay_shape[:2]:
            height, width = overlay_shape[:2]
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        # Convert to float
        img = image.astype(np.float32)

        # Basic non-white mask
        non_white_mask = np.any(img < white_threshold, axis=2)

        # --- NEW: color weighting ---
        # Tennis ball ≈ high R + high G, low B (yellow)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]

        yellow_score = (r + g) - 1.5 * b   # emphasize yellow, suppress blue
        yellow_score = np.clip(yellow_score, 0, None)

        # Normalize weights (avoid explosion)
        weight = 1.0 + 2.0 * (yellow_score / 255.0)  # tune 2.0 → stronger boost

        # Apply mask
        valid = non_white_mask

        image_sum[valid] += (img[valid] * weight[valid, None])
        pixel_count[valid] += weight[valid]

        loaded_count += 1

    if overlay_shape is None or loaded_count == 0:
        raise RuntimeError("No valid images were loaded.")

    overlay = np.full(overlay_shape, 255.0, dtype=np.float32)

    valid_pixels = pixel_count > 0
    overlay[valid_pixels] = image_sum[valid_pixels] / pixel_count[valid_pixels, None]

    overlay = 255.0 * np.power(
        np.clip(overlay / 255.0, 0.0, 1.0),
        gamma,
    )

    return np.clip(overlay, 0, 255).astype(np.uint8), loaded_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay first images from all HDF5 files under one or more folders.",
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
        default=Path("overlay_first_images.png"),
        help="Output image path.",
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
        help="Post-merge gamma correction. Use <1.0, e.g. 0.75, to reduce washout.",
    )

    parser.add_argument(
        "--white-threshold",
        type=int,
        default=245,
        help="Treat pixels as white background when all channels are >= this value.",
    )

    args = parser.parse_args()

    if args.gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {args.gamma}")

    if not 0 <= args.white_threshold <= 255:
        raise ValueError(f"white-threshold must be in [0, 255], got {args.white_threshold}")

    hdf5_files = collect_hdf5_files(args.top_dirs)

    if not hdf5_files:
        raise RuntimeError("No .hdf5/.h5 files found in provided folders.")

    overlay, loaded_count = overlay_images(
        hdf5_files=hdf5_files,
        image_key=args.image_key,
        gamma=args.gamma,
        white_threshold=args.white_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.rotate(overlay_bgr, cv2.ROTATE_180)

    if not cv2.imwrite(str(args.output), overlay_bgr):
        raise RuntimeError(f"Failed to write output image: {args.output}")

    print(f"Input folders: {len(args.top_dirs)}")
    print(f"Found HDF5 files: {len(hdf5_files)}")
    print(f"Successfully blended images: {loaded_count}")
    print(f"Saved overlay to: {args.output}")


if __name__ == "__main__":
    main()