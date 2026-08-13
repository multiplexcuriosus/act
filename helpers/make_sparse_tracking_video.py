#!/usr/bin/env python3
"""Create side-by-side RGB/event tracking videos from position-only HDF5 episodes.

The expected HDF5 layout is:

    observations/images/rgb
    observations/images/event
    observations/sparse_tracking/rgb_2d_px
    observations/sparse_tracking/rgb_valid
    observations/sparse_tracking/event_2d_px
    observations/sparse_tracking/event_valid

The RGB and event images are read frame-by-frame.  The corresponding 2-D
tracker estimate is drawn on each panel before the panels are concatenated.
The event panel also marks the configured tracker crop region without cropping
the image itself.

Examples:

    python3 make_sparse_tracking_video.py episode_0.hdf5
    python3 make_sparse_tracking_video.py --top-dir /path/to/episodes --num-videos 5

By default, output is written to an ``output`` directory next to the input
HDF5 file(s).  Thus, for ``/data/run/episode_0.hdf5``, the default output is
``/data/run/output/episode_0_event_rgb.mp4``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np


EVENT_IMAGE = "observations/images/event"
RGB_IMAGE = "observations/images/rgb"
TRACKING_GROUP = "observations/sparse_tracking"
DEFAULT_EVENT_TRACKER_CONFIG = Path(
    "/home/dyros/jg_ws/src/openmv_cam/config/offline_tracker_example.json"
)


def _load_event_crop_polygon(config_path: Path) -> Optional[np.ndarray]:
    """Load the event tracker's half-open crop boundary as an OpenCV polygon."""
    if not config_path.is_file():
        print(
            f"ERROR: event tracker config is absent: {config_path}; "
            "continuing without a crop-region border",
            file=sys.stderr,
        )
        return None

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    try:
        width = int(config["width"])
        height = int(config["height"])
        x_crop = tuple(int(value) for value in config["x_crop"])
        y_crop = tuple(int(value) for value in config["y_crop"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid event crop configuration in {config_path}: {error}"
        ) from error

    if len(x_crop) != 2:
        raise ValueError(f"x_crop in {config_path} must contain two values")
    if len(y_crop) != 4:
        raise ValueError(f"y_crop in {config_path} must contain four values")
    lower_x, upper_x = x_crop
    left_lower, left_upper, right_lower, right_upper = y_crop
    if not 0 <= lower_x < upper_x <= width:
        raise ValueError(
            f"x_crop in {config_path} must satisfy 0 <= lower < upper <= {width}"
        )
    if not (
        0 <= left_lower < left_upper <= height
        and 0 <= right_lower < right_upper <= height
    ):
        raise ValueError(
            f"y_crop endpoint pairs in {config_path} must satisfy "
            f"0 <= lower < upper <= {height}"
        )

    return np.asarray(
        [
            (lower_x, left_lower),
            (upper_x - 1, right_lower),
            (upper_x - 1, right_upper - 1),
            (lower_x, left_upper - 1),
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)


def _draw_event_crop_border(
    image: np.ndarray, crop_polygon: Optional[np.ndarray]
) -> None:
    """Mark, but do not remove, the event tracker crop region."""
    if crop_polygon is not None:
        cv2.polylines(image, [crop_polygon], True, (255, 0, 255), 2, cv2.LINE_AA)


def _dataset(h5_file: h5py.File, path: str) -> h5py.Dataset:
    """Return a required dataset with a useful error if it is absent."""
    if path not in h5_file:
        raise KeyError(f"required HDF5 dataset is missing: /{path}")
    value = h5_file[path]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"HDF5 object is not a dataset: /{path}")
    return value


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image array to uint8 without changing its spatial shape."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32, copy=False)
    if image.size and float(np.nanmax(image)) <= 1.0 and float(np.nanmin(image)) >= 0.0:
        image = image * 255.0
    return np.clip(np.nan_to_num(image, nan=0.0), 0.0, 255.0).astype(np.uint8)


def _as_bgr(image: np.ndarray, *, rgb_color_order: str, is_event: bool) -> np.ndarray:
    """Normalize an HDF5 image to a 3-channel OpenCV image.

    Event channels are pseudo-colour/time channels, not RGB channels, so their
    order is deliberately preserved.  RGB images can be marked as RGB or BGR
    via the CLI; OpenCV video output itself is BGR.
    """
    image = _to_uint8(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3]
    else:
        raise ValueError(f"unsupported image shape: {image.shape}")

    if not is_event and rgb_color_order == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(image)


def _resize_to_height(image: np.ndarray, height: int, *, is_event: bool) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"cannot resize image with shape {image.shape}")

    scale = float(height) / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    interpolation = cv2.INTER_NEAREST if is_event else (
        cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    )
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _rotate_position_ccw(
    position: Tuple[Optional[float], Optional[float], bool],
    *,
    native_width: int,
) -> Tuple[Optional[float], Optional[float], bool]:
    """Rotate an image-space position 90 degrees counterclockwise."""
    u, v, valid = position
    if u is None or v is None:
        return u, v, valid
    return v, native_width - 1 - u, valid


def _read_position(
    position_dataset: h5py.Dataset,
    valid_dataset: h5py.Dataset,
    index: int,
) -> Tuple[Optional[float], Optional[float], bool]:
    position = np.asarray(position_dataset[index]).reshape(-1)
    if position.size < 2:
        return None, None, False

    u, v = float(position[0]), float(position[1])
    valid = bool(np.asarray(valid_dataset[index]).reshape(-1)[0])
    valid = valid and np.isfinite(u) and np.isfinite(v)
    return (u, v, valid) if valid else (u, v, False)


def _put_text_box(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    *,
    font_scale: float = 0.55,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    background: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(font_scale * 2.0)))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    x, y = origin
    cv2.rectangle(
        image,
        (x - 4, y - text_height - baseline - 4),
        (x + text_width + 4, y + 4),
        background,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_tracker_overlay(
    image: np.ndarray,
    position: Tuple[Optional[float], Optional[float], bool],
    *,
    native_width: int,
    native_height: int,
    label: str,
) -> None:
    """Draw tracker position, validity, and panel label on a resized panel."""
    u, v, valid = position
    panel_height, panel_width = image.shape[:2]
    _put_text_box(image, label, (10, 28), font_scale=0.62)

    if u is None or v is None:
        _put_text_box(image, "tracker position unavailable", (10, panel_height - 12), font_scale=0.48)
        return

    coordinate_text = f"u={u:.1f}, v={v:.1f}"
    if not valid:
        _put_text_box(
            image,
            f"INVALID ({coordinate_text})",
            (10, panel_height - 12),
            font_scale=0.48,
            text_color=(180, 180, 180),
        )
        return

    x = int(round(u * panel_width / native_width))
    y = int(round(v * panel_height / native_height))
    radius = max(5, int(round(min(panel_width, panel_height) * 0.018)))

    # Clip only for drawing.  The text still reports the raw tracker value.
    draw_x = int(np.clip(x, 0, panel_width - 1))
    draw_y = int(np.clip(y, 0, panel_height - 1))
    cv2.drawMarker(
        image,
        (draw_x, draw_y),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=radius * 3,
        thickness=max(1, radius // 3),
        line_type=cv2.LINE_AA,
    )
    cv2.circle(image, (draw_x, draw_y), radius, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    _put_text_box(
        image,
        coordinate_text,
        (10, panel_height - 12),
        font_scale=0.48,
        text_color=(0, 255, 255),
    )


def _get_fps(h5_file: h5py.File, fps_override: Optional[float]) -> float:
    if fps_override is not None:
        if fps_override <= 0:
            raise ValueError("--fps must be positive")
        return fps_override

    fps = h5_file.attrs.get("fps", 30.0)
    if isinstance(fps, np.ndarray):
        fps = fps.reshape(-1)[0]
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = 30.0
    return fps if fps > 0 else 30.0


def create_video(
    hdf5_path: Path,
    output_path: Path,
    *,
    panel_height: int,
    fps_override: Optional[float],
    rgb_color_order: str,
    event_crop_polygon: Optional[np.ndarray] = None,
) -> int:
    """Create one video and return the number of written frames."""
    if panel_height <= 0:
        raise ValueError("--height must be positive")

    with h5py.File(hdf5_path, "r") as h5_file:
        rgb_images = _dataset(h5_file, RGB_IMAGE)
        event_images = _dataset(h5_file, EVENT_IMAGE)
        rgb_position = _dataset(h5_file, f"{TRACKING_GROUP}/rgb_2d_px")
        rgb_valid = _dataset(h5_file, f"{TRACKING_GROUP}/rgb_valid")
        event_position = _dataset(h5_file, f"{TRACKING_GROUP}/event_2d_px")
        event_valid = _dataset(h5_file, f"{TRACKING_GROUP}/event_valid")

        lengths = {
            "RGB images": rgb_images.shape[0],
            "event images": event_images.shape[0],
            "RGB tracking": rgb_position.shape[0],
            "event tracking": event_position.shape[0],
        }
        frame_count = min(lengths.values())
        if frame_count == 0:
            raise ValueError("the HDF5 file contains no frames")
        if len(set(lengths.values())) != 1:
            print(
                f"warning: frame counts differ in {hdf5_path.name}: {lengths}; "
                f"using the first {frame_count} synchronized frames",
                file=sys.stderr,
            )

        # Read one pair only to establish the fixed video dimensions.
        event0 = _as_bgr(
            event_images[0], rgb_color_order=rgb_color_order, is_event=True
        )
        _draw_event_crop_border(event0, event_crop_polygon)
        event0 = cv2.rotate(event0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb0 = _as_bgr(rgb_images[0], rgb_color_order=rgb_color_order, is_event=False)
        event_panel0 = _resize_to_height(event0, panel_height, is_event=True)
        rgb_panel0 = _resize_to_height(rgb0, panel_height, is_event=False)
        video_width = event_panel0.shape[1] + rgb_panel0.shape[1]
        video_height = panel_height
        fps = _get_fps(h5_file, fps_override)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (video_width, video_height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"could not open video writer for {output_path}; "
                "check that OpenCV has an MP4 codec available"
            )

        try:
            for index in range(frame_count):
                if index == 0:
                    event_panel = event_panel0.copy()
                    rgb_panel = rgb_panel0.copy()
                else:
                    event = _as_bgr(
                        event_images[index],
                        rgb_color_order=rgb_color_order,
                        is_event=True,
                    )
                    _draw_event_crop_border(event, event_crop_polygon)
                    event = cv2.rotate(event, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rgb = _as_bgr(
                        rgb_images[index],
                        rgb_color_order=rgb_color_order,
                        is_event=False,
                    )
                    event_panel = _resize_to_height(event, panel_height, is_event=True)
                    rgb_panel = _resize_to_height(rgb, panel_height, is_event=False)

                _draw_tracker_overlay(
                    event_panel,
                    _rotate_position_ccw(
                        _read_position(event_position, event_valid, index),
                        native_width=event_images.shape[2],
                    ),
                    native_width=event_images.shape[1],
                    native_height=event_images.shape[2],
                    label="EVENT | sparse tracker",
                )
                _draw_tracker_overlay(
                    rgb_panel,
                    _read_position(rgb_position, rgb_valid, index),
                    native_width=rgb_images.shape[2],
                    native_height=rgb_images.shape[1],
                    label="RGB | ball tracker",
                )

                combined = np.hstack((event_panel, rgb_panel))
                _put_text_box(
                    combined,
                    f"frame {index + 1}/{frame_count}   t={index / fps:.3f} s",
                    (10, video_height - 12),
                    font_scale=0.48,
                    background=(32, 32, 32),
                )
                writer.write(combined)
        finally:
            writer.release()

    return frame_count


def _episode_files(top_dir: Path) -> list[Path]:
    if not top_dir.is_dir():
        raise FileNotFoundError(f"--top-dir is not a directory: {top_dir}")
    files = sorted(
        [*top_dir.glob("*.hdf5"), *top_dir.glob("*.h5")],
        key=lambda path: path.name,
    )
    if not files:
        raise FileNotFoundError(f"no .hdf5 or .h5 files found directly in {top_dir}")
    return files


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        type=Path,
        help="one HDF5 episode file",
    )
    parser.add_argument(
        "--top-dir",
        type=Path,
        help="directory containing HDF5 episode files; process them in filename order",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        help="with --top-dir, process only the first M files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the default sibling directory named output",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="height of each side-by-side panel in pixels (default: 480)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="output video frame rate (default: 10)",
    )
    parser.add_argument(
        "--event-tracker-config",
        type=Path,
        default=DEFAULT_EVENT_TRACKER_CONFIG,
        help=(
            "JSON providing x_crop/y_crop for the event-image border "
            f"(default: {DEFAULT_EVENT_TRACKER_CONFIG})"
        ),
    )
    parser.add_argument(
        "--rgb-color-order",
        choices=("rgb", "bgr"),
        default="rgb",
        help="channel order stored in observations/images/rgb (default: rgb)",
    )
    args = parser.parse_args(argv)

    if (args.hdf5_path is None) == (args.top_dir is None):
        parser.error("provide exactly one of an HDF5 path or --top-dir")
    if args.num_videos is not None:
        if args.top_dir is None:
            parser.error("--num-videos requires --top-dir")
        if args.num_videos <= 0:
            parser.error("--num-videos must be positive")
    if args.height <= 0:
        parser.error("--height must be positive")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        event_crop_polygon = _load_event_crop_polygon(args.event_tracker_config)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(
            f"ERROR: {error}; continuing without a crop-region border",
            file=sys.stderr,
        )
        event_crop_polygon = None

    if args.top_dir is not None:
        inputs = _episode_files(args.top_dir)
        if args.num_videos is not None:
            inputs = inputs[: args.num_videos]
        default_output_dir = args.top_dir / "output"
    else:
        inputs = [args.hdf5_path]
        default_output_dir = args.hdf5_path.parent / "output"

    output_dir = args.output_dir or default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for input_path in inputs:
        output_path = output_dir / f"{input_path.stem}_event_rgb.mp4"
        print(f"creating {output_path} ...")
        try:
            frame_count = create_video(
                input_path,
                output_path,
                panel_height=args.height,
                fps_override=args.fps,
                rgb_color_order=args.rgb_color_order,
                event_crop_polygon=event_crop_polygon,
            )
            print(f"  wrote {frame_count} frames")
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            failures += 1
            print(f"  ERROR: {error}", file=sys.stderr)

    if failures:
        print(f"completed with {failures} failure(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
