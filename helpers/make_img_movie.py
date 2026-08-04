#!/usr/bin/env python3

import argparse
from pathlib import Path
import h5py
import cv2
import numpy as np


# Common OpenCV arrow-key codes across backends/platforms.
LEFT_KEYS = {81, 2424832, 65361}
RIGHT_KEYS = {83, 2555904, 65363}


def to_uint8_img(img):
    img = np.asarray(img)

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (255 * (img - mn) / (mx - mn)).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    return img


def ensure_bgr(img):
    img = to_uint8_img(img)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[-1] == 1:
        return cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[-1] == 3:
        # HDF5 images are usually RGB, OpenCV wants BGR
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def get_screen_width(default_width=1920):
    # Try Tk first for an actual display width; fall back if unavailable.
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        root.destroy()
        if width > 0:
            return width
    except Exception:
        pass

    return default_width


def resolve_hdf5_path(path_str):
    path = Path(path_str)

    if path.is_file():
        return path

    if path.is_dir():
        episode_files = sorted(path.glob("episode_*.hdf5"))
        if not episode_files:
            episode_files = sorted(path.glob("*.hdf5"))
        if not episode_files:
            raise FileNotFoundError(f"No .hdf5 files found in directory: {path}")

        selected = episode_files[0]
        print(f"[INFO] Input is a directory; using {selected}")
        return selected

    raise FileNotFoundError(f"HDF5 path does not exist: {path}")


def compose_rgb_event_frame(rgb_img, event_img=None, frame_index=None, frame_count=None):
    """Build the same frame used by both interactive replay and MP4 output."""
    rgb_img = ensure_bgr(rgb_img)

    if event_img is None:
        frame = rgb_img
    else:
        event_img = ensure_bgr(event_img)
        event_img = cv2.rotate(event_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if event_img.shape[:2] != rgb_img.shape[:2]:
            event_img = cv2.resize(
                event_img,
                (rgb_img.shape[1], rgb_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        frame = np.hstack([rgb_img, event_img])

    if frame_index is not None and frame_count is not None:
        cv2.putText(
            frame,
            f"frame {frame_index}/{frame_count - 1}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

    return frame


def get_image_datasets(h5_file, hdf5_path, rgb_path, event_path, rgb_only):
    if rgb_path not in h5_file:
        raise KeyError(f"Dataset '{rgb_path}' not found in {hdf5_path}.")

    rgb = h5_file[rgb_path]
    event = None
    if not rgb_only:
        if event_path not in h5_file:
            raise KeyError(
                f"Dataset '{event_path}' not found in {hdf5_path}. "
                "Use --rgb-only to replay just the RGB images."
            )
        event = h5_file[event_path]

    n = len(rgb) if rgb_only else min(len(rgb), len(event))
    if n == 0:
        raise ValueError(f"No image frames found in {hdf5_path}.")

    return rgb, event, n


def write_video(hdf5_path, output_path, rgb_path, event_path, rgb_only, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as h5_file:
        rgb, event, n = get_image_datasets(
            h5_file, hdf5_path, rgb_path, event_path, rgb_only
        )
        first_frame = compose_rgb_event_frame(
            rgb[0], None if rgb_only else event[0], 0, n
        )
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not open MP4 writer for {output_path}")

        try:
            writer.write(first_frame)
            for i in range(1, n):
                frame = compose_rgb_event_frame(
                    rgb[i], None if rgb_only else event[i], i, n
                )
                writer.write(frame)
        finally:
            writer.release()

    print(f"[INFO] Saved {n} frames to {output_path}")
    return n


def select_child_hdf5(child_dir):
    episode_files = sorted(child_dir.glob("episode_*.hdf5"))
    if episode_files:
        return episode_files[0]

    hdf5_files = sorted(child_dir.glob("*.hdf5"))
    return hdf5_files[0] if hdf5_files else None


def run_batch(top_dir, output_dir, args):
    top_dir = Path(top_dir)
    if not top_dir.is_dir():
        raise ValueError(f"--top_dir must be an existing directory: {top_dir}")

    output_dir = Path(output_dir) if output_dir else top_dir / "videos"
    generated = 0
    skipped = 0

    # A dataset collection commonly stores all episode files directly in the
    # supplied directory. Process every one, not just the first episode.
    direct_files = sorted(top_dir.glob("episode_*.hdf5"))
    if not direct_files:
        direct_files = sorted(top_dir.glob("*.hdf5"))

    candidates = [
        (hdf5_path, output_dir / f"{hdf5_path.stem}_video.mp4")
        for hdf5_path in direct_files
    ]

    # Also support collections whose immediate children each contain a
    # dataset, preserving the child name in the output to avoid collisions.
    child_dirs = sorted(path for path in top_dir.iterdir() if path.is_dir())
    for child_dir in child_dirs:
        # Do not treat the default output directory as an input on reruns.
        if child_dir.resolve() == output_dir.resolve():
            continue

        hdf5_path = select_child_hdf5(child_dir)
        if hdf5_path is None:
            print(f"[WARN] Skipping {child_dir}: no .hdf5 file found")
            skipped += 1
            continue

        output_path = output_dir / f"{child_dir.name}_{hdf5_path.stem}_video.mp4"
        candidates.append((hdf5_path, output_path))

    for hdf5_path, output_path in candidates:
        try:
            write_video(
                hdf5_path,
                output_path,
                args.rgb_path,
                args.event_path,
                args.rgb_only,
                args.fps,
            )
            generated += 1
        except Exception as exc:
            print(f"[WARN] Skipping {hdf5_path}: {exc}")
            skipped += 1

    print(f"[INFO] Batch complete: generated={generated}, skipped={skipped}")
    if generated == 0:
        raise RuntimeError(f"No videos were generated from child directories of {top_dir}")

    return generated, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path", nargs="?")
    parser.add_argument("--rgb-path", default="/observations/images/rgb")
    parser.add_argument("--event-path", default="/observations/images/event")
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="Replay only RGB images and do not require an event dataset",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--video",
        action="store_true",
        help="Save a finite MP4 instead of opening the interactive replay window",
    )
    parser.add_argument(
        "--top_dir",
        "--top-dir",
        type=Path,
        help=(
            "Generate videos for all direct HDF5 files and the first HDF5 file "
            "in each immediate child directory"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Video output directory (default: a videos folder beside the input)",
    )
    parser.add_argument("--window", default="rgb_event_replay")
    parser.add_argument(
        "--n-frame-view",
        action="store_true",
        help="Enable side-by-side chronological n-frame view",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=3,
        help="Number of frames to show in n-frame view (1-10)",
    )
    args = parser.parse_args()

    if (args.hdf5_path is None) == (args.top_dir is None):
        parser.error("provide exactly one of hdf5_path or --top_dir")
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    if args.top_dir is not None:
        run_batch(args.top_dir, args.output_dir, args)
        return

    hdf5_path = resolve_hdf5_path(args.hdf5_path)

    if args.video:
        output_dir = args.output_dir or hdf5_path.parent / "videos"
        output_path = output_dir / f"{hdf5_path.stem}_video.mp4"
        write_video(
            hdf5_path,
            output_path,
            args.rgb_path,
            args.event_path,
            args.rgb_only,
            args.fps,
        )
        return

    if args.output_dir is not None:
        parser.error("--output-dir requires --video or --top_dir")

    with h5py.File(hdf5_path, "r") as f:
        rgb, event, n = get_image_datasets(
            f, hdf5_path, args.rgb_path, args.event_path, args.rgb_only
        )
        delay_ms = max(1, int(1000 / args.fps))
        n_view = max(1, min(args.n_frames, 10))
        screen_width = get_screen_width()

        if args.n_frames != n_view:
            print(f"[WARN] --n-frames clipped to {n_view} (allowed range: 1-10)")

        print(f"[INFO] RGB shape:   {rgb.shape}")
        if event is not None:
            print(f"[INFO] Event shape: {event.shape}")
        else:
            print("[INFO] Event shape: disabled (--rgb-only)")
        print(f"[INFO] Playing {n} frames on repeat")
        if args.n_frame_view:
            print(f"[INFO] n-frame view enabled: showing {n_view} frames side-by-side")
            print(f"[INFO] n-frame view target width: {screen_width}px")
        print("[INFO] Controls: SPACE pause/resume, LEFT/RIGHT step (paused only), q/ESC quit")

        cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

        i = 0
        paused = False

        while True:
            if args.n_frame_view:
                tiles = []
                for k in range(n_view):
                    idx = (i + k) % n

                    tile_img = ensure_bgr(rgb[idx] if args.rgb_only else event[idx])

                    cv2.putText(
                        tile_img,
                        f"{idx}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                    )
                    tile_img = cv2.copyMakeBorder(
                        tile_img,
                        2,
                        2,
                        2,
                        2,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                    tiles.append(tile_img)

                frame = np.hstack(tiles)
                end_idx = (i + n_view - 1) % n
                cv2.putText(
                    frame,
                    f"view {i}->{end_idx} / {n-1}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )

                if frame.shape[1] != screen_width and frame.shape[1] > 0:
                    scale = screen_width / float(frame.shape[1])
                    target_h = max(1, int(round(frame.shape[0] * scale)))
                    frame = cv2.resize(
                        frame,
                        (screen_width, target_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
            else:
                frame = compose_rgb_event_frame(
                    rgb[i], None if args.rgb_only else event[i], i, n
                )


            cv2.imshow(args.window, frame)

            key = cv2.waitKeyEx(delay_ms if not paused else 0)
            key8 = key & 0xFF

            if key8 in [ord("q"), 27]:
                break
            elif key8 == ord(" "):
                paused = not paused
            elif paused and (key in LEFT_KEYS or key8 in [ord("a"), ord("h"), ord(",")]):
                i = (i - 1) % n
            elif paused and (key in RIGHT_KEYS or key8 in [ord("d"), ord("l"), ord(".")]):
                i = (i + 1) % n
                i = (i + 1) % n
            elif not paused:
                i = (i + 1) % n

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
