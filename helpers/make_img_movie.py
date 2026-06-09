#!/usr/bin/env python3

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path")
    parser.add_argument("--rgb-path", default="/observations/images/rgb")
    parser.add_argument("--event-path", default="/observations/images/event")
    parser.add_argument("--fps", type=float, default=30.0)
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

    with h5py.File(args.hdf5_path, "r") as f:
        rgb = f[args.rgb_path]
        event = f[args.event_path]

        n = min(len(rgb), len(event))
        delay_ms = max(1, int(1000 / args.fps))
        n_view = max(1, min(args.n_frames, 10))
        screen_width = get_screen_width()

        if args.n_frames != n_view:
            print(f"[WARN] --n-frames clipped to {n_view} (allowed range: 1-10)")

        print(f"[INFO] RGB shape:   {rgb.shape}")
        print(f"[INFO] Event shape: {event.shape}")
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

                    event_img = ensure_bgr(event[idx])

                    #event_img = cv2.rotate(event_img, cv2.ROTATE_180)

                    cv2.putText(
                        event_img,
                        f"{idx}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                    )
                    event_img = cv2.copyMakeBorder(
                        event_img,
                        2,
                        2,
                        2,
                        2,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                    tiles.append(event_img)

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
                rgb_img = ensure_bgr(rgb[i])
                event_img = ensure_bgr(event[i])

                if event_img.shape[:2] != rgb_img.shape[:2]:
                    event_img = cv2.resize(
                        event_img,
                        (rgb_img.shape[1], rgb_img.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                event_img = cv2.rotate(event_img, cv2.ROTATE_180)

                frame = np.hstack([event_img, rgb_img])
                frame = cv2.rotate(frame, cv2.ROTATE_180)

                cv2.putText(frame, f"frame {i}/{n-1}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


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
