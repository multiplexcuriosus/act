#!/usr/bin/env python3

import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


WINDOW_NAME = "episode_viewer"


def find_hdf5_files(path_str: str):
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in [".hdf5", ".h5"]:
            raise ValueError(f"File is not an HDF5 file: {path}")
        return [path]

    files = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in [".hdf5", ".h5"]]
    )
    if not files:
        raise ValueError(f"No .hdf5/.h5 files found in folder: {path}")

    return files


class Episode:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.h5 = None
        self.rgb = None
        self.event = None
        self.qpos = None
        self.timestamps = None
        self.length = 0

    def open(self):
        self.close()
        self.h5 = h5py.File(self.file_path, "r")
        self.rgb = self.h5["observations"]["images"]["rgb"]
        self.event = self.h5["observations"]["images"]["event"]
        self.qpos = self.h5["observations"]["qpos"]
        self.timestamps = self.h5["observations"]["timestamps"]
        self.length = self.rgb.shape[0]

        if self.event.shape[0] != self.length:
            raise RuntimeError(f"Event length mismatch in {self.file_path}")
        if self.qpos.shape[0] != self.length:
            raise RuntimeError(f"qpos length mismatch in {self.file_path}")
        if self.timestamps.shape[0] != self.length:
            raise RuntimeError(f"timestamp length mismatch in {self.file_path}")

    def close(self):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def __del__(self):
        self.close()


def to_bgr_for_display(img: np.ndarray):
    """
    Input RGB or grayscale, output BGR for cv2.imshow.
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    raise ValueError(f"Unsupported image shape for display: {img.shape}")


def annotate(img: np.ndarray, lines, scale=0.7, thickness=2):
    out = img.copy()
    y = 30
    for line in lines:
        cv2.putText(
            out,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += 28
    return out


def resize_to_height(img: np.ndarray, target_h: int):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_NEAREST)


def make_canvas(rgb_frame, event_frame, info_lines):
    rgb_disp = to_bgr_for_display(rgb_frame)
    event_disp = to_bgr_for_display(event_frame)

    # Bring both to same display height
    target_h = max(rgb_disp.shape[0], event_disp.shape[0])
    rgb_disp = resize_to_height(rgb_disp, target_h)
    event_disp = resize_to_height(event_disp, target_h)

    rgb_disp = annotate(rgb_disp, info_lines)
    event_disp = annotate(event_disp, ["EVENT"])

    spacer = np.zeros((target_h, 20, 3), dtype=np.uint8)
    canvas = np.hstack([rgb_disp, spacer, event_disp])
    return canvas


def open_episode(files, ep_idx):
    ep = Episode(files[ep_idx])
    ep.open()
    print(f"[INFO] Opened episode {ep_idx}: {ep.file_path.name}, length={ep.length}", flush=True)
    return ep


def write_marked_outputs(marked_files, output_dir: Path):
    if not marked_files:
        print("[INFO] No files marked for deletion. Skipping bash script generation.")
        return None, None

    txt_path = output_dir / "marked_for_deletion.txt"
    sh_path = output_dir / "delete_marked_files.sh"

    with open(txt_path, "w", encoding="utf-8") as f:
        for path in sorted(marked_files):
            f.write(str(path) + "\n")

    script = f"""#!/usr/bin/env bash
set -euo pipefail

LIST_FILE="{txt_path}"

if [[ ! -f "$LIST_FILE" ]]; then
  echo "List file not found: $LIST_FILE"
  exit 1
fi

while IFS= read -r file_path; do
  [[ -z "$file_path" ]] && continue

  if [[ -f "$file_path" ]]; then
    echo "Deleting: $file_path"
    rm -- "$file_path"
  else
    echo "Skipping missing file: $file_path"
  fi
done < "$LIST_FILE"
"""

    with open(sh_path, "w", encoding="utf-8") as f:
        f.write(script)

    os.chmod(sh_path, 0o755)
    return txt_path, sh_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="HDF5 file or folder containing HDF5 files")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    args = parser.parse_args()

    files = find_hdf5_files(args.input_path)
    multi_episode = len(files) > 1

    print(f"[INFO] Found {len(files)} episode file(s).", flush=True)
    for i, f in enumerate(files):
        print(f"  [{i}] {f.name}", flush=True)

    # Determine output directory for deletion script (folder containing the files)
    out_dir = files[0].parent

    marked_for_deletion: set = set()

    episode_idx = 0
    frame_idx = 0
    paused = False

    episode = open_episode(files, episode_idx)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_delay_ms = max(1, int(round(1000.0 / args.fps)))

    while True:
        if frame_idx >= episode.length:
            frame_idx = 0

        rgb_frame = episode.rgb[frame_idx]
        event_frame = episode.event[frame_idx]
        qpos = episode.qpos[frame_idx]
        ts = episode.timestamps[frame_idx]

        is_marked = episode.file_path.resolve() in marked_for_deletion
        info_lines = [
            f"EP {episode_idx + 1}/{len(files)}: {episode.file_path.name}",
            f"FRAME {frame_idx + 1}/{episode.length}",
            f"t = {ts:.3f}",
            f"qpos dim = {len(qpos)}",
            "SPACE pause | LEFT/RIGHT change episode | D mark del | ESC quit",
        ]
        if is_marked:
            info_lines.append("*** MARKED FOR DELETION ***")
        if paused:
            info_lines.append("PAUSED")

        canvas = make_canvas(rgb_frame, event_frame, info_lines)
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(0 if paused else frame_delay_ms)

        # ESC
        if key == 27:
            break

        # space
        elif key == 32:
            paused = not paused

        # right arrow
        elif key in [83, 2555904]:
            if multi_episode:
                episode.close()
                episode_idx = (episode_idx + 1) % len(files)
                episode = open_episode(files, episode_idx)
                frame_idx = 0

        # left arrow
        elif key in [81, 2424832]:
            if multi_episode:
                episode.close()
                episode_idx = (episode_idx - 1) % len(files)
                episode = open_episode(files, episode_idx)
                frame_idx = 0

        # d: toggle deletion mark for the current episode
        elif key == ord("d"):
            resolved = episode.file_path.resolve()
            if resolved in marked_for_deletion:
                marked_for_deletion.remove(resolved)
                print(f"[UNMARKED] {episode.file_path.name}", flush=True)
            else:
                marked_for_deletion.add(resolved)
                print(f"[MARKED FOR DELETION] {episode.file_path.name}", flush=True)

        # optional: manual stepping while paused
        elif paused and key == ord("n"):
            frame_idx = (frame_idx + 1) % episode.length

        elif paused and key in [ord("a"), ord("b")]:
            frame_idx = (frame_idx - 1) % episode.length

        if not paused:
            frame_idx += 1

    episode.close()
    cv2.destroyAllWindows()

    txt_path, sh_path = write_marked_outputs(marked_for_deletion, out_dir)
    if sh_path is not None:
        print(f"[OK] Marked {len(marked_for_deletion)} file(s) for deletion.")
        print(f"[OK] Delete list: {txt_path}")
        print(f"[OK] Delete script: {sh_path}")
        print(f"[OK] Run  bash {sh_path}  to delete marked files.")


if __name__ == "__main__":
    main()