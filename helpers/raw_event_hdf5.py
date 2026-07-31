#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence, Tuple

import h5py
import numpy as np


LogFn = Optional[Callable[[str], None]]


def _default_log(msg: str) -> None:
    print(msg, flush=True)


def resolve_recording_dir(
    recording_dir: str,
    *,
    allow_missing_raw_events: bool = True,
    logger: LogFn = None,
) -> Tuple[str, Optional[str], str]:
    """
    Resolve a recording directory into (bag_path, raw_events_h5_path, recording_name).

    Directory layout expected:
        recording_dir/
            <recording_name>_bag/       <- ROS2 bag
            <recording_name>_raw_events.h5  <- optional raw event HDF5
    """
    if logger is None:
        logger = _default_log

    recording_dir = os.path.abspath(recording_dir)
    if not os.path.isdir(recording_dir):
        raise RuntimeError(
            f"recording_dir does not exist or is not a directory: {recording_dir}"
        )

    recording_name = os.path.basename(recording_dir)

    exact_bag = os.path.join(recording_dir, f"{recording_name}_bag")
    bag_candidates = []
    for entry in os.listdir(recording_dir):
        full = os.path.join(recording_dir, entry)
        if not os.path.isdir(full):
            continue
        if entry.endswith("_bag") or os.path.exists(os.path.join(full, "metadata.yaml")):
            bag_candidates.append(full)

    if len(bag_candidates) == 0:
        raise RuntimeError(
            "No ROS2 bag directory found in recording_dir. "
            "Expected a *_bag directory or a directory containing metadata.yaml."
        )
    if os.path.isdir(exact_bag):
        bag_path = exact_bag
    elif len(bag_candidates) == 1:
        bag_path = bag_candidates[0]
    else:
        raise RuntimeError(
            "Multiple bag candidates found in recording_dir and no exact match "
            f"for '{recording_name}_bag'. Candidates: {bag_candidates}"
        )

    exact_h5 = os.path.join(recording_dir, f"{recording_name}_raw_events.h5")
    h5_candidates = [
        os.path.join(recording_dir, entry)
        for entry in os.listdir(recording_dir)
        if entry.endswith("_raw_events.h5")
        and os.path.isfile(os.path.join(recording_dir, entry))
    ]

    if len(h5_candidates) == 0:
        if allow_missing_raw_events:
            logger(
                "[WARNING] no *_raw_events.h5 found in recording_dir; "
                "raw-event sidecar was not resolved"
            )
            raw_events_h5_path: Optional[str] = None
        else:
            raise RuntimeError(
                "No *_raw_events.h5 found in recording_dir and raw events are "
                "required for this conversion mode."
            )
    elif os.path.isfile(exact_h5):
        raw_events_h5_path = exact_h5
    elif len(h5_candidates) == 1:
        raw_events_h5_path = h5_candidates[0]
    else:
        raise RuntimeError(
            "Multiple *_raw_events.h5 candidates found in recording_dir and no "
            f"exact match for '{recording_name}_raw_events.h5'. Candidates: {h5_candidates}"
        )

    return os.path.abspath(bag_path), raw_events_h5_path, recording_name


def render_event_frame_from_raw_arrays(
    event_type: np.ndarray,
    event_x: np.ndarray,
    event_y: np.ndarray,
    width: int = 320,
    height: int = 320,
    contrast: float = 4.0,
    step: float = 1.0,
    scaling_mode: str = "legacy_per_frame_max",
    event_clip_count: Optional[float] = None,
) -> np.ndarray:
    """
    Render a single channel event image into uint8.

    legacy_per_frame_max:
      Keep historical behavior from bag_to_hdf5: normalize by per-frame max abs,
      then apply contrast around neutral gray 128.

    signed_log1p_fixed_clip:
      Use fixed clipping denominator across all frames/channels:
      z = sign(acc) * log1p(abs(acc)) / log1p(event_clip_count)
    """
    if scaling_mode not in ("legacy_per_frame_max", "signed_log1p_fixed_clip"):
        raise RuntimeError(f"Unsupported event scaling_mode: {scaling_mode}")

    if event_type.size == 0:
        return np.full((height, width), 128, dtype=np.uint8)

    xs = np.asarray(event_x, dtype=np.int64)
    ys = np.asarray(event_y, dtype=np.int64)
    et = np.asarray(event_type)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not np.any(valid):
        return np.full((height, width), 128, dtype=np.uint8)

    xs = xs[valid]
    ys = ys[valid]
    et = et[valid]

    acc = np.zeros((height, width), dtype=np.float32)
    values = np.where(et == 1, float(step), -float(step)).astype(np.float32)
    np.add.at(acc, (ys, xs), values)

    if scaling_mode == "legacy_per_frame_max":
        max_abs = float(np.max(np.abs(acc)))
        if max_abs > 0.0:
            acc = acc / max_abs
        img = 128.0 + (acc * float(contrast) * 127.0)
        img = np.clip(img, 0.0, 255.0)
        return img.astype(np.uint8)

    if event_clip_count is None or float(event_clip_count) <= 0.0:
        raise RuntimeError(
            "event_clip_count must be positive for signed_log1p_fixed_clip"
        )
    denom = np.log1p(float(event_clip_count))
    z = np.sign(acc) * np.log1p(np.abs(acc)) / denom
    z = np.clip(z, -1.0, 1.0)
    u8 = np.rint(128.0 + 127.0 * z).astype(np.uint8)
    return u8


class RawEventStore:
    def __init__(self, h5_path: str, logger: LogFn = None):
        if logger is None:
            logger = _default_log
        self._log = logger

        self.h5_path = h5_path
        self.f = h5py.File(h5_path, "r")

        required = [
            "events/type",
            "events/x",
            "events/y",
            "events/t_us",
            "packets/ros_t_ns",
            "packets/start_event_idx",
            "packets/end_event_idx",
        ]
        missing = [k for k in required if k not in self.f]
        if missing:
            self.f.close()
            raise ValueError(
                f"Raw event HDF5 missing required datasets: {missing}"
            )

        self.events_type_ds = self.f["events/type"]
        self.events_x_ds = self.f["events/x"]
        self.events_y_ds = self.f["events/y"]
        self.events_t_us_ds = self.f["events/t_us"]
        self.packets_ros_t_ns_ds = self.f["packets/ros_t_ns"]
        self.packets_start_event_idx_ds = self.f["packets/start_event_idx"]
        self.packets_end_event_idx_ds = self.f["packets/end_event_idx"]

        event_shapes = {
            "events/type": self.events_type_ds.shape,
            "events/x": self.events_x_ds.shape,
            "events/y": self.events_y_ds.shape,
            "events/t_us": self.events_t_us_ds.shape,
        }
        for dataset_name, shape in event_shapes.items():
            if len(shape) != 1:
                self.f.close()
                raise ValueError(
                    f"Malformed {dataset_name}: expected 1-D array, got shape {shape}"
                )

        event_lengths = {name: int(shape[0]) for name, shape in event_shapes.items()}
        if len(set(event_lengths.values())) != 1:
            self.f.close()
            raise ValueError(
                "Malformed event arrays: expected equal lengths for "
                f"events/type, events/x, events/y, events/t_us; got {event_lengths}"
            )
        num_events = int(self.events_t_us_ds.shape[0])

        packet_shapes = {
            "packets/ros_t_ns": self.packets_ros_t_ns_ds.shape,
            "packets/start_event_idx": self.packets_start_event_idx_ds.shape,
            "packets/end_event_idx": self.packets_end_event_idx_ds.shape,
        }
        for dataset_name, shape in packet_shapes.items():
            if len(shape) != 1:
                self.f.close()
                raise ValueError(
                    f"Malformed {dataset_name}: expected 1-D array, got shape {shape}"
                )

        packet_lengths = {
            name: int(shape[0]) for name, shape in packet_shapes.items()
        }
        if len(set(packet_lengths.values())) != 1:
            self.f.close()
            raise ValueError(
                "Malformed packet arrays: expected equal lengths for "
                "packets/ros_t_ns, packets/start_event_idx, packets/end_event_idx; "
                f"got {packet_lengths}"
            )
        if int(self.packets_ros_t_ns_ds.shape[0]) <= 0:
            self.f.close()
            raise ValueError(
                "Malformed packets/ros_t_ns: expected at least one packet timestamp"
            )

        self.packet_ros_t_ns = np.asarray(self.packets_ros_t_ns_ds[:], dtype=np.int64)
        self.packet_start_event_idx = np.asarray(
            self.packets_start_event_idx_ds[:], dtype=np.int64
        )
        self.packet_end_event_idx = np.asarray(
            self.packets_end_event_idx_ds[:], dtype=np.int64
        )

        if np.any(self.packet_ros_t_ns < 0):
            self.f.close()
            raise ValueError(
                "Malformed packets/ros_t_ns: packet ROS timestamps must be >= 0"
            )
        if np.any(np.diff(self.packet_ros_t_ns) < 0):
            self.f.close()
            raise ValueError(
                "Malformed packets/ros_t_ns: packet ROS timestamps must be non-decreasing"
            )

        if np.any(np.diff(self.packet_start_event_idx) < 0):
            self.f.close()
            raise ValueError(
                "Malformed packets/start_event_idx: indices must be non-decreasing"
            )
        if np.any(np.diff(self.packet_end_event_idx) < 0):
            self.f.close()
            raise ValueError(
                "Malformed packets/end_event_idx: indices must be non-decreasing"
            )

        if np.any(self.packet_start_event_idx < 0):
            self.f.close()
            raise ValueError(
                "Malformed packets/start_event_idx: values must be >= 0"
            )
        if np.any(self.packet_end_event_idx < 0):
            self.f.close()
            raise ValueError(
                "Malformed packets/end_event_idx: values must be >= 0"
            )
        if np.any(self.packet_start_event_idx > self.packet_end_event_idx):
            self.f.close()
            raise ValueError(
                "Malformed packet indices: require start_event_idx <= end_event_idx for every packet"
            )
        if np.any(self.packet_end_event_idx > num_events):
            self.f.close()
            raise ValueError(
                "Malformed packets/end_event_idx: value exceeds number of events "
                f"({num_events})"
            )

        if "width" not in self.f.attrs:
            self.f.close()
            raise ValueError("Malformed sidecar metadata: missing attribute 'width'")
        if "height" not in self.f.attrs:
            self.f.close()
            raise ValueError("Malformed sidecar metadata: missing attribute 'height'")

        width_attr = self.f.attrs["width"]
        height_attr = self.f.attrs["height"]
        width_float = float(width_attr)
        height_float = float(height_attr)
        if not width_float.is_integer():
            self.f.close()
            raise ValueError(
                f"Malformed sidecar metadata: width must be an integer, got {width_attr}"
            )
        if not height_float.is_integer():
            self.f.close()
            raise ValueError(
                f"Malformed sidecar metadata: height must be an integer, got {height_attr}"
            )

        self.width = int(width_float)
        self.height = int(height_float)
        if self.width <= 0:
            self.f.close()
            raise ValueError(
                f"Malformed sidecar metadata: width must be positive, got {self.width}"
            )
        if self.height <= 0:
            self.f.close()
            raise ValueError(
                f"Malformed sidecar metadata: height must be positive, got {self.height}"
            )

        if self.packet_ros_t_ns.size > 0:
            self.packet_ros_start_ns = int(self.packet_ros_t_ns[0])
            self.packet_ros_end_ns = int(self.packet_ros_t_ns[-1])
            self._log(
                "[INFO] raw events packet ROS time range (sec): "
                f"{self.packet_ros_start_ns * 1e-9:.6f} .. "
                f"{self.packet_ros_end_ns * 1e-9:.6f}"
            )
        else:
            self.packet_ros_start_ns = None
            self.packet_ros_end_ns = None
            self._log("[WARNING] raw events HDF5 has no packets in /packets/ros_t_ns")

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    def _validate_windows_and_mode(
        self,
        windows_ms: Sequence[float],
        mode: str,
    ) -> Tuple[float, float, float]:
        if len(windows_ms) != 3:
            raise RuntimeError(f"windows_ms must have length 3, got {windows_ms}")

        w0_ms, w1_ms, w2_ms = [float(w) for w in windows_ms]
        if not (w0_ms > 0.0 and w1_ms > 0.0 and w2_ms > 0.0):
            raise RuntimeError(f"windows_ms must be positive, got {windows_ms}")
        if not (w0_ms <= w1_ms <= w2_ms):
            raise RuntimeError(
                "windows_ms must be non-decreasing (small to large), "
                f"got {windows_ms}"
            )
        if mode not in ("cumulative", "shifted"):
            raise RuntimeError(f"Unsupported event frame mode: {mode}")

        return w0_ms, w1_ms, w2_ms

    def _causal_event_slice_by_bag_time(
        self,
        bag_t_sec: float,
        max_window_ms: float,
        packet_margin_ms: float,
    ):
        neutral = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        if self.packet_ros_t_ns.size == 0:
            return {
                "neutral": neutral,
                "empty": True,
                "ev_type": None,
                "ev_x": None,
                "ev_y": None,
                "ev_t_us": None,
                "source_packet_ros_t_ns": None,
            }

        now_ros_ns = int(bag_t_sec * 1e9)
        max_window_ns = int(max_window_ms * 1e6)
        margin_ns = int(float(packet_margin_ms) * 1e6)
        start_ros_ns = now_ros_ns - max_window_ns - margin_ns

        p0 = int(np.searchsorted(self.packet_ros_t_ns, start_ros_ns, side="left"))
        p1 = int(np.searchsorted(self.packet_ros_t_ns, now_ros_ns, side="right"))
        if p0 >= p1:
            return {
                "neutral": neutral,
                "empty": True,
                "ev_type": None,
                "ev_x": None,
                "ev_y": None,
                "ev_t_us": None,
                "source_packet_ros_t_ns": None,
            }

        ev_start = int(self.packet_start_event_idx[p0])
        ev_end = int(self.packet_end_event_idx[p1 - 1])
        if ev_end <= ev_start:
            return {
                "neutral": neutral,
                "empty": True,
                "ev_type": None,
                "ev_x": None,
                "ev_y": None,
                "ev_t_us": None,
                "source_packet_ros_t_ns": int(self.packet_ros_t_ns[p1 - 1]),
            }

        ev_type = np.asarray(self.events_type_ds[ev_start:ev_end])
        ev_x = np.asarray(self.events_x_ds[ev_start:ev_end])
        ev_y = np.asarray(self.events_y_ds[ev_start:ev_end])
        ev_t_us = np.asarray(self.events_t_us_ds[ev_start:ev_end], dtype=np.int64)

        if ev_t_us.size == 0:
            return {
                "neutral": neutral,
                "empty": True,
                "ev_type": None,
                "ev_x": None,
                "ev_y": None,
                "ev_t_us": None,
                "source_packet_ros_t_ns": int(self.packet_ros_t_ns[p1 - 1]),
            }

        return {
            "neutral": neutral,
            "empty": False,
            "ev_type": ev_type,
            "ev_x": ev_x,
            "ev_y": ev_y,
            "ev_t_us": ev_t_us,
            "source_packet_ros_t_ns": int(self.packet_ros_t_ns[p1 - 1]),
        }

    def _time_ranges_us(
        self,
        now_event_t_us: int,
        w0_ms: float,
        w1_ms: float,
        w2_ms: float,
        mode: str,
    ):
        w0_us = int(w0_ms * 1e3)
        w1_us = int(w1_ms * 1e3)
        w2_us = int(w2_ms * 1e3)

        if mode == "shifted":
            return [
                (now_event_t_us - w0_us, now_event_t_us),
                (now_event_t_us - w1_us, now_event_t_us - w0_us),
                (now_event_t_us - w2_us, now_event_t_us - w1_us),
            ]

        return [
            (now_event_t_us - w0_us, now_event_t_us),
            (now_event_t_us - w1_us, now_event_t_us),
            (now_event_t_us - w2_us, now_event_t_us),
        ]

    def frame_3chef_with_metadata_at_bag_time(
        self,
        bag_t_sec: float,
        windows_ms: Tuple[float, float, float] = (50.0, 250.0, 1000.0),
        mode: str = "cumulative",
        contrast: float = 4.0,
        step: float = 1.0,
        packet_margin_ms: float = 50.0,
        scaling_mode: str = "legacy_per_frame_max",
        event_clip_count: Optional[float] = None,
    ) -> Tuple[np.ndarray, Optional[int], np.ndarray]:
        w0_ms, w1_ms, w2_ms = self._validate_windows_and_mode(windows_ms, mode)
        result = self._causal_event_slice_by_bag_time(
            bag_t_sec=bag_t_sec,
            max_window_ms=w2_ms,
            packet_margin_ms=packet_margin_ms,
        )

        if result["empty"]:
            return (
                result["neutral"],
                result["source_packet_ros_t_ns"],
                np.zeros((3,), dtype=np.int32),
            )

        ev_type = result["ev_type"]
        ev_x = result["ev_x"]
        ev_y = result["ev_y"]
        ev_t_us = result["ev_t_us"]

        now_event_t_us = int(np.max(ev_t_us))
        ranges = self._time_ranges_us(now_event_t_us, w0_ms, w1_ms, w2_ms, mode)

        channels = []
        counts = []
        for lo_us, hi_us in ranges:
            if hi_us == now_event_t_us:
                mask = (ev_t_us >= lo_us) & (ev_t_us <= hi_us)
            else:
                mask = (ev_t_us >= lo_us) & (ev_t_us < hi_us)

            counts.append(int(np.count_nonzero(mask)))
            if counts[-1] == 0:
                channels.append(np.full((self.height, self.width), 128, dtype=np.uint8))
                continue

            frame = render_event_frame_from_raw_arrays(
                event_type=ev_type[mask],
                event_x=ev_x[mask],
                event_y=ev_y[mask],
                width=self.width,
                height=self.height,
                contrast=contrast,
                step=step,
                scaling_mode=scaling_mode,
                event_clip_count=event_clip_count,
            )
            channels.append(frame)

        return (
            np.stack(channels, axis=2).astype(np.uint8),
            result["source_packet_ros_t_ns"],
            np.asarray(counts, dtype=np.int32),
        )

    def frame_3chef_at_bag_time(
        self,
        bag_t_sec: float,
        windows_ms: Tuple[float, float, float] = (50.0, 250.0, 1000.0),
        mode: str = "cumulative",
        contrast: float = 4.0,
        step: float = 1.0,
        packet_margin_ms: float = 50.0,
        scaling_mode: str = "legacy_per_frame_max",
        event_clip_count: Optional[float] = None,
    ) -> np.ndarray:
        frame, _, _ = self.frame_3chef_with_metadata_at_bag_time(
            bag_t_sec=bag_t_sec,
            windows_ms=windows_ms,
            mode=mode,
            contrast=contrast,
            step=step,
            packet_margin_ms=packet_margin_ms,
            scaling_mode=scaling_mode,
            event_clip_count=event_clip_count,
        )
        return frame
