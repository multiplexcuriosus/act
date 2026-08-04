from types import SimpleNamespace

import h5py
import numpy as np

from src.act.helpers import make_img_movie


def make_hdf5(path, rgb_frames=3, event_frames=2):
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            "/observations/images/rgb",
            data=np.full((rgb_frames, 4, 6, 3), (10, 20, 30), dtype=np.uint8),
        )
        h5_file.create_dataset(
            "/observations/images/event",
            data=np.full((event_frames, 2, 3), 40, dtype=np.uint8),
        )


class FakeVideoWriter:
    instances = []

    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames = []
        self.released = False
        self.instances.append(self)

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


def test_compose_rgb_event_frame_places_rgb_left_and_event_right():
    rgb = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    event = np.array([[20, 30], [40, 50]], dtype=np.uint8)

    frame = make_img_movie.compose_rgb_event_frame(rgb, event)

    assert frame.shape == (2, 4, 3)
    np.testing.assert_array_equal(
        frame[:, :2],
        np.array(
            [
                [[12, 11, 10], [9, 8, 7]],
                [[6, 5, 4], [3, 2, 1]],
            ],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_array_equal(
        frame[:, 2:],
        np.array(
            [
                [[30, 30, 30], [50, 50, 50]],
                [[20, 20, 20], [40, 40, 40]],
            ],
            dtype=np.uint8,
        ),
    )


def test_write_video_uses_shorter_stream_and_releases_writer(tmp_path, monkeypatch):
    hdf5_path = tmp_path / "episode_0.hdf5"
    make_hdf5(hdf5_path, rgb_frames=3, event_frames=2)
    FakeVideoWriter.instances.clear()
    monkeypatch.setattr(make_img_movie.cv2, "VideoWriter", FakeVideoWriter)
    monkeypatch.setattr(make_img_movie.cv2, "VideoWriter_fourcc", lambda *args: 1234)

    count = make_img_movie.write_video(
        hdf5_path,
        tmp_path / "videos" / "episode_0_video.mp4",
        "/observations/images/rgb",
        "/observations/images/event",
        False,
        30.0,
    )

    writer = FakeVideoWriter.instances[0]
    assert count == 2
    assert writer.fps == 30.0
    assert writer.size == (12, 4)
    assert len(writer.frames) == 2
    assert writer.released


def test_batch_selects_preferred_file_and_skips_empty_child(tmp_path, monkeypatch):
    top_dir = tmp_path / "datasets"
    child_a = top_dir / "run_a"
    child_b = top_dir / "run_b"
    child_a.mkdir(parents=True)
    child_b.mkdir()
    (child_a / "other.hdf5").touch()
    (child_a / "episode_2.hdf5").touch()
    (child_a / "episode_1.hdf5").touch()

    calls = []

    def fake_write_video(hdf5_path, output_path, *args):
        calls.append((hdf5_path, output_path))

    monkeypatch.setattr(make_img_movie, "write_video", fake_write_video)
    args = SimpleNamespace(
        rgb_path="/observations/images/rgb",
        event_path="/observations/images/event",
        rgb_only=False,
        fps=30.0,
    )

    generated, skipped = make_img_movie.run_batch(top_dir, None, args)

    assert (generated, skipped) == (1, 1)
    assert calls == [
        (
            child_a / "episode_1.hdf5",
            top_dir / "videos" / "run_a_episode_1_video.mp4",
        )
    ]


def test_batch_continues_after_invalid_dataset(tmp_path, monkeypatch):
    top_dir = tmp_path / "datasets"
    for name in ("bad", "good"):
        child = top_dir / name
        child.mkdir(parents=True, exist_ok=True)
        (child / "episode_0.hdf5").touch()

    def fake_write_video(hdf5_path, output_path, *args):
        if hdf5_path.parent.name == "bad":
            raise KeyError("missing event dataset")

    monkeypatch.setattr(make_img_movie, "write_video", fake_write_video)
    args = SimpleNamespace(
        rgb_path="rgb",
        event_path="event",
        rgb_only=False,
        fps=30.0,
    )

    assert make_img_movie.run_batch(top_dir, tmp_path / "output", args) == (1, 1)


def test_batch_processes_every_hdf5_file_directly_under_top_dir(
    tmp_path, monkeypatch
):
    top_dir = tmp_path / "datasets"
    top_dir.mkdir()
    (top_dir / "episode_1.hdf5").touch()
    (top_dir / "episode_0.hdf5").touch()
    (top_dir / "unrelated.hdf5").touch()
    calls = []

    def fake_write_video(hdf5_path, output_path, *args):
        calls.append((hdf5_path, output_path))

    monkeypatch.setattr(make_img_movie, "write_video", fake_write_video)
    args = SimpleNamespace(
        rgb_path="rgb",
        event_path="event",
        rgb_only=False,
        fps=30.0,
    )

    assert make_img_movie.run_batch(top_dir, None, args) == (2, 0)
    assert calls == [
        (
            top_dir / "episode_0.hdf5",
            top_dir / "videos" / "episode_0_video.mp4",
        ),
        (
            top_dir / "episode_1.hdf5",
            top_dir / "videos" / "episode_1_video.mp4",
        ),
    ]
