from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


HELPERS_DIR = Path(__file__).resolve().parent
ACT_DIR = HELPERS_DIR.parent
for path in (str(ACT_DIR), str(HELPERS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from image_preprocessing import (  # noqa: E402
    mask_and_center_crop_square_rotated_event_image,
    mask_and_left_crop_image,
    mask_and_left_crop_rotated_event_image,
)
import preprocess_hdf5_images as cli  # noqa: E402


def _write_file(
    path: Path,
    *,
    rgb_shape=(3, 3, 6, 3),
    event_shape=(3, 3, 6, 3),
    include_rgb=True,
    include_event=True,
    compression=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as root:
        root.attrs["root_note"] = "preserve me"
        images = root.require_group("observations/images")
        observations = root["observations"]
        observations.attrs["group_note"] = 17
        root.create_dataset("action", data=np.arange(6, dtype=np.float32).reshape(3, 2))
        root["action"].attrs["units"] = "m"
        root["action_alias"] = root["action"]
        root["soft_action"] = h5py.SoftLink("/action")
        kwargs = {"compression": compression, "chunks": True} if compression else {}
        if include_rgb:
            rgb = np.arange(np.prod(rgb_shape), dtype=np.uint8).reshape(rgb_shape)
            images.create_dataset("rgb", data=rgb, **kwargs)
            images["rgb"].attrs["camera"] = "top"
        if include_event:
            event = np.full(event_shape, 128, dtype=np.uint8)
            event[..., -1, :] = 200
            images.create_dataset("event", data=event, **kwargs)
            images["event"].attrs["neutral"] = 128


def _single_plan(source, output, modality="rgb", mask_x=None, crop_x=1, overwrite=False):
    return cli.build_file_plans(
        [(Path(source), Path(output))],
        cli._selected_modalities(modality),
        mask_x,
        crop_x,
        overwrite,
    )[0]


def test_exact_diagonal_boundary_first_middle_last_rows():
    image = np.full((3, 6), 9, dtype=np.uint8)
    result = mask_and_left_crop_image(image, mask_x=(1, 3), crop_x=0, fill_value=0)
    assert np.array_equal(result[0], [0, 0, 9, 9, 9, 9])
    assert np.array_equal(result[1], [0, 0, 0, 9, 9, 9])
    assert np.array_equal(result[2], [0, 0, 0, 0, 9, 9])


def test_one_row_uses_xtop_and_rgb_fill_all_channels():
    image = np.full((1, 5, 3), 7, dtype=np.uint8)
    result = mask_and_left_crop_image(image, mask_x=(2, 4), crop_x=0, fill_value=0)
    assert np.array_equal(result[0, :3], np.zeros((3, 3), dtype=np.uint8))
    assert np.array_equal(result[0, 3:], np.full((2, 3), 7, dtype=np.uint8))


def test_event_fill_128_mask_only():
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    result = mask_and_left_crop_image(image, mask_x=(0, 2), crop_x=0, fill_value=128)
    assert np.all(result[0, 0] == 128)
    assert np.all(result[1, :2] == 128)
    assert np.all(result[2, :3] == 128)
    assert result.shape == image.shape


def test_crop_only_and_combined_mask_then_crop():
    image = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6)
    cropped = mask_and_left_crop_image(image, crop_x=2, fill_value=0)
    assert np.array_equal(cropped, image[:, 2:])
    combined = mask_and_left_crop_image(image, mask_x=(3, 3), crop_x=2, fill_value=0)
    expected = image.copy()
    expected[:, :4] = 0
    assert np.array_equal(combined, expected[:, 2:])
    assert combined.shape == (4, 4)


def test_event_transform_uses_rotated_view_and_maps_back_to_storage():
    image = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6)
    result = mask_and_left_crop_rotated_event_image(
        image, mask_x=(1, 3), crop_x=1, fill_value=128
    )
    manual_rotated = np.rot90(image, k=1)
    manual_transformed = mask_and_left_crop_image(
        manual_rotated, mask_x=(1, 3), crop_x=1, fill_value=128
    )
    expected = np.rot90(manual_transformed, k=-1)
    assert np.array_equal(result, expected)
    assert result.shape == (3, 6)


def test_event_crop_only_becomes_top_crop_in_stored_orientation(tmp_path):
    source = tmp_path / "source.hdf5"
    output = tmp_path / "output.hdf5"
    _write_file(source, include_rgb=False, event_shape=(3, 4, 7, 3))
    with h5py.File(source, "r") as root:
        original = root[cli.DATASET_KEYS["event"]][:]
    plan = _single_plan(source, output, modality="event", crop_x=2)
    assert plan.datasets[0].result_shape == (3, 2, 7, 3)
    cli.process_file(plan, None, 2)
    with h5py.File(output, "r") as root:
        assert np.array_equal(root[cli.DATASET_KEYS["event"]][:], original[:, 2:])
        assert (
            root[cli.DATASET_KEYS["event"]].attrs[
                f"{cli.METADATA_PREFIX}transform_coordinate_frame"
            ]
            == "rotated_90deg_ccw_debug_view"
        )


def test_event_center_square_crop_masks_then_keeps_concentric_region(tmp_path):
    image = np.arange(7 * 7, dtype=np.uint8).reshape(7, 7)
    result = mask_and_center_crop_square_rotated_event_image(
        image, mask_x=(1, 3), square_side=3, fill_value=128
    )
    rotated = np.rot90(image, k=1)
    masked = mask_and_left_crop_image(rotated, mask_x=(1, 3), crop_x=0, fill_value=128)
    expected = np.rot90(masked[2:5, 2:5], k=-1)
    assert np.array_equal(result, expected)
    assert result.shape == (3, 3)

    source = tmp_path / "square.hdf5"
    output = tmp_path / "square_out.hdf5"
    _write_file(source, include_rgb=False, event_shape=(3, 7, 7, 3))
    plan = cli.build_file_plans(
        [(source, output)], ("event",), (1, 3), 0, False, crop_square=5
    )[0]
    assert plan.datasets[0].result_shape == (3, 5, 5, 3)
    cli.process_file(plan, (1, 3), 0, crop_square=5)
    with h5py.File(output, "r") as root:
        event = root[cli.DATASET_KEYS["event"]]
        assert event.shape == (3, 5, 5, 3)
        assert event.attrs[f"{cli.METADATA_PREFIX}center_square_side_px"] == 5


def test_crop_square_rejects_nonsquare_event_and_conflicting_cli(tmp_path):
    source = tmp_path / "nonsquare.hdf5"
    _write_file(source, include_rgb=False, event_shape=(3, 4, 7, 3))
    with pytest.raises(ValueError, match="square event image"):
        cli.build_file_plans(
            [(source, tmp_path / "out.hdf5")],
            ("event",),
            None,
            0,
            False,
            crop_square=3,
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--input",
                str(source),
                "--modality",
                "event",
                "--crop-x",
                "1",
                "--crop-square",
                "3",
            ]
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            ["--input", str(source), "--modality", "rgb", "--crop-square", "3"]
        )


def test_event_preview_selects_top_16_left_half_activity_frames(tmp_path):
    source = tmp_path / "activity.hdf5"
    frames = np.full((20, 4, 6, 3), 128, dtype=np.uint8)
    for index in range(20):
        # Original top rows map to the rotated view's left half.
        frames[index, 0, :, :] = 128 + index
    with h5py.File(source, "w") as root:
        root.create_dataset(cli.DATASET_KEYS["event"], data=frames)
    with h5py.File(source, "r") as root:
        ranked = cli._highest_left_half_event_activity_frames(
            root[cli.DATASET_KEYS["event"]], count=16
        )
    assert [index for index, _ in ranked] == list(range(19, 3, -1))


@pytest.mark.parametrize("shape", [(3, 4, 7), (3, 4, 7, 3)])
def test_rank3_and_rank4_hdf5_outputs_have_exact_width(tmp_path, shape):
    source = tmp_path / "source.hdf5"
    output = tmp_path / "output.hdf5"
    _write_file(source, rgb_shape=shape, include_event=False)
    plan = _single_plan(source, output, crop_x=2)
    cli.process_file(plan, None, 2)
    with h5py.File(output, "r") as root:
        assert root[cli.DATASET_KEYS["rgb"]].shape == (*shape[:2], shape[2] - 2, *shape[3:])


def test_both_modalities_use_their_neutral_values(tmp_path):
    source = tmp_path / "source.hdf5"
    output = tmp_path / "output.hdf5"
    _write_file(source)
    plan = _single_plan(source, output, modality="both", mask_x=(1, 1), crop_x=0)
    cli.process_file(plan, (1, 1), 0)
    with h5py.File(output, "r") as root:
        assert np.all(root[cli.DATASET_KEYS["rgb"]][:, :, :2] == 0)
        assert np.all(root[cli.DATASET_KEYS["event"]][:, :, :2] == 128)


def test_preserves_unrelated_objects_attributes_links_and_creation_properties(tmp_path):
    source = tmp_path / "source.hdf5"
    output = tmp_path / "output.hdf5"
    _write_file(source, include_event=False, compression="gzip")
    plan = _single_plan(source, output, mask_x=(0, 2), crop_x=1)
    cli.process_file(plan, (0, 2), 1)
    with h5py.File(output, "r") as root:
        assert root.attrs["root_note"] == "preserve me"
        assert root["observations"].attrs["group_note"] == 17
        assert np.array_equal(root["action"][:], np.arange(6, dtype=np.float32).reshape(3, 2))
        assert root["action"].attrs["units"] == "m"
        assert root["action_alias"].id == root["action"].id
        assert isinstance(root.get("soft_action", getlink=True), h5py.SoftLink)
        rgb = root[cli.DATASET_KEYS["rgb"]]
        assert rgb.compression == "gzip"
        assert rgb.chunks is not None
        assert rgb.attrs["camera"] == "top"
        assert rgb.attrs[f"{cli.METADATA_PREFIX}tool_version"] == cli.TOOL_VERSION


def test_directory_discovery_sorted_relative_outputs_and_exclusion(tmp_path):
    top = tmp_path / "dataset"
    out = top / "generated"
    _write_file(top / "z" / "episode_2.h5", include_event=False)
    _write_file(top / "a" / "episode_1.hdf5", include_event=False)
    _write_file(out / "old.hdf5", include_event=False)
    discovered = cli.discover_inputs(top, out)
    assert [path.relative_to(top).as_posix() for path in discovered] == [
        "a/episode_1.hdf5",
        "z/episode_2.h5",
    ]
    args = cli.parse_args(
        ["--top-dir", str(top), "--out-dir", str(out), "--modality", "rgb", "--crop-x", "1"]
    )
    _, _, mappings = cli.resolve_paths(args)
    assert [target.relative_to(out).as_posix() for _, target in mappings] == [
        "a/episode_1.hdf5",
        "z/episode_2.h5",
    ]
    plans = cli.build_file_plans(mappings, ("rgb",), None, 1, False)
    cli.process_all(plans, None, 1)
    assert (out / "a" / "episode_1.hdf5").is_file()
    assert (out / "z" / "episode_2.h5").is_file()
    assert cli.representative_file_indices(9, 3) == [0, 4, 8]


def test_rejects_invalid_parameters_missing_keys_and_unsupported_layouts(tmp_path):
    source = tmp_path / "source.hdf5"
    _write_file(source, include_rgb=False, include_event=True)
    with pytest.raises(ValueError, match="Missing required dataset"):
        _single_plan(source, tmp_path / "out.hdf5", modality="rgb")

    bad = tmp_path / "bad.hdf5"
    _write_file(bad, rgb_shape=(3, 4, 7, 4), include_event=False)
    with pytest.raises(ValueError, match="C=3"):
        _single_plan(bad, tmp_path / "bad_out.hdf5")

    valid = tmp_path / "valid.hdf5"
    _write_file(valid, include_event=False)
    with pytest.raises(ValueError, match="crop_x"):
        _single_plan(valid, tmp_path / "valid_out.hdf5", crop_x=6)
    with pytest.raises(ValueError, match="XTOP"):
        _single_plan(valid, tmp_path / "valid_out.hdf5", mask_x=(6, 0), crop_x=0)

    wrong_dtype = tmp_path / "wrong_dtype.hdf5"
    with h5py.File(wrong_dtype, "w") as root:
        root.create_dataset(
            cli.DATASET_KEYS["rgb"], data=np.zeros((2, 3, 4, 3), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="uint8"):
        _single_plan(wrong_dtype, tmp_path / "wrong_dtype_out.hdf5")


def test_xyt_event_channel_count_requires_and_accepts_exact_metadata(tmp_path):
    source = tmp_path / "xyt.hdf5"
    _write_file(source, include_rgb=False, event_shape=(3, 4, 7, 9))
    with pytest.raises(ValueError, match="event_representation"):
        _single_plan(source, tmp_path / "out.hdf5", modality="event")
    with h5py.File(source, "r+") as root:
        root.attrs["event_representation"] = "xyt_signed_voxel_v1"
        root.attrs["event_temporal_bins"] = 9
        root.attrs["image_channels"] = 9
        root.attrs["channels_per_visual_frame"] = 9
    plan = _single_plan(source, tmp_path / "out.hdf5", modality="event")
    assert plan.datasets[0].original_shape[-1] == 9


def test_refuses_overwrite_and_already_processed_dataset(tmp_path):
    source = tmp_path / "source.hdf5"
    output = tmp_path / "output.hdf5"
    _write_file(source, include_event=False)
    _write_file(output, include_event=False)
    with pytest.raises(ValueError, match="already exists"):
        _single_plan(source, output)
    with h5py.File(source, "r+") as root:
        root[cli.DATASET_KEYS["rgb"]].attrs[f"{cli.METADATA_PREFIX}tool_version"] = "old"
    with pytest.raises(ValueError, match="already processed"):
        _single_plan(source, tmp_path / "new.hdf5")


def test_rejects_multiple_hard_links_and_references_to_selected_dataset(tmp_path):
    hardlinked = tmp_path / "hardlinked.hdf5"
    _write_file(hardlinked, include_event=False)
    with h5py.File(hardlinked, "r+") as root:
        root["rgb_alias"] = root[cli.DATASET_KEYS["rgb"]]
    with pytest.raises(ValueError, match="multiple hard links"):
        _single_plan(hardlinked, tmp_path / "hardlinked_out.hdf5")

    referenced = tmp_path / "referenced.hdf5"
    _write_file(referenced, include_event=False)
    with h5py.File(referenced, "r+") as root:
        refs = root.create_dataset("references", shape=(1,), dtype=h5py.ref_dtype)
        refs[0] = root[cli.DATASET_KEYS["rgb"]].ref
    with pytest.raises(ValueError, match="contains a reference"):
        _single_plan(referenced, tmp_path / "referenced_out.hdf5")

    regioned = tmp_path / "regioned.hdf5"
    _write_file(regioned, include_event=False)
    with h5py.File(regioned, "r+") as root:
        refs = root.create_dataset("regions", shape=(1,), dtype=h5py.regionref_dtype)
        refs[0] = root[cli.DATASET_KEYS["rgb"]].regionref[0:1]
    with pytest.raises(ValueError, match="contains a reference"):
        _single_plan(regioned, tmp_path / "regioned_out.hdf5")


def test_noop_arguments_are_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args(["--input", "episode.hdf5", "--modality", "rgb", "--crop-x", "0"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--input", "episode.hdf5", "--modality", "rgb"])


def test_mask_line_width_cli_default_override_and_validation():
    base = ["--input", "episode.hdf5", "--modality", "rgb", "--mask-x", "1", "2"]
    assert cli.parse_args(base).mask_line_width == pytest.approx(0.1)
    assert cli.parse_args(base + ["--mask-line-width", "0.75"]).mask_line_width == pytest.approx(
        0.75
    )
    with pytest.raises(SystemExit):
        cli.parse_args(base + ["--mask-line-width", "0"])
    with pytest.raises(SystemExit):
        cli.parse_args(base + ["--mask-line-width", "nan"])


def test_aborted_confirmation_causes_no_writes(tmp_path, monkeypatch):
    source = tmp_path / "source.hdf5"
    output_dir = tmp_path / "new_parent"
    output = output_dir / "output.hdf5"
    _write_file(source, include_event=False)
    monkeypatch.setattr(builtins, "input", lambda _prompt: "Yes")
    result = cli.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--modality",
            "rgb",
            "--crop-x",
            "1",
            "--no-preview",
        ]
    )
    assert result == 1
    assert not output.exists()
    assert not output_dir.exists()


def test_preview_only_never_writes_or_prompts(tmp_path, monkeypatch):
    source = tmp_path / "source.hdf5"
    output = tmp_path / "output.hdf5"
    _write_file(source, include_event=False)

    def forbidden_input(_prompt):
        raise AssertionError("preview-only must not prompt")

    monkeypatch.setattr(builtins, "input", forbidden_input)
    result = cli.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--modality",
            "rgb",
            "--mask-x",
            "1",
            "2",
            "--no-preview",
            "--preview-only",
        ]
    )
    assert result == 0
    assert not output.exists()


def test_preview_is_saved_without_displaying_window(tmp_path, monkeypatch):
    source = tmp_path / "source.hdf5"
    _write_file(source, include_event=False)
    plan = _single_plan(source, tmp_path / "unused.hdf5", mask_x=(1, 2), crop_x=1)
    simulations = tmp_path / "crop_simulations"
    monkeypatch.setattr(cli, "SIMULATION_DIR", simulations)
    monkeypatch.setattr(
        cli.plt,
        "show",
        lambda: (_ for _ in ()).throw(AssertionError("interactive preview is forbidden")),
    )

    session_dir = cli.preview_plans([plan], 3, (1, 2), 1)

    assert session_dir.parent == simulations
    pngs = list(session_dir.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].stat().st_size > 0
    assert cli.plt.get_fignums() == []


def test_directory_failure_summary_keeps_completed_and_stops(tmp_path, monkeypatch, capsys):
    plans = [
        cli.FilePlan(Path(f"source_{index}"), tmp_path / f"out_{index}.hdf5", (), 0)
        for index in range(3)
    ]

    def fake_process(plan, _mask_x, _crop_x, _crop_square):
        if plan is plans[1]:
            raise RuntimeError("synthetic failure")
        plan.output.write_bytes(b"done")

    monkeypatch.setattr(cli, "process_file", fake_process)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        cli.process_all(plans, None, 1)
    output = capsys.readouterr().out
    assert "Completed: 1/3" in output
    assert f"failed: {plans[1].output}" in output
    assert f"not attempted: {plans[2].output}" in output
    assert plans[0].output.exists()
    assert not plans[2].output.exists()
