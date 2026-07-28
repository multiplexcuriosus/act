#!/usr/bin/env python3

import argparse
import glob
import os

import h5py
import numpy as np


def expand_paths(patterns):
    paths = []

    for pattern in patterns:
        matches = glob.glob(pattern)
        candidates = matches if matches else [pattern]

        for candidate in candidates:
            if os.path.isdir(candidate):
                paths.extend(glob.glob(os.path.join(candidate, "*.hdf5")))
                paths.extend(glob.glob(os.path.join(candidate, "*.h5")))
            else:
                paths.append(candidate)

    return sorted(set(paths))


def format_attribute_value(value):
    """Produce a readable representation of an HDF5 attribute."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):
        return np.array2string(
            value,
            threshold=20,
            edgeitems=5,
            separator=", ",
        )

    return repr(value)


def print_attributes(obj, indent):
    for key, value in obj.attrs.items():
        print(
            f"{indent}@{key} = "
            f"{format_attribute_value(value)}"
        )


def print_hdf5_structure(path, hdf5_file):
    """
    Print all groups, datasets, shapes, dtypes, compression settings,
    and attributes without loading the complete dataset contents.
    """
    print("\n" + "=" * 100)
    print(f"HDF5 STRUCTURE: {path}")
    print("=" * 100)

    print("/  [File]")
    print_attributes(hdf5_file, indent="  ")

    def visitor(name, obj):
        depth = name.count("/") + 1
        indent = "  " * depth
        basename = name.rsplit("/", 1)[-1]

        if isinstance(obj, h5py.Group):
            print(f"{indent}{basename}/  [Group]")
            print_attributes(obj, indent + "  ")
            return

        if isinstance(obj, h5py.Dataset):
            details = [
                f"shape={obj.shape}",
                f"dtype={obj.dtype}",
            ]

            if obj.maxshape != obj.shape:
                details.append(f"maxshape={obj.maxshape}")

            if obj.chunks is not None:
                details.append(f"chunks={obj.chunks}")

            if obj.compression is not None:
                details.append(f"compression={obj.compression}")

                if obj.compression_opts is not None:
                    details.append(
                        f"compression_opts={obj.compression_opts}"
                    )

            if obj.shuffle:
                details.append("shuffle=True")

            if obj.fletcher32:
                details.append("fletcher32=True")

            print(
                f"{indent}{basename}  [Dataset: "
                + ", ".join(details)
                + "]"
            )
            print_attributes(obj, indent + "  ")

    hdf5_file.visititems(visitor)
    print("=" * 100)


def print_counts(name, values, zero_tol):
    negative = int(np.sum(values < -zero_tol))
    zero = int(np.sum(np.abs(values) <= zero_tol))
    positive = int(np.sum(values > zero_tol))
    total = len(values)

    print(f"\n{name}:")
    print(f"  total: {total}")

    if total == 0:
        return

    print(f"  negative:  {negative:6d} ({negative / total:.3%})")
    print(f"  near zero: {zero:6d} ({zero / total:.3%})")
    print(f"  positive:  {positive:6d} ({positive / total:.3%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print HDF5 structure and inspect interception-action balance. "
            "Supports current (T, 1) delta-s actions and legacy "
            "(T, 2) action-plus-commanded-flag files."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="HDF5 files, dataset directories, or glob patterns.",
    )
    parser.add_argument(
        "--zero-tol",
        type=float,
        default=1e-6,
        help=(
            "Absolute delta-s values at or below this threshold are "
            "classified as near zero. Default: 1e-6."
        ),
    )
    parser.add_argument(
        "--structure",
        choices=("none", "first", "all"),
        default="first",
        help=(
            "Print no structures, the complete structure of the first "
            "file, or the structure of every file. Default: first."
        ),
    )
    args = parser.parse_args()

    paths = expand_paths(args.paths)
    if not paths:
        raise RuntimeError("No HDF5 files found.")

    missing_paths = [path for path in paths if not os.path.isfile(path)]
    if missing_paths:
        for path in missing_paths:
            print(f"[WARN] file not found: {path}")

    paths = [path for path in paths if os.path.isfile(path)]
    if not paths:
        raise RuntimeError("None of the resolved paths are valid files.")

    delta_s_values = []
    episode_mean_delta_s = []
    episode_net_delta_s = []
    legacy_flags = []

    current_files = 0
    legacy_files = 0
    skipped_files = 0

    for file_index, path in enumerate(paths):
        try:
            with h5py.File(path, "r") as f:
                if (
                    args.structure == "all"
                    or (args.structure == "first" and file_index == 0)
                ):
                    print_hdf5_structure(path, f)

                if "action" not in f:
                    print(f"[WARN] skipping action analysis for {path}: missing /action")
                    skipped_files += 1
                    continue

                action = np.asarray(f["action"][:])

        except OSError as exc:
            print(f"[WARN] unable to open {path}: {exc}")
            skipped_files += 1
            continue

        if action.ndim != 2:
            print(
                f"[WARN] skipping action analysis for {path}: "
                f"/action shape is {action.shape}, expected (T, D)"
            )
            skipped_files += 1
            continue

        if action.shape[0] == 0:
            print(f"[WARN] skipping {path}: /action is empty")
            skipped_files += 1
            continue

        if action.shape[1] == 1:
            # Current schema: /action[:, 0] contains signed delta_s.
            episode_delta_s = action[:, 0]

            if not np.all(np.isfinite(episode_delta_s)):
                print(f"[WARN] skipping {path}: non-finite delta_s values")
                skipped_files += 1
                continue

            delta_s_values.append(episode_delta_s)
            episode_mean_delta_s.append(np.mean(episode_delta_s))
            episode_net_delta_s.append(np.sum(episode_delta_s))
            current_files += 1

        elif action.shape[1] == 2:
            # Legacy schema: /action[:, 1] is the commanded flag.
            flags = action[:, 1]

            if not np.all(np.isfinite(flags)):
                print(f"[WARN] skipping {path}: non-finite commanded flags")
                skipped_files += 1
                continue

            legacy_flags.append(flags)
            legacy_files += 1

        else:
            print(
                f"[WARN] skipping action analysis for {path}: "
                f"unsupported /action shape {action.shape}"
            )
            skipped_files += 1

    print("\n" + "=" * 100)
    print("DATASET SUMMARY")
    print("=" * 100)
    print(f"files found:                  {len(paths)}")
    print(f"current delta-s files:        {current_files}")
    print(f"legacy commanded-flag files: {legacy_files}")
    print(f"skipped/unsupported files:    {skipped_files}")

    if delta_s_values:
        delta_s = np.concatenate(delta_s_values)
        episode_means = np.asarray(episode_mean_delta_s)
        episode_nets = np.asarray(episode_net_delta_s)

        print("\nCurrent schema: /action shape (T, 1)")
        print(f"action samples: {len(delta_s)}")
        print(
            "delta_s min/mean/max: "
            f"{delta_s.min():+.8f} / "
            f"{delta_s.mean():+.8f} / "
            f"{delta_s.max():+.8f}"
        )
        print(
            "|delta_s| min/mean/max: "
            f"{np.abs(delta_s).min():.8f} / "
            f"{np.abs(delta_s).mean():.8f} / "
            f"{np.abs(delta_s).max():.8f}"
        )

        print_counts(
            "Per-step delta_s balance",
            delta_s,
            args.zero_tol,
        )
        print_counts(
            "Per-episode mean delta_s balance",
            episode_means,
            args.zero_tol,
        )
        print_counts(
            "Per-episode summed delta_s balance",
            episode_nets,
            args.zero_tol,
        )

    if legacy_flags:
        flags = np.concatenate(legacy_flags)
        zero_flags = np.isclose(flags, 0.0)
        one_flags = np.isclose(flags, 1.0)
        valid_binary = zero_flags | one_flags

        print("\nLegacy schema: /action shape (T, 2)")
        print("unique commanded flags:", np.unique(flags))
        print("uncommanded:", int(np.sum(zero_flags)))
        print("commanded:", int(np.sum(one_flags)))
        print("commanded fraction:", float(np.mean(one_flags)))

        if not np.all(valid_binary):
            print(
                "[WARN] commanded-flag column contains values other than 0 or 1"
            )

    if not delta_s_values and not legacy_flags:
        raise RuntimeError("No compatible /action datasets were found.")


if __name__ == "__main__":
    main()