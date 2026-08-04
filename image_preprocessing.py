"""Reusable, HDF5-independent image masking and left-cropping helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


MaskX = Optional[Tuple[int, int]]


def validate_image_transform(
    height: int,
    width: int,
    *,
    mask_x: MaskX,
    crop_x: int,
) -> None:
    """Validate transform coordinates against an image's original geometry."""
    height = int(height)
    width = int(width)
    crop_x = int(crop_x)
    if height <= 0 or width <= 0:
        raise ValueError(f"Image height and width must be positive, got {(height, width)}")
    if not 0 <= crop_x < width:
        raise ValueError(f"crop_x must satisfy 0 <= L < W={width}, got {crop_x}")
    if mask_x is not None:
        if len(mask_x) != 2:
            raise ValueError(f"mask_x must contain XTOP and XBOTTOM, got {mask_x!r}")
        x_top, x_bottom = (int(mask_x[0]), int(mask_x[1]))
        if not 0 <= x_top < width:
            raise ValueError(f"XTOP must satisfy 0 <= XTOP < W={width}, got {x_top}")
        if not 0 <= x_bottom < width:
            raise ValueError(
                f"XBOTTOM must satisfy 0 <= XBOTTOM < W={width}, got {x_bottom}"
            )


def mask_and_left_crop_image(
    image: np.ndarray,
    *,
    mask_x: MaskX = None,
    crop_x: int = 0,
    fill_value: int,
) -> np.ndarray:
    """Mask polygon O-A-B-C, then remove ``crop_x`` columns from the left.

    ``image`` must be a uint8 ``(H, W)`` or ``(H, W, C)`` array. Coordinates
    refer to this input image. For a one-row image the boundary is ``XTOP``.
    """
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"Image dtype must be uint8, got {array.dtype}")
    if array.ndim not in (2, 3):
        raise ValueError(f"Image must have shape (H,W) or (H,W,C), got {array.shape}")
    if array.ndim == 3 and array.shape[2] <= 0:
        raise ValueError(f"Image channel count must be positive, got {array.shape}")

    height, width = int(array.shape[0]), int(array.shape[1])
    validate_image_transform(height, width, mask_x=mask_x, crop_x=crop_x)
    fill_value = int(fill_value)
    if not 0 <= fill_value <= 255:
        raise ValueError(f"fill_value must satisfy 0 <= value <= 255, got {fill_value}")

    transformed = array.copy()
    if mask_x is not None:
        x_top, x_bottom = (int(mask_x[0]), int(mask_x[1]))
        if height == 1:
            boundaries = np.asarray([float(x_top)], dtype=np.float64)
        else:
            rows = np.arange(height, dtype=np.float64)
            boundaries = x_top + (x_bottom - x_top) * rows / float(height - 1)
        mask = np.arange(width, dtype=np.float64)[None, :] <= boundaries[:, None]
        if transformed.ndim == 3:
            transformed[mask, :] = fill_value
        else:
            transformed[mask] = fill_value

    return transformed[:, int(crop_x) :]


def mask_and_left_crop_rotated_event_image(
    image: np.ndarray,
    *,
    mask_x: MaskX = None,
    crop_x: int = 0,
    fill_value: int = 128,
) -> np.ndarray:
    """Apply the transform in a 90-degree-CCW event debug view.

    The result is rotated clockwise back into the stored orientation. A left
    crop in the rotated view therefore removes top rows in stored coordinates.
    """
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"Image must have shape (H,W) or (H,W,C), got {array.shape}")
    rotated = np.rot90(array, k=1, axes=(0, 1))
    transformed = mask_and_left_crop_image(
        rotated,
        mask_x=mask_x,
        crop_x=crop_x,
        fill_value=fill_value,
    )
    return np.rot90(transformed, k=-1, axes=(0, 1))


def mask_and_center_crop_square_rotated_event_image(
    image: np.ndarray,
    *,
    mask_x: MaskX = None,
    square_side: int,
    fill_value: int = 128,
) -> np.ndarray:
    """Mask in the 90-degree-CCW event view, then keep its centered square."""
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"Image must have shape (H,W) or (H,W,C), got {array.shape}")
    rotated = np.rot90(array, k=1, axes=(0, 1))
    height, width = rotated.shape[:2]
    if height != width:
        raise ValueError(
            f"Centered square crop requires a square event image, got {(height, width)}"
        )
    square_side = int(square_side)
    if not 0 < square_side <= width:
        raise ValueError(
            f"square_side must satisfy 0 < N <= image side {width}, got {square_side}"
        )
    masked = mask_and_left_crop_image(
        rotated,
        mask_x=mask_x,
        crop_x=0,
        fill_value=fill_value,
    )
    start = (width - square_side) // 2
    cropped = masked[start : start + square_side, start : start + square_side]
    return np.rot90(cropped, k=-1, axes=(0, 1))
