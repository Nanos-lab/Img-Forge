"""
TIF sliding-window tiler.

Splits large remote sensing TIF images into fixed-size tiles for YOLO inference,
using rasterio's windowed reading to avoid loading the entire image into memory.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass
class TileInfo:
    """Metadata for a single tile extracted from the source TIF."""

    data: np.ndarray  # shape: (H, W, 3) — RGB image data
    col_off: int      # pixel offset of this tile from the left edge of the full image
    row_off: int      # pixel offset of this tile from the top edge of the full image
    tile_width: int   # actual width of this tile (may differ from tile_size at edges)
    tile_height: int  # actual height of this tile


def compute_windows(
    img_width: int,
    img_height: int,
    tile_size: int = 640,
    overlap_ratio: float = 0.15,
) -> List[Window]:
    """
    Compute all sliding-window positions covering the full image.

    Args:
        img_width: full image width in pixels
        img_height: full image height in pixels
        tile_size: side length of each square tile
        overlap_ratio: overlap between adjacent tiles (0.0 ~ 1.0)

    Returns:
        List of rasterio Window objects, one per tile position.
    """
    stride = int(tile_size * (1.0 - overlap_ratio))
    stride = max(stride, 1)  # guard against degenerate overlap

    windows = []
    for row_start in range(0, img_height, stride):
        for col_start in range(0, img_width, stride):
            # Determine actual tile dimensions (handle edge tiles)
            w = min(tile_size, img_width - col_start)
            h = min(tile_size, img_height - row_start)

            # Skip tiles that are too small to be useful
            if w < tile_size * 0.3 or h < tile_size * 0.3:
                continue

            windows.append(Window(col_start, row_start, w, h))

    return windows


def pad_tile_to_size(tile: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Pad a tile to exactly target_size × target_size using reflection padding.

    Args:
        tile: numpy array of shape (H, W, C)
        target_size: desired side length

    Returns:
        Padded array of shape (target_size, target_size, C)
    """
    h, w = tile.shape[:2]
    if h == target_size and w == target_size:
        return tile

    pad_bottom = target_size - h
    pad_right = target_size - w

    if tile.ndim == 3:
        return np.pad(
            tile,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="reflect",
        )
    else:
        return np.pad(
            tile,
            ((0, pad_bottom), (0, pad_right)),
            mode="reflect",
        )


def extract_tile(src: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    """
    Read a single window from the TIF and return an RGB numpy array.

    - Reads only the first 3 bands (RGB). If the image has < 3 bands, repeats
      the available band(s).
    - Converts rasterio's (C, H, W) layout to numpy's (H, W, C).

    Args:
        src: opened rasterio dataset
        window: the window to read

    Returns:
        numpy array of shape (H, W, 3), dtype uint8
    """
    num_bands = src.count

    if num_bands >= 3:
        data = src.read([1, 2, 3], window=window)  # (3, H, W)
    elif num_bands == 2:
        # Two bands: use first as R, second as G, zeros as B
        arr = src.read([1, 2], window=window)  # (2, H, W)
        data = np.zeros((3, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
        data[0] = arr[0]
        data[1] = arr[1]
    else:
        # Single band: repeat as RGB
        arr = src.read(1, window=window)  # (H, W)
        data = np.stack([arr, arr, arr], axis=0)  # (3, H, W)

    # (C, H, W) → (H, W, C)
    data = np.transpose(data, (1, 2, 0))

    # Normalize to uint8 if necessary
    if data.dtype == np.uint16:
        data = (data / 256).astype(np.uint8)
    elif data.dtype != np.uint8:
        # Min-max normalize for other dtypes
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)

    return data


def generate_tiles(
    src: rasterio.io.DatasetReader,
    tile_size: int = 640,
    overlap_ratio: float = 0.15,
) -> List[TileInfo]:
    """
    Generate all tiles from a TIF image for inference.

    Performs windowed reads — the full image is never loaded into memory at once.

    Args:
        src: opened rasterio dataset
        tile_size: side length of each square tile
        overlap_ratio: overlap between adjacent tiles

    Returns:
        List of TileInfo objects ready for inference.
    """
    img_width = src.width
    img_height = src.height
    windows = compute_windows(img_width, img_height, tile_size, overlap_ratio)

    tiles = []
    for window in windows:
        data = extract_tile(src, window)
        data = pad_tile_to_size(data, tile_size)
        tiles.append(
            TileInfo(
                data=data,
                col_off=int(window.col_off),
                row_off=int(window.row_off),
                tile_width=int(window.width),
                tile_height=int(window.height),
            )
        )

    return tiles


def get_image_transform_info(
    src: rasterio.io.DatasetReader,
) -> dict:
    """
    Extract coordinate transform metadata from the TIF.

    Returns a dict with transform, crs, width, and height — used later
    for pixel-to-geo coordinate conversion.
    """
    return {
        "transform": src.transform,
        "crs": src.crs,
        "width": src.width,
        "height": src.height,
    }
