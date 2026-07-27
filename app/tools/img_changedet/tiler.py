"""分块与拼接 —— 将对齐后的影像对切分为图块，推理后拼接回完整概率图。

支持 overlap 和线性羽化混合，消除拼接缝隙。
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Tile:
    """单个图块。"""
    old_patch: np.ndarray   # (tile_size, tile_size, 3) uint8
    new_patch: np.ndarray   # (tile_size, tile_size, 3) uint8
    row: int
    col: int


@dataclass
class TileSet:
    """图块集合及拼接元数据。"""
    tiles: list[Tile]
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    tile_size: int
    grid_rows: int
    grid_cols: int
    overlap: int


def tile_pair(
    old_img: np.ndarray,
    new_img: np.ndarray,
    tile_size: int = 256,
    overlap: int = 8,
) -> TileSet:
    """将对齐后的影像对切分为图块网格。

    Args:
        old_img: 变化前影像 (H, W, C) uint8。
        new_img: 变化后影像 (H, W, C) uint8。
        tile_size: 图块边长（像素）。
        overlap: 相邻图块间的重叠像素数。

    Returns:
        TileSet。
    """
    h, w = old_img.shape[:2]
    stride = tile_size - overlap

    grid_rows = max(1, int(np.ceil((h - overlap) / stride)))
    grid_cols = max(1, int(np.ceil((w - overlap) / stride)))

    padded_h = (grid_rows - 1) * stride + tile_size
    padded_w = (grid_cols - 1) * stride + tile_size
    pad_h = padded_h - h
    pad_w = padded_w - w

    if pad_h > 0 or pad_w > 0:
        pad_spec = ((0, pad_h), (0, pad_w), (0, 0))
        old_pad = np.pad(old_img, pad_spec, mode='constant', constant_values=0)
        new_pad = np.pad(new_img, pad_spec, mode='constant', constant_values=0)
    else:
        old_pad = old_img
        new_pad = new_img

    tiles = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = row * stride
            x0 = col * stride
            tiles.append(Tile(
                old_patch=old_pad[y0:y0 + tile_size, x0:x0 + tile_size],
                new_patch=new_pad[y0:y0 + tile_size, x0:x0 + tile_size],
                row=row, col=col,
            ))

    return TileSet(
        tiles=tiles,
        original_shape=(h, w),
        padded_shape=(padded_h, padded_w),
        tile_size=tile_size,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        overlap=overlap,
    )


def _build_blend_mask(tile_size: int, overlap: int) -> np.ndarray:
    """构建线性羽化混合掩膜。

    边缘 overlap 范围内从 0 线性过渡到 1，
    四个方向取逐元素最小值。
    """
    if overlap <= 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)

    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)

    top = np.ones((tile_size, tile_size), dtype=np.float32)
    top[:overlap, :] = ramp[:, np.newaxis]

    bottom = np.ones((tile_size, tile_size), dtype=np.float32)
    bottom[tile_size - overlap:, :] = ramp[::-1, np.newaxis]

    left = np.ones((tile_size, tile_size), dtype=np.float32)
    left[:, :overlap] = ramp[np.newaxis, :]

    right = np.ones((tile_size, tile_size), dtype=np.float32)
    right[:, tile_size - overlap:] = ramp[::-1][np.newaxis, :]

    return np.minimum(np.minimum(np.minimum(top, bottom), left), right)


def stitch_probability_map(
    patches: list[np.ndarray],
    tile_set: TileSet,
) -> np.ndarray:
    """将图块概率图拼接回完整概率图。

    Args:
        patches: 每个图块的概率图列表 (tile_size, tile_size) float32。
        tile_set: 图块集合及元数据。

    Returns:
        完整概率图 (H, W) float32，已裁剪填充区域。
    """
    ts = tile_set.tile_size
    stride = ts - tile_set.overlap
    padded_h, padded_w = tile_set.padded_shape

    canvas = np.zeros((padded_h, padded_w), dtype=np.float32)
    weights = np.zeros((padded_h, padded_w), dtype=np.float32)
    mask = _build_blend_mask(ts, tile_set.overlap)

    for tile, patch in zip(tile_set.tiles, patches):
        y0 = tile.row * stride
        x0 = tile.col * stride
        canvas[y0:y0 + ts, x0:x0 + ts] += patch * mask
        weights[y0:y0 + ts, x0:x0 + ts] += mask

    # 归一化
    weights = np.maximum(weights, 1e-10)
    canvas /= weights

    # 裁剪回原始尺寸
    orig_h, orig_w = tile_set.original_shape
    return canvas[:orig_h, :orig_w]
