"""位移场插值工具。

将稀疏控制点的偏移量插值为密集位移场。
仅依赖 numpy + OpenCV，无 scipy 依赖。
"""

import cv2
import numpy as np


def grid_interp(
    xs: list[int],
    ys: list[int],
    vals: list[float],
    field_w: int,
    field_h: int,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """将稀疏控制点插值为低分辨率位移场，再上采样到原图尺寸。

    使用纯 numpy 的 IDW 插值（低分辨率网格上直接计算）。

    Args:
        xs: 控制点 x 坐标列表（像素坐标）。
        ys: 控制点 y 坐标列表（像素坐标）。
        vals: 对应偏移量。
        field_w: 低分辨率位移场宽度。
        field_h: 低分辨率位移场高度。
        img_w: 原图宽度。
        img_h: 原图高度。

    Returns:
        与原图同尺寸的位移场数组，shape (img_h, img_w)。
    """
    n_pts = len(xs)
    if n_pts == 0:
        return np.zeros((img_h, img_w), dtype=np.float32)

    if n_pts == 1:
        return np.full((img_h, img_w), vals[0], dtype=np.float32)

    # 低分辨率网格坐标
    fy = np.linspace(0, img_h - 1, field_h)
    fx = np.linspace(0, img_w - 1, field_w)
    gy, gx = np.meshgrid(fy, fx, indexing="ij")

    pts = np.column_stack([xs, ys]).astype(np.float32)
    vals_arr = np.array(vals, dtype=np.float32)

    # 对低分辨率网格上的每个点做 IDW 插值
    # 网格尺寸 field_h x field_w，通常 ~ 64x64 = 4096 个点
    # 控制点通常 20-200 个，暴力计算没问题
    low_res = np.zeros((field_h, field_w), dtype=np.float32)

    for r in range(field_h):
        for c in range(field_w):
            px, py = fx[c], fy[r]
            # 到所有控制点的距离
            dists = np.sqrt((pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2)

            min_dist = dists.min()
            if min_dist < 0.5:
                # 重合点，直接用该控制点的值
                idx = np.argmin(dists)
                low_res[r, c] = vals_arr[idx]
                continue

            # 取最近的 k 个点做 IDW
            k = min(8, n_pts)
            nearest_idx = np.argpartition(dists, k)[:k]
            nearest_dists = dists[nearest_idx]
            nearest_vals = vals_arr[nearest_idx]

            # IDW: weight = 1/d²
            weights = 1.0 / (nearest_dists * nearest_dists + 1e-10)
            weights /= weights.sum()
            low_res[r, c] = np.sum(weights * nearest_vals)

    # 上采样到原图尺寸
    full_res = cv2.resize(low_res, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    return full_res
