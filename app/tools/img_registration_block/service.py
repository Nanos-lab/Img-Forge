"""分块相位相关配准 + 位移场插值。

将重叠区划分为 N×M 网格，每块独立用相位相关算偏移，
插值生成平滑位移场，逐像素 warp 校正。

解决不同区域偏移不一致的问题（如建筑 vs 地面）。
"""

import cv2
import numpy as np

from app.core.config import REGISTRATION_RESPONSE_THRESHOLD, REGISTRATION_MAX_SHIFT


# ============================================================
#  内部辅助
# ============================================================


def _to_grayscale(patch: np.ndarray) -> np.ndarray:
    """将多波段影像转为单波段灰度 float32。"""
    if patch.ndim == 2:
        return patch.astype(np.float32)
    if patch.shape[0] >= 3:
        gray = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]
    else:
        gray = patch[0]
    return gray.astype(np.float32)


def _block_phase_correlate(
    ref_block: np.ndarray,
    src_block: np.ndarray,
) -> tuple[float, float, float]:
    """对单个小块做相位相关，返回 (d_row, d_col, response)。"""
    h, w = ref_block.shape

    if h < 16 or w < 16:
        return 0.0, 0.0, 0.0

    window = cv2.createHanningWindow((w, h), cv2.CV_32F)

    try:
        (dx, dy), response = cv2.phaseCorrelate(ref_block, src_block, window=window)
    except cv2.error:
        return 0.0, 0.0, 0.0

    if response < REGISTRATION_RESPONSE_THRESHOLD:
        return 0.0, 0.0, response

    shift = np.sqrt(dx * dx + dy * dy)
    if shift > REGISTRATION_MAX_SHIFT:
        return 0.0, 0.0, response

    return float(-dy), float(-dx), float(response)


# ============================================================
#  对外接口
# ============================================================


def phase_correlate(
    ref_patch: np.ndarray,
    src_patch: np.ndarray,
) -> tuple[float, float, float]:
    """兼容 img_registration.phase_correlate 的同名接口。

    内部采用分块匹配，返回所有块偏移的中位数作为全局偏移。

    Args:
        ref_patch: 参考影像块，shape (bands, H, W) 或 (H, W)。
        src_patch: 待配准影像块，shape 同 ref_patch。

    Returns:
        (delta_row, delta_col, response):
        - delta_row: 行方向偏移（亚像素）
        - delta_col: 列方向偏移（亚像素）
        - response:  所有块 response 的中位数
    """
    ref_gray = _to_grayscale(ref_patch)
    src_gray = _to_grayscale(src_patch)

    h, w = ref_gray.shape

    # 网格参数
    block_size = min(64, h // 2, w // 2)
    block_size = max(block_size, 32)
    grid_step = block_size // 2

    rows_list = list(range(0, h - block_size + 1, grid_step))
    cols_list = list(range(0, w - block_size + 1, grid_step))

    if not rows_list or not cols_list:
        return 0.0, 0.0, 0.0

    d_rows, d_cols, responses = [], [], []

    for r0 in rows_list:
        for c0 in cols_list:
            r1 = r0 + block_size
            c1 = c0 + block_size
            dr, dc, resp = _block_phase_correlate(
                ref_gray[r0:r1, c0:c1],
                src_gray[r0:r1, c0:c1],
            )
            if resp >= REGISTRATION_RESPONSE_THRESHOLD:
                d_rows.append(dr)
                d_cols.append(dc)
                responses.append(resp)

    if not d_rows:
        return 0.0, 0.0, 0.0

    # 用中位数作为全局偏移，对异常值鲁棒
    med_dr = float(np.median(d_rows))
    med_dc = float(np.median(d_cols))
    med_resp = float(np.median(responses))

    return med_dr, med_dc, med_resp


def block_warp(
    ref_patch: np.ndarray,
    src_patch: np.ndarray,
    block_size: int = 64,
    grid_step: int | None = None,
) -> tuple[np.ndarray, float]:
    """分块匹配 + 位移场插值 + warp 校正。

    先对网格逐块计算偏移量，插值出平滑位移场，
    再对 src_patch 逐像素 warp，校正局部形变。

    Args:
        ref_patch: 参考影像块，shape (bands, H, W) 或 (H, W)。
        src_patch: 待配准影像块，shape 同 ref_patch。
        block_size: 每个匹配块的尺寸（像素，默认 64）。
        grid_step: 网格步长（默认 block_size/2）。

    Returns:
        (warped_src, avg_response):
        - warped_src: 配准后的源影像，shape 与 src_patch 相同
        - avg_response: 所有块的平均响应值
    """
    ref_gray = _to_grayscale(ref_patch)
    src_gray = _to_grayscale(src_patch)

    h, w = ref_gray.shape

    if grid_step is None:
        grid_step = block_size // 2
    grid_step = max(grid_step, 8)

    # ----------------------------------------------------------
    # 1. 网格逐块匹配，收集控制点
    # ----------------------------------------------------------
    rows_list = list(range(0, h - block_size + 1, grid_step))
    cols_list = list(range(0, w - block_size + 1, grid_step))

    if not rows_list or not cols_list:
        # 图像太小，退化为全局相位相关
        dr, dc, resp = phase_correlate(ref_patch, src_patch)
        mat = np.float32([[1, 0, dc], [0, 1, dr]])
        warped = cv2.warpAffine(
            src_gray, mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # 恢复多波段
        if src_patch.ndim == 3:
            bands = src_patch.shape[0]
            result = np.zeros_like(src_patch)
            for b in range(bands):
                result[b] = cv2.warpAffine(
                    src_patch[b].astype(np.float32), mat, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            return result, resp
        else:
            return warped.astype(src_patch.dtype), resp

    # 控制点网格
    grid_ys, grid_xs = [], []
    grid_dy, grid_dx = [], []
    responses = []

    for r0 in rows_list:
        for c0 in cols_list:
            r1 = r0 + block_size
            c1 = c0 + block_size
            dr, dc, resp = _block_phase_correlate(
                ref_gray[r0:r1, c0:c1],
                src_gray[r0:r1, c0:c1],
            )
            if resp >= REGISTRATION_RESPONSE_THRESHOLD:
                grid_ys.append(r0 + block_size // 2)
                grid_xs.append(c0 + block_size // 2)
                grid_dy.append(dr)
                grid_dx.append(dc)
                responses.append(resp)

    if len(grid_ys) < 4:
        # 控制点太少，退化为全局平移
        dr, dc, resp = phase_correlate(ref_patch, src_patch)
        mat = np.float32([[1, 0, dc], [0, 1, dr]])
        if src_patch.ndim == 3:
            bands = src_patch.shape[0]
            result = np.zeros_like(src_patch)
            for b in range(bands):
                result[b] = cv2.warpAffine(
                    src_patch[b].astype(np.float32), mat, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            return result, float(np.median(responses)) if responses else resp
        else:
            warped = cv2.warpAffine(
                src_gray, mat, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return warped.astype(src_patch.dtype), float(np.median(responses)) if responses else resp

    # ----------------------------------------------------------
    # 2. 插值生成平滑位移场
    # ----------------------------------------------------------
    # 用网格控制点插值到每个像素
    # 使用径向基函数（RBF）或网格插值
    # 为提高计算效率，先生成低分辨率位移场再上采样

    scale = 4  # 位移场降采样因子
    field_h = max(8, h // scale)
    field_w = max(8, w // scale)

    # 像素坐标转位移场网格坐标
    points = np.column_stack([grid_xs, grid_ys]).astype(np.float32)

    # 使用 scipy 不可用，用 OpenCV 的 remap + 插值来生成位移场
    # 方案：对稀疏控制点做 Delaunay 三角剖分 + 线性插值
    # OpenCV 没有直接的 RBF 插值，但可以用 Subdiv2D

    # 使用 Delaunay 三角剖分 + 线性插值（通过 OpenCV 的 Subdiv2D）
    # 或直接用 IDW 插值到低分辨率网格再上采样
    from app.tools.img_registration_block._interp import grid_interp

    field_dx = grid_interp(grid_xs, grid_ys, grid_dx, field_w, field_h, w, h)
    field_dy = grid_interp(grid_xs, grid_ys, grid_dy, field_w, field_h, w, h)

    # ----------------------------------------------------------
    # 3. 对每个像素应用位移场
    # ----------------------------------------------------------
    # 构建 remap 映射：map_x, map_y
    map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = map_x + field_dx  # 源像素的列位置
    map_y = map_y + field_dy  # 源像素的行位置

    if src_patch.ndim == 3:
        bands = src_patch.shape[0]
        result = np.zeros_like(src_patch)
        for b in range(bands):
            band_warped = cv2.remap(
                src_patch[b].astype(np.float32),
                map_x, map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            result[b] = band_warped.astype(src_patch.dtype)
    else:
        result = cv2.remap(
            src_gray, map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(src_patch.dtype)

    avg_resp = float(np.median(responses)) if responses else 0.0

    return result, avg_resp
