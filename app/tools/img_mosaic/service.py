"""影像拼接服务层 —— 自定义地理坐标拼接 + 相位相关精配准。

核心逻辑:
1. 打开多个 TIFF 文件，校验 CRS 一致性
2. 计算所有影像的并集地理范围
3. 以第一张影像的分辨率为基准，创建输出画布数组
4. 按输入顺序逐张将影像贴入画布（重叠区域后覆盖前）
5. 贴入前对第 2 张及之后的影像进行相位相关精配准，校正卫星定位误差

精配准机制:
- 对每张新影像，提取其与**画布上已有数据区域**（布尔掩膜标记）的真正重叠区
- 调用 img_registration 模块的 phase_correlate 计算亚像素偏移
- 响应值低于阈值或偏移量过大时自动回退到原始位置
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.core.config import (
    MOSAIC_OUTPUT_SUFFIX,
    OUTPUT_EXTENSION,
    REGISTRATION_ENABLED,
    REGISTRATION_RESPONSE_THRESHOLD,
    REGISTRATION_MAX_SHIFT,
    REGISTRATION_MIN_OVERLAP,
)
from app.core.exceptions import MosaicError
from app.tools.img_registration import phase_correlate


# ============================================================
#  内部函数
# ============================================================

def _validate_crs(datasets: list[rasterio.DatasetReader]) -> None:
    """检查所有影像的 CRS 是否一致。

    Args:
        datasets: 已打开的 rasterio 数据集列表。

    Raises:
        MosaicError: 任一影像缺少 CRS 或 CRS 不一致。
    """
    for ds in datasets:
        if ds.crs is None:
            raise MosaicError(
                f"影像 '{Path(ds.name).name}' 缺少地理参考信息（CRS）"
            )

    ref_crs = datasets[0].crs
    for ds in datasets[1:]:
        if ds.crs != ref_crs:
            raise MosaicError(
                f"影像 CRS 不一致: '{Path(ds.name).name}' 的 CRS={ds.crs}，"
                f"参考影像 CRS={ref_crs}"
            )


def _union_bounds(datasets: list[rasterio.DatasetReader]) -> tuple[float, float, float, float]:
    """计算所有影像地理范围的并集。

    Args:
        datasets: 已打开的 rasterio 数据集列表。

    Returns:
        (left, bottom, right, top) 并集边界。
    """
    left   = min(ds.bounds.left   for ds in datasets)
    bottom = min(ds.bounds.bottom for ds in datasets)
    right  = max(ds.bounds.right  for ds in datasets)
    top    = max(ds.bounds.top    for ds in datasets)
    return left, bottom, right, top


def _paste_image(
    canvas: np.ndarray,
    data: np.ndarray,
    row_off: int,
    col_off: int,
) -> None:
    """将图像数据贴入画布指定位置。

    自动处理目标区域超出画布边界的情况（裁剪），
    同时也处理源数据中因偏移为负需要跳过头部像素的情况。

    Args:
        canvas:  输出画布数组，shape (bands, H, W)。
        data:    待贴入图像数组，shape (bands, h, w)。
        row_off: 图像左上角在画布中的行偏移（可为负）。
        col_off: 图像左上角在画布中的列偏移（可为负）。
    """
    out_h, out_w = canvas.shape[1], canvas.shape[2]
    h, w = data.shape[1], data.shape[2]

    # 计算源数据的有效贴入范围（因偏移为负而可能需要跳过头部）
    src_r0 = max(0, -row_off)
    src_c0 = max(0, -col_off)
    src_r1 = min(h, out_h - row_off)
    src_c1 = min(w, out_w - col_off)

    # 计算画布上的对应目标范围
    dst_r0 = max(0, row_off)
    dst_c0 = max(0, col_off)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)

    if dst_r1 > dst_r0 and dst_c1 > dst_c0:
        canvas[:, dst_r0:dst_r1, dst_c0:dst_c1] = data[:, src_r0:src_r1, src_c0:src_c1]


# ============================================================
#  精配准（调用 img_registration 模块）
# ============================================================


def _compute_registration_shift(
    canvas: np.ndarray,
    data: np.ndarray,
    row_off: int,
    col_off: int,
    filled_mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """计算新影像与画布已贴入区域之间的亚像素偏移。

    通过布尔掩膜精确提取重叠区，调用 img_registration.phase_correlate 计算。

    Args:
        canvas:     当前画布数组，shape (bands, H, W)。
        data:       待贴入影像数组，shape (bands, h, w)。
        row_off:    基于地理坐标的行偏移（画布坐标系）。
        col_off:    基于地理坐标的列偏移（画布坐标系）。
        filled_mask: 画布上已有数据的布尔掩膜，shape (H, W)，
            True=已填充，未传则退化为整张画布。

    Returns:
        (delta_row, delta_col, response):
        - delta_row: 行方向偏移修正量（亚像素），正值=下移
        - delta_col: 列方向偏移修正量（亚像素），正值=右移
        - response:  相位相关响应值 [0, 1]，越高越可靠
    """
    out_h, out_w = canvas.shape[1], canvas.shape[2]
    h, w = data.shape[1], data.shape[2]

    # ------------------------------------------------------------------
    # 1. 计算新影像在画布上的范围
    # ------------------------------------------------------------------
    img_dst_r0 = max(0, row_off)
    img_dst_c0 = max(0, col_off)
    img_dst_r1 = min(out_h, row_off + h)
    img_dst_c1 = min(out_w, col_off + w)

    if img_dst_r1 <= img_dst_r0 or img_dst_c1 <= img_dst_c0:
        return 0.0, 0.0, 0.0

    # ------------------------------------------------------------------
    # 2. 从掩膜中提取重叠区域内的有效数据范围
    #    紧缩到 mask 中 True 像素的最小包围盒
    # ------------------------------------------------------------------
    if filled_mask is not None:
        mask_sub = filled_mask[img_dst_r0:img_dst_r1, img_dst_c0:img_dst_c1]
        rows_has = np.any(mask_sub, axis=1)
        cols_has = np.any(mask_sub, axis=0)

        if not rows_has.any() or not cols_has.any():
            return 0.0, 0.0, 0.0

        r0_loc = int(np.argmax(rows_has))
        r1_loc = int(len(rows_has) - np.argmax(rows_has[::-1]))
        c0_loc = int(np.argmax(cols_has))
        c1_loc = int(len(cols_has) - np.argmax(cols_has[::-1]))

        dst_r0 = img_dst_r0 + r0_loc
        dst_c0 = img_dst_c0 + c0_loc
        dst_r1 = img_dst_r0 + r1_loc
        dst_c1 = img_dst_c0 + c1_loc
    else:
        dst_r0, dst_c0, dst_r1, dst_c1 = img_dst_r0, img_dst_c0, img_dst_r1, img_dst_c1

    overlap_h = dst_r1 - dst_r0
    overlap_w = dst_c1 - dst_c0

    if overlap_h < REGISTRATION_MIN_OVERLAP or overlap_w < REGISTRATION_MIN_OVERLAP:
        return 0.0, 0.0, 0.0

    # 源影像上的对应区域
    src_r0 = dst_r0 - row_off
    src_c0 = dst_c0 - col_off
    src_r1 = src_r0 + overlap_h
    src_c1 = src_c0 + overlap_w

    # ------------------------------------------------------------------
    # 3. 提取重叠区 patch，调用 img_registration 计算偏移
    # ------------------------------------------------------------------
    canvas_patch = canvas[:, dst_r0:dst_r1, dst_c0:dst_c1]
    src_patch = data[:, src_r0:src_r1, src_c0:src_c1]

    if canvas_patch.shape != src_patch.shape:
        return 0.0, 0.0, 0.0

    d_row, d_col, response = phase_correlate(canvas_patch, src_patch)

    # ------------------------------------------------------------------
    # 4. 可靠性校验
    # ------------------------------------------------------------------
    if response < REGISTRATION_RESPONSE_THRESHOLD:
        return 0.0, 0.0, response

    shift_magnitude = np.sqrt(d_row * d_row + d_col * d_col)
    if shift_magnitude > REGISTRATION_MAX_SHIFT:
        return 0.0, 0.0, response

    return d_row, d_col, response


# ============================================================
#  主入口
# ============================================================

def merge_tifs(src_paths: list[str], original_stems: list[str] | None = None) -> str:
    """拼接多个带地理参考的 TIFF 影像。

    以第一个文件的像素分辨率和波段数为基准，将所有影像按地理坐标
    投影到统一画布，重叠区域后覆盖前（列表顺序决定优先级）。

    对第 2 张及之后的影像自动进行相位相关精配准，偏移量直接作用于
    像素贴入位置（不修改影像地理坐标）。

    Args:
        src_paths: 输入 TIFF 文件路径列表，至少 1 个。顺序决定覆盖优先级。
        original_stems: 原始文件名（不含扩展名）列表，用于构建输出文件名。
            未传时使用第一个 src_path 的 stem。

    Returns:
        输出拼接后的 TIFF 文件路径。

    Raises:
        MosaicError: 无法打开影像、CRS 不一致或拼接失败。
    """
    if not src_paths:
        raise MosaicError("至少需要提供 1 个影像文件")

    datasets: list[rasterio.DatasetReader] = []
    try:
        # ------------------------------------------------------------------
        # 1. 打开所有影像
        # ------------------------------------------------------------------
        for p in src_paths:
            datasets.append(rasterio.open(p))

        # ------------------------------------------------------------------
        # 2. 校验 CRS 一致性
        # ------------------------------------------------------------------
        _validate_crs(datasets)

        # ------------------------------------------------------------------
        # 3. 以第一张影像为基准，计算输出画布尺寸
        # ------------------------------------------------------------------
        ref = datasets[0]
        res_x = ref.transform.a   # 像元宽度（地理单位）
        res_y = -ref.transform.e  # 像元高度（地理单位，取正）

        left, bottom, right, top = _union_bounds(datasets)

        out_w = round((right - left) / res_x)
        out_h = round((top - bottom) / res_y)
        out_transform = from_origin(left, top, res_x, res_y)

        # 使用第一张影像的数据类型和波段数创建画布
        canvas = np.zeros((ref.count, out_h, out_w), dtype=ref.dtypes[0])

        # 布尔掩膜：标记画布上哪些像素已贴入数据（True=已填充，False=背景）
        filled_mask = np.zeros((out_h, out_w), dtype=bool)

        # ------------------------------------------------------------------
        # 4. 逐张贴入影像（含精配准）
        # ------------------------------------------------------------------
        for i, ds in enumerate(datasets):
            # 计算该影像在画布中的像素偏移（基于地理坐标）
            row_off = round((top - ds.bounds.top) / res_y)
            col_off = round((ds.bounds.left - left) / res_x)

            data = ds.read()

            # 对第 2 张及之后的影像进行相位相关精配准
            if REGISTRATION_ENABLED and i > 0:
                d_row, d_col, _ = _compute_registration_shift(
                    canvas, data, row_off, col_off,
                    filled_mask=filled_mask,
                )
                # 仅在有显著偏移时应用（像素级别）
                if abs(d_row) > 0.5 or abs(d_col) > 0.5:
                    row_off = round(row_off + d_row)
                    col_off = round(col_off + d_col)

            _paste_image(canvas, data, row_off, col_off)

            # 更新布尔掩膜：标记本次贴入区域为 True
            paste_r0 = max(0, row_off)
            paste_c0 = max(0, col_off)
            paste_r1 = min(out_h, row_off + data.shape[1])
            paste_c1 = min(out_w, col_off + data.shape[2])
            if paste_r1 > paste_r0 and paste_c1 > paste_c0:
                filled_mask[paste_r0:paste_r1, paste_c0:paste_c1] = True

        # ------------------------------------------------------------------
        # 5. 写出结果
        # ------------------------------------------------------------------
        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            height=out_h,
            width=out_w,
            transform=out_transform,
            compress="lzw",
        )

        # 输出文件名 = 所有输入文件名拼接 + 后缀
        if original_stems:
            stems = original_stems
        else:
            stems = [Path(p).stem for p in src_paths]
        output_stem = "_".join(stems) + MOSAIC_OUTPUT_SUFFIX
        dst_path = Path(src_paths[0]).parent / f"{output_stem}{OUTPUT_EXTENSION}"

        with rasterio.open(str(dst_path), "w", **profile) as dst:
            dst.write(canvas)

    except MosaicError:
        raise
    except Exception as exc:
        raise MosaicError(f"影像拼接失败: {exc}")
    finally:
        for ds in datasets:
            ds.close()

    return str(dst_path)
