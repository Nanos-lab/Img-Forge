"""变化检测服务层 —— 完整 pipeline 编排。

流程:
1. 粗配准（地理裁剪到重叠区）
2. 精配准（相位相关像素级对齐）
3. 分块
4. 模型推理
5. 概率图拼接
6. 变化区域提取 + GeoJSON
"""

import time

import cv2
import numpy as np
import rasterio

from app.core.exceptions import ChangeDetectError
from app.tools.img_registration import phase_correlate
from app.tools.img_changedet.predictor import get_predictor
from app.tools.img_changedet.tiler import tile_pair, stitch_probability_map
from app.tools.img_changedet.extractor import extract_changes, build_geojson

_TILE_SIZE = 256
_TILE_OVERLAP = 8


# ============================================================
#  粗配准
# ============================================================


def _coarse_register(
    old_path: str,
    new_path: str,
) -> tuple[np.ndarray, np.ndarray, rasterio.Affine, tuple[int, int], rasterio.CRS]:
    """将两幅影像粗配准到同一地理重叠区。

    计算两幅影像的地理交集，分别裁剪到同一范围。
    分辨率不同时用 cv2.resize 统一。

    Returns:
        (old_cropped, new_cropped, transform, crop_offset, crs):
        - old_cropped: 裁剪后的变化前影像 (H, W, 3) uint8。
        - new_cropped: 裁剪后的变化后影像 (H, W, 3) uint8。
        - transform: 裁剪后影像的 geotransform。
        - crop_offset: (y_min, x_min) 在原始 new 影像中的像素偏移。
        - crs: 裁剪后影像的 CRS。
    """
    with rasterio.open(old_path) as src_old, rasterio.open(new_path) as src_new:
        # 读取影像到内存
        old_full = src_old.read()
        new_full = src_new.read()

        # 转 (H, W, 3) uint8
        def _to_hwc(data, count):
            if count >= 3:
                rgb = data[:3]
            elif count == 1:
                rgb = np.stack([data[0]] * 3, axis=0)
            else:
                rgb = data[:3]
            # 转为 uint8
            if rgb.dtype != np.uint8:
                if rgb.dtype == np.uint16:
                    rgb = (rgb / 256).astype(np.uint8)
                else:
                    # float: min-max 拉伸
                    rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10) * 255).astype(np.uint8)
            return np.transpose(rgb, (1, 2, 0))

        old_hwc = _to_hwc(old_full, src_old.count)
        new_hwc = _to_hwc(new_full, src_new.count)

        h_old, w_old = old_hwc.shape[:2]
        h_new, w_new = new_hwc.shape[:2]

        # 计算地理交集
        ov_left = max(src_old.bounds.left, src_new.bounds.left)
        ov_bottom = max(src_old.bounds.bottom, src_new.bounds.bottom)
        ov_right = min(src_old.bounds.right, src_new.bounds.right)
        ov_top = min(src_old.bounds.top, src_new.bounds.top)

        if ov_right <= ov_left or ov_top <= ov_bottom:
            raise ChangeDetectError("两幅影像无重叠区域，无法进行变化检测")

        # 用新影像的分辨率作为输出分辨率
        res_x = src_new.transform.a
        res_y = -src_new.transform.e
        crs = src_new.crs

        # 将交集地理坐标转为各影像的像素坐标
        def _geo_to_pixel(transform, left, right, top, bottom, img_w, img_h):
            col_start = max(0, round((left - transform.c) / transform.a))
            col_end = min(img_w, round((right - transform.c) / transform.a))
            row_start = max(0, round((top - transform.f) / transform.e))
            row_end = min(img_h, round((bottom - transform.f) / transform.e))
            return row_start, col_start, row_end, col_end

        o_r0, o_c0, o_r1, o_c1 = _geo_to_pixel(
            src_old.transform, ov_left, ov_right, ov_top, ov_bottom, w_old, h_old)
        n_r0, n_c0, n_r1, n_c1 = _geo_to_pixel(
            src_new.transform, ov_left, ov_right, ov_top, ov_bottom, w_new, h_new)

        # 切片裁剪
        old_crop = old_hwc[o_r0:o_r1, o_c0:o_c1]
        new_crop = new_hwc[n_r0:n_r1, n_c0:n_c1]

        # 如果尺寸不同，统一到新影像的尺寸
        if old_crop.shape[:2] != new_crop.shape[:2]:
            old_crop = cv2.resize(
                old_crop,
                (new_crop.shape[1], new_crop.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # 输出 transform: 从交集左上角开始，用新影像的分辨率
        out_transform = rasterio.transform.from_origin(ov_left, ov_top, res_x, res_y)

        # crop_offset: 裁剪区域左上角在新影像原始坐标系中的像素偏移
        off_x = n_c0
        off_y = n_r0

    return old_crop, new_crop, out_transform, (off_y, off_x), crs


# ============================================================
#  精配准
# ============================================================


def _fine_register(
    old_img: np.ndarray,
    new_img: np.ndarray,
) -> np.ndarray:
    """用相位相关对 old 做精配准，使其与 new 像素级对齐。

    Args:
        old_img: 变化前影像 (H, W, 3) uint8。
        new_img: 变化后影像 (H, W, 3) uint8。

    Returns:
        配准后的 old_img (H, W, 3) uint8。
    """
    h, w = old_img.shape[:2]
    if h < 64 or w < 64:
        return old_img.copy()

    # 取重叠中心区域做配准（避免边缘噪声）
    margin_h, margin_w = h // 4, w // 4
    ref_patch = new_img[margin_h:h - margin_h, margin_w:w - margin_w, :]
    src_patch = old_img[margin_h:h - margin_h, margin_w:w - margin_w, :]

    # (H, W, 3) → (3, H, W) for phase_correlate
    ref_t = np.transpose(ref_patch, (2, 0, 1))
    src_t = np.transpose(src_patch, (2, 0, 1))

    dr, dc, response = phase_correlate(ref_t, src_t)

    if abs(dr) < 0.5 and abs(dc) < 0.5:
        return old_img.copy()

    # warp old_img 对齐到 new_img
    mat = np.float32([[1, 0, dc], [0, 1, dr]])
    warped = cv2.warpAffine(
        old_img, mat, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return warped


# ============================================================
#  主入口
# ============================================================


def detect_changes(
    old_path: str,
    new_path: str,
    threshold: float | None = None,
    min_area: int | None = None,
) -> dict:
    """执行完整变化检测流程。

    Args:
        old_path: 变化前影像路径。
        new_path: 变化后影像路径。
        threshold: 二值化阈值，None Otsu 自动。
        min_area: 最小变化区域面积，None 自动。

    Returns:
        GeoJSON dict。
    """
    t_start = time.perf_counter()

    # ---- Step 1: 粗配准 ----
    old_img, new_img, transform, _, crs = _coarse_register(old_path, new_path)
    print(f"Coarse registration: {old_img.shape[1]}x{old_img.shape[0]}")

    # ---- Step 2: 精配准 ----
    old_img = _fine_register(old_img, new_img)
    print("Fine registration done")

    # ---- Step 3: 分块 ----
    tile_set = tile_pair(old_img, new_img, tile_size=_TILE_SIZE, overlap=_TILE_OVERLAP)
    print(f"Tiles: {tile_set.grid_rows}x{tile_set.grid_cols} ({len(tile_set.tiles)} tiles)")

    # ---- Step 4: 推理 ----
    predictor = get_predictor()
    patches = predictor.predict_batch(tile_set)

    # ---- Step 5: 拼接 ----
    prob_map = stitch_probability_map(patches, tile_set)
    print(f"Stitched prob map: {prob_map.shape[1]}x{prob_map.shape[0]}")

    # ---- Step 6: 变化提取 ----
    regions, binary_map = extract_changes(
        prob_map, transform,
        crop_offset=(0, 0),
        min_area=min_area,
        threshold=threshold,
    )

    elapsed_ms = int(round((time.perf_counter() - t_start) * 1000))

    # ---- Step 7: 输出 GeoJSON ----
    geojson = build_geojson(regions, elapsed_ms)
    print(f"Changes: {len(regions)} regions in {elapsed_ms}ms")

    # 可视化画在 new_path 上（GeoJSON 地理坐标 → new_path 像素坐标）
    save_visualization_tiff(new_path, geojson)

    return geojson


def save_visualization_tiff(
    tif_path: str,
    geojson: dict,
) -> str:
    """在原始影像上绘制变化框，保存为可视化 TIFF。

    Args:
        tif_path: 原始 TIFF 路径。
        geojson: 变化检测返回的 GeoJSON dict。

    Returns:
        输出的 TIFF 路径；失败返回空字符串。
    """
    from pathlib import Path as _P

    out_path = _P(tif_path).parent / f"cd_result_{_P(tif_path).stem}.tif"
    try:
        with rasterio.open(tif_path) as _src:
            _raw = _src.read()
            _profile = _src.profile.copy()
            _tf = _src.transform
            if _raw.shape[0] >= 3:
                _vis = np.transpose(_raw[:3], (1, 2, 0))
            else:
                _vis = np.stack([_raw[0]] * 3, axis=2)
            if _vis.dtype == np.uint16:
                _vis = (_vis / 256).astype(np.uint8)
            elif _vis.dtype != np.uint8:
                _vis = np.clip(_vis / (_vis.max() + 1e-10) * 255, 0, 255).astype(np.uint8)
            _vis = np.ascontiguousarray(_vis)
        features = geojson.get("features", [])
        for f in features:
            r_id = f.get("id", 0)
            coords = f["geometry"]["coordinates"][0]
            lons = [pt[0] for pt in coords[:-1]]
            lats = [pt[1] for pt in coords[:-1]]
            rows, cols = rasterio.transform.rowcol(_tf, lons, lats)
            x_min, x_max = min(cols), max(cols)
            y_min, y_max = min(rows), max(rows)
            _vis = cv2.rectangle(_vis, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            _vis = cv2.putText(_vis, f"#{r_id}", (x_min, max(y_min - 5, 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        _profile.update(count=3, dtype=np.uint8, compress="lzw")
        _vis_chw = np.transpose(_vis, (2, 0, 1)).astype(np.uint8)
        with rasterio.open(str(out_path), "w", **_profile) as _dst:
            _dst.write(_vis_chw)
        print(f"Visualization TIFF saved: {out_path}")
        return str(out_path)
    except Exception as _e:
        print(f"Save visualization TIFF failed: {_e}")
        return ""
