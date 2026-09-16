"""img_pansharpen 服务层 —— 全色锐化（Pan Sharpening，Gram-Schmidt 算法）。

核心管线:
1. 读取多光谱（MS）和全色（Pan）影像，校验 CRS / 波段数 / 地理范围覆盖
2. 将 MS 重采样到 Pan 的地理网格（分辨率对齐，cubic 重采样）
3. 构造合成全色分量 I = MS 各波段加权平均（权重可指定，默认等权重）
4. 将真实 Pan 的均值/方差通过线性回归匹配到 I，得到 Pan_matched
5. 对每个 MS 波段计算 GS 增益 g = cov(MS_b, I) / var(I)
   融合：Sharpened_b = MS_b + g * (Pan_matched - I)
6. 反归一化为 MS 原始 dtype → rasterio 写出（地理网格与 Pan 一致）
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from app.core.config import (
    OUTPUT_EXTENSION,
    PANSHARPEN_OUTPUT_SUFFIX,
    PANSHARPEN_RESAMPLING,
)
from app.core.exceptions import PansharpenError


# ============================================================
#  归一化 / 反归一化
# ============================================================


def _normalize(image: np.ndarray, src_dtype: np.dtype) -> tuple[np.ndarray, float, float]:
    """归一化到浮点，返回 (归一化数组, scale, offset) 供反归一化复用。"""
    img = image.astype(np.float64)
    if src_dtype == np.uint8:
        scale = 255.0
    elif src_dtype == np.uint16:
        scale = 65535.0
    else:
        scale = 1.0
    return img / scale, scale, 0.0


def _denormalize(image: np.ndarray, target_dtype: np.dtype, scale: float) -> np.ndarray:
    """从浮点还原为目标数据类型。"""
    img = image * scale
    if target_dtype in (np.uint8, np.uint16):
        info = np.iinfo(target_dtype)
        img = np.clip(img, info.min, info.max)
    return img.astype(target_dtype)


# ============================================================
#  校验
# ============================================================


def _validate_extent_coverage(
    ms_ds: rasterio.DatasetReader,
    pan_ds: rasterio.DatasetReader,
) -> None:
    """校验 MS 的地理范围是否完全覆盖 Pan 的地理范围。

    全色锐化要求两张影像覆盖同一区域（已配准），若 MS 只能覆盖 Pan
    的一部分，输出网格中 MS 缺失的区域会在重采样后变为 0，进而污染
    全局统计量（均值/方差/协方差），导致该区域被赋予看似合理却完全
    虚构的光谱值。为避免这种静默数据污染，范围不完全重叠时直接拒绝。

    容差取半个 Pan 像元，用于吸收边界坐标的浮点/重投影舍入误差。

    Raises:
        PansharpenError: 两者地理范围无交集，或 MS 未完全覆盖 Pan。
    """
    ms_b = ms_ds.bounds
    pan_b = pan_ds.bounds

    inter_left = max(ms_b.left, pan_b.left)
    inter_bottom = max(ms_b.bottom, pan_b.bottom)
    inter_right = min(ms_b.right, pan_b.right)
    inter_top = min(ms_b.top, pan_b.top)

    if inter_right <= inter_left or inter_top <= inter_bottom:
        raise PansharpenError(
            f"多光谱影像与全色影像地理范围无重叠，无法融合: "
            f"MS bounds={tuple(ms_b)}, Pan bounds={tuple(pan_b)}"
        )

    tol_x = abs(pan_ds.transform.a) * 0.5
    tol_y = abs(pan_ds.transform.e) * 0.5

    covers_pan = (
        inter_left <= pan_b.left + tol_x
        and inter_right >= pan_b.right - tol_x
        and inter_bottom <= pan_b.bottom + tol_y
        and inter_top >= pan_b.top - tol_y
    )
    if not covers_pan:
        raise PansharpenError(
            f"多光谱影像未完全覆盖全色影像的地理范围（部分重叠），"
            f"请确认两张影像覆盖同一区域后重试: "
            f"MS bounds={tuple(ms_b)}, Pan bounds={tuple(pan_b)}"
        )


def _validate_inputs(
    ms_ds: rasterio.DatasetReader,
    pan_ds: rasterio.DatasetReader,
    weights: list[float] | None,
) -> None:
    """校验 MS / Pan 影像是否满足全色锐化的前提条件。

    Raises:
        PansharpenError: CRS 缺失/不一致、Pan 非单波段、权重长度不匹配、
            地理范围无重叠或部分重叠等。
    """
    if ms_ds.crs is None or pan_ds.crs is None:
        raise PansharpenError("多光谱影像或全色影像缺少地理参考信息（CRS）")
    if ms_ds.crs != pan_ds.crs:
        raise PansharpenError(
            f"CRS 不一致: MS={ms_ds.crs}，Pan={pan_ds.crs}，请先完成配准/重投影"
        )
    if pan_ds.count != 1:
        raise PansharpenError(f"全色影像应为单波段，实际含 {pan_ds.count} 个波段")
    if ms_ds.count < 1:
        raise PansharpenError("多光谱影像至少需要 1 个波段")
    if weights is not None and len(weights) != ms_ds.count:
        raise PansharpenError(
            f"权重数量 ({len(weights)}) 与多光谱波段数 ({ms_ds.count}) 不一致"
        )
    _validate_extent_coverage(ms_ds, pan_ds)


# ============================================================
#  MS 重采样到 Pan 网格
# ============================================================


def _resample_ms_to_pan(
    ms_ds: rasterio.DatasetReader,
    pan_ds: rasterio.DatasetReader,
) -> np.ndarray:
    """将多光谱影像重采样到全色影像的地理网格（分辨率对齐）。

    Returns:
        重采样后的 MS 数组，shape (bands, pan_height, pan_width)，dtype 同 MS 原始类型。
    """
    ms_data = ms_ds.read()
    resampling = getattr(Resampling, PANSHARPEN_RESAMPLING, Resampling.cubic)

    destination = np.zeros(
        (ms_ds.count, pan_ds.height, pan_ds.width), dtype=ms_data.dtype
    )
    try:
        reproject(
            ms_data,
            destination,
            src_transform=ms_ds.transform,
            src_crs=ms_ds.crs,
            dst_transform=pan_ds.transform,
            dst_crs=pan_ds.crs,
            resampling=resampling,
            num_threads=2,
        )
    except Exception as exc:
        raise PansharpenError(f"多光谱影像重采样到全色网格失败: {exc}")

    return destination


# ============================================================
#  Gram-Schmidt 融合
# ============================================================


def _synthesize_pan(ms_norm: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """构造合成全色分量 I = MS 各波段加权平均。

    Args:
        ms_norm: 归一化后的 MS 数组，shape (bands, H, W)。
        weights: 权重数组，shape (bands,)，已归一化为和为 1。

    Returns:
        合成全色分量，shape (H, W)。
    """
    return np.tensordot(weights, ms_norm, axes=(0, 0))


def _match_histogram(pan_norm: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """将 Pan 的均值/方差线性匹配到合成分量 I，避免融合后整体亮度偏移。

    Pan_matched = (Pan - mean(Pan)) / std(Pan) * std(I) + mean(I)
    """
    pan_mean, pan_std = pan_norm.mean(), pan_norm.std()
    i_mean, i_std = intensity.mean(), intensity.std()

    if pan_std < 1e-8:
        return np.full_like(pan_norm, i_mean)

    return (pan_norm - pan_mean) / pan_std * i_std + i_mean


def _gram_schmidt_fuse(
    ms_norm: np.ndarray,
    pan_matched: np.ndarray,
    intensity: np.ndarray,
) -> np.ndarray:
    """Gram-Schmidt 融合：对每个 MS 波段用 GS 增益注入 Pan 的高频细节。

    g_b = cov(MS_b, I) / var(I)
    Sharpened_b = MS_b + g_b * (Pan_matched - I)

    Args:
        ms_norm:     归一化 MS 数组，shape (bands, H, W)。
        pan_matched: 直方图匹配后的 Pan，shape (H, W)。
        intensity:   合成全色分量 I，shape (H, W)。

    Returns:
        融合后的 MS 数组，shape (bands, H, W)。
    """
    i_var = intensity.var()
    detail = pan_matched - intensity

    if i_var < 1e-12:
        # I 几乎无变化（如全黑/全白影像），直接返回原始 MS，避免除零
        return ms_norm.copy()

    bands = ms_norm.shape[0]
    result = np.empty_like(ms_norm)
    for b in range(bands):
        band = ms_norm[b]
        cov = ((band - band.mean()) * (intensity - intensity.mean())).mean()
        gain = cov / i_var
        result[b] = band + gain * detail

    return result


# ============================================================
#  主入口
# ============================================================


def pansharpen_image(
    ms_path: str,
    pan_path: str,
    weights: list[float] | None = None,
) -> str:
    """对配准好的多光谱 + 全色影像执行 Gram-Schmidt 全色锐化。

    Args:
        ms_path:  低分辨率多光谱影像路径（.tif/.tiff，多波段）。
        pan_path: 高分辨率全色影像路径（.tif/.tiff，单波段）。
        weights:  MS 各波段合成全色分量时的权重，长度需等于 MS 波段数，
            未传时使用等权重。

    Returns:
        输出高分辨率多光谱影像文件路径（地理网格与 Pan 一致）。

    Raises:
        PansharpenError: 处理失败，包括地理范围无重叠或部分重叠。
    """
    try:
        ms_ds = rasterio.open(ms_path)
        pan_ds = rasterio.open(pan_path)
    except Exception as exc:
        raise PansharpenError(f"无法读取影像: {exc}")

    try:
        _validate_inputs(ms_ds, pan_ds, weights)

        # ------------------------------------------------------------------
        # 1. MS 重采样到 Pan 网格
        # ------------------------------------------------------------------
        ms_dtype = ms_ds.dtypes[0]
        ms_resampled = _resample_ms_to_pan(ms_ds, pan_ds)

        pan_data = pan_ds.read(1)
        pan_dtype = pan_ds.dtypes[0]

        # ------------------------------------------------------------------
        # 2. 归一化
        # ------------------------------------------------------------------
        ms_norm, ms_scale, _ = _normalize(ms_resampled, ms_dtype)
        pan_norm, _, _ = _normalize(pan_data, pan_dtype)

        bands = ms_norm.shape[0]
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.sum() <= 0:
                raise PansharpenError("权重之和必须大于 0")
            w = w / w.sum()
        else:
            w = np.full(bands, 1.0 / bands)

        # ------------------------------------------------------------------
        # 3. 合成全色分量 + 直方图匹配 + GS 融合
        # ------------------------------------------------------------------
        intensity = _synthesize_pan(ms_norm, w)
        pan_matched = _match_histogram(pan_norm, intensity)
        fused_norm = _gram_schmidt_fuse(ms_norm, pan_matched, intensity)

        # ------------------------------------------------------------------
        # 4. 反归一化 + 写出
        # ------------------------------------------------------------------
        fused_norm = np.clip(fused_norm, 0.0, None)
        result = _denormalize(fused_norm, ms_dtype, ms_scale)

        profile = pan_ds.profile.copy()
        profile.update(
            driver="GTiff",
            count=bands,
            dtype=ms_dtype,
            compress="lzw",
        )

        ms_stem = Path(ms_path).stem
        pan_stem = Path(pan_path).stem
        dst_path = (
            Path(ms_path).parent
            / f"{ms_stem}_{pan_stem}{PANSHARPEN_OUTPUT_SUFFIX}{OUTPUT_EXTENSION}"
        )

        try:
            with rasterio.open(str(dst_path), "w", **profile) as dst:
                dst.write(result)
        except Exception as exc:
            raise PansharpenError(f"写出融合结果失败: {exc}")

    finally:
        ms_ds.close()
        pan_ds.close()

    return str(dst_path)
