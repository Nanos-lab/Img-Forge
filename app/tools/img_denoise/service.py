"""img_denoise 服务层 —— 噪声抑制与边缘增强。

核心管线:
1. 读取 TIFF → 归一化到 [0, 1]
2. 逐波段独立处理（不依赖颜色语义，任意波段数通用）：
   a. 噪声抑制：双边滤波（保边去噪）
   b. 边缘增强：反锐化蒙版（Unsharp Masking）
3. 反归一化 → rasterio 写出
"""

from pathlib import Path

import cv2
import numpy as np
import rasterio

from app.core.config import DENOISE_OUTPUT_SUFFIX, OUTPUT_EXTENSION
from app.core.exceptions import DenoiseError


# ============================================================
#  归一化 / 反归一化
# ============================================================


def _normalize(image: np.ndarray, src_dtype: np.dtype) -> np.ndarray:
    """归一化到 [0, 1] 浮点。"""
    img = image.astype(np.float32)
    if src_dtype == np.uint8:
        return img / 255.0
    elif src_dtype == np.uint16:
        return img / 65535.0
    else:
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            return (img - vmin) / (vmax - vmin)
        return np.zeros_like(img, dtype=np.float32)


def _denormalize(image: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """从 [0, 1] 还原为目标数据类型。"""
    img = np.clip(image, 0.0, 1.0)
    if target_dtype == np.uint8:
        return (img * 255.0).astype(np.uint8)
    elif target_dtype == np.uint16:
        return (img * 65535.0).astype(np.uint16)
    else:
        return img.astype(target_dtype)


# ============================================================
#  单波段滤波
# ============================================================


def _denoise_band(band: np.ndarray, strength: float) -> np.ndarray:
    """双边滤波噪声抑制，保边去噪。

    Args:
        band:     单波段浮点数组，取值 [0, 1]。
        strength: 降噪强度 [0, 1]，0 不处理。
    """
    if strength <= 0.0:
        return band

    diameter = int(round(3 + strength * 6))       # 3 ~ 9
    sigma_color = 0.05 + strength * 0.35           # 0.05 ~ 0.40
    sigma_space = 3.0 + strength * 12.0            # 3 ~ 15

    band_c = np.ascontiguousarray(band, dtype=np.float32)
    filtered = cv2.bilateralFilter(band_c, diameter, sigma_color, sigma_space)
    return np.clip(filtered, 0.0, 1.0)


def _sharpen_band(band: np.ndarray, strength: float) -> np.ndarray:
    """反锐化蒙版边缘增强。

    Args:
        band:     单波段浮点数组，取值 [0, 1]。
        strength: 锐化强度 [0, 3]，0 不处理。
    """
    if strength <= 0.0:
        return band

    band_c = np.ascontiguousarray(band, dtype=np.float32)
    blurred = cv2.GaussianBlur(band_c, ksize=(0, 0), sigmaX=1.0)
    sharpened = band_c + strength * (band_c - blurred)
    return np.clip(sharpened, 0.0, 1.0)


def _process_band(band: np.ndarray, denoise: float, sharpen: float) -> np.ndarray:
    """单波段完整管线：降噪 → 锐化。"""
    band = _denoise_band(band, denoise)
    band = _sharpen_band(band, sharpen)
    return band


# ============================================================
#  主入口
# ============================================================


def denoise_image(src_path: str, denoise: float, sharpen: float) -> str:
    """对遥感影像执行噪声抑制与边缘增强。

    逐波段独立处理，不依赖颜色语义，适用于任意波段数的影像
    （单波段灰度图、RGB、多光谱均可）。

    Args:
        src_path: 输入 TIFF 文件路径。
        denoise:  降噪强度 [0, 1]，0 不处理，1 最强。
        sharpen:  锐化强度 [0, 3]，0 不处理，值越大边缘增强越明显。

    Returns:
        输出影像文件路径。

    Raises:
        DenoiseError: 处理失败。
    """
    try:
        with rasterio.open(src_path) as src:
            image = src.read()
            profile = src.profile.copy()
    except Exception as exc:
        raise DenoiseError(f"无法读取影像: {exc}")

    bands, height, width = image.shape
    src_dtype = image.dtype

    # 归一化
    norm = _normalize(image, src_dtype)

    # 逐波段处理
    processed = np.stack(
        [_process_band(norm[i], denoise, sharpen) for i in range(bands)],
        axis=0,
    )

    result = _denormalize(processed, src_dtype)

    # 写出
    src_path_obj = Path(src_path)
    dst_path = src_path_obj.parent / f"{src_path_obj.stem}{DENOISE_OUTPUT_SUFFIX}{OUTPUT_EXTENSION}"

    profile.update(driver="GTiff", dtype=src_dtype, count=bands,
                   height=height, width=width, compress="lzw")
    try:
        with rasterio.open(str(dst_path), "w", **profile) as dst:
            dst.write(result)
    except Exception as exc:
        raise DenoiseError(f"写出处理结果失败: {exc}")

    return str(dst_path)
