"""img_enhance_2 服务层 —— Cesium 风格色调增强。

核心管线:
1. 读取 TIFF → 归一化到 [0, 1]
2. 亮度：线性缩放（1=原始，0=全黑，3=最亮）
3. 对比度：以 0.5 为中心缩放偏离量（1=原始，0=统一灰色，5=最大反差）
4. 饱和度：亮度保持的 RGB 混合（1=原始，0=完全灰度，4=最鲜艳）
5. 反归一化 → rasterio 写出

无水体/植被检测，简洁高效。
"""

from pathlib import Path

import numpy as np
import rasterio

from app.core.config import OUTPUT_SUFFIX, OUTPUT_EXTENSION
from app.core.exceptions import EnhancementError


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
#  Cesium 风格色调映射
# ============================================================


def _adjust_brightness(rgb: np.ndarray, brightness: float) -> np.ndarray:
    """亮度：线性缩放。"""
    if brightness == 1.0:
        return rgb
    return np.clip(rgb * brightness, 0.0, 1.0)


def _adjust_contrast(rgb: np.ndarray, contrast: float) -> np.ndarray:
    """对比度：以 0.5 为中心缩放偏离量。"""
    if contrast == 1.0:
        return rgb
    return np.clip((rgb - 0.5) * contrast + 0.5, 0.0, 1.0)


def _adjust_saturation(rgb: np.ndarray, saturation: float) -> np.ndarray:
    """饱和度：亮度保持的 RGB 混合。"""
    if saturation == 1.0:
        return rgb

    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gray = np.stack([gray, gray, gray], axis=2)

    return np.clip(gray * (1.0 - saturation) + rgb * saturation, 0.0, 1.0)


def _enhance_rgb(rgb: np.ndarray,
                 brightness: float,
                 contrast: float,
                 saturation: float) -> np.ndarray:
    """完整色调增强管线，顺序：亮度 → 对比度 → 饱和度。"""
    rgb = _adjust_brightness(rgb, brightness)
    rgb = _adjust_contrast(rgb, contrast)
    rgb = _adjust_saturation(rgb, saturation)
    return rgb


# ============================================================
#  主入口
# ============================================================


def enhance_image(src_path: str,
                  brightness: float,
                  contrast: float,
                  saturation: float) -> str:
    """对遥感影像执行 Cesium 风格色调增强。

    Args:
        src_path:    输入 TIFF 文件路径。
        brightness:  亮度系数 [0, 3]，1 原始值，0 全黑。
        contrast:    对比度系数 [0, 5]，1 原始值，0 统一灰色。
        saturation:  饱和度系数 [0, 4]，1 原始值，0 完全灰度。

    Returns:
        输出影像文件路径。

    Raises:
        EnhancementError: 处理失败。
    """
    try:
        with rasterio.open(src_path) as src:
            image = src.read()
            profile = src.profile.copy()
    except Exception as exc:
        raise EnhancementError(f"无法读取影像: {exc}")

    bands, height, width = image.shape
    if bands < 3:
        raise EnhancementError(f"影像仅含 {bands} 个波段，需要至少 3 个波段")

    src_dtype = image.dtype

    # 归一化
    norm = _normalize(image, src_dtype)

    # 分离 RGB 和额外波段
    rgb = np.transpose(norm[:3], (1, 2, 0))  # (H, W, 3)
    extra = norm[3:]

    # Cesium 风格色调增强
    rgb_enhanced = _enhance_rgb(rgb, brightness, contrast, saturation)

    # 重组
    result_rgb = np.transpose(rgb_enhanced, (2, 0, 1))  # (3, H, W)
    if extra.shape[0] > 0:
        result = np.concatenate([result_rgb, extra], axis=0)
    else:
        result = result_rgb

    result = _denormalize(result, src_dtype)

    # 写出
    src_path_obj = Path(src_path)
    dst_path = src_path_obj.parent / f"{src_path_obj.stem}{OUTPUT_SUFFIX}{OUTPUT_EXTENSION}"

    profile.update(driver="GTiff", dtype=src_dtype, count=bands,
                   height=height, width=width, compress="lzw")
    try:
        with rasterio.open(str(dst_path), "w", **profile) as dst:
            dst.write(result)
    except Exception as exc:
        raise EnhancementError(f"写出增强影像失败: {exc}")

    return str(dst_path)
