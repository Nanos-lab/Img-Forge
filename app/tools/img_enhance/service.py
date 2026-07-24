"""影像增强服务层 —— 基于曲线映射的色调增强算法。

核心处理管线:
1. 读取 TIFF → 归一化 → 分离 RGB / 额外波段
2. RGB 转 HSV，对 V 通道做 gamma 亮度 + S 曲线对比度
3. 基于可见光特征检测水体 / 植被掩膜
4. 对特定区域做通道级色彩增强
5. 还原数据类型 → rasterio 写出（保留投影等几何属性）

一个函数 = 一个处理环节，互不耦合，便于单独测试。
"""

from pathlib import Path

import cv2
import numpy as np
import rasterio

from app.core.config import (
    OUTPUT_SUFFIX, OUTPUT_EXTENSION,
    DEFAULT_WATER_FACTOR, DEFAULT_VEGETATION_FACTOR,
)
from app.core.exceptions import EnhancementError


# ============================================================
#  归一化 / 反归一化
# ============================================================

def _normalize_to_float(image: np.ndarray, src_dtype: np.dtype) -> np.ndarray:
    """将栅格数据归一化到 [0, 1] 浮点范围。

    支持 uint8 (除以 255)、uint16 (除以 65535) 以及 float (min-max 拉伸)。
    """
    image = image.astype(np.float32)
    if src_dtype == np.uint8:
        return image / 255.0
    elif src_dtype == np.uint16:
        return image / 65535.0
    else:
        vmin, vmax = image.min(), image.max()
        if vmax > vmin:
            return (image - vmin) / (vmax - vmin)
        return np.zeros_like(image, dtype=np.float32)


def _denormalize_from_float(image: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """将 [0, 1] 浮点图像还原为目标数据类型。"""
    image = np.clip(image, 0.0, 1.0)
    if target_dtype == np.uint8:
        return (image * 255.0).astype(np.uint8)
    elif target_dtype == np.uint16:
        return (image * 65535.0).astype(np.uint16)
    else:
        return image.astype(target_dtype)


# ============================================================
#  色彩空间转换工具
# ============================================================

def _bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    """BGR [0, 1] 浮点 → HSV [0, 1] 浮点。"""
    bgr_u8 = (np.clip(bgr, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0


def _hsv_to_bgr(hsv: np.ndarray) -> np.ndarray:
    """HSV [0, 1] 浮点 → BGR [0, 1] 浮点。"""
    hsv_u8 = (np.clip(hsv, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


# ============================================================
#  亮度 / 对比度曲线调整
# ============================================================

def _adjust_gamma_brightness(v: np.ndarray, brightness: float) -> np.ndarray:
    """对 V 通道应用 gamma 曲线进行亮度调整。

    gamma = 2^(-brightness)，clamp 到 [0.125, 8.0]
    - brightness = 0 → gamma = 1.0（中性，无变化）
    - brightness > 0 → gamma < 1（提亮，暗部尤为明显）
    - brightness < 0 → gamma > 1（压暗）

    理论依据：摄影曝光模型中，一档曝光对应光量翻倍。
    在 gamma 编码空间中等价于 gamma ∝ 2^(-EV)，
    brightness 以 2 为底指数映射，每 ±1 单位 ≈ ±1 档曝光。
    默认 0.3 → gamma = 2^(-0.3) ≈ 0.81，与旧默认 (0.6 → 0.83) 一致。
    """
    gamma = max(0.125, min(8.0, 2.0 ** (-brightness)))
    v_safe = np.clip(v, 1e-6, 1.0)
    return np.power(v_safe, gamma)


def _adjust_s_curve_contrast(v: np.ndarray, contrast: float) -> np.ndarray:
    """对 V 通道进行对比度调整，支持正负双向。

    正值（增强对比度）：S 曲线（sigmoid）拉伸中间调，
    同时压缩高光和暗部两端，避免线性拉伸导致的过曝和死黑。

    负值（降低对比度）：线性将 V 向 0.5（中灰）压缩，
    V_out = V × (1 - |c|) + 0.5 × |c|，影像趋向扁平。

    0 为中性，无调整。
    """
    if contrast == 0.0:
        return v

    if contrast > 0.0:
        k = 5.0 + contrast * 20.0  # [5, 25]
        v_centered = v - 0.5
        v_s = 1.0 / (1.0 + np.exp(-k * v_centered))
        lo = 1.0 / (1.0 + np.exp(k * 0.5))
        hi = 1.0 / (1.0 + np.exp(-k * 0.5))
        v_s = (v_s - lo) / (hi - lo)
        return v * (1.0 - contrast) + v_s * contrast
    else:
        # contrast < 0: 压向中灰，降低对比度
        c_abs = -contrast
        return v * (1.0 - c_abs) + 0.5 * c_abs


def _adjust_saturation(s: np.ndarray, saturation: float) -> np.ndarray:
    """在 HSV 空间中对 S 通道做线性缩放。

    S_out = S × (1 + saturation)，clamp 到 [0, 1]
    - saturation = 0 → 中性，无变化
    - saturation > 0 → 提升饱和度（色彩更鲜艳）
    - saturation < 0 → 降低饱和度（趋向灰度）
    """
    if saturation == 0.0:
        return s
    return np.clip(s * (1.0 + saturation), 0.0, 1.0)


def _apply_tone_enhancement(bgr: np.ndarray,
                            brightness: float,
                            contrast: float,
                            saturation: float) -> np.ndarray:
    """光度 + 对比度 + 饱和度色调增强管线。

    在 HSV 空间中对 V 通道做 gamma 亮度 + S 曲线对比度，对 S 通道做
    线性缩放饱和度，然后重建 BGR 图像。H（色相）保持不变。
    """
    hsv = _bgr_to_hsv(bgr)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    v = _adjust_gamma_brightness(v, brightness)
    v = _adjust_s_curve_contrast(v, contrast)
    s = _adjust_saturation(s, saturation)

    hsv_out = np.stack([h, s, v], axis=2)
    return _hsv_to_bgr(hsv_out)


# ============================================================
#  水体 / 植被区域检测
# ============================================================

def _detect_water_mask(b: np.ndarray, g: np.ndarray, r: np.ndarray,
                       v: np.ndarray, s: np.ndarray) -> np.ndarray:
    """基于可见光 RGB 特征检测水体区域，返回软掩膜 [0, 1]。

    检测条件（同时满足）:
    - 蓝色通道占优: B > R 且 B > G
    - 低饱和度: S < 0.3（水体颜色不饱和）
    - 较暗: V < 0.5

    掩膜经高斯模糊处理，避免处理边界生硬。
    """
    mask = (b > r) & (b > g) & (v < 0.5) & (s < 0.3)
    mask_float = mask.astype(np.float32)
    mask_float = cv2.GaussianBlur(mask_float, (3, 3), sigmaX=0)
    return mask_float


def _detect_vegetation_mask(b: np.ndarray, g: np.ndarray,
                            r: np.ndarray) -> np.ndarray:
    """基于 Green Leaf Index (GLI) 检测植被区域，返回软掩膜 [0, 1]。

    GLI = (2G - R - B) / (2G + R + B)，范围 [-1, 1]，与影像位深无关，
    同一阈值对 uint8、uint16 均有效，避免 ExG 随动态范围漂移的问题。

    三层保守约束确保只选取真正植被：
    1. hard floor: GLI 必须 > 0.04（绿色通道显著占优）
    2. Otsu 进一步收紧阈值（仅在植被/非植被双峰明显时生效）
    3. 覆盖率封顶 45%（防止 Otsu 在单峰直方图上误判）
    最后以 3×3 高斯模糊柔化掩膜边界。
    """
    # GLI 计算，避免分母为 0
    gli = (2.0 * g - r - b) / (2.0 * g + r + b + 1e-8)

    # 硬底线：GLI 必须 > 0.04（绿色显著占优才算植被）
    hard_floor = 0.04

    # Otsu 自适应阈值（有双峰结构时起效，单峰时取 hard_floor）
    gli_u8 = ((gli + 1.0) / 2.0 * 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(
        gli_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    gli_otsu = (otsu_thresh / 255.0) * 2.0 - 1.0

    # 取较严格者
    threshold = max(gli_otsu, hard_floor)

    mask = (gli > threshold).astype(np.float32)

    # 覆盖率封顶：若 > 45% 像素被判定为植被，改用中位数作为阈值
    coverage = float(mask.mean())
    if coverage > 0.45:
        median_gli = float(np.percentile(gli, 55))  # 只保留 top 45%
        threshold = max(median_gli, hard_floor)
        mask = (gli > threshold).astype(np.float32)

    mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=0)
    return mask


# ============================================================
#  区域色彩增强（通用）
# ============================================================

def _apply_region_enhancement(bgr: np.ndarray, mask: np.ndarray,
                              factor: float, *,
                              boost_channel: int,
                              boost_strength: float,
                              suppress_channel: int,
                              suppress_strength: float,
                              ) -> np.ndarray:
    """通用区域色彩增强 —— 增强某通道、抑制另一通道。

    对 mask 覆盖区域做柔滑的通道级调整，增强强度 = mask × factor × strength，
    无 mask 区域（mask=0）不受影响，过渡区域（mask ∈ (0,1)）平滑渐变。

    Args:
        bgr: BGR 影像，shape (H, W, 3)，值域 [0, 1]。
        mask: 软掩膜 [0, 1]，由检测函数生成。
        factor: 用户指定的增强系数。
        boost_channel:   增强的通道索引（B=0, G=1, R=2）。
        boost_strength:  增强力度（1.0 = 最大翻倍）。
        suppress_channel: 抑制的通道索引。
        suppress_strength: 抑制力度（1.0 = 最大压到 0）。
    """
    result = bgr.copy()
    result[:, :, boost_channel] *= (1.0 + mask * factor * boost_strength)
    result[:, :, suppress_channel] *= (1.0 - mask * factor * suppress_strength)
    return np.clip(result, 0.0, 1.0)


# ============================================================
#  主处理函数
# ============================================================

def enhance_image(src_path: str, brightness: float, contrast: float,
                  saturation: float,
                  water_factor: float | None = None,
                  vegetation_factor: float | None = None) -> str:
    """对遥感影像执行完整的色调增强处理。

    处理顺序：
    1. 读取 → 归一化 → 分离 RGB / 额外波段
    2. 提取原始 HSV 特征（用于水体检测）
    3. 水体区域增强（增强 B 通道、抑制 R 通道）
    4. 植被区域增强（增强 G 通道、抑制 R 通道）
    5. 整体色调调整（亮度 gamma + 对比度 S 曲线 + 饱和度线性缩放）
    6. 还原数据类型 → rasterio 写出

    Args:
        src_path: 输入 TIFF 文件路径。
        brightness: 亮度参数 [0, 2]，默认 0.6。
        contrast: 对比度参数 [0, 1]，默认 0.4。
        saturation: 饱和度参数 [-1, 1]，默认 0（中性）。
        water_factor: 水体增强系数，默认 None（自动使用内部默认值）。
        vegetation_factor: 植被增强系数，默认 None（自动使用内部默认值）。

    Returns:
        输出影像文件路径（与输入同目录，文件名加 _Enhance 后缀）。

    Raises:
        EnhancementError: 读取、处理或写出过程中发生错误。
    """
    # 水体/植被使用内部默认值
    if water_factor is None:
        water_factor = DEFAULT_WATER_FACTOR
    if vegetation_factor is None:
        vegetation_factor = DEFAULT_VEGETATION_FACTOR
    # ------------------------------------------------------------------
    # 1. 读取影像
    # ------------------------------------------------------------------
    try:
        with rasterio.open(src_path) as src:
            image = src.read()           # (bands, H, W)
            profile = src.profile.copy()
    except Exception as exc:
        raise EnhancementError(f"无法读取影像文件: {exc}")

    bands, height, width = image.shape
    if bands < 3:
        raise EnhancementError(
            f"影像仅含 {bands} 个波段，需要至少 3 个波段（RGB）"
        )

    src_dtype = image.dtype

    # ------------------------------------------------------------------
    # 2. 归一化 & 波段分离
    # ------------------------------------------------------------------
    image_norm = _normalize_to_float(image, src_dtype)

    rgb = image_norm[:3]            # (3, H, W)
    others = image_norm[3:]         # (B-3, H, W)，可能存在额外波段

    # (Bands, H, W) → (H, W, 3) RGB → BGR
    rgb_hwc = np.transpose(rgb, (1, 2, 0))
    bgr = rgb_hwc[:, :, ::-1].copy()

    # ------------------------------------------------------------------
    # 3. 从原始 BGR 提取 HSV（用于水体检测，保证检测基于原始影像）
    # ------------------------------------------------------------------
    hsv = _bgr_to_hsv(bgr)
    _, s_val, v_val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # ------------------------------------------------------------------
    # 4. 水体增强（先做区域色彩调整，再做整体色调）
    # ------------------------------------------------------------------
    if water_factor > 0.0:
        water_mask = _detect_water_mask(
            bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2],
            v_val, s_val,
        )
        bgr = _apply_region_enhancement(
            bgr, water_mask, water_factor,
            boost_channel=0,   boost_strength=0.5,   # B 通道
            suppress_channel=2, suppress_strength=0.3, # R 通道
        )

    # ------------------------------------------------------------------
    # 5. 植被增强
    # ------------------------------------------------------------------
    if vegetation_factor > 0.0:
        veg_mask = _detect_vegetation_mask(
            bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2],
        )
        bgr = _apply_region_enhancement(
            bgr, veg_mask, vegetation_factor,
            boost_channel=1,   boost_strength=0.5,   # G 通道
            suppress_channel=2, suppress_strength=0.2, # R 通道
        )

    # ------------------------------------------------------------------
    # 6. 色调增强（亮度 + 对比度 + 饱和度）—— 最后做，覆盖区域调整的生硬边界
    # ------------------------------------------------------------------
    bgr = _apply_tone_enhancement(bgr, brightness, contrast, saturation)

    # ------------------------------------------------------------------
    # 7. 还原 RGB 波段并合并额外波段
    # ------------------------------------------------------------------
    rgb_enhanced = bgr[:, :, ::-1]                       # BGR → RGB
    rgb_enhanced = np.transpose(rgb_enhanced, (2, 0, 1))  # (H, W, 3) → (3, H, W)

    if others.shape[0] > 0:
        result = np.concatenate([rgb_enhanced, others], axis=0)
    else:
        result = rgb_enhanced

    # ------------------------------------------------------------------
    # 8. 还原数据类型并写出
    # ------------------------------------------------------------------
    result = _denormalize_from_float(result, src_dtype)

    src_path_obj = Path(src_path)
    dst_path = src_path_obj.parent / f"{src_path_obj.stem}{OUTPUT_SUFFIX}{OUTPUT_EXTENSION}"

    profile.update(
        driver="GTiff",
        dtype=src_dtype,
        count=bands,
        height=height,
        width=width,
        compress="lzw",
    )
    try:
        with rasterio.open(str(dst_path), "w", **profile) as dst:
            dst.write(result)
    except Exception as exc:
        raise EnhancementError(f"写出增强影像失败: {exc}")

    return str(dst_path)
