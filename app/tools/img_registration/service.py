"""相位相关配准服务层 —— 纯相位相关算法。

与拼接业务解耦，仅提供纯函数 phase_correlate()，
其他模块可通过 import 复用。
"""

import cv2
import numpy as np


# ============================================================
#  内部辅助
# ============================================================


def _to_grayscale(patch: np.ndarray) -> np.ndarray:
    """将多波段影像转为单波段灰度 float32。

    Args:
        patch: 输入影像块，shape (bands, H, W)。

    Returns:
        灰度 float32 数组，shape (H, W)。
    """
    if patch.shape[0] >= 3:
        gray = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]
    else:
        gray = patch[0]
    return gray.astype(np.float32)


# ============================================================
#  对外接口
# ============================================================


def phase_correlate(
    ref_patch: np.ndarray,
    src_patch: np.ndarray,
) -> tuple[float, float, float]:
    """计算 src_patch 相对于 ref_patch 的亚像素偏移。

    内部自动转灰度、加 Hanning 窗后调用 cv2.phaseCorrelate。

    Args:
        ref_patch: 参考影像块，shape (bands, H, W) 或 (H, W)。
        src_patch: 待配准影像块，shape (bands, H, W) 或 (H, W)。
                   尺寸须与 ref_patch 一致。

    Returns:
        (delta_row, delta_col, response):
        - delta_row: 行方向偏移修正量（亚像素），正值=下移
        - delta_col: 列方向偏移修正量（亚像素），正值=右移
        - response:  相位相关响应值 [0, 1]，越高越可靠
    """
    # 确保灰度
    if ref_patch.ndim == 3:
        ref_gray = _to_grayscale(ref_patch)
    else:
        ref_gray = ref_patch.astype(np.float32)

    if src_patch.ndim == 3:
        src_gray = _to_grayscale(src_patch)
    else:
        src_gray = src_patch.astype(np.float32)

    h, w = ref_gray.shape

    # Hanning 窗降低边缘效应
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)

    try:
        (dx, dy), response = cv2.phaseCorrelate(ref_gray, src_gray, window=window)
    except cv2.error:
        return 0.0, 0.0, 0.0

    # dx, dy 表示 src_patch 相对于 ref_patch 的偏移（列, 行）
    # 要抵消此偏移，修正量为反向
    return float(-dy), float(-dx), float(response)
