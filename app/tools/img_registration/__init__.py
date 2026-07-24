"""img_registration 模块 —— 相位相关配准算法。

提供纯函数 phase_correlate()，接收两个影像 patch，返回亚像素偏移量。
不涉及画布、掩膜等拼接业务逻辑，可被 img_mosaic 及其他模块复用。
"""

from app.tools.img_registration.service import phase_correlate

__all__ = ["phase_correlate"]
