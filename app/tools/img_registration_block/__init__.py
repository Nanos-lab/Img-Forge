"""img_registration_block 模块 —— 分块相位相关配准 + 位移场插值。

功能同 img_registration，但内部采用分块计算+位移场插值，
能处理局部形变（建筑与地面不同偏移量）。
"""

from app.tools.img_registration_block.service import block_warp, phase_correlate

__all__ = ["block_warp", "phase_correlate"]
