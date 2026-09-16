"""img_registration —— 相位相关配准算法（共享算法层，Layer 1）。

提供纯函数 phase_correlate()，接收两个影像 patch，返回亚像素偏移量。
无路由、无业务逻辑，供 tools/ 下的业务模块（img_mosaic、img_ortho、
img_changedet 等）复用，不反向依赖任何业务模块。
"""

from app.shared.img_registration.service import phase_correlate

__all__ = ["phase_correlate"]
