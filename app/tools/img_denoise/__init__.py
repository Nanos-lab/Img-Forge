"""img_denoise 模块 —— 噪声抑制与边缘增强。

对任意波段数的遥感影像逐波段独立处理：
- 噪声抑制：双边滤波（保边去噪）
- 边缘增强：反锐化蒙版（Unsharp Masking）

不依赖颜色语义，单波段灰度图、RGB、多光谱影像均可处理。
"""

from app.tools.img_denoise.router import router

__all__ = ["router"]
