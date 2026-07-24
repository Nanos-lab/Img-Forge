"""img_enhance_2 模块 —— Cesium 风格色调增强。

摒弃水体/植被区域检测，采用 Cesium 风格的色调映射曲线：
- 亮度：gamma 校正（幂律曲线）
- 对比度：sigmoid S 曲线
- 饱和度：亮度保持的 RGB 混合

接口与 img_enhance 完全一致，可直接替换。
"""

from app.tools.img_enhance_2.router import router

__all__ = ["router"]
