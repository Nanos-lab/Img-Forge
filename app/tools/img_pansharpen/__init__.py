"""img_pansharpen 模块 —— 全色锐化（Pan Sharpening，Gram-Schmidt 算法）。

输入配准好的低分辨率多光谱影像 + 高分辨率全色影像，融合为高分辨率
多光谱影像，兼顾光谱信息与空间细节。
"""

from app.tools.img_pansharpen.router import router

__all__ = ["router"]
