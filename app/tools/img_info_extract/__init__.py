"""img_info_extract 模块 —— 图片信息提取（基于大模型视觉理解）。

上传单张图片，提取目标名称、标题、分辨率、平台、时间、比例尺六项信息，
图中未标注的信息对应字段返回 null。
"""

from app.tools.img_info_extract.router import router

__all__ = ["router"]
