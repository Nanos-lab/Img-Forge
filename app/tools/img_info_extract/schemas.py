"""img_info_extract 模块的数据模型。"""

from pydantic import BaseModel, Field


class ImageInfoResult(BaseModel):
    """图片信息提取结果。六个字段均允许为空（图中未标注该信息时）。"""

    target: str | None = Field(
        default=None, description="图片中主要拍摄目标的名称"
    )
    title: str | None = Field(
        default=None, description="图片标题"
    )
    resolution: str | None = Field(
        default=None, description="影像分辨率信息"
    )
    platform: str | None = Field(
        default=None, description="拍摄平台/卫星/传感器名称"
    )
    time: str | None = Field(
        default=None, description="拍摄或成图时间"
    )
    scale: str | None = Field(
        default=None, description="比例尺信息"
    )
