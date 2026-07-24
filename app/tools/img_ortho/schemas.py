"""img_ortho 模块的数据模型。"""

from pydantic import BaseModel, Field


class OrthoResult(BaseModel):
    """正射校正处理结果。"""

    message: str = Field(description="处理结果描述")
    original_filename: str = Field(description="原始影像文件名")
    output_filename: str = Field(description="输出文件名")
    output_crs: str = Field(description="输出投影坐标系")
    output_width: int = Field(description="输出影像宽度（像素）")
    output_height: int = Field(description="输出影像高度（像素）")
