"""img_mosaic 模块的数据模型。"""

from pydantic import BaseModel, Field


class MosaicResult(BaseModel):
    """拼接处理结果。"""

    message: str = Field(description="处理结果描述")
    input_count: int = Field(description="输入影像数")
    output_filename: str = Field(description="输出文件名")
    output_width: int = Field(description="输出影像宽度（像素）")
    output_height: int = Field(description="输出影像高度（像素）")
    band_count: int = Field(description="输出影像波段数")
