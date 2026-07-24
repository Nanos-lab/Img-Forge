"""img_enhance_2 模块的数据模型。

接口与 img_enhance.schemas 一致。
"""

from pydantic import BaseModel, Field


class EnhanceParams(BaseModel):
    """影像增强请求参数 —— 三个维度统一 [-1, 1]，0 为中性。"""

    brightness: float = Field(
        default=1.0,
        ge=0.0,
        le=3.0,
        description="亮度系数。1 为原始值，0 为全黑，上限 3。",
    )
    contrast: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="对比度系数。1 为原始值，0 为统一灰色，上限 5。",
    )
    saturation: float = Field(
        default=1.0,
        ge=0.0,
        le=4.0,
        description="饱和度系数。1 为原始值，0 为完全灰度，上限 4。",
    )


class EnhanceResult(BaseModel):
    """影像增强处理结果。"""

    message: str = Field(description="处理结果描述")
    original_filename: str = Field(description="原始文件名")
    output_filename: str = Field(description="输出文件名")
