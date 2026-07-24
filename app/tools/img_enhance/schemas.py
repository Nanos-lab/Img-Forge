"""影像增强模块的数据模型。"""

from pydantic import BaseModel, Field


class EnhanceParams(BaseModel):
    """影像增强请求参数 —— 三个维度统一 [-1, 1]，0 为中性。

    水体和植被增强使用内部默认值，不在此暴露。
    """

    brightness: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "亮度调整系数，gamma = 2^(-brightness)。"
            "0 为中性（无变化），正值提亮，负值压暗。"
            "每 ±1 单位 ≈ 一档曝光量变化。"
        ),
    )
    contrast: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "对比度调整系数。正值通过 S 曲线增强中间调反差，"
            "负值线性压向中灰以降低对比度。0 为中性（无调整）。"
        ),
    )
    saturation: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "饱和度调整系数，S_out = S × (1 + saturation)。"
            "0 为中性（无变化），正值提升饱和度，负值降低饱和度。"
            "saturation = -1 时完全去色为灰度。"
        ),
    )


class EnhanceResult(BaseModel):
    """影像增强处理结果。"""

    message: str = Field(description="处理结果描述")
    original_filename: str = Field(description="原始文件名")
    output_filename: str = Field(description="输出文件名")
