"""img_pansharpen 模块的数据模型。"""

from pydantic import BaseModel, Field


class PansharpenParams(BaseModel):
    """全色锐化请求参数。"""

    weights: str | None = Field(
        default=None,
        description="MS 各波段合成全色分量的权重，逗号分隔，长度需等于 MS 波段数。不传则等权重。",
    )


class PansharpenResult(BaseModel):
    """全色锐化处理结果。"""

    message: str = Field(description="处理结果描述")
    ms_filename: str = Field(description="多光谱原始文件名")
    pan_filename: str = Field(description="全色原始文件名")
    output_filename: str = Field(description="输出文件名")
