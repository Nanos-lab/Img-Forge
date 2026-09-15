"""img_denoise 模块的数据模型。"""

from pydantic import BaseModel, Field


class DenoiseParams(BaseModel):
    """噪声抑制与边缘增强请求参数。"""

    denoise: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="降噪强度。0 不处理，1 最强（双边滤波，保边去噪）。",
    )
    sharpen: float = Field(
        default=0.0,
        ge=0.0,
        le=3.0,
        description="锐化强度。0 不处理，值越大边缘增强越明显（反锐化蒙版）。",
    )


class DenoiseResult(BaseModel):
    """噪声抑制与边缘增强处理结果。"""

    message: str = Field(description="处理结果描述")
    original_filename: str = Field(description="原始文件名")
    output_filename: str = Field(description="输出文件名")
