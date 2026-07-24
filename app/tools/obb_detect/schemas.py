"""obb_detect 模块的数据模型。"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DetectParams(BaseModel):
    """目标检测请求参数。"""

    classes: Optional[str] = Field(
        default="0,1,7,8",
        description=(
            "检测类别，逗号分隔的 DOTA 类别 ID。"
            "0:plane 1:ship 7:harbor 8:bridge"
        ),
    )
    confidence: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="置信度阈值，低于此值的检测结果被过滤。默认 0.25。",
    )


class DetectResult(BaseModel):
    """目标检测处理结果。"""

    message: str = Field(description="处理结果描述")
    original_filename: str
    total_detections: int
    class_counts: dict
    elapsed_seconds: float
