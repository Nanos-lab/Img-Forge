"""img_changedet 模块的数据模型。"""

from pydantic import BaseModel, Field


class ChangeDetectResult(BaseModel):
    """变化检测处理结果。"""
    message: str = Field(description="处理结果描述")
    old_filename: str = Field(description="变化前影像文件名")
    new_filename: str = Field(description="变化后影像文件名")
    change_count: int = Field(description="检测到的变化区域数量")
    elapsed_ms: int = Field(description="处理耗时（毫秒）")
