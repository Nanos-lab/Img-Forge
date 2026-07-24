"""统一 API 响应模型。"""

from typing import Any, Optional

from pydantic import BaseModel


class APIResponse(BaseModel):
    """统一成功响应。"""

    code: int = 200
    message: str = "success"
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    """统一错误响应。"""

    code: int
    message: str
    detail: Optional[str] = None


class EnhanceResultData(BaseModel):
    """增强处理结果数据。"""

    original_filename: str
    output_filename: str
    width: int
    height: int
    band_count: int
    crs: Optional[str] = None
