"""img_enhance_2 模块的 API 路由。

POST /tools/enhance/ — 接口与 img_enhance 完全一致。
"""

import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse

from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, TEMP_DIR
from app.core.exceptions import (
    EnhancementError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from app.tools.img_enhance_2.service import enhance_image

router = APIRouter(prefix="/tools/enhance", tags=["影像增强"])


def _validate_upload(filename: str, file_size: int) -> None:
    """校验上传文件的格式与大小。"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(filename, ALLOWED_EXTENSIONS)
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise FileTooLargeError(file_size, MAX_UPLOAD_SIZE_BYTES)


@router.post(
    "/",
    summary="遥感影像色调增强",
    description=(
        "对上传的遥感影像（TIFF 格式）进行 Cesium 风格色调增强。"
        "调整亮度、对比度、饱和度，无水体/植被区域检测。"
        "输出影像保留原始的投影、分辨率和波段数。"
    ),
)
async def enhance(
    file: UploadFile = File(..., description="待增强的遥感影像（.tif / .tiff）"),
    brightness: float = Form(
        default=1.0, ge=0.0, le=3.0,
        description="亮度系数。1 原始值，0 全黑，上限 3",
    ),
    contrast: float = Form(
        default=1.0, ge=0.0, le=5.0,
        description="对比度系数。1 原始值，0 统一灰色，上限 5",
    ),
    saturation: float = Form(
        default=1.0, ge=0.0, le=4.0,
        description="饱和度系数。1 原始值，0 完全灰度，上限 4",
    ),
) -> FileResponse:
    if not file.filename:
        raise EnhancementError("未提供文件名")
    _validate_upload(file.filename, file.size or 0)

    temp_input = TEMP_DIR / f"{Path(file.filename).stem}_input.tif"
    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise EnhancementError(f"保存上传文件失败: {exc}")

    try:
        output_path = enhance_image(
            src_path=str(temp_input),
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
        )
    except EnhancementError:
        raise
    except Exception as exc:
        raise EnhancementError(f"处理过程中发生未知错误: {exc}")
    finally:
        if temp_input.exists():
            temp_input.unlink(missing_ok=True)

    output_name = Path(output_path).name
    return FileResponse(
        path=output_path,
        filename=output_name,
        media_type="image/tiff",
        headers={
            "X-Original-Filename": quote(file.filename or "unknown", safe=""),
            "X-Output-Filename": quote(output_name, safe=""),
        },
    )
