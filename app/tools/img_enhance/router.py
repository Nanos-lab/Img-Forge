"""影像增强模块的 API 路由。

POST /tools/enhance/ — 上传遥感影像，返回增强后的 TIFF 文件。
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
from app.tools.img_enhance.service import enhance_image

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
        "对上传的遥感影像（TIFF 格式）进行基于曲线映射的色调增强。"
        "调整亮度、对比度、饱和度，同时内部分析水体/植被区域进行"
        "针对性的色彩优化，改善影像的目视判读效果。"
        "输出影像保留原始的投影、分辨率和波段数。"
    ),
)
async def enhance(
    file: UploadFile = File(..., description="待增强的遥感影像（.tif / .tiff）"),
    brightness: float = Form(
        default=0.0, ge=-1.0, le=1.0,
        description="亮度，gamma = 2^(-brightness)。0 中性，正值提亮，负值压暗",
    ),
    contrast: float = Form(
        default=0.0, ge=-1.0, le=1.0,
        description="对比度。正值 S 曲线增强，负值压向中灰降低对比度。0 中性",
    ),
    saturation: float = Form(
        default=0.0, ge=-1.0, le=1.0,
        description="饱和度，S_out = S × (1 + s)。0 中性，正值鲜艳，负值趋灰",
    ),
) -> FileResponse:
    # --- 校验上传文件 ---
    if not file.filename:
        raise EnhancementError("未提供文件名")
    _validate_upload(file.filename, file.size or 0)

    # --- 保存上传文件到临时目录 ---
    temp_input = TEMP_DIR / f"{Path(file.filename).stem}_input.tif"
    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise EnhancementError(f"保存上传文件失败: {exc}")

    # --- 执行增强处理 ---
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
        # 清理临时输入文件
        if temp_input.exists():
            temp_input.unlink(missing_ok=True)

    # --- 返回增强后的影像 ---
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
