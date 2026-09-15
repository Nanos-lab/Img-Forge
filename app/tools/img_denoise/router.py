"""img_denoise 模块的 API 路由。

POST /tools/denoise/
"""

import os
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse

from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, TEMP_DIR
from app.core.exceptions import (
    DenoiseError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from app.tools.img_denoise.service import denoise_image

router = APIRouter(prefix="/tools/denoise", tags=["噪声抑制与边缘增强"])


def _validate_upload(filename: str, file_size: int) -> None:
    """校验上传文件的格式与大小。"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(filename, ALLOWED_EXTENSIONS)
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise FileTooLargeError(file_size, MAX_UPLOAD_SIZE_BYTES)


@router.post(
    "/",
    summary="遥感影像噪声抑制与边缘增强",
    description=(
        "对上传的遥感影像（TIFF 格式）进行噪声抑制（双边滤波，保边去噪）"
        "和边缘增强（反锐化蒙版）。逐波段独立处理，不依赖颜色语义，"
        "适用于单波段灰度图、RGB、多光谱等任意波段数的影像。"
        "输出影像保留原始的投影、分辨率和波段数。"
    ),
)
async def denoise(
    file: UploadFile = File(..., description="待处理的遥感影像（.tif / .tiff）"),
    denoise: float = Form(
        default=0.0, ge=0.0, le=1.0,
        description="降噪强度。0 不处理，1 最强",
    ),
    sharpen: float = Form(
        default=0.0, ge=0.0, le=3.0,
        description="锐化强度。0 不处理，值越大边缘增强越明显",
    ),
    background_tasks: BackgroundTasks = None,
) -> FileResponse:
    if not file.filename:
        raise DenoiseError("未提供文件名")
    _validate_upload(file.filename, file.size or 0)

    temp_input = TEMP_DIR / f"{Path(file.filename).stem}_input.tif"
    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise DenoiseError(f"保存上传文件失败: {exc}")

    try:
        output_path = denoise_image(
            src_path=str(temp_input),
            denoise=denoise,
            sharpen=sharpen,
        )
    except DenoiseError:
        raise
    except Exception as exc:
        raise DenoiseError(f"处理过程中发生未知错误: {exc}")
    finally:
        if temp_input.exists():
            temp_input.unlink(missing_ok=True)

    output_name = Path(output_path).name
    if background_tasks:
        background_tasks.add_task(os.remove, output_path)
    return FileResponse(
        path=output_path,
        filename=output_name,
        media_type="image/tiff",
        headers={
            "X-Original-Filename": quote(file.filename or "unknown", safe=""),
            "X-Output-Filename": quote(output_name, safe=""),
        },
    )
