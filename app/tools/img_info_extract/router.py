"""img_info_extract 模块的 API 路由。

POST /tools/info-extract/ — 上传单张图片，返回提取到的图片信息 JSON。
"""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from app.core.config import (
    INFO_EXTRACT_ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
    TEMP_DIR,
)
from app.core.exceptions import (
    FileTooLargeError,
    InfoExtractError,
    UnsupportedFormatError,
)
from app.tools.img_info_extract.service import extract_image_info

router = APIRouter(prefix="/tools/info-extract", tags=["图片信息提取"])

# 临时文件目录：用随机文件名落盘，规避中文/特殊字符文件名在
# OpenCV（Windows 下 cv2.imread 依赖 fopen，不支持非 ASCII 路径）
# 读取时失败的问题。
INFO_EXTRACT_TEMP_DIR = TEMP_DIR / "info_extract"
INFO_EXTRACT_TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _validate_upload(filename: str, file_size: int) -> None:
    """校验上传文件的格式与大小。"""
    ext = Path(filename).suffix.lower()
    if ext not in INFO_EXTRACT_ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(filename, INFO_EXTRACT_ALLOWED_EXTENSIONS)
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise FileTooLargeError(file_size, MAX_UPLOAD_SIZE_BYTES)


@router.post(
    "/",
    summary="图片信息提取",
    description=(
        "上传单张图片（.jpg/.jpeg/.png/.tif/.tiff），基于大模型视觉理解"
        "提取目标名称、标题、分辨率、平台、时间、比例尺六项信息。"
        "图中未标注的信息对应字段返回 null。"
    ),
)
async def info_extract(
    file: UploadFile = File(..., description="待提取信息的图片"),
) -> JSONResponse:
    if not file.filename:
        raise InfoExtractError("未提供文件名")
    _validate_upload(file.filename, file.size or 0)

    # 用随机文件名落盘，避免原始文件名含中文/特殊字符时
    # OpenCV 在 Windows 上读取失败。
    ext = Path(file.filename).suffix.lower()
    temp_input = INFO_EXTRACT_TEMP_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        try:
            with open(temp_input, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as exc:
            raise InfoExtractError(f"保存上传文件失败: {exc}")

        try:
            result = extract_image_info(str(temp_input))
        except InfoExtractError:
            raise
        except Exception as exc:
            raise InfoExtractError(f"处理过程中发生未知错误: {exc}")
    finally:
        temp_input.unlink(missing_ok=True)

    return JSONResponse(content=result.model_dump())
