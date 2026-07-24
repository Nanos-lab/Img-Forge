"""影像拼接模块的 API 路由。

POST /tools/mosaic/ — 上传多个 TIFF，返回拼接后的 TIFF 文件。
"""

import shutil
from pathlib import Path
from typing import List
from urllib.parse import quote

import os

from fastapi import APIRouter, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse

from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, TEMP_DIR
from app.core.exceptions import (
    MosaicError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from app.tools.img_mosaic.service import merge_tifs

router = APIRouter(prefix="/tools/mosaic", tags=["影像拼接"])


def _validate_upload(filename: str, file_size: int) -> None:
    """校验上传文件的格式与大小。"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(filename, ALLOWED_EXTENSIONS)
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise FileTooLargeError(file_size, MAX_UPLOAD_SIZE_BYTES)


@router.post(
    "/",
    summary="遥感影像拼接",
    description=(
        "上传多个带地理参考的 TIFF 影像，根据坐标信息自动拼接为一张完整影像。"
        "重叠区域后覆盖前（最后上传的影像在最上层）。"
        "要求所有影像 CRS 一致，分辨率会自动统一为第一张影像的分辨率。"
        "输出影像保留地理参考信息。"
    ),
)
async def mosaic(
    files: List[UploadFile] = File(..., description="待拼接的遥感影像列表（.tif / .tiff）"),
    background_tasks: BackgroundTasks = None,
) -> FileResponse:
    if not files:
        raise MosaicError("至少需要上传 1 个影像文件")

    # --- 校验所有文件 ---
    for f in files:
        if not f.filename:
            raise MosaicError("存在未命名的文件")
        _validate_upload(f.filename, f.size or 0)

    # --- 暂存所有上传文件 ---
    temp_paths: list[str] = []
    original_stems: list[str] = []
    try:
        for idx, f in enumerate(files):
            stem = Path(f.filename).stem
            original_stems.append(stem)
            temp_input = TEMP_DIR / f"mosaic_{idx}_{stem}_input.tif"
            with open(temp_input, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            temp_paths.append(str(temp_input))

        # --- 执行拼接 ---
        output_path = merge_tifs(temp_paths, original_stems=original_stems)

    except MosaicError:
        raise
    except Exception as exc:
        raise MosaicError(f"拼接过程中发生未知错误: {exc}")
    finally:
        # 清理所有临时输入文件
        for tp in temp_paths:
            Path(tp).unlink(missing_ok=True)

    # --- 返回拼接后的影像（发送完成后自动清理） ---
    output_name = Path(output_path).name
    if background_tasks:
        background_tasks.add_task(os.remove, output_path)
    return FileResponse(
        path=output_path,
        filename=output_name,
        media_type="image/tiff",
        headers={
            "X-Input-Count": str(len(files)),
            "X-Output-Filename": quote(output_name, safe=""),
        },
    )
