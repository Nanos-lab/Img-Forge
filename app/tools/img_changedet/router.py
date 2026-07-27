"""img_changedet 模块的 API 路由。

POST /tools/changedet/ — 上传两期遥感影像，返回 GeoJSON 格式的变化区域。
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, TEMP_DIR
from app.core.exceptions import (
    ChangeDetectError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from app.tools.img_changedet.service import detect_changes

router = APIRouter(prefix="/tools/changedet", tags=["变化检测"])


def _validate_upload(filename: str, file_size: int) -> None:
    """校验上传文件的格式与大小。"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(filename, ALLOWED_EXTENSIONS)
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise FileTooLargeError(file_size, MAX_UPLOAD_SIZE_BYTES)


@router.post(
    "/",
    summary="遥感影像变化检测",
    description=(
        "上传两期遥感影像（变化前 + 变化后），自动进行配准、"
        "分块推理、变化区域提取，返回 GeoJSON 格式的变化区域。"
    ),
)
async def changedet(
    old: UploadFile = File(..., description="变化前遥感影像（.tif / .tiff）"),
    new: UploadFile = File(..., description="变化后遥感影像（.tif / .tiff）"),
    threshold: float = Form(default=None, ge=0.0, le=1.0, description="变化概率二值化阈值，默认 Otsu 自动"),
    min_area: int = Form(default=None, ge=1, description="最小变化区域面积（像素），默认自动"),
) -> JSONResponse:
    if not old.filename:
        raise ChangeDetectError("未提供变化前影像文件名")
    if not new.filename:
        raise ChangeDetectError("未提供变化后影像文件名")

    _validate_upload(old.filename, old.size or 0)
    _validate_upload(new.filename, new.size or 0)

    temp_old = TEMP_DIR / f"cd_{Path(old.filename).stem}_old.tif"
    temp_new = TEMP_DIR / f"cd_{Path(new.filename).stem}_new.tif"
    temp_paths = [temp_old, temp_new]

    try:
        with open(temp_old, "wb") as buffer:
            shutil.copyfileobj(old.file, buffer)
        with open(temp_new, "wb") as buffer:
            shutil.copyfileobj(new.file, buffer)

        geojson = detect_changes(
            old_path=str(temp_old),
            new_path=str(temp_new),
            threshold=threshold,
            min_area=min_area,
        )

    except ChangeDetectError:
        raise
    except Exception as exc:
        raise ChangeDetectError(f"变化检测处理失败: {exc}")
    finally:
        for tp in temp_paths:
            tp.unlink(missing_ok=True)

    return JSONResponse(content=geojson)
