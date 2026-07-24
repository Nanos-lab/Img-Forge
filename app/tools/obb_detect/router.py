"""OBB 目标检测模块的 API 路由。

POST /tools/obb-detect/ — 上传遥感影像，返回 GeoJSON 检测结果。
"""

import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, TEMP_DIR
from app.core.exceptions import (
    DetectionError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from app.tools.obb_detect.service import detect_objects

router = APIRouter(prefix="/tools/obb-detect", tags=["目标检测"])


def _validate_upload(filename: str, file_size: int) -> None:
    """校验上传文件的格式与大小。"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(filename, ALLOWED_EXTENSIONS)
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise FileTooLargeError(file_size, MAX_UPLOAD_SIZE_BYTES)


@router.post(
    "/",
    summary="遥感影像目标检测",
    description=(
        "对上传的遥感影像（TIFF 格式）进行 YOLOv8-OBB 旋转目标检测。"
        "支持飞机、舰船、港口、桥梁四类目标，"
        "采用滑动窗口分块策略处理大幅面影像，返回 GeoJSON 格式的检测结果。"
    ),
)
async def detect(
    file: UploadFile = File(..., description="待检测的遥感影像（.tif / .tiff）"),
    classes: str = Form(
        default="0,1,7,8",
        description="检测类别，逗号分隔的 DOTA ID。0:plane 1:ship 7:harbor 8:bridge",
    ),
    confidence: float = Form(
        default=0.25, ge=0.01, le=1.0,
        description="置信度阈值，0.25 为推荐值",
    ),
) -> JSONResponse:
    # --- 校验 ---
    if not file.filename:
        raise DetectionError("未提供文件名")
    _validate_upload(file.filename, file.size or 0)

    # --- 解析类别 ---
    class_ids = []
    if classes.strip():
        try:
            class_ids = [int(c.strip()) for c in classes.split(",") if c.strip()]
        except ValueError:
            raise DetectionError(f"无效的类别参数: '{classes}'，应为逗号分隔的整数")

    # --- 暂存上传文件 ---
    temp_input = TEMP_DIR / f"{Path(file.filename).stem}_input.tif"
    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise DetectionError(f"保存上传文件失败: {exc}")

    # --- 执行检测 ---
    try:
        geojson = detect_objects(
            src_path=str(temp_input),
            classes=class_ids,
            confidence=confidence,
        )
    except DetectionError:
        raise
    except Exception as exc:
        raise DetectionError(f"检测过程中发生未知错误: {exc}")
    finally:
        if temp_input.exists():
            temp_input.unlink(missing_ok=True)

    return JSONResponse(content=geojson)
