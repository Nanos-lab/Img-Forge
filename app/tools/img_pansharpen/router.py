"""img_pansharpen 模块的 API 路由。

POST /tools/pansharpen/ — 上传多光谱 + 全色影像，返回融合后的高分辨率多光谱 TIFF。
"""

import os
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse

from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, TEMP_DIR
from app.core.exceptions import (
    FileTooLargeError,
    PansharpenError,
    UnsupportedFormatError,
)
from app.tools.img_pansharpen.service import pansharpen_image

router = APIRouter(prefix="/tools/pansharpen", tags=["全色锐化"])


def _validate_upload(filename: str, file_size: int) -> None:
    """校验上传文件的格式与大小。"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(filename, ALLOWED_EXTENSIONS)
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise FileTooLargeError(file_size, MAX_UPLOAD_SIZE_BYTES)


def _parse_weights(weights: str | None) -> list[float] | None:
    """解析逗号分隔的权重字符串。"""
    if not weights:
        return None
    try:
        return [float(w) for w in weights.split(",")]
    except ValueError:
        raise PansharpenError(f"权重格式错误，应为逗号分隔的数字: '{weights}'")


@router.post(
    "/",
    summary="全色锐化（多光谱 + 全色融合）",
    description=(
        "上传配准好的低分辨率多光谱影像和高分辨率全色影像，"
        "基于 Gram-Schmidt 算法融合为高分辨率多光谱影像，兼顾光谱信息与空间细节。"
        "MS 会自动重采样对齐到 Pan 的地理网格，输出分辨率与 Pan 一致。"
    ),
)
async def pansharpen(
    ms: UploadFile = File(..., description="低分辨率多光谱影像（.tif / .tiff，多波段）"),
    pan: UploadFile = File(..., description="高分辨率全色影像（.tif / .tiff，单波段）"),
    weights: str = Form(
        default=None,
        description="MS 各波段合成全色分量的权重，逗号分隔，长度需等于 MS 波段数。不传则等权重。",
    ),
    background_tasks: BackgroundTasks = None,
) -> FileResponse:
    if not ms.filename or not pan.filename:
        raise PansharpenError("未提供文件名")
    _validate_upload(ms.filename, ms.size or 0)
    _validate_upload(pan.filename, pan.size or 0)

    parsed_weights = _parse_weights(weights)

    temp_ms = TEMP_DIR / f"{Path(ms.filename).stem}_ms_input.tif"
    temp_pan = TEMP_DIR / f"{Path(pan.filename).stem}_pan_input.tif"
    try:
        with open(temp_ms, "wb") as buffer:
            shutil.copyfileobj(ms.file, buffer)
        with open(temp_pan, "wb") as buffer:
            shutil.copyfileobj(pan.file, buffer)
    except Exception as exc:
        raise PansharpenError(f"保存上传文件失败: {exc}")

    try:
        output_path = pansharpen_image(
            ms_path=str(temp_ms),
            pan_path=str(temp_pan),
            weights=parsed_weights,
        )
    except PansharpenError:
        raise
    except Exception as exc:
        raise PansharpenError(f"处理过程中发生未知错误: {exc}")
    finally:
        temp_ms.unlink(missing_ok=True)
        temp_pan.unlink(missing_ok=True)

    output_name = Path(output_path).name
    if background_tasks:
        background_tasks.add_task(os.remove, output_path)
    return FileResponse(
        path=output_path,
        filename=output_name,
        media_type="image/tiff",
        headers={
            "X-MS-Filename": quote(ms.filename or "unknown", safe=""),
            "X-Pan-Filename": quote(pan.filename or "unknown", safe=""),
            "X-Output-Filename": quote(output_name, safe=""),
        },
    )
