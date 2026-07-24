"""正射校正模块的 API 路由。

POST /tools/ortho/ — 上传原始遥感影像 + 可选 RPC 文件，返回正射校正后的 TIFF 文件。

RPC 来源优先级：
1. 上传的 RPC 文件（外置）
2. TIFF 内嵌的 RPC 标签（自动读取）
"""

import os
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse

from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, TEMP_DIR
from app.core.exceptions import (
    OrthoError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from app.tools.img_ortho.service import orthorectify

router = APIRouter(prefix="/tools/ortho", tags=["正射校正"])

# RPC 文件允许的扩展名
RPC_EXTENSIONS = {".rpc", ".rpb", ".txt"}


def _validate_upload(filename: str, file_size: int) -> None:
    """校验上传文件的格式与大小。"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(filename, ALLOWED_EXTENSIONS)
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise FileTooLargeError(file_size, MAX_UPLOAD_SIZE_BYTES)


@router.post(
    "/",
    summary="RPC 正射校正",
    description=(
        "上传原始遥感影像进行正射校正。"
        "RPC 可上传外置文件或使用 TIFF 内嵌标签；"
        "可选上传 DEM 文件用于精确地形校正。"
        "输出默认使用 CGCS2000 高斯-克吕格 3 度带投影。"
    ),
)
async def ortho(
    file: UploadFile = File(..., description="原始遥感影像（.tif / .tiff）"),
    rpc_file: UploadFile = File(default=None, description="RPC 系数文件（.rpc / .rpb，可选，不传则使用 TIFF 内嵌 RPC）"),
    dem_file: UploadFile = File(default=None, description="DEM 高程文件（.tif，可选）"),
    height: float = Form(default=0.0, ge=0.0, description="无 DEM 时的默认高程（米）"),
    dst_crs: str = Form(default="auto", description="输出投影，auto=CGCS2000 高斯-克吕格，或 EPSG:xxxx，或 Proj4 字符串"),
    resolution: float = Form(default=None, ge=0.0, description="输出分辨率（地理单位/像素），默认自动计算"),
    background_tasks: BackgroundTasks = None,
) -> FileResponse:
    if not file.filename:
        raise OrthoError("未提供影像文件名")

    # --- 校验文件 ---
    _validate_upload(file.filename, file.size or 0)

    if rpc_file and rpc_file.filename:
        rpc_ext = Path(rpc_file.filename).suffix.lower()
        if rpc_ext not in RPC_EXTENSIONS:
            raise UnsupportedFormatError(rpc_file.filename, RPC_EXTENSIONS)

    if dem_file:
        _validate_upload(dem_file.filename, dem_file.size or 0)

    # --- 暂存所有上传文件 ---
    src_stem = Path(file.filename).stem
    temp_input = TEMP_DIR / f"ortho_{src_stem}_input.tif"
    temp_rpc: Path | None = None
    temp_dem: Path | None = None

    temp_paths: list[Path] = [temp_input]

    try:
        # 保存影像
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 保存 RPC 文件（可选）
        if rpc_file and rpc_file.filename:
            temp_rpc = TEMP_DIR / f"ortho_{src_stem}_input{Path(rpc_file.filename).suffix}"
            with open(temp_rpc, "wb") as buffer:
                shutil.copyfileobj(rpc_file.file, buffer)
            temp_paths.append(temp_rpc)

        # 保存 DEM（可选）
        if dem_file:
            temp_dem = TEMP_DIR / f"ortho_{src_stem}_dem_input.tif"
            with open(temp_dem, "wb") as buffer:
                shutil.copyfileobj(dem_file.file, buffer)
            temp_paths.append(temp_dem)

        # --- 执行正射校正 ---
        output_path = orthorectify(
            src_path=str(temp_input),
            rpc_path=str(temp_rpc) if temp_rpc else None,
            dst_crs=dst_crs,
            dem_path=str(temp_dem) if temp_dem else None,
            height=height,
            resolution=resolution,
        )

    except OrthoError:
        raise
    except Exception as exc:
        raise OrthoError(f"处理过程中发生未知错误: {exc}")
    finally:
        # 清理所有临时输入文件
        for tp in temp_paths:
            tp.unlink(missing_ok=True)

    # --- 返回正射校正后的影像（发送完成后自动清理） ---
    output_name = Path(output_path).name
    if background_tasks:
        background_tasks.add_task(os.remove, output_path)
    return FileResponse(
        path=output_path,
        filename=output_name,
        media_type="image/tiff",
        headers={
            "X-Original-Filename": quote(file.filename, safe=""),
            "X-Output-Filename": quote(output_name, safe=""),
            "X-Output-CRS": quote(dst_crs, safe=""),
        },
    )
