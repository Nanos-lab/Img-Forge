"""img_ortho 模块的 API 路由。

POST /tools/ortho/ — 几何校正。

根据 reference 的文件类型自动选择模式：
  - .rpc / .rpb / .txt → 正射校正模式（RPC）
  - .tif / .tiff       → 配准模式（将 file 配准到 reference）
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
from app.tools.img_ortho.service import orthorectify, register_to_ref

router = APIRouter(prefix="/tools/ortho", tags=["正射校正"])

TIFF_EXTENSIONS = {".tif", ".tiff"}
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
    summary="几何校正",
    description=(
        "上传 file + reference，根据 reference 文件类型自动切换模式：\n"
        "- .rpc/.rpb/.txt → RPC 正射校正\n"
        "- .tif/.tiff → 图像配准（将 file 配准到 reference）"
    ),
)
async def ortho(
    file: UploadFile = File(..., description="待处理的遥感影像（.tif / .tiff）"),
    reference: UploadFile = File(default=None, description="参考文件：RPC 文件（.rpc/.rpb）进行正射校正，或 TIFF 影像（.tif/.tiff）进行配准"),
    dem_file: UploadFile = File(default=None, description="DEM 高程文件（.tif，正射模式可选）"),
    height: float = Form(default=0.0, ge=0.0, description="无 DEM 时的默认高程（米）"),
    dst_crs: str = Form(default="auto", description="输出投影（正射模式），auto=CGCS2000 高斯-克吕格，或 EPSG:xxxx"),
    resolution: float = Form(default=None, ge=0.0, description="输出分辨率（正射模式），默认自动计算"),
    background_tasks: BackgroundTasks = None,
) -> FileResponse:
    if not file.filename:
        raise OrthoError("未提供影像文件名")
    _validate_upload(file.filename, file.size or 0)

    # --- 判断模式 ---
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix.lower()
        if ref_ext in TIFF_EXTENSIONS:
            mode = "registration"
            _validate_upload(reference.filename, reference.size or 0)
        elif ref_ext in RPC_EXTENSIONS:
            mode = "ortho"
        else:
            raise UnsupportedFormatError(reference.filename, TIFF_EXTENSIONS | RPC_EXTENSIONS)
    else:
        mode = "ortho"  # 无 reference 则尝试内嵌 RPC

    if dem_file:
        _validate_upload(dem_file.filename, dem_file.size or 0)

    # --- 暂存文件 ---
    src_stem = Path(file.filename).stem
    temp_input = TEMP_DIR / f"ortho_{src_stem}_input.tif"
    temp_paths: list[Path] = [temp_input]

    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if mode == "registration":
            # ---- 配准模式 ----
            temp_ref = TEMP_DIR / f"ortho_{Path(reference.filename).stem}_ref.tif"
            with open(temp_ref, "wb") as buffer:
                shutil.copyfileobj(reference.file, buffer)
            temp_paths.append(temp_ref)

            output_path = register_to_ref(
                src_path=str(temp_input),
                ref_path=str(temp_ref),
            )
        else:
            # ---- 正射模式 ----
            temp_rpc: Path | None = None
            temp_dem: Path | None = None

            if reference and reference.filename:
                temp_rpc = TEMP_DIR / f"ortho_{src_stem}_input{ref_ext}"
                with open(temp_rpc, "wb") as buffer:
                    shutil.copyfileobj(reference.file, buffer)
                temp_paths.append(temp_rpc)

            if dem_file:
                temp_dem = TEMP_DIR / f"ortho_{src_stem}_dem_input.tif"
                with open(temp_dem, "wb") as buffer:
                    shutil.copyfileobj(dem_file.file, buffer)
                temp_paths.append(temp_dem)

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
        for tp in temp_paths:
            tp.unlink(missing_ok=True)

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
        },
    )
