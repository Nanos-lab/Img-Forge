"""正射校正服务层 —— 基于 RPC 有理多项式系数的几何校正。

核心逻辑:
1. 解析 RPC 文件（RPB 格式）为 rasterio.rpc.RPC 对象
2. 使用 rasterio.warp.reproject 配合 RPC 系数进行正射校正
3. 可选 DEM 文件用于精确地形校正，无 DEM 时使用常量高度
4. 输出 CRS 默认为 CGCS2000 高斯-克吕格（自动计算 3 度带）
5. 输出 GeoTIFF 并保留地理参考信息
"""

import math
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.rpc import RPC
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
)

from app.core.config import (
    ORTHO_OUTPUT_SUFFIX,
    OUTPUT_EXTENSION,
    ORTHO_DEFAULT_HEIGHT,
    ORTHO_DEFAULT_RESAMPLING,
)
from app.core.exceptions import OrthoError


# ============================================================
#  RPC 文件解析
# ============================================================


def _parse_rpc_file(rpc_path: str) -> RPC:
    """解析 RPC 系数文本文件（RPB 格式）为 rasterio.rpc.RPC 对象。

    Args:
        rpc_path: RPC 文件路径（.rpc / .rpb 格式）。

    Returns:
        rasterio.rpc.RPC 对象。

    Raises:
        OrthoError: 文件格式错误或缺少必要字段。
    """
    params: dict[str, str] = {}

    with open(rpc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            # 格式: KEY: VALUE  或  KEY = VALUE
            if ":" in line:
                key, val = line.split(":", 1)
            elif "=" in line:
                key, val = line.split("=", 1)
            else:
                continue
            params[key.strip().upper()] = val.strip()

    def get_float(key: str) -> float:
        v = params.get(key)
        if v is None:
            raise OrthoError(f"RPC 文件缺少必要字段: {key}")
        return float(v)

    def get_coeffs(prefix: str) -> list[float]:
        """读取 20 个系数: prefix_1 ~ prefix_20"""
        coeffs = []
        for i in range(1, 21):
            key = f"{prefix}_{i}"
            if key not in params:
                raise OrthoError(f"RPC 文件缺少系数: {key}")
            coeffs.append(float(params[key]))
        if len(coeffs) != 20:
            raise OrthoError(f"{prefix} 需要 20 个系数，实际 {len(coeffs)}")
        return coeffs

    try:
        rpc = RPC(
            height_off=get_float("HEIGHT_OFF"),
            height_scale=get_float("HEIGHT_SCALE"),
            lat_off=get_float("LAT_OFF"),
            lat_scale=get_float("LAT_SCALE"),
            long_off=get_float("LONG_OFF"),
            long_scale=get_float("LONG_SCALE"),
            line_off=get_float("LINE_OFF"),
            line_scale=get_float("LINE_SCALE"),
            samp_off=get_float("SAMP_OFF"),
            samp_scale=get_float("SAMP_SCALE"),
            line_num_coeff=get_coeffs("LINE_NUM_COEFF"),
            line_den_coeff=get_coeffs("LINE_DEN_COEFF"),
            samp_num_coeff=get_coeffs("SAMP_NUM_COEFF"),
            samp_den_coeff=get_coeffs("SAMP_DEN_COEFF"),
            err_bias=0,
            err_rand=0,
        )
    except OrthoError:
        raise
    except Exception as exc:
        raise OrthoError(f"RPC 文件解析失败: {exc}")

    return rpc


# ============================================================
#  CRS 自动计算
# ============================================================


def _cgcs2000_gauss_kruger_3deg(center_lon: float) -> CRS:
    """根据中心经度计算 CGCS2000 3 度带高斯-克吕格投影的 CRS。

    Args:
        center_lon: 中心经度（十进制度数）。

    Returns:
        rasterio.crs.CRS 对象。
    """
    # CGCS2000 3-degree Gauss-Kruger zone 25 ~ 45
    # Zone 25: 75°E (EPSG:4544)  ..., Zone 45: 135°E (EPSG:4564)
    zone = round((center_lon - 1.5) / 3)
    zone = max(25, min(45, zone))
    epsg_code = 4519 + zone
    return CRS.from_epsg(epsg_code)


def _resolve_dst_crs(dst_crs: str, center_lon: float) -> CRS:
    """解析输出 CRS 参数。

    Args:
        dst_crs: 用户指定的 CRS，支持 "auto"、EPSG 编码或 Proj4 字符串。
        center_lon: 影像中心经度，用于 auto 模式。

    Returns:
        rasterio.crs.CRS 对象。
    """
    if dst_crs.lower() == "auto":
        return _cgcs2000_gauss_kruger_3deg(center_lon)

    # 尝试作为 EPSG 编码
    if dst_crs.upper().startswith("EPSG:"):
        return CRS.from_epsg(dst_crs.split(":")[1])

    # 作为 Proj4 或 WKT 字符串
    crs = CRS.from_string(dst_crs)
    if crs and crs.is_valid:
        return crs

    raise OrthoError(f"无法识别的 CRS: {dst_crs}")


# ============================================================
#  RPC 来源
# ============================================================


def _read_rpc_from_tiff(src_path: str) -> RPC:
    """从 TIFF 内嵌的 RPC 标签中读取 RPC 系数。

    Args:
        src_path: TIFF 文件路径。

    Returns:
        rasterio.rpc.RPC 对象。

    Raises:
        OrthoError: TIFF 中未找到 RPC 标签或格式异常。
    """
    with rasterio.open(src_path) as src:
        rpc_tags = src.tags(ns="RPC")
    if not rpc_tags:
        raise OrthoError("TIFF 中未找到内嵌 RPC 标签")

    try:
        # RPC 标签中的系数是空格分隔的字符串
        def parse_coeffs(val: str) -> list[float]:
            return [float(x) for x in val.strip().split()]

        rpc = RPC(
            height_off=float(rpc_tags["HEIGHT_OFF"]),
            height_scale=float(rpc_tags["HEIGHT_SCALE"]),
            lat_off=float(rpc_tags["LAT_OFF"]),
            lat_scale=float(rpc_tags["LAT_SCALE"]),
            long_off=float(rpc_tags["LONG_OFF"]),
            long_scale=float(rpc_tags["LONG_SCALE"]),
            line_off=float(rpc_tags["LINE_OFF"]),
            line_scale=float(rpc_tags["LINE_SCALE"]),
            samp_off=float(rpc_tags["SAMP_OFF"]),
            samp_scale=float(rpc_tags["SAMP_SCALE"]),
            line_num_coeff=parse_coeffs(rpc_tags["LINE_NUM_COEFF"]),
            line_den_coeff=parse_coeffs(rpc_tags["LINE_DEN_COEFF"]),
            samp_num_coeff=parse_coeffs(rpc_tags["SAMP_NUM_COEFF"]),
            samp_den_coeff=parse_coeffs(rpc_tags["SAMP_DEN_COEFF"]),
            err_bias=0,
            err_rand=0,
        )
    except KeyError as exc:
        raise OrthoError(f"TIFF 内嵌 RPC 标签缺少字段: {exc}")
    except Exception as exc:
        raise OrthoError(f"读取 TIFF 内嵌 RPC 失败: {exc}")

    return rpc


def orthorectify(
    src_path: str,
    rpc_path: str | None = None,
    dst_crs: str = "auto",
    dem_path: str | None = None,
    height: float = ORTHO_DEFAULT_HEIGHT,
    resolution: float | None = None,
) -> str:
    """对遥感影像进行 RPC 正射校正。

    RPC 来源优先级：
    1. rpc_path 参数（外置 RPC 文件）
    2. TIFF 内嵌的 RPC 标签（自动读取）

    Args:
        src_path:   原始遥感影像 TIFF 路径。
        rpc_path:   RPC 系数文件路径（可选，不传则尝试从 TIFF 内嵌读取）。
        dst_crs:    输出投影，默认 "auto"（CGCS2000 高斯-克吕格自动计算）。
        dem_path:   DEM 高程文件路径（可选），用于精确地形校正。
        height:     无 DEM 时的默认高程（米）。
        resolution: 输出分辨率（地理单位/像素），默认与输入近似。

    Returns:
        输出正射校正后的 TIFF 文件路径。

    Raises:
        OrthoError: RPC 解析失败、正射校正处理失败。
    """
    if not src_path:
        raise OrthoError("未提供原始影像路径")

    # ------------------------------------------------------------------
    # 1. 获取 RPC（优先外置文件，其次 TIFF 内嵌）
    # ------------------------------------------------------------------
    if rpc_path:
        rpc = _parse_rpc_file(rpc_path)
    else:
        rpc = _read_rpc_from_tiff(src_path)

    # ------------------------------------------------------------------
    # 2. 打开源影像
    # ------------------------------------------------------------------
    with rasterio.open(src_path) as src:
        src_data = src.read()
        src_height, src_width = src.height, src.width
        src_dtype = src.dtypes[0]

        # 计算影像中心经度（使用 RPC 提供的大地坐标）
        center_lon = rpc.long_off
        center_lat = rpc.lat_off

        # 解析输出 CRS
        out_crs = _resolve_dst_crs(dst_crs, center_lon)

        # ------------------------------------------------------------------
        # 3. 计算输出变换和尺寸
        # ------------------------------------------------------------------
        # RPC 的参考坐标系为 WGS84 经纬度（EPSG:4326）
        src_crs = CRS.from_epsg(4326)

        if resolution is not None:
            out_transform, out_width, out_height = calculate_default_transform(
                src_crs, out_crs, src_width, src_height,
                rpcs=rpc,
                resolution=resolution,
            )
        else:
            out_transform, out_width, out_height = calculate_default_transform(
                src_crs, out_crs, src_width, src_height,
                rpcs=rpc,
            )

        if out_width < 1 or out_height < 1:
            raise OrthoError(
                f"计算输出尺寸异常: {out_width}x{out_height}"
            )

        # ------------------------------------------------------------------
        # 4. 执行正射校正
        # ------------------------------------------------------------------
        destination = np.zeros(
            (src_data.shape[0], out_height, out_width), dtype=src_dtype
        )

        # 构建传递给 GDAL 的额外参数
        warp_kwargs: dict = {}
        if dem_path:
            warp_kwargs["RPC_DEM"] = dem_path
        warp_kwargs["RPC_HEIGHT"] = str(height)

        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "lanczos": Resampling.lanczos,
        }
        resampling = resampling_map.get(
            ORTHO_DEFAULT_RESAMPLING, Resampling.bilinear
        )

        try:
            reproject(
                src_data,
                destination,
                rpcs=rpc,
                src_crs=src_crs,
                dst_crs=out_crs,
                dst_transform=out_transform,
                resampling=resampling,
                num_threads=2,
                kwargs=warp_kwargs,
            )
        except Exception as exc:
            raise OrthoError(f"正射校正失败: {exc}")

        # ------------------------------------------------------------------
        # 5. 写出结果
        # ------------------------------------------------------------------
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=out_height,
            width=out_width,
            transform=out_transform,
            crs=out_crs,
            compress="lzw",
        )

        src_path_obj = Path(src_path)
        dst_path = (
            src_path_obj.parent
            / f"{src_path_obj.stem}{ORTHO_OUTPUT_SUFFIX}{OUTPUT_EXTENSION}"
        )

        with rasterio.open(str(dst_path), "w", **profile) as dst:
            dst.write(destination)

    return str(dst_path)
