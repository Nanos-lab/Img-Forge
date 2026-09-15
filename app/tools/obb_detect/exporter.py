"""Output exporter — GeoJSON and visualization.

Converts detection results (OBB in full-image pixel coordinates) to:
- GeoJSON FeatureCollection dictionary (for API responses)
- Annotated visualization image (PNG), rendered from a TIF and the
  GeoJSON returned by the obb_detect API
"""

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio

from app.core.config import TEMP_DIR
from app.tools.obb_detect.detector import Detection
from app.tools.obb_detect.merger import obb_to_corners

# cv2 is optional — only needed for visualize()
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

CLASS_COLORS = {
    "plane": (0, 255, 0),
    "ship": (0, 0, 255),
}

DEFAULT_BOX_COLOR = (255, 255, 0)

# 可视化输出默认写入的子目录（相对于 TEMP_DIR）
VISUALIZE_OUTPUT_SUBDIR = "obb_detect"


# ---- coordinate conversion ----

def pixel_to_geo(transform, px: float, py: float) -> tuple:
    """Convert pixel coordinates to geographic (lon, lat) via rasterio transform."""
    lon, lat = rasterio.transform.xy(transform, py, px)
    return (lon, lat)


def geo_to_pixel(transform, lon: float, lat: float) -> tuple:
    """Convert geographic (lon, lat) back to pixel (px, py) via rasterio transform."""
    row, col = rasterio.transform.rowcol(transform, lon, lat)
    return (col, row)


def obb_to_geojson_polygon(det: Detection, transform) -> list:
    """Convert a single OBB detection to a GeoJSON polygon coordinate list.

    Corners from obb_to_corners are in image-pixel clockwise order.
    We walk them in reverse so the GeoJSON ring is counterclockwise (RFC 7946 §3.1.6).
    """
    corners_pixel = obb_to_corners(det.cx, det.cy, det.width, det.height, det.angle)
    coords = []
    for corner in reversed(corners_pixel):
        lon, lat = pixel_to_geo(transform, corner[0], corner[1])
        coords.append([lon, lat])
    coords.append(coords[0])  # close the ring
    return coords


# ---- GeoJSON dict (API output) ----

def build_geojson_dict(
    detections: List[Detection],
    transform,
    source_filename: Optional[str] = None,
) -> dict:
    """Build a GeoJSON FeatureCollection dict (RFC 7946 compliant, no crs member)."""
    features = []
    for i, det in enumerate(detections):
        polygon_coords = obb_to_geojson_polygon(det, transform)
        center_lon, center_lat = pixel_to_geo(transform, det.cx, det.cy)

        features.append({
            "type": "Feature",
            "id": i,
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coords],
            },
            "properties": {
                "class_id": det.cls,
                "class_name": det.class_name,
                "confidence": round(det.confidence, 4),
                "center_lon": round(center_lon, 6),
                "center_lat": round(center_lat, 6),
                "width_px": round(det.width, 2),
                "height_px": round(det.height, 2),
                "angle_rad": round(det.angle, 4),
                "angle_deg": round(math.degrees(det.angle), 2),
                "source_file": source_filename or "",
            },
        })

    class_counts = {}
    for d in detections:
        class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_detections": len(detections),
            "class_counts": class_counts,
        },
    }


# ---- file export ----

def export_geojson(
    detections: List[Detection],
    transform,
    output_path: str,
    source_filename: Optional[str] = None,
) -> str:
    """Export detections as a GeoJSON file."""
    import json

    geojson = build_geojson_dict(detections, transform, source_filename)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    return str(output_path)


# ---- visualization ----

def _read_display_image(src) -> np.ndarray:
    """Read a rasterio dataset into a BGR uint8 numpy array for display.

    Handles band-count fallback (1/2/3+ bands) and dtype normalization
    (uint16 → uint8 by right-shift, other dtypes → min-max scaling).
    """
    num_bands = src.count
    if num_bands >= 3:
        img = src.read([1, 2, 3])
    elif num_bands == 2:
        arr = src.read([1, 2])
        img = np.zeros((3, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
        img[0], img[1] = arr[0], arr[1]
    else:
        arr = src.read(1)
        img = np.stack([arr, arr, arr], axis=0)

    # (C, H, W) → (H, W, C) & normalize to uint8
    img = np.transpose(img, (1, 2, 0))
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        vmin, vmax = img.min(), img.max()
        img = ((img - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def visualize(
    tif_path: str,
    geojson: dict,
    output_path: Optional[str] = None,
    max_side: int = 4000,
    line_width: int = 2,
    font_scale: float = 0.6,
) -> str:
    """Draw OBB detections from a GeoJSON result onto the source TIF and save as PNG.

    Args:
        tif_path: 输入的 TIF 影像路径（即送入 obb_detect 接口的原图）。
        geojson: obb_detect 接口（或 service.detect_objects）返回的
            GeoJSON FeatureCollection dict。每个 Feature 的
            geometry.coordinates 存放的是经纬度（若影像无地理参考，则数值上
            等价于像素坐标），会通过影像的 transform 反算回像素坐标用于绘制。
        output_path: 输出 PNG 路径。默认写入
            temp/obb_detect/{tif文件名}.png。
        max_side: 图像长边超过该值时等比缩小，避免超大图渲染过慢。
        line_width: 检测框线宽（像素）。
        font_scale: 标签文字缩放比例。

    Returns:
        输出 PNG 文件的绝对路径。
    """
    if not _HAS_CV2:
        raise ImportError(
            "opencv-python-headless is required for visualization. "
            "Install it with: pip install opencv-python-headless"
        )

    tif_path = Path(tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"影像文件不存在: {tif_path}")

    if output_path is None:
        output_path = TEMP_DIR / VISUALIZE_OUTPUT_SUBDIR / f"{tif_path.stem}.png"
    output_path = Path(output_path)

    with rasterio.open(str(tif_path)) as src:
        transform = src.transform
        img = _read_display_image(src)

    # Downsample
    h, w = img.shape[:2]
    longest = max(h, w)
    scale = min(1.0, max_side / longest) if longest > max_side else 1.0
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Draw boxes directly from the GeoJSON polygon rings
    features = geojson.get("features", [])
    for feat in features:
        props = feat.get("properties", {})
        geometry = feat.get("geometry", {})
        rings = geometry.get("coordinates", [])
        if not rings or not rings[0]:
            continue
        ring = rings[0][:-1]  # drop the closing duplicate point

        corners = []
        for lon, lat in ring:
            px, py = geo_to_pixel(transform, lon, lat)
            corners.append([px * scale, py * scale])
        corners = np.array(corners, dtype=np.int32)

        class_name = props.get("class_name", "unknown")
        confidence = props.get("confidence", 0.0)
        color = CLASS_COLORS.get(class_name, DEFAULT_BOX_COLOR)

        cv2.polylines(img, [corners], isClosed=True, color=color, thickness=line_width)

        # Label anchored at the top-most corner of the box
        label = f"{class_name} {confidence:.2f}"
        tx = int(corners[:, 0].min())
        ty = max(15, int(corners[:, 1].min()) - 5)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (tx, ty - th - 2), (tx + tw, ty + 2), color, -1)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Summary line: per-class counts + total, taken from GeoJSON metadata
    metadata = geojson.get("metadata", {})
    class_counts = metadata.get("class_counts", {})
    total = metadata.get("total_detections", len(features))
    if class_counts:
        summary = " | ".join(f"{name}: {count}" for name, count in class_counts.items())
        summary = f"{summary} | Total: {total}"
    else:
        summary = f"Total: {total}"
    cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return str(output_path)
