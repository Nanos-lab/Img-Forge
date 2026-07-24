"""Output exporter — GeoJSON and visualization.

Converts detection results (OBB in full-image pixel coordinates) to:
- GeoJSON FeatureCollection dictionary (for API responses)
- Annotated visualization image (PNG)
"""

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio

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


# ---- coordinate conversion ----

def pixel_to_geo(transform, px: float, py: float) -> tuple:
    """Convert pixel coordinates to geographic (lon, lat) via rasterio transform."""
    lon, lat = rasterio.transform.xy(transform, py, px)
    return (lon, lat)


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
    geojson = build_geojson_dict(detections, transform, source_filename)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    return str(output_path)


def visualize(
    src,
    detections: List[Detection],
    output_path: str,
    max_side: int = 4000,
    line_width: int = 2,
    font_scale: float = 0.6,
) -> str:
    """Create an annotated visualization image with OBBs drawn on it."""
    if not _HAS_CV2:
        raise ImportError(
            "opencv-python-headless is required for visualization. "
            "Install it with: pip install opencv-python-headless"
        )

    # Read RGB
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

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Downsample
    h, w = img.shape[:2]
    longest = max(h, w)
    scale = min(1.0, max_side / longest) if longest > max_side else 1.0
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Draw boxes
    for det in detections:
        color = CLASS_COLORS.get(det.class_name, (255, 255, 0))
        cx_s, cy_s = det.cx * scale, det.cy * scale
        w_s, h_s = det.width * scale, det.height * scale
        cos_a, sin_a = math.cos(det.angle), math.sin(det.angle)
        hw, hh = w_s / 2.0, h_s / 2.0
        corners_local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = (corners_local @ rotation.T + np.array([cx_s, cy_s])).astype(np.int32)
        cv2.polylines(img, [corners], isClosed=True, color=color, thickness=line_width)

        # Label
        label = f"{det.class_name} {det.confidence:.2f}"
        tx, ty = max(0, int(cx_s - w_s / 2)), max(15, int(cy_s - hh - 5))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (tx, ty - th - 2), (tx + tw, ty + 2), color, -1)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Summary
    n_planes = sum(1 for d in detections if d.cls == 0)
    n_ships = sum(1 for d in detections if d.cls == 1)
    summary = f"Planes: {n_planes} | Ships: {n_ships} | Total: {len(detections)}"
    cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return str(output_path)
