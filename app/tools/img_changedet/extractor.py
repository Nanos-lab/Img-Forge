"""变化区域提取 —— 从概率图提取变化区域并输出 GeoJSON。

流程：高斯模糊 → 形态学闭运算 → Otsu 阈值 → 连通域 → 合并重叠框 → GeoJSON。
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ChangeRegion:
    """单个变化区域。"""
    id: int
    bbox_pixel: tuple[int, int, int, int]    # (x, y, w, h)
    area_pixels: int
    corners_geo: list[tuple[float, float]]    # 四角 (lon, lat) 顺时针
    centroid_geo: tuple[float, float]         # (lon, lat)


def _otsu_threshold(prob: np.ndarray) -> float:
    """Otsu 自适应阈值。"""
    prob_u8 = (np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thresh_val) / 255.0


def _auto_min_area(h: int, w: int) -> int:
    return max(50, int(h * w * 0.00005))


def _merge_overlapping(regions: list[ChangeRegion]) -> list[ChangeRegion]:
    """迭代合并有重叠的边界框。"""
    if len(regions) <= 1:
        return regions

    boxes = [(r.bbox_pixel[0], r.bbox_pixel[1],
              r.bbox_pixel[0] + r.bbox_pixel[2],
              r.bbox_pixel[1] + r.bbox_pixel[3], r) for r in regions]

    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(boxes)
        for i, (x1, y1, x2, y2, ri) in enumerate(boxes):
            if used[i]:
                continue
            for j, (u1, v1, u2, v2, rj) in enumerate(boxes):
                if i == j or used[j]:
                    continue
                if x1 < u2 and x2 > u1 and y1 < v2 and y2 > v1:
                    ux, uy = min(x1, u1), min(y1, v1)
                    ux2, uy2 = max(x2, u2), max(y2, v2)
                    ri.bbox_pixel = (ux, uy, ux2 - ux, uy2 - uy)
                    ri.area_pixels += rj.area_pixels
                    boxes[i] = (ux, uy, ux2, uy2, ri)
                    used[j] = True
                    changed = True
            if not used[i]:
                merged.append(boxes[i])
        if changed:
            boxes = merged

    return [b[4] for b in boxes]


def _merge_contained(regions: list[ChangeRegion]) -> list[ChangeRegion]:
    """移除完全被更大框包含的小框。"""
    if len(regions) <= 1:
        return regions
    sorted_regions = sorted(regions, key=lambda r: r.area_pixels, reverse=True)
    result = []
    for r in sorted_regions:
        rx, ry, rw, rh = r.bbox_pixel
        contained = False
        for outer in result:
            ox, oy, ow, oh = outer.bbox_pixel
            if rx >= ox and ry >= oy and rx + rw <= ox + ow and ry + rh <= oy + oh:
                outer.area_pixels += r.area_pixels
                contained = True
                break
        if not contained:
            result.append(r)
    return result


def extract_changes(
    change_map: np.ndarray,
    transform,
    *,
    crop_offset: tuple[int, int] = (0, 0),
    min_area: int | None = None,
    threshold: float | None = None,
) -> tuple[list[ChangeRegion], np.ndarray]:
    """从概率图提取变化区域。

    Args:
        change_map: 概率图 (H, W) float32。
        transform: rasterio 仿射变换，像素→地理坐标。
        crop_offset: 裁剪偏移 (y, x)，用于恢复在原始影像中的位置。
        min_area: 最小变化区域面积，None 自动。
        threshold: 二值化阈值，None Otsu 自动。

    Returns:
        (regions, binary_map):
        - regions: 变化区域列表，按面积降序。
        - binary_map: 二值化后的变化掩膜 (H, W) uint8。
    """
    H, W = change_map.shape[:2]

    if threshold is None:
        threshold = _otsu_threshold(change_map)
    if min_area is None:
        min_area = _auto_min_area(H, W)

    # 高斯模糊 + 形态学闭运算
    blur = cv2.GaussianBlur(change_map, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)

    binary = (closed >= threshold).astype(np.uint8)

    if binary.sum() == 0:
        return [], binary

    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=4, ltype=cv2.CV_32S,
    )

    offset_y, offset_x = crop_offset
    regions = []

    for label_id in range(1, stats.shape[0]):
        x, y, w, h, area = stats[label_id]
        if area < min_area:
            continue

        # 四角像素坐标 → 地理坐标
        corners_px = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        if transform is not None:
            corners_geo = []
            for px, py in corners_px:
                lon, lat = transform * (px + offset_x, py + offset_y)
                corners_geo.append((float(lon), float(lat)))
        else:
            corners_geo = [(float(px), float(py)) for px, py in corners_px]

        cx, cy = centroids[label_id]
        if transform is not None:
            clon, clat = transform * (cx + offset_x, cy + offset_y)
        else:
            clon, clat = float(cx), float(cy)

        regions.append(ChangeRegion(
            id=0,
            bbox_pixel=(int(x), int(y), int(w), int(h)),
            area_pixels=int(area),
            corners_geo=corners_geo,
            centroid_geo=(float(clon), float(clat)),
        ))

    regions = _merge_overlapping(regions)
    regions = _merge_contained(regions)
    regions.sort(key=lambda r: r.area_pixels, reverse=True)
    for i, r in enumerate(regions):
        r.id = i + 1

    return regions, binary


def build_geojson(regions: list[ChangeRegion], elapsed_ms: int) -> dict:
    """构建 GeoJSON FeatureCollection。

    Args:
        regions: 变化区域列表。
        elapsed_ms: 处理耗时（毫秒）。

    Returns:
        GeoJSON 字典。
    """
    features = []
    for r in regions:
        # RFC 7946: 外环逆时针，闭合
        coords = [[
            [r.corners_geo[0][0], r.corners_geo[0][1]],  # TL
            [r.corners_geo[1][0], r.corners_geo[1][1]],  # TR
            [r.corners_geo[2][0], r.corners_geo[2][1]],  # BR
            [r.corners_geo[3][0], r.corners_geo[3][1]],  # BL
            [r.corners_geo[0][0], r.corners_geo[0][1]],  # 闭合
        ]]

        features.append({
            "type": "Feature",
            "id": r.id,
            "geometry": {
                "type": "Polygon",
                "coordinates": coords,
            },
            "properties": {
                "area_pixels": r.area_pixels,
                "center_lon": round(r.centroid_geo[0], 6),
                "center_lat": round(r.centroid_geo[1], 6),
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "count": len(features),
            "elapsed_ms": elapsed_ms,
        },
    }
