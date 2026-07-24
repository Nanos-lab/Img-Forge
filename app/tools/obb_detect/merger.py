"""
Detection merger — coordinate mapping and global OBB NMS.

After YOLO runs on each tile independently, this module:
1. Maps tile-local coordinates back to full-image pixel coordinates.
2. Runs global OBB NMS to remove duplicate detections across tile boundaries.
"""

import math
from typing import List

import numpy as np

from app.tools.obb_detect.detector import Detection


# ---- Coordinate conversion --------------------------------------------------

def tile_to_image_coords(detections: List[Detection]) -> List[Detection]:
    """
    Convert detection coordinates from tile-local space to full-image pixel space.

    Mutates each Detection in-place and returns the same list.
    """
    for det in detections:
        det.cx += det.col_off
        det.cy += det.row_off
    return detections


# ---- OBB utilities ----------------------------------------------------------

def obb_to_corners(cx: float, cy: float, w: float, h: float, angle: float):
    """
    Convert an oriented bounding box (cx, cy, w, h, angle) to four corner points.

    YOLOv8-OBB angle convention: angle is in radians, measured from the positive
    x-axis. The box is defined as width along the direction of the angle.

    Returns:
        numpy array of shape (4, 2) — four corner coordinates.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Half-dimensions
    hw = w / 2.0
    hh = h / 2.0

    # Corners in the local (unrotated) frame
    corners_local = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ])

    # Rotation matrix
    rotation = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a],
    ])

    # Rotate and translate
    corners = corners_local @ rotation.T + np.array([cx, cy])

    return corners


def obb_iou(det1: Detection, det2: Detection) -> float:
    """
    Compute IoU between two OBBs using the shoelace formula for polygon area
    and Sutherland–Hodgman for polygon intersection.

    Returns a float in [0.0, 1.0].
    """
    corners1 = obb_to_corners(det1.cx, det1.cy, det1.width, det1.height, det1.angle)
    corners2 = obb_to_corners(det2.cx, det2.cy, det2.width, det2.height, det2.angle)

    intersection = _convex_polygon_intersection(corners1, corners2)
    if intersection is None or len(intersection) < 3:
        return 0.0

    area_inter = _polygon_area(intersection)
    area1 = _polygon_area(corners1)
    area2 = _polygon_area(corners2)
    union = area1 + area2 - area_inter

    if union <= 0:
        return 0.0
    return area_inter / union


# ---- Polygon geometry helpers -----------------------------------------------

def _polygon_area(corners: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    x = corners[:, 0]
    y = corners[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _convex_polygon_intersection(
    poly1: np.ndarray,
    poly2: np.ndarray,
) -> np.ndarray:
    """
    Sutherland–Hodgman clipping: clip poly1 (subject) by each edge of poly2 (clip).

    Both inputs must be convex and in CCW or CW order (consistent).
    Returns the intersection polygon as a numpy array of shape (N, 2), or None.
    """
    output = poly1.copy()

    n2 = len(poly2)
    for i in range(n2):
        if len(output) == 0:
            return None

        edge_start = poly2[i]
        edge_end = poly2[(i + 1) % n2]

        input_poly = output.copy()
        output = []

        n_in = len(input_poly)
        for j in range(n_in):
            current = input_poly[j]
            previous = input_poly[(j - 1) % n_in]

            current_inside = _is_inside(current, edge_start, edge_end)
            previous_inside = _is_inside(previous, edge_start, edge_end)

            if current_inside:
                if not previous_inside:
                    # Leaving → entering: add intersection
                    output.append(_line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                # Entering → leaving: add intersection
                output.append(_line_intersection(previous, current, edge_start, edge_end))

    if len(output) == 0:
        return None
    return np.array(output)


def _is_inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    """Check if point is on the left side of the directed edge (CCW polygon)."""
    return (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    ) >= 0


def _line_intersection(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> np.ndarray:
    """Compute the intersection point of lines p1-p2 and p3-p4."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return p2  # near-parallel: fallback

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])


# ---- Global NMS -------------------------------------------------------------

def obb_nms(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """
    Apply OBB NMS across all detections in full-image coordinates.

    Sorts by confidence descending, then greedily suppresses overlapping boxes.

    Args:
        detections: list of Detection objects in full-image pixel coords
        iou_threshold: IoU threshold above which to suppress

    Returns:
        Filtered list of detections.
    """
    if len(detections) == 0:
        return []

    # Sort by confidence descending
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []
    suppressed = [False] * len(detections)

    for i in range(len(detections)):
        if suppressed[i]:
            continue
        kept.append(detections[i])

        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            # Only suppress same-class detections
            if detections[i].cls != detections[j].cls:
                continue
            iou = obb_iou(detections[i], detections[j])
            if iou > iou_threshold:
                suppressed[j] = True

    return kept


# ---- Public API -------------------------------------------------------------

def merge_detections(
    detections: List[Detection],
    iou_threshold: float = 0.5,
) -> List[Detection]:
    """
    Full merge pipeline:
    1. Convert tile-local coords to full-image pixel coords.
    2. Apply global OBB NMS.

    Args:
        detections: list of Detection objects with tile-local coords
        iou_threshold: IoU threshold for global NMS

    Returns:
        Merged list of detections in full-image pixel coords.
    """
    detections = tile_to_image_coords(detections)
    detections = obb_nms(detections, iou_threshold)
    return detections
