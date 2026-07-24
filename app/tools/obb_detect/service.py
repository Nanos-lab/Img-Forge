"""OBB detection service — full TIF → GeoJSON pipeline.

Steps:
1. Open TIF → get transform metadata
2. Generate tiles (sliding window)
3. YOLOv8-OBB batch inference
4. Merge (tile→image coords + global OBB NMS)
5. Build GeoJSON dict

Model is loaded once as a module-level singleton.
"""

import time
from pathlib import Path
from typing import List, Optional

import rasterio

from app.core.config import (
    OBB_MODEL_PATH,
    OBB_DEFAULT_CONFIDENCE,
    OBB_DEFAULT_IOU,
    OBB_DEFAULT_DEVICE,
    OBB_TILE_SIZE,
    OBB_OVERLAP_RATIO,
    OBB_BATCH_SIZE,
    OBB_NMS_IOU,
)
from app.core.exceptions import DetectionError
from app.tools.obb_detect.detector import Detector
from app.tools.obb_detect.tiler import generate_tiles, get_image_transform_info
from app.tools.obb_detect.merger import merge_detections
from app.tools.obb_detect.exporter import build_geojson_dict

# Resolve model path relative to this module
_MODULE_DIR = Path(__file__).parent


def _resolve_model_path() -> str:
    """Resolve the model file path from config."""
    path = Path(OBB_MODEL_PATH)
    if not path.is_absolute():
        path = _MODULE_DIR / path
    return str(path.resolve())


def detect_objects(
    src_path: str,
    classes: Optional[List[int]] = None,
    confidence: float = OBB_DEFAULT_CONFIDENCE,
) -> dict:
    """Run OBB object detection on a TIFF image.

    Args:
        src_path: input TIF file path.
        classes: list of DOTA class IDs, e.g. [0, 1] for plane+ship.
                 None or empty = detect all 15 classes.
        confidence: detection confidence threshold [0, 1].

    Returns:
        GeoJSON FeatureCollection dict.

    Raises:
        DetectionError: on read/detect/merge failure.
    """
    if classes is None:
        classes = []

    tif_path = Path(src_path)
    if not tif_path.exists():
        raise DetectionError(f"影像文件不存在: {src_path}")

    t_start = time.time()

    # --- create detector for this request ---
    detector = Detector(
        model_path=_resolve_model_path(),
        confidence=confidence,
        iou=OBB_DEFAULT_IOU,
        device=OBB_DEFAULT_DEVICE,
        classes=classes,
    )

    try:
        with rasterio.open(str(tif_path)) as src:
            img_info = get_image_transform_info(src)

            # Generate tiles
            tiles = generate_tiles(
                src,
                tile_size=OBB_TILE_SIZE,
                overlap_ratio=OBB_OVERLAP_RATIO,
            )

            # Warmup on first call
            detector.warmup()

            # Detect
            detections = detector.detect_batch(tiles, batch_size=OBB_BATCH_SIZE)

            # Merge (coord mapping + global OBB NMS)
            detections = merge_detections(
                detections, iou_threshold=OBB_NMS_IOU,
            )

    except DetectionError:
        raise
    except Exception as exc:
        raise DetectionError(f"目标检测失败: {exc}")

    elapsed = time.time() - t_start

    # Build GeoJSON (RFC 7946: no crs member, coordinates are always WGS84)
    geojson = build_geojson_dict(
        detections,
        img_info["transform"],
        source_filename=tif_path.name,
    )
    geojson["metadata"]["elapsed_seconds"] = round(elapsed, 1)
    geojson["metadata"]["image_width"] = img_info["width"]
    geojson["metadata"]["image_height"] = img_info["height"]

    return geojson
