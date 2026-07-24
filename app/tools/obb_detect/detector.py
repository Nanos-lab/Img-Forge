"""YOLOv8-OBB detector wrapper.

Loads a pretrained YOLOv8-OBB model (trained on DOTA v1) and runs inference
on tile batches. Uses a module-level model singleton for efficiency.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from app.tools.obb_detect.tiler import TileInfo

# DOTA v1 — full class name mapping
DOTA_CLASSES = {
    0: "plane",
    1: "ship",
    2: "storage-tank",
    3: "baseball-diamond",
    4: "tennis-court",
    5: "basketball-court",
    6: "ground-track-field",
    7: "harbor",
    8: "bridge",
    9: "large-vehicle",
    10: "small-vehicle",
    11: "helicopter",
    12: "roundabout",
    13: "soccer-ball-field",
    14: "swimming-pool",
}

# Module-level model singleton
_model = None


def _get_model(model_path: str):
    """Load and cache the YOLO model (singleton)."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(model_path)
    return _model


@dataclass
class Detection:
    """A single detection result in tile-local pixel coordinates."""

    cls: int
    class_name: str
    confidence: float
    cx: float
    cy: float
    width: float
    height: float
    angle: float
    col_off: int
    row_off: int


class Detector:
    """YOLOv8-OBB detector wrapper."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        classes: list = None,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self._model = None

        # Build target class filter
        if classes is None or len(classes) == 0:
            self.target_classes = DOTA_CLASSES.copy()
        else:
            self.target_classes = {}
            for cid in classes:
                if cid in DOTA_CLASSES:
                    self.target_classes[cid] = DOTA_CLASSES[cid]

    @property
    def model(self):
        """Lazy-load the YOLO model (singleton)."""
        if self._model is None:
            self._model = _get_model(self.model_path)
        return self._model

    # ---- internal helpers ----

    def _parse_obb_results(self, results) -> List[dict]:
        """Extract OBB detections, filtering to target classes."""
        detections = []
        for result in results:
            if result.obb is None:
                continue
            obb_data = result.obb.data
            obb_cls = result.obb.cls
            if obb_data is None or len(obb_data) == 0:
                continue
            for i in range(len(obb_data)):
                cls_id = int(obb_cls[i].item())
                if cls_id not in self.target_classes:
                    continue
                row = obb_data[i]
                detections.append({
                    "cls": cls_id,
                    "class_name": self.target_classes[cls_id],
                    "confidence": float(row[5]) if len(row) > 5 else float(row[4]),
                    "cx": float(row[0]),
                    "cy": float(row[1]),
                    "width": float(row[2]),
                    "height": float(row[3]),
                    "angle": float(row[4]),
                })
        return detections

    # ---- public API ----

    def detect_batch(
        self, tiles: List[TileInfo], batch_size: int = 8,
    ) -> List[Detection]:
        """Run inference on a batch of tiles."""
        all_detections: List[Detection] = []
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i : i + batch_size]
            batch_images = [t.data for t in batch_tiles]
            results = self.model(
                batch_images,
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            for tile, result in zip(batch_tiles, results):
                raw_dets = self._parse_obb_results([result])
                for d in raw_dets:
                    all_detections.append(Detection(
                        cls=d["cls"],
                        class_name=d["class_name"],
                        confidence=d["confidence"],
                        cx=d["cx"],
                        cy=d["cy"],
                        width=d["width"],
                        height=d["height"],
                        angle=d["angle"],
                        col_off=tile.col_off,
                        row_off=tile.row_off,
                    ))
        return all_detections

    def warmup(self):
        """Warm up the model with a dummy inference."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, conf=0.99, device=self.device, verbose=False)
