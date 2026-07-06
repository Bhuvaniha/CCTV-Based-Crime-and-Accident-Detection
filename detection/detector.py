"""YOLO-based object detector.

Loads a pretrained YOLOv8 model from Ultralytics and runs
inference on video frames to detect persons and vehicles.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import config as cfg
from utils.logger import get_logger

logger = get_logger(__name__)


class DetectionResult:
    """Holds detection results for a single frame."""

    def __init__(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        class_names: List[str],
    ):
        self.frame = frame
        self.boxes = boxes          # shape (N, 4)  xyxy
        self.scores = scores        # shape (N,)
        self.class_ids = class_ids  # shape (N,)
        self.class_names = class_names

    @property
    def person_count(self) -> int:
        return int((self.class_ids == cfg.YOLO_PERSON_CLASS_ID).sum())

    @property
    def vehicle_count(self) -> int:
        return int(np.isin(self.class_ids, list(cfg.YOLO_VEHICLE_CLASS_IDS)).sum())

    @property
    def has_persons(self) -> bool:
        return self.person_count > 0

    @property
    def has_vehicles(self) -> bool:
        return self.vehicle_count > 0


class YOLODetector:
    """Thin wrapper around Ultralytics YOLO.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to a ``.pt`` weights file.  Falls back to config default.
    device : str, optional
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        model_path = model_path or cfg.MODEL_DIR / cfg.YOLO_MODEL_NAME
        self.device = device or cfg.YOLO_DEVICE

        logger.info("Loading YOLO model from %s on device=%s", model_path, self.device)
        self.model = YOLO(str(model_path))
        self._class_names = self.model.names
        logger.info("YOLO loaded.  Classes: %s", self._class_names)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (OpenCV format).

        Returns
        -------
        DetectionResult
        """
        results = self.model(frame, device=self.device, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.empty((0, 4))
        scores = results.boxes.conf.cpu().numpy() if results.boxes is not None else np.empty((0,))
        class_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else np.empty((0,), dtype=int)

        # Filter by confidence
        mask = scores >= cfg.YOLO_CONFIDENCE_THRESHOLD
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        class_names = [self._class_names[cid] for cid in class_ids]

        return DetectionResult(
            frame=frame,
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
        )

    def draw_boxes(self, result: DetectionResult, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Annotate frame with bounding boxes and labels.

        Parameters
        ----------
        result : DetectionResult
        color : tuple, optional

        Returns
        -------
        np.ndarray
        """
        frame = result.frame.copy()
        for box, score, name in zip(result.boxes, result.scores, result.class_names):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
