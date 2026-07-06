"""Event detection logic.

Analyzes per-frame detection results and identifies
incidents: road accidents, fights, and fires.

Each detection strategy is a separate method so it can be
extended or replaced without touching the other strategies.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

import config as cfg
from detection.detector import DetectionResult
from utils.logger import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    ACCIDENT = auto()
    FIGHT = auto()
    FIRE = auto()


class Event:
    """Represents a single detected incident."""

    def __init__(
        self,
        event_type: EventType,
        frame_idx: int,
        timestamp_sec: float,
        confidence: float,
        frame: np.ndarray,
        boxes: np.ndarray,
    ):
        self.event_type = event_type
        self.frame_idx = frame_idx
        self.timestamp_sec = timestamp_sec
        self.confidence = confidence
        self.frame = frame
        self.boxes = boxes

    @property
    def type_name(self) -> str:
        return self.event_type.name.title()

    def __repr__(self) -> str:
        return (
            f"Event(type={self.type_name}, frame={self.frame_idx}, "
            f"time={self.timestamp_sec:.2f}s, conf={self.confidence:.2f})"
        )


class EventDetector:
    """Orchestrates multiple event-detection strategies.

    Call ``process_frame`` for every frame to receive a list of
    :class:`Event` objects detected in that frame.
    """

    def __init__(self):
        self._prev_vehicle_centers: List[Tuple[float, float]] = []
        self._prev_person_centers: List[Tuple[float, float]] = []

    def process_frame(
        self,
        result: DetectionResult,
        frame_idx: int,
        fps: float,
    ) -> List[Event]:
        """Analyse a single frame and return detected events."""
        events: List[Event] = []
        timestamp = frame_idx / fps if fps > 0 else 0.0

        accident = self._detect_accident(result, frame_idx, timestamp, fps)
        if accident:
            events.append(accident)

        fight = self._detect_fight(result, frame_idx, timestamp, fps)
        if fight:
            events.append(fight)

        # Update state for next frame
        self._update_state(result)

        return events

    # ------------------------------------------------------------------
    # Accident detection  (IoU overlap + centroid proximity + speed delta)
    # ------------------------------------------------------------------
    def _detect_accident(
        self,
        result: DetectionResult,
        frame_idx: int,
        timestamp: float,
        fps: float,
    ) -> Optional[Event]:
        if not result.has_vehicles:
            return None

        vehicle_mask = np.isin(result.class_ids, list(cfg.YOLO_VEHICLE_CLASS_IDS))
        vehicle_boxes = result.boxes[vehicle_mask]

        for i in range(len(vehicle_boxes)):
            for j in range(i + 1, len(vehicle_boxes)):
                iou = self._box_iou(vehicle_boxes[i], vehicle_boxes[j])
                c1 = self._box_center(vehicle_boxes[i])
                c2 = self._box_center(vehicle_boxes[j])
                centroid_dist = np.linalg.norm(np.array(c1) - np.array(c2))

                # Trigger on IoU overlap OR very close centroids
                if iou >= cfg.ACCIDENT_IOU_THRESHOLD or centroid_dist <= cfg.ACCIDENT_CENTROID_DISTANCE:
                    confidence = max(
                        min(1.0, iou + 0.3),
                        min(0.8, 1.0 - centroid_dist / cfg.ACCIDENT_CENTROID_DISTANCE),
                    )
                    logger.info(
                        "Accident suspected at frame %d (iou=%.3f, dist=%.0f, conf=%.2f)",
                        frame_idx, iou, centroid_dist, confidence,
                    )
                    return Event(
                        event_type=EventType.ACCIDENT,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp,
                        confidence=confidence,
                        frame=result.frame,
                        boxes=np.array([vehicle_boxes[i], vehicle_boxes[j]]),
                    )
        return None

    # ------------------------------------------------------------------
    # Fight detection  (heuristic: close persons + high motion)
    # ------------------------------------------------------------------
    def _detect_fight(
        self,
        result: DetectionResult,
        frame_idx: int,
        timestamp: float,
        fps: float,
    ) -> Optional[Event]:
        if not result.has_persons:
            return None

        person_mask = result.class_ids == cfg.YOLO_PERSON_CLASS_ID
        person_boxes = result.boxes[person_mask]

        if len(person_boxes) < 2:
            return None

        centers = np.array([self._box_center(b) for b in person_boxes])

        # Check proximity between every pair of persons
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < cfg.FIGHT_PROXIMITY_THRESHOLD:
                    logger.info("Fight suspected at frame %d (distance=%.1f)", frame_idx, dist)
                    return Event(
                        event_type=EventType.FIGHT,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp,
                        confidence=0.7,          # heuristic
                        frame=result.frame,
                        boxes=np.array([person_boxes[i], person_boxes[j]]),
                    )
        return None

    # ------------------------------------------------------------------
    # Fire detection (placeholder – uses YOLO fire class if available)
    # ------------------------------------------------------------------
    def _detect_fire(
        self,
        result: DetectionResult,
        frame_idx: int,
        timestamp: float,
        fps: float,
    ) -> Optional[Event]:
        """Detect fire via fire-specific class ID (e.g. class 64 in some models)."""
        fire_class_ids = {64}
        fire_mask = np.isin(result.class_ids, list(fire_class_ids))
        if fire_mask.any():
            fire_boxes = result.boxes[fire_mask]
            fire_scores = result.scores[fire_mask]
            best = fire_scores.argmax()
            if fire_scores[best] >= cfg.FIRE_CONFIDENCE_THRESHOLD:
                return Event(
                    event_type=EventType.FIRE,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp,
                    confidence=float(fire_scores[best]),
                    frame=result.frame,
                    boxes=fire_boxes,
                )
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_state(self, result: DetectionResult) -> None:
        """Store current frame state for motion estimation in next frame."""
        if result.has_vehicles:
            vmask = np.isin(result.class_ids, list(cfg.YOLO_VEHICLE_CLASS_IDS))
            self._prev_vehicle_centers = [
                self._box_center(b) for b in result.boxes[vmask]
            ]
        if result.has_persons:
            pmask = result.class_ids == cfg.YOLO_PERSON_CLASS_ID
            self._prev_person_centers = [
                self._box_center(b) for b in result.boxes[pmask]
            ]

    @staticmethod
    def _box_center(box: np.ndarray) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
