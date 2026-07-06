"""Video processing pipeline.

Orchestrates the full workflow:
  Open video → YOLO detection → Event detection → Evidence capture → DB logging

Each public method is independently callable so the UI can track progress.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple

import cv2
import numpy as np

import config as cfg
from database.db_manager import DatabaseManager
from detection.detector import YOLODetector, DetectionResult
from events.event_handler import EventDetector, Event, EventType
from utils.logger import get_logger
from utils.video_utils import get_video_capture, frame_generator, create_video_writer

logger = get_logger(__name__)


@dataclass
class ProcessingStats:
    """Aggregate statistics from a processing run."""
    total_frames: int = 0
    processed_frames: int = 0
    events_detected: int = 0
    persons_detected: int = 0
    vehicles_detected: int = 0
    processing_time_sec: float = 0.0
    incidents: List[Dict] = field(default_factory=list)


class VideoProcessor:
    """Processes a video file end-to-end.

    Parameters
    ----------
    video_path : Path
    output_path : Path, optional
        Where to save the annotated video (default: ``processed/<name>_annotated.mp4``).
    """

    def __init__(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
    ):
        self.video_path = video_path
        self.output_path = output_path or (
            cfg.PROCESSED_DIR / f"{video_path.stem}_annotated.mp4"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Lazy init so we can catch ImportError gracefully
        self._detector: Optional[YOLODetector] = None
        self._event_detector = EventDetector()
        self._db = DatabaseManager()

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    @property
    def detector(self) -> YOLODetector:
        if self._detector is None:
            self._detector = YOLODetector()
        return self._detector

    # ------------------------------------------------------------------
    # Core processing loop
    # ------------------------------------------------------------------
    def process(
        self,
        progress_callback=None,
        frame_skip: int = None,
    ) -> ProcessingStats:
        """Run the full processing pipeline.

        Parameters
        ----------
        progress_callback : callable, optional
            ``fn(current_frame, total_frames, message)`` called periodically.
        frame_skip : int, optional
            Process every N-th frame (default from config).

        Returns
        -------
        ProcessingStats
        """
        frame_skip = frame_skip if frame_skip is not None else cfg.PROCESSING_FRAME_SKIP

        cap, info = get_video_capture(self.video_path)
        fps = info["fps"]
        width = info["width"]
        height = info["height"]
        total_frames = info["total_frames"]
        processed_frames_estimate = total_frames // frame_skip

        writer = create_video_writer(self.output_path, fps, width, height)

        stats = ProcessingStats(total_frames=total_frames)
        start_time = time.time()

        event_frame_buffer: List[Tuple[np.ndarray, int]] = []
        last_event_frame = -frame_skip * 5  # prevent duplicate spam

        logger.info("Processing started: %s  (%d frames, skip=%d)",
                     self.video_path.name, total_frames, frame_skip)

        for frame, frame_idx in frame_generator(cap, skip=frame_skip):
            stats.processed_frames += 1

            if progress_callback:
                progress_callback(
                    stats.processed_frames,
                    processed_frames_estimate,
                    f"Frame {frame_idx} / {total_frames}",
                )

            # --- Detection ---
            try:
                det_result = self.detector.detect(frame)
            except Exception as exc:
                logger.error("Detection failed at frame %d: %s", frame_idx, exc)
                continue

            stats.persons_detected += det_result.person_count
            stats.vehicles_detected += det_result.vehicle_count

            # --- Event detection ---
            events = self._event_detector.process_frame(det_result, frame_idx, fps)

            for event in events:
                # Avoid duplicate events within a window
                if frame_idx - last_event_frame < frame_skip * 10:
                    continue
                last_event_frame = frame_idx

                logger.info("EVENT: %s at frame %d (conf=%.2f)",
                            event.type_name, frame_idx, event.confidence)

                # Save evidence
                img_path = self._save_evidence_image(event, frame_idx)
                vid_path = self._save_evidence_clip(event, frame_idx, fps, cap)

                # Log to database
                incident_id = self._db.insert_incident(
                    event_type=event.type_name,
                    timestamp_sec=event.timestamp_sec,
                    confidence=event.confidence,
                    evidence_image=str(img_path) if img_path else "",
                    evidence_video=str(vid_path) if vid_path else "",
                )

                stats.events_detected += 1
                stats.incidents.append({
                    "id": incident_id,
                    "event_type": event.type_name,
                    "timestamp_sec": float(event.timestamp_sec),
                    "confidence": float(event.confidence),
                    "evidence_image": str(img_path),
                    "evidence_video": str(vid_path),
                    "processing_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

            # Write annotated frame
            annotated = self.detector.draw_boxes(det_result)
            writer.write(annotated)

        # Cleanup
        writer.release()
        stats.processing_time_sec = time.time() - start_time

        logger.info("Processing complete: %d events in %.1fs",
                     stats.events_detected, stats.processing_time_sec)
        return stats

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------
    def _save_evidence_image(self, event: Event, frame_idx: int) -> Optional[Path]:
        """Save the event frame as a JPEG evidence image."""
        try:
            event_dir = cfg.EVIDENCE_IMAGE_DIR / event.type_name.lower()
            event_dir.mkdir(parents=True, exist_ok=True)
            filename = f"event_{event.type_name}_frame{frame_idx:06d}.jpg"
            out_path = event_dir / filename
            cv2.imwrite(str(out_path), event.frame)
            logger.debug("Evidence image saved: %s", out_path)
            return out_path
        except Exception as exc:
            logger.error("Failed to save evidence image: %s", exc)
            return None

    def _save_evidence_clip(
        self,
        event: Event,
        frame_idx: int,
        fps: float,
        cap: cv2.VideoCapture,
    ) -> Optional[Path]:
        """Extract a short clip around the event and save as MP4."""
        try:
            half_window = int(fps * cfg.EVIDENCE_CLIP_SECONDS)
            start_frame = max(0, frame_idx - half_window)
            end_frame = frame_idx + half_window

            event_dir = cfg.EVIDENCE_VIDEO_DIR / event.type_name.lower()
            event_dir.mkdir(parents=True, exist_ok=True)
            filename = f"event_{event.type_name}_frame{frame_idx:06d}.mp4"
            out_path = event_dir / filename

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = create_video_writer(out_path, fps, width, height)

            # Seek to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(start_frame, end_frame):
                ret, clip_frame = cap.read()
                if not ret:
                    break
                writer.write(clip_frame)
            writer.release()

            # Seek back to current position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            logger.debug("Evidence clip saved: %s", out_path)
            return out_path
        except Exception as exc:
            logger.error("Failed to save evidence clip: %s", exc)
            return None


# ------------------------------------------------------------------
# Standalone helper for single-frame inference (camera mode)
# ------------------------------------------------------------------
def process_camera_frame(
    frame: np.ndarray,
    detector: YOLODetector,
    event_detector: EventDetector,
    frame_idx: int,
    fps: float,
) -> Tuple[np.ndarray, List[Event]]:
    """Run detection + event detection on a single camera frame.

    Returns
    -------
    annotated_frame : np.ndarray
    events : List[Event]
    """
    det_result = detector.detect(frame)
    events = event_detector.process_frame(det_result, frame_idx, fps)
    annotated = detector.draw_boxes(det_result)
    return annotated, events
