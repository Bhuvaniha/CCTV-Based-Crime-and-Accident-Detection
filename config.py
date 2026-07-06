"""Centralized configuration for the Crime & Accident Detection System.

All configurable parameters are defined here so they can be tuned
without touching business logic.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
EVIDENCE_IMAGE_DIR = BASE_DIR / "evidence" / "images"
EVIDENCE_VIDEO_DIR = BASE_DIR / "evidence" / "videos"

# Event-type subdirectories for organised evidence storage
EVIDENCE_EVENT_TYPES = ("accident", "fight", "fire")

# Ensure all data directories exist
for _dir in (UPLOAD_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR,
             EVIDENCE_IMAGE_DIR, EVIDENCE_VIDEO_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
for _type in EVIDENCE_EVENT_TYPES:
    (EVIDENCE_IMAGE_DIR / _type).mkdir(parents=True, exist_ok=True)
    (EVIDENCE_VIDEO_DIR / _type).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Video / Processing
# ---------------------------------------------------------------------------
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
MAX_UPLOAD_SIZE_MB = 500
PROCESSING_FRAME_SKIP = 2          # process every N-th frame
EVIDENCE_CLIP_SECONDS = 3          # seconds before & after event

# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------
YOLO_MODEL_NAME = "yolov8n.pt"     # nano – fast for V1, swap to s/m/l/x later
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_DEVICE = "cpu"                # "cuda" if GPU available

# COCO class ids we care about
YOLO_PERSON_CLASS_ID = 0
YOLO_VEHICLE_CLASS_IDS = {2, 3, 5, 7}   # car, motorcycle, bus, truck

# ---------------------------------------------------------------------------
# Event Detection
# ---------------------------------------------------------------------------
ACCIDENT_IOU_THRESHOLD = 0.02      # overlap ratio (low = sensitive)
ACCIDENT_CENTROID_DISTANCE = 100   # px — trigger if centroids within this
ACCIDENT_SPEED_THRESHOLD = 5.0     # pixels/frame delta (tunable)
FIGHT_PROXIMITY_THRESHOLD = 80     # px distance between persons
FIGHT_MOTION_THRESHOLD = 5.0       # avg motion magnitude per person
FIRE_CONFIDENCE_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_PATH = BASE_DIR / "incidents.db"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "DEBUG"
LOG_FILE = LOG_DIR / "system.log"
LOG_FORMAT = "%(asctime)s | %(name)-24s | %(levelname)-6s | %(message)s"
