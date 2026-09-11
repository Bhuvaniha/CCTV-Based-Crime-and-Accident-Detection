# CCTV Crime & Accident Detection System

AI-powered surveillance video analysis that detects accidents, fights, and fires in real-time using YOLOv8 object detection. When an event is detected, the system automatically captures evidence (screenshot + video clip) and logs the incident to a database.

## Features

- **Video Upload & Processing** -- Upload CCTV footage (MP4, AVI, MOV up to 500MB) for offline analysis
- **Live Camera Detection** -- Real-time webcam feed with instant event detection
- **Event Detection** -- Heuristic-based detection for:
  - **Accidents** -- Vehicle proximity and overlap analysis (IoU + centroid distance)
  - **Fights** -- Person proximity detection
  - **Fire** -- Placeholder (class-based detection, not yet wired into pipeline)
- **Evidence Capture** -- Automatic screenshots and video clips around detected events
- **Incident Logging** -- SQLite database with timestamps, confidence scores, and evidence paths
- **Evidence Gallery** -- Browse and download evidence images, video clips, and incident logs via a web dashboard

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| UI | Streamlit |
| Object Detection | YOLOv8 (Ultralytics) |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Database | SQLite |

## Project Structure

```
CCTV/
├── run.py                  # Entry point (launches Streamlit app)
├── config.py               # All configurable parameters
├── requirements.txt        # Python dependencies
├── incidents.db            # SQLite database (auto-created)
│
├── app/                    # Streamlit web UI
│   ├── main.py             # App entry point
│   └── ui.py               # UI components (detection, evidence, camera)
│
├── detection/              # Object detection
│   └── detector.py         # YOLOv8 wrapper for person/vehicle detection
│
├── processing/             # Video processing pipeline
│   └── pipeline.py         # End-to-end: detect -> events -> evidence -> DB
│
├── events/                 # Event detection logic
│   └── event_handler.py    # Accident, fight, and fire detection algorithms
│
├── database/               # Data persistence
│   └── db_manager.py       # SQLite CRUD operations
│
├── utils/                  # Shared utilities
│   ├── logger.py           # Logging configuration
│   └── video_utils.py      # Video I/O helpers
│
├── tracking/               # Reserved (future object tracking)
├── models/                 # YOLO model weights (yolov8n.pt)
├── uploads/                # User-uploaded videos
├── processed/              # Annotated output videos
├── evidence/               # Captured evidence
│   ├── images/             # Screenshots (accident/, fight/, fire/)
│   └── videos/             # Video clips (accident/, fight/, fire/)
└── logs/                   # Application logs
```

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd CCTV

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The YOLOv8 nano model (`models/yolov8n.pt`) is included. All directories are auto-created on first run.

## Usage

```bash
# Launch the web app
python run.py

# Or directly with Streamlit
streamlit run app/main.py
```

The app opens at **http://localhost:8501**.

### Video Analysis
1. Go to the **Detection** tab
2. Upload a video file or select the Camera tab for live feed
3. Click **Process Video** to start analysis
4. View annotated video with bounding boxes and detected events

### Evidence Gallery
1. Go to the **Evidence** page from the sidebar
2. Browse evidence images, video clips, or the incident log
3. Download any evidence item directly from the browser

## Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `YOLO_MODEL_NAME` | `yolov8n.pt` | YOLO model (nano = fast, use `yolov8s.pt` for better accuracy) |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence |
| `YOLO_DEVICE` | `cpu` | Set to `cuda` for GPU acceleration |
| `PROCESSING_FRAME_SKIP` | `2` | Process every N-th frame |
| `EVIDENCE_CLIP_SECONDS` | `3` | Seconds before/after event for clips |
| `ACCIDENT_IOU_THRESHOLD` | `0.02` | IoU ratio to trigger accident detection |
| `FIGHT_PROXIMITY_THRESHOLD` | `80` | Pixel distance between persons for fight |
| `MAX_UPLOAD_SIZE_MB` | `500` | Maximum upload file size |

## Limitations

- Fire detection is defined but not wired into the processing pipeline
- Accident and fight detection use distance-based heuristics, not trained event classifiers
- CPU inference may be slow on longer videos; set `YOLO_DEVICE = "cuda"` for GPU support
- The `tracking/` module is a placeholder for future object tracking features
