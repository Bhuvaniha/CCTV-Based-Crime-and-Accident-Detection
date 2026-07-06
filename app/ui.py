"""Reusable UI components for the CCTV Detection System.

Keeps presentation logic separate from business logic.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

import config as cfg
from database.db_manager import DatabaseManager
from utils.logger import get_logger
from utils.video_utils import is_valid_extension

logger = get_logger(__name__)

# Will be set to True if ML deps are missing (graceful degradation)
_ML_AVAILABLE = True
try:
    from processing.pipeline import VideoProcessor, process_camera_frame
    from detection.detector import YOLODetector
    from events.event_handler import EventDetector
except ImportError as exc:
    _ML_AVAILABLE = False
    _ML_IMPORT_ERROR = str(exc)
    logger.warning("ML pipeline not available: %s", exc)

# ---------------------------------------------------------------------------
# Header & Sidebar
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.title("🎥 AI-Powered Crime & Accident Detection")
    st.markdown(
        "Detect **accidents**, **fights**, and **fire** events from "
        "uploaded videos or live camera feed."
    )
    st.divider()


def render_sidebar() -> None:
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/cctv.png", width=80)
        st.markdown("### Detection System v1.0")
        st.markdown("---")

        db = DatabaseManager()
        st.metric("Total Incidents Logged", db.count())

        st.markdown("---")
        st.markdown("**Storage**")
        img_count = sum(len(list(d.glob("*.*"))) for d in cfg.EVIDENCE_IMAGE_DIR.iterdir() if d.is_dir())
        vid_count = sum(len(list(d.glob("*.*"))) for d in cfg.EVIDENCE_VIDEO_DIR.iterdir() if d.is_dir())
        st.metric("Evidence Images", img_count)
        st.metric("Evidence Videos", vid_count)

        st.markdown("---")
        st.caption("Built with YOLOv8 + OpenCV + Streamlit")


# ---------------------------------------------------------------------------
# Detection Page (Upload + Live Camera)
# ---------------------------------------------------------------------------

def render_detection_page() -> None:
    render_header()
    render_sidebar()
    tab1, tab2 = st.tabs(["📁 Upload Video", "📷 Live Camera"])
    with tab1:
        render_upload_tab()
    with tab2:
        render_camera_tab()


# ---------------------------------------------------------------------------
# Upload Tab
# ---------------------------------------------------------------------------

def render_upload_tab() -> None:
    col_upload, col_status = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Choose a video file",
            type=[ext.lstrip(".") for ext in cfg.ALLOWED_VIDEO_EXTENSIONS],
        )

        if uploaded is not None:
            if not is_valid_extension(uploaded.name):
                st.error(
                    f"Unsupported format. Allowed: "
                    f"{', '.join(cfg.ALLOWED_VIDEO_EXTENSIONS)}"
                )
            else:
                save_path = cfg.UPLOAD_DIR / uploaded.name
                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.session_state.video_path = save_path
                st.session_state.processed = False
                st.session_state.processing_done = False
                st.session_state.processing = False
                st.success(f"Uploaded: **{uploaded.name}**")
                st.video(str(save_path))

        # Show preview of previously uploaded or processed video
        if st.session_state.video_path:
            if st.session_state.get("processed") and st.session_state.video_path:
                processed_path = cfg.PROCESSED_DIR / f"{st.session_state.video_path.stem}_annotated.mp4"
                if processed_path.exists():
                    st.video(str(processed_path))
                    st.caption("✅ Processed video (with detection boxes)")
            else:
                st.video(str(st.session_state.video_path))

    with col_status:
        st.markdown("#### Video Info")
        if st.session_state.video_path:
            _show_video_info(st.session_state.video_path)

        st.markdown("#### Status")
        if st.session_state.processing:
            st.info("⏳ Processing video...")
        elif st.session_state.get("processing_done"):
            st.success("✅ Processing complete!")
            if st.button("🔄 New Video", width='stretch'):
                for k in ("video_path", "processing", "processed", "incidents", "stats", "processing_done"):
                    st.session_state[k] = _get_default(k)
                st.rerun()
        elif st.session_state.video_path and not st.session_state.get("processed"):
            if st.button("▶️ Process Video", width='stretch', type="primary"):
                st.session_state.processing = True
                st.rerun()
        else:
            st.caption("Upload a video to start.")

        # ML warning
        if not _ML_AVAILABLE:
            st.warning(f"⚠️ ML deps missing. See error above.")

    # ---------- Trigger processing ----------
    if st.session_state.processing and st.session_state.video_path and not st.session_state.get("processing_done"):
        _run_upload_processing()


def _show_video_info(video_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error("Could not read video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0
    cap.release()

    st.markdown(f"- **File:** `{video_path.name}`")
    st.markdown(f"- **Size:** {w}x{h}")
    st.markdown(f"- **FPS:** {fps:.1f}")
    st.markdown(f"- **Duration:** {duration:.1f}s")
    st.markdown(f"- **Frames:** {total}")


def _render_upload_controls() -> None:
    # ML availability warning
    if not _ML_AVAILABLE:
        st.warning(
            f"⚠️ ML dependencies not installed. Run:\n\n"
            f"```bash\npip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
            f"pip install ultralytics\n```\n\n"
            f"*Error: {_ML_IMPORT_ERROR}*"
        )

    # Reset button
    if st.session_state.video_path and not st.session_state.processing:
        if st.button("🔄 Reset & Upload New Video", width='stretch'):
            for key in ("video_path", "processing", "processed", "incidents", "stats", "processing_done"):
                st.session_state[key] = _get_default(key)
            st.rerun()

    # ---------- Auto-run processing ----------
    if st.session_state.processing and st.session_state.video_path:
        _run_upload_processing()


def _run_upload_processing() -> None:
    """Execute the video processing pipeline with a progress bar."""
    progress_bar = st.progress(0, text="Initialising...")
    status_placeholder = st.empty()

    def _progress(current: int, total: int, msg: str) -> None:
        pct = min(current / max(total, 1), 1.0)
        progress_bar.progress(pct, text=msg)
        status_placeholder.caption(
            f"Processed {current} / {total} frames ({pct:.0%})"
        )

    try:
        processor = VideoProcessor(st.session_state.video_path)
        stats = processor.process(progress_callback=_progress)

        st.session_state.incidents = stats.incidents
        st.session_state.stats = {
            "total_frames": stats.total_frames,
            "processed_frames": stats.processed_frames,
            "events_detected": stats.events_detected,
            "persons_detected": stats.persons_detected,
            "vehicles_detected": stats.vehicles_detected,
            "processing_time_sec": stats.processing_time_sec,
        }
        st.session_state.processed = True
        st.success(
            f"✅ Processing complete! {stats.events_detected} event(s) detected "
            f"in {stats.processing_time_sec:.1f}s."
        )

        # Show annotated video preview
        st.subheader("🎬 Processed Video (with detections)")
        st.video(str(processor.output_path))

        # Show stats summary
        _render_processing_stats(st.session_state.stats)

    except Exception as exc:
        logger.exception("Processing failed")
        st.error(f"Processing failed: {exc}")
    finally:
        progress_bar.empty()
        status_placeholder.empty()
        st.session_state.processing = False
        st.session_state.processing_done = True


def _render_processing_stats(stats: dict) -> None:
    """Display processing statistics in metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Frames", stats.get("total_frames", 0))
    with col2:
        st.metric("Processed Frames", stats.get("processed_frames", 0))
    with col3:
        st.metric("Events Detected", stats.get("events_detected", 0))
    with col4:
        st.metric("Processing Time", f"{stats.get('processing_time_sec', 0):.1f}s")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Persons Detected", stats.get("persons_detected", 0))
    with col2:
        st.metric("Vehicles Detected", stats.get("vehicles_detected", 0))


def _get_default(key: str):
    defaults = {
        "video_path": None,
        "processing": False,
        "processed": False,
        "incidents": [],
        "stats": {},
        "processing_done": False,
    }
    return defaults.get(key)


# ---------------------------------------------------------------------------
# Camera Tab
# ---------------------------------------------------------------------------

def render_camera_tab() -> None:
    st.markdown("### Live Camera Detection")
    st.caption("Access your webcam to detect events in real-time.")

    if not _ML_AVAILABLE:
        st.warning(
            f"⚠️ ML dependencies not installed. Run:\n\n"
            f"```bash\npip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
            f"pip install ultralytics\n```\n\n"
            f"*Error: {_ML_IMPORT_ERROR}*"
        )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        start_disabled = st.session_state.camera_active or not _ML_AVAILABLE
        if st.button(
            "▶️ Start Camera",
            disabled=start_disabled,
            type="primary",
            width='stretch',
        ):
            st.session_state.camera_active = True
            st.session_state.camera_frame_count = 0
            st.session_state.camera_event_count = 0
            st.rerun()

    with col2:
        stop_disabled = not st.session_state.camera_active
        if st.button(
            "⏹️ Stop Camera",
            disabled=stop_disabled,
            width='stretch',
        ):
            st.session_state.camera_active = False
            st.session_state.camera_frame = None
            st.rerun()

    with col3:
        st.caption("Detection runs while camera is active. Events are logged automatically.")

    if st.session_state.camera_active:
        _run_camera_loop()
    else:
        st.info("Camera is off. Click **Start Camera** to begin.")


def _run_camera_loop() -> None:
    """Open webcam, capture + process frames with ML detection until user stops."""
    FRAME_PLACEHOLDER = st.empty()
    alert_placeholder = st.empty()
    status_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access camera. Check permissions.")
        st.session_state.camera_active = False
        return

    try:
        detector = YOLODetector()
        event_detector = EventDetector()
        db = DatabaseManager()
    except Exception as exc:
        st.error(f"Failed to initialise ML models: {exc}")
        st.session_state.camera_active = False
        cap.release()
        return

    frame_count = 0
    event_count = st.session_state.get("camera_event_count", 0)

    try:
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.warning("Lost camera feed.")
                break

            frame_count += 1

            # Process every N-th frame for performance
            if frame_count % cfg.PROCESSING_FRAME_SKIP == 0:
                try:
                    annotated, events = process_camera_frame(
                        frame, detector, event_detector,
                        frame_count, 30.0,  # assume ~30fps
                    )
                    display_frame = annotated
                except Exception as exc:
                    logger.error("Camera detection error: %s", exc)
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                events = []

            # Handle detected events
            for event in events:
                event_count += 1
                # Save evidence
                ts = time.strftime("%H:%M:%S")
                img_name = f"cam_{event.type_name}_{int(time.time())}.jpg"
                img_dir = cfg.EVIDENCE_IMAGE_DIR / event.type_name.lower()
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / img_name
                cv2.imwrite(str(img_path), event.frame)

                db.insert_incident(
                    event_type=event.type_name,
                    timestamp_sec=time.time(),
                    confidence=event.confidence,
                    evidence_image=str(img_path),
                )

                alert_placeholder.error(
                    f"🚨 **{event.type_name}** detected at {ts} "
                    f"(conf: {event.confidence:.2f})"
                )

            # Display the frame
            FRAME_PLACEHOLDER.image(display_frame, channels="RGB", width='stretch')

            status_placeholder.info(
                f"📡 Camera active | Frames: {frame_count} | "
                f"Events: {event_count}"
            )

            st.session_state.camera_event_count = event_count

    finally:
        cap.release()
        st.session_state.camera_active = False
        st.session_state.camera_frame_count = frame_count
        alert_placeholder.success(
            f"Camera stopped. {event_count} event(s) detected."
        )
        st.rerun()


# ---------------------------------------------------------------------------
# Evidence Page
# ---------------------------------------------------------------------------

def render_evidence_page() -> None:
    render_sidebar()
    st.title("📂 Evidence Gallery")
    st.caption("Browse captured evidence images and video clips organised by event type.")

    evidence_tabs = st.tabs(["📸 Images", "🎬 Video Clips", "📋 Incident Log"])

    with evidence_tabs[0]:
        _render_evidence_images()

    with evidence_tabs[1]:
        _render_evidence_videos()

    with evidence_tabs[2]:
        _render_incident_log()


def _render_evidence_images() -> None:
    event_dirs = sorted(
        d for d in cfg.EVIDENCE_IMAGE_DIR.iterdir() if d.is_dir()
    )
    if not event_dirs:
        st.info("No evidence images captured yet. Process a video or run camera detection.")
        return

    for event_dir in event_dirs:
        image_files = sorted(event_dir.glob("*.*"))
        if not image_files:
            continue
        st.subheader(f"🚦 {event_dir.name.title()}")
        cols = st.columns(3)
        for idx, img_path in enumerate(image_files):
            with cols[idx % 3]:
                st.image(str(img_path), width='stretch')
                st.caption(f"{img_path.name}")

                with open(img_path, "rb") as f:
                    st.download_button(
                        label="Download",
                        data=f,
                        file_name=img_path.name,
                        mime="image/jpeg",
                        key=f"dl_img_{img_path.stem}",
                        width='stretch',
                    )


def _render_evidence_videos() -> None:
    event_dirs = sorted(
        d for d in cfg.EVIDENCE_VIDEO_DIR.iterdir() if d.is_dir()
    )
    if not event_dirs:
        st.info("No evidence video clips captured yet.")
        return

    for event_dir in event_dirs:
        video_files = sorted(event_dir.glob("*.*"))
        if not video_files:
            continue
        st.subheader(f"🚦 {event_dir.name.title()}")
        for vid_path in video_files:
            st.video(str(vid_path))
            st.caption(f"{vid_path.name}")

            with open(vid_path, "rb") as f:
                st.download_button(
                    label="Download Clip",
                    data=f,
                    file_name=vid_path.name,
                    mime="video/mp4",
                    key=f"dl_vid_{vid_path.stem}",
                )


def _render_incident_log() -> None:
    db = DatabaseManager()
    incidents = db.get_all_incidents()

    if not incidents:
        st.info("No incidents recorded yet.")
        return

    # Convert to display format
    import pandas as pd

    df = pd.DataFrame(incidents)
    df = df.rename(columns={
        "id": "ID",
        "event_type": "Event Type",
        "timestamp_sec": "Timestamp (s)",
        "confidence": "Confidence",
        "evidence_image": "Image",
        "evidence_video": "Video",
        "processing_date": "Date",
    })

    st.dataframe(df, width='stretch', hide_index=True)

    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Incidents", len(df))
    with col2:
        if "Event Type" in df.columns:
            types = df["Event Type"].value_counts().to_dict()
            most_common = max(types, key=types.get)
            st.metric("Most Common", most_common)
    with col3:
        if "Confidence" in df.columns:
            conf_vals = pd.to_numeric(df["Confidence"], errors="coerce")
            avg = f"{conf_vals.mean():.2f}" if not conf_vals.isna().all() else "N/A"
            st.metric("Confidence Avg", avg)
        else:
            st.metric("Confidence Avg", "N/A")
