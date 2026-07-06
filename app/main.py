"""Streamlit application entry point.

Run with::

    streamlit run app/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.ui import (
    render_detection_page,
    render_evidence_page,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CCTV Crime & Accident Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "video_path": None,
    "processing": False,
    "processed": False,
    "processing_done": False,
    "incidents": [],
    "stats": {},
    "camera_active": False,
    "camera_frame": None,
    "camera_frame_count": 0,
    "camera_event_count": 0,
    "active_tab": "Upload Video",
}
for key, val in _DEFAULTS.items():
    st.session_state.setdefault(key, val)


def main() -> None:
    detection_page = st.Page(
        render_detection_page, title="Detection", icon="🎥", default=True,
    )
    evidence_page = st.Page(
        render_evidence_page, title="Evidence", icon="📂",
    )

    pg = st.navigation([detection_page, evidence_page])
    pg.run()


if __name__ == "__main__":
    main()
