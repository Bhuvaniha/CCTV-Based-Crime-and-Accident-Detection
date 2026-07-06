"""Video I/O utilities.

Thin wrappers around OpenCV ``VideoCapture`` / ``VideoWriter``
for consistent frame reading and processed video writing.
"""

from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

import config as cfg
from utils.logger import get_logger

logger = get_logger(__name__)


def get_video_capture(video_path: Path) -> Tuple[cv2.VideoCapture, dict]:
    """Open a video file and return (cap, info).

    Parameters
    ----------
    video_path : Path

    Returns
    -------
    cap : cv2.VideoCapture
    info : dict  with keys ``fps``, ``width``, ``height``, ``total_frames``

    Raises
    ------
    FileNotFoundError
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    info = dict(fps=fps, width=width, height=height, total_frames=total_frames)
    logger.debug("Opened %s  %s", video_path, info)
    return cap, info


def frame_generator(
    cap: cv2.VideoCapture,
    skip: int = 1,
) -> Generator[Tuple[np.ndarray, int], None, None]:
    """Yield (frame, frame_index) tuples.

    Parameters
    ----------
    cap : cv2.VideoCapture
    skip : int
        Process every ``skip``-th frame (1 = all frames).
    """
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            yield frame, idx
        idx += 1
    cap.release()


def create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    fourcc: str = "mp4v",
) -> cv2.VideoWriter:
    """Create a ``VideoWriter`` for the processed output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))
    logger.debug("VideoWriter created: %s  (%s, %dx%d, %.2f fps)",
                 output_path, fourcc, width, height, fps)
    return writer


def is_valid_extension(filename: str) -> bool:
    """Check if the file extension is in the allowed set."""
    ext = Path(filename).suffix.lower()
    return ext in cfg.ALLOWED_VIDEO_EXTENSIONS
