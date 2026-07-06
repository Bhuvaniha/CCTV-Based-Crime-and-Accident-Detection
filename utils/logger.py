"""Logging configuration for the application.

Provides a single `get_logger` factory so every module uses
consistent formatting and output targets (console + file).
"""

import logging
import sys
from pathlib import Path

import config as cfg


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for the entire application.

    Parameters
    ----------
    name : str
        Typically ``__name__`` from the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:                     # already configured
        return logger

    logger.setLevel(getattr(logging, cfg.LOG_LEVEL.upper(), logging.DEBUG))

    formatter = logging.Formatter(cfg.LOG_FORMAT)

    # --- Console handler ------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler ----------------------------------------------------
    log_path = Path(cfg.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
