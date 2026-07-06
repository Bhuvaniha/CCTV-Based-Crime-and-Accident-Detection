#!/usr/bin/env python3
"""Convenience entry point for the CCTV detection system.

Usage::

    python run.py          # launches Streamlit UI
    python run.py --help   # shows available options
"""

import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().parent / "app" / "main.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
