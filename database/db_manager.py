"""SQLite database layer for incident persistence.

Uses a single ``incidents`` table with the schema defined in the
requirements.  All CRUD operations are encapsulated here so the
rest of the application never writes raw SQL.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import config as cfg
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS incidents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT    NOT NULL,
    timestamp_sec   REAL    NOT NULL,
    confidence      REAL    NOT NULL,
    evidence_image  TEXT,
    evidence_video  TEXT,
    processing_date TEXT    NOT NULL
);
"""

_COLUMNS = ("id", "event_type", "timestamp_sec", "confidence",
            "evidence_image", "evidence_video", "processing_date")


class DatabaseManager:
    """Manages the SQLite connection and incident records.

    Usage::

        db = DatabaseManager()
        db.insert_incident(...)
        all_incidents = db.get_all_incidents()
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or cfg.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            logger.info("Connected to database: %s", self.db_path)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("Database connection closed.")

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(_CREATE_TABLE)
        conn.commit()
        logger.debug("Database schema ensured.")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def insert_incident(
        self,
        event_type: str,
        timestamp_sec: float,
        confidence: float,
        evidence_image: str = "",
        evidence_video: str = "",
    ) -> int:
        """Insert a new incident and return its ID."""
        conn = self._connect()
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            """INSERT INTO incidents
               (event_type, timestamp_sec, confidence,
                evidence_image, evidence_video, processing_date)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event_type, float(timestamp_sec), float(confidence),
             evidence_image, evidence_video, now),
        )
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.info("Inserted incident #%d  type=%s  time=%.2fs", row_id, event_type, timestamp_sec)
        return row_id

    def get_all_incidents(self) -> List[Dict]:
        """Return all incidents as a list of dicts."""
        conn = self._connect()
        rows = conn.execute("SELECT * FROM incidents ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def get_incident(self, incident_id: int) -> Optional[Dict]:
        """Return a single incident or ``None``."""
        conn = self._connect()
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        return dict(row) if row else None

    def delete_incident(self, incident_id: int) -> bool:
        """Delete an incident by ID.  Returns ``True`` if deleted."""
        conn = self._connect()
        cur = conn.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))
        conn.commit()
        return cur.rowcount > 0

    def count(self) -> int:
        """Total number of incidents."""
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
