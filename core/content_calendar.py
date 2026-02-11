"""
CAM Content Calendar

SQLite-backed content pipeline tracker. Every piece of content — script
outlines, video ideas, episode descriptions, research notes — gets an
entry that moves through the pipeline: idea → planned → in_progress →
review → published → archived.

Think of this as the whiteboard in the shop where George tracks what's
being filmed, what needs editing, and what's ready to go live.

Usage:
    from core.content_calendar import ContentCalendar, ContentType, ContentStatus

    cal = ContentCalendar()
    entry = cal.add_entry(
        title="DR650 Valve Adjustment Tutorial",
        content_type=ContentType.VIDEO,
        description="Step-by-step valve check and adjustment on the DR650",
    )
    cal.update_entry(entry.entry_id, status=ContentStatus.IN_PROGRESS.value)
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("cam.content_calendar")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentType(Enum):
    """What kind of content this entry represents."""
    SCRIPT_OUTLINE = "script_outline"
    VIDEO = "video"
    EPISODE_DESCRIPTION = "episode_description"
    RESEARCH = "research"
    GENERAL = "general"


class ContentStatus(Enum):
    """Pipeline stage for a content entry."""
    IDEA = "idea"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    PUBLISHED = "published"
    ARCHIVED = "archived"


# ---------------------------------------------------------------------------
# ContentEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContentEntry:
    """A single piece of content in the pipeline.

    Attributes:
        entry_id:       Unique identifier (UUID string)
        title:          Short title for the content piece
        content_type:   What kind of content (video, script, etc.)
        description:    Longer description of what this content covers
        status:         Current pipeline stage
        scheduled_date: Optional target date (ISO string or None)
        body:           The actual content text (script draft, outline, etc.)
        created_at:     When the entry was created (ISO string)
        updated_at:     When the entry was last modified (ISO string)
        task_id:        Optional link to the originating task
        metadata:       Additional key-value data
    """
    entry_id: str
    title: str
    content_type: str = ContentType.GENERAL.value
    description: str = ""
    status: str = ContentStatus.IDEA.value
    scheduled_date: str | None = None
    body: str = ""
    created_at: str = ""
    updated_at: str = ""
    task_id: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 characters of the entry ID for display."""
        return self.entry_id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/dashboard use."""
        return {
            "entry_id": self.entry_id,
            "short_id": self.short_id,
            "title": self.title,
            "content_type": self.content_type,
            "description": self.description,
            "status": self.status,
            "scheduled_date": self.scheduled_date,
            "body": self.body,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "task_id": self.task_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentEntry":
        """Deserialize from a dictionary."""
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            title=data.get("title", ""),
            content_type=data.get("content_type", ContentType.GENERAL.value),
            description=data.get("description", ""),
            status=data.get("status", ContentStatus.IDEA.value),
            scheduled_date=data.get("scheduled_date"),
            body=data.get("body", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            task_id=data.get("task_id"),
            metadata=data.get("metadata") or {},
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ContentEntry":
        """Convert a SQLite row to a ContentEntry."""
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}

        return cls(
            entry_id=row["entry_id"],
            title=row["title"],
            content_type=row["content_type"],
            description=row["description"],
            status=row["status"],
            scheduled_date=row["scheduled_date"],
            body=row["body"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            task_id=row["task_id"],
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# ContentCalendar
# ---------------------------------------------------------------------------

class ContentCalendar:
    """SQLite-backed content pipeline tracker.

    Stores content entries with CRUD operations and optional change
    callbacks for real-time dashboard updates.

    Args:
        db_path:    Path to the SQLite database file.
        on_change:  Async callback fired after every mutation (add/update/remove).
    """

    def __init__(
        self,
        db_path: str = "data/content_calendar.db",
        on_change: Callable[[], Coroutine] | None = None,
    ):
        self._db_path = db_path
        self._on_change = on_change

        # Ensure parent directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_file), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        logger.info("ContentCalendar initialized (db=%s)", db_file)

    # -------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------

    def _init_db(self):
        """Create the content_entries table and indexes if they don't exist."""
        cur = self._conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS content_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content_type TEXT NOT NULL DEFAULT 'general',
                description TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'idea',
                scheduled_date TEXT,
                body TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                task_id TEXT,
                metadata TEXT
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_status
            ON content_entries(status)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_type
            ON content_entries(content_type)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_scheduled_date
            ON content_entries(scheduled_date)
        """)

        self._conn.commit()

    # -------------------------------------------------------------------
    # Change notification
    # -------------------------------------------------------------------

    async def _notify_change(self):
        """Fire the on_change callback if set."""
        if self._on_change is not None:
            try:
                await self._on_change()
            except Exception:
                logger.debug("Content calendar on_change callback error", exc_info=True)

    # -------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------

    def add_entry(
        self,
        title: str,
        content_type: str = ContentType.GENERAL.value,
        description: str = "",
        scheduled_date: str | None = None,
        body: str = "",
        task_id: str | None = None,
        metadata: dict | None = None,
    ) -> ContentEntry:
        """Create a new content entry and persist it.

        Args:
            title:          Short title for the content piece.
            content_type:   One of ContentType values.
            description:    Longer description.
            scheduled_date: Optional target date (ISO string).
            body:           Content text (script, outline, etc.).
            task_id:        Optional link to originating task.
            metadata:       Optional extra key-value data.

        Returns:
            The created ContentEntry.
        """
        now = datetime.now(timezone.utc).isoformat()
        entry_id = str(uuid.uuid4())
        meta = metadata or {}

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO content_entries
                (entry_id, title, content_type, description, status,
                 scheduled_date, body, created_at, updated_at, task_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id, title, content_type, description,
                ContentStatus.IDEA.value, scheduled_date, body,
                now, now, task_id, json.dumps(meta),
            ),
        )
        self._conn.commit()

        entry = ContentEntry(
            entry_id=entry_id,
            title=title,
            content_type=content_type,
            description=description,
            status=ContentStatus.IDEA.value,
            scheduled_date=scheduled_date,
            body=body,
            created_at=now,
            updated_at=now,
            task_id=task_id,
            metadata=meta,
        )

        logger.info(
            "Content entry created: '%s' (%s) type=%s",
            title, entry.short_id, content_type,
        )
        return entry

    def update_entry(self, entry_id: str, **kwargs) -> ContentEntry | None:
        """Update fields on an existing content entry.

        Allowed fields: title, content_type, description, status,
        scheduled_date, body, task_id, metadata.

        Args:
            entry_id: The entry to update.
            **kwargs: Fields to change.

        Returns:
            The updated ContentEntry, or None if not found.
        """
        allowed = {
            "title", "content_type", "description", "status",
            "scheduled_date", "body", "task_id", "metadata",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_entry(entry_id)

        # Always update the timestamp
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Serialize metadata if present
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [entry_id]

        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE content_entries SET {set_clause} WHERE entry_id = ?",
            values,
        )
        self._conn.commit()

        if cur.rowcount == 0:
            return None

        logger.info("Content entry updated: %s", entry_id[:8])
        return self.get_entry(entry_id)

    def remove_entry(self, entry_id: str) -> bool:
        """Delete a content entry by ID.

        Args:
            entry_id: The entry to delete.

        Returns:
            True if found and removed, False otherwise.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM content_entries WHERE entry_id = ?", (entry_id,))
        self._conn.commit()

        if cur.rowcount > 0:
            logger.info("Content entry removed: %s", entry_id[:8])
            return True
        return False

    def get_entry(self, entry_id: str) -> ContentEntry | None:
        """Return a single content entry by ID, or None."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM content_entries WHERE entry_id = ?", (entry_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return ContentEntry.from_row(row)

    def list_all(
        self,
        status: str | None = None,
        content_type: str | None = None,
    ) -> list[ContentEntry]:
        """Return content entries with optional filters.

        Args:
            status:       Filter by pipeline status.
            content_type: Filter by content type.

        Returns:
            List of ContentEntry objects, newest first.
        """
        conditions = []
        params = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if content_type:
            conditions.append("content_type = ?")
            params.append(content_type)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        cur = self._conn.cursor()
        cur.execute(
            f"SELECT * FROM content_entries {where_clause} ORDER BY updated_at DESC",
            params,
        )
        return [ContentEntry.from_row(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------
    # Dashboard helpers
    # -------------------------------------------------------------------

    def to_broadcast_list(self) -> list[dict]:
        """Return all entries as JSON-serializable dicts for the dashboard."""
        return [e.to_dict() for e in self.list_all()]

    def get_status(self) -> dict:
        """Return a snapshot of content calendar state — counts by status."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT status, COUNT(*) as cnt FROM content_entries GROUP BY status"
        )
        counts = {row["status"]: row["cnt"] for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) FROM content_entries")
        total = cur.fetchone()[0]

        return {
            "total": total,
            "by_status": counts,
            "db_path": self._db_path,
        }

    # -------------------------------------------------------------------
    # Close
    # -------------------------------------------------------------------

    def close(self):
        """Close the SQLite connection. Call on shutdown."""
        if self._conn:
            self._conn.close()
            logger.info("ContentCalendar connection closed")


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    try:
        cal = ContentCalendar(db_path=tmp.name)

        # Add entries
        e1 = cal.add_entry(
            title="DR650 Valve Adjustment Tutorial",
            content_type=ContentType.VIDEO.value,
            description="Step-by-step valve check on the DR650",
        )
        e2 = cal.add_entry(
            title="Highway 20 Research Notes",
            content_type=ContentType.RESEARCH.value,
            description="Route planning and historical landmarks",
            scheduled_date="2026-06-01",
        )

        print(f"Created {e1.short_id}: {e1.title}")
        print(f"Created {e2.short_id}: {e2.title}")

        # Update
        updated = cal.update_entry(e1.entry_id, status=ContentStatus.IN_PROGRESS.value)
        print(f"Updated {updated.short_id}: status={updated.status}")

        # List
        print(f"\nAll entries ({len(cal.list_all())}):")
        for e in cal.list_all():
            print(f"  [{e.status}] {e.title} ({e.short_id})")

        # Status
        print(f"\nStatus: {cal.get_status()}")

        # Remove
        removed = cal.remove_entry(e2.entry_id)
        print(f"\nRemoved {e2.short_id}: {removed}")
        print(f"Entries after removal: {len(cal.list_all())}")

        cal.close()

    finally:
        os.unlink(tmp.name)
        print(f"\nCleaned up temp file: {tmp.name}")
