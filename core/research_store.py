"""
CAM Research Store

SQLite-backed storage for research results. Every time the research agent
completes a query — web search, page scraping, synthesis — the result
gets stored here with sources, model info, and cost tracking.

Follows the same pattern as core/content_calendar.py so dashboard
integration is consistent.

Usage:
    from core.research_store import ResearchStore, ResearchStatus

    store = ResearchStore()
    entry = store.add_entry(
        query="motorcycle diagnostic scanner market 2026",
        summary="The market is dominated by...",
        sources=[{"title": "...", "url": "...", "snippet": "..."}],
    )
    store.update_entry(entry.entry_id, status=ResearchStatus.COMPLETED.value)
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

logger = logging.getLogger("cam.research_store")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResearchStatus(Enum):
    """Status of a research entry."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STALE = "stale"


# ---------------------------------------------------------------------------
# ResearchEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResearchEntry:
    """A single research result.

    Attributes:
        entry_id:      Unique identifier (UUID string).
        query:         The original research query/question.
        summary:       Synthesized research summary text.
        sources:       List of source dicts (title, url, snippet).
        source_count:  Number of search results found.
        pages_fetched: Number of pages actually fetched and read.
        status:        Current status (in_progress, completed, failed, stale).
        model_used:    Model that generated the synthesis.
        tokens_used:   Total tokens consumed by the synthesis call.
        cost_usd:      Estimated cost in USD.
        created_at:    When the entry was created (ISO string).
        updated_at:    When the entry was last modified (ISO string).
        task_id:       Optional link to the originating task.
        metadata:      Additional key-value data.
    """
    entry_id: str
    query: str
    summary: str = ""
    sources: list[dict] = field(default_factory=list)
    source_count: int = 0
    pages_fetched: int = 0
    status: str = ResearchStatus.IN_PROGRESS.value
    model_used: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
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
            "query": self.query,
            "summary": self.summary,
            "sources": self.sources,
            "source_count": self.source_count,
            "pages_fetched": self.pages_fetched,
            "status": self.status,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "task_id": self.task_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ResearchEntry":
        """Convert a SQLite row to a ResearchEntry."""
        try:
            sources = json.loads(row["sources"]) if row["sources"] else []
        except (json.JSONDecodeError, TypeError):
            sources = []
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}

        return cls(
            entry_id=row["entry_id"],
            query=row["query"],
            summary=row["summary"],
            sources=sources,
            source_count=row["source_count"],
            pages_fetched=row["pages_fetched"],
            status=row["status"],
            model_used=row["model_used"],
            tokens_used=row["tokens_used"],
            cost_usd=row["cost_usd"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            task_id=row["task_id"],
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# ResearchStore
# ---------------------------------------------------------------------------

class ResearchStore:
    """SQLite-backed research result storage.

    Stores research entries with CRUD operations and optional change
    callbacks for real-time dashboard updates.

    Args:
        db_path:    Path to the SQLite database file.
        on_change:  Async callback fired after every mutation (add/update/remove).
    """

    def __init__(
        self,
        db_path: str = "data/research.db",
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

        logger.info("ResearchStore initialized (db=%s)", db_file)

    # -------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------

    def _init_db(self):
        """Create the research_entries table and indexes if they don't exist."""
        cur = self._conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS research_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT UNIQUE NOT NULL,
                query TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                sources TEXT NOT NULL DEFAULT '[]',
                source_count INTEGER NOT NULL DEFAULT 0,
                pages_fetched INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'in_progress',
                model_used TEXT NOT NULL DEFAULT '',
                tokens_used INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                task_id TEXT,
                metadata TEXT
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_research_status
            ON research_entries(status)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_research_created_at
            ON research_entries(created_at)
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
                logger.debug("Research store on_change callback error", exc_info=True)

    # -------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------

    def add_entry(
        self,
        query: str,
        summary: str = "",
        sources: list[dict] | None = None,
        task_id: str | None = None,
        metadata: dict | None = None,
    ) -> ResearchEntry:
        """Create a new research entry and persist it.

        Args:
            query:    The research query/question.
            summary:  Synthesized summary text.
            sources:  List of source dicts (title, url, snippet).
            task_id:  Optional link to originating task.
            metadata: Optional extra key-value data.

        Returns:
            The created ResearchEntry.
        """
        now = datetime.now(timezone.utc).isoformat()
        entry_id = str(uuid.uuid4())
        src = sources or []
        meta = metadata or {}

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO research_entries
                (entry_id, query, summary, sources, source_count, pages_fetched,
                 status, model_used, tokens_used, cost_usd,
                 created_at, updated_at, task_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id, query, summary, json.dumps(src),
                len(src), 0,
                ResearchStatus.IN_PROGRESS.value, "", 0, 0.0,
                now, now, task_id, json.dumps(meta),
            ),
        )
        self._conn.commit()

        entry = ResearchEntry(
            entry_id=entry_id,
            query=query,
            summary=summary,
            sources=src,
            source_count=len(src),
            pages_fetched=0,
            status=ResearchStatus.IN_PROGRESS.value,
            model_used="",
            tokens_used=0,
            cost_usd=0.0,
            created_at=now,
            updated_at=now,
            task_id=task_id,
            metadata=meta,
        )

        logger.info("Research entry created: '%s' (%s)", query[:60], entry.short_id)
        return entry

    def update_entry(self, entry_id: str, **kwargs) -> ResearchEntry | None:
        """Update fields on an existing research entry.

        Allowed fields: query, summary, sources, source_count,
        pages_fetched, status, model_used, tokens_used, cost_usd,
        task_id, metadata.

        Args:
            entry_id: The entry to update.
            **kwargs: Fields to change.

        Returns:
            The updated ResearchEntry, or None if not found.
        """
        allowed = {
            "query", "summary", "sources", "source_count",
            "pages_fetched", "status", "model_used", "tokens_used",
            "cost_usd", "task_id", "metadata",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_entry(entry_id)

        # Always update the timestamp
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Serialize JSON fields
        if "sources" in updates and isinstance(updates["sources"], list):
            updates["sources"] = json.dumps(updates["sources"])
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [entry_id]

        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE research_entries SET {set_clause} WHERE entry_id = ?",
            values,
        )
        self._conn.commit()

        if cur.rowcount == 0:
            return None

        logger.info("Research entry updated: %s", entry_id[:8])
        return self.get_entry(entry_id)

    def remove_entry(self, entry_id: str) -> bool:
        """Delete a research entry by ID.

        Args:
            entry_id: The entry to delete.

        Returns:
            True if found and removed, False otherwise.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM research_entries WHERE entry_id = ?", (entry_id,))
        self._conn.commit()

        if cur.rowcount > 0:
            logger.info("Research entry removed: %s", entry_id[:8])
            return True
        return False

    def get_entry(self, entry_id: str) -> ResearchEntry | None:
        """Return a single research entry by ID, or None."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM research_entries WHERE entry_id = ?", (entry_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return ResearchEntry.from_row(row)

    def list_all(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[ResearchEntry]:
        """Return research entries with optional status filter.

        Args:
            status: Filter by research status.
            limit:  Maximum entries to return.

        Returns:
            List of ResearchEntry objects, newest first.
        """
        conditions = []
        params: list[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        params.append(limit)

        cur = self._conn.cursor()
        cur.execute(
            f"SELECT * FROM research_entries {where_clause} ORDER BY updated_at DESC LIMIT ?",
            params,
        )
        return [ResearchEntry.from_row(row) for row in cur.fetchall()]

    def search_by_query(self, keyword: str, limit: int = 20) -> list[ResearchEntry]:
        """Search research entries by keyword in query and summary.

        Args:
            keyword: Search term (LIKE match).
            limit:   Maximum entries to return.

        Returns:
            List of matching ResearchEntry objects.
        """
        pattern = f"%{keyword}%"
        cur = self._conn.cursor()
        cur.execute(
            """SELECT * FROM research_entries
               WHERE query LIKE ? OR summary LIKE ?
               ORDER BY updated_at DESC LIMIT ?""",
            (pattern, pattern, limit),
        )
        return [ResearchEntry.from_row(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------
    # Dashboard helpers
    # -------------------------------------------------------------------

    def to_broadcast_list(self) -> list[dict]:
        """Return all entries as JSON-serializable dicts for the dashboard."""
        return [e.to_dict() for e in self.list_all()]

    def get_status(self) -> dict:
        """Return a snapshot of research store state — counts by status."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT status, COUNT(*) as cnt FROM research_entries GROUP BY status"
        )
        counts = {row["status"]: row["cnt"] for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) FROM research_entries")
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
            logger.info("ResearchStore connection closed")


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    try:
        store = ResearchStore(db_path=tmp.name)

        # Add entries
        e1 = store.add_entry(
            query="motorcycle diagnostic scanner market 2026",
            summary="The market is growing rapidly with new OBD-M tools...",
            sources=[
                {"title": "Scanner Review", "url": "https://example.com/1", "snippet": "Top picks..."},
                {"title": "Market Analysis", "url": "https://example.com/2", "snippet": "Growth trends..."},
            ],
        )
        e2 = store.add_entry(
            query="Portland motorcycle repair demand",
            task_id="test-task-123",
        )

        print(f"Created {e1.short_id}: {e1.query}")
        print(f"Created {e2.short_id}: {e2.query}")

        # Update
        updated = store.update_entry(
            e1.entry_id,
            status=ResearchStatus.COMPLETED.value,
            model_used="glm-4.7-flash",
            tokens_used=1500,
            cost_usd=0.0,
            pages_fetched=3,
        )
        print(f"Updated {updated.short_id}: status={updated.status}, model={updated.model_used}")

        # List
        print(f"\nAll entries ({len(store.list_all())}):")
        for e in store.list_all():
            print(f"  [{e.status}] {e.query} ({e.short_id})")

        # Search
        results = store.search_by_query("motorcycle")
        print(f"\nSearch 'motorcycle': {len(results)} results")

        # Status
        print(f"\nStatus: {store.get_status()}")

        # Remove
        removed = store.remove_entry(e2.entry_id)
        print(f"\nRemoved {e2.short_id}: {removed}")
        print(f"Entries after removal: {len(store.list_all())}")

        store.close()

    finally:
        os.unlink(tmp.name)
        print(f"\nCleaned up temp file: {tmp.name}")
