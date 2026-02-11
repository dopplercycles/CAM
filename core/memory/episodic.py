"""
CAM Episodic Memory

SQLite-backed timestamped conversation logs that let CAM recall past
interactions naturally. Every task interaction gets recorded — what was
asked, what happened, and the outcome — so CAM can reference history
during the THINK phase.

Think of this as the shop logbook — a chronological record of every job,
every question, every diagnostic. When a customer asks "didn't we look
at this last month?", you flip back through the pages.

Usage:
    from core.memory.episodic import EpisodicMemory

    em = EpisodicMemory()
    em.record("user", "What year was the Sportster introduced?",
              context_tags=["research", "harley"])

    results = em.search(keyword="Sportster")
    for ep in results:
        print(f"[{ep.participant}] {ep.content}")
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.memory.episodic")


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A single episode in the conversation history.

    Attributes:
        episode_id:    Unique identifier for this episode
        timestamp:     When the episode was recorded (UTC ISO string)
        participant:   Who was involved — "user", "assistant", "system"
        content:       The text content of this episode
        context_tags:  Searchable tags (e.g. ["think", "task_start"])
        task_id:       Optional link to a specific task
        metadata:      Additional key-value data
    """
    episode_id: str
    timestamp: str
    participant: str
    content: str
    context_tags: list[str] = field(default_factory=list)
    task_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for JSON/dashboard use."""
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "participant": self.participant,
            "content": self.content,
            "context_tags": self.context_tags,
            "task_id": self.task_id,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """SQLite-backed persistent store for CAM's conversation history.

    Records timestamped episodes — every task interaction, question, and
    outcome. Supports keyword search with flexible filters and provides
    status metrics for the dashboard.

    Sync design: record() and search() are synchronous because SQLite
    single-row ops are sub-millisecond. Only summarize() is async because
    it calls the model router. LIKE-based search is adequate for expected
    volume; FTS5 can be added later without API changes.

    Args:
        db_path:         Path to the SQLite database file.
        retention_days:  How long to keep episodes before cleanup.
    """

    def __init__(
        self,
        db_path: str = "data/memory/episodic.db",
        retention_days: int = 365,
    ):
        self._db_path = db_path
        self._retention_days = retention_days

        # Ensure parent directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_file), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        logger.info(
            "EpisodicMemory initialized (db=%s, retention=%d days)",
            db_file, retention_days,
        )

    # -------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------

    def _init_db(self):
        """Create the episodes table and indexes if they don't exist."""
        cur = self._conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                participant TEXT NOT NULL,
                content TEXT NOT NULL,
                context_tags TEXT,
                task_id TEXT,
                metadata TEXT
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_timestamp
            ON episodes(timestamp)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_task_id
            ON episodes(task_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_participant
            ON episodes(participant)
        """)

        self._conn.commit()

    # -------------------------------------------------------------------
    # Record
    # -------------------------------------------------------------------

    def record(
        self,
        participant: str,
        content: str,
        context_tags: list[str] | None = None,
        task_id: str | None = None,
        metadata: dict | None = None,
    ) -> Episode:
        """Record a new episode in the conversation history.

        Args:
            participant:   Who was involved — "user", "assistant", "system"
            content:       The text content to record
            context_tags:  Optional searchable tags
            task_id:       Optional link to a specific task
            metadata:      Optional extra key-value data

        Returns:
            The created Episode object.
        """
        episode_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        tags = context_tags or []
        meta = metadata or {}

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO episodes (episode_id, timestamp, participant, content,
                                  context_tags, task_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                timestamp,
                participant,
                content,
                json.dumps(tags),
                task_id,
                json.dumps(meta),
            ),
        )
        self._conn.commit()

        episode = Episode(
            episode_id=episode_id,
            timestamp=timestamp,
            participant=participant,
            content=content,
            context_tags=tags,
            task_id=task_id,
            metadata=meta,
        )

        logger.debug(
            "Episode recorded [%s] %.80s (task=%s)",
            participant, content, task_id or "none",
        )
        return episode

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------

    def search(
        self,
        keyword: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        participant: str | None = None,
        task_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Episode]:
        """Search episodes with flexible AND-combined filters.

        All filters are optional. When multiple are provided, they are
        combined with AND. Results are returned newest-first.

        Args:
            keyword:      Substring match against content (LIKE %kw%)
            start_time:   Only episodes at or after this ISO timestamp
            end_time:     Only episodes at or before this ISO timestamp
            participant:  Filter by participant type
            task_id:      Filter by task ID
            limit:        Maximum results to return (default 50)
            offset:       Skip this many results (for pagination)

        Returns:
            List of Episode objects, newest first.
        """
        conditions = []
        params = []

        if keyword:
            conditions.append("content LIKE ?")
            params.append(f"%{keyword}%")

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        if participant:
            conditions.append("participant = ?")
            params.append(participant)

        if task_id:
            conditions.append("task_id = ?")
            params.append(task_id)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT * FROM episodes
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        cur = self._conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()

        return [self._row_to_episode(row) for row in rows]

    # -------------------------------------------------------------------
    # Summarize (async — calls model router)
    # -------------------------------------------------------------------

    async def summarize(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        router=None,
    ) -> str:
        """Generate a natural-language summary of episodes in a time range.

        Fetches episodes in the time range, builds a prompt, and calls
        the model router for a summary. Uses task_complexity="simple"
        to keep costs low.

        Args:
            start_time:  Only episodes at or after this ISO timestamp
            end_time:    Only episodes at or before this ISO timestamp
            router:      ModelRouter instance (required)

        Returns:
            A summary string from the model, or an error message.
        """
        if router is None:
            return "[Error] No model router provided for summarization"

        episodes = self.search(start_time=start_time, end_time=end_time, limit=100)
        if not episodes:
            return "No episodes found in the specified time range."

        # Build the prompt from episode content
        lines = []
        for ep in reversed(episodes):  # chronological order for the model
            lines.append(f"[{ep.timestamp}] [{ep.participant}] {ep.content}")

        prompt = (
            "Summarize the following conversation history concisely. "
            "Focus on what tasks were performed, key decisions, and outcomes.\n\n"
            + "\n".join(lines)
        )

        response = await router.route(
            prompt=prompt,
            task_complexity="simple",
            system_prompt="You are CAM, summarizing your own conversation history.",
        )
        return response.text

    # -------------------------------------------------------------------
    # Get recent
    # -------------------------------------------------------------------

    def get_recent(self, count: int = 20) -> list[Episode]:
        """Return the most recent episodes for dashboard display.

        Args:
            count: Maximum number of episodes to return.

        Returns:
            List of Episode objects, newest first.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?",
            (count,),
        )
        rows = cur.fetchall()
        return [self._row_to_episode(row) for row in rows]

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a snapshot of episodic memory state.

        Used by the dashboard memory panel and orchestrator status.
        """
        cur = self._conn.cursor()

        # Total count
        cur.execute("SELECT COUNT(*) FROM episodes")
        total_count = cur.fetchone()[0]

        # Oldest and newest timestamps
        cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM episodes")
        row = cur.fetchone()
        oldest = row[0] if row else None
        newest = row[1] if row else None

        # Count today
        today_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00")
        cur.execute("SELECT COUNT(*) FROM episodes WHERE timestamp >= ?", (today_start,))
        count_today = cur.fetchone()[0]

        return {
            "total_count": total_count,
            "oldest_timestamp": oldest,
            "newest_timestamp": newest,
            "count_today": count_today,
            "db_path": self._db_path,
            "retention_days": self._retention_days,
        }

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------

    def cleanup(self, before_date: str | None = None) -> int:
        """Delete episodes older than the given date.

        If no date is provided, uses the retention_days setting to
        calculate the cutoff.

        Args:
            before_date: ISO timestamp — delete everything before this.

        Returns:
            Number of episodes deleted.
        """
        if before_date is None:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
            before_date = cutoff.isoformat()

        cur = self._conn.cursor()
        cur.execute("DELETE FROM episodes WHERE timestamp < ?", (before_date,))
        deleted = cur.rowcount
        self._conn.commit()

        if deleted > 0:
            logger.info("Episodic cleanup: deleted %d episodes before %s", deleted, before_date)
        return deleted

    # -------------------------------------------------------------------
    # Close
    # -------------------------------------------------------------------

    def close(self):
        """Close the SQLite connection. Call on shutdown."""
        if self._conn:
            self._conn.close()
            logger.info("EpisodicMemory connection closed")

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        """Convert a SQLite row to an Episode dataclass."""
        # Parse JSON fields safely
        try:
            tags = json.loads(row["context_tags"]) if row["context_tags"] else []
        except (json.JSONDecodeError, TypeError):
            tags = []

        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}

        return Episode(
            episode_id=row["episode_id"],
            timestamp=row["timestamp"],
            participant=row["participant"],
            content=row["content"],
            context_tags=tags,
            task_id=row["task_id"],
            metadata=meta,
        )

    def __repr__(self) -> str:
        status = self.get_status()
        return (
            f"EpisodicMemory(episodes={status['total_count']}, "
            f"db={self._db_path})"
        )


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    # Use a temp file so we don't clobber real data
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    try:
        em = EpisodicMemory(db_path=tmp.name, retention_days=30)

        # Record some episodes
        em.record("user", "What year was the Sportster introduced?",
                  context_tags=["research", "harley"])
        em.record("assistant", "The Harley-Davidson Sportster debuted in 1957.",
                  context_tags=["research", "harley"])
        em.record("system", "Processing task: research M-8 recall",
                  context_tags=["think", "task_start"],
                  task_id="test-task-001")
        em.record("assistant", "Task completed: M-8 recall research done",
                  context_tags=["iterate", "task_complete"],
                  task_id="test-task-001")
        em.record("user", "Draft a script about barn find CB750",
                  context_tags=["content", "youtube"])

        print(f"\n{em}")
        print(f"Status: {em.get_status()}")

        # Search by keyword
        print("\n--- Search: 'Sportster' ---")
        results = em.search(keyword="Sportster")
        for ep in results:
            print(f"  [{ep.participant}] {ep.content[:80]}")

        # Search by task_id
        print("\n--- Search: task_id='test-task-001' ---")
        results = em.search(task_id="test-task-001")
        for ep in results:
            print(f"  [{ep.participant}] {ep.content[:80]}")

        # Get recent
        print("\n--- Recent (3) ---")
        recent = em.get_recent(3)
        for ep in recent:
            print(f"  [{ep.participant}] {ep.content[:80]}")

        # Cleanup test
        deleted = em.cleanup(before_date="2099-01-01T00:00:00")
        print(f"\nCleanup (everything): {deleted} deleted")
        print(f"After cleanup: {em.get_status()}")

        em.close()

    finally:
        os.unlink(tmp.name)
        print(f"\nCleaned up temp file: {tmp.name}")
