"""
CAM Security Audit Log

SQLite-backed persistent audit trail for all security-relevant actions.
Every task that goes through the orchestrator's permission check gets
logged here with its tier, risk level, and result (executed, approved,
rejected, blocked, timeout).

This is separate from the EventLogger (in-memory ring buffer for general
activity). The audit log is:
    - Persistent (SQLite, survives restarts)
    - Queryable (filter by risk, actor, time)
    - Exportable (for review / compliance)

Follows the same pattern as core/content_calendar.py.

Usage:
    from security.audit import SecurityAuditLog

    audit = SecurityAuditLog()
    entry = audit.log_action(
        action_type="publish_content",
        actor="dashboard",
        target="publish video to youtube",
        result="approved",
        risk_level="medium",
        tier=2,
        task_id="abc123",
    )
    recent = audit.get_recent(limit=50)
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger("cam.security_audit")


# ---------------------------------------------------------------------------
# AuditEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """A single security audit log entry.

    Attributes:
        entry_id:         Unique identifier (UUID string)
        timestamp:        When this action occurred (ISO string)
        action_type:      Permission action type (e.g. 'publish_content', 'shell_command')
        actor:            Who initiated the action ('orchestrator', 'dashboard', agent name)
        target:           Task description (truncated for display)
        result:           Outcome: 'executed', 'approved', 'rejected', 'blocked', 'timeout', 'pending'
        risk_level:       'low', 'medium', 'high', or 'critical'
        tier:             Permission tier (1, 2, or 3)
        task_id:          Originating task ID (if applicable)
        approved_by:      Who approved (None, 'dashboard', or specific user)
        approval_time_s:  Seconds from request to approval (None if not applicable)
        metadata:         Additional key-value data
    """
    entry_id: str
    timestamp: str
    action_type: str
    actor: str
    target: str
    result: str
    risk_level: str
    tier: int
    task_id: str = ""
    approved_by: str | None = None
    approval_time_s: float | None = None
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
            "timestamp": self.timestamp,
            "action_type": self.action_type,
            "actor": self.actor,
            "target": self.target,
            "result": self.result,
            "risk_level": self.risk_level,
            "tier": self.tier,
            "task_id": self.task_id,
            "approved_by": self.approved_by,
            "approval_time_s": self.approval_time_s,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "AuditEntry":
        """Convert a SQLite row to an AuditEntry."""
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}

        return cls(
            entry_id=row["entry_id"],
            timestamp=row["timestamp"],
            action_type=row["action_type"],
            actor=row["actor"],
            target=row["target"],
            result=row["result"],
            risk_level=row["risk_level"],
            tier=row["tier"],
            task_id=row["task_id"] or "",
            approved_by=row["approved_by"],
            approval_time_s=row["approval_time_s"],
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# SecurityAuditLog
# ---------------------------------------------------------------------------

class SecurityAuditLog:
    """SQLite-backed security audit trail.

    Logs every action that passes through the permission system with
    its tier, risk level, and result. Provides filtering and status
    methods for the dashboard.

    Args:
        db_path:    Path to the SQLite database file.
        on_change:  Async callback fired after every new log entry.
    """

    def __init__(
        self,
        db_path: str = "data/security_audit.db",
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

        logger.info("SecurityAuditLog initialized (db=%s)", db_file)

    # -------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------

    def _init_db(self):
        """Create the audit_log table and indexes if they don't exist."""
        cur = self._conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                action_type TEXT NOT NULL,
                actor TEXT NOT NULL,
                target TEXT NOT NULL DEFAULT '',
                result TEXT NOT NULL,
                risk_level TEXT NOT NULL DEFAULT 'low',
                tier INTEGER NOT NULL DEFAULT 1,
                task_id TEXT,
                approved_by TEXT,
                approval_time_s REAL,
                metadata TEXT
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
            ON audit_log(timestamp)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_risk_level
            ON audit_log(risk_level)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_actor
            ON audit_log(actor)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_result
            ON audit_log(result)
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
                logger.debug("Security audit on_change callback error", exc_info=True)

    # -------------------------------------------------------------------
    # Log action
    # -------------------------------------------------------------------

    def log_action(
        self,
        action_type: str,
        actor: str,
        target: str,
        result: str,
        risk_level: str = "low",
        tier: int = 1,
        task_id: str = "",
        approved_by: str | None = None,
        approval_time_s: float | None = None,
        metadata: dict | None = None,
    ) -> AuditEntry:
        """Record a security-relevant action in the audit log.

        Args:
            action_type:      Permission action type (e.g. 'publish_content')
            actor:            Who initiated ('orchestrator', 'dashboard', agent name)
            target:           Task description (truncated to 500 chars)
            result:           'executed', 'approved', 'rejected', 'blocked', 'timeout', 'pending'
            risk_level:       'low', 'medium', 'high', or 'critical'
            tier:             Permission tier (1, 2, or 3)
            task_id:          Originating task ID
            approved_by:      Who approved (None or 'dashboard')
            approval_time_s:  Seconds from request to approval
            metadata:         Additional data

        Returns:
            The created AuditEntry.
        """
        now = datetime.now(timezone.utc).isoformat()
        entry_id = str(uuid.uuid4())
        meta = metadata or {}

        # Truncate target for storage
        target_truncated = target[:500]

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO audit_log
                (entry_id, timestamp, action_type, actor, target, result,
                 risk_level, tier, task_id, approved_by, approval_time_s, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id, now, action_type, actor, target_truncated, result,
                risk_level, tier, task_id, approved_by, approval_time_s,
                json.dumps(meta),
            ),
        )
        self._conn.commit()

        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=now,
            action_type=action_type,
            actor=actor,
            target=target_truncated,
            result=result,
            risk_level=risk_level,
            tier=tier,
            task_id=task_id,
            approved_by=approved_by,
            approval_time_s=approval_time_s,
            metadata=meta,
        )

        log_func = logger.warning if tier >= 3 else logger.info
        log_func(
            "Audit: [tier %d/%s] %s → %s (actor=%s, task=%s)",
            tier, risk_level, action_type, result, actor, task_id[:8] if task_id else "none",
        )

        return entry

    # -------------------------------------------------------------------
    # Update an existing entry (for approval flow)
    # -------------------------------------------------------------------

    def update_result(
        self,
        entry_id: str,
        result: str,
        approved_by: str | None = None,
        approval_time_s: float | None = None,
    ) -> AuditEntry | None:
        """Update the result of an existing audit entry.

        Used when a Tier 2 action transitions from 'pending' to
        'approved', 'rejected', or 'timeout'.

        Args:
            entry_id:         The entry to update.
            result:           New result value.
            approved_by:      Who approved (if applicable).
            approval_time_s:  Time taken to approve (if applicable).

        Returns:
            The updated AuditEntry, or None if not found.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE audit_log
            SET result = ?, approved_by = ?, approval_time_s = ?
            WHERE entry_id = ?
            """,
            (result, approved_by, approval_time_s, entry_id),
        )
        self._conn.commit()

        if cur.rowcount == 0:
            return None

        return self.get_entry(entry_id)

    # -------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------

    def get_entry(self, entry_id: str) -> AuditEntry | None:
        """Return a single audit entry by ID, or None."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM audit_log WHERE entry_id = ?", (entry_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return AuditEntry.from_row(row)

    def get_recent(self, limit: int = 100) -> list[AuditEntry]:
        """Return the most recent audit entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of AuditEntry objects, newest first.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [AuditEntry.from_row(row) for row in cur.fetchall()]

    def filter_entries(
        self,
        risk_level: str | None = None,
        actor: str | None = None,
        result: str | None = None,
        tier: int | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Return audit entries matching the given filters.

        Args:
            risk_level: Filter by risk level ('low', 'medium', 'critical').
            actor:      Filter by actor name.
            result:     Filter by result ('executed', 'approved', etc.).
            tier:       Filter by permission tier (1, 2, or 3).
            limit:      Maximum entries to return.

        Returns:
            List of AuditEntry objects, newest first.
        """
        conditions = []
        params: list = []

        if risk_level:
            conditions.append("risk_level = ?")
            params.append(risk_level)
        if actor:
            conditions.append("actor = ?")
            params.append(actor)
        if result:
            conditions.append("result = ?")
            params.append(result)
        if tier is not None:
            conditions.append("tier = ?")
            params.append(tier)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        params.append(limit)

        cur = self._conn.cursor()
        cur.execute(
            f"SELECT * FROM audit_log {where_clause} ORDER BY timestamp DESC LIMIT ?",
            params,
        )
        return [AuditEntry.from_row(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------
    # Dashboard helpers
    # -------------------------------------------------------------------

    def to_broadcast_list(self) -> list[dict]:
        """Return recent entries as JSON-serializable dicts for the dashboard."""
        return [e.to_dict() for e in self.get_recent(limit=200)]

    def get_status(self) -> dict:
        """Return a snapshot of audit log state — counts by result and risk."""
        cur = self._conn.cursor()

        cur.execute(
            "SELECT result, COUNT(*) as cnt FROM audit_log GROUP BY result"
        )
        by_result = {row["result"]: row["cnt"] for row in cur.fetchall()}

        cur.execute(
            "SELECT risk_level, COUNT(*) as cnt FROM audit_log GROUP BY risk_level"
        )
        by_risk = {row["risk_level"]: row["cnt"] for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) FROM audit_log")
        total = cur.fetchone()[0]

        # Distinct actors for filter dropdown
        cur.execute("SELECT DISTINCT actor FROM audit_log ORDER BY actor")
        actors = [row["actor"] for row in cur.fetchall()]

        return {
            "total": total,
            "by_result": by_result,
            "by_risk": by_risk,
            "actors": actors,
            "db_path": self._db_path,
        }

    # -------------------------------------------------------------------
    # Close
    # -------------------------------------------------------------------

    def close(self):
        """Close the SQLite connection. Call on shutdown."""
        if self._conn:
            self._conn.close()
            logger.info("SecurityAuditLog connection closed")


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    try:
        audit = SecurityAuditLog(db_path=tmp.name)

        # Log some actions
        e1 = audit.log_action(
            action_type="query_model",
            actor="orchestrator",
            target="what is a sportster",
            result="executed",
            risk_level="low",
            tier=1,
            task_id="task-001",
        )
        print(f"Logged: {e1.short_id} — {e1.action_type} → {e1.result} (tier {e1.tier})")

        e2 = audit.log_action(
            action_type="publish_content",
            actor="orchestrator",
            target="publish video to youtube",
            result="pending",
            risk_level="medium",
            tier=2,
            task_id="task-002",
        )
        print(f"Logged: {e2.short_id} — {e2.action_type} → {e2.result} (tier {e2.tier})")

        # Update approval
        e2_updated = audit.update_result(
            entry_id=e2.entry_id,
            result="approved",
            approved_by="dashboard",
            approval_time_s=3.5,
        )
        print(f"Updated: {e2_updated.short_id} → {e2_updated.result} (by {e2_updated.approved_by})")

        e3 = audit.log_action(
            action_type="modify_security",
            actor="orchestrator",
            target="disable security checks",
            result="blocked",
            risk_level="critical",
            tier=3,
            task_id="task-003",
        )
        print(f"Logged: {e3.short_id} — {e3.action_type} → {e3.result} (tier {e3.tier})")

        # Query
        print(f"\nRecent entries ({len(audit.get_recent())}):")
        for e in audit.get_recent():
            print(f"  [{e.risk_level}] {e.action_type} → {e.result} (tier {e.tier})")

        # Filter
        critical = audit.filter_entries(risk_level="critical")
        print(f"\nCritical entries: {len(critical)}")

        # Status
        status = audit.get_status()
        print(f"\nStatus: {status}")

        audit.close()

    finally:
        os.unlink(tmp.name)
        print(f"\nCleaned up temp file: {tmp.name}")
