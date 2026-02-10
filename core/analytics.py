"""
CAM Analytics — SQLite-Backed Task & Model Call History

Persists task completion records and model call records so analytics
survive server restarts. Provides aggregate metrics for the dashboard:
success rates, average completion times, tasks per agent, and model costs.

CLAUDE.md: "Cost tracking on every API call" + "Full action logging"

Usage:
    from core.analytics import Analytics

    analytics = Analytics(db_path="data/analytics.db")
    analytics.record_task(task)
    analytics.record_model_call(model="kimi-k2.5", backend="moonshot",
                                tokens=1200, latency_ms=450.0,
                                cost_usd=0.0012, task_short_id="a1b2c3d4")
    summary = analytics.get_summary()
    analytics.close()
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.task import Task


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("cam.analytics")


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class Analytics:
    """SQLite-backed analytics for task history and model cost tracking.

    Follows the EventLogger/HealthMonitor pattern — standalone class,
    receives events via method calls from server.py, no global state.

    Args:
        db_path: Path to the SQLite database file. Parent dirs created
                 automatically if they don't exist.
    """

    def __init__(self, db_path: str = "data/analytics.db"):
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_file), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        logger.info("Analytics initialized (db=%s)", db_file)

    # -------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        cur = self._conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS task_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                short_id TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                complexity TEXT NOT NULL,
                source TEXT NOT NULL,
                assigned_agent TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                recorded_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_call_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                backend TEXT NOT NULL,
                tokens INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                cost_usd REAL NOT NULL,
                task_short_id TEXT,
                recorded_at TEXT NOT NULL
            )
        """)

        # Indexes for common query patterns
        cur.execute("CREATE INDEX IF NOT EXISTS idx_task_status ON task_records(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_task_agent ON task_records(assigned_agent)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model_backend ON model_call_records(backend)")

        self._conn.commit()

    # -------------------------------------------------------------------
    # Recording events
    # -------------------------------------------------------------------

    def record_task(self, task: Task):
        """Record a completed or failed task in the database.

        Computes duration_seconds from created_at → completed_at.
        Silently catches errors so analytics never break the main loop.

        Args:
            task: The Task object (must have completed_at set).
        """
        try:
            now = datetime.now(timezone.utc)
            completed_at = task.completed_at or now

            # Duration in seconds
            duration = (completed_at - task.created_at).total_seconds()

            self._conn.execute(
                """INSERT INTO task_records
                   (task_id, short_id, description, status, complexity, source,
                    assigned_agent, created_at, completed_at, duration_seconds, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task.task_id,
                    task.short_id,
                    task.description,
                    task.status.value,
                    task.complexity.value,
                    task.source,
                    task.assigned_agent,
                    task.created_at.isoformat(),
                    completed_at.isoformat(),
                    duration,
                    now.isoformat(),
                ),
            )
            self._conn.commit()
            logger.info("Recorded task %s (%s)", task.short_id, task.status.value)

        except Exception as e:
            logger.error("Failed to record task %s: %s", task.short_id, e)

    def record_model_call(
        self,
        model: str,
        backend: str,
        tokens: int,
        latency_ms: float,
        cost_usd: float,
        task_short_id: str | None = None,
    ):
        """Record a model router call in the database.

        Args:
            model:          Model name (e.g. "kimi-k2.5", "claude-sonnet").
            backend:        Backend provider (e.g. "moonshot", "anthropic", "ollama").
            tokens:         Total tokens used.
            latency_ms:     Round-trip latency in milliseconds.
            cost_usd:       Estimated cost in USD.
            task_short_id:  Short ID of the associated task, if any.
        """
        try:
            now = datetime.now(timezone.utc)
            self._conn.execute(
                """INSERT INTO model_call_records
                   (model, backend, tokens, latency_ms, cost_usd, task_short_id, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (model, backend, tokens, latency_ms, cost_usd, task_short_id, now.isoformat()),
            )
            self._conn.commit()

        except Exception as e:
            logger.error("Failed to record model call: %s", e)

    # -------------------------------------------------------------------
    # Aggregate queries
    # -------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Compute aggregate analytics for the dashboard.

        Returns a dict with:
            total_tasks, completed_tasks, failed_tasks, success_rate (0-100)
            avg_completion_seconds (completed only)
            tasks_per_agent (agent_name -> count)
            total_model_cost_usd, total_model_calls, total_model_tokens
            cost_by_backend (backend -> {calls, tokens, cost_usd})
        """
        cur = self._conn.cursor()

        # --- Task aggregates ---
        cur.execute("SELECT COUNT(*) FROM task_records")
        total_tasks = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM task_records WHERE status = 'completed'")
        completed_tasks = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM task_records WHERE status = 'failed'")
        failed_tasks = cur.fetchone()[0]

        success_rate = 0.0
        if total_tasks > 0:
            success_rate = round((completed_tasks / total_tasks) * 100, 1)

        cur.execute("SELECT AVG(duration_seconds) FROM task_records WHERE status = 'completed'")
        row = cur.fetchone()
        avg_completion = round(row[0], 1) if row[0] is not None else 0.0

        # Tasks per agent (NULL → "(model fallback)")
        cur.execute("""
            SELECT COALESCE(assigned_agent, '(model fallback)') AS agent, COUNT(*) AS cnt
            FROM task_records
            GROUP BY agent
            ORDER BY cnt DESC
        """)
        tasks_per_agent = {row["agent"]: row["cnt"] for row in cur.fetchall()}

        # --- Model call aggregates ---
        cur.execute("SELECT COUNT(*), COALESCE(SUM(tokens), 0), COALESCE(SUM(cost_usd), 0) FROM model_call_records")
        model_row = cur.fetchone()
        total_model_calls = model_row[0]
        total_model_tokens = model_row[1]
        total_model_cost = round(model_row[2], 6)

        # Cost breakdown by backend
        cur.execute("""
            SELECT backend, COUNT(*) AS calls, SUM(tokens) AS tokens, SUM(cost_usd) AS cost
            FROM model_call_records
            GROUP BY backend
        """)
        cost_by_backend = {}
        for row in cur.fetchall():
            cost_by_backend[row["backend"]] = {
                "calls": row["calls"],
                "tokens": row["tokens"],
                "cost_usd": round(row["cost"], 6),
            }

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": success_rate,
            "avg_completion_seconds": avg_completion,
            "tasks_per_agent": tasks_per_agent,
            "total_model_cost_usd": total_model_cost,
            "total_model_calls": total_model_calls,
            "total_model_tokens": total_model_tokens,
            "cost_by_backend": cost_by_backend,
        }

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------

    def close(self):
        """Close the SQLite connection."""
        try:
            self._conn.close()
            logger.info("Analytics database connection closed")
        except Exception:
            pass
