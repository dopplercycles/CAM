"""
Offline Mode and Graceful Degradation for CAM.

Monitors internet and Ollama connectivity, queues operations when offline,
caches dashboard data to local JSON, and processes queued items when
connectivity returns.

Critical for a mobile business — George needs to log service work and
access customer data even when cell service is spotty.  All local
operations (service records, CRM, inventory, scheduling) continue
normally regardless of connectivity state.

States:
  online   — internet + Ollama both available
  degraded — one of the two is down (partial functionality)
  offline  — no internet connectivity
"""

import asyncio
import json
import logging
import socket
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("cam.offline")


# ── Constants ───────────────────────────────────────────────────────

STATE_ONLINE = "online"
STATE_DEGRADED = "degraded"
STATE_OFFLINE = "offline"

OP_TYPES = (
    "telegram_message",
    "task",
    "scout_scan",
    "api_call",
    "notification",
)

# Connectivity check targets
DNS_CHECK_HOST = "dns.google"
DNS_CHECK_PORT = 443
DNS_CHECK_TIMEOUT = 5

OLLAMA_DEFAULT_URL = "http://localhost:11434"

# Monitoring intervals
DEFAULT_CHECK_INTERVAL = 30       # seconds when online
FAST_CHECK_INTERVAL = 10          # seconds when degraded/offline
CACHE_INTERVAL_CHECKS = 3         # cache dashboard data every N checks


# ── Dataclass ───────────────────────────────────────────────────────

@dataclass
class QueuedOperation:
    """An operation queued for when connectivity returns."""

    queue_id: str
    op_type: str
    payload: dict
    status: str       # pending, processing, completed, failed
    error: str
    created_at: str
    processed_at: str

    @property
    def short_id(self) -> str:
        return self.queue_id[:8]

    def to_dict(self) -> dict:
        return {
            "queue_id": self.queue_id,
            "short_id": self.short_id,
            "op_type": self.op_type,
            "payload": self.payload,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "processed_at": self.processed_at,
        }

    @staticmethod
    def from_row(row) -> "QueuedOperation":
        payload = row["payload"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                payload = {}
        return QueuedOperation(
            queue_id=row["queue_id"],
            op_type=row["op_type"],
            payload=payload,
            status=row["status"],
            error=row["error"],
            created_at=row["created_at"],
            processed_at=row["processed_at"],
        )


# ── OfflineManager ──────────────────────────────────────────────────

class OfflineManager:
    """Monitors connectivity and manages graceful degradation.

    - Periodic checks for internet (TCP to dns.google:443) and
      Ollama (/api/tags endpoint)
    - Queues operations that need connectivity in SQLite
    - Caches essential dashboard data to local JSON
    - Processes queued items when connectivity returns
    - Logs all state transitions for diagnostics
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        db_path: str = "data/offline.db",
        ollama_url: str = OLLAMA_DEFAULT_URL,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        on_change: Callable | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._ollama_url = ollama_url.rstrip("/")
        self._check_interval = check_interval
        self._on_change = on_change

        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        # Current state
        self._state = STATE_ONLINE
        self._internet_available = True
        self._ollama_available = True
        self._last_check = ""
        self._state_since = self._now()
        self._check_count = 0
        self._transitions: list[dict] = []

        # Background loop control
        self._running = False

        # Operation handlers (set via set_handlers())
        self._handlers: dict[str, Callable] = {}

        # Dashboard cache callback (set via set_cache_provider())
        self._cache_provider: Callable | None = None

        logger.info(
            "OfflineManager initialized (cache=%s, db=%s, ollama=%s, interval=%ds)",
            self.cache_dir, db_path, self._ollama_url, check_interval,
        )

    # ── Schema ──────────────────────────────────────────────────────

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS offline_queue (
                queue_id     TEXT PRIMARY KEY,
                op_type      TEXT NOT NULL,
                payload      TEXT NOT NULL DEFAULT '{}',
                status       TEXT NOT NULL DEFAULT 'pending',
                error        TEXT DEFAULT '',
                created_at   TEXT NOT NULL,
                processed_at TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_oq_status
                ON offline_queue(status);
            CREATE INDEX IF NOT EXISTS idx_oq_created
                ON offline_queue(created_at);

            CREATE TABLE IF NOT EXISTS state_log (
                log_id      TEXT PRIMARY KEY,
                old_state   TEXT NOT NULL,
                new_state   TEXT NOT NULL,
                internet    INTEGER NOT NULL,
                ollama      INTEGER NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sl_created
                ON state_log(created_at);
        """)
        self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _fire_change(self):
        if self._on_change:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._on_change())
            except RuntimeError:
                pass

    # ── Handler & cache registration ────────────────────────────────

    def set_handlers(self, **handlers):
        """Register async handlers for operation types when reconnecting.

        Example:
            set_handlers(
                telegram_message=send_queued_telegram,
                scout_scan=run_queued_scan,
            )
        """
        self._handlers = handlers

    def set_cache_provider(self, provider: Callable):
        """Set a callable that returns a dict of dashboard data to cache.

        The provider should return something like:
            {"agent_status": {...}, "task_status": {...}, ...}
        """
        self._cache_provider = provider

    # ── Connectivity checks ─────────────────────────────────────────

    def check_internet(self) -> bool:
        """Test internet by attempting TCP connect to dns.google:443."""
        try:
            sock = socket.create_connection(
                (DNS_CHECK_HOST, DNS_CHECK_PORT),
                timeout=DNS_CHECK_TIMEOUT,
            )
            sock.close()
            return True
        except (OSError, socket.timeout):
            return False

    def check_ollama(self) -> bool:
        """Test Ollama availability via /api/tags."""
        try:
            req = Request(f"{self._ollama_url}/api/tags", method="GET")
            with urlopen(req, timeout=DNS_CHECK_TIMEOUT) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def check_connectivity(self) -> dict:
        """Run full connectivity check.

        Returns dict with state, internet, ollama, changed.
        """
        loop = asyncio.get_running_loop()
        internet = await loop.run_in_executor(None, self.check_internet)
        ollama = await loop.run_in_executor(None, self.check_ollama)

        old_state = self._state
        self._internet_available = internet
        self._ollama_available = ollama
        self._last_check = self._now()
        self._check_count += 1

        # Determine new state
        if internet and ollama:
            new_state = STATE_ONLINE
        elif internet or ollama:
            new_state = STATE_DEGRADED
        else:
            new_state = STATE_OFFLINE

        changed = new_state != old_state

        if changed:
            self._state = new_state
            self._state_since = self._now()
            transition = {
                "old_state": old_state,
                "new_state": new_state,
                "internet": internet,
                "ollama": ollama,
                "timestamp": self._now(),
            }
            self._transitions.append(transition)
            if len(self._transitions) > 50:
                self._transitions = self._transitions[-50:]

            # Persist transition
            self._conn.execute(
                "INSERT INTO state_log VALUES (?,?,?,?,?,?)",
                (uuid.uuid4().hex, old_state, new_state,
                 int(internet), int(ollama), self._now()),
            )
            self._conn.commit()

            logger.info(
                "Connectivity: %s -> %s (internet=%s, ollama=%s)",
                old_state, new_state, internet, ollama,
            )

            # Process queue when coming back online
            if new_state == STATE_ONLINE and old_state != STATE_ONLINE:
                await self._process_queue()

            self._fire_change()

        return {
            "state": self._state,
            "internet": internet,
            "ollama": ollama,
            "changed": changed,
        }

    # ── Monitoring loop ─────────────────────────────────────────────

    async def start_monitoring(self):
        """Background loop: check connectivity + cache dashboard data."""
        self._running = True
        await self.check_connectivity()
        logger.info(
            "Offline monitoring loop started (interval=%ds)",
            self._check_interval,
        )

        checks_since_cache = 0
        while self._running:
            interval = (
                FAST_CHECK_INTERVAL
                if self._state != STATE_ONLINE
                else self._check_interval
            )
            await asyncio.sleep(interval)
            if not self._running:
                break
            try:
                await self.check_connectivity()
            except Exception as e:
                logger.warning("Connectivity check error: %s", e)

            # Periodically cache dashboard data
            checks_since_cache += 1
            if checks_since_cache >= CACHE_INTERVAL_CHECKS:
                checks_since_cache = 0
                self._snapshot_dashboard_cache()

    def stop_monitoring(self):
        """Signal the monitoring loop to stop."""
        self._running = False

    def _snapshot_dashboard_cache(self):
        """Cache current dashboard data to disk via the provider callback."""
        if not self._cache_provider:
            return
        try:
            sections = self._cache_provider()
            if isinstance(sections, dict):
                for key, data in sections.items():
                    self.cache_data(key, data)
        except Exception as e:
            logger.debug("Dashboard cache snapshot failed: %s", e)

    # ── Operation queue ─────────────────────────────────────────────

    def queue_operation(
        self, op_type: str, payload: dict,
    ) -> QueuedOperation:
        """Queue an operation for retry when connectivity returns."""
        queue_id = uuid.uuid4().hex
        now = self._now()
        self._conn.execute(
            "INSERT INTO offline_queue VALUES (?,?,?,?,?,?,?)",
            (queue_id, op_type, json.dumps(payload, default=str),
             "pending", "", now, ""),
        )
        self._conn.commit()
        logger.info("Queued %s operation (%s)", op_type, queue_id[:8])
        self._fire_change()
        return QueuedOperation(
            queue_id=queue_id, op_type=op_type, payload=payload,
            status="pending", error="", created_at=now, processed_at="",
        )

    def list_queue(
        self, status: str = "", limit: int = 100,
    ) -> list[QueuedOperation]:
        """List queued operations, optionally filtered by status."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM offline_queue WHERE status = ? "
                "ORDER BY created_at ASC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM offline_queue "
                "ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [QueuedOperation.from_row(r) for r in rows]

    def queue_depth(self) -> dict:
        """Count of pending operations by type."""
        rows = self._conn.execute(
            "SELECT op_type, COUNT(*) as cnt FROM offline_queue "
            "WHERE status = 'pending' GROUP BY op_type",
        ).fetchall()
        by_type = {r["op_type"]: r["cnt"] for r in rows}
        total = sum(by_type.values())
        return {"total": total, "by_type": by_type}

    async def _process_queue(self):
        """Process pending queued operations when connectivity returns."""
        pending = self.list_queue(status="pending")
        if not pending:
            return

        logger.info("Processing %d queued operations", len(pending))
        processed = 0
        for op in pending:
            try:
                self._conn.execute(
                    "UPDATE offline_queue SET status = 'processing' "
                    "WHERE queue_id = ?",
                    (op.queue_id,),
                )
                self._conn.commit()

                await self._execute_operation(op)

                self._conn.execute(
                    "UPDATE offline_queue SET status = 'completed', "
                    "processed_at = ? WHERE queue_id = ?",
                    (self._now(), op.queue_id),
                )
                self._conn.commit()
                processed += 1
            except Exception as e:
                logger.warning(
                    "Failed to process queued op %s: %s", op.short_id, e,
                )
                self._conn.execute(
                    "UPDATE offline_queue SET status = 'failed', error = ? "
                    "WHERE queue_id = ?",
                    (str(e)[:500], op.queue_id),
                )
                self._conn.commit()

        logger.info(
            "Processed %d/%d queued operations", processed, len(pending),
        )
        self._fire_change()

    async def _execute_operation(self, op: QueuedOperation):
        """Execute a single queued operation via registered handler."""
        handler = self._handlers.get(op.op_type)
        if handler:
            await handler(op.payload)
        else:
            logger.debug(
                "No handler for op_type '%s', marking completed",
                op.op_type,
            )

    async def force_process_queue(self) -> int:
        """Manually trigger queue processing. Returns count processed."""
        pending = self.list_queue(status="pending")
        if not pending:
            return 0
        await self._process_queue()
        return len(pending)

    def clear_queue(self, status: str = "") -> int:
        """Clear queued operations. If status given, only clear that status."""
        if status:
            cur = self._conn.execute(
                "DELETE FROM offline_queue WHERE status = ?", (status,),
            )
        else:
            cur = self._conn.execute("DELETE FROM offline_queue")
        self._conn.commit()
        self._fire_change()
        return cur.rowcount

    def remove_operation(self, queue_id: str) -> bool:
        """Remove a single queued operation."""
        cur = self._conn.execute(
            "DELETE FROM offline_queue WHERE queue_id = ?", (queue_id,),
        )
        self._conn.commit()
        if cur.rowcount:
            self._fire_change()
        return cur.rowcount > 0

    # ── Dashboard cache ─────────────────────────────────────────────

    def cache_data(self, key: str, data: Any):
        """Cache a dashboard data section to local JSON file."""
        path = self.cache_dir / f"{key}.json"
        try:
            path.write_text(
                json.dumps(data, default=str), encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to cache '%s': %s", key, e)

    def get_cached_data(self, key: str) -> Any | None:
        """Retrieve cached dashboard data section."""
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_cached_sections(self) -> list[str]:
        """Return names of all cached data sections."""
        return sorted(p.stem for p in self.cache_dir.glob("*.json"))

    def clear_cache(self) -> int:
        """Remove all cached JSON files."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count

    # ── State history ───────────────────────────────────────────────

    def get_state_history(self, limit: int = 20) -> list[dict]:
        """Return recent connectivity state transitions."""
        rows = self._conn.execute(
            "SELECT * FROM state_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "old_state": r["old_state"],
                "new_state": r["new_state"],
                "internet": bool(r["internet"]),
                "ollama": bool(r["ollama"]),
                "timestamp": r["created_at"],
            }
            for r in rows
        ]

    # ── Properties ──────────────────────────────────────────────────

    @property
    def is_online(self) -> bool:
        return self._state == STATE_ONLINE

    @property
    def is_offline(self) -> bool:
        return self._state == STATE_OFFLINE

    @property
    def internet_available(self) -> bool:
        return self._internet_available

    @property
    def ollama_available(self) -> bool:
        return self._ollama_available

    @property
    def state(self) -> str:
        return self._state

    # ── Status / broadcast ──────────────────────────────────────────

    def get_status(self) -> dict:
        """Dashboard status summary."""
        depth = self.queue_depth()
        return {
            "state": self._state,
            "internet": self._internet_available,
            "ollama": self._ollama_available,
            "last_check": self._last_check,
            "state_since": self._state_since,
            "check_count": self._check_count,
            "check_interval": (
                FAST_CHECK_INTERVAL
                if self._state != STATE_ONLINE
                else self._check_interval
            ),
            "queue_depth": depth["total"],
            "queue_by_type": depth["by_type"],
            "cached_sections": self.list_cached_sections(),
        }

    def to_broadcast_dict(self) -> dict:
        """Full state for WS broadcast."""
        return {
            "status": self.get_status(),
            "queue": [op.to_dict() for op in self.list_queue(limit=50)],
            "state_history": self.get_state_history(limit=10),
        }

    # ── Cleanup ─────────────────────────────────────────────────────

    def close(self):
        self.stop_monitoring()
        self._conn.close()
        logger.info("OfflineManager closed")
