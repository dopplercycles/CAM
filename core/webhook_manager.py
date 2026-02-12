"""
CAM Webhook Manager

Handles outbound event-triggered webhooks (HMAC-SHA256 signatures, retry with
exponential backoff) and inbound webhooks that create tasks.  Persists state to
a dedicated SQLite database (data/webhooks.db).

Usage:
    from core.webhook_manager import WebhookManager

    wm = WebhookManager(db_path="data/webhooks.db", task_queue=tq,
                        event_logger=el, on_change=broadcast_fn)
    await wm.evaluate_event(event_dict)          # outbound delivery
    result = await wm.process_inbound(payload, ip, sig)  # inbound
"""

import asyncio
import hashlib
import hmac
import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

import httpx

logger = logging.getLogger("cam.webhooks")

# ---------------------------------------------------------------------------
# Defaults (overridable via config)
# ---------------------------------------------------------------------------
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE = 10       # seconds
DEFAULT_RETRY_MAX = 3600      # 1 hour cap
DEFAULT_RETRY_CHECK = 15      # seconds between retry sweeps
DEFAULT_MAX_HISTORY = 500
CAM_VERSION = "0.1.0"


class WebhookManager:
    """Manages outbound and inbound webhooks with SQLite persistence."""

    def __init__(
        self,
        db_path: str = "data/webhooks.db",
        task_queue: Any = None,
        event_logger: Any = None,
        on_change: Callable[[], Awaitable[None]] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_seconds: int = DEFAULT_RETRY_BASE,
        retry_max_seconds: int = DEFAULT_RETRY_MAX,
        retry_check_interval: int = DEFAULT_RETRY_CHECK,
        max_delivery_history: int = DEFAULT_MAX_HISTORY,
        inbound_secret: str = "",
    ):
        self.db_path = db_path
        self.task_queue = task_queue
        self.event_logger = event_logger
        self.on_change = on_change
        self.max_retries = max_retries
        self.retry_base = retry_base_seconds
        self.retry_max = retry_max_seconds
        self.retry_check_interval = retry_check_interval
        self.max_delivery_history = max_delivery_history
        self.inbound_secret = inbound_secret

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        logger.info("WebhookManager initialised — db=%s", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self):
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS webhook_endpoints (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint_id     TEXT NOT NULL,
                name            TEXT NOT NULL,
                direction       TEXT NOT NULL DEFAULT 'outbound',
                url             TEXT NOT NULL DEFAULT '',
                secret          TEXT NOT NULL DEFAULT '',
                event_filters   TEXT NOT NULL DEFAULT '[]',
                severity_filter TEXT NOT NULL DEFAULT 'all',
                enabled         INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_endpoint_id
                ON webhook_endpoints(endpoint_id);
            CREATE INDEX IF NOT EXISTS idx_direction
                ON webhook_endpoints(direction);
            CREATE INDEX IF NOT EXISTS idx_enabled
                ON webhook_endpoints(enabled);

            CREATE TABLE IF NOT EXISTS webhook_deliveries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                delivery_id     TEXT NOT NULL,
                endpoint_id     TEXT NOT NULL,
                event_category  TEXT NOT NULL DEFAULT '',
                event_message   TEXT NOT NULL DEFAULT '',
                payload         TEXT NOT NULL DEFAULT '{}',
                status          TEXT NOT NULL DEFAULT 'pending',
                http_status     INTEGER,
                response_body   TEXT,
                attempt_count   INTEGER NOT NULL DEFAULT 0,
                max_attempts    INTEGER NOT NULL DEFAULT 5,
                next_retry_at   REAL,
                error           TEXT,
                created_at      TEXT NOT NULL,
                completed_at    TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_del_status
                ON webhook_deliveries(status);
            CREATE INDEX IF NOT EXISTS idx_del_endpoint
                ON webhook_deliveries(endpoint_id);
            CREATE INDEX IF NOT EXISTS idx_del_created
                ON webhook_deliveries(created_at);

            CREATE TABLE IF NOT EXISTS webhook_inbound_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id          TEXT NOT NULL,
                source_ip       TEXT NOT NULL DEFAULT '',
                endpoint_id     TEXT NOT NULL DEFAULT '',
                payload         TEXT NOT NULL DEFAULT '{}',
                task_id         TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'received',
                error           TEXT,
                created_at      TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Endpoint CRUD
    # ------------------------------------------------------------------

    def register_endpoint(
        self,
        name: str,
        url: str = "",
        direction: str = "outbound",
        secret: str = "",
        event_filters: list[str] | None = None,
        severity_filter: str = "all",
        enabled: bool = True,
    ) -> dict:
        """Create and persist a new webhook endpoint."""
        endpoint_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        filters_json = json.dumps(event_filters or [])
        self._conn.execute(
            """INSERT INTO webhook_endpoints
               (endpoint_id, name, direction, url, secret,
                event_filters, severity_filter, enabled, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (endpoint_id, name, direction, url, secret,
             filters_json, severity_filter, 1 if enabled else 0, now, now),
        )
        self._conn.commit()
        logger.info("Registered webhook endpoint %s (%s, %s)", name, direction, endpoint_id[:8])
        return self._endpoint_dict(endpoint_id)

    def update_endpoint(self, endpoint_id: str, **kwargs) -> dict | None:
        """Update mutable fields on an existing endpoint."""
        row = self._conn.execute(
            "SELECT * FROM webhook_endpoints WHERE endpoint_id = ?",
            (endpoint_id,),
        ).fetchone()
        if not row:
            return None

        allowed = {"name", "url", "secret", "event_filters",
                   "severity_filter", "enabled", "direction"}
        sets, vals = [], []
        for key, val in kwargs.items():
            if key not in allowed:
                continue
            if key == "event_filters":
                val = json.dumps(val if val is not None else [])
            if key == "enabled":
                val = 1 if val else 0
            sets.append(f"{key} = ?")
            vals.append(val)

        if not sets:
            return self._endpoint_dict(endpoint_id)

        sets.append("updated_at = ?")
        vals.append(datetime.now(timezone.utc).isoformat())
        vals.append(endpoint_id)
        self._conn.execute(
            f"UPDATE webhook_endpoints SET {', '.join(sets)} WHERE endpoint_id = ?",
            vals,
        )
        self._conn.commit()
        logger.info("Updated webhook endpoint %s", endpoint_id[:8])
        return self._endpoint_dict(endpoint_id)

    def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete an endpoint and its delivery history."""
        cur = self._conn.execute(
            "DELETE FROM webhook_endpoints WHERE endpoint_id = ?",
            (endpoint_id,),
        )
        self._conn.execute(
            "DELETE FROM webhook_deliveries WHERE endpoint_id = ?",
            (endpoint_id,),
        )
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Deleted webhook endpoint %s", endpoint_id[:8])
        return deleted

    def get_endpoint(self, endpoint_id: str) -> dict | None:
        """Look up a single endpoint by ID."""
        return self._endpoint_dict(endpoint_id)

    def list_endpoints(self, direction: str | None = None) -> list[dict]:
        """Return all endpoints, optionally filtered by direction."""
        if direction:
            rows = self._conn.execute(
                "SELECT * FROM webhook_endpoints WHERE direction = ? ORDER BY created_at DESC",
                (direction,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM webhook_endpoints ORDER BY created_at DESC",
            ).fetchall()
        return [self._row_to_endpoint(r) for r in rows]

    # ------------------------------------------------------------------
    # Outbound delivery
    # ------------------------------------------------------------------

    async def evaluate_event(self, event_dict: dict) -> None:
        """Match an event against all enabled outbound endpoints and fire deliveries."""
        category = event_dict.get("category", "")
        severity = event_dict.get("severity", "info")

        endpoints = self._conn.execute(
            "SELECT * FROM webhook_endpoints WHERE direction = 'outbound' AND enabled = 1",
        ).fetchall()

        for ep in endpoints:
            ep_dict = self._row_to_endpoint(ep)
            # Check event category filter
            filters = ep_dict.get("event_filters", [])
            if filters and category not in filters:
                continue
            # Check severity filter
            sev_filter = ep_dict.get("severity_filter", "all")
            if sev_filter != "all" and severity != sev_filter:
                continue

            await self._deliver_to_endpoint(ep_dict["endpoint_id"], event_dict)

    async def _deliver_to_endpoint(self, endpoint_id: str, event_dict: dict) -> None:
        """Create a delivery record and attempt HTTP POST."""
        ep = self._endpoint_dict(endpoint_id)
        if not ep:
            return

        delivery_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        payload = {
            "event": {
                "timestamp": event_dict.get("timestamp", now),
                "severity": event_dict.get("severity", "info"),
                "category": event_dict.get("category", ""),
                "message": event_dict.get("message", ""),
                "details": event_dict.get("details", {}),
            },
            "webhook": {
                "delivery_id": delivery_id,
                "endpoint_name": ep["name"],
                "cam_version": CAM_VERSION,
            },
        }

        self._conn.execute(
            """INSERT INTO webhook_deliveries
               (delivery_id, endpoint_id, event_category, event_message,
                payload, status, attempt_count, max_attempts, created_at)
               VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?)""",
            (delivery_id, endpoint_id, event_dict.get("category", ""),
             event_dict.get("message", ""), json.dumps(payload),
             self.max_retries, now),
        )
        self._conn.commit()

        await self._deliver(delivery_id, ep, payload)

    async def _deliver(self, delivery_id: str, endpoint: dict, payload: dict) -> None:
        """HTTP POST with HMAC signature. Updates delivery status."""
        url = endpoint.get("url", "")
        if not url:
            self._update_delivery(delivery_id, status="failed", error="No URL configured")
            return

        payload_bytes = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Event": payload.get("event", {}).get("category", ""),
            "X-Webhook-Delivery-ID": delivery_id,
            "User-Agent": "CAM-Webhook/1.0",
        }
        secret = endpoint.get("secret", "")
        if secret:
            sig = self._compute_signature(secret, payload_bytes)
            headers["X-Webhook-Signature"] = sig

        # Increment attempt count
        row = self._conn.execute(
            "SELECT attempt_count FROM webhook_deliveries WHERE delivery_id = ?",
            (delivery_id,),
        ).fetchone()
        attempt = (row["attempt_count"] if row else 0) + 1
        self._conn.execute(
            "UPDATE webhook_deliveries SET attempt_count = ?, status = 'pending' WHERE delivery_id = ?",
            (attempt, delivery_id),
        )
        self._conn.commit()

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(
                connect=10.0, read=30.0, write=10.0, pool=10.0
            )) as client:
                resp = await client.post(url, content=payload_bytes, headers=headers)

            resp_body = resp.text[:500]  # truncate response
            if 200 <= resp.status_code < 300:
                self._update_delivery(
                    delivery_id,
                    status="success",
                    http_status=resp.status_code,
                    response_body=resp_body,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                logger.info("Webhook delivered %s → %s (%d)",
                            delivery_id[:8], url[:60], resp.status_code)
            else:
                error_msg = f"HTTP {resp.status_code}: {resp_body[:200]}"
                if attempt >= self.max_retries:
                    self._update_delivery(
                        delivery_id, status="failed",
                        http_status=resp.status_code,
                        response_body=resp_body, error=error_msg,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )
                    logger.warning("Webhook failed permanently %s → %s",
                                   delivery_id[:8], error_msg)
                else:
                    self._schedule_retry(delivery_id, attempt)
                    self._update_delivery(
                        delivery_id, status="retrying",
                        http_status=resp.status_code,
                        response_body=resp_body, error=error_msg,
                    )
                    logger.info("Webhook retry scheduled %s attempt %d/%d",
                                delivery_id[:8], attempt, self.max_retries)

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            if attempt >= self.max_retries:
                self._update_delivery(
                    delivery_id, status="failed", error=error_msg,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                logger.warning("Webhook failed permanently %s → %s",
                               delivery_id[:8], error_msg)
            else:
                self._schedule_retry(delivery_id, attempt)
                self._update_delivery(
                    delivery_id, status="retrying", error=error_msg,
                )
                logger.info("Webhook retry scheduled %s attempt %d/%d",
                            delivery_id[:8], attempt, self.max_retries)

        # Trim old deliveries
        self._trim_deliveries()

        # Broadcast update
        if self.on_change:
            try:
                await self.on_change()
            except Exception:
                pass

    def _schedule_retry(self, delivery_id: str, attempt: int) -> None:
        """Set next_retry_at using exponential backoff."""
        delay = min(self.retry_base * (2 ** attempt), self.retry_max)
        next_at = time.time() + delay
        self._conn.execute(
            "UPDATE webhook_deliveries SET next_retry_at = ? WHERE delivery_id = ?",
            (next_at, delivery_id),
        )
        self._conn.commit()

    async def process_retries(self) -> None:
        """Background loop — picks due retries and re-delivers."""
        while True:
            try:
                await asyncio.sleep(self.retry_check_interval)
                now = time.time()
                rows = self._conn.execute(
                    """SELECT d.*, e.endpoint_id as ep_id, e.name, e.url, e.secret,
                              e.event_filters, e.severity_filter, e.enabled
                       FROM webhook_deliveries d
                       JOIN webhook_endpoints e ON d.endpoint_id = e.endpoint_id
                       WHERE d.status = 'retrying' AND d.next_retry_at <= ?
                       LIMIT 10""",
                    (now,),
                ).fetchall()

                for row in rows:
                    ep = {
                        "endpoint_id": row["ep_id"],
                        "name": row["name"],
                        "url": row["url"],
                        "secret": row["secret"],
                    }
                    try:
                        payload = json.loads(row["payload"])
                    except (json.JSONDecodeError, TypeError):
                        payload = {}
                    await self._deliver(row["delivery_id"], ep, payload)

            except asyncio.CancelledError:
                logger.info("Webhook retry loop cancelled")
                break
            except Exception as exc:
                logger.warning("Webhook retry loop error: %s", exc)

    # ------------------------------------------------------------------
    # Inbound webhooks
    # ------------------------------------------------------------------

    async def process_inbound(
        self, payload: dict, source_ip: str = "", provided_signature: str = ""
    ) -> dict:
        """Validate an inbound webhook, log it, and create a task."""
        log_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Signature validation (if inbound_secret is configured)
        if self.inbound_secret:
            expected = self._compute_signature(
                self.inbound_secret,
                json.dumps(payload).encode(),
            )
            if not hmac.compare_digest(expected, provided_signature or ""):
                self._conn.execute(
                    """INSERT INTO webhook_inbound_log
                       (log_id, source_ip, payload, status, error, created_at)
                       VALUES (?, ?, ?, 'rejected', 'Invalid signature', ?)""",
                    (log_id, source_ip, json.dumps(payload), now),
                )
                self._conn.commit()
                logger.warning("Inbound webhook rejected — bad signature from %s", source_ip)
                return {"ok": False, "error": "Invalid signature", "log_id": log_id}

        # Create task from payload
        task_id = ""
        try:
            if self.task_queue is not None:
                description = payload.get("description", payload.get("message", "Webhook task"))
                from core.task import TaskComplexity
                task = self.task_queue.add_task(
                    description=str(description),
                    source="webhook",
                    complexity=TaskComplexity.LOW,
                )
                task_id = task.short_id
                status = "converted"
                if self.event_logger:
                    self.event_logger.info(
                        "webhook",
                        f"Inbound webhook created task {task_id}",
                        source_ip=source_ip,
                    )
            else:
                status = "received"
        except Exception as exc:
            status = "error"
            task_id = ""
            logger.warning("Inbound webhook task creation failed: %s", exc)

        self._conn.execute(
            """INSERT INTO webhook_inbound_log
               (log_id, source_ip, payload, task_id, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (log_id, source_ip, json.dumps(payload), task_id, status, now),
        )
        self._conn.commit()

        if self.on_change:
            try:
                await self.on_change()
            except Exception:
                pass

        return {"ok": True, "log_id": log_id, "task_id": task_id, "status": status}

    # ------------------------------------------------------------------
    # Status / queries
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Endpoint counts, delivery counts by status, success rate."""
        ep_total = self._conn.execute(
            "SELECT COUNT(*) FROM webhook_endpoints"
        ).fetchone()[0]
        ep_enabled = self._conn.execute(
            "SELECT COUNT(*) FROM webhook_endpoints WHERE enabled = 1"
        ).fetchone()[0]

        success = self._conn.execute(
            "SELECT COUNT(*) FROM webhook_deliveries WHERE status = 'success'"
        ).fetchone()[0]
        failed = self._conn.execute(
            "SELECT COUNT(*) FROM webhook_deliveries WHERE status = 'failed'"
        ).fetchone()[0]
        retrying = self._conn.execute(
            "SELECT COUNT(*) FROM webhook_deliveries WHERE status = 'retrying'"
        ).fetchone()[0]
        pending = self._conn.execute(
            "SELECT COUNT(*) FROM webhook_deliveries WHERE status = 'pending'"
        ).fetchone()[0]
        total_delivered = success + failed + retrying + pending

        inbound_count = self._conn.execute(
            "SELECT COUNT(*) FROM webhook_inbound_log"
        ).fetchone()[0]

        success_rate = (
            round(success / (success + failed) * 100, 1)
            if (success + failed) > 0 else 100.0
        )

        return {
            "endpoints_total": ep_total,
            "endpoints_enabled": ep_enabled,
            "deliveries_success": success,
            "deliveries_failed": failed,
            "deliveries_retrying": retrying,
            "deliveries_pending": pending,
            "deliveries_total": total_delivered,
            "success_rate": success_rate,
            "inbound_count": inbound_count,
        }

    def get_recent_deliveries(
        self, limit: int = 20, endpoint_id: str | None = None
    ) -> list[dict]:
        """Return recent delivery log entries, newest first."""
        if endpoint_id:
            rows = self._conn.execute(
                """SELECT * FROM webhook_deliveries
                   WHERE endpoint_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (endpoint_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM webhook_deliveries
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_delivery(r) for r in rows]

    def to_broadcast_dict(self) -> dict:
        """Full state for WebSocket push."""
        return {
            "endpoints": self.list_endpoints(),
            "status": self.get_status(),
            "recent_deliveries": self.get_recent_deliveries(limit=30),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_signature(secret: str, payload_bytes: bytes) -> str:
        """HMAC-SHA256 hex digest."""
        return hmac.new(
            secret.encode(), payload_bytes, hashlib.sha256
        ).hexdigest()

    def _endpoint_dict(self, endpoint_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM webhook_endpoints WHERE endpoint_id = ?",
            (endpoint_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_endpoint(row)

    @staticmethod
    def _row_to_endpoint(row) -> dict:
        try:
            filters = json.loads(row["event_filters"])
        except (json.JSONDecodeError, TypeError):
            filters = []
        return {
            "endpoint_id": row["endpoint_id"],
            "name": row["name"],
            "direction": row["direction"],
            "url": row["url"],
            "secret": row["secret"],
            "event_filters": filters,
            "severity_filter": row["severity_filter"],
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _row_to_delivery(row) -> dict:
        return {
            "delivery_id": row["delivery_id"],
            "endpoint_id": row["endpoint_id"],
            "event_category": row["event_category"],
            "event_message": row["event_message"],
            "status": row["status"],
            "http_status": row["http_status"],
            "response_body": row["response_body"],
            "attempt_count": row["attempt_count"],
            "max_attempts": row["max_attempts"],
            "error": row["error"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }

    def _update_delivery(self, delivery_id: str, **kwargs) -> None:
        """Update delivery fields."""
        sets, vals = [], []
        for k, v in kwargs.items():
            sets.append(f"{k} = ?")
            vals.append(v)
        if not sets:
            return
        vals.append(delivery_id)
        self._conn.execute(
            f"UPDATE webhook_deliveries SET {', '.join(sets)} WHERE delivery_id = ?",
            vals,
        )
        self._conn.commit()

    def _trim_deliveries(self) -> None:
        """Keep only max_delivery_history most recent deliveries."""
        count = self._conn.execute(
            "SELECT COUNT(*) FROM webhook_deliveries"
        ).fetchone()[0]
        if count > self.max_delivery_history:
            excess = count - self.max_delivery_history
            self._conn.execute(
                """DELETE FROM webhook_deliveries WHERE id IN (
                       SELECT id FROM webhook_deliveries
                       WHERE status IN ('success', 'failed')
                       ORDER BY created_at ASC LIMIT ?
                   )""",
                (excess,),
            )
            self._conn.commit()

    def close(self):
        """Close SQLite connection."""
        try:
            self._conn.close()
            logger.info("WebhookManager closed")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    async def _self_test():
        """Quick self-test for WebhookManager."""
        db_path = tempfile.mktemp(suffix=".db")
        print(f"Self-test using temp db: {db_path}")

        wm = WebhookManager(db_path=db_path)

        # Register endpoints
        ep1 = wm.register_endpoint(
            name="Test Outbound",
            url="https://httpbin.org/post",
            direction="outbound",
            event_filters=["task", "agent"],
            severity_filter="all",
        )
        print(f"  Registered: {ep1['name']} ({ep1['endpoint_id'][:8]})")

        ep2 = wm.register_endpoint(
            name="Test Inbound",
            direction="inbound",
        )
        print(f"  Registered: {ep2['name']} ({ep2['endpoint_id'][:8]})")

        # List endpoints
        eps = wm.list_endpoints()
        print(f"  Total endpoints: {len(eps)}")
        assert len(eps) == 2, "Expected 2 endpoints"

        # Filter by direction
        outbound = wm.list_endpoints(direction="outbound")
        assert len(outbound) == 1, "Expected 1 outbound"

        # Update
        updated = wm.update_endpoint(ep1["endpoint_id"], name="Updated Outbound")
        assert updated["name"] == "Updated Outbound"
        print(f"  Updated name: {updated['name']}")

        # Toggle
        toggled = wm.update_endpoint(ep1["endpoint_id"], enabled=False)
        assert not toggled["enabled"]
        print(f"  Toggled enabled: {toggled['enabled']}")

        # Status
        status = wm.get_status()
        print(f"  Status: {status}")
        assert status["endpoints_total"] == 2
        assert status["endpoints_enabled"] == 1  # ep2 is still enabled

        # Inbound
        result = await wm.process_inbound(
            {"description": "Test inbound task"},
            source_ip="127.0.0.1",
        )
        print(f"  Inbound result: {result}")
        assert result["ok"]

        # Delete
        deleted = wm.delete_endpoint(ep2["endpoint_id"])
        assert deleted
        print(f"  Deleted ep2: {deleted}")

        # Broadcast dict
        bd = wm.to_broadcast_dict()
        assert "endpoints" in bd
        assert "status" in bd
        assert "recent_deliveries" in bd
        print(f"  Broadcast dict keys: {list(bd.keys())}")

        # HMAC
        sig = wm._compute_signature("mysecret", b'{"test": true}')
        assert len(sig) == 64  # hex sha256
        print(f"  HMAC signature: {sig[:16]}...")

        wm.close()
        Path(db_path).unlink(missing_ok=True)
        print("\nAll self-tests passed!")

    asyncio.run(_self_test())
