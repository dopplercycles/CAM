"""
Performance Monitor and Optimizer for CAM.

Tracks system resource usage (CPU, memory, disk), CAM process metrics,
model response times, and agent WebSocket latency.  Stores time-series
data in SQLite with automatic retention pruning (24h detail → 7d hourly
→ 90d daily).  Raises threshold-based alerts and generates optimization
recommendations that George must approve before they execute (Tier 2).

Follows the OfflineManager pattern: SQLite-backed, async background loop,
broadcast callback.
"""

import asyncio
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

import psutil

logger = logging.getLogger("cam.performance")


# ── Constants ───────────────────────────────────────────────────────

SAMPLE_INTERVAL = 15          # seconds between metric samples
PRUNE_INTERVAL = 3600         # seconds between retention cleanup
OPTIMIZE_INTERVAL = 300       # seconds between optimization scans

DETAILED_RETENTION_HOURS = 24
HOURLY_RETENTION_DAYS = 7
DAILY_RETENTION_DAYS = 90

DEFAULT_THRESHOLDS: dict[str, float] = {
    "cpu_percent": 85.0,
    "memory_percent": 80.0,
    "disk_percent": 90.0,
    "response_time_ms": 5000.0,
    "ws_latency_ms": 500.0,
}


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class MetricSample:
    """Single point-in-time system metrics snapshot."""

    sample_id: str = ""
    timestamp: str = ""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    process_cpu: float = 0.0
    process_memory_mb: float = 0.0
    active_agents: int = 0
    active_tasks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "timestamp": self.timestamp,
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "memory_used_mb": round(self.memory_used_mb, 1),
            "memory_total_mb": round(self.memory_total_mb, 1),
            "disk_percent": round(self.disk_percent, 1),
            "disk_used_gb": round(self.disk_used_gb, 2),
            "disk_total_gb": round(self.disk_total_gb, 2),
            "process_cpu": round(self.process_cpu, 1),
            "process_memory_mb": round(self.process_memory_mb, 1),
            "active_agents": self.active_agents,
            "active_tasks": self.active_tasks,
        }

    @staticmethod
    def from_row(row: sqlite3.Row) -> "MetricSample":
        return MetricSample(
            sample_id=row["sample_id"],
            timestamp=row["timestamp"],
            cpu_percent=row["cpu_percent"] or 0.0,
            memory_percent=row["memory_percent"] or 0.0,
            memory_used_mb=row["memory_used_mb"] or 0.0,
            memory_total_mb=row["memory_total_mb"] or 0.0,
            disk_percent=row["disk_percent"] or 0.0,
            disk_used_gb=row["disk_used_gb"] or 0.0,
            disk_total_gb=row["disk_total_gb"] or 0.0,
            process_cpu=row["process_cpu"] or 0.0,
            process_memory_mb=row["process_memory_mb"] or 0.0,
            active_agents=row["active_agents"] or 0,
            active_tasks=row["active_tasks"] or 0,
        )


@dataclass
class AlertEvent:
    """A threshold-breach alert (warning or critical)."""

    alert_id: str = ""
    alert_type: str = ""
    severity: str = "warning"    # "warning" | "critical"
    message: str = ""
    value: float = 0.0
    threshold: float = 0.0
    created_at: str = ""
    resolved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "value": round(self.value, 1),
            "threshold": round(self.threshold, 1),
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }

    @staticmethod
    def from_row(row: sqlite3.Row) -> "AlertEvent":
        return AlertEvent(
            alert_id=row["alert_id"],
            alert_type=row["alert_type"],
            severity=row["severity"],
            message=row["message"],
            value=row["value"] or 0.0,
            threshold=row["threshold"] or 0.0,
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )


@dataclass
class OptimizationRec:
    """An actionable optimization recommendation (Tier 2 — George must approve)."""

    rec_id: str = ""
    category: str = ""
    title: str = ""
    description: str = ""
    impact: str = ""
    action: str = ""
    status: str = "pending"      # "pending" | "applied" | "dismissed"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rec_id": self.rec_id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "impact": self.impact,
            "action": self.action,
            "status": self.status,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_row(row: sqlite3.Row) -> "OptimizationRec":
        return OptimizationRec(
            rec_id=row["rec_id"],
            category=row["category"],
            title=row["title"],
            description=row["description"],
            impact=row["impact"],
            action=row["action"],
            status=row["status"],
            created_at=row["created_at"],
        )


# ── PerformanceMonitor ──────────────────────────────────────────────

class PerformanceMonitor:
    """
    Real-time system performance monitoring with SQLite persistence.

    Collects CPU, memory, disk, and process-level metrics every
    ``sample_interval`` seconds.  Checks thresholds for alerts, prunes
    old data on a schedule, and periodically scans for optimisation
    opportunities.

    Args:
        db_path:          Path for the performance SQLite database.
        sample_interval:  Seconds between metric samples.
        analytics_db:     Path to the analytics DB (model call latency).
        on_change:        Async callback fired after each sample cycle.
        health_monitor:   Optional HealthMonitor for WS latency data.
        registry:         Optional agent registry for active-agent count.
        task_queue:       Optional task queue for active-task count.
    """

    def __init__(
        self,
        db_path: str = "data/performance.db",
        sample_interval: int = SAMPLE_INTERVAL,
        analytics_db: str = "data/analytics.db",
        on_change: Callable[[], Coroutine] | None = None,
        health_monitor: Any = None,
        registry: Any = None,
        task_queue: Any = None,
    ):
        self._db_path = db_path
        self._sample_interval = sample_interval
        self._analytics_db = analytics_db
        self._on_change = on_change
        self._health_monitor = health_monitor
        self._registry = registry
        self._task_queue = task_queue

        self._running = False
        self._thresholds = dict(DEFAULT_THRESHOLDS)
        self._latest_sample: MetricSample | None = None
        self._total_samples = 0

        # psutil handle for CAM's own process
        self._process = psutil.Process(os.getpid())

        # Timers for periodic tasks
        self._last_prune = 0.0
        self._last_optimize = 0.0

        # Open database
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        logger.info("PerformanceMonitor initialised (interval=%ds, db=%s)",
                     sample_interval, db_path)

    # ── Schema ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        cur = self._conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS metric_samples (
                sample_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                cpu_percent REAL,
                memory_percent REAL,
                memory_used_mb REAL,
                memory_total_mb REAL,
                disk_percent REAL,
                disk_used_gb REAL,
                disk_total_gb REAL,
                process_cpu REAL,
                process_memory_mb REAL,
                active_agents INTEGER,
                active_tasks INTEGER
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_metric_ts ON metric_samples(timestamp)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS metric_hourly (
                hour TEXT PRIMARY KEY,
                cpu_avg REAL,
                cpu_max REAL,
                mem_avg REAL,
                mem_max REAL,
                disk_avg REAL,
                disk_max REAL,
                sample_count INTEGER
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                value REAL,
                threshold REAL,
                created_at TEXT,
                resolved_at TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS optimizations (
                rec_id TEXT PRIMARY KEY,
                category TEXT,
                title TEXT,
                description TEXT,
                impact TEXT,
                action TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT
            )
        """)

        self._conn.commit()

        # Count existing samples for status reporting
        row = cur.execute("SELECT COUNT(*) FROM metric_samples").fetchone()
        self._total_samples = row[0] if row else 0

    # ── Background loop ─────────────────────────────────────────────

    async def start_monitoring(self) -> None:
        """Main async loop — call as ``asyncio.create_task(pm.start_monitoring())``."""
        self._running = True
        logger.info("Performance monitoring loop started")
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Collect metrics (psutil blocks, so run in executor)
                sample = await loop.run_in_executor(None, self._read_system_metrics)
                self._store_sample(sample)
                self._latest_sample = sample
                self._total_samples += 1

                # Check thresholds for alerts
                self._check_alerts(sample)

                # Periodic pruning
                now = asyncio.get_event_loop().time()
                if now - self._last_prune > PRUNE_INTERVAL:
                    await loop.run_in_executor(None, self._prune_old_data)
                    self._last_prune = now

                # Periodic optimization scan
                if now - self._last_optimize > OPTIMIZE_INTERVAL:
                    await loop.run_in_executor(None, self._run_optimization_scan)
                    self._last_optimize = now

                # Broadcast to dashboard
                await self._fire_change()

            except Exception:
                logger.exception("Error in performance monitoring loop")

            await asyncio.sleep(self._sample_interval)

    def stop_monitoring(self) -> None:
        """Signal the background loop to stop."""
        self._running = False
        logger.info("Performance monitoring stopping")

    async def _fire_change(self) -> None:
        """Safely invoke the on_change callback."""
        if self._on_change:
            try:
                await self._on_change()
            except Exception:
                logger.exception("Error in performance on_change callback")

    # ── Metric collection ───────────────────────────────────────────

    def _read_system_metrics(self) -> MetricSample:
        """Read system and process metrics via psutil (runs in executor)."""
        now = datetime.now(timezone.utc).isoformat()

        # System-level
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # CAM process
        try:
            proc_cpu = self._process.cpu_percent()
            proc_mem_mb = self._process.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_cpu = 0.0
            proc_mem_mb = 0.0

        # Active agents count (from registry if available)
        active_agents = 0
        if self._registry:
            try:
                agents = self._registry.list_agents()
                active_agents = sum(
                    1 for a in agents
                    if getattr(a, "status", "") == "online"
                )
            except Exception:
                pass

        # Active tasks count (from task queue if available)
        active_tasks = 0
        if self._task_queue:
            try:
                active_tasks = self._task_queue.depth().get("pending", 0)
            except Exception:
                pass

        return MetricSample(
            sample_id=uuid.uuid4().hex[:16],
            timestamp=now,
            cpu_percent=cpu,
            memory_percent=mem.percent,
            memory_used_mb=mem.used / (1024 * 1024),
            memory_total_mb=mem.total / (1024 * 1024),
            disk_percent=disk.percent,
            disk_used_gb=disk.used / (1024 ** 3),
            disk_total_gb=disk.total / (1024 ** 3),
            process_cpu=proc_cpu,
            process_memory_mb=proc_mem_mb,
            active_agents=active_agents,
            active_tasks=active_tasks,
        )

    def _store_sample(self, s: MetricSample) -> None:
        """Persist a metric sample to SQLite."""
        try:
            self._conn.execute(
                """INSERT INTO metric_samples
                   (sample_id, timestamp, cpu_percent, memory_percent,
                    memory_used_mb, memory_total_mb, disk_percent,
                    disk_used_gb, disk_total_gb, process_cpu,
                    process_memory_mb, active_agents, active_tasks)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    s.sample_id, s.timestamp, s.cpu_percent, s.memory_percent,
                    s.memory_used_mb, s.memory_total_mb, s.disk_percent,
                    s.disk_used_gb, s.disk_total_gb, s.process_cpu,
                    s.process_memory_mb, s.active_agents, s.active_tasks,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.exception("Failed to store metric sample")

    # ── Alert checking ──────────────────────────────────────────────

    def _check_alerts(self, sample: MetricSample) -> None:
        """Compare current sample against thresholds and create/resolve alerts."""
        checks = [
            ("cpu_percent", sample.cpu_percent, self._thresholds["cpu_percent"]),
            ("memory_percent", sample.memory_percent, self._thresholds["memory_percent"]),
            ("disk_percent", sample.disk_percent, self._thresholds["disk_percent"]),
        ]

        now = datetime.now(timezone.utc).isoformat()

        for alert_type, value, threshold in checks:
            warning_threshold = threshold * 0.8
            # Check for active (unresolved) alert of this type
            row = self._conn.execute(
                "SELECT alert_id, severity FROM alerts WHERE alert_type = ? AND resolved_at IS NULL",
                (alert_type,),
            ).fetchone()

            if value >= threshold:
                # Critical
                if row and row["severity"] == "critical":
                    continue  # already have a critical alert
                if row:
                    # Upgrade warning → critical
                    self._conn.execute(
                        "UPDATE alerts SET severity = 'critical', value = ? WHERE alert_id = ?",
                        (value, row["alert_id"]),
                    )
                else:
                    self._conn.execute(
                        """INSERT INTO alerts (alert_id, alert_type, severity, message,
                           value, threshold, created_at)
                           VALUES (?,?,?,?,?,?,?)""",
                        (
                            uuid.uuid4().hex[:16], alert_type, "critical",
                            f"{alert_type} at {value:.1f}% (threshold: {threshold:.1f}%)",
                            value, threshold, now,
                        ),
                    )
                self._conn.commit()

            elif value >= warning_threshold:
                # Warning
                if row:
                    continue  # already alerting
                self._conn.execute(
                    """INSERT INTO alerts (alert_id, alert_type, severity, message,
                       value, threshold, created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        uuid.uuid4().hex[:16], alert_type, "warning",
                        f"{alert_type} at {value:.1f}% (warning threshold: {warning_threshold:.1f}%)",
                        value, threshold, now,
                    ),
                )
                self._conn.commit()

            else:
                # Below warning — resolve any active alert
                if row:
                    self._conn.execute(
                        "UPDATE alerts SET resolved_at = ? WHERE alert_id = ?",
                        (now, row["alert_id"]),
                    )
                    self._conn.commit()

    # ── Retention pruning ───────────────────────────────────────────

    def _prune_old_data(self) -> None:
        """Aggregate old samples into hourly buckets and delete stale data."""
        now = datetime.now(timezone.utc)
        detail_cutoff = (now - timedelta(hours=DETAILED_RETENTION_HOURS)).isoformat()
        hourly_cutoff = (now - timedelta(days=HOURLY_RETENTION_DAYS)).isoformat()

        try:
            # Aggregate samples older than 24h into hourly buckets
            rows = self._conn.execute(
                """SELECT strftime('%%Y-%%m-%%dT%%H:00:00', timestamp) AS hour,
                          AVG(cpu_percent) AS cpu_avg, MAX(cpu_percent) AS cpu_max,
                          AVG(memory_percent) AS mem_avg, MAX(memory_percent) AS mem_max,
                          AVG(disk_percent) AS disk_avg, MAX(disk_percent) AS disk_max,
                          COUNT(*) AS sample_count
                   FROM metric_samples
                   WHERE timestamp < ?
                   GROUP BY hour""",
                (detail_cutoff,),
            ).fetchall()

            for r in rows:
                self._conn.execute(
                    """INSERT OR REPLACE INTO metric_hourly
                       (hour, cpu_avg, cpu_max, mem_avg, mem_max,
                        disk_avg, disk_max, sample_count)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (r["hour"], r["cpu_avg"], r["cpu_max"], r["mem_avg"],
                     r["mem_max"], r["disk_avg"], r["disk_max"], r["sample_count"]),
                )

            # Delete aggregated detail samples
            self._conn.execute(
                "DELETE FROM metric_samples WHERE timestamp < ?",
                (detail_cutoff,),
            )

            # Delete hourly data older than retention
            self._conn.execute(
                "DELETE FROM metric_hourly WHERE hour < ?",
                (hourly_cutoff,),
            )

            # Delete resolved alerts older than 7 days
            alert_cutoff = (now - timedelta(days=7)).isoformat()
            self._conn.execute(
                "DELETE FROM alerts WHERE resolved_at IS NOT NULL AND resolved_at < ?",
                (alert_cutoff,),
            )

            self._conn.commit()
            logger.debug("Performance data pruning complete")
        except Exception:
            logger.exception("Error pruning performance data")

    # ── Optimization scan ───────────────────────────────────────────

    def _run_optimization_scan(self) -> None:
        """Scan for optimization opportunities (recommend-only, never auto-apply)."""
        now = datetime.now(timezone.utc).isoformat()

        # 1. Large log files
        log_dir = Path("data/logs")
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                try:
                    size_mb = log_file.stat().st_size / (1024 * 1024)
                    if size_mb > 50:
                        self._add_recommendation(
                            category="disk",
                            title=f"Large log file: {log_file.name}",
                            description=f"{log_file.name} is {size_mb:.0f} MB. "
                                        "Truncating to last 1000 lines frees disk space.",
                            impact=f"Free ~{size_mb - 1:.0f} MB disk space",
                            action=f"truncate_log:{log_file}",
                            now=now,
                        )
                except OSError:
                    pass

        # 2. Total database size
        data_dir = Path("data")
        if data_dir.exists():
            total_db_mb = 0.0
            for db_file in data_dir.glob("*.db"):
                try:
                    total_db_mb += db_file.stat().st_size / (1024 * 1024)
                except OSError:
                    pass
            if total_db_mb > 500:
                self._add_recommendation(
                    category="disk",
                    title="Database files exceed 500 MB",
                    description=f"Total database size is {total_db_mb:.0f} MB. "
                                "Running VACUUM on each database can reclaim unused space.",
                    impact="Potentially reclaim 10-30% disk space",
                    action="vacuum_databases",
                    now=now,
                )

        # 3. Model latency degradation
        try:
            latency = self.get_model_latency_by_tier()
            for model, stats in latency.items():
                if stats.get("avg_1h") and stats.get("avg_24h"):
                    if stats["avg_1h"] > stats["avg_24h"] * 1.5:
                        self._add_recommendation(
                            category="performance",
                            title=f"Model latency spike: {model}",
                            description=f"{model} average latency last hour "
                                        f"({stats['avg_1h']:.0f} ms) is 1.5x the "
                                        f"24h average ({stats['avg_24h']:.0f} ms).",
                            impact="Slower response times for users",
                            action=f"check_model:{model}",
                            now=now,
                        )
        except Exception:
            pass

        # 4. Disk growth trend
        try:
            week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            oldest = self._conn.execute(
                "SELECT disk_percent FROM metric_samples WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
                (week_ago,),
            ).fetchone()
            newest = self._conn.execute(
                "SELECT disk_percent FROM metric_samples ORDER BY timestamp DESC LIMIT 1",
            ).fetchone()
            if oldest and newest:
                growth = (newest["disk_percent"] or 0) - (oldest["disk_percent"] or 0)
                if growth > 5.0:
                    self._add_recommendation(
                        category="disk",
                        title="Rapid disk usage growth",
                        description=f"Disk usage grew {growth:.1f}% in the last 7 days. "
                                    "Review data directories for large or unnecessary files.",
                        impact="May run out of disk space",
                        action="disk_cleanup",
                        now=now,
                    )
        except Exception:
            pass

        # 5. Stale analytics data
        try:
            if Path(self._analytics_db).exists():
                aconn = sqlite3.connect(self._analytics_db)
                cutoff_90d = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
                row = aconn.execute(
                    "SELECT COUNT(*) FROM model_call_records WHERE recorded_at < ?",
                    (cutoff_90d,),
                ).fetchone()
                aconn.close()
                if row and row[0] > 1000:
                    self._add_recommendation(
                        category="maintenance",
                        title="Stale analytics data",
                        description=f"{row[0]} model call records are older than 90 days. "
                                    "Purging reduces database size and query times.",
                        impact="Smaller analytics DB, faster queries",
                        action="purge_old_analytics",
                        now=now,
                    )
        except Exception:
            pass

        logger.debug("Optimization scan complete")

    def _add_recommendation(self, category: str, title: str, description: str,
                            impact: str, action: str, now: str) -> None:
        """Insert a recommendation if no pending duplicate exists."""
        existing = self._conn.execute(
            "SELECT rec_id FROM optimizations WHERE category = ? AND action = ? AND status = 'pending'",
            (category, action),
        ).fetchone()
        if existing:
            return

        self._conn.execute(
            """INSERT INTO optimizations
               (rec_id, category, title, description, impact, action, status, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (uuid.uuid4().hex[:16], category, title, description, impact, action, "pending", now),
        )
        self._conn.commit()

    # ── Apply / dismiss recommendations ─────────────────────────────

    def apply_recommendation(self, rec_id: str) -> dict[str, Any]:
        """
        Execute a recommendation action (Tier 2 — George clicked Apply).

        Returns a result dict with ``ok`` and ``message`` keys.
        """
        row = self._conn.execute(
            "SELECT * FROM optimizations WHERE rec_id = ?", (rec_id,),
        ).fetchone()
        if not row:
            return {"ok": False, "message": "Recommendation not found"}

        action = row["action"]
        result: dict[str, Any] = {"ok": False, "message": "Unknown action"}

        try:
            if action.startswith("truncate_log:"):
                result = self._action_truncate_log(action.split(":", 1)[1])
            elif action == "vacuum_databases":
                result = self._action_vacuum_databases()
            elif action == "purge_old_analytics":
                result = self._action_purge_old_analytics()
            elif action.startswith("check_model:"):
                # Informational — no automated action, just acknowledge
                result = {"ok": True, "message": f"Acknowledged — monitor {action.split(':', 1)[1]} latency"}
            elif action == "disk_cleanup":
                result = {"ok": True, "message": "Acknowledged — review data directories manually"}
            else:
                result = {"ok": False, "message": f"No handler for action: {action}"}
        except Exception as exc:
            result = {"ok": False, "message": str(exc)}

        # Mark as applied
        self._conn.execute(
            "UPDATE optimizations SET status = 'applied' WHERE rec_id = ?",
            (rec_id,),
        )
        self._conn.commit()
        logger.info("Applied recommendation %s: %s → %s", rec_id, action, result.get("message"))
        return result

    def dismiss_recommendation(self, rec_id: str) -> bool:
        """Mark a recommendation as dismissed."""
        cur = self._conn.execute(
            "UPDATE optimizations SET status = 'dismissed' WHERE rec_id = ? AND status = 'pending'",
            (rec_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── Recommendation action handlers ──────────────────────────────

    def _action_truncate_log(self, path_str: str) -> dict[str, Any]:
        """Truncate a log file to its last 1000 lines."""
        path = Path(path_str)
        if not path.exists():
            return {"ok": False, "message": f"File not found: {path}"}
        if not str(path).startswith("data/"):
            return {"ok": False, "message": "Can only truncate files in data/"}

        lines = path.read_text().splitlines()
        original_size = path.stat().st_size
        kept = lines[-1000:] if len(lines) > 1000 else lines
        path.write_text("\n".join(kept) + "\n")
        new_size = path.stat().st_size
        freed = (original_size - new_size) / (1024 * 1024)
        return {"ok": True, "message": f"Truncated {path.name} — freed {freed:.1f} MB"}

    def _action_vacuum_databases(self) -> dict[str, Any]:
        """Run VACUUM on each .db file in data/."""
        results = []
        for db_file in Path("data").glob("*.db"):
            try:
                before = db_file.stat().st_size
                conn = sqlite3.connect(str(db_file))
                conn.execute("VACUUM")
                conn.close()
                after = db_file.stat().st_size
                freed = (before - after) / (1024 * 1024)
                if freed > 0.1:
                    results.append(f"{db_file.name}: freed {freed:.1f} MB")
            except Exception as exc:
                results.append(f"{db_file.name}: error — {exc}")
        msg = "; ".join(results) if results else "All databases already compact"
        return {"ok": True, "message": msg}

    def _action_purge_old_analytics(self) -> dict[str, Any]:
        """Delete analytics records older than 90 days."""
        if not Path(self._analytics_db).exists():
            return {"ok": False, "message": "Analytics DB not found"}

        cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        conn = sqlite3.connect(self._analytics_db)
        cur = conn.execute("DELETE FROM model_call_records WHERE recorded_at < ?", (cutoff,))
        deleted = cur.rowcount
        conn.execute("VACUUM")
        conn.commit()
        conn.close()
        return {"ok": True, "message": f"Purged {deleted} records older than 90 days"}

    # ── Cross-module queries ────────────────────────────────────────

    def get_model_latency_by_tier(self) -> dict[str, dict[str, float | None]]:
        """Read model_call_records from analytics.db, return avg latency by model."""
        if not Path(self._analytics_db).exists():
            return {}

        result: dict[str, dict[str, float | None]] = {}
        try:
            conn = sqlite3.connect(self._analytics_db)
            now = datetime.now(timezone.utc)
            cutoff_1h = (now - timedelta(hours=1)).isoformat()
            cutoff_24h = (now - timedelta(hours=24)).isoformat()

            # Last 24h average by model
            rows_24h = conn.execute(
                """SELECT model, AVG(latency_ms) AS avg_lat, COUNT(*) AS cnt
                   FROM model_call_records
                   WHERE recorded_at >= ?
                   GROUP BY model""",
                (cutoff_24h,),
            ).fetchall()

            for r in rows_24h:
                result[r[0]] = {"avg_24h": round(r[1], 1), "calls_24h": r[2], "avg_1h": None, "calls_1h": 0}

            # Last 1h average by model
            rows_1h = conn.execute(
                """SELECT model, AVG(latency_ms) AS avg_lat, COUNT(*) AS cnt
                   FROM model_call_records
                   WHERE recorded_at >= ?
                   GROUP BY model""",
                (cutoff_1h,),
            ).fetchall()

            for r in rows_1h:
                if r[0] in result:
                    result[r[0]]["avg_1h"] = round(r[1], 1)
                    result[r[0]]["calls_1h"] = r[2]
                else:
                    result[r[0]] = {"avg_24h": None, "calls_24h": 0, "avg_1h": round(r[1], 1), "calls_1h": r[2]}

            conn.close()
        except Exception:
            logger.exception("Error reading model latency from analytics DB")

        return result

    def get_ws_latency(self) -> dict[str, float | None]:
        """Read last_ping_rtt_ms from HealthMonitor for each agent."""
        if not self._health_monitor:
            return {}

        result: dict[str, float | None] = {}
        try:
            for agent_id, metrics in self._health_monitor._metrics.items():
                result[agent_id] = (
                    round(metrics.last_ping_rtt_ms, 1)
                    if metrics.last_ping_rtt_ms is not None else None
                )
        except Exception:
            logger.exception("Error reading WS latency from health monitor")

        return result

    def get_db_sizes(self) -> list[dict[str, Any]]:
        """Return name + size for each .db file in data/."""
        sizes: list[dict[str, Any]] = []
        data_dir = Path("data")
        if not data_dir.exists():
            return sizes

        for db_file in sorted(data_dir.glob("*.db")):
            try:
                size_bytes = db_file.stat().st_size
                sizes.append({
                    "name": db_file.name,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                })
            except OSError:
                pass

        return sizes

    # ── Status / broadcast ──────────────────────────────────────────

    def get_recent_samples(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent metric samples."""
        rows = self._conn.execute(
            "SELECT * FROM metric_samples ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [MetricSample.from_row(r).to_dict() for r in reversed(rows)]

    def get_active_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return unresolved alerts, most recent first."""
        rows = self._conn.execute(
            "SELECT * FROM alerts WHERE resolved_at IS NULL ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [AlertEvent.from_row(r).to_dict() for r in rows]

    def get_recommendations(self) -> list[dict[str, Any]]:
        """Return pending optimization recommendations."""
        rows = self._conn.execute(
            "SELECT * FROM optimizations WHERE status = 'pending' ORDER BY created_at DESC",
        ).fetchall()
        return [OptimizationRec.from_row(r).to_dict() for r in rows]

    def get_hourly_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Return hourly aggregate data for the given time window."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self._conn.execute(
            "SELECT * FROM metric_hourly WHERE hour >= ? ORDER BY hour ASC",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_status(self) -> dict[str, Any]:
        """Summary status dict."""
        return {
            "monitoring": self._running,
            "sample_interval": self._sample_interval,
            "thresholds": dict(self._thresholds),
            "latest_sample": self._latest_sample.to_dict() if self._latest_sample else None,
            "active_alerts": len(self.get_active_alerts(limit=100)),
            "total_samples": self._total_samples,
        }

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Full state snapshot for dashboard broadcast."""
        return {
            "status": self.get_status(),
            "recent_samples": self.get_recent_samples(20),
            "model_latency": self.get_model_latency_by_tier(),
            "ws_latency": self.get_ws_latency(),
            "db_sizes": self.get_db_sizes(),
            "alerts": self.get_active_alerts(20),
            "recommendations": self.get_recommendations(),
            "history_hourly": self.get_hourly_history(24),
        }

    def set_threshold(self, key: str, value: float) -> bool:
        """Update a threshold at runtime. Returns True if key was valid."""
        if key not in self._thresholds:
            return False
        self._thresholds[key] = value
        logger.info("Threshold updated: %s = %.1f", key, value)
        return True

    def close(self) -> None:
        """Stop monitoring and close the database."""
        self.stop_monitoring()
        try:
            self._conn.close()
        except Exception:
            pass
        logger.info("PerformanceMonitor closed")
