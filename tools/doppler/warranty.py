"""
Warranty & Recall Tracker for CAM / Doppler Cycles.

Tracks warranty status and recall notices for customer vehicles.  Monitors
expiration windows (30/60/90 days), checks vehicles against known NHTSA
recalls, and proactively notifies customers about relevant recalls and
expiring warranties.

Integrates with:
  - CRMStore              (customer lookups, vehicles, notes)
  - ServiceRecordStore    (vehicle data)
  - EmailTemplateManager  (expiration/recall notifications)
  - NotificationManager   (dashboard alerts)
  - KnowledgeIngest       (recall database import)

SQLite-backed, single-file module — same pattern as feedback.py, ride_log.py.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("cam.warranty")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COVERAGE_TYPES = ("factory", "extended", "powertrain", "aftermarket", "dealer")

COVERAGE_LABELS = {
    "factory": "Factory Warranty",
    "extended": "Extended Warranty",
    "powertrain": "Powertrain",
    "aftermarket": "Aftermarket",
    "dealer": "Dealer Warranty",
}

WARRANTY_STATUSES = ("active", "expiring_soon", "expired")

# Alert windows in days
EXPIRY_WINDOWS = (30, 60, 90)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WarrantyItem:
    """A warranty record tied to a specific customer vehicle.

    Attributes:
        warranty_id:        Unique identifier (UUID string).
        vehicle_id:         Link to ServiceRecordStore vehicle_id.
        customer_id:        Link to CRM customer_id.
        component:          What's covered (e.g. 'engine', 'full vehicle').
        start_date:         Coverage start (YYYY-MM-DD).
        end_date:           Coverage end (YYYY-MM-DD).
        coverage_type:      One of COVERAGE_TYPES.
        provider:           Warranty provider name.
        documentation_path: Path to warranty docs (PDF, photo, etc.).
        mileage_limit:      Max mileage for coverage (0 = unlimited).
        notes:              Free-text notes.
        created_at:         Record creation timestamp (ISO-8601).
        updated_at:         Last modification timestamp (ISO-8601).
    """
    warranty_id: str
    vehicle_id: str = ""
    customer_id: str = ""
    component: str = ""
    start_date: str = ""
    end_date: str = ""
    coverage_type: str = "factory"
    provider: str = ""
    documentation_path: str = ""
    mileage_limit: int = 0
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def short_id(self) -> str:
        return self.warranty_id[:8]

    @property
    def status(self) -> str:
        """Compute warranty status from end_date."""
        if not self.end_date:
            return "active"
        try:
            end = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        except ValueError:
            return "active"
        today = datetime.now(timezone.utc).date()
        if end < today:
            return "expired"
        if (end - today).days <= 90:
            return "expiring_soon"
        return "active"

    @property
    def days_remaining(self) -> Optional[int]:
        """Days until warranty expires, or None if no end date."""
        if not self.end_date:
            return None
        try:
            end = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        except ValueError:
            return None
        return (end - datetime.now(timezone.utc).date()).days

    def to_dict(self) -> dict[str, Any]:
        return {
            "warranty_id": self.warranty_id,
            "vehicle_id": self.vehicle_id,
            "customer_id": self.customer_id,
            "component": self.component,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "coverage_type": self.coverage_type,
            "provider": self.provider,
            "documentation_path": self.documentation_path,
            "mileage_limit": self.mileage_limit,
            "notes": self.notes,
            "status": self.status,
            "days_remaining": self.days_remaining,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_row(row: sqlite3.Row) -> "WarrantyItem":
        r = dict(row)
        return WarrantyItem(
            warranty_id=r.get("warranty_id", ""),
            vehicle_id=r.get("vehicle_id", ""),
            customer_id=r.get("customer_id", ""),
            component=r.get("component", ""),
            start_date=r.get("start_date", ""),
            end_date=r.get("end_date", ""),
            coverage_type=r.get("coverage_type", "factory"),
            provider=r.get("provider", ""),
            documentation_path=r.get("documentation_path", ""),
            mileage_limit=int(r.get("mileage_limit", 0)),
            notes=r.get("notes", ""),
            created_at=r.get("created_at", ""),
            updated_at=r.get("updated_at", ""),
        )


@dataclass
class RecallNotice:
    """A safety recall notice (typically from NHTSA).

    Attributes:
        recall_id:     Unique identifier (UUID string).
        nhtsa_id:      NHTSA campaign number (e.g. '24V-123').
        make:          Manufacturer.
        model:         Model name.
        year_start:    First affected model year.
        year_end:      Last affected model year.
        component:     Affected component/system.
        description:   Recall description.
        remedy:        Recommended fix / dealer remedy.
        date_issued:   When the recall was issued (YYYY-MM-DD).
        severity:      'safety', 'emissions', 'compliance'.
        notes:         Additional notes.
        created_at:    Record creation timestamp (ISO-8601).
        updated_at:    Last modification timestamp (ISO-8601).
    """
    recall_id: str
    nhtsa_id: str = ""
    make: str = ""
    model: str = ""
    year_start: int = 0
    year_end: int = 0
    component: str = ""
    description: str = ""
    remedy: str = ""
    date_issued: str = ""
    severity: str = "safety"
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def short_id(self) -> str:
        return self.recall_id[:8]

    @property
    def year_range(self) -> str:
        """Human-readable year range."""
        if self.year_start == self.year_end or self.year_end == 0:
            return str(self.year_start)
        return f"{self.year_start}-{self.year_end}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall_id": self.recall_id,
            "nhtsa_id": self.nhtsa_id,
            "make": self.make,
            "model": self.model,
            "year_start": self.year_start,
            "year_end": self.year_end,
            "year_range": self.year_range,
            "component": self.component,
            "description": self.description,
            "remedy": self.remedy,
            "date_issued": self.date_issued,
            "severity": self.severity,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_row(row: sqlite3.Row) -> "RecallNotice":
        r = dict(row)
        return RecallNotice(
            recall_id=r.get("recall_id", ""),
            nhtsa_id=r.get("nhtsa_id", ""),
            make=r.get("make", ""),
            model=r.get("model", ""),
            year_start=int(r.get("year_start", 0)),
            year_end=int(r.get("year_end", 0)),
            component=r.get("component", ""),
            description=r.get("description", ""),
            remedy=r.get("remedy", ""),
            date_issued=r.get("date_issued", ""),
            severity=r.get("severity", "safety"),
            notes=r.get("notes", ""),
            created_at=r.get("created_at", ""),
            updated_at=r.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class WarrantyRecallManager:
    """Manages warranty items and recall notices with SQLite persistence.

    Provides CRUD for both warranties and recalls, expiration alerting,
    vehicle-recall cross-referencing, and customer notification via email
    templates and the notification manager.
    """

    def __init__(
        self,
        db_path: str = "data/warranty.db",
        *,
        crm_store: Any = None,
        service_store: Any = None,
        notification_manager: Any = None,
        email_template_manager: Any = None,
        on_change: Optional[Callable[[], Coroutine]] = None,
        on_model_call: Optional[Callable] = None,
    ):
        self._crm = crm_store
        self._service_store = service_store
        self._notifications = notification_manager
        self._email = email_template_manager
        self._on_change = on_change
        self._on_model_call = on_model_call

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("WarrantyRecallManager initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS warranties (
                warranty_id        TEXT PRIMARY KEY,
                vehicle_id         TEXT DEFAULT '',
                customer_id        TEXT DEFAULT '',
                component          TEXT DEFAULT '',
                start_date         TEXT DEFAULT '',
                end_date           TEXT DEFAULT '',
                coverage_type      TEXT DEFAULT 'factory',
                provider           TEXT DEFAULT '',
                documentation_path TEXT DEFAULT '',
                mileage_limit      INTEGER DEFAULT 0,
                notes              TEXT DEFAULT '',
                created_at         TEXT NOT NULL,
                updated_at         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_war_vehicle  ON warranties(vehicle_id);
            CREATE INDEX IF NOT EXISTS idx_war_customer ON warranties(customer_id);
            CREATE INDEX IF NOT EXISTS idx_war_end      ON warranties(end_date);
            CREATE INDEX IF NOT EXISTS idx_war_type     ON warranties(coverage_type);

            CREATE TABLE IF NOT EXISTS recalls (
                recall_id    TEXT PRIMARY KEY,
                nhtsa_id     TEXT DEFAULT '',
                make         TEXT DEFAULT '',
                model        TEXT DEFAULT '',
                year_start   INTEGER DEFAULT 0,
                year_end     INTEGER DEFAULT 0,
                component    TEXT DEFAULT '',
                description  TEXT DEFAULT '',
                remedy       TEXT DEFAULT '',
                date_issued  TEXT DEFAULT '',
                severity     TEXT DEFAULT 'safety',
                notes        TEXT DEFAULT '',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rec_make  ON recalls(make);
            CREATE INDEX IF NOT EXISTS idx_rec_model ON recalls(model);
            CREATE INDEX IF NOT EXISTS idx_rec_nhtsa ON recalls(nhtsa_id);

            CREATE TABLE IF NOT EXISTS recall_alerts (
                alert_id    TEXT PRIMARY KEY,
                recall_id   TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                vehicle_id  TEXT DEFAULT '',
                notified_at TEXT NOT NULL,
                method      TEXT DEFAULT 'dashboard',
                FOREIGN KEY (recall_id) REFERENCES recalls(recall_id)
            );
            CREATE INDEX IF NOT EXISTS idx_alert_cust ON recall_alerts(customer_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    def _fire_change(self):
        """Trigger the async on_change callback if set."""
        if self._on_change is None:
            return
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._on_change())
        except RuntimeError:
            pass

    # ==================================================================
    # WARRANTY CRUD
    # ==================================================================

    def add_warranty(
        self,
        vehicle_id: str = "",
        customer_id: str = "",
        component: str = "",
        start_date: str = "",
        end_date: str = "",
        coverage_type: str = "factory",
        provider: str = "",
        documentation_path: str = "",
        mileage_limit: int = 0,
        notes: str = "",
    ) -> WarrantyItem:
        """Create a new warranty record."""
        if coverage_type not in COVERAGE_TYPES:
            coverage_type = "factory"

        now = self._now()
        wid = str(uuid.uuid4())

        self._conn.execute(
            """INSERT INTO warranties
               (warranty_id, vehicle_id, customer_id, component, start_date,
                end_date, coverage_type, provider, documentation_path,
                mileage_limit, notes, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (wid, vehicle_id, customer_id, component, start_date,
             end_date, coverage_type, provider, documentation_path,
             mileage_limit, notes, now, now),
        )
        self._conn.commit()

        item = self.get_warranty(wid)
        logger.info("Warranty added %s — %s %s (%s)", item.short_id,
                     coverage_type, component, vehicle_id or "no vehicle")

        # CRM note
        if customer_id and self._crm:
            try:
                self._crm.add_note(
                    customer_id,
                    f"Warranty added: {COVERAGE_LABELS.get(coverage_type, coverage_type)} "
                    f"— {component or 'full vehicle'}, expires {end_date or 'N/A'}",
                    category="warranty",
                )
            except Exception:
                logger.debug("CRM note failed for warranty %s", wid, exc_info=True)

        self._fire_change()
        return item

    def update_warranty(self, warranty_id: str, **kwargs) -> Optional[WarrantyItem]:
        """Update a warranty record. Only supplied fields are changed."""
        existing = self.get_warranty(warranty_id)
        if not existing:
            return None

        allowed = {
            "vehicle_id", "customer_id", "component", "start_date",
            "end_date", "coverage_type", "provider", "documentation_path",
            "mileage_limit", "notes",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return existing

        if "coverage_type" in updates and updates["coverage_type"] not in COVERAGE_TYPES:
            updates["coverage_type"] = existing.coverage_type
        if "mileage_limit" in updates:
            updates["mileage_limit"] = int(updates["mileage_limit"])

        updates["updated_at"] = self._now()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [warranty_id]
        self._conn.execute(
            f"UPDATE warranties SET {set_clause} WHERE warranty_id = ?", vals,
        )
        self._conn.commit()
        logger.info("Warranty updated %s", warranty_id[:8])
        self._fire_change()
        return self.get_warranty(warranty_id)

    def delete_warranty(self, warranty_id: str) -> bool:
        """Delete a warranty record."""
        cur = self._conn.execute(
            "DELETE FROM warranties WHERE warranty_id = ?", (warranty_id,),
        )
        self._conn.commit()
        if cur.rowcount > 0:
            logger.info("Warranty deleted %s", warranty_id[:8])
            self._fire_change()
            return True
        return False

    def get_warranty(self, warranty_id: str) -> Optional[WarrantyItem]:
        """Fetch a single warranty by ID."""
        cur = self._conn.execute(
            "SELECT * FROM warranties WHERE warranty_id = ?", (warranty_id,),
        )
        row = cur.fetchone()
        return WarrantyItem.from_row(row) if row else None

    def list_warranties(
        self,
        customer_id: str = "",
        vehicle_id: str = "",
        coverage_type: str = "",
        status: str = "",
        limit: int = 200,
    ) -> list[WarrantyItem]:
        """List warranties with optional filters."""
        query = "SELECT * FROM warranties WHERE 1=1"
        params: list[Any] = []

        if customer_id:
            query += " AND customer_id = ?"
            params.append(customer_id)
        if vehicle_id:
            query += " AND vehicle_id = ?"
            params.append(vehicle_id)
        if coverage_type:
            query += " AND coverage_type = ?"
            params.append(coverage_type)

        query += " ORDER BY end_date ASC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        items = [WarrantyItem.from_row(r) for r in rows]

        # Post-filter by computed status if requested
        if status:
            items = [w for w in items if w.status == status]

        return items

    def check_warranty(self, vehicle_id: str) -> list[WarrantyItem]:
        """Check all warranties for a specific vehicle.

        Returns all warranty items (active, expiring, expired) for the
        given vehicle, sorted by end_date ascending.
        """
        return self.list_warranties(vehicle_id=vehicle_id)

    # ==================================================================
    # RECALL CRUD
    # ==================================================================

    def add_recall(
        self,
        nhtsa_id: str = "",
        make: str = "",
        model: str = "",
        year_start: int = 0,
        year_end: int = 0,
        component: str = "",
        description: str = "",
        remedy: str = "",
        date_issued: str = "",
        severity: str = "safety",
        notes: str = "",
    ) -> RecallNotice:
        """Add a recall notice to the database."""
        now = self._now()
        rid = str(uuid.uuid4())

        # Default year_end to year_start if not provided
        if year_end == 0 and year_start > 0:
            year_end = year_start

        self._conn.execute(
            """INSERT INTO recalls
               (recall_id, nhtsa_id, make, model, year_start, year_end,
                component, description, remedy, date_issued, severity,
                notes, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (rid, nhtsa_id, make, model, year_start, year_end,
             component, description, remedy, date_issued, severity,
             notes, now, now),
        )
        self._conn.commit()

        recall = self.get_recall(rid)
        logger.info("Recall added %s — %s %s %s (%s)",
                     recall.short_id, make, model, recall.year_range, nhtsa_id)
        self._fire_change()
        return recall

    def update_recall(self, recall_id: str, **kwargs) -> Optional[RecallNotice]:
        """Update a recall notice."""
        existing = self.get_recall(recall_id)
        if not existing:
            return None

        allowed = {
            "nhtsa_id", "make", "model", "year_start", "year_end",
            "component", "description", "remedy", "date_issued",
            "severity", "notes",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return existing

        if "year_start" in updates:
            updates["year_start"] = int(updates["year_start"])
        if "year_end" in updates:
            updates["year_end"] = int(updates["year_end"])

        updates["updated_at"] = self._now()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [recall_id]
        self._conn.execute(
            f"UPDATE recalls SET {set_clause} WHERE recall_id = ?", vals,
        )
        self._conn.commit()
        logger.info("Recall updated %s", recall_id[:8])
        self._fire_change()
        return self.get_recall(recall_id)

    def delete_recall(self, recall_id: str) -> bool:
        """Delete a recall notice."""
        cur = self._conn.execute(
            "DELETE FROM recalls WHERE recall_id = ?", (recall_id,),
        )
        self._conn.commit()
        if cur.rowcount > 0:
            logger.info("Recall deleted %s", recall_id[:8])
            self._fire_change()
            return True
        return False

    def get_recall(self, recall_id: str) -> Optional[RecallNotice]:
        """Fetch a single recall by ID."""
        cur = self._conn.execute(
            "SELECT * FROM recalls WHERE recall_id = ?", (recall_id,),
        )
        row = cur.fetchone()
        return RecallNotice.from_row(row) if row else None

    def list_recalls(
        self,
        make: str = "",
        model: str = "",
        severity: str = "",
        limit: int = 200,
    ) -> list[RecallNotice]:
        """List recalls with optional filters."""
        query = "SELECT * FROM recalls WHERE 1=1"
        params: list[Any] = []

        if make:
            query += " AND LOWER(make) = LOWER(?)"
            params.append(make)
        if model:
            query += " AND LOWER(model) = LOWER(?)"
            params.append(model)
        if severity:
            query += " AND severity = ?"
            params.append(severity)

        query += " ORDER BY date_issued DESC LIMIT ?"
        params.append(limit)

        return [RecallNotice.from_row(r) for r in self._conn.execute(query, params)]

    # ==================================================================
    # CROSS-REFERENCE: Scan customer vehicles against recalls
    # ==================================================================

    def scan_recalls(self, customer_id: str = "") -> list[dict[str, Any]]:
        """Check customer vehicles against known recalls.

        If customer_id is given, only scan that customer's vehicles.
        Otherwise scan all vehicles in the service record store.

        Returns a list of matches: [{vehicle, recall, customer_id}].
        """
        matches: list[dict[str, Any]] = []

        if self._service_store is None:
            logger.debug("No service store — cannot scan vehicles for recalls")
            return matches

        # Get vehicles to check
        if customer_id and self._crm:
            try:
                profile = self._crm.get_customer_profile(customer_id)
                vehicles = profile.get("vehicles", [])
            except Exception:
                vehicles = []
        else:
            try:
                vehicles = [v.to_dict() for v in self._service_store.list_vehicles()]
            except Exception:
                vehicles = []

        # Get all recalls
        recalls = self.list_recalls()
        if not recalls:
            return matches

        for v in vehicles:
            v_make = (v.get("make", "") if isinstance(v, dict) else getattr(v, "make", "")).lower()
            v_model = (v.get("model", "") if isinstance(v, dict) else getattr(v, "model", "")).lower()
            try:
                v_year = int(v.get("year", 0) if isinstance(v, dict) else getattr(v, "year", 0))
            except (ValueError, TypeError):
                v_year = 0
            v_id = v.get("vehicle_id", "") if isinstance(v, dict) else getattr(v, "vehicle_id", "")

            for recall in recalls:
                if (recall.make.lower() == v_make
                        and recall.model.lower() == v_model
                        and recall.year_start <= v_year <= recall.year_end):
                    matches.append({
                        "vehicle": v if isinstance(v, dict) else v.to_dict(),
                        "recall": recall.to_dict(),
                        "customer_id": (v.get("owner_id", "") if isinstance(v, dict)
                                        else getattr(v, "owner_id", "")),
                        "vehicle_id": v_id,
                    })

        logger.info("Recall scan complete — %d match(es) found", len(matches))
        return matches

    # ==================================================================
    # EXPIRATION ALERTS
    # ==================================================================

    def alert_expiring(self, days: int = 90) -> list[WarrantyItem]:
        """Find warranties expiring within the given number of days.

        Returns list of WarrantyItem objects that are still active but
        have end_date within the window.  Commonly used with 30, 60, 90.
        """
        today = datetime.now(timezone.utc).date()
        cutoff = (today + timedelta(days=days)).isoformat()
        today_str = today.isoformat()

        rows = self._conn.execute(
            """SELECT * FROM warranties
               WHERE end_date != '' AND end_date >= ? AND end_date <= ?
               ORDER BY end_date ASC""",
            (today_str, cutoff),
        ).fetchall()

        items = [WarrantyItem.from_row(r) for r in rows]
        logger.info("Found %d warranty(ies) expiring within %d days", len(items), days)
        return items

    def notify_expiring_warranties(self, days: int = 30) -> int:
        """Send notifications for warranties expiring within the window.

        Uses NotificationManager for dashboard alerts.  Returns count of
        notifications sent.
        """
        expiring = self.alert_expiring(days)
        count = 0

        for w in expiring:
            # Dashboard notification
            if self._notifications:
                try:
                    self._notifications.emit(
                        level="warning",
                        title="Warranty Expiring",
                        message=(
                            f"{COVERAGE_LABELS.get(w.coverage_type, w.coverage_type)} "
                            f"for vehicle {w.vehicle_id[:8] or 'unknown'} "
                            f"({w.component or 'full vehicle'}) "
                            f"expires {w.end_date} ({w.days_remaining} days)"
                        ),
                        source="warranty",
                    )
                    count += 1
                except Exception:
                    logger.debug("Notification failed for warranty %s", w.short_id,
                                 exc_info=True)

        logger.info("Sent %d expiring warranty notification(s)", count)
        return count

    def notify_recall_matches(self, matches: list[dict[str, Any]]) -> int:
        """Send dashboard notifications for recall matches.

        Takes the output of scan_recalls(). Returns count of notifications sent.
        """
        count = 0
        for m in matches:
            recall = m.get("recall", {})
            vehicle = m.get("vehicle", {})
            customer_id = m.get("customer_id", "")

            if self._notifications:
                try:
                    self._notifications.emit(
                        level="alert",
                        title="Recall Match Found",
                        message=(
                            f"{vehicle.get('year', '')} {vehicle.get('make', '')} "
                            f"{vehicle.get('model', '')} — "
                            f"{recall.get('component', 'Unknown')}: "
                            f"{recall.get('description', '')[:120]}"
                        ),
                        source="warranty",
                    )
                    count += 1
                except Exception:
                    logger.debug("Notification failed for recall match", exc_info=True)

            # Record the alert
            try:
                self._conn.execute(
                    """INSERT INTO recall_alerts
                       (alert_id, recall_id, customer_id, vehicle_id, notified_at, method)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (str(uuid.uuid4()), recall.get("recall_id", ""),
                     customer_id, m.get("vehicle_id", ""),
                     self._now(), "dashboard"),
                )
                self._conn.commit()
            except Exception:
                logger.debug("Failed to record recall alert", exc_info=True)

        logger.info("Sent %d recall match notification(s)", count)
        return count

    # ==================================================================
    # STATUS & BROADCAST
    # ==================================================================

    def get_status(self) -> dict[str, Any]:
        """Aggregate stats for the dashboard."""
        cur = self._conn.cursor()

        total = cur.execute("SELECT COUNT(*) FROM warranties").fetchone()[0]
        total_recalls = cur.execute("SELECT COUNT(*) FROM recalls").fetchone()[0]

        # Count active/expiring/expired by querying dates
        today = datetime.now(timezone.utc).date().isoformat()
        cutoff_90 = (datetime.now(timezone.utc).date() + timedelta(days=90)).isoformat()

        active = cur.execute(
            "SELECT COUNT(*) FROM warranties WHERE end_date = '' OR end_date > ?",
            (cutoff_90,),
        ).fetchone()[0]

        expiring = cur.execute(
            """SELECT COUNT(*) FROM warranties
               WHERE end_date != '' AND end_date >= ? AND end_date <= ?""",
            (today, cutoff_90),
        ).fetchone()[0]

        expired = cur.execute(
            "SELECT COUNT(*) FROM warranties WHERE end_date != '' AND end_date < ?",
            (today,),
        ).fetchone()[0]

        # Alerts sent
        alerts_sent = cur.execute("SELECT COUNT(*) FROM recall_alerts").fetchone()[0]

        # Coverage types breakdown
        type_rows = cur.execute(
            "SELECT coverage_type, COUNT(*) as cnt FROM warranties GROUP BY coverage_type"
        ).fetchall()
        by_type = {r["coverage_type"]: r["cnt"] for r in type_rows}

        return {
            "total_warranties": total,
            "active": active,
            "expiring_soon": expiring,
            "expired": expired,
            "total_recalls": total_recalls,
            "alerts_sent": alerts_sent,
            "by_type": by_type,
        }

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Full state snapshot for WebSocket broadcast."""
        return {
            "warranties": [w.to_dict() for w in self.list_warranties()],
            "recalls": [r.to_dict() for r in self.list_recalls()],
            "expiring": [w.to_dict() for w in self.alert_expiring(90)],
            "status": self.get_status(),
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
                logger.info("WarrantyRecallManager closed")
            except Exception:
                pass
