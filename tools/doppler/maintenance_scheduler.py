"""
Recurring Maintenance Scheduler for CAM / Doppler Cycles.

Manages preventive maintenance schedules for customer vehicles.  Defines
standard maintenance intervals by make/model (oil changes, valve adjustments,
brake fluid, coolant, chain/belt service, tire replacement, fork service),
tracks each vehicle's current mileage and last-service dates, and calculates
which services are due or upcoming.

Key capabilities:
  - calculate_due()          — what's due/upcoming for a specific vehicle
  - generate_reminders()     — customer notifications for upcoming maintenance
  - maintenance_forecast()   — projected work across all customers for 30/60/90 days

This is proactive revenue generation: instead of waiting for customers to call
with problems, CAM reminds them before issues arise.

Integrates with:
  - ServiceRecordStore   (vehicle data, service history)
  - CRMStore             (customer lookups)
  - NotificationManager  (dashboard alerts)

SQLite-backed, single-file module — same pattern as warranty.py, feedback.py.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("cam.maintenance_scheduler")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard motorcycle maintenance service types
SERVICE_TYPES = (
    "oil_change",
    "valve_adjustment",
    "brake_fluid",
    "coolant",
    "chain_service",
    "belt_service",
    "tire_replacement",
    "fork_service",
    "spark_plugs",
    "air_filter",
    "brake_pads",
    "final_drive",
)

SERVICE_LABELS = {
    "oil_change": "Oil Change",
    "valve_adjustment": "Valve Adjustment",
    "brake_fluid": "Brake Fluid Flush",
    "coolant": "Coolant Flush",
    "chain_service": "Chain Service",
    "belt_service": "Belt Service",
    "tire_replacement": "Tire Replacement",
    "fork_service": "Fork Service",
    "spark_plugs": "Spark Plugs",
    "air_filter": "Air Filter",
    "brake_pads": "Brake Pads",
    "final_drive": "Final Drive Service",
}

# Estimated labor hours and typical revenue per service type
# Used for revenue forecasting — can be overridden per schedule
SERVICE_ESTIMATES = {
    "oil_change":        {"labor_hours": 0.5, "parts_est": 35,  "labor_rate": 75},
    "valve_adjustment":  {"labor_hours": 2.0, "parts_est": 15,  "labor_rate": 75},
    "brake_fluid":       {"labor_hours": 0.75, "parts_est": 20, "labor_rate": 75},
    "coolant":           {"labor_hours": 0.75, "parts_est": 25, "labor_rate": 75},
    "chain_service":     {"labor_hours": 0.5, "parts_est": 15,  "labor_rate": 75},
    "belt_service":      {"labor_hours": 1.5, "parts_est": 80,  "labor_rate": 75},
    "tire_replacement":  {"labor_hours": 1.0, "parts_est": 200, "labor_rate": 75},
    "fork_service":      {"labor_hours": 2.5, "parts_est": 40,  "labor_rate": 75},
    "spark_plugs":       {"labor_hours": 0.5, "parts_est": 25,  "labor_rate": 75},
    "air_filter":        {"labor_hours": 0.25, "parts_est": 20, "labor_rate": 75},
    "brake_pads":        {"labor_hours": 1.0, "parts_est": 60,  "labor_rate": 75},
    "final_drive":       {"labor_hours": 1.0, "parts_est": 30,  "labor_rate": 75},
}

# Default intervals (miles) — generic motorcycle defaults.
# Make/model specific schedules override these.
DEFAULT_INTERVALS = {
    "oil_change":        3000,
    "valve_adjustment":  15000,
    "brake_fluid":       12000,
    "coolant":           24000,
    "chain_service":     3000,
    "belt_service":      50000,
    "tire_replacement":  12000,
    "fork_service":      25000,
    "spark_plugs":       12000,
    "air_filter":        12000,
    "brake_pads":        12000,
    "final_drive":       24000,
}

# Common make/model overrides — miles between services.
# Keys are "make|model" lowercase.  Missing services fall back to defaults.
MAKE_MODEL_INTERVALS = {
    "harley-davidson|*": {
        "oil_change": 5000,
        "valve_adjustment": 0,  # 0 = not applicable (hydraulic lifters)
        "chain_service": 0,     # belt drive
        "belt_service": 50000,
        "final_drive": 0,       # belt, no shaft/chain final drive
    },
    "ducati|*": {
        "oil_change": 7500,
        "valve_adjustment": 7500,  # desmo valves — tighter intervals
        "belt_service": 15000,     # timing belts
        "chain_service": 3000,
    },
    "honda|*": {
        "oil_change": 4000,
        "valve_adjustment": 16000,
    },
    "yamaha|*": {
        "oil_change": 4000,
        "valve_adjustment": 26600,
    },
    "suzuki|*": {
        "oil_change": 3700,
        "valve_adjustment": 14500,
    },
    "kawasaki|*": {
        "oil_change": 4000,
        "valve_adjustment": 15000,
    },
    "bmw|*": {
        "oil_change": 6000,
        "valve_adjustment": 12000,
        "chain_service": 0,     # shaft drive
        "belt_service": 0,
        "final_drive": 24000,   # shaft drive service
    },
}


def _get_intervals(make: str, model: str) -> dict[str, int]:
    """Resolve maintenance intervals for a make/model.

    Checks make|model first, then make|* wildcard, then defaults.
    Returns dict of service_type -> miles_interval (0 = not applicable).
    """
    intervals = dict(DEFAULT_INTERVALS)

    # Check make|* wildcard
    key_wild = f"{make.lower()}|*"
    if key_wild in MAKE_MODEL_INTERVALS:
        intervals.update(MAKE_MODEL_INTERVALS[key_wild])

    # Check make|model specific
    key_exact = f"{make.lower()}|{model.lower()}"
    if key_exact in MAKE_MODEL_INTERVALS:
        intervals.update(MAKE_MODEL_INTERVALS[key_exact])

    return intervals


def _estimate_revenue(service_type: str) -> float:
    """Estimate revenue for a service type (labor + parts)."""
    est = SERVICE_ESTIMATES.get(service_type, {})
    labor = est.get("labor_hours", 1.0) * est.get("labor_rate", 75)
    parts = est.get("parts_est", 0)
    return labor + parts


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MaintenanceSchedule:
    """A maintenance schedule entry for a specific vehicle + service type.

    Attributes:
        schedule_id:       Unique identifier (UUID string).
        vehicle_id:        Link to ServiceRecordStore vehicle.
        customer_id:       Link to CRM customer.
        service_type:      One of SERVICE_TYPES.
        interval_miles:    Miles between services.
        interval_months:   Months between services (0 = mileage only).
        last_service_date: When last performed (YYYY-MM-DD or '').
        last_service_miles: Mileage at last service.
        current_mileage:   Vehicle's current mileage reading.
        notes:             Free-text notes.
        enabled:           Whether this schedule is active.
        created_at:        Record creation timestamp (ISO-8601).
        updated_at:        Last modification timestamp (ISO-8601).
    """
    schedule_id: str
    vehicle_id: str = ""
    customer_id: str = ""
    service_type: str = "oil_change"
    interval_miles: int = 3000
    interval_months: int = 0
    last_service_date: str = ""
    last_service_miles: int = 0
    current_mileage: int = 0
    notes: str = ""
    enabled: int = 1
    created_at: str = ""
    updated_at: str = ""

    @property
    def short_id(self) -> str:
        return self.schedule_id[:8]

    @property
    def label(self) -> str:
        return SERVICE_LABELS.get(self.service_type, self.service_type)

    @property
    def miles_since_service(self) -> int:
        """Miles driven since the last service."""
        if self.current_mileage > 0 and self.last_service_miles > 0:
            return max(0, self.current_mileage - self.last_service_miles)
        return 0

    @property
    def miles_until_due(self) -> int:
        """Miles remaining until service is due.  Negative = overdue."""
        if self.interval_miles <= 0:
            return 999999  # not applicable
        return self.interval_miles - self.miles_since_service

    @property
    def due_status(self) -> str:
        """'overdue', 'due_soon' (within 500 mi), 'upcoming', or 'ok'."""
        remaining = self.miles_until_due
        if remaining <= 0:
            return "overdue"
        if remaining <= 500:
            return "due_soon"
        if remaining <= self.interval_miles * 0.25:
            return "upcoming"
        return "ok"

    @property
    def months_since_service(self) -> Optional[int]:
        """Months since last service, or None if no date recorded."""
        if not self.last_service_date:
            return None
        try:
            last = datetime.strptime(self.last_service_date, "%Y-%m-%d").date()
        except ValueError:
            return None
        today = datetime.now(timezone.utc).date()
        return (today.year - last.year) * 12 + (today.month - last.month)

    @property
    def time_overdue(self) -> bool:
        """Whether the schedule is overdue by time (if interval_months > 0)."""
        if self.interval_months <= 0:
            return False
        months = self.months_since_service
        if months is None:
            return False
        return months >= self.interval_months

    @property
    def estimated_revenue(self) -> float:
        return _estimate_revenue(self.service_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "vehicle_id": self.vehicle_id,
            "customer_id": self.customer_id,
            "service_type": self.service_type,
            "label": self.label,
            "interval_miles": self.interval_miles,
            "interval_months": self.interval_months,
            "last_service_date": self.last_service_date,
            "last_service_miles": self.last_service_miles,
            "current_mileage": self.current_mileage,
            "miles_since_service": self.miles_since_service,
            "miles_until_due": self.miles_until_due,
            "due_status": self.due_status,
            "months_since_service": self.months_since_service,
            "time_overdue": self.time_overdue,
            "estimated_revenue": self.estimated_revenue,
            "notes": self.notes,
            "enabled": bool(self.enabled),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_row(row: sqlite3.Row) -> "MaintenanceSchedule":
        r = dict(row)
        return MaintenanceSchedule(
            schedule_id=r.get("schedule_id", ""),
            vehicle_id=r.get("vehicle_id", ""),
            customer_id=r.get("customer_id", ""),
            service_type=r.get("service_type", "oil_change"),
            interval_miles=int(r.get("interval_miles", 3000)),
            interval_months=int(r.get("interval_months", 0)),
            last_service_date=r.get("last_service_date", ""),
            last_service_miles=int(r.get("last_service_miles", 0)),
            current_mileage=int(r.get("current_mileage", 0)),
            notes=r.get("notes", ""),
            enabled=int(r.get("enabled", 1)),
            created_at=r.get("created_at", ""),
            updated_at=r.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class MaintenanceSchedulerManager:
    """Manages preventive maintenance schedules with SQLite persistence.

    Tracks maintenance intervals by make/model, calculates what's due,
    generates customer reminders, and forecasts projected work/revenue
    for the next 30/60/90 days.
    """

    def __init__(
        self,
        db_path: str = "data/maintenance_scheduler.db",
        *,
        service_store: Any = None,
        crm_store: Any = None,
        notification_manager: Any = None,
        on_change: Optional[Callable[[], Coroutine]] = None,
    ):
        self._service_store = service_store
        self._crm = crm_store
        self._notifications = notification_manager
        self._on_change = on_change

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("MaintenanceSchedulerManager initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS maintenance_schedules (
                schedule_id       TEXT PRIMARY KEY,
                vehicle_id        TEXT DEFAULT '',
                customer_id       TEXT DEFAULT '',
                service_type      TEXT DEFAULT 'oil_change',
                interval_miles    INTEGER DEFAULT 3000,
                interval_months   INTEGER DEFAULT 0,
                last_service_date TEXT DEFAULT '',
                last_service_miles INTEGER DEFAULT 0,
                current_mileage   INTEGER DEFAULT 0,
                notes             TEXT DEFAULT '',
                enabled           INTEGER DEFAULT 1,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ms_vehicle  ON maintenance_schedules(vehicle_id);
            CREATE INDEX IF NOT EXISTS idx_ms_customer ON maintenance_schedules(customer_id);
            CREATE INDEX IF NOT EXISTS idx_ms_type     ON maintenance_schedules(service_type);
            CREATE INDEX IF NOT EXISTS idx_ms_enabled  ON maintenance_schedules(enabled);

            CREATE TABLE IF NOT EXISTS reminder_log (
                reminder_id  TEXT PRIMARY KEY,
                schedule_id  TEXT NOT NULL,
                customer_id  TEXT DEFAULT '',
                vehicle_id   TEXT DEFAULT '',
                service_type TEXT DEFAULT '',
                sent_at      TEXT NOT NULL,
                method       TEXT DEFAULT 'dashboard',
                FOREIGN KEY (schedule_id) REFERENCES maintenance_schedules(schedule_id)
            );
            CREATE INDEX IF NOT EXISTS idx_rem_schedule ON reminder_log(schedule_id);
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
        if self._on_change is None:
            return
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._on_change())
        except RuntimeError:
            pass

    # ==================================================================
    # SCHEDULE CRUD
    # ==================================================================

    def add_schedule(
        self,
        vehicle_id: str = "",
        customer_id: str = "",
        service_type: str = "oil_change",
        interval_miles: int = 0,
        interval_months: int = 0,
        last_service_date: str = "",
        last_service_miles: int = 0,
        current_mileage: int = 0,
        notes: str = "",
    ) -> MaintenanceSchedule:
        """Create a maintenance schedule for a vehicle + service type.

        If interval_miles is 0, attempts to look up the vehicle's make/model
        from the service store and use the appropriate default interval.
        """
        if service_type not in SERVICE_TYPES:
            service_type = "oil_change"

        # Auto-resolve interval from make/model if not specified
        if interval_miles <= 0:
            interval_miles = self._resolve_interval(vehicle_id, service_type)

        now = self._now()
        sid = str(uuid.uuid4())

        self._conn.execute(
            """INSERT INTO maintenance_schedules
               (schedule_id, vehicle_id, customer_id, service_type,
                interval_miles, interval_months, last_service_date,
                last_service_miles, current_mileage, notes, enabled,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (sid, vehicle_id, customer_id, service_type,
             interval_miles, interval_months, last_service_date,
             last_service_miles, current_mileage, notes, now, now),
        )
        self._conn.commit()

        sched = self.get_schedule(sid)
        logger.info("Schedule added %s — %s for vehicle %s (every %d mi)",
                     sched.short_id, service_type, vehicle_id[:8] if vehicle_id else "?",
                     interval_miles)
        self._fire_change()
        return sched

    def _resolve_interval(self, vehicle_id: str, service_type: str) -> int:
        """Look up the vehicle's make/model and return the correct interval."""
        if vehicle_id and self._service_store:
            try:
                vehicles = self._service_store.list_vehicles()
                for v in vehicles:
                    vid = v.vehicle_id if hasattr(v, "vehicle_id") else v.get("vehicle_id", "")
                    if vid == vehicle_id:
                        make = v.make if hasattr(v, "make") else v.get("make", "")
                        model = v.model if hasattr(v, "model") else v.get("model", "")
                        intervals = _get_intervals(make, model)
                        return intervals.get(service_type, DEFAULT_INTERVALS.get(service_type, 3000))
            except Exception:
                logger.debug("Could not resolve vehicle %s for intervals", vehicle_id[:8],
                             exc_info=True)
        return DEFAULT_INTERVALS.get(service_type, 3000)

    def update_schedule(self, schedule_id: str, **kwargs) -> Optional[MaintenanceSchedule]:
        """Update a maintenance schedule."""
        existing = self.get_schedule(schedule_id)
        if not existing:
            return None

        allowed = {
            "vehicle_id", "customer_id", "service_type", "interval_miles",
            "interval_months", "last_service_date", "last_service_miles",
            "current_mileage", "notes", "enabled",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return existing

        for int_field in ("interval_miles", "interval_months", "last_service_miles",
                          "current_mileage", "enabled"):
            if int_field in updates:
                updates[int_field] = int(updates[int_field])

        updates["updated_at"] = self._now()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [schedule_id]
        self._conn.execute(
            f"UPDATE maintenance_schedules SET {set_clause} WHERE schedule_id = ?", vals,
        )
        self._conn.commit()
        logger.info("Schedule updated %s", schedule_id[:8])
        self._fire_change()
        return self.get_schedule(schedule_id)

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a maintenance schedule."""
        cur = self._conn.execute(
            "DELETE FROM maintenance_schedules WHERE schedule_id = ?", (schedule_id,),
        )
        self._conn.commit()
        if cur.rowcount > 0:
            logger.info("Schedule deleted %s", schedule_id[:8])
            self._fire_change()
            return True
        return False

    def get_schedule(self, schedule_id: str) -> Optional[MaintenanceSchedule]:
        """Fetch a single schedule by ID."""
        cur = self._conn.execute(
            "SELECT * FROM maintenance_schedules WHERE schedule_id = ?", (schedule_id,),
        )
        row = cur.fetchone()
        return MaintenanceSchedule.from_row(row) if row else None

    def list_schedules(
        self,
        vehicle_id: str = "",
        customer_id: str = "",
        service_type: str = "",
        enabled_only: bool = True,
        limit: int = 500,
    ) -> list[MaintenanceSchedule]:
        """List schedules with optional filters."""
        query = "SELECT * FROM maintenance_schedules WHERE 1=1"
        params: list[Any] = []

        if vehicle_id:
            query += " AND vehicle_id = ?"
            params.append(vehicle_id)
        if customer_id:
            query += " AND customer_id = ?"
            params.append(customer_id)
        if service_type:
            query += " AND service_type = ?"
            params.append(service_type)
        if enabled_only:
            query += " AND enabled = 1"

        query += " ORDER BY vehicle_id, service_type LIMIT ?"
        params.append(limit)

        return [MaintenanceSchedule.from_row(r) for r in self._conn.execute(query, params)]

    def update_mileage(self, vehicle_id: str, mileage: int) -> int:
        """Update current_mileage for all schedules on a vehicle.

        Returns count of updated schedules.
        """
        now = self._now()
        cur = self._conn.execute(
            """UPDATE maintenance_schedules
               SET current_mileage = ?, updated_at = ?
               WHERE vehicle_id = ? AND enabled = 1""",
            (mileage, now, vehicle_id),
        )
        self._conn.commit()
        count = cur.rowcount
        if count > 0:
            logger.info("Updated mileage to %d for %d schedule(s) on vehicle %s",
                         mileage, count, vehicle_id[:8])
            self._fire_change()
        return count

    def record_service(
        self,
        vehicle_id: str,
        service_type: str,
        date: str = "",
        mileage: int = 0,
    ) -> Optional[MaintenanceSchedule]:
        """Record that a service was performed — updates last_service fields.

        Finds the matching schedule for this vehicle+service_type, updates
        last_service_date and last_service_miles, and optionally updates
        current_mileage.
        """
        rows = self._conn.execute(
            """SELECT * FROM maintenance_schedules
               WHERE vehicle_id = ? AND service_type = ? AND enabled = 1
               LIMIT 1""",
            (vehicle_id, service_type),
        ).fetchall()

        if not rows:
            return None

        sched = MaintenanceSchedule.from_row(rows[0])
        now = self._now()
        service_date = date or datetime.now(timezone.utc).date().isoformat()

        updates = {
            "last_service_date": service_date,
            "updated_at": now,
        }
        if mileage > 0:
            updates["last_service_miles"] = mileage
            updates["current_mileage"] = mileage

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [sched.schedule_id]
        self._conn.execute(
            f"UPDATE maintenance_schedules SET {set_clause} WHERE schedule_id = ?", vals,
        )
        self._conn.commit()
        logger.info("Recorded %s service for vehicle %s at %d mi",
                     service_type, vehicle_id[:8], mileage)
        self._fire_change()
        return self.get_schedule(sched.schedule_id)

    # ==================================================================
    # AUTO-SETUP: Create schedules for a vehicle from make/model defaults
    # ==================================================================

    def setup_vehicle(
        self,
        vehicle_id: str,
        customer_id: str = "",
        make: str = "",
        model: str = "",
        current_mileage: int = 0,
    ) -> list[MaintenanceSchedule]:
        """Auto-create maintenance schedules for a vehicle based on make/model.

        Skips service types with interval = 0 (not applicable for that make).
        Skips service types that already have a schedule for this vehicle.
        """
        intervals = _get_intervals(make, model)

        # Get existing schedules for this vehicle
        existing = {s.service_type for s in self.list_schedules(vehicle_id=vehicle_id)}

        created = []
        for stype, miles in intervals.items():
            if miles <= 0:
                continue  # not applicable for this make
            if stype in existing:
                continue  # already has a schedule

            sched = self.add_schedule(
                vehicle_id=vehicle_id,
                customer_id=customer_id,
                service_type=stype,
                interval_miles=miles,
                current_mileage=current_mileage,
            )
            created.append(sched)

        logger.info("Setup vehicle %s (%s %s) — created %d schedule(s)",
                     vehicle_id[:8], make, model, len(created))
        return created

    # ==================================================================
    # CALCULATE DUE
    # ==================================================================

    def calculate_due(
        self,
        vehicle_id: str = "",
        customer_id: str = "",
    ) -> list[dict[str, Any]]:
        """Determine which services are due or upcoming for a vehicle/customer.

        Returns list of dicts with schedule info + due_status, sorted by
        urgency (overdue first, then due_soon, upcoming, ok).
        """
        schedules = self.list_schedules(
            vehicle_id=vehicle_id,
            customer_id=customer_id,
        )

        results = []
        for s in schedules:
            if s.interval_miles <= 0:
                continue

            entry = s.to_dict()

            # Also check time-based overdue
            if s.time_overdue and entry["due_status"] == "ok":
                entry["due_status"] = "due_soon"
                entry["time_overdue"] = True

            results.append(entry)

        # Sort by urgency
        priority = {"overdue": 0, "due_soon": 1, "upcoming": 2, "ok": 3}
        results.sort(key=lambda x: (priority.get(x["due_status"], 4), x.get("miles_until_due", 999999)))

        return results

    # ==================================================================
    # GENERATE REMINDERS
    # ==================================================================

    def generate_reminders(self, include_upcoming: bool = True) -> list[dict[str, Any]]:
        """Generate customer notifications for maintenance that's due.

        Sends dashboard notifications via NotificationManager for overdue
        and due_soon items.  Returns list of reminder records created.

        Args:
            include_upcoming: Also notify for 'upcoming' items (default True).
        """
        schedules = self.list_schedules()
        reminders: list[dict[str, Any]] = []

        notify_statuses = {"overdue", "due_soon"}
        if include_upcoming:
            notify_statuses.add("upcoming")

        for s in schedules:
            if s.interval_miles <= 0:
                continue
            if s.due_status not in notify_statuses and not s.time_overdue:
                continue

            # Check if we already sent a reminder recently (within 7 days)
            recent = self._conn.execute(
                """SELECT COUNT(*) FROM reminder_log
                   WHERE schedule_id = ? AND sent_at > ?""",
                (s.schedule_id,
                 (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()),
            ).fetchone()[0]

            if recent > 0:
                continue

            # Send notification
            status_label = s.due_status.replace("_", " ").title()
            if s.time_overdue:
                status_label = "Time Overdue"

            message = (
                f"{s.label} for vehicle {s.vehicle_id[:8] if s.vehicle_id else 'unknown'} "
                f"— {status_label}"
            )
            if s.miles_until_due <= 0:
                message += f" (overdue by {abs(s.miles_until_due)} mi)"
            else:
                message += f" ({s.miles_until_due} mi remaining)"

            level = "alert" if s.due_status == "overdue" else "warning"

            if self._notifications:
                try:
                    self._notifications.emit(
                        level=level,
                        title="Maintenance Due",
                        message=message,
                        source="maintenance_scheduler",
                    )
                except Exception:
                    logger.debug("Notification failed for schedule %s", s.short_id,
                                 exc_info=True)

            # Log the reminder
            rid = str(uuid.uuid4())
            now = self._now()
            self._conn.execute(
                """INSERT INTO reminder_log
                   (reminder_id, schedule_id, customer_id, vehicle_id,
                    service_type, sent_at, method)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (rid, s.schedule_id, s.customer_id, s.vehicle_id,
                 s.service_type, now, "dashboard"),
            )
            self._conn.commit()

            reminders.append({
                "reminder_id": rid,
                "schedule_id": s.schedule_id,
                "customer_id": s.customer_id,
                "vehicle_id": s.vehicle_id,
                "service_type": s.service_type,
                "label": s.label,
                "due_status": s.due_status,
                "miles_until_due": s.miles_until_due,
                "estimated_revenue": s.estimated_revenue,
            })

        logger.info("Generated %d maintenance reminder(s)", len(reminders))
        return reminders

    # ==================================================================
    # MAINTENANCE FORECAST
    # ==================================================================

    def maintenance_forecast(self, days: int = 90) -> dict[str, Any]:
        """Project maintenance work across all customers for the next N days.

        Estimates which services will come due based on average daily mileage
        (estimated from last_service data).  Groups by 30-day windows.

        Returns:
            {
                "period_days": 90,
                "total_services": int,
                "total_estimated_revenue": float,
                "windows": [
                    {"label": "0-30 days", "services": [...], "revenue": float},
                    {"label": "31-60 days", ...},
                    {"label": "61-90 days", ...},
                ],
                "by_service_type": {service_type: {"count": int, "revenue": float}},
                "by_customer": {customer_id: {"count": int, "revenue": float}},
            }
        """
        schedules = self.list_schedules()

        # Default average daily mileage if we can't calculate
        default_daily_miles = 30  # ~11,000 miles/year

        # Build forecast windows
        window_size = 30
        num_windows = max(1, days // window_size)
        windows = []
        for i in range(num_windows):
            windows.append({
                "label": f"{i * window_size + (1 if i > 0 else 0)}-{(i + 1) * window_size} days",
                "services": [],
                "revenue": 0.0,
            })

        by_service = {}
        by_customer = {}
        total_services = 0
        total_revenue = 0.0

        for s in schedules:
            if s.interval_miles <= 0:
                continue

            # Estimate daily mileage for this vehicle
            daily_miles = default_daily_miles
            if s.last_service_date and s.last_service_miles > 0 and s.current_mileage > 0:
                try:
                    last_dt = datetime.strptime(s.last_service_date, "%Y-%m-%d").date()
                    days_since = (datetime.now(timezone.utc).date() - last_dt).days
                    if days_since > 7:
                        miles_since = s.current_mileage - s.last_service_miles
                        daily_miles = max(5, miles_since / days_since)
                except (ValueError, ZeroDivisionError):
                    pass

            # Project when the service will be due
            remaining = s.miles_until_due
            if remaining <= 0:
                # Already overdue — goes in window 0
                days_until_due = 0
            else:
                days_until_due = remaining / daily_miles if daily_miles > 0 else 999

            if days_until_due > days:
                continue  # Outside forecast window

            # Determine which window
            window_idx = min(int(days_until_due / window_size), num_windows - 1)

            entry = {
                "schedule_id": s.schedule_id,
                "vehicle_id": s.vehicle_id,
                "customer_id": s.customer_id,
                "service_type": s.service_type,
                "label": s.label,
                "estimated_days": round(days_until_due),
                "estimated_revenue": s.estimated_revenue,
                "due_status": s.due_status,
            }

            windows[window_idx]["services"].append(entry)
            windows[window_idx]["revenue"] += s.estimated_revenue

            # Aggregate by service type
            if s.service_type not in by_service:
                by_service[s.service_type] = {"count": 0, "revenue": 0.0, "label": s.label}
            by_service[s.service_type]["count"] += 1
            by_service[s.service_type]["revenue"] += s.estimated_revenue

            # Aggregate by customer
            cid = s.customer_id or "unknown"
            if cid not in by_customer:
                by_customer[cid] = {"count": 0, "revenue": 0.0}
            by_customer[cid]["count"] += 1
            by_customer[cid]["revenue"] += s.estimated_revenue

            total_services += 1
            total_revenue += s.estimated_revenue

        return {
            "period_days": days,
            "total_services": total_services,
            "total_estimated_revenue": round(total_revenue, 2),
            "windows": windows,
            "by_service_type": by_service,
            "by_customer": by_customer,
        }

    # ==================================================================
    # STATUS & BROADCAST
    # ==================================================================

    def get_status(self) -> dict[str, Any]:
        """Aggregate stats for the dashboard."""
        schedules = self.list_schedules()

        overdue = sum(1 for s in schedules if s.due_status == "overdue" or s.time_overdue)
        due_soon = sum(1 for s in schedules if s.due_status == "due_soon")
        upcoming = sum(1 for s in schedules if s.due_status == "upcoming")
        total_vehicles = len(set(s.vehicle_id for s in schedules if s.vehicle_id))

        # Estimated revenue for overdue + due_soon
        actionable_revenue = sum(
            s.estimated_revenue for s in schedules
            if s.due_status in ("overdue", "due_soon") or s.time_overdue
        )

        reminders_sent = self._conn.execute(
            "SELECT COUNT(*) FROM reminder_log"
        ).fetchone()[0]

        return {
            "total_schedules": len(schedules),
            "total_vehicles": total_vehicles,
            "overdue": overdue,
            "due_soon": due_soon,
            "upcoming": upcoming,
            "actionable_revenue": round(actionable_revenue, 2),
            "reminders_sent": reminders_sent,
        }

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Full state snapshot for WebSocket broadcast."""
        due = self.calculate_due()
        forecast = self.maintenance_forecast(90)
        return {
            "schedules": [s.to_dict() for s in self.list_schedules()],
            "due": due,
            "forecast": forecast,
            "status": self.get_status(),
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
                logger.info("MaintenanceSchedulerManager closed")
            except Exception:
                pass
