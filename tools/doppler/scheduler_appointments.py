"""Appointment Scheduler for Doppler Cycles mobile diagnostics.

Manages scheduled appointments with GPS-based travel time estimation,
availability checking, automatic reminders, and calendar views (day/week).

Storage: ``data/appointments.db`` (separate from business.db).
"""

import json
import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ServiceType(str, Enum):
    diagnostic = "diagnostic"
    routine_maintenance = "routine_maintenance"
    repair = "repair"
    consultation = "consultation"
    pickup_delivery = "pickup_delivery"


class ScheduledAppointmentStatus(str, Enum):
    scheduled = "scheduled"
    confirmed = "confirmed"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"
    no_show = "no_show"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ScheduledAppointment:
    """A scheduled appointment for Doppler Cycles mobile diagnostics."""

    appointment_id: str
    customer_id: str
    customer_name: str
    vehicle_id: str = ""
    vehicle_summary: str = ""
    date: str = ""  # YYYY-MM-DD
    time_slot: str = ""  # HH:MM
    duration_estimate: int = 60  # minutes
    location_address: str = ""
    location_lat: float = 0.0
    location_lon: float = 0.0
    service_type: str = "diagnostic"
    status: str = "scheduled"
    estimated_cost: float = 0.0
    notes: str = ""
    reminder_sent: int = 0  # bitmask: 1=24hr, 2=1hr
    travel_minutes_from_prev: float = 0.0  # computed, not stored
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.appointment_id[:8]

    def to_dict(self) -> dict[str, Any]:
        return {
            "appointment_id": self.appointment_id,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "vehicle_id": self.vehicle_id,
            "vehicle_summary": self.vehicle_summary,
            "date": self.date,
            "time_slot": self.time_slot,
            "duration_estimate": self.duration_estimate,
            "location_address": self.location_address,
            "location_lat": self.location_lat,
            "location_lon": self.location_lon,
            "service_type": self.service_type,
            "status": self.status,
            "estimated_cost": self.estimated_cost,
            "notes": self.notes,
            "reminder_sent": self.reminder_sent,
            "travel_minutes_from_prev": self.travel_minutes_from_prev,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "short_id": self.short_id,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ScheduledAppointment":
        """Create from a SQLite Row (does NOT set travel_minutes_from_prev)."""
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return cls(
            appointment_id=row["appointment_id"],
            customer_id=row["customer_id"],
            customer_name=row["customer_name"],
            vehicle_id=row["vehicle_id"] or "",
            vehicle_summary=row["vehicle_summary"] or "",
            date=row["date"],
            time_slot=row["time_slot"],
            duration_estimate=row["duration_estimate"],
            location_address=row["location_address"] or "",
            location_lat=row["location_lat"] or 0.0,
            location_lon=row["location_lon"] or 0.0,
            service_type=row["service_type"] or "diagnostic",
            status=row["status"] or "scheduled",
            estimated_cost=row["estimated_cost"] or 0.0,
            notes=row["notes"] or "",
            reminder_sent=row["reminder_sent"] or 0,
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# AppointmentScheduler
# ---------------------------------------------------------------------------


class AppointmentScheduler:
    """Manages scheduled appointments with travel time and reminders.

    Args:
        db_path: Path to the SQLite database file.
        on_change: Async callback fired after any mutation (for WS broadcast).
        home_lat: Home base latitude (Gresham, OR default).
        home_lon: Home base longitude.
        avg_speed_mph: Average travel speed for time estimates.
        road_factor: Multiplier on haversine distance to approximate road distance.
    """

    def __init__(
        self,
        db_path: str = "data/appointments.db",
        on_change: Callable[[], Coroutine] | None = None,
        home_lat: float = 45.4976,
        home_lon: float = -122.4302,
        avg_speed_mph: float = 30.0,
        road_factor: float = 1.4,
    ):
        self.db_path = db_path
        self._on_change = on_change
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.avg_speed_mph = avg_speed_mph
        self.road_factor = road_factor

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("AppointmentScheduler initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS scheduled_appointments (
                appointment_id   TEXT PRIMARY KEY,
                customer_id      TEXT NOT NULL,
                customer_name    TEXT NOT NULL DEFAULT '',
                vehicle_id       TEXT DEFAULT '',
                vehicle_summary  TEXT DEFAULT '',
                date             TEXT NOT NULL,
                time_slot        TEXT NOT NULL,
                duration_estimate INTEGER DEFAULT 60,
                location_address TEXT DEFAULT '',
                location_lat     REAL DEFAULT 0.0,
                location_lon     REAL DEFAULT 0.0,
                service_type     TEXT DEFAULT 'diagnostic',
                status           TEXT DEFAULT 'scheduled',
                estimated_cost   REAL DEFAULT 0.0,
                notes            TEXT DEFAULT '',
                reminder_sent    INTEGER DEFAULT 0,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                metadata         TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_appt_date
                ON scheduled_appointments(date);
            CREATE INDEX IF NOT EXISTS idx_appt_status
                ON scheduled_appointments(status);
            CREATE INDEX IF NOT EXISTS idx_appt_customer
                ON scheduled_appointments(customer_id);
            CREATE INDEX IF NOT EXISTS idx_appt_reminder
                ON scheduled_appointments(reminder_sent, status, date);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Travel time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Return straight-line distance in miles between two coordinates."""
        if (lat1 == 0.0 and lon1 == 0.0) or (lat2 == 0.0 and lon2 == 0.0):
            return 0.0
        R = 3958.8  # Earth radius in miles
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def estimate_travel_minutes(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Estimated travel time in minutes between two points."""
        miles = self.haversine_distance(lat1, lon1, lat2, lon2)
        if miles == 0.0:
            return 0.0
        road_miles = miles * self.road_factor
        return (road_miles / self.avg_speed_mph) * 60.0

    def _compute_travel_for_day(
        self, appointments: list[ScheduledAppointment]
    ) -> list[ScheduledAppointment]:
        """Add travel_minutes_from_prev to an ordered list of appointments.

        First appointment travels from home base. Each subsequent appointment
        travels from the previous one's location.
        """
        prev_lat, prev_lon = self.home_lat, self.home_lon
        for appt in appointments:
            appt.travel_minutes_from_prev = self.estimate_travel_minutes(
                prev_lat, prev_lon, appt.location_lat, appt.location_lon
            )
            if appt.location_lat != 0.0 or appt.location_lon != 0.0:
                prev_lat, prev_lon = appt.location_lat, appt.location_lon
        return appointments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _row_to_appt(self, row: sqlite3.Row | None) -> ScheduledAppointment | None:
        if row is None:
            return None
        return ScheduledAppointment.from_row(row)

    async def _notify(self):
        """Fire the on_change callback if set."""
        if self._on_change:
            try:
                await self._on_change()
            except Exception:
                logger.exception("AppointmentScheduler on_change callback failed")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def book(
        self,
        customer_id: str,
        customer_name: str,
        vehicle_id: str = "",
        vehicle_summary: str = "",
        date: str = "",
        time_slot: str = "",
        duration_estimate: int = 60,
        location_address: str = "",
        location_lat: float = 0.0,
        location_lon: float = 0.0,
        service_type: str = "diagnostic",
        estimated_cost: float = 0.0,
        notes: str = "",
        metadata: dict | None = None,
    ) -> ScheduledAppointment:
        """Create a new scheduled appointment."""
        now = self._now_iso()
        appt_id = str(uuid.uuid4())
        meta_json = json.dumps(metadata or {})

        self._conn.execute(
            """INSERT INTO scheduled_appointments
               (appointment_id, customer_id, customer_name, vehicle_id,
                vehicle_summary, date, time_slot, duration_estimate,
                location_address, location_lat, location_lon,
                service_type, status, estimated_cost, notes,
                reminder_sent, created_at, updated_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?)""",
            (
                appt_id, customer_id, customer_name, vehicle_id,
                vehicle_summary, date, time_slot, duration_estimate,
                location_address, location_lat, location_lon,
                service_type, "scheduled", estimated_cost, notes,
                now, now, meta_json,
            ),
        )
        self._conn.commit()
        logger.info("Booked appointment %s for %s on %s %s", appt_id[:8], customer_name, date, time_slot)
        return ScheduledAppointment(
            appointment_id=appt_id,
            customer_id=customer_id,
            customer_name=customer_name,
            vehicle_id=vehicle_id,
            vehicle_summary=vehicle_summary,
            date=date,
            time_slot=time_slot,
            duration_estimate=duration_estimate,
            location_address=location_address,
            location_lat=location_lat,
            location_lon=location_lon,
            service_type=service_type,
            status="scheduled",
            estimated_cost=estimated_cost,
            notes=notes,
            reminder_sent=0,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

    def reschedule(
        self,
        appointment_id: str,
        new_date: str | None = None,
        new_time_slot: str | None = None,
        new_duration: int | None = None,
    ) -> ScheduledAppointment | None:
        """Reschedule an appointment. Resets reminder_sent to 0."""
        row = self._conn.execute(
            "SELECT * FROM scheduled_appointments WHERE appointment_id = ?",
            (appointment_id,),
        ).fetchone()
        if row is None:
            return None

        now = self._now_iso()
        updates = {"updated_at": now, "reminder_sent": 0}
        if new_date is not None:
            updates["date"] = new_date
        if new_time_slot is not None:
            updates["time_slot"] = new_time_slot
        if new_duration is not None:
            updates["duration_estimate"] = new_duration

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [appointment_id]
        self._conn.execute(
            f"UPDATE scheduled_appointments SET {set_clause} WHERE appointment_id = ?",
            vals,
        )
        self._conn.commit()
        logger.info("Rescheduled appointment %s", appointment_id[:8])
        return self.get_appointment(appointment_id)

    def cancel(self, appointment_id: str, reason: str = "") -> ScheduledAppointment | None:
        """Cancel an appointment."""
        row = self._conn.execute(
            "SELECT * FROM scheduled_appointments WHERE appointment_id = ?",
            (appointment_id,),
        ).fetchone()
        if row is None:
            return None

        now = self._now_iso()
        # Append cancel reason to notes if provided
        current_notes = row["notes"] or ""
        if reason:
            current_notes = f"{current_notes}\n[Cancelled: {reason}]".strip()

        self._conn.execute(
            """UPDATE scheduled_appointments
               SET status = 'cancelled', notes = ?, updated_at = ?
               WHERE appointment_id = ?""",
            (current_notes, now, appointment_id),
        )
        self._conn.commit()
        logger.info("Cancelled appointment %s", appointment_id[:8])
        return self.get_appointment(appointment_id)

    def update_appointment(
        self, appointment_id: str, **kwargs: Any
    ) -> ScheduledAppointment | None:
        """Update arbitrary fields on an appointment."""
        allowed = {
            "customer_id", "customer_name", "vehicle_id", "vehicle_summary",
            "date", "time_slot", "duration_estimate", "location_address",
            "location_lat", "location_lon", "service_type", "status",
            "estimated_cost", "notes", "metadata",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_appointment(appointment_id)

        # Serialize metadata if present
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])

        updates["updated_at"] = self._now_iso()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [appointment_id]
        cur = self._conn.execute(
            f"UPDATE scheduled_appointments SET {set_clause} WHERE appointment_id = ?",
            vals,
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return None
        logger.info("Updated appointment %s", appointment_id[:8])
        return self.get_appointment(appointment_id)

    def get_appointment(self, appointment_id: str) -> ScheduledAppointment | None:
        """Fetch a single appointment by ID."""
        row = self._conn.execute(
            "SELECT * FROM scheduled_appointments WHERE appointment_id = ?",
            (appointment_id,),
        ).fetchone()
        return self._row_to_appt(row)

    def remove_appointment(self, appointment_id: str) -> bool:
        """Permanently delete an appointment."""
        cur = self._conn.execute(
            "DELETE FROM scheduled_appointments WHERE appointment_id = ?",
            (appointment_id,),
        )
        self._conn.commit()
        removed = cur.rowcount > 0
        if removed:
            logger.info("Deleted appointment %s", appointment_id[:8])
        return removed

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_day_schedule(
        self, target_date: str, include_cancelled: bool = False
    ) -> list[ScheduledAppointment]:
        """Return appointments for a given date, ordered by time_slot."""
        if include_cancelled:
            rows = self._conn.execute(
                "SELECT * FROM scheduled_appointments WHERE date = ? ORDER BY time_slot",
                (target_date,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM scheduled_appointments
                   WHERE date = ? AND status != 'cancelled'
                   ORDER BY time_slot""",
                (target_date,),
            ).fetchall()
        appts = [ScheduledAppointment.from_row(r) for r in rows]
        return self._compute_travel_for_day(appts)

    def get_week_view(self, start_date: str) -> dict[str, list[dict]]:
        """Return 7 days of appointments keyed by YYYY-MM-DD.

        Each day's appointments include computed travel times.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        result = {}
        for offset in range(7):
            day = start + timedelta(days=offset)
            day_str = day.isoformat()
            appts = self.get_day_schedule(day_str)
            result[day_str] = [a.to_dict() for a in appts]
        return result

    def check_availability(
        self, target_date: str, time_slot: str, duration_minutes: int = 60
    ) -> dict:
        """Check if a time slot is available and suggest alternatives if not.

        Returns dict with 'available' (bool), 'conflicts' (list), and
        'suggested_times' (list of HH:MM strings).
        """
        # Parse requested window
        req_start = self._parse_time(time_slot)
        req_end = req_start + timedelta(minutes=duration_minutes)

        # Get all non-cancelled appointments for the day
        rows = self._conn.execute(
            """SELECT * FROM scheduled_appointments
               WHERE date = ? AND status NOT IN ('cancelled', 'no_show')
               ORDER BY time_slot""",
            (target_date,),
        ).fetchall()

        conflicts = []
        busy_windows = []
        for row in rows:
            appt = ScheduledAppointment.from_row(row)
            appt_start = self._parse_time(appt.time_slot)
            appt_end = appt_start + timedelta(minutes=appt.duration_estimate)
            busy_windows.append((appt_start, appt_end))

            # Check overlap
            if req_start < appt_end and req_end > appt_start:
                conflicts.append(appt.to_dict())

        # Suggest available times (8:00-18:00 in 30-min increments)
        suggested = []
        if conflicts:
            base_date = datetime.strptime(target_date, "%Y-%m-%d")
            for hour in range(8, 18):
                for minute in (0, 30):
                    candidate_start = base_date.replace(hour=hour, minute=minute)
                    candidate_end = candidate_start + timedelta(minutes=duration_minutes)
                    if candidate_end.hour > 18:
                        continue
                    overlap = any(
                        candidate_start < be and candidate_end > bs
                        for bs, be in busy_windows
                    )
                    if not overlap:
                        suggested.append(f"{hour:02d}:{minute:02d}")
            suggested = suggested[:6]  # Limit to 6 suggestions

        return {
            "available": len(conflicts) == 0,
            "conflicts": conflicts,
            "suggested_times": suggested,
        }

    def list_appointments(
        self, limit: int = 100, status_filter: str | None = None
    ) -> list[ScheduledAppointment]:
        """Return recent appointments, optionally filtered by status."""
        if status_filter:
            rows = self._conn.execute(
                """SELECT * FROM scheduled_appointments
                   WHERE status = ?
                   ORDER BY date DESC, time_slot DESC LIMIT ?""",
                (status_filter, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM scheduled_appointments
                   ORDER BY date DESC, time_slot DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [ScheduledAppointment.from_row(r) for r in rows]

    def get_customer_appointments(
        self, customer_id: str
    ) -> list[ScheduledAppointment]:
        """Return all appointments for a given customer."""
        rows = self._conn.execute(
            """SELECT * FROM scheduled_appointments
               WHERE customer_id = ?
               ORDER BY date DESC, time_slot DESC""",
            (customer_id,),
        ).fetchall()
        return [ScheduledAppointment.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------

    def get_pending_reminders(self) -> list[dict]:
        """Find appointments needing 24hr or 1hr reminders.

        Scans SCHEDULED/CONFIRMED appointments within the next 25 hours
        where the relevant reminder bitmask bit has not been set.
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=25)
        today_str = now.strftime("%Y-%m-%d")
        tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        rows = self._conn.execute(
            """SELECT * FROM scheduled_appointments
               WHERE status IN ('scheduled', 'confirmed')
                 AND date IN (?, ?)
                 AND reminder_sent < 3
               ORDER BY date, time_slot""",
            (today_str, tomorrow_str),
        ).fetchall()

        pending = []
        for row in rows:
            appt = ScheduledAppointment.from_row(row)
            try:
                appt_dt = datetime.strptime(
                    f"{appt.date} {appt.time_slot}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            hours_until = (appt_dt - now).total_seconds() / 3600.0
            if hours_until < 0:
                continue

            # 24-hour reminder: between 23-25 hours out, bit 1 not set
            if 0 < hours_until <= 25 and not (appt.reminder_sent & 1):
                if hours_until > 1:  # Don't send 24hr reminder if <1hr away
                    pending.append({
                        "appointment": appt.to_dict(),
                        "reminder_type": "24hr",
                        "hours_until": round(hours_until, 1),
                    })

            # 1-hour reminder: within 1.5 hours, bit 2 not set
            if 0 < hours_until <= 1.5 and not (appt.reminder_sent & 2):
                pending.append({
                    "appointment": appt.to_dict(),
                    "reminder_type": "1hr",
                    "hours_until": round(hours_until, 1),
                })

        return pending

    def mark_reminder_sent(self, appointment_id: str, reminder_type: str) -> bool:
        """Set the reminder bitmask bit for a sent reminder.

        reminder_type: '24hr' sets bit 1, '1hr' sets bit 2.
        """
        bit = 1 if reminder_type == "24hr" else 2
        cur = self._conn.execute(
            """UPDATE scheduled_appointments
               SET reminder_sent = reminder_sent | ?
               WHERE appointment_id = ?""",
            (bit, appointment_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def _parse_time(self, time_str: str) -> datetime:
        """Parse HH:MM to a datetime (date portion is arbitrary)."""
        parts = time_str.split(":")
        h = int(parts[0]) if len(parts) > 0 else 0
        m = int(parts[1]) if len(parts) > 1 else 0
        return datetime(2000, 1, 1, h, m)

    def get_status(self) -> dict:
        """Return summary counts for dashboard display."""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        monday = datetime.now(timezone.utc).date()
        monday -= timedelta(days=monday.weekday())
        sunday = monday + timedelta(days=6)

        total = self._conn.execute(
            "SELECT COUNT(*) FROM scheduled_appointments WHERE status != 'cancelled'"
        ).fetchone()[0]

        upcoming = self._conn.execute(
            """SELECT COUNT(*) FROM scheduled_appointments
               WHERE date >= ? AND status IN ('scheduled', 'confirmed')""",
            (today_str,),
        ).fetchone()[0]

        today_count = self._conn.execute(
            """SELECT COUNT(*) FROM scheduled_appointments
               WHERE date = ? AND status != 'cancelled'""",
            (today_str,),
        ).fetchone()[0]

        week_count = self._conn.execute(
            """SELECT COUNT(*) FROM scheduled_appointments
               WHERE date >= ? AND date <= ? AND status != 'cancelled'""",
            (monday.isoformat(), sunday.isoformat()),
        ).fetchone()[0]

        return {
            "total": total,
            "upcoming": upcoming,
            "today_count": today_count,
            "this_week_count": week_count,
        }

    def to_broadcast_dict(self) -> dict:
        """Full state for dashboard broadcast."""
        today = datetime.now(timezone.utc).date()
        monday = today - timedelta(days=today.weekday())

        # Get all non-cancelled upcoming appointments for the list
        upcoming = self.list_appointments(limit=200, status_filter=None)

        return {
            "appointments": [a.to_dict() for a in upcoming],
            "status": self.get_status(),
            "current_week": self.get_week_view(monday.isoformat()),
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("AppointmentScheduler closed")
