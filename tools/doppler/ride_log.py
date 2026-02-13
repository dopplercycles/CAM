"""
Ride and Service Log for CAM / Doppler Cycles.

Tracks every ride — service calls, commutes, content rides, personal trips,
Highway 20 prep — for tax deduction documentation (business miles vs personal
miles).  Calculates fuel efficiency and generates IRS-ready mileage reports.

Essential bookkeeping for a mobile motorcycle service business where the
DR650 is both the shop truck and the content platform.

SQLite-backed, single-file module — same pattern as training.py, photo_docs.py,
and invoicing.py.
"""

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PURPOSE_TYPES = ("service_call", "commute", "content_ride", "personal", "highway_20")

# Business-deductible purposes per IRS rules
BUSINESS_PURPOSES = ("service_call", "commute", "content_ride", "highway_20")

# IRS standard mileage rate for 2026 (dollars per mile)
IRS_MILEAGE_RATE = 0.70

PURPOSE_LABELS = {
    "service_call": "Service Call",
    "commute": "Commute",
    "content_ride": "Content Ride",
    "personal": "Personal",
    "highway_20": "Highway 20",
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RideEntry:
    """A single ride log entry."""
    ride_id: str = ""
    date: str = ""
    start_time: str = ""
    end_time: str = ""
    start_location: str = ""
    end_location: str = ""
    distance: float = 0.0
    purpose: str = "personal"
    weather_conditions: str = ""
    fuel_used: float = 0.0
    odometer_start: float = 0.0
    odometer_end: float = 0.0
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def short_id(self) -> str:
        return self.ride_id[:8] if self.ride_id else ""

    @property
    def duration_minutes(self) -> Optional[int]:
        """Calculate ride duration from start_time and end_time (HH:MM format)."""
        if not self.start_time or not self.end_time:
            return None
        try:
            start = datetime.strptime(self.start_time, "%H:%M")
            end = datetime.strptime(self.end_time, "%H:%M")
            delta = (end - start).total_seconds() / 60
            # Handle rides crossing midnight
            if delta < 0:
                delta += 24 * 60
            return int(delta)
        except (ValueError, TypeError):
            return None

    @property
    def mpg(self) -> Optional[float]:
        """Miles per gallon — distance / fuel_used if both > 0."""
        if self.distance > 0 and self.fuel_used > 0:
            return round(self.distance / self.fuel_used, 1)
        return None

    def to_dict(self) -> dict:
        return {
            "ride_id": self.ride_id,
            "short_id": self.short_id,
            "date": self.date,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_location": self.start_location,
            "end_location": self.end_location,
            "distance": self.distance,
            "purpose": self.purpose,
            "purpose_label": PURPOSE_LABELS.get(self.purpose, self.purpose),
            "weather_conditions": self.weather_conditions,
            "fuel_used": self.fuel_used,
            "odometer_start": self.odometer_start,
            "odometer_end": self.odometer_end,
            "notes": self.notes,
            "duration_minutes": self.duration_minutes,
            "mpg": self.mpg,
            "is_business": self.purpose in BUSINESS_PURPOSES,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_row(row) -> "RideEntry":
        """Build a RideEntry from a sqlite3.Row."""
        r = dict(row)
        return RideEntry(
            ride_id=r["ride_id"],
            date=r.get("date", ""),
            start_time=r.get("start_time", ""),
            end_time=r.get("end_time", ""),
            start_location=r.get("start_location", ""),
            end_location=r.get("end_location", ""),
            distance=float(r.get("distance", 0)),
            purpose=r.get("purpose", "personal"),
            weather_conditions=r.get("weather_conditions", ""),
            fuel_used=float(r.get("fuel_used", 0)),
            odometer_start=float(r.get("odometer_start", 0)),
            odometer_end=float(r.get("odometer_end", 0)),
            notes=r.get("notes", ""),
            created_at=r.get("created_at", ""),
            updated_at=r.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# RideLogManager
# ---------------------------------------------------------------------------

class RideLogManager:
    """SQLite-backed ride and mileage log for Doppler Cycles.

    Tracks rides by purpose (service call, commute, content ride, personal,
    Highway 20), calculates fuel efficiency, and generates IRS-ready mileage
    reports for tax deduction documentation.

    Args:
        db_path:    Path to the SQLite database file.
        on_change:  Async callback fired after any state mutation
                    (for broadcasting updates to the dashboard).
    """

    def __init__(
        self,
        db_path: str = "data/ride_log.db",
        on_change: Optional[Callable[[], Coroutine]] = None,
    ):
        self._db_path = db_path
        self._on_change = on_change

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("RideLogManager initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        """Create the rides table if it doesn't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS rides (
                ride_id            TEXT PRIMARY KEY,
                date               TEXT NOT NULL,
                start_time         TEXT DEFAULT '',
                end_time           TEXT DEFAULT '',
                start_location     TEXT DEFAULT '',
                end_location       TEXT DEFAULT '',
                distance           REAL DEFAULT 0,
                purpose            TEXT DEFAULT 'personal',
                weather_conditions TEXT DEFAULT '',
                fuel_used          REAL DEFAULT 0,
                odometer_start     REAL DEFAULT 0,
                odometer_end       REAL DEFAULT 0,
                notes              TEXT DEFAULT '',
                created_at         TEXT NOT NULL,
                updated_at         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rides_date ON rides(date);
            CREATE INDEX IF NOT EXISTS idx_rides_purpose ON rides(purpose);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now(self) -> str:
        """ISO-8601 timestamp."""
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _fire_change(self):
        """Schedule the on_change callback if one was provided."""
        if self._on_change:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._on_change())
            except RuntimeError:
                pass  # no running loop — skip broadcast

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def log_ride(
        self,
        date: str,
        start_time: str = "",
        end_time: str = "",
        start_location: str = "",
        end_location: str = "",
        distance: float = 0.0,
        purpose: str = "personal",
        weather_conditions: str = "",
        fuel_used: float = 0.0,
        odometer_start: float = 0.0,
        odometer_end: float = 0.0,
        notes: str = "",
    ) -> RideEntry:
        """Log a new ride.

        Args:
            date:               YYYY-MM-DD format.
            start_time:         HH:MM (24h), optional.
            end_time:           HH:MM (24h), optional.
            start_location:     Free text starting point.
            end_location:       Free text destination.
            distance:           Miles ridden.
            purpose:            One of PURPOSE_TYPES.
            weather_conditions: Free text weather description.
            fuel_used:          Gallons consumed.
            odometer_start:     Odometer reading at start.
            odometer_end:       Odometer reading at end.
            notes:              Any additional notes.

        Returns:
            The newly created RideEntry.
        """
        if purpose not in PURPOSE_TYPES:
            purpose = "personal"

        # Auto-calculate distance from odometer if not provided directly
        if distance == 0 and odometer_start > 0 and odometer_end > odometer_start:
            distance = odometer_end - odometer_start

        now = self._now()
        entry = RideEntry(
            ride_id=str(uuid.uuid4()),
            date=date,
            start_time=start_time,
            end_time=end_time,
            start_location=start_location,
            end_location=end_location,
            distance=distance,
            purpose=purpose,
            weather_conditions=weather_conditions,
            fuel_used=fuel_used,
            odometer_start=odometer_start,
            odometer_end=odometer_end,
            notes=notes,
            created_at=now,
            updated_at=now,
        )

        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO rides
               (ride_id, date, start_time, end_time, start_location,
                end_location, distance, purpose, weather_conditions,
                fuel_used, odometer_start, odometer_end, notes,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.ride_id, entry.date, entry.start_time, entry.end_time,
                entry.start_location, entry.end_location, entry.distance,
                entry.purpose, entry.weather_conditions, entry.fuel_used,
                entry.odometer_start, entry.odometer_end, entry.notes,
                entry.created_at, entry.updated_at,
            ),
        )
        self._conn.commit()
        logger.info("Ride logged: %s — %s, %.1f mi (%s)",
                     entry.short_id, entry.date, entry.distance, entry.purpose)
        self._fire_change()
        return entry

    def update_ride(self, ride_id: str, **kwargs) -> Optional[RideEntry]:
        """Update fields on an existing ride.

        Args:
            ride_id:  The ride UUID.
            **kwargs: Fields to update (must be valid RideEntry fields).

        Returns:
            Updated RideEntry, or None if ride_id not found.
        """
        existing = self.get_ride(ride_id)
        if not existing:
            return None

        allowed = {
            "date", "start_time", "end_time", "start_location", "end_location",
            "distance", "purpose", "weather_conditions", "fuel_used",
            "odometer_start", "odometer_end", "notes",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return existing

        # Validate purpose if being changed
        if "purpose" in updates and updates["purpose"] not in PURPOSE_TYPES:
            updates["purpose"] = "personal"

        updates["updated_at"] = self._now()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [ride_id]

        cur = self._conn.cursor()
        cur.execute(f"UPDATE rides SET {set_clause} WHERE ride_id = ?", values)
        self._conn.commit()

        logger.info("Ride updated: %s", ride_id[:8])
        self._fire_change()
        return self.get_ride(ride_id)

    def delete_ride(self, ride_id: str) -> bool:
        """Delete a ride entry.

        Args:
            ride_id: The ride UUID.

        Returns:
            True if a ride was deleted, False if not found.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM rides WHERE ride_id = ?", (ride_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Ride deleted: %s", ride_id[:8])
            self._fire_change()
        return deleted

    def get_ride(self, ride_id: str) -> Optional[RideEntry]:
        """Get a single ride by ID.

        Args:
            ride_id: The ride UUID.

        Returns:
            RideEntry or None if not found.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM rides WHERE ride_id = ?", (ride_id,))
        row = cur.fetchone()
        return RideEntry.from_row(row) if row else None

    def list_rides(
        self,
        purpose: str = "",
        month: str = "",
        limit: int = 50,
    ) -> list[RideEntry]:
        """List rides, newest first.

        Args:
            purpose: Filter by purpose type (optional).
            month:   Filter by month in YYYY-MM format (optional).
            limit:   Max results to return.

        Returns:
            List of RideEntry objects.
        """
        query = "SELECT * FROM rides WHERE 1=1"
        params: list[Any] = []

        if purpose and purpose in PURPOSE_TYPES:
            query += " AND purpose = ?"
            params.append(purpose)

        if month:
            query += " AND date LIKE ?"
            params.append(f"{month}%")

        query += " ORDER BY date DESC, start_time DESC LIMIT ?"
        params.append(limit)

        cur = self._conn.cursor()
        cur.execute(query, params)
        return [RideEntry.from_row(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Fuel tracking
    # ------------------------------------------------------------------

    def fuel_tracker(self, months: int = 6) -> dict:
        """Calculate fuel efficiency stats over the last N months.

        Args:
            months: Number of months to look back.

        Returns:
            Dict with total_gallons, total_miles, avg_mpg, and monthly breakdown.
        """
        # Calculate the start date
        now = datetime.utcnow()
        year = now.year
        month = now.month - months
        while month <= 0:
            month += 12
            year -= 1
        start_date = f"{year:04d}-{month:02d}-01"

        cur = self._conn.cursor()
        cur.execute(
            """SELECT date, distance, fuel_used FROM rides
               WHERE date >= ? AND (distance > 0 OR fuel_used > 0)
               ORDER BY date""",
            (start_date,),
        )
        rows = cur.fetchall()

        total_gallons = 0.0
        total_miles = 0.0
        monthly: dict[str, dict] = {}

        for row in rows:
            d = dict(row)
            month_key = d["date"][:7]  # YYYY-MM
            dist = float(d["distance"] or 0)
            fuel = float(d["fuel_used"] or 0)

            total_gallons += fuel
            total_miles += dist

            if month_key not in monthly:
                monthly[month_key] = {"month": month_key, "gallons": 0, "miles": 0}
            monthly[month_key]["gallons"] += fuel
            monthly[month_key]["miles"] += dist

        # Calculate MPG for each month
        monthly_list = []
        for m in sorted(monthly.values(), key=lambda x: x["month"]):
            m["mpg"] = round(m["miles"] / m["gallons"], 1) if m["gallons"] > 0 else None
            m["gallons"] = round(m["gallons"], 2)
            m["miles"] = round(m["miles"], 1)
            monthly_list.append(m)

        return {
            "total_gallons": round(total_gallons, 2),
            "total_miles": round(total_miles, 1),
            "avg_mpg": round(total_miles / total_gallons, 1) if total_gallons > 0 else None,
            "monthly": monthly_list,
        }

    # ------------------------------------------------------------------
    # Mileage reports (tax documentation)
    # ------------------------------------------------------------------

    def mileage_report(self, year: int = 0) -> dict:
        """Generate an IRS-ready mileage report for a given year.

        Groups rides by purpose, calculates business vs personal miles, and
        estimates the standard mileage deduction.

        Args:
            year: Tax year to report on (defaults to current year).

        Returns:
            Dict with year, totals, business/personal breakdown, by_purpose,
            monthly breakdown, IRS rate, and estimated deduction.
        """
        if year == 0:
            year = datetime.utcnow().year

        cur = self._conn.cursor()
        cur.execute(
            """SELECT * FROM rides
               WHERE date LIKE ?
               ORDER BY date""",
            (f"{year}-%",),
        )
        rows = cur.fetchall()

        total_miles = 0.0
        business_miles = 0.0
        personal_miles = 0.0
        by_purpose: dict[str, dict] = {}
        monthly: dict[str, dict] = {}

        for row in rows:
            entry = RideEntry.from_row(row)
            dist = entry.distance

            total_miles += dist

            if entry.purpose in BUSINESS_PURPOSES:
                business_miles += dist
            else:
                personal_miles += dist

            # By purpose
            if entry.purpose not in by_purpose:
                by_purpose[entry.purpose] = {"count": 0, "miles": 0.0}
            by_purpose[entry.purpose]["count"] += 1
            by_purpose[entry.purpose]["miles"] += dist

            # Monthly
            month_key = entry.date[:7]
            if month_key not in monthly:
                monthly[month_key] = {"month": month_key, "business": 0.0, "personal": 0.0}
            if entry.purpose in BUSINESS_PURPOSES:
                monthly[month_key]["business"] += dist
            else:
                monthly[month_key]["personal"] += dist

        # Round miles in by_purpose
        for p in by_purpose.values():
            p["miles"] = round(p["miles"], 1)

        # Sort and round monthly
        monthly_list = []
        for m in sorted(monthly.values(), key=lambda x: x["month"]):
            m["business"] = round(m["business"], 1)
            m["personal"] = round(m["personal"], 1)
            monthly_list.append(m)

        return {
            "year": year,
            "total_miles": round(total_miles, 1),
            "business_miles": round(business_miles, 1),
            "personal_miles": round(personal_miles, 1),
            "business_pct": round(business_miles / total_miles * 100, 1) if total_miles > 0 else 0,
            "by_purpose": by_purpose,
            "monthly": monthly_list,
            "irs_rate": IRS_MILEAGE_RATE,
            "estimated_deduction": round(business_miles * IRS_MILEAGE_RATE, 2),
        }

    # ------------------------------------------------------------------
    # Monthly summary
    # ------------------------------------------------------------------

    def monthly_summary(self, months: int = 6) -> list[dict]:
        """Per-month summary of riding activity.

        Args:
            months: Number of months to look back.

        Returns:
            List of dicts with total_rides, total_miles, business_miles,
            personal_miles, fuel_used, avg_mpg per month.
        """
        now = datetime.utcnow()
        year = now.year
        month = now.month - months
        while month <= 0:
            month += 12
            year -= 1
        start_date = f"{year:04d}-{month:02d}-01"

        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM rides WHERE date >= ? ORDER BY date",
            (start_date,),
        )
        rows = cur.fetchall()

        buckets: dict[str, dict] = {}
        for row in rows:
            entry = RideEntry.from_row(row)
            month_key = entry.date[:7]

            if month_key not in buckets:
                buckets[month_key] = {
                    "month": month_key,
                    "total_rides": 0,
                    "total_miles": 0.0,
                    "business_miles": 0.0,
                    "personal_miles": 0.0,
                    "fuel_used": 0.0,
                }

            b = buckets[month_key]
            b["total_rides"] += 1
            b["total_miles"] += entry.distance
            if entry.purpose in BUSINESS_PURPOSES:
                b["business_miles"] += entry.distance
            else:
                b["personal_miles"] += entry.distance
            b["fuel_used"] += entry.fuel_used

        result = []
        for b in sorted(buckets.values(), key=lambda x: x["month"]):
            b["total_miles"] = round(b["total_miles"], 1)
            b["business_miles"] = round(b["business_miles"], 1)
            b["personal_miles"] = round(b["personal_miles"], 1)
            b["fuel_used"] = round(b["fuel_used"], 2)
            b["avg_mpg"] = (
                round(b["total_miles"] / b["fuel_used"], 1)
                if b["fuel_used"] > 0 else None
            )
            result.append(b)

        return result

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Summary status for the dashboard header cards."""
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) FROM rides")
        total_rides = cur.fetchone()[0]

        cur.execute("SELECT COALESCE(SUM(distance), 0) FROM rides")
        total_miles = round(cur.fetchone()[0], 1)

        cur.execute(
            "SELECT COALESCE(SUM(distance), 0) FROM rides WHERE purpose IN (?, ?, ?, ?)",
            BUSINESS_PURPOSES,
        )
        business_miles = round(cur.fetchone()[0], 1)

        # This month
        month_start = datetime.utcnow().strftime("%Y-%m-01")
        cur.execute(
            "SELECT COALESCE(SUM(distance), 0) FROM rides WHERE date >= ?",
            (month_start,),
        )
        this_month_miles = round(cur.fetchone()[0], 1)

        cur.execute(
            "SELECT COUNT(*) FROM rides WHERE date >= ?",
            (month_start,),
        )
        rides_this_month = cur.fetchone()[0]

        # Average MPG (only rides with fuel data)
        cur.execute(
            """SELECT SUM(distance), SUM(fuel_used) FROM rides
               WHERE fuel_used > 0 AND distance > 0"""
        )
        row = cur.fetchone()
        total_dist = row[0] or 0
        total_fuel = row[1] or 0
        avg_mpg = round(total_dist / total_fuel, 1) if total_fuel > 0 else None

        return {
            "total_rides": total_rides,
            "total_miles": total_miles,
            "business_miles": business_miles,
            "this_month_miles": this_month_miles,
            "avg_mpg": avg_mpg,
            "rides_this_month": rides_this_month,
        }

    def to_broadcast_dict(self) -> dict:
        """Full state dict for broadcasting to the dashboard."""
        rides = self.list_rides(limit=50)
        return {
            "rides": [r.to_dict() for r in rides],
            "status": self.get_status(),
            "fuel_stats": self.fuel_tracker(months=6),
            "mileage_report": self.mileage_report(),
            "monthly_summary": self.monthly_summary(months=6),
        }

    def close(self):
        """Close the SQLite connection. Call on shutdown."""
        try:
            self._conn.close()
            logger.info("RideLogManager database closed")
        except Exception:
            pass
