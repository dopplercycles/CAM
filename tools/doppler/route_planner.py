"""Route Planner for Doppler Cycles mobile service.

Pulls the day's appointments from the scheduler, optimizes stop order to
minimize total drive time, and produces a full timeline (leave home → stops
→ return home).  Completed routes store actual vs estimated times so
estimates improve over time.

Storage: ``data/routes.db`` (separate SQLite database).
"""

import itertools
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

from tools.doppler.scheduler_appointments import AppointmentScheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RouteStop:
    """A single stop on a planned route."""

    stop_id: str
    route_id: str
    stop_order: int
    appointment_id: str
    customer_name: str
    address: str
    lat: float
    lon: float
    service_type: str
    duration_minutes: int
    travel_minutes_from_prev: float
    travel_miles_from_prev: float
    eta: str  # HH:MM
    planned_departure: str  # HH:MM
    actual_arrival: str = ""
    actual_departure: str = ""

    @property
    def short_id(self) -> str:
        return self.stop_id[:8]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_id": self.stop_id,
            "route_id": self.route_id,
            "stop_order": self.stop_order,
            "appointment_id": self.appointment_id,
            "customer_name": self.customer_name,
            "address": self.address,
            "lat": self.lat,
            "lon": self.lon,
            "service_type": self.service_type,
            "duration_minutes": self.duration_minutes,
            "travel_minutes_from_prev": round(self.travel_minutes_from_prev, 1),
            "travel_miles_from_prev": round(self.travel_miles_from_prev, 1),
            "eta": self.eta,
            "planned_departure": self.planned_departure,
            "actual_arrival": self.actual_arrival,
            "actual_departure": self.actual_departure,
            "short_id": self.short_id,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "RouteStop":
        return cls(
            stop_id=row["stop_id"],
            route_id=row["route_id"],
            stop_order=row["stop_order"],
            appointment_id=row["appointment_id"],
            customer_name=row["customer_name"],
            address=row["address"],
            lat=row["lat"],
            lon=row["lon"],
            service_type=row["service_type"],
            duration_minutes=row["duration_minutes"],
            travel_minutes_from_prev=row["travel_minutes_from_prev"],
            travel_miles_from_prev=row["travel_miles_from_prev"],
            eta=row["eta"],
            planned_departure=row["planned_departure"],
            actual_arrival=row["actual_arrival"] or "",
            actual_departure=row["actual_departure"] or "",
        )


@dataclass
class PlannedRoute:
    """A full day's route with ordered stops."""

    route_id: str
    date: str
    status: str  # "planned" / "in_progress" / "completed"
    total_distance_miles: float
    total_drive_minutes: float
    total_service_minutes: float
    depart_time: str  # HH:MM
    estimated_return: str  # HH:MM
    actual_return: str  # HH:MM or ""
    stop_count: int
    created_at: str
    metadata: dict = field(default_factory=dict)
    stops: list[RouteStop] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "date": self.date,
            "status": self.status,
            "total_distance_miles": round(self.total_distance_miles, 1),
            "total_drive_minutes": round(self.total_drive_minutes, 1),
            "total_service_minutes": round(self.total_service_minutes, 1),
            "depart_time": self.depart_time,
            "estimated_return": self.estimated_return,
            "actual_return": self.actual_return,
            "stop_count": self.stop_count,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "stops": [s.to_dict() for s in self.stops],
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row, stops: list[RouteStop] | None = None) -> "PlannedRoute":
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return cls(
            route_id=row["route_id"],
            date=row["date"],
            status=row["status"],
            total_distance_miles=row["total_distance_miles"],
            total_drive_minutes=row["total_drive_minutes"],
            total_service_minutes=row["total_service_minutes"],
            depart_time=row["depart_time"],
            estimated_return=row["estimated_return"],
            actual_return=row["actual_return"] or "",
            stop_count=row["stop_count"],
            created_at=row["created_at"],
            metadata=metadata,
            stops=stops or [],
        )


# ---------------------------------------------------------------------------
# RoutePlanner
# ---------------------------------------------------------------------------


class RoutePlanner:
    """Plans and tracks optimised daily routes for mobile service calls.

    Args:
        db_path: Path to the SQLite database file.
        on_change: Async callback fired after any mutation (for WS broadcast).
        appointment_scheduler: AppointmentScheduler instance to pull day
            appointments and reuse travel estimation helpers.
    """

    def __init__(
        self,
        db_path: str = "data/routes.db",
        on_change: Callable[[], Coroutine] | None = None,
        appointment_scheduler: AppointmentScheduler | None = None,
    ):
        self.db_path = db_path
        self._on_change = on_change
        self._scheduler = appointment_scheduler

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("RoutePlanner initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS routes (
                route_id              TEXT PRIMARY KEY,
                date                  TEXT NOT NULL,
                status                TEXT NOT NULL DEFAULT 'planned',
                total_distance_miles  REAL DEFAULT 0.0,
                total_drive_minutes   REAL DEFAULT 0.0,
                total_service_minutes REAL DEFAULT 0.0,
                depart_time           TEXT NOT NULL,
                estimated_return      TEXT NOT NULL,
                actual_return         TEXT,
                stop_count            INTEGER DEFAULT 0,
                created_at            TEXT NOT NULL,
                metadata              TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_routes_date
                ON routes(date);

            CREATE TABLE IF NOT EXISTS route_stops (
                stop_id                TEXT PRIMARY KEY,
                route_id               TEXT NOT NULL,
                stop_order             INTEGER NOT NULL,
                appointment_id         TEXT,
                customer_name          TEXT DEFAULT '',
                address                TEXT DEFAULT '',
                lat                    REAL DEFAULT 0.0,
                lon                    REAL DEFAULT 0.0,
                service_type           TEXT DEFAULT '',
                duration_minutes       INTEGER DEFAULT 60,
                travel_minutes_from_prev REAL DEFAULT 0.0,
                travel_miles_from_prev REAL DEFAULT 0.0,
                eta                    TEXT DEFAULT '',
                planned_departure      TEXT DEFAULT '',
                actual_arrival         TEXT,
                actual_departure       TEXT,
                FOREIGN KEY (route_id) REFERENCES routes(route_id)
            );

            CREATE INDEX IF NOT EXISTS idx_route_stops_route
                ON route_stops(route_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def _notify(self):
        """Fire the on_change callback if set."""
        if self._on_change:
            try:
                await self._on_change()
            except Exception:
                logger.exception("RoutePlanner on_change callback failed")

    @property
    def _home_lat(self) -> float:
        return self._scheduler.home_lat if self._scheduler else 45.4976

    @property
    def _home_lon(self) -> float:
        return self._scheduler.home_lon if self._scheduler else -122.4302

    def _travel_miles(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in miles via the scheduler."""
        if self._scheduler:
            return AppointmentScheduler.haversine_distance(lat1, lon1, lat2, lon2)
        return 0.0

    def _travel_minutes(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Estimated drive time in minutes via the scheduler."""
        if self._scheduler:
            return self._scheduler.estimate_travel_minutes(lat1, lon1, lat2, lon2)
        return 0.0

    @staticmethod
    def _add_minutes(time_str: str, minutes: float) -> str:
        """Add minutes to an HH:MM string and return HH:MM."""
        parts = time_str.split(":")
        h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        base = datetime(2000, 1, 1, h, m)
        result = base + timedelta(minutes=minutes)
        return result.strftime("%H:%M")

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _optimize_stops(self, appointments: list, home_lat: float, home_lon: float) -> list:
        """Reorder appointments to minimise total travel distance.

        For ≤7 stops: brute-force all permutations (optimal).
        For >7 stops: nearest-neighbor greedy from home base.
        """
        if len(appointments) <= 1:
            return list(appointments)

        def total_cost(perm):
            """Total haversine miles for a given permutation."""
            cost = 0.0
            prev_lat, prev_lon = home_lat, home_lon
            for appt in perm:
                cost += self._travel_miles(prev_lat, prev_lon, appt.location_lat, appt.location_lon)
                prev_lat, prev_lon = appt.location_lat, appt.location_lon
            # Return-home leg
            cost += self._travel_miles(prev_lat, prev_lon, home_lat, home_lon)
            return cost

        if len(appointments) <= 7:
            # Brute-force — try every permutation, pick minimum
            best = None
            best_cost = float("inf")
            for perm in itertools.permutations(appointments):
                c = total_cost(perm)
                if c < best_cost:
                    best_cost = c
                    best = list(perm)
            return best or list(appointments)
        else:
            # Nearest-neighbor greedy
            remaining = list(appointments)
            ordered = []
            prev_lat, prev_lon = home_lat, home_lon
            while remaining:
                nearest = min(
                    remaining,
                    key=lambda a: self._travel_miles(prev_lat, prev_lon, a.location_lat, a.location_lon),
                )
                ordered.append(nearest)
                prev_lat, prev_lon = nearest.location_lat, nearest.location_lon
                remaining.remove(nearest)
            return ordered

    # ------------------------------------------------------------------
    # Timeline builder
    # ------------------------------------------------------------------

    def _build_timeline(
        self, ordered_appointments: list, depart_time: str, home_lat: float, home_lon: float
    ) -> tuple[list[RouteStop], float, float, float, str]:
        """Build a timed route from an ordered list of appointments.

        Returns:
            (stops, total_distance_miles, total_drive_minutes,
             total_service_minutes, estimated_return_time)
        """
        stops: list[RouteStop] = []
        total_dist = 0.0
        total_drive = 0.0
        total_service = 0.0
        prev_lat, prev_lon = home_lat, home_lon
        current_time = depart_time

        for order, appt in enumerate(ordered_appointments, start=1):
            # Travel from previous location
            miles = self._travel_miles(prev_lat, prev_lon, appt.location_lat, appt.location_lon)
            minutes = self._travel_minutes(prev_lat, prev_lon, appt.location_lat, appt.location_lon)
            total_dist += miles
            total_drive += minutes

            eta = self._add_minutes(current_time, minutes)
            duration = appt.duration_estimate or 60
            total_service += duration
            planned_departure = self._add_minutes(eta, duration)

            stop = RouteStop(
                stop_id=str(uuid.uuid4()),
                route_id="",  # set after route is created
                stop_order=order,
                appointment_id=appt.appointment_id,
                customer_name=appt.customer_name,
                address=appt.location_address,
                lat=appt.location_lat,
                lon=appt.location_lon,
                service_type=appt.service_type,
                duration_minutes=duration,
                travel_minutes_from_prev=minutes,
                travel_miles_from_prev=miles,
                eta=eta,
                planned_departure=planned_departure,
            )
            stops.append(stop)
            prev_lat, prev_lon = appt.location_lat, appt.location_lon
            current_time = planned_departure

        # Return-home leg
        if stops:
            return_miles = self._travel_miles(prev_lat, prev_lon, home_lat, home_lon)
            return_minutes = self._travel_minutes(prev_lat, prev_lon, home_lat, home_lon)
            total_dist += return_miles
            total_drive += return_minutes
            estimated_return = self._add_minutes(current_time, return_minutes)
        else:
            estimated_return = depart_time

        return stops, total_dist, total_drive, total_service, estimated_return

    # ------------------------------------------------------------------
    # Core public methods
    # ------------------------------------------------------------------

    def plan_route(self, date_str: str, depart_time: str = "08:00") -> PlannedRoute:
        """Plan (or replan) an optimised route for the given date.

        Fetches the day's appointments from the scheduler, optimises stop
        order, builds a timeline, and saves the result to the database.
        If a planned route already exists for that date it is replaced.
        """
        if not self._scheduler:
            raise RuntimeError("No appointment scheduler configured")

        # Get the day's appointments
        appointments = self._scheduler.get_day_schedule(date_str)
        # Filter to appointments with valid coordinates
        appointments = [a for a in appointments if not (a.location_lat == 0.0 and a.location_lon == 0.0)]

        home_lat, home_lon = self._home_lat, self._home_lon

        # Optimise order
        ordered = self._optimize_stops(appointments, home_lat, home_lon)

        # Build timeline
        stops, total_dist, total_drive, total_service, est_return = self._build_timeline(
            ordered, depart_time, home_lat, home_lon
        )

        # Create route record
        route_id = str(uuid.uuid4())
        now = self._now_iso()

        # Set route_id on all stops
        for s in stops:
            s.route_id = route_id

        # Upsert: delete any existing planned route for this date
        existing = self._conn.execute(
            "SELECT route_id FROM routes WHERE date = ?", (date_str,)
        ).fetchall()
        for row in existing:
            self._conn.execute("DELETE FROM route_stops WHERE route_id = ?", (row["route_id"],))
            self._conn.execute("DELETE FROM routes WHERE route_id = ?", (row["route_id"],))

        # Insert route
        self._conn.execute(
            """INSERT INTO routes
               (route_id, date, status, total_distance_miles, total_drive_minutes,
                total_service_minutes, depart_time, estimated_return, actual_return,
                stop_count, created_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                route_id, date_str, "planned", total_dist, total_drive,
                total_service, depart_time, est_return, None,
                len(stops), now, "{}",
            ),
        )

        # Insert stops
        for stop in stops:
            self._conn.execute(
                """INSERT INTO route_stops
                   (stop_id, route_id, stop_order, appointment_id, customer_name,
                    address, lat, lon, service_type, duration_minutes,
                    travel_minutes_from_prev, travel_miles_from_prev,
                    eta, planned_departure, actual_arrival, actual_departure)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    stop.stop_id, route_id, stop.stop_order, stop.appointment_id,
                    stop.customer_name, stop.address, stop.lat, stop.lon,
                    stop.service_type, stop.duration_minutes,
                    stop.travel_minutes_from_prev, stop.travel_miles_from_prev,
                    stop.eta, stop.planned_departure, None, None,
                ),
            )

        self._conn.commit()
        logger.info("Planned route %s for %s: %d stops, %.1f mi", route_id[:8], date_str, len(stops), total_dist)

        return PlannedRoute(
            route_id=route_id,
            date=date_str,
            status="planned",
            total_distance_miles=total_dist,
            total_drive_minutes=total_drive,
            total_service_minutes=total_service,
            depart_time=depart_time,
            estimated_return=est_return,
            actual_return="",
            stop_count=len(stops),
            created_at=now,
            stops=stops,
        )

    def get_day_plan(self, date_str: str) -> PlannedRoute | None:
        """Fetch the existing planned route for a date, or None."""
        row = self._conn.execute(
            "SELECT * FROM routes WHERE date = ? ORDER BY created_at DESC LIMIT 1",
            (date_str,),
        ).fetchone()
        if row is None:
            return None

        stop_rows = self._conn.execute(
            "SELECT * FROM route_stops WHERE route_id = ? ORDER BY stop_order",
            (row["route_id"],),
        ).fetchall()
        stops = [RouteStop.from_row(sr) for sr in stop_rows]
        return PlannedRoute.from_row(row, stops)

    def start_route(self, route_id: str) -> PlannedRoute | None:
        """Mark a route as in_progress."""
        cur = self._conn.execute(
            "UPDATE routes SET status = 'in_progress' WHERE route_id = ?",
            (route_id,),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return None
        logger.info("Route %s started", route_id[:8])
        return self._get_route_by_id(route_id)

    def complete_stop(
        self,
        route_id: str,
        stop_order: int,
        actual_arrival: str = "",
        actual_departure: str = "",
    ) -> RouteStop | None:
        """Record actual times for a stop.

        If all stops now have actual times, the route is marked completed.
        """
        cur = self._conn.execute(
            """UPDATE route_stops
               SET actual_arrival = ?, actual_departure = ?
               WHERE route_id = ? AND stop_order = ?""",
            (actual_arrival, actual_departure, route_id, stop_order),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return None

        logger.info("Route %s stop %d completed", route_id[:8], stop_order)

        # Check if all stops are complete
        incomplete = self._conn.execute(
            """SELECT COUNT(*) FROM route_stops
               WHERE route_id = ? AND (actual_arrival IS NULL OR actual_arrival = '')""",
            (route_id,),
        ).fetchone()[0]
        if incomplete == 0:
            self._conn.execute(
                "UPDATE routes SET status = 'completed' WHERE route_id = ?",
                (route_id,),
            )
            self._conn.commit()
            logger.info("Route %s auto-completed (all stops done)", route_id[:8])

        row = self._conn.execute(
            "SELECT * FROM route_stops WHERE route_id = ? AND stop_order = ?",
            (route_id, stop_order),
        ).fetchone()
        return RouteStop.from_row(row) if row else None

    def complete_route(self, route_id: str, actual_return: str = "") -> PlannedRoute | None:
        """Mark a route as completed, optionally recording actual return time."""
        cur = self._conn.execute(
            "UPDATE routes SET status = 'completed', actual_return = ? WHERE route_id = ?",
            (actual_return or None, route_id),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return None
        logger.info("Route %s completed (return: %s)", route_id[:8], actual_return or "N/A")
        return self._get_route_by_id(route_id)

    def get_route_history(self, limit: int = 20) -> list[PlannedRoute]:
        """Return past routes, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM routes ORDER BY date DESC, created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        routes = []
        for row in rows:
            stop_rows = self._conn.execute(
                "SELECT * FROM route_stops WHERE route_id = ? ORDER BY stop_order",
                (row["route_id"],),
            ).fetchall()
            stops = [RouteStop.from_row(sr) for sr in stop_rows]
            routes.append(PlannedRoute.from_row(row, stops))
        return routes

    def get_accuracy_stats(self) -> dict:
        """Calculate accuracy stats from completed routes with actual times.

        Returns:
            dict with total_routes, total_stops, avg_arrival_variance_min,
            avg_service_variance_min, on_time_pct.
        """
        total_routes = self._conn.execute(
            "SELECT COUNT(*) FROM routes WHERE status = 'completed'"
        ).fetchone()[0]

        # Get stops with both estimated and actual arrival times
        rows = self._conn.execute(
            """SELECT eta, actual_arrival, planned_departure, actual_departure,
                      duration_minutes
               FROM route_stops
               WHERE actual_arrival IS NOT NULL AND actual_arrival != ''
                 AND eta IS NOT NULL AND eta != ''"""
        ).fetchall()

        total_stops = len(rows)
        if total_stops == 0:
            return {
                "total_routes": total_routes,
                "total_stops": 0,
                "avg_arrival_variance_min": 0.0,
                "avg_service_variance_min": 0.0,
                "on_time_pct": 0.0,
            }

        arrival_variances = []
        service_variances = []
        on_time_count = 0

        for row in rows:
            # Arrival variance (actual - estimated, in minutes)
            try:
                est_arr = self._parse_hhmm(row["eta"])
                act_arr = self._parse_hhmm(row["actual_arrival"])
                diff = (act_arr - est_arr).total_seconds() / 60.0
                arrival_variances.append(diff)
                # "On time" = arrived within 10 minutes of ETA
                if abs(diff) <= 10:
                    on_time_count += 1
            except (ValueError, TypeError):
                pass

            # Service duration variance (actual vs planned)
            try:
                if row["actual_departure"] and row["actual_arrival"]:
                    act_dep = self._parse_hhmm(row["actual_departure"])
                    act_arr2 = self._parse_hhmm(row["actual_arrival"])
                    actual_service = (act_dep - act_arr2).total_seconds() / 60.0
                    planned_service = row["duration_minutes"] or 60
                    service_variances.append(actual_service - planned_service)
            except (ValueError, TypeError):
                pass

        avg_arrival = sum(arrival_variances) / len(arrival_variances) if arrival_variances else 0.0
        avg_service = sum(service_variances) / len(service_variances) if service_variances else 0.0
        on_time_pct = (on_time_count / len(arrival_variances) * 100) if arrival_variances else 0.0

        return {
            "total_routes": total_routes,
            "total_stops": total_stops,
            "avg_arrival_variance_min": round(avg_arrival, 1),
            "avg_service_variance_min": round(avg_service, 1),
            "on_time_pct": round(on_time_pct, 1),
        }

    @staticmethod
    def _parse_hhmm(time_str: str) -> datetime:
        """Parse HH:MM into a datetime (date portion arbitrary)."""
        parts = time_str.strip().split(":")
        h = int(parts[0]) if len(parts) > 0 else 0
        m = int(parts[1]) if len(parts) > 1 else 0
        return datetime(2000, 1, 1, h, m)

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Summary status for dashboard display."""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_route = self.get_day_plan(today_str)

        total_routes = self._conn.execute(
            "SELECT COUNT(*) FROM routes"
        ).fetchone()[0]

        total_dist = self._conn.execute(
            "SELECT COALESCE(SUM(total_distance_miles), 0) FROM routes WHERE status = 'completed'"
        ).fetchone()[0]

        return {
            "today_route_status": today_route.status if today_route else "none",
            "today_stops": today_route.stop_count if today_route else 0,
            "total_routes": total_routes,
            "total_distance_all_time": round(total_dist, 1),
        }

    def to_broadcast_dict(self) -> dict:
        """Full state for dashboard broadcast."""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_route = self.get_day_plan(today_str)

        history = self.get_route_history(limit=10)

        return {
            "today_route": today_route.to_dict() if today_route else None,
            "status": self.get_status(),
            "route_history": [r.to_dict() for r in history],
            "accuracy_stats": self.get_accuracy_stats(),
        }

    def _get_route_by_id(self, route_id: str) -> PlannedRoute | None:
        """Fetch a route and its stops by ID."""
        row = self._conn.execute(
            "SELECT * FROM routes WHERE route_id = ?", (route_id,)
        ).fetchone()
        if row is None:
            return None
        stop_rows = self._conn.execute(
            "SELECT * FROM route_stops WHERE route_id = ? ORDER BY stop_order",
            (row["route_id"],),
        ).fetchall()
        stops = [RouteStop.from_row(sr) for sr in stop_rows]
        return PlannedRoute.from_row(row, stops)

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("RoutePlanner closed")
