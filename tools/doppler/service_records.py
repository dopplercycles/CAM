"""
CAM Service Record System

Manages vehicles and service records for Doppler Cycles' mobile diagnostic
business. Stores data in a separate SQLite database (data/service_records.db)
and generates professional PDF service reports using fpdf2.

Links to BusinessStore customers via owner_id / customer_id (same UUIDs),
but uses denormalized names to avoid cross-DB JOINs.

Usage:
    from tools.doppler.service_records import ServiceRecordStore

    store = ServiceRecordStore(db_path="data/service_records.db")
    vehicle = store.add_vehicle(year="2019", make="Harley-Davidson",
                                model="Street Glide", vin="1HD1KTP...",
                                owner_id="abc-123", owner_name="Mike R.")
    record = store.add_record(vehicle_id=vehicle.vehicle_id,
                              customer_id=vehicle.owner_id,
                              customer_name=vehicle.owner_name,
                              date="2026-02-10", service_type="diagnostic",
                              services_performed=["Engine diagnostic", "Code read"],
                              labor_hours=2.0)
    pdf_path = store.generate_pdf(record.record_id)
"""

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

from fpdf import FPDF


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("cam.service_records")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Vehicle:
    """A customer's motorcycle / vehicle.

    Attributes:
        vehicle_id:  Unique identifier (UUID string).
        year:        Model year.
        make:        Manufacturer (e.g. Harley-Davidson).
        model:       Model name (e.g. Street Glide).
        vin:         Vehicle Identification Number.
        owner_id:    Link to BusinessStore customer_id.
        owner_name:  Denormalized owner name (avoids cross-DB JOINs).
        color:       Vehicle color.
        mileage:     Current mileage reading.
        notes:       Free-text notes.
        created_at:  When the record was created (ISO string).
        updated_at:  When the record was last modified (ISO string).
        metadata:    Additional key-value data.
    """
    vehicle_id: str
    year: str = ""
    make: str = ""
    model: str = ""
    vin: str = ""
    owner_id: str = ""
    owner_name: str = ""
    color: str = ""
    mileage: int = 0
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 characters of the vehicle ID for display."""
        return self.vehicle_id[:8]

    @property
    def display_name(self) -> str:
        """Human-readable vehicle name: 'year make model'."""
        parts = [p for p in (self.year, self.make, self.model) if p]
        return " ".join(parts) or "Unknown Vehicle"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/dashboard use."""
        return {
            "vehicle_id": self.vehicle_id,
            "short_id": self.short_id,
            "display_name": self.display_name,
            "year": self.year,
            "make": self.make,
            "model": self.model,
            "vin": self.vin,
            "owner_id": self.owner_id,
            "owner_name": self.owner_name,
            "color": self.color,
            "mileage": self.mileage,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Vehicle":
        """Convert a SQLite row to a Vehicle."""
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return cls(
            vehicle_id=row["vehicle_id"],
            year=row["year"],
            make=row["make"],
            model=row["model"],
            vin=row["vin"],
            owner_id=row["owner_id"],
            owner_name=row["owner_name"],
            color=row["color"],
            mileage=row["mileage"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=meta,
        )


@dataclass
class ServiceRecord:
    """A service record for a vehicle.

    Attributes:
        record_id:          Unique identifier (UUID string).
        vehicle_id:         Link to vehicles table.
        customer_id:        Link to BusinessStore customer.
        customer_name:      Denormalized customer name.
        vehicle_summary:    Denormalized: "2019 Harley-Davidson Street Glide".
        date:               Service date (YYYY-MM-DD).
        service_type:       Type: diagnostic, maintenance, repair, inspection.
        services_performed: List of service description strings.
        parts_used:         List of dicts: {description, part_number, quantity, unit_cost}.
        labor_hours:        Hours of labor.
        labor_rate:         Hourly rate in USD (default $75).
        parts_total:        Calculated sum of parts costs.
        labor_total:        Calculated labor_hours * labor_rate.
        total_cost:         parts_total + labor_total.
        notes:              Free-text notes.
        recommendations:    Future maintenance recommendations.
        photos_path:        Path to photos directory.
        video_path:         Path to video file.
        appointment_id:     Optional link to BusinessStore appointment.
        invoice_id:         Optional link to BusinessStore invoice.
        created_at:         When the record was created (ISO string).
        updated_at:         When the record was last modified (ISO string).
        metadata:           Additional key-value data.
    """
    record_id: str
    vehicle_id: str = ""
    customer_id: str = ""
    customer_name: str = ""
    vehicle_summary: str = ""
    date: str = ""
    service_type: str = ""
    services_performed: list[str] = field(default_factory=list)
    parts_used: list[dict] = field(default_factory=list)
    labor_hours: float = 0.0
    labor_rate: float = 75.0
    parts_total: float = 0.0
    labor_total: float = 0.0
    total_cost: float = 0.0
    notes: str = ""
    recommendations: str = ""
    photos_path: str = ""
    video_path: str = ""
    appointment_id: str = ""
    invoice_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 characters of the record ID for display."""
        return self.record_id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/dashboard use."""
        return {
            "record_id": self.record_id,
            "short_id": self.short_id,
            "vehicle_id": self.vehicle_id,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "vehicle_summary": self.vehicle_summary,
            "date": self.date,
            "service_type": self.service_type,
            "services_performed": self.services_performed,
            "parts_used": self.parts_used,
            "labor_hours": self.labor_hours,
            "labor_rate": self.labor_rate,
            "parts_total": self.parts_total,
            "labor_total": self.labor_total,
            "total_cost": self.total_cost,
            "notes": self.notes,
            "recommendations": self.recommendations,
            "photos_path": self.photos_path,
            "video_path": self.video_path,
            "appointment_id": self.appointment_id,
            "invoice_id": self.invoice_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ServiceRecord":
        """Convert a SQLite row to a ServiceRecord."""
        try:
            services = json.loads(row["services_performed"]) if row["services_performed"] else []
        except (json.JSONDecodeError, TypeError):
            services = []
        try:
            parts = json.loads(row["parts_used"]) if row["parts_used"] else []
        except (json.JSONDecodeError, TypeError):
            parts = []
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return cls(
            record_id=row["record_id"],
            vehicle_id=row["vehicle_id"],
            customer_id=row["customer_id"],
            customer_name=row["customer_name"],
            vehicle_summary=row["vehicle_summary"],
            date=row["date"],
            service_type=row["service_type"],
            services_performed=services,
            parts_used=parts,
            labor_hours=row["labor_hours"],
            labor_rate=row["labor_rate"],
            parts_total=row["parts_total"],
            labor_total=row["labor_total"],
            total_cost=row["total_cost"],
            notes=row["notes"],
            recommendations=row["recommendations"],
            photos_path=row["photos_path"],
            video_path=row["video_path"],
            appointment_id=row["appointment_id"],
            invoice_id=row["invoice_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# ServiceRecordStore — SQLite-backed vehicle + service record storage
# ---------------------------------------------------------------------------

class ServiceRecordStore:
    """SQLite-backed service record storage.

    Separate database with two tables: vehicles and service_records.
    Follows the BusinessStore pattern with optional change callbacks
    for real-time dashboard updates.

    Args:
        db_path:    Path to the SQLite database file.
        on_change:  Async callback fired after every mutation.
    """

    def __init__(
        self,
        db_path: str = "data/service_records.db",
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

        logger.info("ServiceRecordStore initialized (db=%s)", db_file)

    # -------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------

    def _init_db(self):
        """Create vehicles and service_records tables if they don't exist."""
        cur = self._conn.cursor()

        # Vehicles table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id TEXT UNIQUE NOT NULL,
                year TEXT NOT NULL DEFAULT '',
                make TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                vin TEXT NOT NULL DEFAULT '',
                owner_id TEXT NOT NULL DEFAULT '',
                owner_name TEXT NOT NULL DEFAULT '',
                color TEXT NOT NULL DEFAULT '',
                mileage INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_vehicles_owner
            ON vehicles(owner_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_vehicles_vin
            ON vehicles(vin)
        """)

        # Service records table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS service_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT UNIQUE NOT NULL,
                vehicle_id TEXT NOT NULL DEFAULT '',
                customer_id TEXT NOT NULL DEFAULT '',
                customer_name TEXT NOT NULL DEFAULT '',
                vehicle_summary TEXT NOT NULL DEFAULT '',
                date TEXT NOT NULL DEFAULT '',
                service_type TEXT NOT NULL DEFAULT '',
                services_performed TEXT NOT NULL DEFAULT '[]',
                parts_used TEXT NOT NULL DEFAULT '[]',
                labor_hours REAL NOT NULL DEFAULT 0.0,
                labor_rate REAL NOT NULL DEFAULT 75.0,
                parts_total REAL NOT NULL DEFAULT 0.0,
                labor_total REAL NOT NULL DEFAULT 0.0,
                total_cost REAL NOT NULL DEFAULT 0.0,
                notes TEXT NOT NULL DEFAULT '',
                recommendations TEXT NOT NULL DEFAULT '',
                photos_path TEXT NOT NULL DEFAULT '',
                video_path TEXT NOT NULL DEFAULT '',
                appointment_id TEXT NOT NULL DEFAULT '',
                invoice_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_vehicle
            ON service_records(vehicle_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_customer
            ON service_records(customer_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_date
            ON service_records(date)
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
                logger.debug("Service store on_change callback error", exc_info=True)

    # -------------------------------------------------------------------
    # Vehicle CRUD
    # -------------------------------------------------------------------

    def add_vehicle(
        self,
        year: str = "",
        make: str = "",
        model: str = "",
        vin: str = "",
        owner_id: str = "",
        owner_name: str = "",
        color: str = "",
        mileage: int = 0,
        notes: str = "",
        metadata: dict | None = None,
    ) -> Vehicle:
        """Create a new vehicle record.

        Args:
            year:       Model year.
            make:       Manufacturer.
            model:      Model name.
            vin:        Vehicle Identification Number.
            owner_id:   Link to BusinessStore customer.
            owner_name: Customer name (denormalized).
            color:      Vehicle color.
            mileage:    Current mileage.
            notes:      Free-text notes.
            metadata:   Optional extra data.

        Returns:
            The created Vehicle.
        """
        now = datetime.now(timezone.utc).isoformat()
        vehicle_id = str(uuid.uuid4())
        meta = metadata or {}

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO vehicles
                (vehicle_id, year, make, model, vin, owner_id, owner_name,
                 color, mileage, notes, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (vehicle_id, year, make, model, vin, owner_id, owner_name,
             color, mileage, notes, now, now, json.dumps(meta)),
        )
        self._conn.commit()

        vehicle = Vehicle(
            vehicle_id=vehicle_id, year=year, make=make, model=model,
            vin=vin, owner_id=owner_id, owner_name=owner_name,
            color=color, mileage=mileage, notes=notes,
            created_at=now, updated_at=now, metadata=meta,
        )
        logger.info("Vehicle added: %s (%s)", vehicle.display_name, vehicle.short_id)
        return vehicle

    def update_vehicle(self, vehicle_id: str, **kwargs) -> Vehicle | None:
        """Update a vehicle record.

        Args:
            vehicle_id: The vehicle's UUID.
            **kwargs:   Fields to update.

        Returns:
            The updated Vehicle, or None if not found.
        """
        existing = self.get_vehicle(vehicle_id)
        if existing is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        allowed = {"year", "make", "model", "vin", "owner_id", "owner_name",
                    "color", "mileage", "notes", "metadata"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}

        if not updates:
            return existing

        set_parts = ["updated_at = ?"]
        params: list[Any] = [now]

        for key, val in updates.items():
            if key == "metadata":
                set_parts.append(f"{key} = ?")
                params.append(json.dumps(val))
            else:
                set_parts.append(f"{key} = ?")
                params.append(val)

        params.append(vehicle_id)
        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE vehicles SET {', '.join(set_parts)} WHERE vehicle_id = ?",
            params,
        )
        self._conn.commit()

        logger.info("Vehicle updated: %s", vehicle_id[:8])
        return self.get_vehicle(vehicle_id)

    def get_vehicle(self, vehicle_id: str) -> Vehicle | None:
        """Retrieve a vehicle by ID.

        Args:
            vehicle_id: The vehicle's UUID.

        Returns:
            The Vehicle, or None if not found.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,))
        row = cur.fetchone()
        return Vehicle.from_row(row) if row else None

    def remove_vehicle(self, vehicle_id: str) -> bool:
        """Delete a vehicle record.

        Args:
            vehicle_id: The vehicle's UUID.

        Returns:
            True if deleted, False if not found.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM vehicles WHERE vehicle_id = ?", (vehicle_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Vehicle removed: %s", vehicle_id[:8])
        return deleted

    def list_vehicles(self, owner_id: str | None = None, limit: int = 100) -> list[Vehicle]:
        """List vehicles, optionally filtered by owner.

        Args:
            owner_id: Filter to this customer's vehicles.
            limit:    Maximum records to return.

        Returns:
            List of Vehicle objects.
        """
        cur = self._conn.cursor()
        if owner_id:
            cur.execute(
                "SELECT * FROM vehicles WHERE owner_id = ? ORDER BY updated_at DESC LIMIT ?",
                (owner_id, limit),
            )
        else:
            cur.execute("SELECT * FROM vehicles ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [Vehicle.from_row(row) for row in cur.fetchall()]

    def search_vehicles(self, query: str) -> list[Vehicle]:
        """Search vehicles by make, model, VIN, or owner name.

        Args:
            query: Search string (case-insensitive partial match).

        Returns:
            List of matching Vehicle objects.
        """
        cur = self._conn.cursor()
        like = f"%{query}%"
        cur.execute(
            """SELECT * FROM vehicles
               WHERE make LIKE ? OR model LIKE ? OR vin LIKE ? OR owner_name LIKE ?
               ORDER BY updated_at DESC LIMIT 50""",
            (like, like, like, like),
        )
        return [Vehicle.from_row(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------
    # Service Record CRUD
    # -------------------------------------------------------------------

    def _calc_totals(
        self,
        parts_used: list[dict],
        labor_hours: float,
        labor_rate: float,
    ) -> tuple[float, float, float]:
        """Calculate parts_total, labor_total, and total_cost.

        Args:
            parts_used:  List of part dicts with quantity and unit_cost.
            labor_hours: Hours of labor.
            labor_rate:  Hourly rate.

        Returns:
            Tuple of (parts_total, labor_total, total_cost).
        """
        parts_total = 0.0
        for part in parts_used:
            qty = float(part.get("quantity", 0))
            cost = float(part.get("unit_cost", 0))
            parts_total += qty * cost
        labor_total = labor_hours * labor_rate
        return round(parts_total, 2), round(labor_total, 2), round(parts_total + labor_total, 2)

    def add_record(
        self,
        vehicle_id: str = "",
        customer_id: str = "",
        customer_name: str = "",
        date: str = "",
        service_type: str = "",
        services_performed: list[str] | None = None,
        parts_used: list[dict] | None = None,
        labor_hours: float = 0.0,
        labor_rate: float = 75.0,
        notes: str = "",
        recommendations: str = "",
        photos_path: str = "",
        video_path: str = "",
        appointment_id: str = "",
        invoice_id: str = "",
        metadata: dict | None = None,
    ) -> ServiceRecord:
        """Create a new service record.

        Auto-calculates parts_total, labor_total, total_cost from parts_used
        and labor fields. Looks up the vehicle to build vehicle_summary.

        Args:
            vehicle_id:         Link to vehicles table.
            customer_id:        Link to BusinessStore customer.
            customer_name:      Customer name (denormalized).
            date:               Service date (YYYY-MM-DD).
            service_type:       diagnostic, maintenance, repair, or inspection.
            services_performed: List of service descriptions.
            parts_used:         List of part dicts.
            labor_hours:        Hours of labor.
            labor_rate:         Hourly rate in USD.
            notes:              Free-text notes.
            recommendations:    Future maintenance recommendations.
            photos_path:        Path to photos.
            video_path:         Path to video.
            appointment_id:     Optional link to appointment.
            invoice_id:         Optional link to invoice.
            metadata:           Optional extra data.

        Returns:
            The created ServiceRecord.
        """
        now = datetime.now(timezone.utc).isoformat()
        record_id = str(uuid.uuid4())
        meta = metadata or {}
        services = services_performed or []
        parts = parts_used or []

        # Build vehicle summary from the vehicle record
        vehicle_summary = ""
        vehicle = self.get_vehicle(vehicle_id)
        if vehicle:
            vehicle_summary = vehicle.display_name

        parts_total, labor_total, total_cost = self._calc_totals(parts, labor_hours, labor_rate)

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO service_records
                (record_id, vehicle_id, customer_id, customer_name,
                 vehicle_summary, date, service_type, services_performed,
                 parts_used, labor_hours, labor_rate, parts_total,
                 labor_total, total_cost, notes, recommendations,
                 photos_path, video_path, appointment_id, invoice_id,
                 created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (record_id, vehicle_id, customer_id, customer_name,
             vehicle_summary, date, service_type, json.dumps(services),
             json.dumps(parts), labor_hours, labor_rate, parts_total,
             labor_total, total_cost, notes, recommendations,
             photos_path, video_path, appointment_id, invoice_id,
             now, now, json.dumps(meta)),
        )
        self._conn.commit()

        record = ServiceRecord(
            record_id=record_id, vehicle_id=vehicle_id,
            customer_id=customer_id, customer_name=customer_name,
            vehicle_summary=vehicle_summary, date=date,
            service_type=service_type, services_performed=services,
            parts_used=parts, labor_hours=labor_hours, labor_rate=labor_rate,
            parts_total=parts_total, labor_total=labor_total,
            total_cost=total_cost, notes=notes, recommendations=recommendations,
            photos_path=photos_path, video_path=video_path,
            appointment_id=appointment_id, invoice_id=invoice_id,
            created_at=now, updated_at=now, metadata=meta,
        )
        logger.info("Service record added: %s for %s (%s)",
                     record.short_id, vehicle_summary, customer_name)
        return record

    def update_record(self, record_id: str, **kwargs) -> ServiceRecord | None:
        """Update a service record.

        Recalculates totals if parts_used, labor_hours, or labor_rate change.

        Args:
            record_id: The record's UUID.
            **kwargs:  Fields to update.

        Returns:
            The updated ServiceRecord, or None if not found.
        """
        existing = self.get_record(record_id)
        if existing is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        allowed = {"vehicle_id", "customer_id", "customer_name", "vehicle_summary",
                    "date", "service_type", "services_performed", "parts_used",
                    "labor_hours", "labor_rate", "notes", "recommendations",
                    "photos_path", "video_path", "appointment_id", "invoice_id",
                    "metadata"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}

        if not updates:
            return existing

        # Recalculate totals if cost-affecting fields changed
        parts = updates.get("parts_used", existing.parts_used)
        hours = float(updates.get("labor_hours", existing.labor_hours))
        rate = float(updates.get("labor_rate", existing.labor_rate))
        parts_total, labor_total, total_cost = self._calc_totals(parts, hours, rate)

        # Always update totals alongside the other fields
        updates["parts_total"] = parts_total
        updates["labor_total"] = labor_total
        updates["total_cost"] = total_cost

        set_parts_sql = ["updated_at = ?"]
        params: list[Any] = [now]

        json_fields = {"services_performed", "parts_used", "metadata"}
        for key, val in updates.items():
            set_parts_sql.append(f"{key} = ?")
            if key in json_fields:
                params.append(json.dumps(val))
            else:
                params.append(val)

        params.append(record_id)
        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE service_records SET {', '.join(set_parts_sql)} WHERE record_id = ?",
            params,
        )
        self._conn.commit()

        logger.info("Service record updated: %s", record_id[:8])
        return self.get_record(record_id)

    def get_record(self, record_id: str) -> ServiceRecord | None:
        """Retrieve a service record by ID.

        Args:
            record_id: The record's UUID.

        Returns:
            The ServiceRecord, or None if not found.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM service_records WHERE record_id = ?", (record_id,))
        row = cur.fetchone()
        return ServiceRecord.from_row(row) if row else None

    def remove_record(self, record_id: str) -> bool:
        """Delete a service record.

        Args:
            record_id: The record's UUID.

        Returns:
            True if deleted, False if not found.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM service_records WHERE record_id = ?", (record_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Service record removed: %s", record_id[:8])
        return deleted

    def list_records(
        self,
        vehicle_id: str | None = None,
        customer_id: str | None = None,
        limit: int = 100,
    ) -> list[ServiceRecord]:
        """List service records with optional filters.

        Args:
            vehicle_id:  Filter to records for this vehicle.
            customer_id: Filter to records for this customer.
            limit:       Maximum records to return.

        Returns:
            List of ServiceRecord objects, newest first.
        """
        cur = self._conn.cursor()
        conditions = []
        params: list[Any] = []

        if vehicle_id:
            conditions.append("vehicle_id = ?")
            params.append(vehicle_id)
        if customer_id:
            conditions.append("customer_id = ?")
            params.append(customer_id)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        cur.execute(
            f"SELECT * FROM service_records {where} ORDER BY date DESC, created_at DESC LIMIT ?",
            params,
        )
        return [ServiceRecord.from_row(row) for row in cur.fetchall()]

    def get_vehicle_history(self, vehicle_id: str) -> list[ServiceRecord]:
        """Get all service records for a vehicle, newest first.

        Args:
            vehicle_id: The vehicle's UUID.

        Returns:
            List of ServiceRecord objects.
        """
        return self.list_records(vehicle_id=vehicle_id)

    def get_customer_history(self, customer_id: str) -> list[ServiceRecord]:
        """Get all service records for a customer, newest first.

        Args:
            customer_id: The customer's UUID.

        Returns:
            List of ServiceRecord objects.
        """
        return self.list_records(customer_id=customer_id)

    # -------------------------------------------------------------------
    # PDF Generation
    # -------------------------------------------------------------------

    def generate_pdf(self, record_id: str) -> str | None:
        """Generate a professional PDF service report.

        Creates a PDF at data/service_reports/{record_id}.pdf with Doppler
        Cycles branding, vehicle info, services, parts table, labor, costs,
        notes, and recommendations.

        Args:
            record_id: The service record's UUID.

        Returns:
            Path to the generated PDF file, or None if record not found.
        """
        record = self.get_record(record_id)
        if record is None:
            return None

        vehicle = self.get_vehicle(record.vehicle_id)

        # Ensure output directory exists
        report_dir = Path("data/service_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = str(report_dir / f"{record_id}.pdf")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- Header ---
        pdf.set_font("Helvetica", "B", 22)
        pdf.cell(0, 10, "DOPPLER CYCLES", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 6, "Mobile Motorcycle Diagnostics", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.cell(0, 6, "Portland Metro Area  |  Gresham, Oregon", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        # --- Title ---
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "SERVICE REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        # --- Vehicle Info Box ---
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "  Vehicle Information", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 10)

        if vehicle:
            pdf.cell(95, 6, f"  Vehicle: {vehicle.display_name}", new_x="RIGHT")
            pdf.cell(95, 6, f"Color: {vehicle.color or 'N/A'}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(95, 6, f"  VIN: {vehicle.vin or 'N/A'}", new_x="RIGHT")
            pdf.cell(95, 6, f"Mileage: {vehicle.mileage:,} mi" if vehicle.mileage else "Mileage: N/A",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.cell(95, 6, f"  Owner: {record.customer_name or vehicle.owner_name}",
                     new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(0, 6, f"  Vehicle: {record.vehicle_summary or 'N/A'}",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 6, f"  Owner: {record.customer_name}",
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # --- Service Details ---
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "  Service Details", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(95, 6, f"  Date: {record.date}", new_x="RIGHT")
        pdf.cell(95, 6, f"Type: {record.service_type.title() if record.service_type else 'N/A'}",
                 new_x="LMARGIN", new_y="NEXT")

        if record.services_performed:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "  Services Performed:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            for svc in record.services_performed:
                pdf.cell(0, 5, f"    - {svc}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # --- Parts Table ---
        if record.parts_used:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "  Parts Used", new_x="LMARGIN", new_y="NEXT", fill=True)

            # Table header
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(70, 6, "  Description", border="B")
            pdf.cell(35, 6, "Part #", border="B", align="C")
            pdf.cell(20, 6, "Qty", border="B", align="C")
            pdf.cell(30, 6, "Unit Cost", border="B", align="R")
            pdf.cell(35, 6, "Total", border="B", align="R", new_x="LMARGIN", new_y="NEXT")

            # Table rows
            pdf.set_font("Helvetica", "", 9)
            for part in record.parts_used:
                desc = part.get("description", "")
                pn = part.get("part_number", "")
                qty = int(part.get("quantity", 0))
                uc = float(part.get("unit_cost", 0))
                line_total = qty * uc
                pdf.cell(70, 5, f"  {desc}")
                pdf.cell(35, 5, pn, align="C")
                pdf.cell(20, 5, str(qty), align="C")
                pdf.cell(30, 5, f"${uc:.2f}", align="R")
                pdf.cell(35, 5, f"${line_total:.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

        # --- Labor ---
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "  Labor", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  {record.labor_hours:.1f} hours x ${record.labor_rate:.2f}/hr = ${record.labor_total:.2f}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # --- Cost Summary ---
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "  Cost Summary", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(120, 6, "  Parts Total:", new_x="RIGHT")
        pdf.cell(70, 6, f"${record.parts_total:.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(120, 6, "  Labor Total:", new_x="RIGHT")
        pdf.cell(70, 6, f"${record.labor_total:.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(120, 7, "  GRAND TOTAL:", new_x="RIGHT")
        pdf.cell(70, 7, f"${record.total_cost:.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # --- Notes ---
        if record.notes:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "  Notes", new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, f"  {record.notes}")
            pdf.ln(4)

        # --- Recommendations ---
        if record.recommendations:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "  Recommendations", new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, f"  {record.recommendations}")
            pdf.ln(4)

        # --- Footer ---
        pdf.ln(8)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, "Thank you for choosing Doppler Cycles", new_x="LMARGIN", new_y="NEXT", align="C")
        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        pdf.cell(0, 5, f"Report generated: {generated}", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.output(pdf_path)
        logger.info("PDF generated: %s", pdf_path)
        return pdf_path

    # -------------------------------------------------------------------
    # Dashboard methods
    # -------------------------------------------------------------------

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Return all service data for dashboard broadcast.

        Returns:
            Dict with vehicles, records, and status.
        """
        return {
            "vehicles": [v.to_dict() for v in self.list_vehicles()],
            "records": [r.to_dict() for r in self.list_records()],
            "status": self.get_status(),
        }

    def get_status(self) -> dict[str, Any]:
        """Return summary stats for the dashboard.

        Returns:
            Dict with total_vehicles, total_records, records_this_month, total_revenue.
        """
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) FROM vehicles")
        total_vehicles = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM service_records")
        total_records = cur.fetchone()[0]

        # Records this month
        month_start = datetime.now().strftime("%Y-%m-01")
        cur.execute(
            "SELECT COUNT(*) FROM service_records WHERE date >= ?",
            (month_start,),
        )
        records_this_month = cur.fetchone()[0]

        cur.execute("SELECT COALESCE(SUM(total_cost), 0) FROM service_records")
        total_revenue = cur.fetchone()[0]

        return {
            "total_vehicles": total_vehicles,
            "total_records": total_records,
            "records_this_month": records_this_month,
            "total_revenue": round(total_revenue, 2),
        }

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------

    def close(self):
        """Close the SQLite connection."""
        self._conn.close()
        logger.info("ServiceRecordStore closed")


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    try:
        store = ServiceRecordStore(db_path=tmp.name)

        # --- Vehicle CRUD ---
        print("\n=== Vehicle CRUD ===")
        v1 = store.add_vehicle(
            year="2019", make="Harley-Davidson", model="Street Glide",
            vin="1HD1KTP19KB123456", owner_id="cust-001", owner_name="Mike R.",
            color="Vivid Black", mileage=45200,
        )
        print(f"  Created: {v1.display_name} ({v1.short_id})")
        assert v1.display_name == "2019 Harley-Davidson Street Glide"

        v2 = store.add_vehicle(
            year="2015", make="Yamaha", model="FZ-09",
            vin="JYARN33E0FA012345", owner_id="cust-002", owner_name="Sarah K.",
            color="Matte Gray", mileage=22100,
        )
        print(f"  Created: {v2.display_name} ({v2.short_id})")

        updated = store.update_vehicle(v1.vehicle_id, mileage=45500)
        assert updated is not None
        assert updated.mileage == 45500
        print(f"  Updated mileage: {updated.mileage}")

        vehicles = store.list_vehicles()
        assert len(vehicles) == 2
        print(f"  Listed: {len(vehicles)} vehicles")

        results = store.search_vehicles("Harley")
        assert len(results) == 1
        print(f"  Search 'Harley': {len(results)} result")

        owner_vehicles = store.list_vehicles(owner_id="cust-001")
        assert len(owner_vehicles) == 1
        print(f"  Owner filter: {len(owner_vehicles)} vehicle")

        # --- Service Record CRUD ---
        print("\n=== Service Record CRUD ===")
        r1 = store.add_record(
            vehicle_id=v1.vehicle_id,
            customer_id="cust-001",
            customer_name="Mike R.",
            date="2026-02-10",
            service_type="diagnostic",
            services_performed=["Engine diagnostic", "Code read", "Compression test"],
            parts_used=[
                {"description": "Oil filter", "part_number": "HD-6371", "quantity": 1, "unit_cost": 12.00},
                {"description": "Spark plugs", "part_number": "NGK-6289", "quantity": 4, "unit_cost": 12.00},
            ],
            labor_hours=2.0,
            labor_rate=75.0,
            notes="Found loose exhaust bracket on right side.",
            recommendations="Replace exhaust gasket at next service. Check belt tension.",
        )
        print(f"  Created: {r1.short_id} — {r1.vehicle_summary}")
        assert r1.parts_total == 60.0   # 12 + 4*12 = 60
        assert r1.labor_total == 150.0  # 2 * 75
        assert r1.total_cost == 210.0
        print(f"  Totals: parts=${r1.parts_total}, labor=${r1.labor_total}, total=${r1.total_cost}")

        r2 = store.add_record(
            vehicle_id=v2.vehicle_id,
            customer_id="cust-002",
            customer_name="Sarah K.",
            date="2026-02-08",
            service_type="maintenance",
            services_performed=["Oil change", "Chain clean and lube"],
            labor_hours=1.0,
        )
        print(f"  Created: {r2.short_id} — {r2.vehicle_summary}")

        updated_r = store.update_record(r1.record_id, labor_hours=2.5)
        assert updated_r is not None
        assert updated_r.labor_total == 187.5
        print(f"  Updated labor: {updated_r.labor_hours}hrs = ${updated_r.labor_total}")

        history = store.get_vehicle_history(v1.vehicle_id)
        assert len(history) == 1
        print(f"  Vehicle history: {len(history)} record(s)")

        records = store.list_records()
        assert len(records) == 2
        print(f"  All records: {len(records)}")

        # --- PDF Generation ---
        print("\n=== PDF Generation ===")
        pdf_path = store.generate_pdf(r1.record_id)
        assert pdf_path is not None
        assert os.path.exists(pdf_path)
        size = os.path.getsize(pdf_path)
        print(f"  Generated: {pdf_path} ({size:,} bytes)")
        # Clean up test PDF
        os.unlink(pdf_path)

        # --- Status ---
        print("\n=== Status ===")
        status = store.get_status()
        print(f"  Vehicles: {status['total_vehicles']}")
        print(f"  Records: {status['total_records']}")
        print(f"  This month: {status['records_this_month']}")
        print(f"  Revenue: ${status['total_revenue']:.2f}")

        # --- Broadcast dict ---
        bd = store.to_broadcast_dict()
        assert "vehicles" in bd
        assert "records" in bd
        assert "status" in bd
        print(f"  Broadcast: {len(bd['vehicles'])} vehicles, {len(bd['records'])} records")

        # --- Deletion ---
        print("\n=== Deletion ===")
        assert store.remove_record(r2.record_id)
        assert len(store.list_records()) == 1
        print("  Removed record r2")

        assert store.remove_vehicle(v2.vehicle_id)
        assert len(store.list_vehicles()) == 1
        print("  Removed vehicle v2")

        store.close()
        print("\nAll tests passed!")

    finally:
        os.unlink(tmp.name)
        # Clean up report dir if empty
        report_dir = Path("data/service_reports")
        if report_dir.exists() and not list(report_dir.iterdir()):
            report_dir.rmdir()
