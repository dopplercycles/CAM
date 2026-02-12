"""
CAM Business Agent

Local in-process agent that intercepts business-related tasks and handles
them with business-specific prompting. Manages customer contacts, service
appointments, invoices, and parts/tools inventory for Doppler Cycles.

"Local" means this runs inside the server process — no WebSocket, no
network hop. It's checked after the research agent in dispatch_to_agent().
If the task isn't business-related, it returns None and the flow
continues to remote agents unchanged.

Data lives in a single SQLite database with four tables: customers,
appointments, invoices, and inventory. All CRUD is handled by the
BusinessStore class, which follows the ResearchStore/ScoutStore pattern.

Usage:
    from agents.business_agent import BusinessAgent, BusinessStore

    store = BusinessStore(db_path="data/business.db")
    agent = BusinessAgent(
        router=orchestrator.router,
        persona=persona,
        long_term_memory=long_term_memory,
        business_store=store,
        event_logger=event_logger,
    )

    # Called from dispatch_to_agent() — returns str or None
    result = await agent.try_handle(task, plan)
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("cam.business_agent")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AppointmentStatus(Enum):
    """Status of a service appointment."""
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class InvoiceStatus(Enum):
    """Status of an invoice."""
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Customer:
    """A Doppler Cycles customer.

    Attributes:
        customer_id: Unique identifier (UUID string).
        name:        Customer's full name.
        phone:       Phone number.
        email:       Email address.
        bike_info:   Description of customer's motorcycle(s).
        notes:       Free-text notes about the customer.
        created_at:  When the record was created (ISO string).
        updated_at:  When the record was last modified (ISO string).
        metadata:    Additional key-value data.
    """
    customer_id: str
    name: str
    phone: str = ""
    email: str = ""
    bike_info: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 characters of the customer ID for display."""
        return self.customer_id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/dashboard use."""
        return {
            "customer_id": self.customer_id,
            "short_id": self.short_id,
            "name": self.name,
            "phone": self.phone,
            "email": self.email,
            "bike_info": self.bike_info,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Customer":
        """Convert a SQLite row to a Customer."""
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return cls(
            customer_id=row["customer_id"],
            name=row["name"],
            phone=row["phone"],
            email=row["email"],
            bike_info=row["bike_info"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=meta,
        )


@dataclass
class Appointment:
    """A service appointment.

    Attributes:
        appointment_id: Unique identifier (UUID string).
        customer_id:    Link to the customer record.
        customer_name:  Denormalized customer name (avoids JOINs).
        date:           Appointment date (YYYY-MM-DD).
        time:           Appointment time (HH:MM).
        bike:           Motorcycle being serviced.
        service_type:   Type of service (diagnostic, maintenance, etc.).
        status:         Current status (scheduled, confirmed, etc.).
        notes:          Free-text notes.
        location:       Service location (address or "mobile").
        estimated_cost: Estimated cost in USD.
        created_at:     When the record was created (ISO string).
        updated_at:     When the record was last modified (ISO string).
        metadata:       Additional key-value data.
    """
    appointment_id: str
    customer_id: str = ""
    customer_name: str = ""
    date: str = ""
    time: str = ""
    bike: str = ""
    service_type: str = ""
    status: str = AppointmentStatus.SCHEDULED.value
    notes: str = ""
    location: str = ""
    estimated_cost: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 characters of the appointment ID for display."""
        return self.appointment_id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/dashboard use."""
        return {
            "appointment_id": self.appointment_id,
            "short_id": self.short_id,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "date": self.date,
            "time": self.time,
            "bike": self.bike,
            "service_type": self.service_type,
            "status": self.status,
            "notes": self.notes,
            "location": self.location,
            "estimated_cost": self.estimated_cost,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Appointment":
        """Convert a SQLite row to an Appointment."""
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return cls(
            appointment_id=row["appointment_id"],
            customer_id=row["customer_id"],
            customer_name=row["customer_name"],
            date=row["date"],
            time=row["time"],
            bike=row["bike"],
            service_type=row["service_type"],
            status=row["status"],
            notes=row["notes"],
            location=row["location"],
            estimated_cost=row["estimated_cost"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=meta,
        )


@dataclass
class Invoice:
    """A service invoice.

    Attributes:
        invoice_id:     Unique identifier (UUID string).
        invoice_number: Human-readable number (DC-0001 format).
        customer_id:    Link to the customer record.
        customer_name:  Denormalized customer name (avoids JOINs).
        date:           Invoice date (YYYY-MM-DD).
        items:          List of line item dicts (description, qty, unit_price).
        labor_hours:    Total labor hours billed.
        labor_rate:     Hourly labor rate in USD (default $75).
        subtotal:       Sum of parts + labor before tax.
        tax_rate:       Tax rate as decimal (0.0 for Oregon — no sales tax).
        total:          Final total after tax.
        status:         Current status (draft, sent, paid, etc.).
        notes:          Free-text notes.
        appointment_id: Optional link to the originating appointment.
        created_at:     When the record was created (ISO string).
        updated_at:     When the record was last modified (ISO string).
        metadata:       Additional key-value data.
    """
    invoice_id: str
    invoice_number: str = ""
    customer_id: str = ""
    customer_name: str = ""
    date: str = ""
    items: list[dict] = field(default_factory=list)
    labor_hours: float = 0.0
    labor_rate: float = 75.0
    subtotal: float = 0.0
    tax_rate: float = 0.0
    total: float = 0.0
    status: str = InvoiceStatus.DRAFT.value
    notes: str = ""
    appointment_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 characters of the invoice ID for display."""
        return self.invoice_id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/dashboard use."""
        return {
            "invoice_id": self.invoice_id,
            "short_id": self.short_id,
            "invoice_number": self.invoice_number,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "date": self.date,
            "items": self.items,
            "labor_hours": self.labor_hours,
            "labor_rate": self.labor_rate,
            "subtotal": self.subtotal,
            "tax_rate": self.tax_rate,
            "total": self.total,
            "status": self.status,
            "notes": self.notes,
            "appointment_id": self.appointment_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Invoice":
        """Convert a SQLite row to an Invoice."""
        try:
            items = json.loads(row["items"]) if row["items"] else []
        except (json.JSONDecodeError, TypeError):
            items = []
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return cls(
            invoice_id=row["invoice_id"],
            invoice_number=row["invoice_number"],
            customer_id=row["customer_id"],
            customer_name=row["customer_name"],
            date=row["date"],
            items=items,
            labor_hours=row["labor_hours"],
            labor_rate=row["labor_rate"],
            subtotal=row["subtotal"],
            tax_rate=row["tax_rate"],
            total=row["total"],
            status=row["status"],
            notes=row["notes"],
            appointment_id=row["appointment_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=meta,
        )


@dataclass
class InventoryItem:
    """A parts/tools inventory item.

    Attributes:
        item_id:           Unique identifier (UUID string).
        name:              Item name.
        category:          Category (parts, tools, consumables, etc.).
        quantity:          Current quantity on hand.
        cost:              Unit cost in USD.
        location:          Where the item is stored.
        reorder_threshold: Reorder when quantity drops to this level.
        supplier:          Supplier name.
        part_number:       Manufacturer/supplier part number.
        notes:             Free-text notes.
        created_at:        When the record was created (ISO string).
        updated_at:        When the record was last modified (ISO string).
        metadata:          Additional key-value data.
    """
    item_id: str
    name: str = ""
    category: str = ""
    quantity: int = 0
    cost: float = 0.0
    location: str = ""
    reorder_threshold: int = 0
    supplier: str = ""
    part_number: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 characters of the item ID for display."""
        return self.item_id[:8]

    @property
    def needs_reorder(self) -> bool:
        """True if quantity is at or below the reorder threshold."""
        return self.reorder_threshold > 0 and self.quantity <= self.reorder_threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/dashboard use."""
        return {
            "item_id": self.item_id,
            "short_id": self.short_id,
            "name": self.name,
            "category": self.category,
            "quantity": self.quantity,
            "cost": self.cost,
            "location": self.location,
            "reorder_threshold": self.reorder_threshold,
            "supplier": self.supplier,
            "part_number": self.part_number,
            "notes": self.notes,
            "needs_reorder": self.needs_reorder,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "InventoryItem":
        """Convert a SQLite row to an InventoryItem."""
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return cls(
            item_id=row["item_id"],
            name=row["name"],
            category=row["category"],
            quantity=row["quantity"],
            cost=row["cost"],
            location=row["location"],
            reorder_threshold=row["reorder_threshold"],
            supplier=row["supplier"],
            part_number=row["part_number"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# BusinessStore — single SQLite DB, four tables
# ---------------------------------------------------------------------------

class BusinessStore:
    """SQLite-backed business data storage.

    Single database with four tables: customers, appointments, invoices,
    inventory. Follows the ResearchStore/ScoutStore pattern with optional
    change callbacks for real-time dashboard updates.

    Args:
        db_path:    Path to the SQLite database file.
        on_change:  Async callback fired after every mutation.
    """

    def __init__(
        self,
        db_path: str = "data/business.db",
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

        logger.info("BusinessStore initialized (db=%s)", db_file)

    # -------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------

    def _init_db(self):
        """Create all four tables and indexes if they don't exist."""
        cur = self._conn.cursor()

        # Customers table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                phone TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL DEFAULT '',
                bike_info TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_customers_name
            ON customers(name)
        """)

        # Appointments table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                appointment_id TEXT UNIQUE NOT NULL,
                customer_id TEXT NOT NULL DEFAULT '',
                customer_name TEXT NOT NULL DEFAULT '',
                date TEXT NOT NULL DEFAULT '',
                time TEXT NOT NULL DEFAULT '',
                bike TEXT NOT NULL DEFAULT '',
                service_type TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'scheduled',
                notes TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                estimated_cost REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_appointments_date
            ON appointments(date)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_appointments_status
            ON appointments(status)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_appointments_customer
            ON appointments(customer_id)
        """)

        # Invoices table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id TEXT UNIQUE NOT NULL,
                invoice_number TEXT NOT NULL DEFAULT '',
                customer_id TEXT NOT NULL DEFAULT '',
                customer_name TEXT NOT NULL DEFAULT '',
                date TEXT NOT NULL DEFAULT '',
                items TEXT NOT NULL DEFAULT '[]',
                labor_hours REAL NOT NULL DEFAULT 0.0,
                labor_rate REAL NOT NULL DEFAULT 75.0,
                subtotal REAL NOT NULL DEFAULT 0.0,
                tax_rate REAL NOT NULL DEFAULT 0.0,
                total REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'draft',
                notes TEXT NOT NULL DEFAULT '',
                appointment_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_invoices_status
            ON invoices(status)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_invoices_customer
            ON invoices(customer_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_invoices_number
            ON invoices(invoice_number)
        """)

        # Inventory table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT '',
                quantity INTEGER NOT NULL DEFAULT 0,
                cost REAL NOT NULL DEFAULT 0.0,
                location TEXT NOT NULL DEFAULT '',
                reorder_threshold INTEGER NOT NULL DEFAULT 0,
                supplier TEXT NOT NULL DEFAULT '',
                part_number TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_inventory_category
            ON inventory(category)
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
                logger.debug("Business store on_change callback error", exc_info=True)

    # -------------------------------------------------------------------
    # Customer CRUD
    # -------------------------------------------------------------------

    def add_customer(
        self,
        name: str,
        phone: str = "",
        email: str = "",
        bike_info: str = "",
        notes: str = "",
        metadata: dict | None = None,
    ) -> Customer:
        """Create a new customer record.

        Args:
            name:      Customer's full name.
            phone:     Phone number.
            email:     Email address.
            bike_info: Motorcycle description.
            notes:     Free-text notes.
            metadata:  Optional extra data.

        Returns:
            The created Customer.
        """
        now = datetime.now(timezone.utc).isoformat()
        customer_id = str(uuid.uuid4())
        meta = metadata or {}

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO customers
                (customer_id, name, phone, email, bike_info, notes,
                 created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (customer_id, name, phone, email, bike_info, notes,
             now, now, json.dumps(meta)),
        )
        self._conn.commit()

        customer = Customer(
            customer_id=customer_id, name=name, phone=phone, email=email,
            bike_info=bike_info, notes=notes, created_at=now, updated_at=now,
            metadata=meta,
        )
        logger.info("Customer added: %s (%s)", name, customer.short_id)
        return customer

    def update_customer(self, customer_id: str, **kwargs) -> Customer | None:
        """Update a customer record.

        Args:
            customer_id: The customer's UUID.
            **kwargs:    Fields to update (name, phone, email, bike_info, notes, metadata).

        Returns:
            The updated Customer, or None if not found.
        """
        existing = self.get_customer(customer_id)
        if existing is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        allowed = {"name", "phone", "email", "bike_info", "notes", "metadata"}
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

        params.append(customer_id)
        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE customers SET {', '.join(set_parts)} WHERE customer_id = ?",
            params,
        )
        self._conn.commit()

        logger.info("Customer updated: %s", customer_id[:8])
        return self.get_customer(customer_id)

    def get_customer(self, customer_id: str) -> Customer | None:
        """Retrieve a customer by ID.

        Args:
            customer_id: The customer's UUID.

        Returns:
            The Customer, or None if not found.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
        row = cur.fetchone()
        return Customer.from_row(row) if row else None

    def remove_customer(self, customer_id: str) -> bool:
        """Delete a customer record.

        Args:
            customer_id: The customer's UUID.

        Returns:
            True if deleted, False if not found.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM customers WHERE customer_id = ?", (customer_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Customer removed: %s", customer_id[:8])
        return deleted

    def list_customers(self, limit: int = 100) -> list[Customer]:
        """List all customers, ordered by name.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of Customer objects.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM customers ORDER BY name ASC LIMIT ?", (limit,))
        return [Customer.from_row(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------
    # Appointment CRUD
    # -------------------------------------------------------------------

    def add_appointment(
        self,
        customer_id: str = "",
        customer_name: str = "",
        date: str = "",
        time: str = "",
        bike: str = "",
        service_type: str = "",
        status: str = AppointmentStatus.SCHEDULED.value,
        notes: str = "",
        location: str = "",
        estimated_cost: float = 0.0,
        metadata: dict | None = None,
    ) -> Appointment:
        """Create a new appointment record.

        Args:
            customer_id:    Link to customer.
            customer_name:  Denormalized name.
            date:           Date string (YYYY-MM-DD).
            time:           Time string (HH:MM).
            bike:           Motorcycle being serviced.
            service_type:   Type of service.
            status:         Initial status.
            notes:          Free-text notes.
            location:       Service location.
            estimated_cost: Estimated cost.
            metadata:       Optional extra data.

        Returns:
            The created Appointment.
        """
        now = datetime.now(timezone.utc).isoformat()
        appointment_id = str(uuid.uuid4())
        meta = metadata or {}

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO appointments
                (appointment_id, customer_id, customer_name, date, time,
                 bike, service_type, status, notes, location, estimated_cost,
                 created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (appointment_id, customer_id, customer_name, date, time,
             bike, service_type, status, notes, location, estimated_cost,
             now, now, json.dumps(meta)),
        )
        self._conn.commit()

        appt = Appointment(
            appointment_id=appointment_id, customer_id=customer_id,
            customer_name=customer_name, date=date, time=time,
            bike=bike, service_type=service_type, status=status,
            notes=notes, location=location, estimated_cost=estimated_cost,
            created_at=now, updated_at=now, metadata=meta,
        )
        logger.info("Appointment added: %s for %s on %s", appt.short_id, customer_name, date)
        return appt

    def update_appointment(self, appointment_id: str, **kwargs) -> Appointment | None:
        """Update an appointment record.

        Args:
            appointment_id: The appointment's UUID.
            **kwargs:       Fields to update.

        Returns:
            The updated Appointment, or None if not found.
        """
        existing = self.get_appointment(appointment_id)
        if existing is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        allowed = {"customer_id", "customer_name", "date", "time", "bike",
                    "service_type", "status", "notes", "location",
                    "estimated_cost", "metadata"}
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

        params.append(appointment_id)
        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE appointments SET {', '.join(set_parts)} WHERE appointment_id = ?",
            params,
        )
        self._conn.commit()

        logger.info("Appointment updated: %s", appointment_id[:8])
        return self.get_appointment(appointment_id)

    def get_appointment(self, appointment_id: str) -> Appointment | None:
        """Retrieve an appointment by ID.

        Args:
            appointment_id: The appointment's UUID.

        Returns:
            The Appointment, or None if not found.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,))
        row = cur.fetchone()
        return Appointment.from_row(row) if row else None

    def remove_appointment(self, appointment_id: str) -> bool:
        """Delete an appointment record.

        Args:
            appointment_id: The appointment's UUID.

        Returns:
            True if deleted, False if not found.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM appointments WHERE appointment_id = ?", (appointment_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Appointment removed: %s", appointment_id[:8])
        return deleted

    def list_appointments(self, limit: int = 100) -> list[Appointment]:
        """List all appointments, ordered by date descending (newest first).

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of Appointment objects.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM appointments ORDER BY date DESC, time DESC LIMIT ?",
            (limit,),
        )
        return [Appointment.from_row(row) for row in cur.fetchall()]

    def get_customer_appointments(self, customer_id: str) -> list[Appointment]:
        """Get all appointments for a specific customer.

        Args:
            customer_id: The customer's UUID.

        Returns:
            List of Appointment objects, newest first.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM appointments WHERE customer_id = ? ORDER BY date DESC",
            (customer_id,),
        )
        return [Appointment.from_row(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------
    # Invoice CRUD
    # -------------------------------------------------------------------

    def _next_invoice_number(self, prefix: str = "DC") -> str:
        """Generate the next auto-increment invoice number (e.g., DC-0001).

        Args:
            prefix: Invoice number prefix.

        Returns:
            The next invoice number string.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT invoice_number FROM invoices WHERE invoice_number LIKE ? "
            "ORDER BY id DESC LIMIT 1",
            (f"{prefix}-%",),
        )
        row = cur.fetchone()
        if row and row["invoice_number"]:
            try:
                last_num = int(row["invoice_number"].split("-")[1])
                return f"{prefix}-{last_num + 1:04d}"
            except (IndexError, ValueError):
                pass
        return f"{prefix}-0001"

    def add_invoice(
        self,
        customer_id: str = "",
        customer_name: str = "",
        date: str = "",
        items: list[dict] | None = None,
        labor_hours: float = 0.0,
        labor_rate: float = 75.0,
        status: str = InvoiceStatus.DRAFT.value,
        notes: str = "",
        appointment_id: str = "",
        metadata: dict | None = None,
        prefix: str = "DC",
    ) -> Invoice:
        """Create a new invoice record.

        Auto-generates the invoice number (DC-0001 format) and calculates
        subtotal/total from items and labor.

        Args:
            customer_id:    Link to customer.
            customer_name:  Denormalized name.
            date:           Invoice date (YYYY-MM-DD).
            items:          List of line item dicts with description, qty, unit_price.
            labor_hours:    Labor hours billed.
            labor_rate:     Hourly rate in USD.
            status:         Initial status.
            notes:          Free-text notes.
            appointment_id: Optional link to appointment.
            metadata:       Optional extra data.
            prefix:         Invoice number prefix.

        Returns:
            The created Invoice.
        """
        now = datetime.now(timezone.utc).isoformat()
        invoice_id = str(uuid.uuid4())
        invoice_number = self._next_invoice_number(prefix)
        line_items = items or []
        meta = metadata or {}

        if not date:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Calculate totals
        parts_total = sum(
            item.get("qty", 0) * item.get("unit_price", 0.0)
            for item in line_items
        )
        labor_total = labor_hours * labor_rate
        subtotal = parts_total + labor_total
        tax_rate = 0.0  # Oregon — no sales tax
        total = subtotal * (1 + tax_rate)

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO invoices
                (invoice_id, invoice_number, customer_id, customer_name, date,
                 items, labor_hours, labor_rate, subtotal, tax_rate, total,
                 status, notes, appointment_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (invoice_id, invoice_number, customer_id, customer_name, date,
             json.dumps(line_items), labor_hours, labor_rate, subtotal,
             tax_rate, total, status, notes, appointment_id,
             now, now, json.dumps(meta)),
        )
        self._conn.commit()

        inv = Invoice(
            invoice_id=invoice_id, invoice_number=invoice_number,
            customer_id=customer_id, customer_name=customer_name, date=date,
            items=line_items, labor_hours=labor_hours, labor_rate=labor_rate,
            subtotal=subtotal, tax_rate=tax_rate, total=total,
            status=status, notes=notes, appointment_id=appointment_id,
            created_at=now, updated_at=now, metadata=meta,
        )
        logger.info("Invoice added: %s (%s) — $%.2f", invoice_number, inv.short_id, total)
        return inv

    def update_invoice(self, invoice_id: str, **kwargs) -> Invoice | None:
        """Update an invoice record.

        If items, labor_hours, or labor_rate change, subtotal/total are recalculated.

        Args:
            invoice_id: The invoice's UUID.
            **kwargs:   Fields to update.

        Returns:
            The updated Invoice, or None if not found.
        """
        existing = self.get_invoice(invoice_id)
        if existing is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        allowed = {"customer_id", "customer_name", "date", "items",
                    "labor_hours", "labor_rate", "status", "notes",
                    "appointment_id", "metadata"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}

        if not updates:
            return existing

        # Recalculate totals if financial fields changed
        line_items = updates.get("items", existing.items)
        labor_hours = updates.get("labor_hours", existing.labor_hours)
        labor_rate = updates.get("labor_rate", existing.labor_rate)

        if "items" in updates or "labor_hours" in updates or "labor_rate" in updates:
            parts_total = sum(
                item.get("qty", 0) * item.get("unit_price", 0.0)
                for item in line_items
            )
            labor_total = labor_hours * labor_rate
            subtotal = parts_total + labor_total
            total = subtotal  # Oregon: no tax
            updates["subtotal"] = subtotal
            updates["total"] = total

        set_parts = ["updated_at = ?"]
        params: list[Any] = [now]

        for key, val in updates.items():
            if key == "items":
                set_parts.append(f"{key} = ?")
                params.append(json.dumps(val))
            elif key == "metadata":
                set_parts.append(f"{key} = ?")
                params.append(json.dumps(val))
            else:
                set_parts.append(f"{key} = ?")
                params.append(val)

        params.append(invoice_id)
        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE invoices SET {', '.join(set_parts)} WHERE invoice_id = ?",
            params,
        )
        self._conn.commit()

        logger.info("Invoice updated: %s", invoice_id[:8])
        return self.get_invoice(invoice_id)

    def get_invoice(self, invoice_id: str) -> Invoice | None:
        """Retrieve an invoice by ID.

        Args:
            invoice_id: The invoice's UUID.

        Returns:
            The Invoice, or None if not found.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM invoices WHERE invoice_id = ?", (invoice_id,))
        row = cur.fetchone()
        return Invoice.from_row(row) if row else None

    def remove_invoice(self, invoice_id: str) -> bool:
        """Delete an invoice record.

        Args:
            invoice_id: The invoice's UUID.

        Returns:
            True if deleted, False if not found.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM invoices WHERE invoice_id = ?", (invoice_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Invoice removed: %s", invoice_id[:8])
        return deleted

    def list_invoices(self, limit: int = 100) -> list[Invoice]:
        """List all invoices, ordered by date descending (newest first).

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of Invoice objects.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM invoices ORDER BY date DESC LIMIT ?", (limit,))
        return [Invoice.from_row(row) for row in cur.fetchall()]

    def get_customer_invoices(self, customer_id: str) -> list[Invoice]:
        """Get all invoices for a specific customer.

        Args:
            customer_id: The customer's UUID.

        Returns:
            List of Invoice objects, newest first.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM invoices WHERE customer_id = ? ORDER BY date DESC",
            (customer_id,),
        )
        return [Invoice.from_row(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------
    # Inventory CRUD
    # -------------------------------------------------------------------

    def add_inventory_item(
        self,
        name: str,
        category: str = "",
        quantity: int = 0,
        cost: float = 0.0,
        location: str = "",
        reorder_threshold: int = 0,
        supplier: str = "",
        part_number: str = "",
        notes: str = "",
        metadata: dict | None = None,
    ) -> InventoryItem:
        """Create a new inventory item.

        Args:
            name:              Item name.
            category:          Category (parts, tools, etc.).
            quantity:          Initial quantity.
            cost:              Unit cost in USD.
            location:          Storage location.
            reorder_threshold: Reorder point.
            supplier:          Supplier name.
            part_number:       Part/model number.
            notes:             Free-text notes.
            metadata:          Optional extra data.

        Returns:
            The created InventoryItem.
        """
        now = datetime.now(timezone.utc).isoformat()
        item_id = str(uuid.uuid4())
        meta = metadata or {}

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO inventory
                (item_id, name, category, quantity, cost, location,
                 reorder_threshold, supplier, part_number, notes,
                 created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (item_id, name, category, quantity, cost, location,
             reorder_threshold, supplier, part_number, notes,
             now, now, json.dumps(meta)),
        )
        self._conn.commit()

        item = InventoryItem(
            item_id=item_id, name=name, category=category, quantity=quantity,
            cost=cost, location=location, reorder_threshold=reorder_threshold,
            supplier=supplier, part_number=part_number, notes=notes,
            created_at=now, updated_at=now, metadata=meta,
        )
        logger.info("Inventory item added: %s (%s)", name, item.short_id)
        return item

    def update_inventory_item(self, item_id: str, **kwargs) -> InventoryItem | None:
        """Update an inventory item.

        Args:
            item_id:  The item's UUID.
            **kwargs: Fields to update.

        Returns:
            The updated InventoryItem, or None if not found.
        """
        existing = self.get_inventory_item(item_id)
        if existing is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        allowed = {"name", "category", "quantity", "cost", "location",
                    "reorder_threshold", "supplier", "part_number", "notes", "metadata"}
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

        params.append(item_id)
        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE inventory SET {', '.join(set_parts)} WHERE item_id = ?",
            params,
        )
        self._conn.commit()

        logger.info("Inventory item updated: %s", item_id[:8])
        return self.get_inventory_item(item_id)

    def get_inventory_item(self, item_id: str) -> InventoryItem | None:
        """Retrieve an inventory item by ID.

        Args:
            item_id: The item's UUID.

        Returns:
            The InventoryItem, or None if not found.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM inventory WHERE item_id = ?", (item_id,))
        row = cur.fetchone()
        return InventoryItem.from_row(row) if row else None

    def remove_inventory_item(self, item_id: str) -> bool:
        """Delete an inventory item.

        Args:
            item_id: The item's UUID.

        Returns:
            True if deleted, False if not found.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM inventory WHERE item_id = ?", (item_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Inventory item removed: %s", item_id[:8])
        return deleted

    def list_inventory(self, limit: int = 200) -> list[InventoryItem]:
        """List all inventory items, ordered by name.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of InventoryItem objects.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM inventory ORDER BY name ASC LIMIT ?", (limit,))
        return [InventoryItem.from_row(row) for row in cur.fetchall()]

    def get_low_stock_items(self) -> list[InventoryItem]:
        """Get all inventory items that need reordering.

        Returns:
            List of InventoryItem objects where quantity <= reorder_threshold.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM inventory WHERE reorder_threshold > 0 "
            "AND quantity <= reorder_threshold ORDER BY name ASC",
        )
        return [InventoryItem.from_row(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------
    # Dashboard helpers
    # -------------------------------------------------------------------

    def to_broadcast_list(self) -> dict[str, list[dict]]:
        """Return all business data for dashboard broadcast.

        Returns:
            Dict with customers, appointments, invoices, and inventory lists.
        """
        return {
            "customers": [c.to_dict() for c in self.list_customers()],
            "appointments": [a.to_dict() for a in self.list_appointments()],
            "invoices": [i.to_dict() for i in self.list_invoices()],
            "inventory": [item.to_dict() for item in self.list_inventory()],
        }

    def get_status(self) -> dict[str, Any]:
        """Return summary status for the dashboard.

        Returns:
            Dict with counts and key metrics.
        """
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) FROM customers")
        customer_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM appointments WHERE status IN ('scheduled', 'confirmed')")
        upcoming_appts = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM appointments")
        total_appts = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM invoices WHERE status = 'draft'")
        draft_invoices = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM invoices WHERE status = 'paid'")
        paid_invoices = cur.fetchone()[0]

        cur.execute("SELECT COALESCE(SUM(total), 0) FROM invoices WHERE status = 'paid'")
        total_revenue = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM invoices")
        total_invoices = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM inventory")
        inventory_count = cur.fetchone()[0]

        low_stock = len(self.get_low_stock_items())

        return {
            "customer_count": customer_count,
            "upcoming_appointments": upcoming_appts,
            "total_appointments": total_appts,
            "draft_invoices": draft_invoices,
            "paid_invoices": paid_invoices,
            "total_invoices": total_invoices,
            "total_revenue": round(total_revenue, 2),
            "inventory_count": inventory_count,
            "low_stock_count": low_stock,
        }

    def close(self):
        """Close the database connection."""
        self._conn.close()
        logger.info("BusinessStore closed")

    # -------------------------------------------------------------------
    # Sample data seeding
    # -------------------------------------------------------------------

    def seed_sample_data(self):
        """Insert sample records if tables are empty.

        Only inserts when all four tables have zero rows, so it's safe
        to call on every startup.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM customers")
        if cur.fetchone()[0] > 0:
            logger.info("BusinessStore already has data — skipping seed")
            return

        logger.info("Seeding sample business data")

        # --- Customers ---
        mike = self.add_customer(
            name="Mike Rodriguez",
            phone="503-555-0142",
            email="mike.r@email.com",
            bike_info="2019 Harley-Davidson Street Glide",
            notes="Regular customer. Referred by Dave.",
        )
        sarah = self.add_customer(
            name="Sarah Chen",
            phone="503-555-0198",
            email="sarah.chen@email.com",
            bike_info="2021 Yamaha MT-07",
            notes="New rider, very careful about maintenance schedules.",
        )
        dave = self.add_customer(
            name="Dave Thompson",
            phone="503-555-0167",
            email="",
            bike_info="1982 Honda CB750",
            notes="Vintage bike enthusiast. Prefers text over email.",
        )

        # --- Appointments ---
        self.add_appointment(
            customer_id=mike.customer_id,
            customer_name=mike.name,
            date="2026-02-15",
            time="10:00",
            bike="2019 Street Glide",
            service_type="Diagnostic",
            status=AppointmentStatus.SCHEDULED.value,
            notes="Check engine light on. Customer reports intermittent rough idle.",
            location="Customer's garage — SE Portland",
            estimated_cost=150.0,
        )
        self.add_appointment(
            customer_id=sarah.customer_id,
            customer_name=sarah.name,
            date="2026-02-10",
            time="14:00",
            bike="2021 MT-07",
            service_type="Maintenance",
            status=AppointmentStatus.COMPLETED.value,
            notes="5000-mile service: oil change, chain adjustment, brake check.",
            location="Customer's apartment — NW Portland",
            estimated_cost=175.0,
        )

        # --- Invoice (for Sarah's completed maintenance) ---
        self.add_invoice(
            customer_id=sarah.customer_id,
            customer_name=sarah.name,
            date="2026-02-10",
            items=[
                {"description": "10W-40 Synthetic Oil (3 qt)", "qty": 1, "unit_price": 28.50},
                {"description": "Oil Filter — Yamaha OEM", "qty": 1, "unit_price": 12.00},
            ],
            labor_hours=1.5,
            labor_rate=75.0,
            status=InvoiceStatus.PAID.value,
            notes="5000-mile service completed. Everything looks good.",
        )

        # --- Inventory ---
        self.add_inventory_item(
            name="OBD-M Diagnostic Scanner",
            category="tools",
            quantity=1,
            cost=349.99,
            location="Mobile kit",
            reorder_threshold=0,
            notes="Primary diagnostic tool for late-model bikes.",
        )
        self.add_inventory_item(
            name="10W-40 Synthetic Oil",
            category="consumables",
            quantity=12,
            cost=9.50,
            location="Garage shelf A",
            reorder_threshold=4,
            supplier="Parts Unlimited",
            part_number="PU-1040-QT",
        )
        self.add_inventory_item(
            name="Oil Filters (assorted)",
            category="parts",
            quantity=3,
            cost=11.00,
            location="Garage shelf A",
            reorder_threshold=2,
            supplier="Parts Unlimited",
        )
        self.add_inventory_item(
            name="Spark Plugs (NGK)",
            category="parts",
            quantity=8,
            cost=4.50,
            location="Garage shelf B",
            reorder_threshold=5,
            supplier="NGK Direct",
            part_number="NGK-CR8E",
        )
        self.add_inventory_item(
            name="Multimeter",
            category="tools",
            quantity=1,
            cost=89.99,
            location="Mobile kit",
            reorder_threshold=0,
            notes="Fluke 117 — essential for electrical diagnostics.",
        )

        logger.info("Sample business data seeded successfully")


# ---------------------------------------------------------------------------
# Business detection keywords — aligned with task_classifier.py
# ---------------------------------------------------------------------------

BUSINESS_KEYWORDS = [
    "customer", "client", "contact", "customer list", "add customer",
    "appointment", "schedule", "book", "reschedule", "cancel appointment",
    "invoice", "bill", "charge", "payment", "receipt", "billing",
    "inventory", "stock", "parts", "tools", "reorder", "low stock",
    "business", "revenue", "pricing", "service history",
    "crm", "customer profile", "last service",
    "who has a", "customer note", "customer tag",
]

# Business-specific system prompt for model calls
BUSINESS_SYSTEM_PROMPT = """
You are also serving as the Business Agent for Doppler Cycles, a mobile
motorcycle diagnostics and repair business in Portland, Oregon.

Key business context:
- Labor rate: $75/hour
- Oregon has NO sales tax — do not add tax to invoices
- Mobile service: George goes to the customer's location
- Shop counter test: would George say this to a customer standing in
  front of him? If not, rewrite it.
- Customer privacy: never expose customer contact details in public content

When handling business tasks:
- Be direct and actionable — George needs info he can act on
- For scheduling, consider Portland metro travel time between appointments
- For invoicing, itemize parts and labor separately
- For inventory, flag items that need reordering
- Always maintain professional but approachable tone
"""


# ---------------------------------------------------------------------------
# BusinessAgent
# ---------------------------------------------------------------------------

class BusinessAgent:
    """Local business agent — intercepts business tasks in dispatch_to_agent().

    Handles business-related tasks (customers, appointments, invoices,
    inventory) with business-specific prompting. Stores results in
    long-term memory and uses the BusinessStore for data context.

    Args:
        router:           ModelRouter instance for business model calls.
        persona:          Persona instance for building system prompts.
        long_term_memory: LongTermMemory for context retrieval and result storage.
        business_store:   BusinessStore for customer/appointment/invoice/inventory data.
        event_logger:     EventLogger for audit trail.
        on_model_call:    Optional callback for model cost tracking.
    """

    def __init__(
        self,
        router,
        persona,
        long_term_memory,
        business_store: BusinessStore,
        event_logger,
        on_model_call: Callable | None = None,
        crm_store=None,
    ):
        self.router = router
        self.persona = persona
        self.long_term = long_term_memory
        self.store = business_store
        self.event_logger = event_logger
        self._on_model_call = on_model_call
        self.crm_store = crm_store

        logger.info("BusinessAgent initialized")

    # -------------------------------------------------------------------
    # Detection
    # -------------------------------------------------------------------

    def is_business_task(self, description: str) -> bool:
        """Check if a task description is business-related.

        Uses lowercase keyword matching against BUSINESS_KEYWORDS.

        Args:
            description: The task description text.

        Returns:
            True if the task appears to be business-related.
        """
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in BUSINESS_KEYWORDS)

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    async def try_handle(self, task, plan: dict) -> str | None:
        """Try to handle a task as business work.

        Called from dispatch_to_agent() after the research agent.
        Returns the business result string if handled, or None to let
        the task fall through to remote agents / model fallback.

        Args:
            task: The Task object being dispatched.
            plan: The plan dict from the THINK phase.

        Returns:
            Business result string, or None if not a business task.
        """
        if not self.is_business_task(task.description):
            return None

        logger.info(
            "Business agent handling task %s: %s",
            task.short_id, task.description[:80],
        )

        try:
            # Build context from the business store
            biz_context = self._build_context(task.description)

            # Get relevant long-term memory
            ltm_context = self._get_ltm_context(task.description)

            # Make a business-specific model call
            result = await self._business_call(task, biz_context, ltm_context)

            # Store result in long-term memory
            if result and len(result) > 50:
                try:
                    ltm_content = f"Business task: {task.description}\nResult: {result[:500]}"
                    self.long_term.store(
                        content=ltm_content,
                        category="knowledge",
                        metadata={
                            "task_id": task.task_id,
                            "source": "business_agent",
                        },
                    )
                except Exception:
                    logger.debug("Business agent LTM store failed (non-fatal)", exc_info=True)

            return result

        except Exception as e:
            logger.warning(
                "Business agent failed for task %s: %s",
                task.short_id, e,
            )
            # Return None to let the task fall through to other agents
            return None

    # -------------------------------------------------------------------
    # Context building
    # -------------------------------------------------------------------

    def _build_context(self, description: str) -> str:
        """Build business context from the store based on task keywords.

        Pulls relevant data: customer list if "customer" mentioned,
        upcoming appointments if "appointment" mentioned, low stock
        if "inventory" mentioned, etc.

        Args:
            description: Task description text.

        Returns:
            Formatted context string with relevant business data.
        """
        desc_lower = description.lower()
        parts: list[str] = []

        # Customer context
        if any(kw in desc_lower for kw in ["customer", "client", "contact"]):
            customers = self.store.list_customers(limit=20)
            if customers:
                lines = [f"  - {c.name} ({c.bike_info})" for c in customers]
                parts.append(f"Current customers ({len(customers)}):\n" + "\n".join(lines))

        # Appointment context
        if any(kw in desc_lower for kw in ["appointment", "schedule", "book"]):
            appts = self.store.list_appointments(limit=20)
            upcoming = [a for a in appts if a.status in ("scheduled", "confirmed")]
            if upcoming:
                lines = [
                    f"  - {a.date} {a.time}: {a.customer_name} — {a.service_type} ({a.status})"
                    for a in upcoming
                ]
                parts.append(f"Upcoming appointments ({len(upcoming)}):\n" + "\n".join(lines))

        # Invoice context
        if any(kw in desc_lower for kw in ["invoice", "bill", "payment", "revenue"]):
            invoices = self.store.list_invoices(limit=20)
            if invoices:
                lines = [
                    f"  - {i.invoice_number}: {i.customer_name} — ${i.total:.2f} ({i.status})"
                    for i in invoices
                ]
                parts.append(f"Recent invoices ({len(invoices)}):\n" + "\n".join(lines))

        # Inventory context
        if any(kw in desc_lower for kw in ["inventory", "stock", "parts", "tools", "reorder"]):
            low_stock = self.store.get_low_stock_items()
            if low_stock:
                lines = [
                    f"  - {item.name}: {item.quantity} on hand (reorder at {item.reorder_threshold})"
                    for item in low_stock
                ]
                parts.append(f"Low stock items ({len(low_stock)}):\n" + "\n".join(lines))

            all_items = self.store.list_inventory(limit=50)
            if all_items:
                lines = [f"  - {item.name}: {item.quantity} ({item.category})" for item in all_items]
                parts.append(f"Full inventory ({len(all_items)} items):\n" + "\n".join(lines))

        # CRM context — richer customer profiles, vehicles, notes
        crm_keywords = ["crm", "customer profile", "service history", "last service",
                         "who has a", "customer note", "customer tag"]
        if self.crm_store and any(kw in desc_lower for kw in crm_keywords):
            try:
                crm_customers = self.crm_store.list_customers(limit=30)
                if crm_customers:
                    lines = []
                    for cc in crm_customers:
                        # Get vehicles from service history
                        hist = self.crm_store.get_service_history(cc.customer_id)
                        vehicles = [
                            f"{v.get('year', '')} {v.get('make', '')} {v.get('model', '')}".strip()
                            for v in hist.get("vehicles", [])
                        ]
                        veh_str = ", ".join(vehicles) if vehicles else "no vehicles"
                        tags_str = ", ".join(cc.tags) if cc.tags else ""
                        last = cc.last_contact[:10] if cc.last_contact else "unknown"
                        notes = self.crm_store.get_notes(cc.customer_id, limit=3)
                        note_str = "; ".join(n.content for n in notes) if notes else ""
                        line = f"  - {cc.name} | Vehicles: {veh_str} | Last contact: {last}"
                        if tags_str:
                            line += f" | Tags: {tags_str}"
                        if note_str:
                            line += f" | Notes: {note_str}"
                        lines.append(line)
                    parts.append(f"CRM customer profiles ({len(crm_customers)}):\n" + "\n".join(lines))
            except Exception:
                pass  # CRM context is supplemental; don't block on errors

        # General business status
        status = self.store.get_status()
        parts.append(
            f"Business summary: {status['customer_count']} customers, "
            f"{status['upcoming_appointments']} upcoming appointments, "
            f"{status['draft_invoices']} draft invoices, "
            f"${status['total_revenue']:.2f} total revenue, "
            f"{status['low_stock_count']} items low on stock"
        )

        return "\n\n".join(parts)

    # -------------------------------------------------------------------
    # Model call
    # -------------------------------------------------------------------

    async def _business_call(self, task, biz_context: str, ltm_context: str) -> str:
        """Make a business-specific model call.

        Uses tier2 complexity — customer-facing work per task_classifier Rule 2.

        Args:
            task:        The task being handled.
            biz_context: Business data context from the store.
            ltm_context: Relevant long-term memory context.

        Returns:
            The model's response text.
        """
        system_prompt = self.persona.build_system_prompt() + BUSINESS_SYSTEM_PROMPT

        # Build user prompt
        parts = [f"Business task: {task.description}"]

        if biz_context:
            parts.append(f"\n\nCurrent business data:\n{biz_context}")

        if ltm_context:
            parts.append(f"\n\n{ltm_context}")

        prompt = "".join(parts)

        logger.info(
            "Business agent making model call for %s (prompt ~%d chars)",
            task.short_id, len(prompt),
        )

        response = await self.router.route(
            prompt=prompt,
            task_complexity="tier2",
            system_prompt=system_prompt,
        )

        # Notify model call listener for cost tracking
        if self._on_model_call is not None:
            try:
                self._on_model_call(
                    model=response.model,
                    backend=response.backend,
                    tokens=response.total_tokens,
                    latency_ms=response.latency_ms,
                    cost_usd=response.cost_usd,
                    task_short_id=task.short_id,
                )
            except Exception:
                logger.debug("Business agent model call callback error", exc_info=True)

        logger.info(
            "Business agent call complete for %s: model=%s, tokens=%d",
            task.short_id, response.model, response.total_tokens,
        )

        return response.text

    # -------------------------------------------------------------------
    # LTM context
    # -------------------------------------------------------------------

    def _get_ltm_context(self, description: str) -> str:
        """Query long-term memory for relevant business knowledge.

        Args:
            description: Task description for the LTM query.

        Returns:
            Formatted context string, or empty string if nothing relevant.
        """
        try:
            ltm_results = self.long_term.query(description, top_k=3)
            relevant = [r for r in ltm_results if r.score > 0.3]
            if relevant:
                ltm_lines = [f"- [{r.category}] {r.content}" for r in relevant]
                return (
                    "\nRelevant past business knowledge from memory:\n"
                    + "\n".join(ltm_lines)
                )
        except Exception:
            logger.debug("Business agent LTM query failed (non-fatal)", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Self-test — run with: python3 agents/business_agent.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("BusinessStore self-test")
    print("=" * 60)

    # Create a temp DB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    try:
        store = BusinessStore(db_path=tmp_db)

        # --- Customer CRUD ---
        print("\n--- Customer CRUD ---")
        c = store.add_customer(name="Test Customer", phone="555-1234", bike_info="2020 Honda CB500F")
        print(f"  Added: {c.name} ({c.short_id})")

        c2 = store.get_customer(c.customer_id)
        assert c2 is not None
        assert c2.name == "Test Customer"
        print(f"  Retrieved: {c2.name}")

        c3 = store.update_customer(c.customer_id, phone="555-5678", notes="Updated")
        assert c3 is not None
        assert c3.phone == "555-5678"
        print(f"  Updated phone: {c3.phone}")

        customers = store.list_customers()
        assert len(customers) == 1
        print(f"  Listed: {len(customers)} customer(s)")

        # --- Appointment CRUD ---
        print("\n--- Appointment CRUD ---")
        a = store.add_appointment(
            customer_id=c.customer_id, customer_name=c.name,
            date="2026-03-01", time="10:00", bike="CB500F",
            service_type="Diagnostic",
        )
        print(f"  Added: {a.short_id} on {a.date}")

        a2 = store.update_appointment(a.appointment_id, status="confirmed")
        assert a2 is not None
        assert a2.status == "confirmed"
        print(f"  Updated status: {a2.status}")

        appts = store.get_customer_appointments(c.customer_id)
        assert len(appts) == 1
        print(f"  Customer appointments: {len(appts)}")

        # --- Invoice CRUD ---
        print("\n--- Invoice CRUD ---")
        inv = store.add_invoice(
            customer_id=c.customer_id, customer_name=c.name,
            items=[{"description": "Oil Change", "qty": 1, "unit_price": 28.50}],
            labor_hours=1.0, labor_rate=75.0,
        )
        print(f"  Added: {inv.invoice_number} — ${inv.total:.2f}")
        assert inv.invoice_number == "DC-0001"
        assert inv.total == 103.50  # 28.50 parts + 75.00 labor

        inv2 = store.add_invoice(customer_name="Another")
        assert inv2.invoice_number == "DC-0002"
        print(f"  Auto-increment: {inv2.invoice_number}")

        inv3 = store.update_invoice(inv.invoice_id, status="paid")
        assert inv3 is not None
        assert inv3.status == "paid"
        print(f"  Updated status: {inv3.status}")

        # --- Inventory CRUD ---
        print("\n--- Inventory CRUD ---")
        item = store.add_inventory_item(
            name="Test Oil", category="consumables",
            quantity=3, reorder_threshold=5,
        )
        print(f"  Added: {item.name} (qty={item.quantity})")
        assert item.needs_reorder is True
        print(f"  Needs reorder: {item.needs_reorder}")

        low = store.get_low_stock_items()
        assert len(low) == 1
        print(f"  Low stock items: {len(low)}")

        item2 = store.update_inventory_item(item.item_id, quantity=10)
        assert item2 is not None
        assert item2.quantity == 10
        assert item2.needs_reorder is False
        print(f"  Updated qty: {item2.quantity}, needs reorder: {item2.needs_reorder}")

        # --- Delete tests ---
        print("\n--- Delete tests ---")
        assert store.remove_inventory_item(item.item_id) is True
        assert store.remove_invoice(inv.invoice_id) is True
        assert store.remove_invoice(inv2.invoice_id) is True
        assert store.remove_appointment(a.appointment_id) is True
        assert store.remove_customer(c.customer_id) is True
        print("  All deletes successful")

        # --- Sample data seeding ---
        print("\n--- Sample data seeding ---")
        store.seed_sample_data()
        status = store.get_status()
        print(f"  Customers: {status['customer_count']}")
        print(f"  Appointments: {status['total_appointments']}")
        print(f"  Invoices: {status['total_invoices']}")
        print(f"  Inventory: {status['inventory_count']}")
        print(f"  Low stock: {status['low_stock_count']}")
        assert status["customer_count"] == 3
        assert status["total_appointments"] == 2
        assert status["total_invoices"] == 1
        assert status["inventory_count"] == 5

        # --- Broadcast format ---
        print("\n--- Broadcast format ---")
        broadcast = store.to_broadcast_list()
        assert "customers" in broadcast
        assert "appointments" in broadcast
        assert "invoices" in broadcast
        assert "inventory" in broadcast
        print(f"  Keys: {list(broadcast.keys())}")

        # Second seed should be a no-op
        store.seed_sample_data()
        assert store.get_status()["customer_count"] == 3
        print("  Second seed: no-op (correct)")

        store.close()
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    finally:
        os.unlink(tmp_db)
