"""CRM (Customer Relationship Manager) for Doppler Cycles.

Provides a richer customer layer on top of the basic BusinessStore customers,
with tags, timestamped notes, preferred contact methods, address fields,
and service-history integration via ServiceRecordStore.

Storage: ``data/crm.db`` (separate from business.db for clean separation).
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CRMCustomer:
    """An enriched customer record for Doppler Cycles CRM."""

    customer_id: str
    name: str
    phone: str = ""
    email: str = ""
    address: str = ""
    preferred_contact_method: str = "phone"  # phone / email / text
    tags: list[str] = field(default_factory=list)
    notes_summary: str = ""
    date_added: str = ""
    last_contact: str = ""
    business_customer_id: str = ""  # links to BusinessStore customer_id
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.customer_id[:8]

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "phone": self.phone,
            "email": self.email,
            "address": self.address,
            "preferred_contact_method": self.preferred_contact_method,
            "tags": self.tags,
            "notes_summary": self.notes_summary,
            "date_added": self.date_added,
            "last_contact": self.last_contact,
            "business_customer_id": self.business_customer_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "short_id": self.short_id,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "CRMCustomer":
        try:
            tags = json.loads(row["tags"]) if row["tags"] else []
        except (json.JSONDecodeError, TypeError):
            tags = []
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return cls(
            customer_id=row["customer_id"],
            name=row["name"],
            phone=row["phone"] or "",
            email=row["email"] or "",
            address=row["address"] or "",
            preferred_contact_method=row["preferred_contact_method"] or "phone",
            tags=tags,
            notes_summary=row["notes_summary"] or "",
            date_added=row["date_added"] or "",
            last_contact=row["last_contact"] or "",
            business_customer_id=row["business_customer_id"] or "",
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
            metadata=metadata,
        )


@dataclass
class CustomerNote:
    """A timestamped note attached to a CRM customer."""

    note_id: str
    customer_id: str
    content: str = ""
    category: str = "general"  # general / service / followup / reminder
    created_at: str = ""

    @property
    def short_id(self) -> str:
        return self.note_id[:8]

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "customer_id": self.customer_id,
            "content": self.content,
            "category": self.category,
            "created_at": self.created_at,
            "short_id": self.short_id,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "CustomerNote":
        return cls(
            note_id=row["note_id"],
            customer_id=row["customer_id"],
            content=row["content"] or "",
            category=row["category"] or "general",
            created_at=row["created_at"] or "",
        )


# ---------------------------------------------------------------------------
# CRMStore
# ---------------------------------------------------------------------------

class CRMStore:
    """SQLite-backed CRM for Doppler Cycles.

    Args:
        db_path:       Path to the SQLite database file.
        service_store: Optional ServiceRecordStore for vehicle/history lookups.
        on_change:     Async callback fired after any mutation.
    """

    def __init__(
        self,
        db_path: str = "data/crm.db",
        service_store: Any | None = None,
        on_change: Callable[[], Coroutine] | None = None,
    ):
        self._db_path = db_path
        self._service_store = service_store
        self._on_change = on_change

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        logger.info("CRMStore initialized — db=%s", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS crm_customers (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id             TEXT UNIQUE NOT NULL,
                name                    TEXT NOT NULL,
                phone                   TEXT DEFAULT '',
                email                   TEXT DEFAULT '',
                address                 TEXT DEFAULT '',
                preferred_contact_method TEXT DEFAULT 'phone',
                tags                    TEXT DEFAULT '[]',
                notes_summary           TEXT DEFAULT '',
                date_added              TEXT,
                last_contact            TEXT,
                business_customer_id    TEXT DEFAULT '',
                created_at              TEXT NOT NULL,
                updated_at              TEXT NOT NULL,
                metadata                TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_crm_name
                ON crm_customers(name);
            CREATE INDEX IF NOT EXISTS idx_crm_phone
                ON crm_customers(phone);
            CREATE INDEX IF NOT EXISTS idx_crm_email
                ON crm_customers(email);
            CREATE INDEX IF NOT EXISTS idx_crm_biz_id
                ON crm_customers(business_customer_id);

            CREATE TABLE IF NOT EXISTS customer_notes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id     TEXT UNIQUE NOT NULL,
                customer_id TEXT NOT NULL,
                content     TEXT DEFAULT '',
                category    TEXT DEFAULT 'general',
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_note_customer
                ON customer_notes(customer_id);
            CREATE INDEX IF NOT EXISTS idx_note_created
                ON customer_notes(created_at);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _notify_change(self) -> None:
        """Fire the on_change callback if set."""
        if self._on_change is not None:
            try:
                await self._on_change()
            except Exception:
                logger.debug("CRMStore on_change callback error", exc_info=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Customer CRUD
    # ------------------------------------------------------------------

    def add_customer(
        self,
        name: str,
        phone: str = "",
        email: str = "",
        address: str = "",
        preferred_contact_method: str = "phone",
        tags: list[str] | None = None,
        notes_summary: str = "",
        metadata: dict | None = None,
    ) -> CRMCustomer:
        """Create a new CRM customer record."""
        now = self._now()
        customer = CRMCustomer(
            customer_id=str(uuid.uuid4()),
            name=name,
            phone=phone,
            email=email,
            address=address,
            preferred_contact_method=preferred_contact_method,
            tags=tags or [],
            notes_summary=notes_summary,
            date_added=now,
            last_contact=now,
            business_customer_id="",
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO crm_customers
               (customer_id, name, phone, email, address,
                preferred_contact_method, tags, notes_summary,
                date_added, last_contact, business_customer_id,
                created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                customer.customer_id,
                customer.name,
                customer.phone,
                customer.email,
                customer.address,
                customer.preferred_contact_method,
                json.dumps(customer.tags),
                customer.notes_summary,
                customer.date_added,
                customer.last_contact,
                customer.business_customer_id,
                customer.created_at,
                customer.updated_at,
                json.dumps(customer.metadata),
            ),
        )
        self._conn.commit()
        logger.info("CRM customer added: %s (%s)", customer.name, customer.short_id)
        return customer

    def update_customer(self, customer_id: str, **kwargs: Any) -> CRMCustomer | None:
        """Update fields on an existing CRM customer.

        Allowed fields: name, phone, email, address,
        preferred_contact_method, tags, notes_summary,
        last_contact, business_customer_id, metadata.
        """
        allowed = {
            "name", "phone", "email", "address",
            "preferred_contact_method", "tags", "notes_summary",
            "last_contact", "business_customer_id", "metadata",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_customer(customer_id)

        updates["updated_at"] = self._now()

        # Serialize JSON fields
        if "tags" in updates:
            updates["tags"] = json.dumps(updates["tags"])
        if "metadata" in updates:
            updates["metadata"] = json.dumps(updates["metadata"])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [customer_id]

        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE crm_customers SET {set_clause} WHERE customer_id = ?",
            values,
        )
        self._conn.commit()

        customer = self.get_customer(customer_id)
        if customer:
            logger.info("CRM customer updated: %s (%s)", customer.name, customer.short_id)
        return customer

    def get_customer(self, customer_id: str) -> CRMCustomer | None:
        """Fetch a single CRM customer by ID."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM crm_customers WHERE customer_id = ?",
            (customer_id,),
        )
        row = cur.fetchone()
        return CRMCustomer.from_row(row) if row else None

    def remove_customer(self, customer_id: str) -> bool:
        """Delete a CRM customer and all their notes."""
        cur = self._conn.cursor()
        cur.execute(
            "DELETE FROM customer_notes WHERE customer_id = ?",
            (customer_id,),
        )
        cur.execute(
            "DELETE FROM crm_customers WHERE customer_id = ?",
            (customer_id,),
        )
        deleted = cur.rowcount > 0
        self._conn.commit()
        if deleted:
            logger.info("CRM customer removed: %s", customer_id[:8])
        return deleted

    def list_customers(self, limit: int = 100) -> list[CRMCustomer]:
        """List all CRM customers ordered by name."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM crm_customers ORDER BY name LIMIT ?",
            (limit,),
        )
        return [CRMCustomer.from_row(r) for r in cur.fetchall()]

    def search_customers(self, query: str) -> list[CRMCustomer]:
        """Search customers by name, phone, email, address, tags, or notes.

        Also searches vehicles via service_store (if available) and includes
        customers who own matching vehicles.
        """
        like = f"%{query}%"
        cur = self._conn.cursor()
        cur.execute(
            """SELECT * FROM crm_customers
               WHERE name LIKE ? OR phone LIKE ? OR email LIKE ?
                  OR address LIKE ? OR tags LIKE ? OR notes_summary LIKE ?
               ORDER BY name""",
            (like, like, like, like, like, like),
        )
        results = {r["customer_id"]: CRMCustomer.from_row(r) for r in cur.fetchall()}

        # Also search vehicles in service_store
        if self._service_store is not None:
            try:
                vehicles = self._service_store.search_vehicles(query)
                owner_ids = {v.owner_id for v in vehicles if v.owner_id}
                if owner_ids:
                    placeholders = ",".join("?" for _ in owner_ids)
                    cur.execute(
                        f"""SELECT * FROM crm_customers
                            WHERE business_customer_id IN ({placeholders})""",
                        list(owner_ids),
                    )
                    for r in cur.fetchall():
                        cid = r["customer_id"]
                        if cid not in results:
                            results[cid] = CRMCustomer.from_row(r)
            except Exception:
                logger.debug("CRM vehicle search failed", exc_info=True)

        return sorted(results.values(), key=lambda c: c.name)

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    def add_note(
        self,
        customer_id: str,
        content: str,
        category: str = "general",
    ) -> CustomerNote:
        """Add a timestamped note to a customer and update last_contact."""
        now = self._now()
        note = CustomerNote(
            note_id=str(uuid.uuid4()),
            customer_id=customer_id,
            content=content,
            category=category,
            created_at=now,
        )
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO customer_notes
               (note_id, customer_id, content, category, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (note.note_id, note.customer_id, note.content, note.category, note.created_at),
        )
        # Update last_contact on the customer
        cur.execute(
            "UPDATE crm_customers SET last_contact = ?, updated_at = ? WHERE customer_id = ?",
            (now, now, customer_id),
        )
        self._conn.commit()
        logger.info("CRM note added for %s: %s", customer_id[:8], note.short_id)
        return note

    def get_notes(self, customer_id: str, limit: int = 50) -> list[CustomerNote]:
        """Get notes for a customer, newest first."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT * FROM customer_notes
               WHERE customer_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (customer_id, limit),
        )
        return [CustomerNote.from_row(r) for r in cur.fetchall()]

    def remove_note(self, note_id: str) -> bool:
        """Delete a single customer note."""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM customer_notes WHERE note_id = ?", (note_id,))
        deleted = cur.rowcount > 0
        self._conn.commit()
        if deleted:
            logger.info("CRM note removed: %s", note_id[:8])
        return deleted

    # ------------------------------------------------------------------
    # Service history integration
    # ------------------------------------------------------------------

    def get_service_history(self, customer_id: str) -> dict[str, Any]:
        """Get vehicles and service records for a CRM customer.

        Uses the linked business_customer_id to query ServiceRecordStore.
        """
        customer = self.get_customer(customer_id)
        if not customer or not customer.business_customer_id:
            return {"vehicles": [], "records": []}
        if self._service_store is None:
            return {"vehicles": [], "records": []}

        biz_id = customer.business_customer_id
        try:
            vehicles = self._service_store.list_vehicles(owner_id=biz_id)
            records = self._service_store.get_customer_history(biz_id)
            return {
                "vehicles": [v.to_dict() for v in vehicles],
                "records": [r.to_dict() for r in records],
            }
        except Exception:
            logger.debug("CRM service history lookup failed", exc_info=True)
            return {"vehicles": [], "records": []}

    def get_customer_profile(self, customer_id: str) -> dict[str, Any]:
        """Full customer profile: info + notes + service history."""
        customer = self.get_customer(customer_id)
        if not customer:
            return {}
        notes = self.get_notes(customer_id)
        history = self.get_service_history(customer_id)
        return {
            "customer": customer.to_dict(),
            "notes": [n.to_dict() for n in notes],
            "vehicles": history["vehicles"],
            "records": history["records"][:10],  # last 10 service records
        }

    # ------------------------------------------------------------------
    # Import from BusinessStore
    # ------------------------------------------------------------------

    def import_from_business_store(self, business_store: Any) -> int:
        """Idempotent import of existing BusinessStore customers into CRM.

        Checks business_customer_id to skip duplicates. Returns the number
        of newly imported customers.
        """
        if business_store is None:
            return 0

        try:
            biz_customers = business_store.list_customers(limit=10000)
        except Exception:
            logger.warning("Failed to list BusinessStore customers for import", exc_info=True)
            return 0

        # Collect existing business_customer_id links to avoid dupes
        cur = self._conn.cursor()
        cur.execute("SELECT business_customer_id FROM crm_customers WHERE business_customer_id != ''")
        existing_biz_ids = {r["business_customer_id"] for r in cur.fetchall()}

        imported = 0
        now = self._now()
        for biz_cust in biz_customers:
            if biz_cust.customer_id in existing_biz_ids:
                continue

            customer_id = str(uuid.uuid4())
            cur.execute(
                """INSERT INTO crm_customers
                   (customer_id, name, phone, email, address,
                    preferred_contact_method, tags, notes_summary,
                    date_added, last_contact, business_customer_id,
                    created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    customer_id,
                    biz_cust.name,
                    biz_cust.phone or "",
                    biz_cust.email or "",
                    "",  # address — not in BusinessStore
                    "phone",  # default
                    "[]",  # tags
                    biz_cust.notes or "",
                    biz_cust.created_at or now,
                    biz_cust.updated_at or now,
                    biz_cust.customer_id,  # link back
                    now,
                    now,
                    json.dumps(biz_cust.metadata if hasattr(biz_cust, "metadata") else {}),
                ),
            )
            imported += 1

        self._conn.commit()
        if imported:
            logger.info("CRM imported %d customers from BusinessStore", imported)
        return imported

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Full state for dashboard broadcast."""
        return {
            "customers": [c.to_dict() for c in self.list_customers()],
            "status": self.get_status(),
        }

    def get_status(self) -> dict[str, Any]:
        """Summary statistics for dashboard display."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM crm_customers")
        total = cur.fetchone()[0]

        # Customers added this month
        month_start = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0,
        ).isoformat()
        cur.execute(
            "SELECT COUNT(*) FROM crm_customers WHERE created_at >= ?",
            (month_start,),
        )
        this_month = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM customer_notes")
        total_notes = cur.fetchone()[0]

        return {
            "total_customers": total,
            "customers_this_month": this_month,
            "total_notes": total_notes,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info("CRMStore closed")
