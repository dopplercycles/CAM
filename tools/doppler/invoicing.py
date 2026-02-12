"""
Invoice & Billing System for Doppler Cycles.

Provides professional invoicing for mobile diagnostic services with:
- Sequential invoice numbering (DC-2026-0001)
- Line items, labor, parts, tax tracking
- PDF generation (fpdf2, matching service record branding)
- Payment status lifecycle (draft → sent → paid / overdue / cancelled)
- Service record integration (one-click invoice from completed work)
- Monthly revenue reporting for dashboard charts

SQLite-backed, single-file module -- same pattern as service_records.py.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from fpdf import FPDF

logger = logging.getLogger(__name__)

# Valid status transitions
VALID_STATUSES = {"draft", "sent", "paid", "overdue", "cancelled"}
PAYMENT_METHODS = {"cash", "check", "card", "zelle", "venmo", ""}


# ---------------------------------------------------------------------------
# InvoiceRecord dataclass
# ---------------------------------------------------------------------------

@dataclass
class InvoiceRecord:
    """A single invoice."""
    invoice_id: str = ""
    invoice_number: str = ""
    customer_id: str = ""
    customer_name: str = ""
    customer_email: str = ""
    customer_phone: str = ""
    service_record_id: str = ""
    date: str = ""                      # YYYY-MM-DD
    due_date: str = ""                  # YYYY-MM-DD
    line_items: list = field(default_factory=list)  # [{description, qty, unit_price, total}]
    labor_hours: float = 0.0
    labor_rate: float = 75.0
    parts_total: float = 0.0
    labor_total: float = 0.0
    subtotal: float = 0.0
    tax_rate: float = 0.0              # Oregon = 0%
    tax_amount: float = 0.0
    total: float = 0.0
    status: str = "draft"
    payment_method: str = ""
    paid_date: str = ""
    notes: str = ""
    appointment_id: str = ""
    pdf_path: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.invoice_id[:8] if self.invoice_id else ""

    def to_dict(self) -> dict:
        return {
            "invoice_id": self.invoice_id,
            "invoice_number": self.invoice_number,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "customer_email": self.customer_email,
            "customer_phone": self.customer_phone,
            "service_record_id": self.service_record_id,
            "date": self.date,
            "due_date": self.due_date,
            "line_items": self.line_items,
            "labor_hours": self.labor_hours,
            "labor_rate": self.labor_rate,
            "parts_total": self.parts_total,
            "labor_total": self.labor_total,
            "subtotal": self.subtotal,
            "tax_rate": self.tax_rate,
            "tax_amount": self.tax_amount,
            "total": self.total,
            "status": self.status,
            "payment_method": self.payment_method,
            "paid_date": self.paid_date,
            "notes": self.notes,
            "appointment_id": self.appointment_id,
            "pdf_path": self.pdf_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_row(row) -> "InvoiceRecord":
        """Build an InvoiceRecord from a sqlite3.Row."""
        # sqlite3.Row supports [] access but not .get(), so use dict()
        r = dict(row)
        return InvoiceRecord(
            invoice_id=r["invoice_id"],
            invoice_number=r["invoice_number"],
            customer_id=r.get("customer_id", ""),
            customer_name=r.get("customer_name", ""),
            customer_email=r.get("customer_email", ""),
            customer_phone=r.get("customer_phone", ""),
            service_record_id=r.get("service_record_id", ""),
            date=r.get("date", ""),
            due_date=r.get("due_date", ""),
            line_items=json.loads(r.get("line_items", "[]")),
            labor_hours=float(r.get("labor_hours", 0)),
            labor_rate=float(r.get("labor_rate", 75)),
            parts_total=float(r.get("parts_total", 0)),
            labor_total=float(r.get("labor_total", 0)),
            subtotal=float(r.get("subtotal", 0)),
            tax_rate=float(r.get("tax_rate", 0)),
            tax_amount=float(r.get("tax_amount", 0)),
            total=float(r.get("total", 0)),
            status=r.get("status", "draft"),
            payment_method=r.get("payment_method", ""),
            paid_date=r.get("paid_date", ""),
            notes=r.get("notes", ""),
            appointment_id=r.get("appointment_id", ""),
            pdf_path=r.get("pdf_path", ""),
            created_at=r.get("created_at", ""),
            updated_at=r.get("updated_at", ""),
            metadata=json.loads(r.get("metadata", "{}")),
        )


# ---------------------------------------------------------------------------
# InvoiceManager
# ---------------------------------------------------------------------------

class InvoiceManager:
    """SQLite-backed invoice management for Doppler Cycles."""

    def __init__(
        self,
        db_path: str = "data/invoices.db",
        on_change: Optional[Callable[[], Coroutine]] = None,
        service_store: Any = None,
        default_labor_rate: float = 75.0,
        invoice_prefix: str = "DC",
    ):
        self._db_path = db_path
        self._on_change = on_change
        self._service_store = service_store
        self._default_labor_rate = default_labor_rate
        self._invoice_prefix = invoice_prefix

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("InvoiceManager initialized (db=%s, prefix=%s)", db_path, invoice_prefix)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS invoices (
                invoice_id       TEXT PRIMARY KEY,
                invoice_number   TEXT UNIQUE NOT NULL,
                customer_id      TEXT DEFAULT '',
                customer_name    TEXT DEFAULT '',
                customer_email   TEXT DEFAULT '',
                customer_phone   TEXT DEFAULT '',
                service_record_id TEXT DEFAULT '',
                date             TEXT NOT NULL,
                due_date         TEXT NOT NULL,
                line_items       TEXT DEFAULT '[]',
                labor_hours      REAL DEFAULT 0,
                labor_rate       REAL DEFAULT 75,
                parts_total      REAL DEFAULT 0,
                labor_total      REAL DEFAULT 0,
                subtotal         REAL DEFAULT 0,
                tax_rate         REAL DEFAULT 0,
                tax_amount       REAL DEFAULT 0,
                total            REAL DEFAULT 0,
                status           TEXT DEFAULT 'draft',
                payment_method   TEXT DEFAULT '',
                paid_date        TEXT DEFAULT '',
                notes            TEXT DEFAULT '',
                appointment_id   TEXT DEFAULT '',
                pdf_path         TEXT DEFAULT '',
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                metadata         TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_inv_date ON invoices(date);
            CREATE INDEX IF NOT EXISTS idx_inv_status ON invoices(status);
            CREATE INDEX IF NOT EXISTS idx_inv_customer ON invoices(customer_id);
            CREATE INDEX IF NOT EXISTS idx_inv_service ON invoices(service_record_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Invoice numbering -- DC-2026-0001
    # ------------------------------------------------------------------

    def _next_invoice_number(self, year: int = 0) -> str:
        """Generate the next sequential invoice number for the given year."""
        if year == 0:
            year = date.today().year
        prefix = f"{self._invoice_prefix}-{year}-"
        row = self._conn.execute(
            "SELECT invoice_number FROM invoices WHERE invoice_number LIKE ? ORDER BY invoice_number DESC LIMIT 1",
            (f"{prefix}%",),
        ).fetchone()
        if row:
            try:
                last_serial = int(row["invoice_number"].split("-")[-1])
            except (ValueError, IndexError):
                last_serial = 0
            return f"{prefix}{last_serial + 1:04d}"
        return f"{prefix}0001"

    # ------------------------------------------------------------------
    # Financial calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_totals(
        line_items: list,
        labor_hours: float,
        labor_rate: float,
        tax_rate: float = 0.0,
    ) -> dict:
        """Compute parts_total, labor_total, subtotal, tax_amount, total."""
        parts_total = sum(
            float(item.get("qty", 0)) * float(item.get("unit_price", 0))
            for item in line_items
        )
        labor_total = labor_hours * labor_rate
        subtotal = parts_total + labor_total
        tax_amount = subtotal * (tax_rate / 100.0)
        total = subtotal + tax_amount
        return {
            "parts_total": round(parts_total, 2),
            "labor_total": round(labor_total, 2),
            "subtotal": round(subtotal, 2),
            "tax_amount": round(tax_amount, 2),
            "total": round(total, 2),
        }

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_invoice(
        self,
        customer_name: str,
        customer_id: str = "",
        customer_email: str = "",
        customer_phone: str = "",
        service_record_id: str = "",
        invoice_date: str = "",
        due_date: str = "",
        line_items: Optional[list] = None,
        labor_hours: float = 0.0,
        labor_rate: Optional[float] = None,
        tax_rate: float = 0.0,
        notes: str = "",
        appointment_id: str = "",
        metadata: Optional[dict] = None,
    ) -> InvoiceRecord:
        """Create a new invoice in draft status."""
        inv_id = str(uuid.uuid4())
        now = datetime.now().isoformat(timespec="seconds")
        inv_date = invoice_date or date.today().isoformat()
        if not due_date:
            try:
                d = date.fromisoformat(inv_date)
            except ValueError:
                d = date.today()
            due_date = (d + timedelta(days=30)).isoformat()

        if labor_rate is None:
            labor_rate = self._default_labor_rate
        items = line_items or []
        # Ensure each line item has a 'total' field
        for item in items:
            item["total"] = round(float(item.get("qty", 0)) * float(item.get("unit_price", 0)), 2)

        year = int(inv_date[:4]) if len(inv_date) >= 4 else date.today().year
        inv_number = self._next_invoice_number(year)

        totals = self._calc_totals(items, labor_hours, labor_rate, tax_rate)

        rec = InvoiceRecord(
            invoice_id=inv_id,
            invoice_number=inv_number,
            customer_id=customer_id,
            customer_name=customer_name,
            customer_email=customer_email,
            customer_phone=customer_phone,
            service_record_id=service_record_id,
            date=inv_date,
            due_date=due_date,
            line_items=items,
            labor_hours=labor_hours,
            labor_rate=labor_rate,
            tax_rate=tax_rate,
            notes=notes,
            appointment_id=appointment_id,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            **totals,
        )

        self._conn.execute(
            """INSERT INTO invoices (
                invoice_id, invoice_number, customer_id, customer_name,
                customer_email, customer_phone, service_record_id,
                date, due_date, line_items, labor_hours, labor_rate,
                parts_total, labor_total, subtotal, tax_rate, tax_amount, total,
                status, payment_method, paid_date, notes, appointment_id,
                pdf_path, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.invoice_id, rec.invoice_number, rec.customer_id,
                rec.customer_name, rec.customer_email, rec.customer_phone,
                rec.service_record_id, rec.date, rec.due_date,
                json.dumps(rec.line_items), rec.labor_hours, rec.labor_rate,
                rec.parts_total, rec.labor_total, rec.subtotal,
                rec.tax_rate, rec.tax_amount, rec.total,
                rec.status, rec.payment_method, rec.paid_date,
                rec.notes, rec.appointment_id, rec.pdf_path,
                rec.created_at, rec.updated_at, json.dumps(rec.metadata),
            ),
        )
        self._conn.commit()
        logger.info("Invoice created: %s (%s) for %s — $%.2f",
                     inv_number, inv_id[:8], customer_name, rec.total)
        return rec

    def create_from_service_record(
        self,
        record_id: str,
        parts_markup: float = 0.0,
    ) -> Optional[InvoiceRecord]:
        """Create an invoice from an existing service record.

        Pulls customer info, parts_used → line_items, labor hours/rate.
        parts_markup is a percentage added to part unit costs (0.0 = no markup).
        """
        if self._service_store is None:
            logger.warning("Cannot create from service record -- no service_store configured")
            return None

        record = self._service_store.get_record(record_id)
        if record is None:
            logger.warning("Service record not found: %s", record_id)
            return None

        # Build line items from parts_used
        line_items = []
        for part in (record.parts_used or []):
            qty = float(part.get("quantity", 1))
            unit_cost = float(part.get("unit_cost", 0))
            if parts_markup > 0:
                unit_cost = round(unit_cost * (1 + parts_markup / 100), 2)
            line_items.append({
                "description": part.get("description", "Part"),
                "qty": qty,
                "unit_price": unit_cost,
                "total": round(qty * unit_cost, 2),
            })

        return self.create_invoice(
            customer_name=record.customer_name or "Customer",
            customer_id=getattr(record, "customer_id", ""),
            service_record_id=record_id,
            line_items=line_items,
            labor_hours=record.labor_hours,
            labor_rate=record.labor_rate,
            notes=f"Service: {', '.join(record.services_performed or [])}" if record.services_performed else "",
            appointment_id=getattr(record, "appointment_id", ""),
        )

    def get_invoice(self, invoice_id: str) -> Optional[InvoiceRecord]:
        """Fetch a single invoice by ID."""
        row = self._conn.execute(
            "SELECT * FROM invoices WHERE invoice_id = ?", (invoice_id,)
        ).fetchone()
        return InvoiceRecord.from_row(row) if row else None

    def get_invoice_by_number(self, invoice_number: str) -> Optional[InvoiceRecord]:
        """Fetch a single invoice by its display number (e.g. DC-2026-0001)."""
        row = self._conn.execute(
            "SELECT * FROM invoices WHERE invoice_number = ?", (invoice_number,)
        ).fetchone()
        return InvoiceRecord.from_row(row) if row else None

    def update_invoice(self, invoice_id: str, **kwargs) -> Optional[InvoiceRecord]:
        """Update an existing invoice. Recalculates totals if financial fields change."""
        inv = self.get_invoice(invoice_id)
        if inv is None:
            return None

        financial_fields = {"line_items", "labor_hours", "labor_rate", "tax_rate"}
        needs_recalc = bool(financial_fields & set(kwargs.keys()))

        # Apply updates
        for key, val in kwargs.items():
            if hasattr(inv, key):
                setattr(inv, key, val)

        if needs_recalc:
            # Ensure line item totals are computed
            for item in inv.line_items:
                item["total"] = round(float(item.get("qty", 0)) * float(item.get("unit_price", 0)), 2)
            totals = self._calc_totals(inv.line_items, inv.labor_hours, inv.labor_rate, inv.tax_rate)
            for k, v in totals.items():
                setattr(inv, k, v)

        inv.updated_at = datetime.now().isoformat(timespec="seconds")

        self._conn.execute(
            """UPDATE invoices SET
                customer_id=?, customer_name=?, customer_email=?, customer_phone=?,
                service_record_id=?, date=?, due_date=?, line_items=?,
                labor_hours=?, labor_rate=?, parts_total=?, labor_total=?,
                subtotal=?, tax_rate=?, tax_amount=?, total=?,
                status=?, payment_method=?, paid_date=?, notes=?,
                appointment_id=?, pdf_path=?, updated_at=?, metadata=?
            WHERE invoice_id=?""",
            (
                inv.customer_id, inv.customer_name, inv.customer_email, inv.customer_phone,
                inv.service_record_id, inv.date, inv.due_date, json.dumps(inv.line_items),
                inv.labor_hours, inv.labor_rate, inv.parts_total, inv.labor_total,
                inv.subtotal, inv.tax_rate, inv.tax_amount, inv.total,
                inv.status, inv.payment_method, inv.paid_date, inv.notes,
                inv.appointment_id, inv.pdf_path, inv.updated_at, json.dumps(inv.metadata),
                inv.invoice_id,
            ),
        )
        self._conn.commit()
        logger.info("Invoice updated: %s (%s)", inv.invoice_number, inv.short_id)
        return inv

    def delete_invoice(self, invoice_id: str) -> bool:
        """Delete an invoice. Returns True if deleted."""
        cur = self._conn.execute(
            "DELETE FROM invoices WHERE invoice_id = ?", (invoice_id,)
        )
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Invoice deleted: %s", invoice_id[:8])
        return deleted

    # ------------------------------------------------------------------
    # Payment & status
    # ------------------------------------------------------------------

    def mark_paid(
        self,
        invoice_id: str,
        payment_method: str = "",
        paid_date: str = "",
    ) -> Optional[InvoiceRecord]:
        """Mark an invoice as paid."""
        return self.update_invoice(
            invoice_id,
            status="paid",
            payment_method=payment_method,
            paid_date=paid_date or date.today().isoformat(),
        )

    def mark_sent(self, invoice_id: str) -> Optional[InvoiceRecord]:
        """Mark an invoice as sent (awaiting payment)."""
        return self.update_invoice(invoice_id, status="sent")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_invoices(self, status: Optional[str] = None, limit: int = 100) -> list[InvoiceRecord]:
        """List invoices, optionally filtered by status."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM invoices WHERE status = ? ORDER BY date DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM invoices ORDER BY date DESC LIMIT ?", (limit,)
            ).fetchall()
        return [InvoiceRecord.from_row(r) for r in rows]

    def get_customer_invoices(self, customer_id: str) -> list[InvoiceRecord]:
        """Get all invoices for a specific customer."""
        rows = self._conn.execute(
            "SELECT * FROM invoices WHERE customer_id = ? ORDER BY date DESC",
            (customer_id,),
        ).fetchall()
        return [InvoiceRecord.from_row(r) for r in rows]

    def get_outstanding(self) -> list[InvoiceRecord]:
        """Get invoices that are sent or overdue, ordered by due date."""
        rows = self._conn.execute(
            "SELECT * FROM invoices WHERE status IN ('sent', 'overdue') ORDER BY due_date ASC"
        ).fetchall()
        return [InvoiceRecord.from_row(r) for r in rows]

    def get_outstanding_total(self) -> float:
        """Sum of totals for all outstanding invoices."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(total), 0) as t FROM invoices WHERE status IN ('sent', 'overdue')"
        ).fetchone()
        return round(float(row["t"]), 2)

    def get_monthly_revenue(self, months: int = 6) -> list[dict]:
        """Revenue from paid invoices by month for the last N months.

        Returns: [{month: "2026-02", revenue: float, count: int}, ...]
        """
        cutoff = (date.today().replace(day=1) - timedelta(days=30 * (months - 1))).replace(day=1)
        rows = self._conn.execute(
            """SELECT substr(paid_date, 1, 7) as month,
                      SUM(total) as revenue,
                      COUNT(*) as cnt
               FROM invoices
               WHERE status = 'paid' AND paid_date >= ?
               GROUP BY month
               ORDER BY month ASC""",
            (cutoff.isoformat(),),
        ).fetchall()
        return [{"month": r["month"], "revenue": round(float(r["revenue"]), 2), "count": int(r["cnt"])}
                for r in rows]

    def check_overdue(self) -> list[InvoiceRecord]:
        """Find sent invoices past their due date, mark them overdue.

        Returns the list of newly-overdue invoices.
        """
        today_str = date.today().isoformat()
        rows = self._conn.execute(
            "SELECT * FROM invoices WHERE status = 'sent' AND due_date < ?",
            (today_str,),
        ).fetchall()
        newly_overdue = []
        for row in rows:
            inv = InvoiceRecord.from_row(row)
            updated = self.update_invoice(inv.invoice_id, status="overdue")
            if updated:
                newly_overdue.append(updated)
                logger.info("Invoice now overdue: %s (%s)", updated.invoice_number, updated.short_id)
        return newly_overdue

    # ------------------------------------------------------------------
    # PDF generation (fpdf2 -- matches service_records.py branding)
    # ------------------------------------------------------------------

    def generate_pdf(self, invoice_id: str) -> Optional[str]:
        """Generate a professional PDF invoice.

        Returns the file path, or None if invoice not found.
        """
        inv = self.get_invoice(invoice_id)
        if inv is None:
            return None

        pdf_dir = Path("data/invoices")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = str(pdf_dir / f"{invoice_id}.pdf")

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
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "INVOICE", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(2)

        # --- Invoice info ---
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "  Invoice Details", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(95, 6, f"  Invoice #: {inv.invoice_number}", new_x="RIGHT")
        pdf.cell(95, 6, f"Date: {inv.date}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(95, 6, f"  Status: {inv.status.upper()}", new_x="RIGHT")
        pdf.cell(95, 6, f"Due Date: {inv.due_date}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # --- Bill To ---
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "  Bill To", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  {inv.customer_name}", new_x="LMARGIN", new_y="NEXT")
        if inv.customer_email:
            pdf.cell(0, 6, f"  {inv.customer_email}", new_x="LMARGIN", new_y="NEXT")
        if inv.customer_phone:
            pdf.cell(0, 6, f"  {inv.customer_phone}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # --- Line Items Table ---
        if inv.line_items:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "  Parts & Materials", new_x="LMARGIN", new_y="NEXT", fill=True)

            # Table header
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(80, 6, "  Description", border="B")
            pdf.cell(25, 6, "Qty", border="B", align="C")
            pdf.cell(35, 6, "Unit Price", border="B", align="R")
            pdf.cell(40, 6, "Total", border="B", align="R", new_x="LMARGIN", new_y="NEXT")

            # Table rows
            pdf.set_font("Helvetica", "", 9)
            for item in inv.line_items:
                desc = str(item.get("description", ""))
                qty = float(item.get("qty", 0))
                up = float(item.get("unit_price", 0))
                lt = float(item.get("total", qty * up))
                pdf.cell(80, 5, f"  {desc}")
                pdf.cell(25, 5, f"{qty:g}", align="C")
                pdf.cell(35, 5, f"${up:.2f}", align="R")
                pdf.cell(40, 5, f"${lt:.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

        # --- Labor ---
        if inv.labor_hours > 0:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "  Labor", new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6,
                     f"  {inv.labor_hours:.1f} hours x ${inv.labor_rate:.2f}/hr = ${inv.labor_total:.2f}",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

        # --- Cost Summary ---
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "  Cost Summary", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(120, 6, "  Parts Total:", new_x="RIGHT")
        pdf.cell(70, 6, f"${inv.parts_total:.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(120, 6, "  Labor Total:", new_x="RIGHT")
        pdf.cell(70, 6, f"${inv.labor_total:.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(120, 6, "  Subtotal:", new_x="RIGHT")
        pdf.cell(70, 6, f"${inv.subtotal:.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
        if inv.tax_rate > 0:
            pdf.cell(120, 6, f"  Tax ({inv.tax_rate:.1f}%):", new_x="RIGHT")
            pdf.cell(70, 6, f"${inv.tax_amount:.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(120, 8, "  TOTAL:", new_x="RIGHT")
        pdf.cell(70, 8, f"${inv.total:.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # --- Payment Terms ---
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  Payment Terms: Net 30 -Payment due by {inv.due_date}",
                 new_x="LMARGIN", new_y="NEXT")
        if inv.status == "paid":
            pdf.set_font("Helvetica", "B", 10)
            method = f" ({inv.payment_method})" if inv.payment_method else ""
            pdf.cell(0, 6, f"  PAID on {inv.paid_date}{method}",
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # --- Notes ---
        if inv.notes:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "  Notes", new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, f"  {inv.notes}")
            pdf.ln(4)

        # --- Footer ---
        pdf.ln(8)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, "Thank you for choosing Doppler Cycles", new_x="LMARGIN", new_y="NEXT", align="C")
        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        pdf.cell(0, 5, f"Invoice generated: {generated}", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.output(pdf_path)

        # Store the path on the record
        self.update_invoice(invoice_id, pdf_path=pdf_path)
        logger.info("Invoice PDF generated: %s", pdf_path)
        return pdf_path

    # ------------------------------------------------------------------
    # Email placeholder
    # ------------------------------------------------------------------

    def send_invoice(self, invoice_id: str) -> dict:
        """Attempt to send an invoice (placeholder -- email not yet configured).

        Generates PDF if needed, marks the invoice as sent.
        """
        inv = self.get_invoice(invoice_id)
        if inv is None:
            return {"ok": False, "message": "Invoice not found"}

        # Generate PDF if it doesn't exist
        if not inv.pdf_path or not Path(inv.pdf_path).exists():
            self.generate_pdf(invoice_id)

        # Mark as sent
        self.mark_sent(invoice_id)

        return {
            "ok": False,
            "message": "Email not configured -- PDF generated for manual sending",
            "invoice_number": inv.invoice_number,
            "pdf_path": inv.pdf_path or f"data/invoices/{invoice_id}.pdf",
        }

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Summary status dict for dashboard cards."""
        counts = {}
        for s in VALID_STATUSES:
            row = self._conn.execute(
                "SELECT COUNT(*) as c FROM invoices WHERE status = ?", (s,)
            ).fetchone()
            counts[s] = int(row["c"])

        total_row = self._conn.execute("SELECT COUNT(*) as c FROM invoices").fetchone()

        # Revenue this month
        month_prefix = date.today().strftime("%Y-%m")
        rev_row = self._conn.execute(
            "SELECT COALESCE(SUM(total), 0) as r FROM invoices WHERE status = 'paid' AND substr(paid_date, 1, 7) = ?",
            (month_prefix,),
        ).fetchone()

        return {
            "total_invoices": int(total_row["c"]),
            "draft": counts.get("draft", 0),
            "sent": counts.get("sent", 0),
            "paid": counts.get("paid", 0),
            "overdue": counts.get("overdue", 0),
            "cancelled": counts.get("cancelled", 0),
            "outstanding_total": self.get_outstanding_total(),
            "revenue_this_month": round(float(rev_row["r"]), 2),
        }

    def to_broadcast_dict(self) -> dict:
        """Full state for dashboard WS broadcast."""
        return {
            "invoices": [inv.to_dict() for inv in self.list_invoices(limit=200)],
            "status": self.get_status(),
            "monthly_revenue": self.get_monthly_revenue(months=6),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close the database connection."""
        self._conn.close()
        logger.info("InvoiceManager closed")
