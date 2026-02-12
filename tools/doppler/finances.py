"""
Financial Dashboard & Runway Monitor for Doppler Cycles.

Tracks income and expenses in a standalone SQLite DB with:
- Manual transaction entry (income + expenses by category)
- Automatic income sync from paid invoices (idempotent via reference_id)
- Monthly summary, trend analysis, and category breakdowns
- Runway projection based on rolling 3-month averages
- Dashboard broadcast integration via on_change callback

Separate DB (data/finances.db) — same pattern as invoicing.py / inventory.py.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# Valid categories for income and expense transactions
INCOME_CATEGORIES = {
    "diagnostic_service", "repair", "consultation",
    "content_revenue", "parts_sale",
}
EXPENSE_CATEGORIES = {
    "parts_cost", "fuel", "tools", "subscriptions",
    "hosting", "insurance", "misc",
}


# ---------------------------------------------------------------------------
# Transaction dataclass
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """A single financial transaction (income or expense)."""
    txn_id: str = ""
    date: str = ""                  # YYYY-MM-DD
    type: str = ""                  # "income" or "expense"
    category: str = ""
    amount: float = 0.0             # Always positive
    description: str = ""
    reference_id: str = ""          # Links to invoice_id or other source
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.txn_id[:8] if self.txn_id else ""

    def to_dict(self) -> dict:
        return {
            "txn_id": self.txn_id,
            "date": self.date,
            "type": self.type,
            "category": self.category,
            "amount": self.amount,
            "description": self.description,
            "reference_id": self.reference_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_row(row) -> "Transaction":
        """Build a Transaction from a sqlite3.Row."""
        r = dict(row)
        return Transaction(
            txn_id=r["txn_id"],
            date=r.get("date", ""),
            type=r.get("type", ""),
            category=r.get("category", ""),
            amount=float(r.get("amount", 0)),
            description=r.get("description", ""),
            reference_id=r.get("reference_id", ""),
            created_at=r.get("created_at", ""),
            metadata=json.loads(r.get("metadata", "{}")),
        )


# ---------------------------------------------------------------------------
# FinanceTracker
# ---------------------------------------------------------------------------

class FinanceTracker:
    """SQLite-backed finance tracking for Doppler Cycles."""

    def __init__(
        self,
        db_path: str = "data/finances.db",
        on_change: Optional[Callable[[], Coroutine]] = None,
        invoice_manager=None,
    ):
        self._db_path = db_path
        self._on_change = on_change
        self._invoice_manager = invoice_manager

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("FinanceTracker initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS transactions (
                txn_id        TEXT PRIMARY KEY,
                date          TEXT NOT NULL,
                type          TEXT NOT NULL,
                category      TEXT NOT NULL,
                amount        REAL NOT NULL,
                description   TEXT DEFAULT '',
                reference_id  TEXT DEFAULT '',
                created_at    TEXT NOT NULL,
                metadata      TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date);
            CREATE INDEX IF NOT EXISTS idx_txn_type ON transactions(type);
            CREATE INDEX IF NOT EXISTS idx_txn_category ON transactions(category);
            CREATE INDEX IF NOT EXISTS idx_txn_reference ON transactions(reference_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Notify callback
    # ------------------------------------------------------------------

    async def _notify(self):
        """Fire the on_change callback if set."""
        if self._on_change:
            try:
                await self._on_change()
            except Exception as e:
                logger.error("Finance on_change callback failed: %s", e)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_transaction(
        self,
        date_str: str,
        txn_type: str,
        category: str,
        amount: float,
        description: str = "",
        reference_id: str = "",
        metadata: Optional[dict] = None,
    ) -> Transaction:
        """Add a new transaction. Returns the created Transaction."""
        if txn_type not in ("income", "expense"):
            raise ValueError(f"Invalid type '{txn_type}', must be 'income' or 'expense'")
        valid_cats = INCOME_CATEGORIES if txn_type == "income" else EXPENSE_CATEGORIES
        if category not in valid_cats:
            raise ValueError(f"Invalid category '{category}' for type '{txn_type}'")
        if amount <= 0:
            raise ValueError("Amount must be positive")

        txn = Transaction(
            txn_id=str(uuid.uuid4()),
            date=date_str,
            type=txn_type,
            category=category,
            amount=round(amount, 2),
            description=description,
            reference_id=reference_id,
            created_at=datetime.utcnow().isoformat(),
            metadata=metadata or {},
        )
        self._conn.execute(
            """INSERT INTO transactions
               (txn_id, date, type, category, amount, description, reference_id, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                txn.txn_id, txn.date, txn.type, txn.category, txn.amount,
                txn.description, txn.reference_id, txn.created_at,
                json.dumps(txn.metadata),
            ),
        )
        self._conn.commit()
        logger.info("Transaction added: %s %s $%.2f (%s)", txn.short_id, txn.type, txn.amount, txn.category)
        return txn

    def update_transaction(self, txn_id: str, **kwargs) -> Optional[Transaction]:
        """Update fields on an existing transaction. Returns updated Transaction or None."""
        row = self._conn.execute(
            "SELECT * FROM transactions WHERE txn_id = ?", (txn_id,)
        ).fetchone()
        if not row:
            return None

        txn = Transaction.from_row(row)
        allowed = {"date", "type", "category", "amount", "description", "reference_id", "metadata"}
        updates = {}
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if k == "type" and v not in ("income", "expense"):
                raise ValueError(f"Invalid type '{v}'")
            if k == "category":
                txn_type = kwargs.get("type", txn.type)
                valid_cats = INCOME_CATEGORIES if txn_type == "income" else EXPENSE_CATEGORIES
                if v not in valid_cats:
                    raise ValueError(f"Invalid category '{v}' for type '{txn_type}'")
            if k == "amount":
                v = round(float(v), 2)
                if v <= 0:
                    raise ValueError("Amount must be positive")
            if k == "metadata":
                v = json.dumps(v)
            updates[k] = v

        if not updates:
            return txn

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [txn_id]
        self._conn.execute(
            f"UPDATE transactions SET {set_clause} WHERE txn_id = ?", vals
        )
        self._conn.commit()

        row = self._conn.execute(
            "SELECT * FROM transactions WHERE txn_id = ?", (txn_id,)
        ).fetchone()
        updated = Transaction.from_row(row)
        logger.info("Transaction updated: %s", updated.short_id)
        return updated

    def delete_transaction(self, txn_id: str) -> bool:
        """Delete a transaction by ID. Returns True if deleted."""
        cur = self._conn.execute(
            "DELETE FROM transactions WHERE txn_id = ?", (txn_id,)
        )
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Transaction deleted: %s", txn_id[:8])
        return deleted

    def list_transactions(
        self,
        txn_type: Optional[str] = None,
        category: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 200,
    ) -> list[Transaction]:
        """List transactions with optional filters."""
        clauses = []
        params = []
        if txn_type:
            clauses.append("type = ?")
            params.append(txn_type)
        if category:
            clauses.append("category = ?")
            params.append(category)
        if start_date:
            clauses.append("date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("date <= ?")
            params.append(end_date)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM transactions{where} ORDER BY date DESC LIMIT ?", params
        ).fetchall()
        return [Transaction.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Invoice sync
    # ------------------------------------------------------------------

    def sync_from_invoices(self) -> int:
        """Pull paid invoices and upsert as income transactions.

        Uses reference_id = invoice_id to deduplicate. Returns count of
        new transactions created.
        """
        if not self._invoice_manager:
            logger.debug("No invoice_manager set, skipping sync")
            return 0

        paid_invoices = self._invoice_manager.list_invoices(status="paid", limit=9999)
        created = 0
        for inv in paid_invoices:
            # Skip if already synced
            existing = self._conn.execute(
                "SELECT txn_id FROM transactions WHERE reference_id = ?",
                (inv.invoice_id,),
            ).fetchone()
            if existing:
                continue

            self.add_transaction(
                date_str=inv.paid_date or inv.date,
                txn_type="income",
                category="diagnostic_service",
                amount=inv.total,
                description=f"Invoice {inv.invoice_number} — {inv.customer_name}",
                reference_id=inv.invoice_id,
                metadata={"source": "invoice_sync", "invoice_number": inv.invoice_number},
            )
            created += 1

        if created:
            logger.info("Invoice sync: %d new transactions", created)
        return created

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def monthly_summary(self, months: int = 12) -> list[dict]:
        """Monthly income/expense/net grouped by YYYY-MM, most recent first."""
        rows = self._conn.execute("""
            SELECT
                substr(date, 1, 7) AS month,
                SUM(CASE WHEN type = 'income' THEN amount ELSE 0 END) AS income,
                SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END) AS expenses,
                COUNT(*) AS txn_count
            FROM transactions
            GROUP BY month
            ORDER BY month DESC
            LIMIT ?
        """, (months,)).fetchall()

        return [
            {
                "month": r["month"],
                "income": round(float(r["income"]), 2),
                "expenses": round(float(r["expenses"]), 2),
                "net": round(float(r["income"]) - float(r["expenses"]), 2),
                "txn_count": int(r["txn_count"]),
            }
            for r in rows
        ]

    def runway_calculator(self, current_balance: float, months_lookback: int = 3) -> dict:
        """Calculate runway based on rolling average of recent months.

        Returns dict with balance, averages, runway months/date, and status.
        """
        summary = self.monthly_summary(months=months_lookback)

        if not summary:
            return {
                "current_balance": round(current_balance, 2),
                "avg_monthly_income": 0,
                "avg_monthly_expenses": 0,
                "avg_monthly_net": 0,
                "monthly_burn_rate": 0,
                "runway_months": None,
                "runway_date": None,
                "status": "stable",
            }

        n = len(summary)
        avg_income = round(sum(m["income"] for m in summary) / n, 2)
        avg_expenses = round(sum(m["expenses"] for m in summary) / n, 2)
        avg_net = round(avg_income - avg_expenses, 2)
        burn_rate = round(max(avg_expenses - avg_income, 0), 2)

        # Determine status
        if avg_net > 0:
            status = "growing"
        elif burn_rate == 0:
            status = "stable"
        else:
            status = "burning"

        # Runway calculation
        runway_months = None
        runway_date = None
        if burn_rate > 0:
            runway_months = round(current_balance / burn_rate, 1) if burn_rate else None
            if runway_months is not None:
                from dateutil.relativedelta import relativedelta
                try:
                    rd = relativedelta(months=int(runway_months))
                    runway_date = (date.today() + rd).isoformat()
                except Exception:
                    runway_date = None

        return {
            "current_balance": round(current_balance, 2),
            "avg_monthly_income": avg_income,
            "avg_monthly_expenses": avg_expenses,
            "avg_monthly_net": avg_net,
            "monthly_burn_rate": burn_rate,
            "runway_months": runway_months,
            "runway_date": runway_date,
            "status": status,
        }

    def trend_analysis(self, months: int = 6) -> list[dict]:
        """Monthly trend with percent change from previous month."""
        summary = self.monthly_summary(months=months)
        # summary is newest-first; reverse for chronological processing
        summary.reverse()

        result = []
        for i, m in enumerate(summary):
            entry = {
                "month": m["month"],
                "income": m["income"],
                "expenses": m["expenses"],
                "net": m["net"],
                "income_change_pct": None,
                "expense_change_pct": None,
            }
            if i > 0:
                prev = summary[i - 1]
                if prev["income"] > 0:
                    entry["income_change_pct"] = round(
                        ((m["income"] - prev["income"]) / prev["income"]) * 100, 1
                    )
                if prev["expenses"] > 0:
                    entry["expense_change_pct"] = round(
                        ((m["expenses"] - prev["expenses"]) / prev["expenses"]) * 100, 1
                    )
            result.append(entry)

        return result

    def top_expense_categories(self, months: int = 3) -> list[dict]:
        """Top expense categories by total over the last N months."""
        rows = self._conn.execute("""
            SELECT category, SUM(amount) AS total
            FROM transactions
            WHERE type = 'expense'
              AND date >= date('now', ?)
            GROUP BY category
            ORDER BY total DESC
        """, (f"-{months} months",)).fetchall()

        grand_total = sum(float(r["total"]) for r in rows) or 1
        return [
            {
                "category": r["category"],
                "total": round(float(r["total"]), 2),
                "percentage": round(float(r["total"]) / grand_total * 100, 1),
            }
            for r in rows
        ]

    def revenue_by_service_type(self, months: int = 3) -> list[dict]:
        """Income breakdown by category over the last N months."""
        rows = self._conn.execute("""
            SELECT category, SUM(amount) AS total
            FROM transactions
            WHERE type = 'income'
              AND date >= date('now', ?)
            GROUP BY category
            ORDER BY total DESC
        """, (f"-{months} months",)).fetchall()

        grand_total = sum(float(r["total"]) for r in rows) or 1
        return [
            {
                "category": r["category"],
                "total": round(float(r["total"]), 2),
                "percentage": round(float(r["total"]) / grand_total * 100, 1),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Summary status dict for dashboard cards."""
        row = self._conn.execute("""
            SELECT
                COALESCE(SUM(CASE WHEN type = 'income' THEN amount ELSE 0 END), 0) AS total_income,
                COALESCE(SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END), 0) AS total_expenses,
                COUNT(*) AS txn_count
            FROM transactions
        """).fetchone()

        month_prefix = date.today().strftime("%Y-%m")
        month_row = self._conn.execute("""
            SELECT
                COALESCE(SUM(CASE WHEN type = 'income' THEN amount ELSE 0 END), 0) AS month_income,
                COALESCE(SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END), 0) AS month_expenses
            FROM transactions
            WHERE substr(date, 1, 7) = ?
        """, (month_prefix,)).fetchone()

        total_income = round(float(row["total_income"]), 2)
        total_expenses = round(float(row["total_expenses"]), 2)

        return {
            "total_income": total_income,
            "total_expenses": total_expenses,
            "net": round(total_income - total_expenses, 2),
            "this_month_income": round(float(month_row["month_income"]), 2),
            "this_month_expenses": round(float(month_row["month_expenses"]), 2),
            "txn_count": int(row["txn_count"]),
        }

    def to_broadcast_dict(self, current_balance: float = 0) -> dict:
        """Full state for dashboard WS broadcast."""
        return {
            "transactions": [t.to_dict() for t in self.list_transactions(limit=100)],
            "status": self.get_status(),
            "monthly_summary": self.monthly_summary(months=12),
            "runway": self.runway_calculator(current_balance),
            "top_expenses": self.top_expense_categories(months=3),
            "revenue_by_type": self.revenue_by_service_type(months=3),
            "trend": self.trend_analysis(months=6),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close the database connection."""
        self._conn.close()
        logger.info("FinanceTracker closed")
