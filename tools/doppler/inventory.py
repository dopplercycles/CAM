"""
Parts Inventory Manager for Doppler Cycles.

Tracks parts and supplies for mobile diagnostic work with:
- Full CRUD for parts (filters, fluids, electrical, brakes, etc.)
- Usage logging per service record or manual adjustment
- Low-stock alerts via notification callback
- Cost/margin reporting for dashboard summary cards
- Top-used-parts aggregation for usage charts

Separate SQLite DB (data/inventory.db) — does not touch business.db.
Same pattern as invoicing.py / service_records.py.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# Valid part categories
CATEGORIES = {
    "filters", "fluids", "electrical", "brakes",
    "drivetrain", "fasteners", "consumables", "tools",
}


# ---------------------------------------------------------------------------
# Part dataclass
# ---------------------------------------------------------------------------

@dataclass
class Part:
    """A single inventory part."""
    part_id: str = ""
    part_number: str = ""
    name: str = ""
    description: str = ""
    category: str = ""
    quantity_on_hand: int = 0
    reorder_point: int = 0
    cost: float = 0.0           # George's cost
    retail_price: float = 0.0   # Customer price
    supplier: str = ""
    location: str = ""
    last_ordered: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.part_id[:8] if self.part_id else ""

    @property
    def needs_reorder(self) -> bool:
        """True when qty is at or below reorder point (and reorder is enabled)."""
        return self.reorder_point > 0 and self.quantity_on_hand <= self.reorder_point

    @property
    def margin(self) -> float:
        """Markup percentage: ((retail - cost) / cost) * 100."""
        if self.cost <= 0:
            return 0.0
        return ((self.retail_price - self.cost) / self.cost) * 100

    @property
    def inventory_value(self) -> float:
        """Total value at cost for qty on hand."""
        return self.quantity_on_hand * self.cost

    def to_dict(self) -> dict:
        return {
            "part_id": self.part_id,
            "part_number": self.part_number,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "quantity_on_hand": self.quantity_on_hand,
            "reorder_point": self.reorder_point,
            "cost": self.cost,
            "retail_price": self.retail_price,
            "supplier": self.supplier,
            "location": self.location,
            "last_ordered": self.last_ordered,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            # Computed properties
            "short_id": self.short_id,
            "needs_reorder": self.needs_reorder,
            "margin": round(self.margin, 1),
            "inventory_value": round(self.inventory_value, 2),
        }

    @staticmethod
    def from_row(row) -> "Part":
        """Build a Part from a sqlite3.Row."""
        r = dict(row)
        return Part(
            part_id=r["part_id"],
            part_number=r.get("part_number", ""),
            name=r.get("name", ""),
            description=r.get("description", ""),
            category=r.get("category", ""),
            quantity_on_hand=int(r.get("quantity_on_hand", 0)),
            reorder_point=int(r.get("reorder_point", 0)),
            cost=float(r.get("cost", 0)),
            retail_price=float(r.get("retail_price", 0)),
            supplier=r.get("supplier", ""),
            location=r.get("location", ""),
            last_ordered=r.get("last_ordered", ""),
            notes=r.get("notes", ""),
            created_at=r.get("created_at", ""),
            updated_at=r.get("updated_at", ""),
            metadata=json.loads(r.get("metadata", "{}")),
        )


# ---------------------------------------------------------------------------
# UsageEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class UsageEntry:
    """A single usage log entry."""
    log_id: str = ""
    part_id: str = ""
    part_number: str = ""
    part_name: str = ""
    quantity_used: int = 0
    service_record_id: str = ""
    reason: str = ""            # "service_record" / "manual" / "adjustment"
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "log_id": self.log_id,
            "part_id": self.part_id,
            "part_number": self.part_number,
            "part_name": self.part_name,
            "quantity_used": self.quantity_used,
            "service_record_id": self.service_record_id,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_row(row) -> "UsageEntry":
        """Build a UsageEntry from a sqlite3.Row."""
        r = dict(row)
        return UsageEntry(
            log_id=r["log_id"],
            part_id=r.get("part_id", ""),
            part_number=r.get("part_number", ""),
            part_name=r.get("part_name", ""),
            quantity_used=int(r.get("quantity_used", 0)),
            service_record_id=r.get("service_record_id", ""),
            reason=r.get("reason", ""),
            timestamp=r.get("timestamp", ""),
            metadata=json.loads(r.get("metadata", "{}")),
        )


# ---------------------------------------------------------------------------
# InventoryManager
# ---------------------------------------------------------------------------

class InventoryManager:
    """SQLite-backed parts inventory for Doppler Cycles."""

    def __init__(
        self,
        db_path: str = "data/inventory.db",
        on_change: Optional[Callable[[], Coroutine]] = None,
        notification_callback: Optional[Callable] = None,
    ):
        self._db_path = db_path
        self._on_change = on_change
        self._notification_callback = notification_callback

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("InventoryManager initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS parts (
                part_id          TEXT PRIMARY KEY,
                part_number      TEXT DEFAULT '',
                name             TEXT NOT NULL DEFAULT '',
                description      TEXT DEFAULT '',
                category         TEXT DEFAULT '',
                quantity_on_hand INTEGER DEFAULT 0,
                reorder_point    INTEGER DEFAULT 0,
                cost             REAL DEFAULT 0,
                retail_price     REAL DEFAULT 0,
                supplier         TEXT DEFAULT '',
                location         TEXT DEFAULT '',
                last_ordered     TEXT DEFAULT '',
                notes            TEXT DEFAULT '',
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                metadata         TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS usage_log (
                log_id            TEXT PRIMARY KEY,
                part_id           TEXT NOT NULL,
                part_number       TEXT DEFAULT '',
                part_name         TEXT DEFAULT '',
                quantity_used     INTEGER DEFAULT 0,
                service_record_id TEXT DEFAULT '',
                reason            TEXT DEFAULT 'manual',
                timestamp         TEXT NOT NULL,
                metadata          TEXT DEFAULT '{}'
            );
        """)
        # Indexes for common queries
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_parts_part_number ON parts(part_number)",
            "CREATE INDEX IF NOT EXISTS idx_parts_name ON parts(name)",
            "CREATE INDEX IF NOT EXISTS idx_parts_category ON parts(category)",
            "CREATE INDEX IF NOT EXISTS idx_usage_part_id ON usage_log(part_id)",
            "CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_log(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_usage_service_record ON usage_log(service_record_id)",
        ]:
            self._conn.execute(idx_sql)
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_part(
        self,
        name: str,
        part_number: str = "",
        description: str = "",
        category: str = "",
        quantity_on_hand: int = 0,
        reorder_point: int = 0,
        cost: float = 0.0,
        retail_price: float = 0.0,
        supplier: str = "",
        location: str = "",
        notes: str = "",
        metadata: Optional[dict] = None,
    ) -> Part:
        """Add a new part to inventory."""
        now = datetime.utcnow().isoformat()
        part = Part(
            part_id=str(uuid.uuid4()),
            part_number=part_number.strip(),
            name=name.strip(),
            description=description.strip(),
            category=category.strip().lower(),
            quantity_on_hand=max(0, int(quantity_on_hand)),
            reorder_point=max(0, int(reorder_point)),
            cost=float(cost),
            retail_price=float(retail_price),
            supplier=supplier.strip(),
            location=location.strip(),
            notes=notes.strip(),
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        self._conn.execute(
            """INSERT INTO parts (part_id, part_number, name, description,
               category, quantity_on_hand, reorder_point, cost, retail_price,
               supplier, location, last_ordered, notes, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                part.part_id, part.part_number, part.name, part.description,
                part.category, part.quantity_on_hand, part.reorder_point,
                part.cost, part.retail_price, part.supplier, part.location,
                part.last_ordered, part.notes, part.created_at, part.updated_at,
                json.dumps(part.metadata),
            ),
        )
        self._conn.commit()
        logger.info("Part added: %s (%s) — qty %d", part.name, part.short_id, part.quantity_on_hand)
        self._check_reorder_alert(part)
        return part

    def update_part(self, part_id: str, **kwargs) -> Optional[Part]:
        """Update an existing part. Returns updated Part or None if not found."""
        part = self.get_part(part_id)
        if not part:
            return None

        allowed = {
            "part_number", "name", "description", "category",
            "quantity_on_hand", "reorder_point", "cost", "retail_price",
            "supplier", "location", "last_ordered", "notes", "metadata",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return part

        # Type coercion
        if "quantity_on_hand" in updates:
            updates["quantity_on_hand"] = max(0, int(updates["quantity_on_hand"]))
        if "reorder_point" in updates:
            updates["reorder_point"] = max(0, int(updates["reorder_point"]))
        if "cost" in updates:
            updates["cost"] = float(updates["cost"])
        if "retail_price" in updates:
            updates["retail_price"] = float(updates["retail_price"])
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])
        elif "metadata" in updates and isinstance(updates["metadata"], str):
            pass  # already JSON string

        updates["updated_at"] = datetime.utcnow().isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [part_id]
        self._conn.execute(
            f"UPDATE parts SET {set_clause} WHERE part_id = ?", values
        )
        self._conn.commit()

        updated = self.get_part(part_id)
        if updated:
            logger.info("Part updated: %s (%s)", updated.name, updated.short_id)
            self._check_reorder_alert(updated)
        return updated

    def get_part(self, part_id: str) -> Optional[Part]:
        """Get a part by its UUID."""
        row = self._conn.execute(
            "SELECT * FROM parts WHERE part_id = ?", (part_id,)
        ).fetchone()
        return Part.from_row(row) if row else None

    def get_part_by_number(self, part_number: str) -> Optional[Part]:
        """Get a part by its part number."""
        if not part_number:
            return None
        row = self._conn.execute(
            "SELECT * FROM parts WHERE part_number = ? COLLATE NOCASE", (part_number.strip(),)
        ).fetchone()
        return Part.from_row(row) if row else None

    def delete_part(self, part_id: str) -> bool:
        """Delete a part by ID. Returns True if deleted."""
        cur = self._conn.execute(
            "DELETE FROM parts WHERE part_id = ?", (part_id,)
        )
        self._conn.commit()
        if cur.rowcount > 0:
            logger.info("Part deleted: %s", part_id[:8])
            return True
        return False

    def list_parts(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 200,
    ) -> list:
        """List parts with optional category filter and search.

        Search matches name, part_number, or description via LIKE.
        """
        sql = "SELECT * FROM parts WHERE 1=1"
        params = []

        if category:
            sql += " AND category = ?"
            params.append(category.strip().lower())

        if search:
            term = f"%{search.strip()}%"
            sql += " AND (name LIKE ? OR part_number LIKE ? OR description LIKE ?)"
            params.extend([term, term, term])

        sql += " ORDER BY name ASC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [Part.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def use_part(
        self,
        part_id: str,
        quantity: int = 1,
        service_record_id: str = "",
        reason: str = "service_record",
    ) -> Optional[UsageEntry]:
        """Decrement part qty, log usage. Returns UsageEntry or None if part not found."""
        part = self.get_part(part_id)
        if not part:
            return None

        quantity = max(1, int(quantity))
        new_qty = max(0, part.quantity_on_hand - quantity)

        self._conn.execute(
            "UPDATE parts SET quantity_on_hand = ?, updated_at = ? WHERE part_id = ?",
            (new_qty, datetime.utcnow().isoformat(), part_id),
        )

        entry = UsageEntry(
            log_id=str(uuid.uuid4()),
            part_id=part.part_id,
            part_number=part.part_number,
            part_name=part.name,
            quantity_used=quantity,
            service_record_id=service_record_id,
            reason=reason,
            timestamp=datetime.utcnow().isoformat(),
        )
        self._conn.execute(
            """INSERT INTO usage_log (log_id, part_id, part_number, part_name,
               quantity_used, service_record_id, reason, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.log_id, entry.part_id, entry.part_number, entry.part_name,
                entry.quantity_used, entry.service_record_id, entry.reason,
                entry.timestamp, json.dumps(entry.metadata),
            ),
        )
        self._conn.commit()
        logger.info(
            "Part used: %s x%d (%s) — qty now %d",
            part.name, quantity, reason, new_qty,
        )

        # Refresh part to check reorder with updated qty
        updated = self.get_part(part_id)
        if updated:
            self._check_reorder_alert(updated)

        return entry

    def use_part_by_number(
        self,
        part_number: str,
        quantity: int = 1,
        service_record_id: str = "",
        reason: str = "service_record",
    ) -> Optional[UsageEntry]:
        """Look up a part by number and use it. Returns None silently if no match."""
        part = self.get_part_by_number(part_number)
        if not part:
            return None
        return self.use_part(part.part_id, quantity, service_record_id, reason)

    def get_usage_history(
        self,
        part_id: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """Get usage history, optionally filtered by part_id."""
        if part_id:
            rows = self._conn.execute(
                "SELECT * FROM usage_log WHERE part_id = ? ORDER BY timestamp DESC LIMIT ?",
                (part_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM usage_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [UsageEntry.from_row(r) for r in rows]

    def get_top_used_parts(self, limit: int = 10) -> list:
        """Get most-used parts by total quantity consumed.

        Returns: [{part_id, part_name, part_number, total_used}]
        """
        rows = self._conn.execute(
            """SELECT part_id, part_name, part_number,
                      SUM(quantity_used) AS total_used
               FROM usage_log
               GROUP BY part_id
               ORDER BY total_used DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "part_id": dict(r)["part_id"],
                "part_name": dict(r)["part_name"],
                "part_number": dict(r)["part_number"],
                "total_used": int(dict(r)["total_used"]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_low_stock(self) -> list:
        """Get all parts at or below reorder point."""
        rows = self._conn.execute(
            """SELECT * FROM parts
               WHERE reorder_point > 0 AND quantity_on_hand <= reorder_point
               ORDER BY quantity_on_hand ASC""",
        ).fetchall()
        return [Part.from_row(r) for r in rows]

    def reorder_check(self) -> list:
        """Alias for get_low_stock() — used by WS handler."""
        return self.get_low_stock()

    def cost_report(self) -> dict:
        """Generate inventory cost/margin report.

        Returns: {total_parts, total_value_at_cost, total_value_at_retail,
                  potential_margin, low_stock_count,
                  categories: {cat: {count, value_at_cost, value_at_retail}}}
        """
        parts = self.list_parts(limit=10000)
        categories = {}
        total_cost = 0.0
        total_retail = 0.0
        low_stock = 0

        for p in parts:
            val_cost = p.quantity_on_hand * p.cost
            val_retail = p.quantity_on_hand * p.retail_price
            total_cost += val_cost
            total_retail += val_retail
            if p.needs_reorder:
                low_stock += 1

            cat = p.category or "uncategorized"
            if cat not in categories:
                categories[cat] = {"count": 0, "value_at_cost": 0.0, "value_at_retail": 0.0}
            categories[cat]["count"] += 1
            categories[cat]["value_at_cost"] += val_cost
            categories[cat]["value_at_retail"] += val_retail

        # Round category values
        for cat_data in categories.values():
            cat_data["value_at_cost"] = round(cat_data["value_at_cost"], 2)
            cat_data["value_at_retail"] = round(cat_data["value_at_retail"], 2)

        return {
            "total_parts": len(parts),
            "total_value_at_cost": round(total_cost, 2),
            "total_value_at_retail": round(total_retail, 2),
            "potential_margin": round(total_retail - total_cost, 2),
            "low_stock_count": low_stock,
            "categories": categories,
        }

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Quick status dict for summary cards."""
        total = self._conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
        low = self._conn.execute(
            "SELECT COUNT(*) FROM parts WHERE reorder_point > 0 AND quantity_on_hand <= reorder_point"
        ).fetchone()[0]
        value_row = self._conn.execute(
            "SELECT COALESCE(SUM(quantity_on_hand * cost), 0) FROM parts"
        ).fetchone()
        total_value = round(float(value_row[0]), 2)
        retail_row = self._conn.execute(
            "SELECT COALESCE(SUM(quantity_on_hand * retail_price), 0) FROM parts"
        ).fetchone()
        total_retail = round(float(retail_row[0]), 2)

        return {
            "total_parts": total,
            "low_stock_count": low,
            "total_value": total_value,
            "total_retail": total_retail,
        }

    def to_broadcast_dict(self) -> dict:
        """Full state payload for dashboard WS broadcast."""
        return {
            "parts": [p.to_dict() for p in self.list_parts(limit=200)],
            "usage_history": [u.to_dict() for u in self.get_usage_history(limit=50)],
            "top_used": self.get_top_used_parts(limit=10),
            "status": self.get_status(),
            "cost_report": self.cost_report(),
        }

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _check_reorder_alert(self, part: Part):
        """Fire a low-stock notification if part needs reorder."""
        if part.needs_reorder and self._notification_callback:
            body = (
                f"{part.name} (P/N: {part.part_number or 'N/A'}) — "
                f"qty {part.quantity_on_hand}, reorder point {part.reorder_point}"
            )
            try:
                self._notification_callback(
                    "warning",
                    "Low Stock Alert",
                    body,
                    "inventory",
                )
            except Exception as e:
                logger.warning("Failed to send low-stock notification: %s", e)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close the database connection."""
        self._conn.close()
        logger.info("InventoryManager closed")
