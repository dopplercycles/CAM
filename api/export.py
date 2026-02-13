"""
Data Export and Reporting API — comprehensive export for all CAM data.

Provides:
  - CSV/JSON export for customers, vehicles, service records, invoices,
    financials, scout listings, ride logs, and content pipeline data
  - Date range filtering and field selection
  - Full database backup (SQLite dump as ZIP)
  - Monthly business report bundle (financial summary, customer activity,
    content performance, service statistics, ride log for tax prep)
  - Export history tracking
"""

import csv
import io
import json
import logging
import sqlite3
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Query
from fastapi.responses import Response

logger = logging.getLogger("cam.export")


# ── Export configuration ────────────────────────────────────────────

EXPORT_TYPES = {
    "customers": {
        "label": "Customer Data",
        "fields": [
            "customer_id", "name", "phone", "email", "address",
            "preferred_contact_method", "tags", "notes_summary",
            "date_added", "last_contact", "created_at",
        ],
    },
    "vehicles": {
        "label": "Vehicles",
        "fields": [
            "vehicle_id", "year", "make", "model", "vin",
            "owner_id", "owner_name", "color", "mileage", "notes",
            "created_at", "updated_at",
        ],
    },
    "service_records": {
        "label": "Service Records",
        "fields": [
            "record_id", "vehicle_id", "customer_id", "customer_name",
            "vehicle_summary", "date", "service_type", "services_performed",
            "parts_used", "labor_hours", "labor_rate", "parts_total",
            "labor_total", "total_cost", "notes", "recommendations",
            "created_at",
        ],
    },
    "invoices": {
        "label": "Invoices",
        "fields": [
            "invoice_id", "invoice_number", "customer_id", "customer_name",
            "date", "due_date", "line_items", "labor_hours", "labor_rate",
            "parts_total", "labor_total", "subtotal", "tax_rate",
            "tax_amount", "total", "status", "payment_method",
            "paid_date", "notes", "created_at",
        ],
    },
    "financials": {
        "label": "Financial Transactions",
        "fields": [
            "txn_id", "date", "type", "category", "amount",
            "description", "reference_id", "created_at",
        ],
    },
    "scout_listings": {
        "label": "Scout Listings",
        "fields": [
            "entry_id", "title", "price", "url", "source",
            "location", "date_found", "status", "deal_score",
            "score_reason", "make", "model", "year", "snippet",
            "created_at",
        ],
    },
    "ride_logs": {
        "label": "Ride Logs",
        "fields": [
            "ride_id", "date", "start_time", "end_time",
            "start_location", "end_location", "distance", "purpose",
            "weather_conditions", "fuel_used", "odometer_start",
            "odometer_end", "notes", "duration_minutes", "mpg",
            "is_business", "created_at",
        ],
    },
    "content_pipeline": {
        "label": "Content Pipeline",
        "fields": [
            "pipeline_id", "title", "topic", "stage", "status",
            "created_at", "updated_at", "completed_at",
        ],
    },
}

EXPORT_FORMATS = ("csv", "json")

# Database files included in full backup
DB_FILES = [
    "data/crm.db",
    "data/service_records.db",
    "data/invoices.db",
    "data/finances.db",
    "data/scout.db",
    "data/ride_log.db",
    "data/content_pipeline.db",
    "data/appointments.db",
    "data/inventory.db",
    "data/feedback.db",
    "data/warranty.db",
    "data/maintenance_scheduler.db",
    "data/analytics.db",
    "data/content_calendar.db",
    "data/market_monitor.db",
    "data/photo_docs.db",
    "data/email_communications.db",
    "data/training.db",
    "data/plugins.db",
]


# ── Dataclass ───────────────────────────────────────────────────────

@dataclass
class ExportRecord:
    """Tracks a completed export for history."""

    export_id: str
    data_type: str
    format: str
    row_count: int
    file_size: int
    date_from: str
    date_to: str
    created_at: str

    @property
    def short_id(self) -> str:
        return self.export_id[:8]

    def to_dict(self) -> dict:
        return {
            "export_id": self.export_id,
            "short_id": self.short_id,
            "data_type": self.data_type,
            "data_type_label": EXPORT_TYPES.get(
                self.data_type, {}
            ).get("label", self.data_type),
            "format": self.format,
            "row_count": self.row_count,
            "file_size": self.file_size,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_row(row) -> "ExportRecord":
        return ExportRecord(
            export_id=row["export_id"],
            data_type=row["data_type"],
            format=row["format"],
            row_count=row["row_count"],
            file_size=row["file_size"],
            date_from=row["date_from"],
            date_to=row["date_to"],
            created_at=row["created_at"],
        )


# ── ExportManager ───────────────────────────────────────────────────

class ExportManager:
    """Manages data export for all CAM modules."""

    def __init__(
        self,
        export_dir: str = "data/exports",
        db_path: str = "data/exports.db",
        on_change=None,
    ):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._on_change = on_change
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._managers: dict[str, Any] = {}
        logger.info(
            "ExportManager initialized (dir=%s, db=%s)",
            self.export_dir, db_path,
        )

    # ── Schema ──────────────────────────────────────────────────────

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS export_history (
                export_id   TEXT PRIMARY KEY,
                data_type   TEXT NOT NULL,
                format      TEXT NOT NULL,
                row_count   INTEGER DEFAULT 0,
                file_size   INTEGER DEFAULT 0,
                date_from   TEXT DEFAULT '',
                date_to     TEXT DEFAULT '',
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_eh_created
                ON export_history(created_at);
        """)
        self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _fire_change(self):
        if self._on_change:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._on_change())
            except RuntimeError:
                pass

    # ── Manager references ──────────────────────────────────────────

    def set_managers(self, **managers):
        """Store references to data managers for export access.

        Expected keys: crm, service_records, invoices, finances, scout,
        ride_log, content_pipeline.
        """
        self._managers = managers

    # ── Data extraction ─────────────────────────────────────────────

    def _get_data(
        self,
        data_type: str,
        date_from: str = "",
        date_to: str = "",
        fields: list[str] | None = None,
    ) -> list[dict]:
        """Extract rows from the appropriate manager."""
        extractors = {
            "customers": self._extract_customers,
            "vehicles": self._extract_vehicles,
            "service_records": self._extract_service_records,
            "invoices": self._extract_invoices,
            "financials": self._extract_financials,
            "scout_listings": self._extract_scout_listings,
            "ride_logs": self._extract_ride_logs,
            "content_pipeline": self._extract_content_pipeline,
        }
        extractor = extractors.get(data_type)
        if not extractor:
            return []
        rows = extractor(date_from, date_to)
        if fields:
            rows = [{k: r.get(k, "") for k in fields} for r in rows]
        return rows

    @staticmethod
    def _filter_by_date(
        rows: list[dict], date_field: str,
        date_from: str, date_to: str,
    ) -> list[dict]:
        """Filter rows by date range on the given field."""
        filtered = []
        for r in rows:
            val = str(r.get(date_field, ""))[:10]
            if date_from and val < date_from:
                continue
            if date_to and val > date_to:
                continue
            filtered.append(r)
        return filtered

    def _extract_customers(self, date_from: str, date_to: str) -> list[dict]:
        crm = self._managers.get("crm")
        if not crm:
            return []
        customers = crm.list_customers(limit=10000)
        rows = [c.to_dict() for c in customers]
        if date_from or date_to:
            rows = self._filter_by_date(rows, "created_at", date_from, date_to)
        return rows

    def _extract_vehicles(self, date_from: str, date_to: str) -> list[dict]:
        srs = self._managers.get("service_records")
        if not srs:
            return []
        vehicles = srs.list_vehicles(limit=10000)
        rows = [v.to_dict() for v in vehicles]
        if date_from or date_to:
            rows = self._filter_by_date(rows, "created_at", date_from, date_to)
        return rows

    def _extract_service_records(self, date_from: str, date_to: str) -> list[dict]:
        srs = self._managers.get("service_records")
        if not srs:
            return []
        records = srs.list_records(limit=10000)
        rows = [r.to_dict() for r in records]
        if date_from or date_to:
            rows = self._filter_by_date(rows, "date", date_from, date_to)
        # Flatten list fields for CSV compatibility
        for r in rows:
            if isinstance(r.get("services_performed"), list):
                r["services_performed"] = "; ".join(r["services_performed"])
            if isinstance(r.get("parts_used"), list):
                r["parts_used"] = "; ".join(
                    f"{p.get('name', '')} x{p.get('qty', 1)} ${p.get('cost', 0)}"
                    for p in r["parts_used"]
                    if isinstance(p, dict)
                )
        return rows

    def _extract_invoices(self, date_from: str, date_to: str) -> list[dict]:
        inv = self._managers.get("invoices")
        if not inv:
            return []
        invoices = inv.list_invoices(limit=10000)
        rows = [i.to_dict() for i in invoices]
        if date_from or date_to:
            rows = self._filter_by_date(rows, "date", date_from, date_to)
        for r in rows:
            if isinstance(r.get("line_items"), list):
                r["line_items"] = "; ".join(
                    f"{li.get('description', '')} ${li.get('amount', 0)}"
                    for li in r["line_items"]
                    if isinstance(li, dict)
                )
        return rows

    def _extract_financials(self, date_from: str, date_to: str) -> list[dict]:
        fin = self._managers.get("finances")
        if not fin:
            return []
        kwargs: dict[str, Any] = {"limit": 10000}
        if date_from:
            kwargs["start_date"] = date_from
        if date_to:
            kwargs["end_date"] = date_to
        txns = fin.list_transactions(**kwargs)
        return [t.to_dict() for t in txns]

    def _extract_scout_listings(self, date_from: str, date_to: str) -> list[dict]:
        scout = self._managers.get("scout")
        if not scout:
            return []
        listings = scout.list_all(limit=10000)
        rows = [entry.to_dict() for entry in listings]
        if date_from or date_to:
            rows = self._filter_by_date(rows, "date_found", date_from, date_to)
        return rows

    def _extract_ride_logs(self, date_from: str, date_to: str) -> list[dict]:
        rl = self._managers.get("ride_log")
        if not rl:
            return []
        rides = rl.list_rides(limit=10000)
        rows = [r.to_dict() for r in rides]
        if date_from or date_to:
            rows = self._filter_by_date(rows, "date", date_from, date_to)
        return rows

    def _extract_content_pipeline(self, date_from: str, date_to: str) -> list[dict]:
        cp = self._managers.get("content_pipeline")
        if not cp:
            return []
        pipelines = cp.list_all()
        rows = pipelines if isinstance(pipelines, list) else []
        rows = [
            r if isinstance(r, dict) else r.to_dict()
            for r in rows
            if isinstance(r, dict) or hasattr(r, "to_dict")
        ]
        if date_from or date_to:
            rows = self._filter_by_date(rows, "created_at", date_from, date_to)
        return rows

    # ── Format conversion ───────────────────────────────────────────

    @staticmethod
    def _to_csv_bytes(rows: list[dict]) -> bytes:
        """Convert rows to CSV bytes."""
        if not rows:
            return b""
        output = io.StringIO()
        writer = csv.DictWriter(
            output, fieldnames=rows[0].keys(), extrasaction="ignore",
        )
        writer.writeheader()
        for r in rows:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (list, dict)):
                    clean[k] = json.dumps(v, default=str)
                else:
                    clean[k] = v
            writer.writerow(clean)
        return output.getvalue().encode("utf-8")

    @staticmethod
    def _to_json_bytes(rows: list[dict]) -> bytes:
        """Convert rows to JSON bytes."""
        return json.dumps(rows, indent=2, default=str).encode("utf-8")

    # ── Export execution ────────────────────────────────────────────

    def export_data(
        self,
        data_type: str,
        fmt: str = "csv",
        date_from: str = "",
        date_to: str = "",
        fields: list[str] | None = None,
    ) -> tuple[bytes, str, int]:
        """Export data in the requested format.

        Returns (bytes, content_type, row_count).
        """
        rows = self._get_data(data_type, date_from, date_to, fields)

        if fmt == "json":
            data = self._to_json_bytes(rows)
            content_type = "application/json"
        else:
            data = self._to_csv_bytes(rows)
            content_type = "text/csv"

        self._record_export(
            data_type, fmt, len(rows), len(data), date_from, date_to,
        )
        return data, content_type, len(rows)

    def _record_export(
        self, data_type: str, fmt: str, row_count: int,
        file_size: int, date_from: str, date_to: str,
    ):
        export_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO export_history VALUES (?,?,?,?,?,?,?,?)",
            (export_id, data_type, fmt, row_count, file_size,
             date_from, date_to, self._now()),
        )
        self._conn.commit()
        self._fire_change()

    # ── Database backup ─────────────────────────────────────────────

    def database_backup(self) -> tuple[bytes, str]:
        """Create a ZIP containing copies of all SQLite databases.

        Returns (zip_bytes, filename).
        """
        buf = io.BytesIO()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"cam_backup_{timestamp}.zip"

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for db_file in DB_FILES:
                db_path = Path(db_file)
                if db_path.exists():
                    zf.write(db_path, db_path.name)
            # Also include settings
            settings = Path("config/settings.toml")
            if settings.exists():
                zf.write(settings, "config/settings.toml")

        data = buf.getvalue()
        self._record_export("database_backup", "zip", 0, len(data), "", "")
        return data, filename

    # ── Monthly business report ─────────────────────────────────────

    def monthly_report(self, year: int, month: int) -> tuple[bytes, str]:
        """Generate end-of-month business package as a ZIP bundle.

        Contents:
          - summary.json         — overview statistics
          - financial_transactions.csv
          - customer_activity.csv
          - service_records.csv
          - invoices.csv
          - content_performance.csv
          - ride_log.csv         — for tax prep
        """
        month_str = f"{year}-{month:02d}"
        date_from = f"{year}-{month:02d}-01"
        if month == 12:
            date_to = f"{year + 1}-01-01"
        else:
            date_to = f"{year}-{month + 1:02d}-01"

        buf = io.BytesIO()
        filename = f"cam_monthly_report_{month_str}.zip"

        summary: dict[str, Any] = {
            "report": "CAM Monthly Business Report",
            "period": month_str,
            "generated_at": self._now(),
            "sections": {},
        }

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Financial transactions
            fin_rows = self._get_data("financials", date_from, date_to)
            if fin_rows:
                zf.writestr(
                    "financial_transactions.csv",
                    self._to_csv_bytes(fin_rows),
                )
            fin = self._managers.get("finances")
            if fin:
                fin_summary = fin.monthly_summary(months=1)
                top_expenses = fin.top_expense_categories(months=1)
                revenue_by_type = fin.revenue_by_service_type(months=1)
                summary["sections"]["financial"] = {
                    "monthly": fin_summary,
                    "top_expenses": top_expenses,
                    "revenue_by_type": revenue_by_type,
                    "transaction_count": len(fin_rows),
                }

            # Customer activity
            cust_rows = self._get_data("customers", date_from, date_to)
            if cust_rows:
                zf.writestr(
                    "customer_activity.csv",
                    self._to_csv_bytes(cust_rows),
                )
            crm = self._managers.get("crm")
            summary["sections"]["customers"] = {
                "new_customers": len(cust_rows),
                "total_customers": (
                    crm.get_status().get("total_customers", 0) if crm else 0
                ),
            }

            # Service statistics
            svc_rows = self._get_data("service_records", date_from, date_to)
            if svc_rows:
                zf.writestr(
                    "service_records.csv",
                    self._to_csv_bytes(svc_rows),
                )
            total_revenue = sum(
                float(r.get("total_cost", 0) or 0) for r in svc_rows
            )
            total_labor = sum(
                float(r.get("labor_hours", 0) or 0) for r in svc_rows
            )
            summary["sections"]["services"] = {
                "total_services": len(svc_rows),
                "total_revenue": round(total_revenue, 2),
                "total_labor_hours": round(total_labor, 1),
                "avg_ticket": (
                    round(total_revenue / len(svc_rows), 2) if svc_rows else 0
                ),
            }

            # Invoice summary
            inv_rows = self._get_data("invoices", date_from, date_to)
            if inv_rows:
                zf.writestr("invoices.csv", self._to_csv_bytes(inv_rows))
            paid = [r for r in inv_rows if r.get("status") == "paid"]
            outstanding = [
                r for r in inv_rows
                if r.get("status") in ("sent", "overdue")
            ]
            summary["sections"]["invoices"] = {
                "total_invoices": len(inv_rows),
                "paid": len(paid),
                "outstanding": len(outstanding),
                "paid_revenue": round(
                    sum(float(r.get("total", 0) or 0) for r in paid), 2,
                ),
                "outstanding_amount": round(
                    sum(float(r.get("total", 0) or 0) for r in outstanding), 2,
                ),
            }

            # Content performance
            content_rows = self._get_data(
                "content_pipeline", date_from, date_to,
            )
            if content_rows:
                zf.writestr(
                    "content_performance.csv",
                    self._to_csv_bytes(content_rows),
                )
            completed = [
                r for r in content_rows if r.get("status") == "completed"
            ]
            summary["sections"]["content"] = {
                "total_pieces": len(content_rows),
                "completed": len(completed),
                "in_progress": len(content_rows) - len(completed),
            }

            # Ride log for tax prep
            ride_rows = self._get_data("ride_logs", date_from, date_to)
            if ride_rows:
                zf.writestr("ride_log.csv", self._to_csv_bytes(ride_rows))
            biz_miles = sum(
                float(r.get("distance", 0) or 0)
                for r in ride_rows if r.get("is_business")
            )
            personal_miles = sum(
                float(r.get("distance", 0) or 0)
                for r in ride_rows if not r.get("is_business")
            )
            summary["sections"]["mileage"] = {
                "total_rides": len(ride_rows),
                "business_miles": round(biz_miles, 1),
                "personal_miles": round(personal_miles, 1),
                "total_miles": round(biz_miles + personal_miles, 1),
                "estimated_deduction": round(biz_miles * 0.70, 2),
            }

            # Write summary JSON
            zf.writestr(
                "summary.json",
                json.dumps(summary, indent=2, default=str).encode("utf-8"),
            )

        data = buf.getvalue()
        self._record_export(
            "monthly_report", "zip", 0, len(data), date_from, date_to,
        )
        return data, filename

    # ── Export history ──────────────────────────────────────────────

    def list_history(self, limit: int = 50) -> list[ExportRecord]:
        """Return recent export history."""
        rows = self._conn.execute(
            "SELECT * FROM export_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [ExportRecord.from_row(r) for r in rows]

    def clear_history(self) -> int:
        """Clear all export history records."""
        cur = self._conn.execute("DELETE FROM export_history")
        self._conn.commit()
        self._fire_change()
        return cur.rowcount

    # ── Available fields query ──────────────────────────────────────

    @staticmethod
    def get_available_fields(data_type: str) -> list[str]:
        """Return the list of available fields for a data type."""
        info = EXPORT_TYPES.get(data_type, {})
        return info.get("fields", [])

    @staticmethod
    def get_export_types() -> dict:
        """Return all available export types with labels and fields."""
        return {
            k: {"label": v["label"], "fields": v["fields"]}
            for k, v in EXPORT_TYPES.items()
        }

    # ── Status / broadcast ──────────────────────────────────────────

    def get_status(self) -> dict:
        """Dashboard status summary."""
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM export_history",
        ).fetchone()
        total = row["cnt"] if row else 0

        recent = self._conn.execute(
            "SELECT * FROM export_history ORDER BY created_at DESC LIMIT 1",
        ).fetchone()

        return {
            "total_exports": total,
            "last_export": (
                ExportRecord.from_row(recent).to_dict() if recent else None
            ),
            "available_types": list(EXPORT_TYPES.keys()),
            "available_formats": list(EXPORT_FORMATS),
        }

    def to_broadcast_dict(self) -> dict:
        """Full state for WS broadcast."""
        return {
            "history": [r.to_dict() for r in self.list_history(limit=20)],
            "status": self.get_status(),
            "export_types": self.get_export_types(),
        }

    # ── HTTP route registration ─────────────────────────────────────

    def register_routes(self, app) -> None:
        """Register FastAPI HTTP download endpoints on the app."""
        mgr = self

        @app.get("/api/export/{data_type}")
        async def export_data_endpoint(
            data_type: str,
            format: str = Query("csv", pattern="^(csv|json)$"),
            date_from: str = Query("", description="Start date YYYY-MM-DD"),
            date_to: str = Query("", description="End date YYYY-MM-DD"),
            fields: str = Query("", description="Comma-separated field names"),
        ):
            """Download exported data as CSV or JSON."""
            if data_type not in EXPORT_TYPES:
                return Response(
                    content=json.dumps({
                        "error": f"Unknown export type: {data_type}",
                    }),
                    status_code=400,
                    media_type="application/json",
                )
            field_list = (
                [f.strip() for f in fields.split(",") if f.strip()]
                if fields else None
            )
            data, content_type, _count = mgr.export_data(
                data_type, format, date_from, date_to, field_list,
            )
            ext = "json" if format == "json" else "csv"
            timestamp = datetime.utcnow().strftime("%Y%m%d")
            fname = f"cam_{data_type}_{timestamp}.{ext}"
            return Response(
                content=data,
                media_type=content_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{fname}"',
                },
            )

        @app.get("/api/export-backup/full")
        async def full_backup_endpoint():
            """Download full database backup as ZIP."""
            data, fname = mgr.database_backup()
            return Response(
                content=data,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{fname}"',
                },
            )

        @app.get("/api/export-report/monthly")
        async def monthly_report_endpoint(
            year: int = Query(..., description="Report year"),
            month: int = Query(
                ..., ge=1, le=12, description="Report month 1-12",
            ),
        ):
            """Download monthly business report bundle as ZIP."""
            data, fname = mgr.monthly_report(year, month)
            return Response(
                content=data,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{fname}"',
                },
            )

        logger.info(
            "Export routes registered: /api/export/{type}, "
            "/api/export-backup/full, /api/export-report/monthly",
        )

    # ── Cleanup ─────────────────────────────────────────────────────

    def close(self):
        self._conn.close()
        logger.info("ExportManager closed")
