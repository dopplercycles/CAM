"""Scheduled reporting engine for Doppler Cycles.

Generates daily, weekly, and monthly operational reports pulling data from
all existing data stores: analytics, business, scout, service records,
market analyzer, content calendar, event logger, and agent registry.

Reports are viewable as inline HTML in the dashboard and downloadable as PDF.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class Report:
    """A generated operational report."""

    report_id: str
    report_type: str          # "daily", "weekly", "monthly"
    title: str
    period_start: str         # ISO datetime
    period_end: str           # ISO datetime
    html_content: str = ""
    pdf_path: str = ""
    summary: str = ""
    data_json: str = "{}"     # raw data snapshot (JSON string)
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 characters of the report UUID for display."""
        return self.report_id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for list views — omits bulky html_content."""
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "title": self.title,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "pdf_path": self.pdf_path,
            "summary": self.summary,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Serialize including html_content for detail views."""
        d = self.to_dict()
        d["html_content"] = self.html_content
        return d

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Report:
        """Construct a Report from a SQLite row."""
        return cls(
            report_id=row["report_id"],
            report_type=row["report_type"],
            title=row["title"],
            period_start=row["period_start"],
            period_end=row["period_end"],
            html_content=row["html_content"] or "",
            pdf_path=row["pdf_path"] or "",
            summary=row["summary"] or "",
            data_json=row["data_json"] or "{}",
            created_at=row["created_at"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


class ReportEngine:
    """Generates and manages operational reports for Doppler Cycles.

    Pulls data from all available stores and renders HTML + PDF reports.
    Any data source can be None — collectors degrade gracefully.
    """

    def __init__(
        self,
        db_path: str = "data/reports.db",
        pdf_dir: str = "data/reports",
        analytics=None,
        business_store=None,
        scout_store=None,
        event_logger=None,
        content_calendar=None,
        service_store=None,
        market_analyzer=None,
        registry=None,
        on_change: Callable[[], Awaitable[None]] | None = None,
    ):
        self._db_path = db_path
        self._pdf_dir = pdf_dir
        self._analytics = analytics
        self._business_store = business_store
        self._scout_store = scout_store
        self._event_logger = event_logger
        self._content_calendar = content_calendar
        self._service_store = service_store
        self._market_analyzer = market_analyzer
        self._registry = registry
        self._on_change = on_change

        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

        logger.info("ReportEngine initialized (db=%s, pdf_dir=%s)", db_path, pdf_dir)

    def _create_tables(self):
        """Create the reports table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT UNIQUE NOT NULL,
                report_type TEXT NOT NULL,
                title TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                html_content TEXT,
                pdf_path TEXT,
                summary TEXT,
                data_json TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at)
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Data collectors — each wrapped in try/except, returns {} on failure
    # ------------------------------------------------------------------

    def _collect_analytics_data(self) -> dict:
        """Collect task and model cost analytics."""
        try:
            if self._analytics is None:
                return {}
            return self._analytics.get_summary()
        except Exception as e:
            logger.warning("Failed to collect analytics data: %s", e)
            return {}

    def _collect_business_data(self) -> dict:
        """Collect business status, appointments, and invoices."""
        try:
            if self._business_store is None:
                return {}
            status = self._business_store.get_status()
            appointments = [a.to_dict() for a in self._business_store.list_appointments(limit=20)]
            invoices = [i.to_dict() for i in self._business_store.list_invoices(limit=20)]
            return {"status": status, "appointments": appointments, "invoices": invoices}
        except Exception as e:
            logger.warning("Failed to collect business data: %s", e)
            return {}

    def _collect_scout_data(self) -> dict:
        """Collect motorcycle scout listings and status."""
        try:
            if self._scout_store is None:
                return {}
            status = self._scout_store.get_status()
            hot_deals = [l.to_dict() for l in self._scout_store.list_all(min_score=7, limit=10)]
            return {"status": status, "hot_deals": hot_deals}
        except Exception as e:
            logger.warning("Failed to collect scout data: %s", e)
            return {}

    def _collect_events_data(self, count: int = 50) -> dict:
        """Collect recent activity log events."""
        try:
            if self._event_logger is None:
                return {}
            events = self._event_logger.get_recent(count)
            return {"events": events, "count": len(events)}
        except Exception as e:
            logger.warning("Failed to collect events data: %s", e)
            return {}

    def _collect_content_data(self) -> dict:
        """Collect content calendar status and entries."""
        try:
            if self._content_calendar is None:
                return {}
            status = self._content_calendar.get_status()
            entries = [e.to_dict() for e in self._content_calendar.list_all()]
            return {"status": status, "entries": entries}
        except Exception as e:
            logger.warning("Failed to collect content data: %s", e)
            return {}

    def _collect_service_data(self) -> dict:
        """Collect service record status."""
        try:
            if self._service_store is None:
                return {}
            return self._service_store.get_status()
        except Exception as e:
            logger.warning("Failed to collect service data: %s", e)
            return {}

    def _collect_market_data(self) -> dict:
        """Collect market analyzer summary."""
        try:
            if self._market_analyzer is None:
                return {}
            return self._market_analyzer.get_summary()
        except Exception as e:
            logger.warning("Failed to collect market data: %s", e)
            return {}

    def _collect_agent_data(self) -> dict:
        """Collect agent registry status and agent list."""
        try:
            if self._registry is None:
                return {}
            status = self._registry.get_status()
            agents = [a.to_dict() for a in self._registry.list_all()]
            return {"status": status, "agents": agents}
        except Exception as e:
            logger.warning("Failed to collect agent data: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, report_type: str) -> Report:
        """Generate a report of the specified type.

        Args:
            report_type: One of "daily", "weekly", "monthly".

        Returns:
            The generated Report object.
        """
        now = datetime.now(timezone.utc)
        if report_type == "daily":
            return self._generate_daily(now)
        elif report_type == "weekly":
            return self._generate_weekly(now)
        elif report_type == "monthly":
            return self._generate_monthly(now)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def _generate_daily(self, now: datetime) -> Report:
        """Generate a daily summary report covering the last 24 hours."""
        period_end = now
        period_start = now - timedelta(hours=24)
        title = f"Daily Summary — {now.strftime('%b %d, %Y')}"
        period_label = now.strftime("%B %d, %Y")

        analytics = self._collect_analytics_data()
        events = self._collect_events_data(count=30)
        agents = self._collect_agent_data()
        scout = self._collect_scout_data()

        sections = []
        data = {}

        # Task overview
        if analytics:
            data["analytics"] = analytics
            sections.append(("Task Overview", self._html_kv_table([
                ("Total Tasks", analytics.get("total_tasks", 0)),
                ("Completed", analytics.get("completed_tasks", 0)),
                ("Failed", analytics.get("failed_tasks", 0)),
                ("Success Rate", f"{analytics.get('success_rate', 0):.0f}%"),
            ])))

        # Model costs
        if analytics and analytics.get("total_model_calls"):
            sections.append(("Model Costs", self._html_kv_table([
                ("Total Calls", analytics.get("total_model_calls", 0)),
                ("Total Tokens", f"{analytics.get('total_model_tokens', 0):,}"),
                ("Total Cost", f"${analytics.get('total_model_cost_usd', 0):.4f}"),
            ])))

        # Agent status
        if agents:
            data["agents"] = agents
            status = agents.get("status", {})
            sections.append(("Agent Status", self._html_kv_table([
                ("Total Agents", status.get("total", 0)),
                ("Online", status.get("online", 0)),
                ("Offline", status.get("offline", 0)),
                ("Busy", status.get("busy", 0)),
            ])))

        # Scout hot deals
        if scout and scout.get("hot_deals"):
            data["scout"] = scout
            rows = []
            for deal in scout["hot_deals"][:5]:
                rows.append([
                    deal.get("title", "Unknown"),
                    f"${deal.get('price', 0):,.0f}" if deal.get("price") else "N/A",
                    str(deal.get("deal_score", 0)),
                ])
            sections.append(("Scout Hot Deals", self._html_table(
                ["Listing", "Price", "Score"], rows
            )))

        # Recent events
        if events and events.get("events"):
            data["events"] = events
            event_list = events["events"][-10:]  # last 10
            rows = [[
                e.get("timestamp", "")[:16],
                e.get("level", ""),
                e.get("message", "")[:80],
            ] for e in event_list]
            sections.append(("Recent Activity", self._html_table(
                ["Time", "Level", "Message"], rows
            )))

        summary = self._build_summary(analytics, agents)
        html = self._build_html(title, period_label, sections)
        report = self._save_report(
            report_type="daily", title=title,
            period_start=period_start, period_end=period_end,
            html_content=html, sections_data=data, summary=summary,
        )
        self._build_pdf(report.report_id, title, period_label, sections, "daily")
        return report

    def _generate_weekly(self, now: datetime) -> Report:
        """Generate a weekly report covering the last 7 days."""
        period_end = now
        period_start = now - timedelta(days=7)
        start_str = period_start.strftime("%b %d")
        end_str = now.strftime("%b %d, %Y")
        title = f"Weekly Report — {start_str} - {end_str}"
        period_label = f"{start_str} – {end_str}"

        analytics = self._collect_analytics_data()
        business = self._collect_business_data()
        content = self._collect_content_data()
        service = self._collect_service_data()
        market = self._collect_market_data()
        scout = self._collect_scout_data()

        sections = []
        data = {}

        # Business overview
        if business:
            data["business"] = business
            bs = business.get("status", {})
            sections.append(("Business Overview", self._html_kv_table([
                ("Customers", bs.get("customer_count", 0)),
                ("Upcoming Appointments", bs.get("upcoming_appointments", 0)),
                ("Total Appointments", bs.get("total_appointments", 0)),
                ("Paid Invoices", bs.get("paid_invoices", 0)),
                ("Total Revenue", f"${bs.get('total_revenue', 0):,.2f}"),
            ])))

        # Appointments
        if business and business.get("appointments"):
            rows = []
            for apt in business["appointments"][:10]:
                rows.append([
                    apt.get("customer_name", "Unknown"),
                    apt.get("date", ""),
                    apt.get("service_type", ""),
                    apt.get("status", ""),
                ])
            sections.append(("Recent Appointments", self._html_table(
                ["Customer", "Date", "Service", "Status"], rows
            )))

        # Service records
        if service:
            data["service"] = service
            sections.append(("Service Records", self._html_kv_table([
                ("Total Vehicles", service.get("total_vehicles", 0)),
                ("Total Records", service.get("total_records", 0)),
                ("This Month", service.get("records_this_month", 0)),
                ("Revenue", f"${service.get('total_revenue', 0):,.2f}"),
            ])))

        # Content pipeline
        if content:
            data["content"] = content
            cs = content.get("status", {})
            by_status = cs.get("by_status", {})
            sections.append(("Content Pipeline", self._html_kv_table([
                ("Total Entries", cs.get("total", 0)),
                ("Draft", by_status.get("draft", 0)),
                ("In Progress", by_status.get("in_progress", 0)),
                ("Published", by_status.get("published", 0)),
            ])))

        # Market trends
        if market:
            data["market"] = market
            sections.append(("Market Analytics", self._html_kv_table([
                ("Tracked Makes", market.get("tracked_makes", 0)),
                ("Total Snapshots", market.get("total_snapshots", 0)),
                ("Reports", market.get("report_count", 0)),
            ])))

        # Scout status
        if scout:
            data["scout"] = scout
            ss = scout.get("status", {})
            sections.append(("Scout Status", self._html_kv_table([
                ("Total Listings", ss.get("total", 0)),
                ("Hot Deals", ss.get("hot_deals", 0)),
            ])))

        # Model costs
        if analytics:
            data["analytics"] = analytics
            sections.append(("Cost Summary", self._html_kv_table([
                ("Model Calls", analytics.get("total_model_calls", 0)),
                ("Total Tokens", f"{analytics.get('total_model_tokens', 0):,}"),
                ("Total Cost", f"${analytics.get('total_model_cost_usd', 0):.4f}"),
                ("Task Success Rate", f"{analytics.get('success_rate', 0):.0f}%"),
            ])))

        summary = self._build_summary(analytics, business=business, service=service)
        html = self._build_html(title, period_label, sections)
        report = self._save_report(
            report_type="weekly", title=title,
            period_start=period_start, period_end=period_end,
            html_content=html, sections_data=data, summary=summary,
        )
        self._build_pdf(report.report_id, title, period_label, sections, "weekly")
        return report

    def _generate_monthly(self, now: datetime) -> Report:
        """Generate a monthly report with comprehensive metrics."""
        period_end = now
        period_start = now - timedelta(days=30)
        title = f"Monthly Report — {now.strftime('%B %Y')}"
        period_label = now.strftime("%B %Y")

        analytics = self._collect_analytics_data()
        business = self._collect_business_data()
        content = self._collect_content_data()
        service = self._collect_service_data()
        market = self._collect_market_data()
        scout = self._collect_scout_data()
        agents = self._collect_agent_data()
        events = self._collect_events_data(count=100)

        sections = []
        data = {}

        # Business summary
        if business:
            data["business"] = business
            bs = business.get("status", {})
            sections.append(("Business Summary", self._html_kv_table([
                ("Customers", bs.get("customer_count", 0)),
                ("Total Appointments", bs.get("total_appointments", 0)),
                ("Total Invoices", bs.get("total_invoices", 0)),
                ("Draft Invoices", bs.get("draft_invoices", 0)),
                ("Paid Invoices", bs.get("paid_invoices", 0)),
                ("Total Revenue", f"${bs.get('total_revenue', 0):,.2f}"),
                ("Inventory Items", bs.get("inventory_count", 0)),
                ("Low Stock Items", bs.get("low_stock_count", 0)),
            ])))

        # Invoice breakdown
        if business and business.get("invoices"):
            rows = []
            for inv in business["invoices"][:15]:
                rows.append([
                    inv.get("invoice_number", ""),
                    inv.get("customer_name", "Unknown"),
                    f"${inv.get('total', 0):,.2f}" if inv.get("total") else "$0.00",
                    inv.get("status", ""),
                    inv.get("date", ""),
                ])
            sections.append(("Invoices", self._html_table(
                ["#", "Customer", "Total", "Status", "Date"], rows
            )))

        # Service records
        if service:
            data["service"] = service
            sections.append(("Service Records", self._html_kv_table([
                ("Total Vehicles", service.get("total_vehicles", 0)),
                ("Total Records", service.get("total_records", 0)),
                ("Records This Month", service.get("records_this_month", 0)),
                ("Service Revenue", f"${service.get('total_revenue', 0):,.2f}"),
            ])))

        # Content pipeline
        if content:
            data["content"] = content
            cs = content.get("status", {})
            by_status = cs.get("by_status", {})
            sections.append(("Content Pipeline", self._html_kv_table([
                ("Total Entries", cs.get("total", 0)),
                ("Draft", by_status.get("draft", 0)),
                ("In Progress", by_status.get("in_progress", 0)),
                ("Review", by_status.get("review", 0)),
                ("Scheduled", by_status.get("scheduled", 0)),
                ("Published", by_status.get("published", 0)),
            ])))

        # Market analytics
        if market:
            data["market"] = market
            sections.append(("Market Analytics", self._html_kv_table([
                ("Tracked Makes", market.get("tracked_makes", 0)),
                ("Total Snapshots", market.get("total_snapshots", 0)),
                ("Reports Generated", market.get("report_count", 0)),
                ("Latest Data", market.get("latest_date", "N/A")),
            ])))
            # Undervalued listings
            if market.get("undervalued"):
                rows = []
                for item in market["undervalued"][:5]:
                    rows.append([
                        str(item.get("year", "")),
                        item.get("make", ""),
                        item.get("model", ""),
                        f"${item.get('avg_price', 0):,.0f}" if item.get("avg_price") else "N/A",
                    ])
                sections.append(("Undervalued Models", self._html_table(
                    ["Year", "Make", "Model", "Avg Price"], rows
                )))

        # Scout
        if scout:
            data["scout"] = scout
            ss = scout.get("status", {})
            sections.append(("Motorcycle Scout", self._html_kv_table([
                ("Total Listings", ss.get("total", 0)),
                ("Hot Deals", ss.get("hot_deals", 0)),
            ])))

        # Agent status
        if agents:
            data["agents"] = agents
            status = agents.get("status", {})
            sections.append(("Agent Swarm", self._html_kv_table([
                ("Total Agents", status.get("total", 0)),
                ("Online", status.get("online", 0)),
                ("Offline", status.get("offline", 0)),
            ])))

        # Task & cost analytics
        if analytics:
            data["analytics"] = analytics
            sections.append(("Task & Cost Analytics", self._html_kv_table([
                ("Total Tasks", analytics.get("total_tasks", 0)),
                ("Completed", analytics.get("completed_tasks", 0)),
                ("Failed", analytics.get("failed_tasks", 0)),
                ("Success Rate", f"{analytics.get('success_rate', 0):.0f}%"),
                ("Model Calls", analytics.get("total_model_calls", 0)),
                ("Total Tokens", f"{analytics.get('total_model_tokens', 0):,}"),
                ("Total Model Cost", f"${analytics.get('total_model_cost_usd', 0):.4f}"),
            ])))
            # Cost by backend
            cost_by_backend = analytics.get("cost_by_backend", {})
            if cost_by_backend:
                rows = []
                for backend, info in cost_by_backend.items():
                    rows.append([
                        backend,
                        str(info.get("calls", 0)),
                        f"{info.get('tokens', 0):,}",
                        f"${info.get('cost_usd', 0):.4f}",
                    ])
                sections.append(("Cost by Backend", self._html_table(
                    ["Backend", "Calls", "Tokens", "Cost"], rows
                )))

        summary = self._build_summary(analytics, business=business, service=service)
        html = self._build_html(title, period_label, sections)
        report = self._save_report(
            report_type="monthly", title=title,
            period_start=period_start, period_end=period_end,
            html_content=html, sections_data=data, summary=summary,
        )
        self._build_pdf(report.report_id, title, period_label, sections, "monthly")
        return report

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _build_summary(self, analytics: dict = None, agents: dict = None,
                       business: dict = None, service: dict = None) -> str:
        """Build a one-line summary string for the report."""
        parts = []
        if analytics:
            parts.append(f"{analytics.get('completed_tasks', 0)} tasks completed")
            cost = analytics.get("total_model_cost_usd", 0)
            if cost > 0:
                parts.append(f"${cost:.4f} model cost")
        if business:
            bs = business.get("status", {})
            rev = bs.get("total_revenue", 0)
            if rev > 0:
                parts.append(f"${rev:,.2f} revenue")
        if service:
            recs = service.get("records_this_month", 0)
            if recs > 0:
                parts.append(f"{recs} service records this month")
        if agents:
            status = agents.get("status", {})
            online = status.get("online", 0)
            parts.append(f"{online} agents online")
        return " | ".join(parts) if parts else "No data available"

    def _html_kv_table(self, rows: list[tuple[str, Any]]) -> str:
        """Build a key-value table as HTML."""
        html = '<table style="width:100%;border-collapse:collapse;margin:0.5rem 0;">'
        for label, value in rows:
            html += (
                f'<tr>'
                f'<td style="padding:0.3rem 0.6rem;color:var(--text-secondary);'
                f'font-size:0.85rem;white-space:nowrap;">{_esc(label)}</td>'
                f'<td style="padding:0.3rem 0.6rem;color:var(--text-primary);'
                f'font-size:0.85rem;font-weight:500;">{_esc(str(value))}</td>'
                f'</tr>'
            )
        html += '</table>'
        return html

    def _html_table(self, headers: list[str], rows: list[list[str]]) -> str:
        """Build a data table with headers as HTML."""
        html = '<table style="width:100%;border-collapse:collapse;margin:0.5rem 0;">'
        html += '<thead><tr>'
        for h in headers:
            html += (
                f'<th style="padding:0.35rem 0.6rem;text-align:left;'
                f'font-size:0.8rem;color:var(--text-secondary);'
                f'border-bottom:1px solid var(--border-primary);">{_esc(h)}</th>'
            )
        html += '</tr></thead><tbody>'
        for row in rows:
            html += '<tr>'
            for cell in row:
                html += (
                    f'<td style="padding:0.3rem 0.6rem;font-size:0.82rem;'
                    f'color:var(--text-primary);border-bottom:1px solid '
                    f'var(--border-primary);">{_esc(str(cell))}</td>'
                )
            html += '</tr>'
        html += '</tbody></table>'
        return html

    def _build_html(self, title: str, period_label: str,
                    sections: list[tuple[str, str]]) -> str:
        """Wrap report sections in inline-styled HTML for the dashboard panel.

        Uses CSS variables so it inherits the dashboard theme.
        No <html>/<body> — this renders inline in the panel.
        """
        html = f"""
<div style="font-family:inherit;color:var(--text-primary);">
  <div style="text-align:center;margin-bottom:1rem;">
    <div style="font-size:1.1rem;font-weight:700;">{_esc(title)}</div>
    <div style="font-size:0.85rem;color:var(--text-secondary);">
      Period: {_esc(period_label)}
    </div>
    <div style="font-size:0.75rem;color:var(--text-tertiary);margin-top:0.25rem;">
      Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
    </div>
  </div>
"""
        for heading, body in sections:
            html += f"""
  <div style="margin-bottom:1rem;">
    <div style="font-size:0.9rem;font-weight:600;color:var(--text-primary);
         padding:0.4rem 0.6rem;background:var(--bg-tertiary);
         border-radius:4px;margin-bottom:0.4rem;">
      {_esc(heading)}
    </div>
    {body}
  </div>
"""
        html += '</div>'
        return html

    # ------------------------------------------------------------------
    # PDF builder — follows service_records.py fpdf2 pattern
    # ------------------------------------------------------------------

    def _build_pdf(self, report_id: str, title: str, period_label: str,
                   sections: list[tuple[str, str]], report_type: str) -> str | None:
        """Generate a PDF version of the report.

        Args:
            report_id: UUID of the report.
            title: Report title.
            period_label: Human-readable period string.
            sections: List of (heading, html) tuples.
            report_type: "daily", "weekly", or "monthly".

        Returns:
            Path to the generated PDF, or None on error.
        """
        try:
            from fpdf import FPDF
        except ImportError:
            logger.warning("fpdf2 not installed -- skipping PDF generation")
            return None

        try:
            pdf_path = os.path.join(self._pdf_dir, f"{report_id}.pdf")
            # Sanitize text for Helvetica (ASCII-only font)
            safe_title = _pdf_safe(title)
            safe_period = _pdf_safe(period_label)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            # --- Header ---
            pdf.set_font("Helvetica", "B", 22)
            pdf.cell(0, 10, "DOPPLER CYCLES", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 6, "Mobile Motorcycle Diagnostics", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.cell(0, 6, "Portland Metro Area  |  Gresham, Oregon",
                     new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(4)

            # --- Title ---
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 8, safe_title.upper(), new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"Period: {safe_period}", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(6)

            # --- Sections ---
            for heading, html_body in sections:
                # Section header with gray fill
                pdf.set_fill_color(240, 240, 240)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 7, f"  {_pdf_safe(heading)}", new_x="LMARGIN", new_y="NEXT", fill=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.ln(2)

                # Parse simple data from HTML tables
                rows = _extract_table_rows(html_body)
                for row in rows:
                    if len(row) == 2:
                        # Key-value pair
                        pdf.cell(60, 5, f"  {_pdf_safe(row[0])}", new_x="RIGHT")
                        pdf.cell(0, 5, _pdf_safe(row[1]), new_x="LMARGIN", new_y="NEXT")
                    else:
                        # Data row - fit columns evenly
                        col_width = 190 // max(len(row), 1)
                        for cell_val in row:
                            text = _pdf_safe(cell_val[:40])
                            pdf.cell(col_width, 5, f"  {text}", new_x="RIGHT")
                        pdf.ln()

                pdf.ln(4)

            # --- Footer ---
            pdf.ln(4)
            pdf.set_font("Helvetica", "I", 8)
            generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            pdf.cell(0, 5, f"Generated: {generated_at} | Report ID: {report_id[:8]}",
                     new_x="LMARGIN", new_y="NEXT", align="C")

            pdf.output(pdf_path)

            # Update the report's pdf_path in the database
            self._conn.execute(
                "UPDATE reports SET pdf_path = ? WHERE report_id = ?",
                (pdf_path, report_id),
            )
            self._conn.commit()

            logger.info("PDF generated: %s", pdf_path)
            return pdf_path

        except Exception as e:
            logger.error("Failed to generate PDF for report %s: %s", report_id[:8], e)
            return None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def _save_report(self, report_type: str, title: str,
                     period_start: datetime, period_end: datetime,
                     html_content: str, sections_data: dict,
                     summary: str) -> Report:
        """Persist a report to the database and return the Report object."""
        report_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        report = Report(
            report_id=report_id,
            report_type=report_type,
            title=title,
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
            html_content=html_content,
            summary=summary,
            data_json=json.dumps(sections_data, default=str),
            created_at=now_iso,
        )

        self._conn.execute("""
            INSERT INTO reports
                (report_id, report_type, title, period_start, period_end,
                 html_content, pdf_path, summary, data_json, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report.report_id, report.report_type, report.title,
            report.period_start, report.period_end, report.html_content,
            report.pdf_path, report.summary, report.data_json,
            report.created_at, json.dumps(report.metadata),
        ))
        self._conn.commit()

        logger.info("Report saved: %s [%s] %s", report.short_id, report_type, title)
        return report

    def get_report(self, report_id: str) -> Report | None:
        """Retrieve a report by its UUID.

        Args:
            report_id: The report UUID.

        Returns:
            Report object or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        return Report.from_row(row) if row else None

    def list_reports(self, report_type: str | None = None,
                     limit: int = 50) -> list[Report]:
        """List reports, optionally filtered by type.

        Args:
            report_type: Optional filter — "daily", "weekly", or "monthly".
            limit: Maximum number of reports to return.

        Returns:
            List of Report objects, newest first.
        """
        if report_type:
            rows = self._conn.execute(
                "SELECT * FROM reports WHERE report_type = ? ORDER BY created_at DESC LIMIT ?",
                (report_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM reports ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [Report.from_row(r) for r in rows]

    def delete_report(self, report_id: str) -> bool:
        """Delete a report and its PDF file.

        Args:
            report_id: The report UUID.

        Returns:
            True if the report was found and deleted.
        """
        report = self.get_report(report_id)
        if report is None:
            return False

        # Remove PDF file if it exists
        if report.pdf_path:
            try:
                Path(report.pdf_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to delete PDF %s: %s", report.pdf_path, e)

        self._conn.execute("DELETE FROM reports WHERE report_id = ?", (report_id,))
        self._conn.commit()
        logger.info("Report deleted: %s", report_id[:8])
        return True

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def to_broadcast_dict(self) -> dict:
        """Return data suitable for broadcasting to dashboard clients.

        Includes report list (without html_content) and status summary.
        """
        reports = [r.to_dict() for r in self.list_reports(limit=50)]
        return {"reports": reports, "status": self.get_status()}

    def get_status(self) -> dict:
        """Return summary statistics about reports.

        Returns:
            Dict with total_reports, daily_count, weekly_count,
            monthly_count, last_daily, last_weekly, last_monthly.
        """
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) FROM reports")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM reports WHERE report_type = 'daily'")
        daily = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM reports WHERE report_type = 'weekly'")
        weekly = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM reports WHERE report_type = 'monthly'")
        monthly = cur.fetchone()[0]

        def last_of_type(rtype: str) -> str:
            row = cur.execute(
                "SELECT created_at FROM reports WHERE report_type = ? "
                "ORDER BY created_at DESC LIMIT 1", (rtype,)
            ).fetchone()
            return row[0] if row else ""

        return {
            "total_reports": total,
            "daily_count": daily,
            "weekly_count": weekly,
            "monthly_count": monthly,
            "last_daily": last_of_type("daily"),
            "last_weekly": last_of_type("weekly"),
            "last_monthly": last_of_type("monthly"),
        }

    def close(self):
        """Close the SQLite connection."""
        self._conn.close()
        logger.info("ReportEngine closed")


# ======================================================================
# Module-level helpers
# ======================================================================

def _esc(text: str) -> str:
    """Minimal HTML escaping for report content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _pdf_safe(text: str) -> str:
    """Replace Unicode characters unsupported by Helvetica with ASCII equivalents."""
    return (
        text.replace("\u2014", "-")    # em dash
        .replace("\u2013", "-")         # en dash
        .replace("\u2018", "'")         # left single quote
        .replace("\u2019", "'")         # right single quote
        .replace("\u201c", '"')         # left double quote
        .replace("\u201d", '"')         # right double quote
        .replace("\u2026", "...")        # ellipsis
        .replace("\u2022", "*")          # bullet
    )


def _extract_table_rows(html: str) -> list[list[str]]:
    """Extract text content from simple HTML table rows for PDF rendering.

    Handles both <th> and <td> elements. Returns a list of rows,
    where each row is a list of cell text values.
    """
    import re
    rows = []
    # Match each <tr>...</tr> block
    for tr_match in re.finditer(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL):
        tr_content = tr_match.group(1)
        cells = []
        # Match <td> or <th> elements
        for cell_match in re.finditer(r'<t[dh][^>]*>(.*?)</t[dh]>', tr_content, re.DOTALL):
            # Strip HTML tags from cell content
            text = re.sub(r'<[^>]+>', '', cell_match.group(1)).strip()
            # Unescape basic HTML entities
            text = text.replace("&amp;", "&").replace("&lt;", "<")
            text = text.replace("&gt;", ">").replace("&quot;", '"')
            cells.append(text)
        if cells:
            rows.append(cells)
    return rows
