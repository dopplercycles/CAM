"""
Email Template System for Doppler Cycles.

Generates personalized customer emails using CRM + service data via
the model router.  Outputs both plain text and HTML.  Stores sent
communications in CRM history.

SMTP sending defaults to off (copy-ready output); enable via
config/settings.toml [email] section.

SQLite-backed, single-file module — same pattern as invoicing.py.
"""

import json
import logging
import smtplib
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPLATE_TYPES = (
    "appointment_confirmation",
    "appointment_reminder",
    "service_complete",
    "invoice_sent",
    "follow_up_30day",
    "seasonal_maintenance_reminder",
    "thank_you",
)

EMAIL_STATUSES = ("draft", "queued", "sent", "failed")

# Human-readable labels for the dashboard dropdown
TEMPLATE_LABELS = {
    "appointment_confirmation": "Appointment Confirmation",
    "appointment_reminder": "Appointment Reminder",
    "service_complete": "Service Complete Summary",
    "invoice_sent": "Invoice Cover Letter",
    "follow_up_30day": "30-Day Follow-Up",
    "seasonal_maintenance_reminder": "Seasonal Maintenance Reminder",
    "thank_you": "Thank You",
}

# Current month → season mapping for seasonal prompts
_MONTH_TO_SEASON = {
    1: "winter", 2: "winter", 3: "spring", 4: "spring",
    5: "spring", 6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall", 12: "winter",
}


# ---------------------------------------------------------------------------
# EmailRecord dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmailRecord:
    """A single email communication record."""

    email_id: str = ""
    customer_id: str = ""
    customer_name: str = ""
    customer_email: str = ""
    template_type: str = ""
    subject: str = ""
    body_text: str = ""
    body_html: str = ""
    status: str = "draft"
    sent_at: str = ""
    error: str = ""
    context_data: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.email_id[:8] if self.email_id else ""

    def to_dict(self) -> dict:
        return {
            "email_id": self.email_id,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "customer_email": self.customer_email,
            "template_type": self.template_type,
            "subject": self.subject,
            "body_text": self.body_text,
            "body_html": self.body_html,
            "status": self.status,
            "sent_at": self.sent_at,
            "error": self.error,
            "context_data": self.context_data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "short_id": self.short_id,
        }

    @staticmethod
    def from_row(row) -> "EmailRecord":
        """Build an EmailRecord from a sqlite3.Row."""
        d = dict(row)
        return EmailRecord(
            email_id=d.get("email_id", ""),
            customer_id=d.get("customer_id", ""),
            customer_name=d.get("customer_name", ""),
            customer_email=d.get("customer_email", ""),
            template_type=d.get("template_type", ""),
            subject=d.get("subject", ""),
            body_text=d.get("body_text", ""),
            body_html=d.get("body_html", ""),
            status=d.get("status", "draft"),
            sent_at=d.get("sent_at", ""),
            error=d.get("error", ""),
            context_data=json.loads(d.get("context_data") or "{}"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            metadata=json.loads(d.get("metadata") or "{}"),
        )


# ---------------------------------------------------------------------------
# EmailTemplateManager
# ---------------------------------------------------------------------------

class EmailTemplateManager:
    """Manages customer email generation, queueing, and sending.

    Args:
        db_path:       Path to the SQLite database file.
        on_change:     Async callback fired after any mutation (for dashboard broadcast).
        router:        ModelRouter instance for AI-powered email generation.
        crm_store:     CRMStore instance for customer data lookups.
        service_store: ServiceRecordStore instance for service history.
        config:        CAMConfig object (reads config.email for SMTP settings).
    """

    def __init__(
        self,
        db_path: str = "data/email_communications.db",
        on_change: Optional[Callable[[], Coroutine]] = None,
        router: Any = None,
        crm_store: Any = None,
        service_store: Any = None,
        config: Any = None,
    ):
        self._db_path = db_path
        self._on_change = on_change
        self._router = router
        self._crm_store = crm_store
        self._service_store = service_store
        self._config = config

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("EmailTemplateManager initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS email_communications (
                email_id        TEXT PRIMARY KEY,
                customer_id     TEXT DEFAULT '',
                customer_name   TEXT DEFAULT '',
                customer_email  TEXT DEFAULT '',
                template_type   TEXT DEFAULT '',
                subject         TEXT DEFAULT '',
                body_text       TEXT DEFAULT '',
                body_html       TEXT DEFAULT '',
                status          TEXT DEFAULT 'draft',
                sent_at         TEXT DEFAULT '',
                error           TEXT DEFAULT '',
                context_data    TEXT DEFAULT '{}',
                created_at      TEXT DEFAULT '',
                updated_at      TEXT DEFAULT '',
                metadata        TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_em_customer   ON email_communications(customer_id);
            CREATE INDEX IF NOT EXISTS idx_em_template   ON email_communications(template_type);
            CREATE INDEX IF NOT EXISTS idx_em_status     ON email_communications(status);
            CREATE INDEX IF NOT EXISTS idx_em_created    ON email_communications(created_at);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    async def _notify(self):
        """Fire the on_change callback if set."""
        if self._on_change:
            try:
                await self._on_change()
            except Exception:
                logger.debug("EmailTemplateManager on_change callback error", exc_info=True)

    def _email_cfg(self) -> Any:
        """Return the email config section (or None)."""
        if self._config is None:
            return None
        return getattr(self._config, "email", None)

    # ------------------------------------------------------------------
    # Context gathering
    # ------------------------------------------------------------------

    def _gather_context(
        self,
        customer_id: str,
        template_type: str,
        extra_data: dict | None = None,
    ) -> dict:
        """Build a context dict for prompt injection.

        Pulls customer profile from CRM + service store, merges
        template-specific extra_data, and adds temporal info.
        """
        ctx: dict[str, Any] = {
            "template_type": template_type,
            "generated_at": self._now(),
        }

        # Current time info for seasonal relevance
        now = datetime.now(timezone.utc)
        ctx["current_month"] = now.strftime("%B")
        ctx["current_year"] = str(now.year)
        ctx["current_season"] = _MONTH_TO_SEASON.get(now.month, "")

        # Pull CRM profile if available
        if self._crm_store and customer_id:
            try:
                profile = self._crm_store.get_customer_profile(customer_id)
                if profile:
                    ctx["customer"] = profile.get("customer", {})
                    ctx["vehicles"] = profile.get("vehicles", [])
                    ctx["service_records"] = profile.get("records", [])[:5]
                    ctx["notes"] = profile.get("notes", [])[:5]
            except Exception:
                logger.debug("Failed to gather CRM context for %s", customer_id, exc_info=True)

        # Merge extra data (appointment details, invoice info, etc.)
        if extra_data:
            ctx["extra"] = extra_data

        return ctx

    # ------------------------------------------------------------------
    # Prompt builders (per template type)
    # ------------------------------------------------------------------

    def _build_prompt(self, template_type: str, ctx: dict) -> str:
        """Build the generation prompt for a specific template type."""
        customer = ctx.get("customer", {})
        name = customer.get("name", "Valued Customer")
        vehicles = ctx.get("vehicles", [])
        vehicle_str = ", ".join(
            v.get("year_make_model", v.get("description", "motorcycle"))
            for v in vehicles[:3]
        ) if vehicles else "motorcycle"
        extra = ctx.get("extra", {})

        base = (
            f"Write a professional email from Doppler Cycles (Portland, OR mobile "
            f"motorcycle diagnostics) to {name}.\n"
            f"Customer's vehicle(s): {vehicle_str}.\n"
        )

        prompts = {
            "appointment_confirmation": (
                f"{base}"
                f"Template: Appointment Confirmation\n"
                f"Appointment date/time: {extra.get('date', 'TBD')} at {extra.get('time', 'TBD')}\n"
                f"Location: {extra.get('location', 'Customer location')}\n"
                f"Services: {extra.get('services', 'Diagnostic inspection')}\n"
                f"Write a warm confirmation email. Include what to expect, "
                f"remind them to have the bike accessible, and provide contact info."
            ),
            "appointment_reminder": (
                f"{base}"
                f"Template: Appointment Reminder\n"
                f"Appointment date/time: {extra.get('date', 'TBD')} at {extra.get('time', 'TBD')}\n"
                f"Days until appointment: {extra.get('days_until', '1')}\n"
                f"Services: {extra.get('services', 'Diagnostic inspection')}\n"
                f"Write a friendly reminder. Keep it brief and helpful."
            ),
            "service_complete": (
                f"{base}"
                f"Template: Service Complete Summary\n"
                f"Services performed: {extra.get('services_performed', 'Diagnostic inspection')}\n"
                f"Parts used: {extra.get('parts_used', 'None')}\n"
                f"Recommendations: {extra.get('recommendations', 'None')}\n"
                f"Total cost: {extra.get('total_cost', 'See invoice')}\n"
                f"Summarize the work done, highlight important findings, "
                f"and mention recommended follow-up maintenance."
            ),
            "invoice_sent": (
                f"{base}"
                f"Template: Invoice Cover Letter\n"
                f"Invoice number: {extra.get('invoice_number', '')}\n"
                f"Total: {extra.get('total', '')}\n"
                f"Payment terms: {extra.get('payment_terms', 'Due on receipt')}\n"
                f"Write a professional cover letter for the attached invoice. "
                f"Include payment instructions (Zelle, Venmo, check, or cash)."
            ),
            "follow_up_30day": (
                f"{base}"
                f"Template: 30-Day Follow-Up\n"
                f"Last service date: {extra.get('last_service_date', 'recently')}\n"
                f"Services performed: {extra.get('services_performed', 'service work')}\n"
                f"Recommendations from last visit: {extra.get('recommendations', 'None')}\n"
                f"Write a genuine check-in. Ask how the bike is running, "
                f"remind them of any recommendations, offer to help with questions."
            ),
            "seasonal_maintenance_reminder": (
                f"{base}"
                f"Template: Seasonal Maintenance Reminder\n"
                f"Current season: {ctx.get('current_season', 'this season')}\n"
                f"Month: {ctx.get('current_month', '')}\n"
                f"Write season-specific maintenance tips for their bike(s). "
                f"Include relevant checks (tires, fluids, battery, chain) for "
                f"the {ctx.get('current_season', '')} riding season in the Pacific Northwest."
            ),
            "thank_you": (
                f"{base}"
                f"Template: Thank You\n"
                f"Customer since: {customer.get('created_at', 'recently')}\n"
                f"Total services: {extra.get('total_services', 'multiple')}\n"
                f"Write a genuine thank-you note. Mention their loyalty, "
                f"reference their bike(s) by name, keep it personal and warm."
            ),
        }

        prompt = prompts.get(template_type, f"{base}Write a professional email.")

        prompt += (
            "\n\nFormat your response EXACTLY as:\n"
            "SUBJECT: <email subject line>\n"
            "BODY:\n<email body text>\n\n"
            "Use plain text only. No HTML tags. Keep it concise (under 200 words). "
            "Sign off as Doppler Cycles — George & Cam."
        )

        return prompt

    # ------------------------------------------------------------------
    # Email generation (async, uses router)
    # ------------------------------------------------------------------

    async def generate_email(
        self,
        customer_id: str,
        template_type: str,
        extra_data: dict | None = None,
    ) -> EmailRecord:
        """Generate a personalized email using the model router.

        Routes at task_complexity="routine" (free local Ollama).
        Creates an EmailRecord in draft status and logs to CRM notes.
        """
        if template_type not in TEMPLATE_TYPES:
            raise ValueError(f"Unknown template type: {template_type}. Valid: {TEMPLATE_TYPES}")

        ctx = self._gather_context(customer_id, template_type, extra_data)
        prompt = self._build_prompt(template_type, ctx)

        system_prompt = (
            "You are writing emails for Doppler Cycles, a mobile motorcycle "
            "diagnostics and repair business in Portland, Oregon. Owner: George. "
            "Tone: professional but warm, rider-to-rider. Never pushy or salesy. "
            "Think of how a trusted mechanic talks to a regular customer."
        )

        # Route to local model (free)
        subject = f"Doppler Cycles — {TEMPLATE_LABELS.get(template_type, template_type)}"
        body_text = ""

        if self._router:
            try:
                resp = await self._router.route(
                    prompt=prompt,
                    task_complexity="routine",
                    system_prompt=system_prompt,
                )
                raw = resp.text if hasattr(resp, "text") else str(resp)
                subject, body_text = self._parse_response(raw, template_type)
            except Exception:
                logger.exception("Email generation failed for %s/%s", customer_id, template_type)
                body_text = f"[Generation failed — please write manually]\n\nTemplate: {template_type}"
        else:
            body_text = f"[No model router configured — please write manually]\n\nTemplate: {template_type}"

        # Get customer info from context
        customer = ctx.get("customer", {})
        customer_name = customer.get("name", "")
        customer_email = customer.get("email", "")

        # Generate HTML version
        body_html = self._wrap_html(subject, body_text)

        # Create the record
        now = self._now()
        email_id = str(uuid.uuid4())
        rec = EmailRecord(
            email_id=email_id,
            customer_id=customer_id,
            customer_name=customer_name,
            customer_email=customer_email,
            template_type=template_type,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            status="draft",
            context_data=ctx,
            created_at=now,
            updated_at=now,
        )

        self._conn.execute(
            """INSERT INTO email_communications
               (email_id, customer_id, customer_name, customer_email,
                template_type, subject, body_text, body_html,
                status, sent_at, error, context_data,
                created_at, updated_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rec.email_id, rec.customer_id, rec.customer_name,
                rec.customer_email, rec.template_type, rec.subject,
                rec.body_text, rec.body_html, rec.status,
                rec.sent_at, rec.error, json.dumps(rec.context_data),
                rec.created_at, rec.updated_at, json.dumps(rec.metadata),
            ),
        )
        self._conn.commit()

        # Log to CRM notes
        if self._crm_store and customer_id:
            try:
                self._crm_store.add_note(
                    customer_id=customer_id,
                    content=f"Email generated: {TEMPLATE_LABELS.get(template_type, template_type)} — \"{subject}\"",
                    category="communication",
                )
            except Exception:
                logger.debug("Failed to log email to CRM notes", exc_info=True)

        logger.info(
            "Email generated: %s [%s] for customer %s",
            rec.short_id, template_type, customer_id[:8] if customer_id else "unknown",
        )
        await self._notify()
        return rec

    def _parse_response(self, raw: str, template_type: str) -> tuple[str, str]:
        """Parse the model response into (subject, body_text)."""
        subject = f"Doppler Cycles — {TEMPLATE_LABELS.get(template_type, template_type)}"
        body_text = raw.strip()

        # Try to parse SUBJECT: / BODY: format
        lines = raw.strip().split("\n")
        for i, line in enumerate(lines):
            if line.upper().startswith("SUBJECT:"):
                subject = line[len("SUBJECT:"):].strip()
                # Everything after "BODY:" line is the body
                rest = "\n".join(lines[i + 1:])
                if rest.upper().startswith("BODY:"):
                    rest = rest[len("BODY:"):].strip()
                elif "\nBODY:" in rest.upper():
                    idx = rest.upper().index("\nBODY:")
                    rest = rest[idx + len("\nBODY:"):].strip()
                body_text = rest.strip()
                break

        return subject, body_text

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------

    def _wrap_html(self, subject: str, body_text: str) -> str:
        """Wrap plain text body in an email-compatible HTML template.

        Uses inline CSS for maximum email client compatibility.
        """
        # Escape HTML entities in user content
        import html
        safe_subject = html.escape(subject)
        # Convert plain text to HTML paragraphs
        paragraphs = body_text.strip().split("\n\n")
        body_paragraphs = ""
        for p in paragraphs:
            lines = p.strip().split("\n")
            content = "<br>".join(html.escape(line) for line in lines)
            body_paragraphs += f'<p style="margin: 0 0 16px 0; line-height: 1.6;">{content}</p>\n'

        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_subject}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f4f4f4; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #f4f4f4;">
    <tr>
      <td align="center" style="padding: 24px 16px;">
        <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width: 600px; width: 100%;">
          <!-- Header -->
          <tr>
            <td style="background-color: #1a1a2e; padding: 24px 32px; border-radius: 8px 8px 0 0; text-align: center;">
              <h1 style="margin: 0; color: #ffffff; font-size: 22px; font-weight: 700; letter-spacing: 1px;">
                DOPPLER CYCLES
              </h1>
              <p style="margin: 4px 0 0 0; color: #a0a0c0; font-size: 13px;">
                Mobile Motorcycle Diagnostics &amp; Repair — Portland, OR
              </p>
            </td>
          </tr>
          <!-- Body -->
          <tr>
            <td style="background-color: #ffffff; padding: 32px; border-left: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0;">
              <h2 style="margin: 0 0 20px 0; color: #1a1a2e; font-size: 18px; font-weight: 600;">
                {safe_subject}
              </h2>
              <div style="color: #333333; font-size: 15px;">
                {body_paragraphs}
              </div>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="background-color: #f8f8fc; padding: 20px 32px; border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 8px 8px; text-align: center;">
              <p style="margin: 0 0 8px 0; color: #666; font-size: 13px;">
                <strong>Doppler Cycles</strong> — George
              </p>
              <p style="margin: 0 0 4px 0; color: #888; font-size: 12px;">
                Mobile service · Portland metro area
              </p>
              <p style="margin: 0; color: #888; font-size: 12px;">
                cam@dopplercycles.com
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

    # ------------------------------------------------------------------
    # SMTP sending
    # ------------------------------------------------------------------

    def send_email(self, email_id: str) -> bool:
        """Send an email via SMTP or mark as sent (copy-ready mode).

        If config.email.enabled is False, marks as 'sent' without
        actually sending (copy-ready mode for manual pasting).

        Returns True on success, False on failure.
        """
        rec = self.get_email(email_id)
        if not rec:
            logger.warning("send_email: email %s not found", email_id)
            return False

        if rec.status not in ("draft", "queued"):
            logger.warning("send_email: email %s has status '%s', expected draft/queued", email_id, rec.status)
            return False

        cfg = self._email_cfg()
        enabled = getattr(cfg, "enabled", False) if cfg else False
        now = self._now()

        if not enabled:
            # Copy-ready mode — just mark as sent
            self._conn.execute(
                "UPDATE email_communications SET status='sent', sent_at=?, updated_at=? WHERE email_id=?",
                (now, now, email_id),
            )
            self._conn.commit()
            logger.info("Email %s marked sent (copy-ready mode)", rec.short_id)
            return True

        # Actually send via SMTP
        try:
            smtp_host = getattr(cfg, "smtp_host", "")
            smtp_port = getattr(cfg, "smtp_port", 587)
            smtp_user = getattr(cfg, "smtp_user", "")
            smtp_password = getattr(cfg, "smtp_password", "")
            from_addr = getattr(cfg, "from_address", "cam@dopplercycles.com")
            use_tls = getattr(cfg, "use_tls", True)

            to_addr = rec.customer_email or getattr(cfg, "to_address", "")
            if not to_addr:
                raise ValueError("No recipient email address")

            msg = MIMEMultipart("alternative")
            msg["Subject"] = rec.subject
            msg["From"] = from_addr
            msg["To"] = to_addr

            msg.attach(MIMEText(rec.body_text, "plain"))
            msg.attach(MIMEText(rec.body_html, "html"))

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if use_tls:
                    server.starttls()
                if smtp_user:
                    server.login(smtp_user, smtp_password)
                server.sendmail(from_addr, [to_addr], msg.as_string())

            self._conn.execute(
                "UPDATE email_communications SET status='sent', sent_at=?, error='', updated_at=? WHERE email_id=?",
                (now, now, email_id),
            )
            self._conn.commit()
            logger.info("Email %s sent to %s", rec.short_id, to_addr)
            return True

        except Exception as e:
            err = str(e)
            self._conn.execute(
                "UPDATE email_communications SET status='failed', error=?, updated_at=? WHERE email_id=?",
                (err, now, email_id),
            )
            self._conn.commit()
            logger.exception("Failed to send email %s", rec.short_id)
            return False

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def queue_email(self, email_id: str) -> bool:
        """Move an email from draft to queued status."""
        now = self._now()
        cur = self._conn.execute(
            "UPDATE email_communications SET status='queued', updated_at=? WHERE email_id=? AND status='draft'",
            (now, email_id),
        )
        self._conn.commit()
        if cur.rowcount > 0:
            logger.info("Email %s queued", email_id[:8])
            return True
        return False

    def send_queued(self) -> int:
        """Send all queued emails. Returns count of successfully sent."""
        rows = self._conn.execute(
            "SELECT email_id FROM email_communications WHERE status='queued' ORDER BY created_at"
        ).fetchall()

        sent_count = 0
        for row in rows:
            if self.send_email(row["email_id"]):
                sent_count += 1
        return sent_count

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_email(self, email_id: str) -> Optional[EmailRecord]:
        """Retrieve a single email record by ID."""
        row = self._conn.execute(
            "SELECT * FROM email_communications WHERE email_id=?", (email_id,)
        ).fetchone()
        return EmailRecord.from_row(row) if row else None

    def delete_email(self, email_id: str) -> bool:
        """Delete an email record."""
        cur = self._conn.execute(
            "DELETE FROM email_communications WHERE email_id=?", (email_id,)
        )
        self._conn.commit()
        if cur.rowcount > 0:
            logger.info("Email %s deleted", email_id[:8])
            return True
        return False

    def list_emails(
        self,
        customer_id: str = "",
        template_type: str = "",
        status: str = "",
        limit: int = 50,
    ) -> list[EmailRecord]:
        """List emails with optional filters."""
        clauses = []
        params: list[Any] = []

        if customer_id:
            clauses.append("customer_id=?")
            params.append(customer_id)
        if template_type:
            clauses.append("template_type=?")
            params.append(template_type)
        if status:
            clauses.append("status=?")
            params.append(status)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM email_communications{where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [EmailRecord.from_row(r) for r in rows]

    def get_customer_history(self, customer_id: str) -> list[dict]:
        """Get all email communications for a customer (as dicts for WS)."""
        rows = self._conn.execute(
            "SELECT * FROM email_communications WHERE customer_id=? ORDER BY created_at DESC",
            (customer_id,),
        ).fetchall()
        return [EmailRecord.from_row(r).to_dict() for r in rows]

    # ------------------------------------------------------------------
    # Dashboard / broadcast
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Aggregate stats for the dashboard status bar."""
        row = self._conn.execute("""
            SELECT
                COUNT(*)                                     AS total,
                SUM(CASE WHEN status='draft'  THEN 1 ELSE 0 END) AS drafts,
                SUM(CASE WHEN status='queued' THEN 1 ELSE 0 END) AS queued,
                SUM(CASE WHEN status='sent'   THEN 1 ELSE 0 END) AS sent,
                SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed
            FROM email_communications
        """).fetchone()
        return {
            "total": row["total"] or 0,
            "drafts": row["drafts"] or 0,
            "queued": row["queued"] or 0,
            "sent": row["sent"] or 0,
            "failed": row["failed"] or 0,
        }

    def to_broadcast_dict(self) -> dict:
        """Full state snapshot for WebSocket broadcast."""
        emails = self.list_emails(limit=100)
        return {
            "emails": [e.to_dict() for e in emails],
            "status": self.get_status(),
            "template_types": list(TEMPLATE_TYPES),
            "template_labels": TEMPLATE_LABELS,
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("EmailTemplateManager closed")
