"""
Customer Feedback System for CAM / Doppler Cycles.

Collects and manages customer feedback after service completion.  Tracks
multi-category ratings (service quality, communication, timeliness, value),
analyzes sentiment via the model router, calculates NPS and satisfaction
averages, and queues positive reviews for testimonial use in content.

Integrates with:
  - CRMStore          (customer lookups, notes)
  - ServiceRecordStore (service context for follow-ups)
  - EmailTemplateManager (follow-up message generation)
  - ModelRouter        (sentiment analysis)
  - NotificationManager (flagging negative reviews)

SQLite-backed, single-file module — same pattern as training.py, ride_log.py.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("cam.feedback")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RATING_CATEGORIES = ("service_quality", "communication", "timeliness", "value")

RATING_LABELS = {
    "service_quality": "Service Quality",
    "communication": "Communication",
    "timeliness": "Timeliness",
    "value": "Value",
}

FEEDBACK_STATUSES = ("pending", "reviewed", "responded", "archived")

# NPS thresholds (on a 1-5 scale mapped to NPS convention):
#   5 = Promoter, 4 = Passive, 1-3 = Detractor
NPS_PROMOTER_MIN = 5
NPS_PASSIVE_MIN = 4

SENTIMENT_ANALYSIS_PROMPT = """\
You are a customer service analyst.  Analyze the following customer feedback
for a mobile motorcycle diagnostics and repair business.

Customer: {customer_name}
Service date: {service_date}
Overall rating: {overall_rating}/5
Category ratings: {category_ratings}

Comments:
{comments}

Provide a JSON object with these keys:
- sentiment: "positive", "neutral", or "negative"
- key_themes: array of 1-3 short theme strings
- action_needed: boolean (true if negative or if customer mentions unresolved issue)
- summary: one-sentence summary of the feedback
- suggested_response: brief suggested response if action is needed, else ""

Return ONLY the JSON object — no markdown fences, no commentary.
"""

FOLLOW_UP_TEMPLATE = """\
Hi {customer_name},

Thanks for choosing Doppler Cycles for your recent {service_type} on {service_date}. \
I hope everything is running smoothly!

I'd really appreciate a few minutes of your time to share how the experience went. \
Your feedback helps me improve and keeps me honest about the quality of work I deliver.

You can rate your experience on:
- Service Quality (1-5)
- Communication (1-5)
- Timeliness (1-5)
- Value (1-5)

And if you have any comments, concerns, or things I should do differently next time, \
I want to hear them.

{follow_up_link}

Thanks again for your business.

George
Doppler Cycles — Mobile Motorcycle Diagnostics & Repair
"""


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Feedback:
    """A single customer feedback entry."""
    feedback_id: str = ""
    customer_id: str = ""
    customer_name: str = ""
    service_record_id: str = ""
    date: str = ""
    overall_rating: int = 0
    service_quality: int = 0
    communication: int = 0
    timeliness: int = 0
    value: int = 0
    comments: str = ""
    sentiment: str = ""
    sentiment_data: dict = field(default_factory=dict)
    follow_up_needed: bool = False
    resolved: bool = False
    resolved_notes: str = ""
    testimonial_approved: bool = False
    testimonial_queued: bool = False
    status: str = "pending"
    created_at: str = ""
    updated_at: str = ""

    @property
    def short_id(self) -> str:
        return self.feedback_id[:8] if self.feedback_id else ""

    @property
    def avg_rating(self) -> float:
        """Average of all category ratings that are > 0."""
        cats = [self.service_quality, self.communication,
                self.timeliness, self.value]
        rated = [r for r in cats if r > 0]
        return round(sum(rated) / len(rated), 1) if rated else 0.0

    @property
    def nps_category(self) -> str:
        """NPS category based on overall rating (1-5 scale)."""
        if self.overall_rating >= NPS_PROMOTER_MIN:
            return "promoter"
        elif self.overall_rating >= NPS_PASSIVE_MIN:
            return "passive"
        return "detractor"

    def to_dict(self) -> dict:
        return {
            "feedback_id": self.feedback_id,
            "short_id": self.short_id,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "service_record_id": self.service_record_id,
            "date": self.date,
            "overall_rating": self.overall_rating,
            "service_quality": self.service_quality,
            "communication": self.communication,
            "timeliness": self.timeliness,
            "value": self.value,
            "avg_rating": self.avg_rating,
            "comments": self.comments,
            "sentiment": self.sentiment,
            "sentiment_data": self.sentiment_data,
            "follow_up_needed": self.follow_up_needed,
            "resolved": self.resolved,
            "resolved_notes": self.resolved_notes,
            "testimonial_approved": self.testimonial_approved,
            "testimonial_queued": self.testimonial_queued,
            "nps_category": self.nps_category,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_row(row) -> "Feedback":
        """Build a Feedback from a sqlite3.Row."""
        r = dict(row)
        return Feedback(
            feedback_id=r["feedback_id"],
            customer_id=r.get("customer_id", ""),
            customer_name=r.get("customer_name", ""),
            service_record_id=r.get("service_record_id", ""),
            date=r.get("date", ""),
            overall_rating=int(r.get("overall_rating", 0)),
            service_quality=int(r.get("service_quality", 0)),
            communication=int(r.get("communication", 0)),
            timeliness=int(r.get("timeliness", 0)),
            value=int(r.get("value", 0)),
            comments=r.get("comments", ""),
            sentiment=r.get("sentiment", ""),
            sentiment_data=json.loads(r.get("sentiment_data", "{}")),
            follow_up_needed=bool(r.get("follow_up_needed", 0)),
            resolved=bool(r.get("resolved", 0)),
            resolved_notes=r.get("resolved_notes", ""),
            testimonial_approved=bool(r.get("testimonial_approved", 0)),
            testimonial_queued=bool(r.get("testimonial_queued", 0)),
            status=r.get("status", "pending"),
            created_at=r.get("created_at", ""),
            updated_at=r.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# FeedbackManager
# ---------------------------------------------------------------------------

class FeedbackManager:
    """SQLite-backed customer feedback manager for Doppler Cycles.

    Collects multi-category ratings, analyzes sentiment via model router,
    calculates NPS and satisfaction trends, flags negative reviews for
    immediate attention, and queues positive feedback for testimonial use.

    Args:
        db_path:                Path to the SQLite database file.
        router:                 ModelRouter for sentiment analysis (optional).
        crm_store:              CRMStore for customer lookups (optional).
        service_store:          ServiceRecordStore for service context (optional).
        notification_manager:   NotificationManager for flagging (optional).
        on_change:              Async callback on state mutations.
        on_model_call:          Callback for model usage tracking.
    """

    def __init__(
        self,
        db_path: str = "data/feedback.db",
        router: Any = None,
        crm_store: Any = None,
        service_store: Any = None,
        notification_manager: Any = None,
        on_change: Optional[Callable[[], Coroutine]] = None,
        on_model_call: Optional[Callable] = None,
    ):
        self._db_path = db_path
        self._router = router
        self._crm = crm_store
        self._service_store = service_store
        self._notification_mgr = notification_manager
        self._on_change = on_change
        self._on_model_call = on_model_call

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("FeedbackManager initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        """Create the feedback table if it doesn't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id        TEXT PRIMARY KEY,
                customer_id        TEXT DEFAULT '',
                customer_name      TEXT DEFAULT '',
                service_record_id  TEXT DEFAULT '',
                date               TEXT NOT NULL,
                overall_rating     INTEGER DEFAULT 0,
                service_quality    INTEGER DEFAULT 0,
                communication      INTEGER DEFAULT 0,
                timeliness         INTEGER DEFAULT 0,
                value              INTEGER DEFAULT 0,
                comments           TEXT DEFAULT '',
                sentiment          TEXT DEFAULT '',
                sentiment_data     TEXT DEFAULT '{}',
                follow_up_needed   INTEGER DEFAULT 0,
                resolved           INTEGER DEFAULT 0,
                resolved_notes     TEXT DEFAULT '',
                testimonial_approved INTEGER DEFAULT 0,
                testimonial_queued INTEGER DEFAULT 0,
                status             TEXT DEFAULT 'pending',
                created_at         TEXT NOT NULL,
                updated_at         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fb_customer ON feedback(customer_id);
            CREATE INDEX IF NOT EXISTS idx_fb_date ON feedback(date);
            CREATE INDEX IF NOT EXISTS idx_fb_status ON feedback(status);
            CREATE INDEX IF NOT EXISTS idx_fb_sentiment ON feedback(sentiment);

            CREATE TABLE IF NOT EXISTS feedback_requests (
                request_id         TEXT PRIMARY KEY,
                customer_id        TEXT NOT NULL,
                customer_name      TEXT DEFAULT '',
                service_record_id  TEXT DEFAULT '',
                service_date       TEXT DEFAULT '',
                send_after         TEXT NOT NULL,
                sent               INTEGER DEFAULT 0,
                feedback_id        TEXT DEFAULT '',
                created_at         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fbr_send ON feedback_requests(sent, send_after);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now(self) -> str:
        """ISO-8601 UTC timestamp."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _fire_change(self):
        """Schedule the on_change callback if one was provided."""
        if self._on_change:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._on_change())
            except RuntimeError:
                pass

    @staticmethod
    def _clamp_rating(val: int) -> int:
        """Clamp a rating value to 0-5 range."""
        return max(0, min(5, int(val)))

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def submit_feedback(
        self,
        customer_id: str = "",
        customer_name: str = "",
        service_record_id: str = "",
        date: str = "",
        overall_rating: int = 0,
        service_quality: int = 0,
        communication: int = 0,
        timeliness: int = 0,
        value: int = 0,
        comments: str = "",
        testimonial_approved: bool = False,
    ) -> Feedback:
        """Submit a new feedback entry.

        Args:
            customer_id:         CRM customer UUID (optional).
            customer_name:       Customer name for display.
            service_record_id:   Linked service record UUID (optional).
            date:                Feedback date (YYYY-MM-DD, defaults to today).
            overall_rating:      Overall satisfaction 1-5.
            service_quality:     Service quality rating 1-5.
            communication:       Communication rating 1-5.
            timeliness:          Timeliness rating 1-5.
            value:               Value rating 1-5.
            comments:            Free-text customer comments.
            testimonial_approved: Customer gave permission for testimonial use.

        Returns:
            The newly created Feedback entry.
        """
        if not date:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Look up customer name from CRM if not provided
        if customer_id and not customer_name and self._crm:
            try:
                cust = self._crm.get_customer(customer_id)
                if cust:
                    customer_name = cust.name
            except Exception:
                pass

        now = self._now()
        fb = Feedback(
            feedback_id=str(uuid.uuid4()),
            customer_id=customer_id,
            customer_name=customer_name,
            service_record_id=service_record_id,
            date=date,
            overall_rating=self._clamp_rating(overall_rating),
            service_quality=self._clamp_rating(service_quality),
            communication=self._clamp_rating(communication),
            timeliness=self._clamp_rating(timeliness),
            value=self._clamp_rating(value),
            comments=comments,
            testimonial_approved=testimonial_approved,
            status="pending",
            created_at=now,
            updated_at=now,
        )

        # Auto-flag low ratings for follow-up
        if fb.overall_rating > 0 and fb.overall_rating <= 2:
            fb.follow_up_needed = True

        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO feedback
               (feedback_id, customer_id, customer_name, service_record_id,
                date, overall_rating, service_quality, communication,
                timeliness, value, comments, sentiment, sentiment_data,
                follow_up_needed, resolved, resolved_notes,
                testimonial_approved, testimonial_queued, status,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fb.feedback_id, fb.customer_id, fb.customer_name,
                fb.service_record_id, fb.date, fb.overall_rating,
                fb.service_quality, fb.communication, fb.timeliness,
                fb.value, fb.comments, fb.sentiment,
                json.dumps(fb.sentiment_data), int(fb.follow_up_needed),
                int(fb.resolved), fb.resolved_notes,
                int(fb.testimonial_approved), int(fb.testimonial_queued),
                fb.status, fb.created_at, fb.updated_at,
            ),
        )
        self._conn.commit()

        logger.info("Feedback submitted: %s — %s, rating=%d/5",
                     fb.short_id, fb.customer_name, fb.overall_rating)

        # Notify on negative feedback
        if fb.follow_up_needed and self._notification_mgr:
            self._notification_mgr.emit(
                "warning",
                "Negative Feedback Received",
                f"{fb.customer_name} rated {fb.overall_rating}/5: {fb.comments[:100]}",
                "feedback",
            )

        # Add CRM note if available
        if fb.customer_id and self._crm:
            try:
                self._crm.add_note(
                    customer_id=fb.customer_id,
                    content=f"Feedback received: {fb.overall_rating}/5 overall. {fb.comments[:200]}",
                    category="feedback",
                )
            except Exception as exc:
                logger.warning("Failed to add CRM note for feedback: %s", exc)

        self._fire_change()
        return fb

    def update_feedback(self, feedback_id: str, **kwargs) -> Optional[Feedback]:
        """Update fields on an existing feedback entry.

        Args:
            feedback_id: The feedback UUID.
            **kwargs:    Fields to update.

        Returns:
            Updated Feedback, or None if not found.
        """
        existing = self.get_feedback(feedback_id)
        if not existing:
            return None

        allowed = {
            "overall_rating", "service_quality", "communication",
            "timeliness", "value", "comments", "sentiment", "sentiment_data",
            "follow_up_needed", "resolved", "resolved_notes",
            "testimonial_approved", "testimonial_queued", "status",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return existing

        # Clamp ratings
        for rating_field in ("overall_rating", "service_quality",
                             "communication", "timeliness", "value"):
            if rating_field in updates:
                updates[rating_field] = self._clamp_rating(updates[rating_field])

        # Serialize JSON fields
        if "sentiment_data" in updates:
            updates["sentiment_data"] = json.dumps(updates["sentiment_data"])

        # Convert booleans to int for SQLite
        for bool_field in ("follow_up_needed", "resolved",
                           "testimonial_approved", "testimonial_queued"):
            if bool_field in updates:
                updates[bool_field] = int(bool(updates[bool_field]))

        updates["updated_at"] = self._now()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [feedback_id]

        cur = self._conn.cursor()
        cur.execute(f"UPDATE feedback SET {set_clause} WHERE feedback_id = ?", values)
        self._conn.commit()

        logger.info("Feedback updated: %s", feedback_id[:8])
        self._fire_change()
        return self.get_feedback(feedback_id)

    def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a feedback entry.

        Returns:
            True if deleted, False if not found.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM feedback WHERE feedback_id = ?", (feedback_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Feedback deleted: %s", feedback_id[:8])
            self._fire_change()
        return deleted

    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """Get a single feedback entry by ID."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM feedback WHERE feedback_id = ?", (feedback_id,))
        row = cur.fetchone()
        return Feedback.from_row(row) if row else None

    def list_feedback(
        self,
        status: str = "",
        sentiment: str = "",
        flagged_only: bool = False,
        limit: int = 50,
    ) -> list[Feedback]:
        """List feedback entries, newest first.

        Args:
            status:       Filter by status (pending, reviewed, responded, archived).
            sentiment:    Filter by sentiment (positive, neutral, negative).
            flagged_only: Only show entries needing follow-up.
            limit:        Max results.

        Returns:
            List of Feedback objects.
        """
        query = "SELECT * FROM feedback WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if sentiment:
            query += " AND sentiment = ?"
            params.append(sentiment)
        if flagged_only:
            query += " AND follow_up_needed = 1 AND resolved = 0"

        query += " ORDER BY date DESC, created_at DESC LIMIT ?"
        params.append(limit)

        cur = self._conn.cursor()
        cur.execute(query, params)
        return [Feedback.from_row(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Follow-up requests
    # ------------------------------------------------------------------

    def request_feedback(
        self,
        customer_id: str,
        service_record_id: str = "",
        service_date: str = "",
        delay_days: int = 3,
    ) -> dict:
        """Schedule a feedback request to be sent after a delay.

        Generates a follow-up message to send to the customer N days after
        service completion.

        Args:
            customer_id:       CRM customer UUID.
            service_record_id: Related service record UUID.
            service_date:      Date of service (YYYY-MM-DD).
            delay_days:        Days to wait before sending (default 3).

        Returns:
            Dict with request_id, customer_name, send_after, message.
        """
        customer_name = ""
        service_type = "service"

        # Look up customer info
        if self._crm:
            try:
                cust = self._crm.get_customer(customer_id)
                if cust:
                    customer_name = cust.name
            except Exception:
                pass

        # Look up service record details
        if service_record_id and self._service_store:
            try:
                record = self._service_store.get_record(service_record_id)
                if record:
                    service_type = record.service_type or "service"
                    if not service_date:
                        service_date = record.date
                    if not customer_name:
                        customer_name = record.customer_name
            except Exception:
                pass

        if not service_date:
            service_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        send_after = (
            datetime.strptime(service_date, "%Y-%m-%d") + timedelta(days=delay_days)
        ).strftime("%Y-%m-%d")

        now = self._now()
        request_id = str(uuid.uuid4())

        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO feedback_requests
               (request_id, customer_id, customer_name, service_record_id,
                service_date, send_after, sent, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
            (request_id, customer_id, customer_name, service_record_id,
             service_date, send_after, now),
        )
        self._conn.commit()

        # Generate the follow-up message text
        message = FOLLOW_UP_TEMPLATE.format(
            customer_name=customer_name or "there",
            service_type=service_type,
            service_date=service_date,
            follow_up_link="(Feedback can be submitted through the Doppler Cycles dashboard)",
        )

        logger.info("Feedback request scheduled: %s for %s (send after %s)",
                     request_id[:8], customer_name, send_after)

        return {
            "request_id": request_id,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "service_record_id": service_record_id,
            "send_after": send_after,
            "message": message,
        }

    def get_pending_requests(self) -> list[dict]:
        """Get feedback requests that are due to be sent.

        Returns:
            List of request dicts that are past their send_after date
            and haven't been sent yet.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT * FROM feedback_requests
               WHERE sent = 0 AND send_after <= ?
               ORDER BY send_after""",
            (today,),
        )
        return [dict(row) for row in cur.fetchall()]

    def mark_request_sent(self, request_id: str) -> bool:
        """Mark a feedback request as sent."""
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE feedback_requests SET sent = 1 WHERE request_id = ?",
            (request_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def list_requests(self, limit: int = 20) -> list[dict]:
        """List recent feedback requests."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT * FROM feedback_requests
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Sentiment analysis
    # ------------------------------------------------------------------

    async def analyze_sentiment(self, feedback_id: str) -> dict:
        """Analyze sentiment of a feedback entry using the model router.

        Args:
            feedback_id: The feedback UUID to analyze.

        Returns:
            Dict with sentiment, key_themes, action_needed, summary.
        """
        fb = self.get_feedback(feedback_id)
        if not fb:
            return {"error": "Feedback not found"}

        if not fb.comments:
            # No text to analyze — infer from ratings
            if fb.overall_rating >= 4:
                sentiment = "positive"
            elif fb.overall_rating >= 3:
                sentiment = "neutral"
            else:
                sentiment = "negative"
            result = {
                "sentiment": sentiment,
                "key_themes": [],
                "action_needed": fb.overall_rating <= 2,
                "summary": f"Rating-only feedback: {fb.overall_rating}/5",
                "suggested_response": "",
            }
            self.update_feedback(feedback_id, sentiment=sentiment, sentiment_data=result)
            return result

        cat_ratings = ", ".join(
            f"{RATING_LABELS[c]}: {getattr(fb, c)}/5"
            for c in RATING_CATEGORIES if getattr(fb, c) > 0
        )

        prompt = SENTIMENT_ANALYSIS_PROMPT.format(
            customer_name=fb.customer_name or "Anonymous",
            service_date=fb.date,
            overall_rating=fb.overall_rating,
            category_ratings=cat_ratings or "not provided",
            comments=fb.comments,
        )

        result = {}
        if self._router:
            try:
                if self._on_model_call:
                    self._on_model_call("sentiment_analysis", "simple")
                response = await self._router.route(
                    prompt=prompt,
                    task_complexity="simple",
                )
                text = response.text.strip()
                # Strip markdown fences
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text
                    if text.endswith("```"):
                        text = text[:-3]
                result = json.loads(text)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Sentiment analysis parse failed: %s", e)

        if not result:
            # Fallback: simple keyword analysis
            result = self._fallback_sentiment(fb)

        sentiment = result.get("sentiment", "neutral")
        action_needed = result.get("action_needed", False)

        # Update the feedback entry
        self.update_feedback(
            feedback_id,
            sentiment=sentiment,
            sentiment_data=result,
            follow_up_needed=action_needed or fb.follow_up_needed,
        )

        # Notify on negative sentiment
        if sentiment == "negative" and self._notification_mgr:
            self._notification_mgr.emit(
                "warning",
                "Negative Feedback Flagged",
                f"{fb.customer_name}: {result.get('summary', fb.comments[:80])}",
                "feedback",
            )

        logger.info("Sentiment analyzed: %s — %s", feedback_id[:8], sentiment)
        return result

    def _fallback_sentiment(self, fb: Feedback) -> dict:
        """Simple keyword-based sentiment fallback without model router."""
        text = fb.comments.lower()
        positive_words = {"great", "excellent", "amazing", "perfect", "love",
                          "awesome", "fantastic", "wonderful", "best", "recommend",
                          "happy", "pleased", "satisfied", "professional", "thorough"}
        negative_words = {"bad", "terrible", "awful", "worst", "disappointed",
                          "poor", "rude", "slow", "overcharged", "broken",
                          "wrong", "never", "waste", "complaint", "unacceptable"}

        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)

        if neg_count > pos_count or fb.overall_rating <= 2:
            sentiment = "negative"
        elif pos_count > neg_count or fb.overall_rating >= 4:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "key_themes": [],
            "action_needed": sentiment == "negative",
            "summary": f"{'Positive' if sentiment == 'positive' else 'Negative' if sentiment == 'negative' else 'Neutral'} feedback — {fb.overall_rating}/5 overall",
            "suggested_response": "",
        }

    # ------------------------------------------------------------------
    # NPS and satisfaction metrics
    # ------------------------------------------------------------------

    def calculate_nps(self, months: int = 0) -> dict:
        """Calculate Net Promoter Score.

        NPS = % Promoters - % Detractors
        On our 1-5 scale: 5 = Promoter, 4 = Passive, 1-3 = Detractor.

        Args:
            months: Number of months to look back (0 = all time).

        Returns:
            Dict with nps, promoters, passives, detractors, total, pct breakdown.
        """
        query = "SELECT overall_rating FROM feedback WHERE overall_rating > 0"
        params: list[Any] = []

        if months > 0:
            now = datetime.now(timezone.utc)
            y, m = now.year, now.month - months
            while m <= 0:
                m += 12
                y -= 1
            start = f"{y:04d}-{m:02d}-01"
            query += " AND date >= ?"
            params.append(start)

        cur = self._conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()

        promoters = passives = detractors = 0
        for row in rows:
            rating = row[0]
            if rating >= NPS_PROMOTER_MIN:
                promoters += 1
            elif rating >= NPS_PASSIVE_MIN:
                passives += 1
            else:
                detractors += 1

        total = promoters + passives + detractors
        if total == 0:
            return {
                "nps": 0, "promoters": 0, "passives": 0, "detractors": 0,
                "total": 0, "promoter_pct": 0, "detractor_pct": 0,
            }

        promoter_pct = round(promoters / total * 100, 1)
        detractor_pct = round(detractors / total * 100, 1)
        nps = round(promoter_pct - detractor_pct, 1)

        return {
            "nps": nps,
            "promoters": promoters,
            "passives": passives,
            "detractors": detractors,
            "total": total,
            "promoter_pct": promoter_pct,
            "detractor_pct": detractor_pct,
        }

    def satisfaction_averages(self, months: int = 0) -> dict:
        """Calculate average ratings across all categories.

        Args:
            months: Number of months to look back (0 = all time).

        Returns:
            Dict with average for each category and overall.
        """
        query = """SELECT
            AVG(CASE WHEN overall_rating > 0 THEN overall_rating END) as avg_overall,
            AVG(CASE WHEN service_quality > 0 THEN service_quality END) as avg_service_quality,
            AVG(CASE WHEN communication > 0 THEN communication END) as avg_communication,
            AVG(CASE WHEN timeliness > 0 THEN timeliness END) as avg_timeliness,
            AVG(CASE WHEN value > 0 THEN value END) as avg_value,
            COUNT(*) as total
            FROM feedback WHERE overall_rating > 0"""
        params: list[Any] = []

        if months > 0:
            now = datetime.now(timezone.utc)
            y, m = now.year, now.month - months
            while m <= 0:
                m += 12
                y -= 1
            start = f"{y:04d}-{m:02d}-01"
            query += " AND date >= ?"
            params.append(start)

        cur = self._conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()

        def safe_round(val):
            return round(val, 1) if val else 0.0

        return {
            "overall": safe_round(row["avg_overall"]),
            "service_quality": safe_round(row["avg_service_quality"]),
            "communication": safe_round(row["avg_communication"]),
            "timeliness": safe_round(row["avg_timeliness"]),
            "value": safe_round(row["avg_value"]),
            "total_responses": row["total"],
        }

    def satisfaction_trend(self, months: int = 6) -> list[dict]:
        """Monthly satisfaction trend data.

        Args:
            months: Number of months to look back.

        Returns:
            List of dicts with month, avg_rating, count per month.
        """
        now = datetime.now(timezone.utc)
        y, m = now.year, now.month - months
        while m <= 0:
            m += 12
            y -= 1
        start = f"{y:04d}-{m:02d}-01"

        cur = self._conn.cursor()
        cur.execute(
            """SELECT
                SUBSTR(date, 1, 7) as month,
                AVG(overall_rating) as avg_rating,
                COUNT(*) as count
               FROM feedback
               WHERE overall_rating > 0 AND date >= ?
               GROUP BY SUBSTR(date, 1, 7)
               ORDER BY month""",
            (start,),
        )
        return [
            {
                "month": row["month"],
                "avg_rating": round(row["avg_rating"], 1),
                "count": row["count"],
            }
            for row in cur.fetchall()
        ]

    # ------------------------------------------------------------------
    # Testimonials
    # ------------------------------------------------------------------

    def queue_testimonial(self, feedback_id: str) -> Optional[Feedback]:
        """Queue a positive feedback entry for testimonial use in content.

        Only queues if testimonial_approved is True and rating >= 4.

        Args:
            feedback_id: The feedback UUID.

        Returns:
            Updated Feedback if queued, None if conditions not met.
        """
        fb = self.get_feedback(feedback_id)
        if not fb:
            return None

        if not fb.testimonial_approved:
            logger.info("Cannot queue testimonial: not approved by customer")
            return None

        if fb.overall_rating < 4:
            logger.info("Cannot queue testimonial: rating too low (%d/5)",
                        fb.overall_rating)
            return None

        return self.update_feedback(feedback_id, testimonial_queued=True)

    def get_testimonials(self, limit: int = 20) -> list[Feedback]:
        """Get feedback entries queued as testimonials."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT * FROM feedback
               WHERE testimonial_queued = 1 AND testimonial_approved = 1
               ORDER BY date DESC LIMIT ?""",
            (limit,),
        )
        return [Feedback.from_row(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Summary status for dashboard header cards."""
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) FROM feedback")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM feedback WHERE follow_up_needed = 1 AND resolved = 0")
        flagged = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM feedback WHERE testimonial_queued = 1")
        testimonials = cur.fetchone()[0]

        # This month
        month_start = datetime.now(timezone.utc).strftime("%Y-%m-01")
        cur.execute("SELECT COUNT(*) FROM feedback WHERE date >= ?", (month_start,))
        this_month = cur.fetchone()[0]

        # Average overall rating
        cur.execute(
            "SELECT AVG(overall_rating) FROM feedback WHERE overall_rating > 0"
        )
        row = cur.fetchone()
        avg_rating = round(row[0], 1) if row[0] else 0.0

        # Pending requests
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cur.execute(
            "SELECT COUNT(*) FROM feedback_requests WHERE sent = 0 AND send_after <= ?",
            (today,),
        )
        pending_requests = cur.fetchone()[0]

        nps = self.calculate_nps()

        return {
            "total_feedback": total,
            "flagged": flagged,
            "testimonials": testimonials,
            "this_month": this_month,
            "avg_rating": avg_rating,
            "pending_requests": pending_requests,
            "nps": nps["nps"],
        }

    def to_broadcast_dict(self) -> dict:
        """Full state dict for broadcasting to the dashboard."""
        feedback_list = self.list_feedback(limit=50)
        return {
            "feedback": [f.to_dict() for f in feedback_list],
            "status": self.get_status(),
            "nps": self.calculate_nps(),
            "satisfaction": self.satisfaction_averages(),
            "trend": self.satisfaction_trend(months=6),
            "testimonials": [t.to_dict() for t in self.get_testimonials(limit=10)],
            "pending_requests": self.get_pending_requests(),
        }

    def close(self):
        """Close the SQLite connection. Call on shutdown."""
        try:
            self._conn.close()
            logger.info("FeedbackManager closed")
        except Exception:
            pass
