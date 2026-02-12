"""
Doppler Scout — Motorcycle Listing Scraper

Monitors Craigslist and Facebook Marketplace for motorcycle deals matching
configurable criteria. Uses DuckDuckGo site-scoped queries (no direct
scraping of CL/FB), stores listings in SQLite, scores deal quality with
the local model router, and sends notifications for hot deals.

Follows the same pattern as core/research_store.py for SQLite storage
and dashboard integration.

Usage:
    from tools.doppler.scout import ScoutStore, DopplerScout, SearchCriteria

    store = ScoutStore()
    scout = DopplerScout(store, router=router)
    listings = await scout.scan()
"""

import asyncio
import json
import logging
import re
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

logger = logging.getLogger("cam.scout")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ListingStatus(Enum):
    """Status of a scout listing."""
    NEW = "new"
    REVIEWED = "reviewed"
    FLAGGED = "flagged"
    DISMISSED = "dismissed"


# ---------------------------------------------------------------------------
# SearchCriteria dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchCriteria:
    """Configurable search parameters for motorcycle listings.

    Attributes:
        makes:            Motorcycle makes to search for.
        models:           Specific models (empty = all).
        year_min:         Minimum year (0 = no filter).
        year_max:         Maximum year (0 = no filter).
        price_min:        Minimum price in USD.
        price_max:        Maximum price in USD.
        location:         Geographic search center.
        radius_miles:     Search radius from location.
        keywords:         Additional search terms to include.
        exclude_keywords: Terms that disqualify a listing.
    """
    makes: list[str] = field(default_factory=lambda: [
        "honda", "yamaha", "suzuki", "kawasaki", "harley-davidson", "ducati",
    ])
    models: list[str] = field(default_factory=list)
    year_min: int = 0
    year_max: int = 0
    price_min: int = 0
    price_max: int = 5000
    location: str = "Portland OR"
    radius_miles: int = 50
    keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ScoutListing dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoutListing:
    """A single motorcycle listing found by the scout.

    Attributes:
        entry_id:     Unique identifier (UUID string).
        title:        Listing title from search result.
        price:        Asking price in USD (0 if not found).
        url:          Original listing URL.
        source:       Where it was found (craigslist, facebook).
        location:     Listed location if available.
        date_found:   When the scout found this listing (ISO string).
        status:       Current review status (new, reviewed, flagged, dismissed).
        deal_score:   1-10 deal quality score from model.
        score_reason: Model's explanation for the score.
        make:         Motorcycle make (parsed from title).
        model:        Motorcycle model (parsed from title).
        year:         Model year (parsed from title, 0 if unknown).
        snippet:      Search result snippet text.
        created_at:   When the entry was created (ISO string).
        updated_at:   When the entry was last modified (ISO string).
        metadata:     Additional key-value data.
    """
    entry_id: str
    title: str
    price: int = 0
    url: str = ""
    source: str = ""
    location: str = ""
    date_found: str = ""
    status: str = ListingStatus.NEW.value
    deal_score: int = 0
    score_reason: str = ""
    make: str = ""
    model: str = ""
    year: int = 0
    snippet: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 characters of the entry ID for display."""
        return self.entry_id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/dashboard use."""
        return {
            "entry_id": self.entry_id,
            "short_id": self.short_id,
            "title": self.title,
            "price": self.price,
            "url": self.url,
            "source": self.source,
            "location": self.location,
            "date_found": self.date_found,
            "status": self.status,
            "deal_score": self.deal_score,
            "score_reason": self.score_reason,
            "make": self.make,
            "model": self.model,
            "year": self.year,
            "snippet": self.snippet,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ScoutListing":
        """Convert a SQLite row to a ScoutListing."""
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}

        return cls(
            entry_id=row["entry_id"],
            title=row["title"],
            price=row["price"],
            url=row["url"],
            source=row["source"],
            location=row["location"],
            date_found=row["date_found"],
            status=row["status"],
            deal_score=row["deal_score"],
            score_reason=row["score_reason"],
            make=row["make"],
            model=row["model"],
            year=row["year"],
            snippet=row["snippet"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# ScoutStore — SQLite-backed listing storage
# ---------------------------------------------------------------------------

class ScoutStore:
    """SQLite-backed motorcycle listing storage.

    Stores scout listings with CRUD operations and optional change
    callbacks for real-time dashboard updates.

    Args:
        db_path:    Path to the SQLite database file.
        on_change:  Async callback fired after every mutation (add/update/remove).
    """

    def __init__(
        self,
        db_path: str = "data/scout.db",
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

        logger.info("ScoutStore initialized (db=%s)", db_file)

    # -------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------

    def _init_db(self):
        """Create the scout_listings table and indexes if they don't exist."""
        cur = self._conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS scout_listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                price INTEGER NOT NULL DEFAULT 0,
                url TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                date_found TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'new',
                deal_score INTEGER NOT NULL DEFAULT 0,
                score_reason TEXT NOT NULL DEFAULT '',
                make TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                year INTEGER NOT NULL DEFAULT 0,
                snippet TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_scout_status
            ON scout_listings(status)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_scout_deal_score
            ON scout_listings(deal_score)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_scout_date_found
            ON scout_listings(date_found)
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_scout_url
            ON scout_listings(url)
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
                logger.debug("Scout store on_change callback error", exc_info=True)

    # -------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------

    def add_entry(
        self,
        title: str,
        url: str,
        source: str = "",
        price: int = 0,
        location: str = "",
        make: str = "",
        model: str = "",
        year: int = 0,
        snippet: str = "",
        metadata: dict | None = None,
    ) -> ScoutListing | None:
        """Create a new scout listing and persist it.

        Checks for URL duplicates first — returns None if the listing
        already exists.

        Args:
            title:    Listing title.
            url:      Original listing URL.
            source:   Source platform (craigslist, facebook).
            price:    Asking price in USD.
            location: Listed location.
            make:     Motorcycle make.
            model:    Motorcycle model.
            year:     Model year.
            snippet:  Search result snippet.
            metadata: Optional extra key-value data.

        Returns:
            The created ScoutListing, or None if duplicate URL.
        """
        # URL dedup check
        if url and self.get_by_url(url) is not None:
            logger.debug("Duplicate URL skipped: %s", url[:80])
            return None

        now = datetime.now(timezone.utc).isoformat()
        entry_id = str(uuid.uuid4())
        meta = metadata or {}

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO scout_listings
                (entry_id, title, price, url, source, location, date_found,
                 status, deal_score, score_reason, make, model, year, snippet,
                 created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id, title, price, url, source, location, now,
                ListingStatus.NEW.value, 0, "", make, model, year, snippet,
                now, now, json.dumps(meta),
            ),
        )
        self._conn.commit()

        entry = ScoutListing(
            entry_id=entry_id,
            title=title,
            price=price,
            url=url,
            source=source,
            location=location,
            date_found=now,
            status=ListingStatus.NEW.value,
            deal_score=0,
            score_reason="",
            make=make,
            model=model,
            year=year,
            snippet=snippet,
            created_at=now,
            updated_at=now,
            metadata=meta,
        )

        logger.info("Scout listing created: '%s' (%s)", title[:60], entry.short_id)
        return entry

    def update_entry(self, entry_id: str, **kwargs) -> ScoutListing | None:
        """Update fields on an existing scout listing.

        Allowed fields: title, price, url, source, location, status,
        deal_score, score_reason, make, model, year, snippet, metadata.

        Args:
            entry_id: The entry to update.
            **kwargs: Fields to change.

        Returns:
            The updated ScoutListing, or None if not found.
        """
        allowed = {
            "title", "price", "url", "source", "location", "status",
            "deal_score", "score_reason", "make", "model", "year",
            "snippet", "metadata",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_entry(entry_id)

        # Always update the timestamp
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Serialize JSON fields
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [entry_id]

        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE scout_listings SET {set_clause} WHERE entry_id = ?",
            values,
        )
        self._conn.commit()

        if cur.rowcount == 0:
            return None

        logger.info("Scout listing updated: %s", entry_id[:8])
        return self.get_entry(entry_id)

    def get_entry(self, entry_id: str) -> ScoutListing | None:
        """Return a single scout listing by ID, or None."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM scout_listings WHERE entry_id = ?", (entry_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return ScoutListing.from_row(row)

    def remove_entry(self, entry_id: str) -> bool:
        """Delete a scout listing by ID.

        Args:
            entry_id: The entry to delete.

        Returns:
            True if found and removed, False otherwise.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM scout_listings WHERE entry_id = ?", (entry_id,))
        self._conn.commit()

        if cur.rowcount > 0:
            logger.info("Scout listing removed: %s", entry_id[:8])
            return True
        return False

    def get_by_url(self, url: str) -> ScoutListing | None:
        """Return a scout listing by URL, or None."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM scout_listings WHERE url = ?", (url,))
        row = cur.fetchone()
        if row is None:
            return None
        return ScoutListing.from_row(row)

    def list_all(
        self,
        status: str | None = None,
        min_score: int = 0,
        limit: int = 50,
    ) -> list[ScoutListing]:
        """Return scout listings with optional filters.

        Args:
            status:    Filter by listing status.
            min_score: Minimum deal score.
            limit:     Maximum entries to return.

        Returns:
            List of ScoutListing objects, highest score first.
        """
        conditions = []
        params: list[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)

        if min_score > 0:
            conditions.append("deal_score >= ?")
            params.append(min_score)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        params.append(limit)

        cur = self._conn.cursor()
        cur.execute(
            f"SELECT * FROM scout_listings {where_clause} ORDER BY deal_score DESC, updated_at DESC LIMIT ?",
            params,
        )
        return [ScoutListing.from_row(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------
    # Dashboard helpers
    # -------------------------------------------------------------------

    def to_broadcast_list(self) -> list[dict]:
        """Return all entries as JSON-serializable dicts for the dashboard."""
        return [e.to_dict() for e in self.list_all()]

    def get_status(self) -> dict:
        """Return a snapshot of scout store state — counts by status."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT status, COUNT(*) as cnt FROM scout_listings GROUP BY status"
        )
        counts = {row["status"]: row["cnt"] for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) FROM scout_listings")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM scout_listings WHERE deal_score >= 8")
        hot_deals = cur.fetchone()[0]

        return {
            "total": total,
            "by_status": counts,
            "hot_deals": hot_deals,
            "db_path": self._db_path,
        }

    # -------------------------------------------------------------------
    # Close
    # -------------------------------------------------------------------

    def close(self):
        """Close the SQLite connection. Call on shutdown."""
        if self._conn:
            self._conn.close()
            logger.info("ScoutStore connection closed")


# ---------------------------------------------------------------------------
# DopplerScout — search + score engine
# ---------------------------------------------------------------------------

class DopplerScout:
    """Searches for motorcycle listings and scores deal quality.

    Uses DuckDuckGo site-scoped queries to find listings on Craigslist
    and Facebook Marketplace, then scores each with the local model router.

    Args:
        store:                ScoutStore instance for persistence.
        router:               ModelRouter for deal scoring (optional).
        criteria:             SearchCriteria for filtering (optional).
        web_tool:             WebTool instance (created internally if None).
        event_logger:         EventLogger for activity logging (optional).
        notification_manager: NotificationManager for hot deal alerts (optional).
        on_model_call:        Callback for model call tracking (optional).
    """

    SCORING_SYSTEM_PROMPT = (
        "You are a motorcycle deal evaluator. Given a listing title, price, "
        "and snippet, rate the deal quality on a scale of 1-10.\n\n"
        "Scoring guidelines:\n"
        "- 9-10: Exceptional deal — well below market, popular model, good condition indicators\n"
        "- 7-8: Good deal — fair price, desirable bike, no red flags\n"
        "- 5-6: Average — market price, nothing special but not bad\n"
        "- 3-4: Below average — overpriced, questionable condition, 'project' bikes\n"
        "- 1-2: Poor — major red flags, 'parts only', salvage title, suspicious\n\n"
        "Red flags: 'parts only', 'salvage', 'no title', 'needs work', 'project'\n"
        "Good signs: 'low miles', 'clean title', 'garage kept', 'service records'\n\n"
        "Respond with ONLY a JSON object: {\"score\": <1-10>, \"reason\": \"<brief explanation>\"}"
    )

    NOTIFICATION_THRESHOLD = 8

    def __init__(
        self,
        store: ScoutStore,
        router=None,
        criteria: SearchCriteria | None = None,
        web_tool=None,
        event_logger=None,
        notification_manager=None,
        on_model_call=None,
    ):
        self.store = store
        self.router = router
        self.criteria = criteria or SearchCriteria()
        self.event_logger = event_logger
        self.notification_manager = notification_manager
        self.on_model_call = on_model_call

        # Create WebTool internally if not provided
        if web_tool is not None:
            self.web = web_tool
        else:
            from tools.web import WebTool
            self.web = WebTool()

        logger.info("DopplerScout initialized (makes=%s)", self.criteria.makes)

    # -------------------------------------------------------------------
    # Query building
    # -------------------------------------------------------------------

    def _build_queries(self) -> list[tuple[str, str]]:
        """Generate DuckDuckGo search queries from criteria.

        Returns:
            List of (query_string, source_name) tuples.
        """
        queries = []
        location = self.criteria.location

        for make in self.criteria.makes:
            # Craigslist query
            cl_query = f"site:craigslist.org motorcycle {make} {location}"
            queries.append((cl_query, "craigslist"))

            # Facebook Marketplace query
            fb_query = f"site:facebook.com/marketplace motorcycle {make} {location}"
            queries.append((fb_query, "facebook"))

        return queries

    # -------------------------------------------------------------------
    # Listing parsing
    # -------------------------------------------------------------------

    def _parse_listing(self, result, source: str) -> dict:
        """Extract structured data from a search result.

        Parses price, year, and make/model from the title and snippet
        using regex patterns.

        Args:
            result: SearchResult from WebTool.
            source: Source platform name.

        Returns:
            Dict with parsed listing fields.
        """
        title = result.title or ""
        snippet = result.snippet or ""
        combined = f"{title} {snippet}"

        # Extract price: $1,234 or $1234
        price = 0
        price_match = re.search(r'\$[\d,]+', combined)
        if price_match:
            price_str = price_match.group().replace('$', '').replace(',', '')
            try:
                price = int(price_str)
            except ValueError:
                pass

        # Extract year: 4-digit year between 1950-2029
        year = 0
        year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', combined)
        if year_match:
            year = int(year_match.group())

        # Extract make from title (match against known makes)
        make = ""
        model = ""
        title_lower = title.lower()
        known_makes = [
            "honda", "yamaha", "suzuki", "kawasaki", "harley-davidson",
            "harley", "ducati", "bmw", "triumph", "indian", "ktm",
            "aprilia", "moto guzzi", "royal enfield",
        ]
        for m in known_makes:
            if m in title_lower:
                make = m
                if m == "harley":
                    make = "harley-davidson"
                break

        return {
            "title": title,
            "url": result.url,
            "source": source,
            "price": price,
            "year": year,
            "make": make,
            "model": model,
            "snippet": snippet,
        }

    # -------------------------------------------------------------------
    # Deal scoring
    # -------------------------------------------------------------------

    async def _score_listing(self, listing: ScoutListing) -> ScoutListing:
        """Score a listing's deal quality using the model router.

        Args:
            listing: The listing to score.

        Returns:
            The listing with deal_score and score_reason updated.
        """
        if self.router is None:
            # No router available — assign default score
            listing.deal_score = 5
            listing.score_reason = "No model available for scoring"
            return listing

        prompt = (
            f"Listing: {listing.title}\n"
            f"Price: ${listing.price}\n"
            f"Year: {listing.year or 'unknown'}\n"
            f"Make: {listing.make or 'unknown'}\n"
            f"Snippet: {listing.snippet[:200]}\n"
        )

        try:
            response = await self.router.route(
                prompt,
                task_complexity="tier1",
                system_prompt=self.SCORING_SYSTEM_PROMPT,
            )

            if self.on_model_call:
                self.on_model_call(response)

            # Parse JSON response — try json.loads first, then regex, then default
            score, reason = self._parse_score_response(response.text)
            listing.deal_score = score
            listing.score_reason = reason

        except Exception as e:
            logger.warning("Scoring failed for listing %s: %s", listing.short_id, e)
            listing.deal_score = 5
            listing.score_reason = f"Scoring error: {e}"

        return listing

    @staticmethod
    def _parse_score_response(text: str) -> tuple[int, str]:
        """Parse score and reason from model response text.

        Tries JSON parsing first, then regex extraction, then defaults.

        Args:
            text: Raw model response text.

        Returns:
            Tuple of (score, reason).
        """
        # Try JSON parsing
        try:
            data = json.loads(text.strip())
            score = int(data.get("score", 5))
            reason = str(data.get("reason", ""))
            return max(1, min(10, score)), reason
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Try extracting JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                score = int(data.get("score", 5))
                reason = str(data.get("reason", ""))
                return max(1, min(10, score)), reason
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Try regex extraction
        score_match = re.search(r'"score"\s*:\s*(\d+)', text)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        if score_match:
            score = int(score_match.group(1))
            reason = reason_match.group(1) if reason_match else ""
            return max(1, min(10, score)), reason

        # Default
        return 5, "Could not parse model response"

    # -------------------------------------------------------------------
    # Notifications
    # -------------------------------------------------------------------

    def _notify_deal(self, listing: ScoutListing):
        """Send notification for a hot deal (score >= threshold).

        Args:
            listing: The high-scoring listing.
        """
        if listing.deal_score < self.NOTIFICATION_THRESHOLD:
            return

        if self.event_logger:
            self.event_logger.info(
                "scout",
                f"Hot deal found: {listing.title[:60]} (score: {listing.deal_score})",
                url=listing.url,
                price=listing.price,
                score=listing.deal_score,
            )

        if self.notification_manager:
            self.notification_manager.emit(
                "info",
                "Hot Motorcycle Deal",
                f"Score {listing.deal_score}/10: {listing.title[:80]} — ${listing.price}",
                "scout",
            )

    # -------------------------------------------------------------------
    # Main scan
    # -------------------------------------------------------------------

    async def scan(self) -> list[ScoutListing]:
        """Run a full scan for motorcycle listings.

        Builds DuckDuckGo queries, searches, deduplicates by URL,
        stores new listings, scores each, and notifies on hot deals.

        Returns:
            List of newly found ScoutListing objects.
        """
        queries = self._build_queries()
        new_listings: list[ScoutListing] = []
        seen_urls: set[str] = set()

        if self.event_logger:
            self.event_logger.info(
                "scout",
                f"Scout scan started ({len(queries)} queries)",
                makes=self.criteria.makes,
                location=self.criteria.location,
            )

        for i, (query, source) in enumerate(queries):
            # Rate limiting: 1s delay between queries to avoid DDG throttling
            if i > 0:
                await asyncio.sleep(1.0)

            try:
                results = self.web.search(query, max_results=8)
            except Exception as e:
                logger.warning("Search failed for query '%s': %s", query[:60], e)
                continue

            for result in results:
                url = result.url
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                # Parse listing data from search result
                parsed = self._parse_listing(result, source)

                # Apply price filter
                if parsed["price"] > 0:
                    if self.criteria.price_min > 0 and parsed["price"] < self.criteria.price_min:
                        continue
                    if self.criteria.price_max > 0 and parsed["price"] > self.criteria.price_max:
                        continue

                # Apply exclude keywords filter
                combined_lower = f"{parsed['title']} {parsed['snippet']}".lower()
                if any(kw.lower() in combined_lower for kw in self.criteria.exclude_keywords):
                    continue

                # Store the listing (dedup by URL happens inside add_entry)
                entry = self.store.add_entry(
                    title=parsed["title"],
                    url=parsed["url"],
                    source=parsed["source"],
                    price=parsed["price"],
                    make=parsed["make"],
                    model=parsed["model"],
                    year=parsed["year"],
                    snippet=parsed["snippet"],
                    location=self.criteria.location,
                )
                if entry is not None:
                    new_listings.append(entry)

        # Score each new listing
        for listing in new_listings:
            listing = await self._score_listing(listing)
            self.store.update_entry(
                listing.entry_id,
                deal_score=listing.deal_score,
                score_reason=listing.score_reason,
            )
            self._notify_deal(listing)

        if self.event_logger:
            self.event_logger.info(
                "scout",
                f"Scout scan complete: {len(new_listings)} new listings",
                new_count=len(new_listings),
                total_queries=len(queries),
            )

        logger.info("Scout scan complete: %d new listings from %d queries",
                     len(new_listings), len(queries))

        return new_listings

    # -------------------------------------------------------------------
    # Re-score a single listing
    # -------------------------------------------------------------------

    async def score_single(self, entry_id: str) -> ScoutListing | None:
        """Re-score a specific listing.

        Args:
            entry_id: The listing entry ID.

        Returns:
            The updated ScoutListing, or None if not found.
        """
        listing = self.store.get_entry(entry_id)
        if listing is None:
            return None

        listing = await self._score_listing(listing)
        self.store.update_entry(
            listing.entry_id,
            deal_score=listing.deal_score,
            score_reason=listing.score_reason,
        )

        self._notify_deal(listing)
        return listing


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    try:
        store = ScoutStore(db_path=tmp.name)

        # Add entries
        e1 = store.add_entry(
            title="2019 Honda CB500F - Low Miles, Clean Title",
            url="https://portland.craigslist.org/mlt/mcd/1234567890.html",
            source="craigslist",
            price=4200,
            make="honda",
            model="CB500F",
            year=2019,
            snippet="Only 3,200 miles. Garage kept, service records available.",
        )
        e2 = store.add_entry(
            title="2015 Yamaha FZ-07 Project Bike",
            url="https://portland.craigslist.org/mlt/mcd/9876543210.html",
            source="craigslist",
            price=2800,
            make="yamaha",
            model="FZ-07",
            year=2015,
            snippet="Needs carb work. Runs but rough. No title.",
        )

        print(f"Created {e1.short_id}: {e1.title}")
        print(f"Created {e2.short_id}: {e2.title}")

        # Test dedup
        dup = store.add_entry(
            title="Duplicate listing",
            url="https://portland.craigslist.org/mlt/mcd/1234567890.html",
        )
        print(f"Duplicate returned None: {dup is None}")

        # Update
        updated = store.update_entry(
            e1.entry_id,
            deal_score=8,
            score_reason="Great price for a low-mileage CB500F",
            status=ListingStatus.FLAGGED.value,
        )
        print(f"Updated {updated.short_id}: score={updated.deal_score}, status={updated.status}")

        # List
        print(f"\nAll entries ({len(store.list_all())}):")
        for e in store.list_all():
            print(f"  [score={e.deal_score}] {e.title} ({e.short_id})")

        # List with min_score filter
        hot = store.list_all(min_score=8)
        print(f"\nHot deals (score >= 8): {len(hot)}")

        # Status
        print(f"\nStatus: {store.get_status()}")

        # Remove
        removed = store.remove_entry(e2.entry_id)
        print(f"\nRemoved {e2.short_id}: {removed}")
        print(f"Entries after removal: {len(store.list_all())}")

        store.close()

    finally:
        os.unlink(tmp.name)
        print(f"\nCleaned up temp file: {tmp.name}")
