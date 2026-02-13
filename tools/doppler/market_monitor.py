"""
Competitive Market Monitor for CAM / Doppler Cycles.

Tracks the local motorcycle service landscape around Gresham and Portland.
Monitors for competing mobile mechanics and shops via web search, stores
competitor profiles, and generates competitive analysis reports using the
model router.

Features:
  - Competitor CRUD with structured profiles (services, pricing, ratings)
  - Web scanning via DuckDuckGo (Craigslist, Google, Yelp) for discovery
  - Model-router-powered competitive analysis and gap identification
  - New-competitor alerts via NotificationManager
  - Pricing comparison matrix across service types
  - Knowledge base integration for long-term trend analysis

SQLite-backed, single-file module — same pattern as scout.py, training.py.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("cam.market_monitor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPETITOR_TYPES = (
    "mobile_mechanic",
    "independent_shop",
    "dealer_service",
    "specialty_shop",
    "chain_shop",
)

COMPETITOR_TYPE_LABELS = {
    "mobile_mechanic": "Mobile Mechanic",
    "independent_shop": "Independent Shop",
    "dealer_service": "Dealer Service",
    "specialty_shop": "Specialty Shop",
    "chain_shop": "Chain Shop",
}

SERVICE_CATEGORIES = (
    "diagnostics",
    "oil_change",
    "tire_service",
    "brake_service",
    "electrical",
    "engine_repair",
    "suspension",
    "general_maintenance",
    "custom_work",
    "inspection",
)

SCAN_SOURCES = ("craigslist", "google", "yelp")

# Portland metro area coverage zones
COVERAGE_ZONES = (
    "Gresham",
    "Portland",
    "Beaverton",
    "Hillsboro",
    "Lake Oswego",
    "Tigard",
    "Milwaukie",
    "Oregon City",
    "Clackamas",
    "Troutdale",
    "Vancouver WA",
)

# Analysis prompts for the model router
COMPETITIVE_ANALYSIS_PROMPT = """\
You are a business analyst specializing in the motorcycle service industry.
Analyze the competitive landscape for Doppler Cycles, a mobile motorcycle
diagnostics and repair service based in Gresham, Oregon.

Doppler's advantages:
- Mobile service (comes to the customer)
- 20+ years motorcycle industry experience, AMI certified
- Factory trained (Harley-Davidson, Yamaha, Ducati)
- AI-assisted diagnostics

Here are the current competitors in the Portland metro area:

{competitors_json}

Provide a concise analysis covering:
1. **Market Position** — Where Doppler stands relative to competitors
2. **Service Gaps** — Services not well-covered by existing competitors
3. **Pricing Opportunities** — Where Doppler can compete or differentiate
4. **Geographic Gaps** — Underserved areas in the metro
5. **Threats** — Competitors to watch closely
6. **Recommendations** — Top 3 actionable recommendations

Be specific and practical. This is for a solo operator bootstrapping a business.
Keep it under 600 words.
"""

GAP_ANALYSIS_PROMPT = """\
You are a market analyst. Given the competitor data below, identify specific
service gaps and underserved areas in the Portland/Gresham motorcycle service
market.

Competitors:
{competitors_json}

Coverage zones analyzed: {zones}

Return a JSON object with these keys:
- service_gaps: array of {{service, description, opportunity_level}} where
  opportunity_level is "high", "medium", or "low"
- geographic_gaps: array of {{area, description, competitor_count}}
- pricing_gaps: array of {{service, avg_price, opportunity}}

Return ONLY the JSON object — no markdown fences, no commentary.
"""


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Competitor:
    """A competitor business profile."""
    competitor_id: str = ""
    name: str = ""
    comp_type: str = "independent_shop"
    services: list = field(default_factory=list)
    pricing: dict = field(default_factory=dict)
    rating: float = 0.0
    review_count: int = 0
    location: str = ""
    coverage_area: str = ""
    source: str = "manual"
    source_url: str = ""
    phone: str = ""
    website: str = ""
    notes: str = ""
    status: str = "active"
    first_seen: str = ""
    last_seen: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def short_id(self) -> str:
        return self.competitor_id[:8] if self.competitor_id else ""

    @property
    def type_label(self) -> str:
        return COMPETITOR_TYPE_LABELS.get(self.comp_type, self.comp_type)

    def to_dict(self) -> dict:
        return {
            "competitor_id": self.competitor_id,
            "short_id": self.short_id,
            "name": self.name,
            "comp_type": self.comp_type,
            "type_label": self.type_label,
            "services": self.services,
            "pricing": self.pricing,
            "rating": self.rating,
            "review_count": self.review_count,
            "location": self.location,
            "coverage_area": self.coverage_area,
            "source": self.source,
            "source_url": self.source_url,
            "phone": self.phone,
            "website": self.website,
            "notes": self.notes,
            "status": self.status,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_row(row) -> "Competitor":
        """Build a Competitor from a sqlite3.Row."""
        r = dict(row)
        return Competitor(
            competitor_id=r["competitor_id"],
            name=r.get("name", ""),
            comp_type=r.get("comp_type", "independent_shop"),
            services=json.loads(r.get("services", "[]")),
            pricing=json.loads(r.get("pricing", "{}")),
            rating=float(r.get("rating", 0)),
            review_count=int(r.get("review_count", 0)),
            location=r.get("location", ""),
            coverage_area=r.get("coverage_area", ""),
            source=r.get("source", "manual"),
            source_url=r.get("source_url", ""),
            phone=r.get("phone", ""),
            website=r.get("website", ""),
            notes=r.get("notes", ""),
            status=r.get("status", "active"),
            first_seen=r.get("first_seen", ""),
            last_seen=r.get("last_seen", ""),
            created_at=r.get("created_at", ""),
            updated_at=r.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# MarketMonitor
# ---------------------------------------------------------------------------

class MarketMonitor:
    """SQLite-backed competitive market monitor for Doppler Cycles.

    Tracks competing motorcycle service businesses in the Portland metro area,
    generates competitive analysis via the model router, and alerts on new
    competitor discoveries.

    Args:
        db_path:                Path to the SQLite database file.
        router:                 ModelRouter for analysis generation (optional).
        web_tool:               WebTool instance for scanning (optional).
        notification_manager:   NotificationManager for alerts (optional).
        long_term_memory:       LongTermMemory for storing insights (optional).
        on_change:              Async callback on state mutations.
        on_model_call:          Callback for model usage tracking.
    """

    def __init__(
        self,
        db_path: str = "data/market_monitor.db",
        router: Any = None,
        web_tool: Any = None,
        notification_manager: Any = None,
        long_term_memory: Any = None,
        on_change: Optional[Callable[[], Coroutine]] = None,
        on_model_call: Optional[Callable] = None,
    ):
        self._db_path = db_path
        self._router = router
        self._web_tool = web_tool
        self._notification_mgr = notification_manager
        self._ltm = long_term_memory
        self._on_change = on_change
        self._on_model_call = on_model_call

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("MarketMonitor initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        """Create the competitors and scans tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS competitors (
                competitor_id  TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                comp_type      TEXT DEFAULT 'independent_shop',
                services       TEXT DEFAULT '[]',
                pricing        TEXT DEFAULT '{}',
                rating         REAL DEFAULT 0,
                review_count   INTEGER DEFAULT 0,
                location       TEXT DEFAULT '',
                coverage_area  TEXT DEFAULT '',
                source         TEXT DEFAULT 'manual',
                source_url     TEXT DEFAULT '',
                phone          TEXT DEFAULT '',
                website        TEXT DEFAULT '',
                notes          TEXT DEFAULT '',
                status         TEXT DEFAULT 'active',
                first_seen     TEXT NOT NULL,
                last_seen      TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_comp_status ON competitors(status);
            CREATE INDEX IF NOT EXISTS idx_comp_type ON competitors(comp_type);
            CREATE INDEX IF NOT EXISTS idx_comp_location ON competitors(location);

            CREATE TABLE IF NOT EXISTS market_scans (
                scan_id        TEXT PRIMARY KEY,
                source         TEXT NOT NULL,
                query          TEXT DEFAULT '',
                results_count  INTEGER DEFAULT 0,
                new_found      INTEGER DEFAULT 0,
                scan_data      TEXT DEFAULT '{}',
                created_at     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_scan_date ON market_scans(created_at);

            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id    TEXT PRIMARY KEY,
                analysis_type  TEXT DEFAULT 'competitive',
                title          TEXT DEFAULT '',
                content        TEXT DEFAULT '',
                data           TEXT DEFAULT '{}',
                created_at     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_analysis_type ON analyses(analysis_type);
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

    # ------------------------------------------------------------------
    # Competitor CRUD
    # ------------------------------------------------------------------

    def add_competitor(
        self,
        name: str,
        comp_type: str = "independent_shop",
        services: list[str] | None = None,
        pricing: dict | None = None,
        rating: float = 0.0,
        review_count: int = 0,
        location: str = "",
        coverage_area: str = "",
        source: str = "manual",
        source_url: str = "",
        phone: str = "",
        website: str = "",
        notes: str = "",
    ) -> Competitor:
        """Add a new competitor to the database.

        Args:
            name:           Business name.
            comp_type:      One of COMPETITOR_TYPES.
            services:       List of service categories offered.
            pricing:        Dict of service → price/range strings.
            rating:         Review rating (0-5 stars).
            review_count:   Number of reviews.
            location:       Business location/address.
            coverage_area:  Geographic coverage description.
            source:         Where discovered (manual, craigslist, google, yelp).
            source_url:     URL of the listing/profile.
            phone:          Phone number.
            website:        Business website.
            notes:          Any additional notes.

        Returns:
            The newly created Competitor.
        """
        if comp_type not in COMPETITOR_TYPES:
            comp_type = "independent_shop"

        now = self._now()
        comp = Competitor(
            competitor_id=str(uuid.uuid4()),
            name=name,
            comp_type=comp_type,
            services=services or [],
            pricing=pricing or {},
            rating=rating,
            review_count=review_count,
            location=location,
            coverage_area=coverage_area,
            source=source,
            source_url=source_url,
            phone=phone,
            website=website,
            notes=notes,
            status="active",
            first_seen=now,
            last_seen=now,
            created_at=now,
            updated_at=now,
        )

        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO competitors
               (competitor_id, name, comp_type, services, pricing, rating,
                review_count, location, coverage_area, source, source_url,
                phone, website, notes, status, first_seen, last_seen,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                comp.competitor_id, comp.name, comp.comp_type,
                json.dumps(comp.services), json.dumps(comp.pricing),
                comp.rating, comp.review_count, comp.location,
                comp.coverage_area, comp.source, comp.source_url,
                comp.phone, comp.website, comp.notes, comp.status,
                comp.first_seen, comp.last_seen, comp.created_at,
                comp.updated_at,
            ),
        )
        self._conn.commit()
        logger.info("Competitor added: %s — %s (%s)",
                     comp.short_id, comp.name, comp.type_label)
        self._fire_change()
        return comp

    def update_competitor(self, competitor_id: str, **kwargs) -> Optional[Competitor]:
        """Update fields on an existing competitor.

        Args:
            competitor_id:  The competitor UUID.
            **kwargs:       Fields to update.

        Returns:
            Updated Competitor, or None if not found.
        """
        existing = self.get_competitor(competitor_id)
        if not existing:
            return None

        allowed = {
            "name", "comp_type", "services", "pricing", "rating",
            "review_count", "location", "coverage_area", "source",
            "source_url", "phone", "website", "notes", "status",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return existing

        if "comp_type" in updates and updates["comp_type"] not in COMPETITOR_TYPES:
            updates["comp_type"] = "independent_shop"

        # Serialize JSON fields
        if "services" in updates:
            updates["services"] = json.dumps(updates["services"])
        if "pricing" in updates:
            updates["pricing"] = json.dumps(updates["pricing"])

        updates["updated_at"] = self._now()
        updates["last_seen"] = self._now()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [competitor_id]

        cur = self._conn.cursor()
        cur.execute(f"UPDATE competitors SET {set_clause} WHERE competitor_id = ?", values)
        self._conn.commit()

        logger.info("Competitor updated: %s", competitor_id[:8])
        self._fire_change()
        return self.get_competitor(competitor_id)

    def delete_competitor(self, competitor_id: str) -> bool:
        """Delete a competitor entry.

        Returns:
            True if deleted, False if not found.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM competitors WHERE competitor_id = ?", (competitor_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Competitor deleted: %s", competitor_id[:8])
            self._fire_change()
        return deleted

    def get_competitor(self, competitor_id: str) -> Optional[Competitor]:
        """Get a single competitor by ID."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM competitors WHERE competitor_id = ?", (competitor_id,))
        row = cur.fetchone()
        return Competitor.from_row(row) if row else None

    def list_competitors(
        self,
        status: str = "",
        comp_type: str = "",
        location: str = "",
        limit: int = 50,
    ) -> list[Competitor]:
        """List competitors with optional filters.

        Args:
            status:     Filter by status (active, closed, unverified).
            comp_type:  Filter by competitor type.
            location:   Filter by location (substring match).
            limit:      Max results.

        Returns:
            List of Competitor objects, newest first.
        """
        query = "SELECT * FROM competitors WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if comp_type:
            query += " AND comp_type = ?"
            params.append(comp_type)
        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        cur = self._conn.cursor()
        cur.execute(query, params)
        return [Competitor.from_row(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Web scanning for competitor discovery
    # ------------------------------------------------------------------

    async def scan_for_competitors(self, source: str = "google") -> dict:
        """Scan web sources for competing motorcycle service businesses.

        Uses DuckDuckGo site-scoped queries to search Craigslist, Google
        Business listings, and Yelp for motorcycle mechanics in the Portland
        metro area.

        Args:
            source: Which source to scan (google, craigslist, yelp).

        Returns:
            Dict with scan results: results_count, new_found, matches.
        """
        if not self._web_tool:
            from tools.web import WebTool
            self._web_tool = WebTool()

        queries = self._build_scan_queries(source)
        all_matches = []
        new_found = 0

        for query_str in queries:
            results = self._web_tool.search(query_str, max_results=10)
            for result in results:
                match = self._parse_service_listing(result, source)
                if match and not self._is_duplicate(match["name"]):
                    all_matches.append(match)

        # Add new competitors found
        for match in all_matches:
            existing = self._find_by_name(match["name"])
            if not existing:
                self.add_competitor(
                    name=match["name"],
                    comp_type=match.get("comp_type", "independent_shop"),
                    location=match.get("location", "Portland OR"),
                    source=source,
                    source_url=match.get("url", ""),
                    notes=f"Auto-discovered via {source} scan",
                )
                new_found += 1
                self.alert_new_competitor(match["name"], source)

        # Record the scan
        scan_id = str(uuid.uuid4())
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO market_scans
               (scan_id, source, query, results_count, new_found, scan_data, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                scan_id, source,
                "; ".join(queries),
                len(all_matches), new_found,
                json.dumps([m.get("name", "") for m in all_matches]),
                self._now(),
            ),
        )
        self._conn.commit()

        logger.info("Market scan (%s): %d results, %d new competitors",
                     source, len(all_matches), new_found)
        self._fire_change()

        return {
            "scan_id": scan_id,
            "source": source,
            "results_count": len(all_matches),
            "new_found": new_found,
            "matches": all_matches,
        }

    def _build_scan_queries(self, source: str) -> list[str]:
        """Build search queries for competitor discovery."""
        queries = []
        base_terms = [
            "motorcycle mechanic",
            "mobile motorcycle repair",
            "motorcycle diagnostics",
            "motorcycle service",
        ]

        if source == "craigslist":
            for term in base_terms:
                queries.append(f"site:craigslist.org {term} Portland OR Gresham")
        elif source == "yelp":
            for term in base_terms[:2]:  # Fewer queries for Yelp
                queries.append(f"site:yelp.com {term} Portland Oregon")
        else:  # google / general
            for term in base_terms:
                queries.append(f"{term} Portland OR Gresham Oregon")

        return queries

    def _parse_service_listing(self, result, source: str) -> Optional[dict]:
        """Extract competitor info from a search result."""
        title = result.title or ""
        snippet = result.snippet or ""
        url = result.url or ""

        # Skip irrelevant results
        skip_terms = ("parts", "for sale", "helmet", "gear", "accessory",
                      "insurance", "training", "school", "lesson")
        combined_lower = f"{title} {snippet}".lower()
        if any(term in combined_lower for term in skip_terms):
            return None

        # Must contain service-related terms
        service_terms = ("mechanic", "repair", "service", "diagnostic",
                         "maintenance", "shop", "garage", "mobile")
        if not any(term in combined_lower for term in service_terms):
            return None

        # Determine type from content
        comp_type = "independent_shop"
        if "mobile" in combined_lower:
            comp_type = "mobile_mechanic"
        elif "dealer" in combined_lower:
            comp_type = "dealer_service"
        elif any(chain in combined_lower for chain in
                 ("cycle gear", "revzilla", "jiffy", "valvoline")):
            comp_type = "chain_shop"

        # Clean up the name (use title, strip common suffixes)
        name = title.split(" - ")[0].split(" | ")[0].strip()
        if len(name) > 80:
            name = name[:80]

        return {
            "name": name,
            "comp_type": comp_type,
            "url": url,
            "snippet": snippet[:200],
            "location": "Portland OR",
            "source": source,
        }

    def _is_duplicate(self, name: str) -> bool:
        """Check if a competitor name is already tracked (fuzzy match)."""
        return self._find_by_name(name) is not None

    def _find_by_name(self, name: str) -> Optional[Competitor]:
        """Find a competitor by name (case-insensitive)."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM competitors WHERE LOWER(name) = LOWER(?)",
            (name.strip(),),
        )
        row = cur.fetchone()
        return Competitor.from_row(row) if row else None

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def alert_new_competitor(self, name: str, source: str):
        """Send a notification when a new competitor is discovered.

        Args:
            name:   Competitor business name.
            source: Where they were found.
        """
        if self._notification_mgr:
            self._notification_mgr.emit(
                "info",
                "New Competitor Found",
                f"Discovered via {source}: {name}",
                "market_monitor",
            )
        logger.info("New competitor alert: %s (via %s)", name, source)

    # ------------------------------------------------------------------
    # Pricing comparison
    # ------------------------------------------------------------------

    def pricing_matrix(self) -> dict:
        """Build a pricing comparison matrix across competitors.

        Returns:
            Dict with service_types (rows), competitors (columns), and
            a matrix of prices. Also includes Doppler's estimated position.
        """
        competitors = self.list_competitors(status="active")
        if not competitors:
            return {"services": [], "competitors": [], "matrix": {}, "summary": {}}

        # Collect all service types and build matrix
        all_services: set[str] = set()
        for comp in competitors:
            all_services.update(comp.pricing.keys())

        services = sorted(all_services)
        matrix: dict[str, dict[str, str]] = {}

        for service in services:
            matrix[service] = {}
            for comp in competitors:
                matrix[service][comp.name] = comp.pricing.get(service, "—")

        # Summary: average pricing per service where data exists
        summary: dict[str, dict] = {}
        for service in services:
            prices = []
            for comp in competitors:
                price_str = comp.pricing.get(service, "")
                if price_str:
                    # Try to extract numeric values
                    import re
                    nums = re.findall(r'\d+', str(price_str))
                    prices.extend(int(n) for n in nums if int(n) > 0)
            if prices:
                summary[service] = {
                    "avg": round(sum(prices) / len(prices)),
                    "min": min(prices),
                    "max": max(prices),
                    "data_points": len(prices),
                }

        return {
            "services": services,
            "competitors": [c.name for c in competitors],
            "matrix": matrix,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Analysis (model-router powered)
    # ------------------------------------------------------------------

    async def generate_analysis(self, analysis_type: str = "competitive") -> dict:
        """Generate a competitive analysis report using the model router.

        Args:
            analysis_type: Type of analysis (competitive, gaps, pricing, full).

        Returns:
            Dict with analysis_id, title, content, data.
        """
        competitors = self.list_competitors(status="active")
        comp_data = [c.to_dict() for c in competitors]
        now = self._now()
        analysis_id = str(uuid.uuid4())

        content = ""
        analysis_data = {}

        if analysis_type == "gaps":
            content, analysis_data = await self._analyze_gaps(comp_data)
        elif analysis_type == "pricing":
            pricing = self.pricing_matrix()
            content = self._format_pricing_report(pricing)
            analysis_data = pricing
        elif analysis_type == "full":
            # Run competitive + gaps + pricing
            comp_content = await self._analyze_competitive(comp_data)
            _, gap_data = await self._analyze_gaps(comp_data)
            pricing = self.pricing_matrix()
            content = comp_content + "\n\n---\n\n" + self._format_pricing_report(pricing)
            analysis_data = {"gaps": gap_data, "pricing": pricing}
        else:
            # Default: competitive analysis
            content = await self._analyze_competitive(comp_data)

        title = f"{analysis_type.title()} Analysis — {now[:10]}"

        # Store analysis
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO analyses
               (analysis_id, analysis_type, title, content, data, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (analysis_id, analysis_type, title, content,
             json.dumps(analysis_data, default=str), now),
        )
        self._conn.commit()

        # Store insight in long-term memory if available
        if self._ltm and content:
            try:
                self._ltm.store(
                    content=f"Market Analysis ({analysis_type}): {content[:500]}",
                    metadata={
                        "source": "market_monitor",
                        "type": f"analysis_{analysis_type}",
                        "date": now[:10],
                        "competitor_count": len(competitors),
                    },
                )
            except Exception as exc:
                logger.warning("Failed to store analysis in LTM: %s", exc)

        logger.info("Analysis generated: %s (%s)", analysis_id[:8], analysis_type)
        self._fire_change()

        return {
            "analysis_id": analysis_id,
            "analysis_type": analysis_type,
            "title": title,
            "content": content,
            "data": analysis_data,
            "created_at": now,
        }

    async def _analyze_competitive(self, comp_data: list[dict]) -> str:
        """Run competitive analysis via model router."""
        prompt = COMPETITIVE_ANALYSIS_PROMPT.format(
            competitors_json=json.dumps(comp_data, indent=2)
        )

        if self._router:
            try:
                if self._on_model_call:
                    self._on_model_call("market_analysis", "simple")
                response = await self._router.route(
                    prompt=prompt,
                    task_complexity="simple",
                    system_prompt="You are a motorcycle industry business analyst.",
                )
                return response.text
            except Exception as e:
                logger.warning("Model analysis failed: %s", e)

        # Fallback: basic text analysis
        return self._fallback_analysis(comp_data)

    async def _analyze_gaps(self, comp_data: list[dict]) -> tuple[str, dict]:
        """Run gap analysis via model router."""
        prompt = GAP_ANALYSIS_PROMPT.format(
            competitors_json=json.dumps(comp_data, indent=2),
            zones=", ".join(COVERAGE_ZONES),
        )

        gap_data = {"service_gaps": [], "geographic_gaps": [], "pricing_gaps": []}

        if self._router:
            try:
                if self._on_model_call:
                    self._on_model_call("gap_analysis", "simple")
                response = await self._router.route(
                    prompt=prompt,
                    task_complexity="simple",
                )
                # Try to parse JSON from response
                text = response.text.strip()
                # Strip markdown fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text
                    if text.endswith("```"):
                        text = text[:-3]
                try:
                    gap_data = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("Gap analysis returned non-JSON, using raw text")
                    return text, gap_data
            except Exception as e:
                logger.warning("Model gap analysis failed: %s", e)

        # Format gap data as readable text
        lines = ["## Service & Market Gap Analysis\n"]
        for gap in gap_data.get("service_gaps", []):
            level = gap.get("opportunity_level", "medium")
            icon = {"high": "!!!", "medium": "!!", "low": "!"}.get(level, "!")
            lines.append(f"- **{gap.get('service', '?')}** [{icon}] — {gap.get('description', '')}")

        if gap_data.get("geographic_gaps"):
            lines.append("\n### Geographic Gaps\n")
            for gap in gap_data["geographic_gaps"]:
                lines.append(f"- **{gap.get('area', '?')}** — {gap.get('description', '')} "
                           f"({gap.get('competitor_count', 0)} competitors)")

        return "\n".join(lines), gap_data

    def _fallback_analysis(self, comp_data: list[dict]) -> str:
        """Generate basic analysis without model router."""
        lines = ["# Competitive Landscape — Portland Metro\n"]

        # Count by type
        type_counts: dict[str, int] = {}
        for c in comp_data:
            t = c.get("comp_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        lines.append("## Competitor Breakdown\n")
        for t, count in sorted(type_counts.items()):
            label = COMPETITOR_TYPE_LABELS.get(t, t)
            lines.append(f"- **{label}**: {count}")

        # Mobile mechanic count (direct competitors)
        mobile = sum(1 for c in comp_data if c.get("comp_type") == "mobile_mechanic")
        lines.append(f"\n**Direct competitors (mobile mechanics):** {mobile}")
        lines.append(f"**Total tracked:** {len(comp_data)}")

        # Location distribution
        locations: dict[str, int] = {}
        for c in comp_data:
            loc = c.get("location", "Unknown")
            locations[loc] = locations.get(loc, 0) + 1

        if locations:
            lines.append("\n## Geographic Distribution\n")
            for loc, count in sorted(locations.items(), key=lambda x: -x[1]):
                lines.append(f"- **{loc}**: {count}")

        lines.append("\n*For detailed analysis, configure the model router.*")
        return "\n".join(lines)

    def _format_pricing_report(self, pricing_data: dict) -> str:
        """Format pricing matrix as readable markdown."""
        lines = ["## Pricing Comparison\n"]
        summary = pricing_data.get("summary", {})

        if not summary:
            lines.append("*No pricing data available yet.*")
            return "\n".join(lines)

        for service, stats in sorted(summary.items()):
            lines.append(f"- **{service}**: avg ${stats['avg']}, "
                        f"range ${stats['min']}–${stats['max']} "
                        f"({stats['data_points']} data points)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Analysis retrieval
    # ------------------------------------------------------------------

    def get_analysis(self, analysis_id: str) -> Optional[dict]:
        """Get a specific analysis by ID."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,))
        row = cur.fetchone()
        if not row:
            return None
        r = dict(row)
        return {
            "analysis_id": r["analysis_id"],
            "analysis_type": r["analysis_type"],
            "title": r["title"],
            "content": r["content"],
            "data": json.loads(r.get("data", "{}")),
            "created_at": r["created_at"],
        }

    def list_analyses(self, limit: int = 20) -> list[dict]:
        """List recent analyses."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT analysis_id, analysis_type, title, created_at
               FROM analyses ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_latest_analysis(self) -> Optional[dict]:
        """Get the most recent analysis."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        r = dict(row)
        return {
            "analysis_id": r["analysis_id"],
            "analysis_type": r["analysis_type"],
            "title": r["title"],
            "content": r["content"],
            "data": json.loads(r.get("data", "{}")),
            "created_at": r["created_at"],
        }

    # ------------------------------------------------------------------
    # Scan history
    # ------------------------------------------------------------------

    def list_scans(self, limit: int = 20) -> list[dict]:
        """List recent market scans."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT scan_id, source, query, results_count, new_found, created_at
               FROM market_scans ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Summary status for dashboard header cards."""
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) FROM competitors WHERE status = 'active'")
        active_competitors = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM competitors WHERE comp_type = 'mobile_mechanic' AND status = 'active'")
        mobile_competitors = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT location) FROM competitors WHERE status = 'active'")
        areas_covered = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM market_scans")
        total_scans = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM analyses")
        total_analyses = cur.fetchone()[0]

        # Last scan date
        cur.execute("SELECT MAX(created_at) FROM market_scans")
        row = cur.fetchone()
        last_scan = row[0] if row and row[0] else ""

        # Average competitor rating
        cur.execute(
            "SELECT AVG(rating) FROM competitors WHERE status = 'active' AND rating > 0"
        )
        row = cur.fetchone()
        avg_rating = round(row[0], 1) if row and row[0] else 0

        return {
            "active_competitors": active_competitors,
            "mobile_competitors": mobile_competitors,
            "areas_covered": areas_covered,
            "total_scans": total_scans,
            "total_analyses": total_analyses,
            "last_scan": last_scan,
            "avg_rating": avg_rating,
        }

    def to_broadcast_dict(self) -> dict:
        """Full state dict for broadcasting to the dashboard."""
        competitors = self.list_competitors(limit=50)
        return {
            "competitors": [c.to_dict() for c in competitors],
            "status": self.get_status(),
            "pricing_matrix": self.pricing_matrix(),
            "recent_scans": self.list_scans(limit=10),
            "latest_analysis": self.get_latest_analysis(),
        }

    def close(self):
        """Close the SQLite connection. Call on shutdown."""
        try:
            self._conn.close()
            logger.info("MarketMonitor closed")
        except Exception:
            pass
