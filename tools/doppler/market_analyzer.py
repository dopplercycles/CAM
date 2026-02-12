"""
Doppler Market Analyzer — Price Tracking & Trend Analysis

Captures daily price snapshots from scout listings, computes trends
(rising/falling/stable), detects undervalued listings vs historical
averages, and generates downloadable market reports via the model router.

Uses a separate SQLite database (market_analytics.db) to keep market
data independent of the scout store.

Usage:
    from tools.doppler.market_analyzer import MarketAnalyzer

    analyzer = MarketAnalyzer(scout_store=store)
    count = analyzer.take_snapshot()
    trends = analyzer.get_all_trends(days=30)
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Coroutine


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("cam.market_analyzer")


# ---------------------------------------------------------------------------
# MarketAnalyzer — SQLite-backed price tracking and trend analysis
# ---------------------------------------------------------------------------

class MarketAnalyzer:
    """Tracks motorcycle market prices over time and generates analytics.

    Reads current listings from the ScoutStore, aggregates daily snapshots
    by make/model/location, computes price trends, and detects undervalued
    listings.  Reports are generated via the model router using local models
    (task_complexity="simple") for zero API cost.

    Args:
        db_path:       Path to the SQLite database file.
        scout_store:   A ScoutStore instance to read current listings from.
        router:        ModelRouter for generating reports (can be set later).
        on_change:     Async callback fired after data mutations.
        on_model_call: Callback for tracking model usage in analytics.
    """

    def __init__(
        self,
        db_path: str = "data/market_analytics.db",
        scout_store: Any = None,
        router: Any = None,
        on_change: Callable[[], Coroutine] | None = None,
        on_model_call: Callable | None = None,
    ):
        self._db_path = db_path
        self._scout_store = scout_store
        self.router = router
        self._on_change = on_change
        self._on_model_call = on_model_call

        # Ensure parent directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_file), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        logger.info("MarketAnalyzer initialized (db=%s)", db_file)

    # -------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        cur = self._conn.cursor()

        # Daily price observations per make/model/location
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT NOT NULL,
                make TEXT NOT NULL,
                model TEXT NOT NULL,
                avg_price REAL NOT NULL DEFAULT 0,
                min_price INTEGER NOT NULL DEFAULT 0,
                max_price INTEGER NOT NULL DEFAULT 0,
                listing_count INTEGER NOT NULL DEFAULT 0,
                location TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_unique
            ON market_snapshots(snapshot_date, make, model, location)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshot_make_model
            ON market_snapshots(make, model)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshot_date
            ON market_snapshots(snapshot_date)
        """)

        # Generated market reports
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT UNIQUE NOT NULL,
                report_date TEXT NOT NULL,
                report_type TEXT NOT NULL DEFAULT 'weekly',
                title TEXT NOT NULL DEFAULT '',
                summary_text TEXT NOT NULL DEFAULT '',
                data_json TEXT,
                created_at TEXT NOT NULL
            )
        """)

        self._conn.commit()
        logger.info("MarketAnalyzer schema initialized")

    # -------------------------------------------------------------------
    # Snapshot — capture current prices from scout listings
    # -------------------------------------------------------------------

    def take_snapshot(self) -> int:
        """Group current scout listings by make/model/location and store daily averages.

        Uses INSERT OR IGNORE so it's safe to call multiple times per day —
        only the first snapshot for each make/model/location per day is kept.

        Returns:
            Number of new snapshot rows inserted.
        """
        if not self._scout_store:
            logger.warning("No scout_store configured — cannot take snapshot")
            return 0

        listings = self._scout_store.list_all()
        if not listings:
            logger.info("No scout listings to snapshot")
            return 0

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now = datetime.now(timezone.utc).isoformat()

        # Group listings by (make, location) — aggregate across models within make
        groups: dict[tuple[str, str, str], list[int]] = {}
        for entry in listings:
            make = (entry.make or "unknown").lower().strip()
            model = (entry.model or "unknown").lower().strip()
            location = (entry.location or "").strip()
            if entry.price > 0:
                key = (make, model, location)
                groups.setdefault(key, []).append(entry.price)

        if not groups:
            logger.info("No listings with prices to snapshot")
            return 0

        cur = self._conn.cursor()
        inserted = 0

        for (make, model, location), prices in groups.items():
            avg_price = round(sum(prices) / len(prices), 2)
            min_price = min(prices)
            max_price = max(prices)
            count = len(prices)

            cur.execute("""
                INSERT OR IGNORE INTO market_snapshots
                (snapshot_date, make, model, avg_price, min_price, max_price,
                 listing_count, location, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (today, make, model, avg_price, min_price, max_price,
                  count, location, now))

            if cur.rowcount > 0:
                inserted += 1

        self._conn.commit()
        logger.info("Snapshot taken: %d new rows for %s", inserted, today)
        return inserted

    # -------------------------------------------------------------------
    # Price history and trends
    # -------------------------------------------------------------------

    def get_price_history(
        self,
        make: str,
        model: str = "",
        days: int = 30,
        location: str = "",
    ) -> list[dict]:
        """Return daily snapshots for a make/model, date ascending.

        Args:
            make:     Motorcycle make (case-insensitive).
            model:    Specific model (empty = aggregate all models for make).
            days:     How many days of history to return.
            location: Filter by location (empty = all locations).

        Returns:
            List of snapshot dicts with date, avg_price, min, max, count.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        cur = self._conn.cursor()

        if model:
            if location:
                cur.execute("""
                    SELECT snapshot_date, avg_price, min_price, max_price,
                           listing_count, location
                    FROM market_snapshots
                    WHERE LOWER(make) = LOWER(?) AND LOWER(model) = LOWER(?)
                      AND location = ? AND snapshot_date >= ?
                    ORDER BY snapshot_date ASC
                """, (make, model, location, cutoff))
            else:
                cur.execute("""
                    SELECT snapshot_date, avg_price, min_price, max_price,
                           listing_count, location
                    FROM market_snapshots
                    WHERE LOWER(make) = LOWER(?) AND LOWER(model) = LOWER(?)
                      AND snapshot_date >= ?
                    ORDER BY snapshot_date ASC
                """, (make, model, cutoff))
        else:
            if location:
                cur.execute("""
                    SELECT snapshot_date,
                           ROUND(AVG(avg_price), 2) as avg_price,
                           MIN(min_price) as min_price,
                           MAX(max_price) as max_price,
                           SUM(listing_count) as listing_count,
                           ? as location
                    FROM market_snapshots
                    WHERE LOWER(make) = LOWER(?) AND location = ?
                      AND snapshot_date >= ?
                    GROUP BY snapshot_date
                    ORDER BY snapshot_date ASC
                """, (location, make, location, cutoff))
            else:
                cur.execute("""
                    SELECT snapshot_date,
                           ROUND(AVG(avg_price), 2) as avg_price,
                           MIN(min_price) as min_price,
                           MAX(max_price) as max_price,
                           SUM(listing_count) as listing_count,
                           '' as location
                    FROM market_snapshots
                    WHERE LOWER(make) = LOWER(?) AND snapshot_date >= ?
                    GROUP BY snapshot_date
                    ORDER BY snapshot_date ASC
                """, (make, cutoff))

        return [dict(row) for row in cur.fetchall()]

    def get_trend(self, make: str, model: str = "", days: int = 30) -> dict:
        """Compare recent 7-day avg vs older avg to determine trend direction.

        Uses a ±5% threshold to distinguish rising/falling from stable.

        Args:
            make:  Motorcycle make.
            model: Specific model (empty = aggregate).
            days:  Total window to analyze.

        Returns:
            Dict with make, model, direction (rising/falling/stable),
            change_pct, recent_avg, older_avg, data_points.
        """
        history = self.get_price_history(make, model, days)

        if len(history) < 2:
            return {
                "make": make,
                "model": model,
                "direction": "stable",
                "change_pct": 0.0,
                "recent_avg": history[0]["avg_price"] if history else 0,
                "older_avg": history[0]["avg_price"] if history else 0,
                "data_points": len(history),
            }

        # Split into recent (last 7 days) and older
        recent_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        recent = [h for h in history if h["snapshot_date"] >= recent_cutoff]
        older = [h for h in history if h["snapshot_date"] < recent_cutoff]

        # If all data is in the recent window, split in half
        if not older:
            mid = len(history) // 2
            older = history[:mid]
            recent = history[mid:]

        if not recent or not older:
            avg = history[-1]["avg_price"] if history else 0
            return {
                "make": make,
                "model": model,
                "direction": "stable",
                "change_pct": 0.0,
                "recent_avg": avg,
                "older_avg": avg,
                "data_points": len(history),
            }

        recent_avg = sum(h["avg_price"] for h in recent) / len(recent)
        older_avg = sum(h["avg_price"] for h in older) / len(older)

        if older_avg > 0:
            change_pct = round(((recent_avg - older_avg) / older_avg) * 100, 1)
        else:
            change_pct = 0.0

        if change_pct > 5:
            direction = "rising"
        elif change_pct < -5:
            direction = "falling"
        else:
            direction = "stable"

        return {
            "make": make,
            "model": model,
            "direction": direction,
            "change_pct": change_pct,
            "recent_avg": round(recent_avg, 2),
            "older_avg": round(older_avg, 2),
            "data_points": len(history),
        }

    def get_all_trends(self, days: int = 30) -> list[dict]:
        """Get trends for all tracked make/model combos, sorted by absolute change.

        Returns:
            List of trend dicts, sorted by |change_pct| descending.
        """
        cur = self._conn.cursor()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        cur.execute("""
            SELECT DISTINCT make, model FROM market_snapshots
            WHERE snapshot_date >= ?
            ORDER BY make, model
        """, (cutoff,))

        combos = [(row["make"], row["model"]) for row in cur.fetchall()]
        trends = [self.get_trend(make, model, days) for make, model in combos]
        trends.sort(key=lambda t: abs(t["change_pct"]), reverse=True)
        return trends

    # -------------------------------------------------------------------
    # Undervalued listing detection
    # -------------------------------------------------------------------

    def detect_undervalued(self, threshold_pct: float = 20.0) -> list[dict]:
        """Find current listings priced significantly below 30-day average for their make.

        Args:
            threshold_pct: Minimum discount percentage to flag as undervalued.

        Returns:
            List of dicts with listing info and discount percentage.
        """
        if not self._scout_store:
            return []

        # Get 30-day averages per make
        cur = self._conn.cursor()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        cur.execute("""
            SELECT LOWER(make) as make,
                   ROUND(AVG(avg_price), 2) as avg_30d
            FROM market_snapshots
            WHERE snapshot_date >= ?
            GROUP BY LOWER(make)
        """, (cutoff,))

        avg_by_make = {row["make"]: row["avg_30d"] for row in cur.fetchall()}

        if not avg_by_make:
            return []

        listings = self._scout_store.list_all()
        undervalued = []

        for entry in listings:
            make = (entry.make or "").lower().strip()
            if not make or make not in avg_by_make or entry.price <= 0:
                continue

            avg_price = avg_by_make[make]
            if avg_price <= 0:
                continue

            discount_pct = round(((avg_price - entry.price) / avg_price) * 100, 1)
            if discount_pct >= threshold_pct:
                undervalued.append({
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "price": entry.price,
                    "make": entry.make,
                    "model": entry.model,
                    "year": entry.year,
                    "location": entry.location,
                    "avg_price": avg_price,
                    "discount_pct": discount_pct,
                    "deal_score": entry.deal_score,
                    "url": entry.url,
                })

        undervalued.sort(key=lambda d: d["discount_pct"], reverse=True)
        return undervalued

    # -------------------------------------------------------------------
    # Location summary
    # -------------------------------------------------------------------

    def get_location_summary(self) -> list[dict]:
        """Group current scout listings by location with counts and averages.

        Returns:
            List of dicts with location, count, avg_price, hot_deals count.
        """
        if not self._scout_store:
            return []

        listings = self._scout_store.list_all()
        loc_groups: dict[str, dict] = {}

        for entry in listings:
            loc = (entry.location or "Unknown").strip()
            if loc not in loc_groups:
                loc_groups[loc] = {"prices": [], "hot": 0}
            if entry.price > 0:
                loc_groups[loc]["prices"].append(entry.price)
            if entry.deal_score >= 8:
                loc_groups[loc]["hot"] += 1

        result = []
        for loc, data in loc_groups.items():
            prices = data["prices"]
            result.append({
                "location": loc,
                "count": len(prices),
                "avg_price": round(sum(prices) / len(prices), 2) if prices else 0,
                "hot_deals": data["hot"],
            })

        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    # -------------------------------------------------------------------
    # Report generation
    # -------------------------------------------------------------------

    async def generate_report(self, report_type: str = "weekly") -> dict:
        """Assemble market data and generate a summary report via model router.

        Uses task_complexity="simple" for local model execution (zero API cost).

        Args:
            report_type: Type of report ("weekly", "monthly", "custom").

        Returns:
            Dict with report_id, title, summary_text, report_date.
        """
        # Gather data for the report
        trends = self.get_all_trends(days=30)
        undervalued = self.detect_undervalued()
        locations = self.get_location_summary()

        now = datetime.now(timezone.utc)
        report_date = now.strftime("%Y-%m-%d")
        report_id = str(uuid.uuid4())

        data_payload = {
            "report_type": report_type,
            "date": report_date,
            "trends": trends,
            "undervalued": undervalued[:10],
            "locations": locations,
        }

        # Generate summary via model router if available
        summary_text = ""
        if self.router:
            system_prompt = (
                "You are a motorcycle market analyst. Generate a concise markdown "
                "report covering: current market conditions, notable price trends, "
                "best deals available, and geographic observations. Be specific with "
                "numbers. Keep it under 500 words."
            )

            data_str = json.dumps(data_payload, indent=2)
            prompt = f"Generate a motorcycle market report for {report_date}:\n\n{data_str}"

            try:
                if self._on_model_call:
                    self._on_model_call("market_report", "simple")
                response = await self.router.route(
                    prompt=prompt,
                    task_complexity="simple",
                    system_prompt=system_prompt,
                )
                summary_text = response.text
            except Exception as e:
                logger.warning("Model report generation failed: %s", e)
                summary_text = self._generate_fallback_report(data_payload)
        else:
            summary_text = self._generate_fallback_report(data_payload)

        title = f"Market Report — {report_date} ({report_type})"

        # Store report
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO market_reports
            (report_id, report_date, report_type, title, summary_text, data_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (report_id, report_date, report_type, title, summary_text,
              json.dumps(data_payload), now.isoformat()))
        self._conn.commit()

        logger.info("Market report generated: %s", report_id[:8])
        return {
            "report_id": report_id,
            "title": title,
            "summary_text": summary_text,
            "report_date": report_date,
            "report_type": report_type,
        }

    def _generate_fallback_report(self, data: dict) -> str:
        """Generate a basic markdown report without the model router."""
        lines = [f"# Motorcycle Market Report — {data['date']}\n"]

        trends = data.get("trends", [])
        if trends:
            lines.append("## Price Trends\n")
            for t in trends[:10]:
                arrow = {"rising": "↑", "falling": "↓", "stable": "→"}.get(t["direction"], "→")
                make_model = t["make"]
                if t["model"]:
                    make_model += f" {t['model']}"
                lines.append(
                    f"- **{make_model.title()}** {arrow} {t['change_pct']:+.1f}% "
                    f"(avg ${t['recent_avg']:,.0f}, {t['data_points']} data points)"
                )
            lines.append("")

        undervalued = data.get("undervalued", [])
        if undervalued:
            lines.append("## Undervalued Listings\n")
            for u in undervalued[:5]:
                lines.append(
                    f"- **{u['title']}** — ${u['price']:,} "
                    f"({u['discount_pct']:.0f}% below avg ${u['avg_price']:,.0f})"
                )
            lines.append("")

        locations = data.get("locations", [])
        if locations:
            lines.append("## Geographic Summary\n")
            for loc in locations[:8]:
                hot_str = f", {loc['hot_deals']} hot deals" if loc["hot_deals"] else ""
                lines.append(
                    f"- **{loc['location']}** — {loc['count']} listings, "
                    f"avg ${loc['avg_price']:,.0f}{hot_str}"
                )
            lines.append("")

        if not trends and not undervalued:
            lines.append("*No market data available yet. Take snapshots to build history.*\n")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Report retrieval
    # -------------------------------------------------------------------

    def get_latest_report(self) -> dict | None:
        """Return the most recent market report."""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT report_id, report_date, report_type, title, summary_text,
                   data_json, created_at
            FROM market_reports
            ORDER BY created_at DESC LIMIT 1
        """)
        row = cur.fetchone()
        return self._row_to_report(row) if row else None

    def get_report(self, report_id: str) -> dict | None:
        """Return a specific report by its UUID."""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT report_id, report_date, report_type, title, summary_text,
                   data_json, created_at
            FROM market_reports
            WHERE report_id = ?
        """, (report_id,))
        row = cur.fetchone()
        return self._row_to_report(row) if row else None

    def list_reports(self, limit: int = 20) -> list[dict]:
        """Return recent report metadata (without full summary text)."""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT report_id, report_date, report_type, title, created_at
            FROM market_reports
            ORDER BY created_at DESC LIMIT ?
        """, (limit,))
        return [
            {
                "report_id": row["report_id"],
                "report_date": row["report_date"],
                "report_type": row["report_type"],
                "title": row["title"],
                "created_at": row["created_at"],
            }
            for row in cur.fetchall()
        ]

    def _row_to_report(self, row: sqlite3.Row) -> dict:
        """Convert a report DB row to a dict."""
        data_json = row["data_json"]
        try:
            data = json.loads(data_json) if data_json else {}
        except (json.JSONDecodeError, TypeError):
            data = {}
        return {
            "report_id": row["report_id"],
            "report_date": row["report_date"],
            "report_type": row["report_type"],
            "title": row["title"],
            "summary_text": row["summary_text"],
            "data": data,
            "created_at": row["created_at"],
        }

    # -------------------------------------------------------------------
    # Dashboard summary
    # -------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return full dashboard payload with all analytics data.

        Returns:
            Dict with trends, undervalued listings, location summary,
            snapshot counts, and latest report info.
        """
        cur = self._conn.cursor()

        # Count distinct makes tracked
        cur.execute("SELECT COUNT(DISTINCT make) FROM market_snapshots")
        tracked_makes = cur.fetchone()[0]

        # Total snapshots
        cur.execute("SELECT COUNT(*) FROM market_snapshots")
        total_snapshots = cur.fetchone()[0]

        # Latest snapshot date
        cur.execute("SELECT MAX(snapshot_date) FROM market_snapshots")
        row = cur.fetchone()
        latest_date = row[0] if row and row[0] else ""

        # Report count
        cur.execute("SELECT COUNT(*) FROM market_reports")
        report_count = cur.fetchone()[0]

        # Latest report ID
        latest_report = self.get_latest_report()
        latest_report_id = latest_report["report_id"] if latest_report else ""

        return {
            "tracked_makes": tracked_makes,
            "total_snapshots": total_snapshots,
            "latest_date": latest_date,
            "report_count": report_count,
            "latest_report_id": latest_report_id,
            "trends": self.get_all_trends(days=30),
            "undervalued": self.detect_undervalued(),
            "location_summary": self.get_location_summary(),
        }

    # -------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------

    def export_report_markdown(self, report_id: str) -> str | None:
        """Return report markdown text for download.

        Args:
            report_id: UUID of the report to export.

        Returns:
            Markdown string, or None if report not found.
        """
        report = self.get_report(report_id)
        if not report:
            return None
        return report["summary_text"]

    # -------------------------------------------------------------------
    # Close
    # -------------------------------------------------------------------

    def close(self):
        """Close the SQLite connection."""
        if self._conn:
            self._conn.close()
            logger.info("MarketAnalyzer closed")


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import tempfile
    import os
    import sys

    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

    from tools.doppler.scout import ScoutStore

    tmp_market = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_scout = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_market.close()
    tmp_scout.close()

    passed = 0
    failed = 0

    def check(label, condition):
        global passed, failed
        if condition:
            print(f"  PASS  {label}")
            passed += 1
        else:
            print(f"  FAIL  {label}")
            failed += 1

    try:
        # Set up scout store with test data
        store = ScoutStore(db_path=tmp_scout.name)
        store.add_entry(
            title="2019 Honda CB500F - Low Miles",
            url="https://example.com/honda1",
            source="craigslist",
            price=4200,
            make="honda",
            model="CB500F",
            year=2019,
            location="Portland",
        )
        store.add_entry(
            title="2015 Honda CB300R - Clean Title",
            url="https://example.com/honda2",
            source="craigslist",
            price=2800,
            make="honda",
            model="CB300R",
            year=2015,
            location="Portland",
        )
        store.add_entry(
            title="2018 Yamaha FZ-07",
            url="https://example.com/yamaha1",
            source="craigslist",
            price=3500,
            make="yamaha",
            model="FZ-07",
            year=2018,
            location="Salem",
        )
        store.add_entry(
            title="2004 Ducati Monster 620 - Project",
            url="https://example.com/ducati1",
            source="facebook",
            price=1500,
            make="ducati",
            model="Monster 620",
            year=2004,
            location="Portland",
        )

        print("=" * 60)
        print("MarketAnalyzer Self-Test")
        print("=" * 60)

        # --- Test initialization ---
        print("\n--- Initialization ---")
        analyzer = MarketAnalyzer(db_path=tmp_market.name, scout_store=store)
        check("Analyzer created", analyzer is not None)
        check("DB path set", analyzer._db_path == tmp_market.name)

        # --- Test snapshot ---
        print("\n--- Snapshot ---")
        count = analyzer.take_snapshot()
        check("Snapshot returns count > 0", count > 0)
        print(f"    Inserted {count} snapshot rows")

        # Take again — should insert 0 (same day, INSERT OR IGNORE)
        count2 = analyzer.take_snapshot()
        check("Duplicate snapshot returns 0", count2 == 0)

        # --- Test price history ---
        print("\n--- Price History ---")
        history = analyzer.get_price_history("honda", days=30)
        check("Honda history has entries", len(history) > 0)
        if history:
            check("History has avg_price", "avg_price" in history[0])
            print(f"    Honda avg: ${history[0]['avg_price']}")

        # --- Test trends ---
        print("\n--- Trends ---")
        trend = analyzer.get_trend("honda", days=30)
        check("Honda trend has direction", trend["direction"] in ("rising", "falling", "stable"))
        check("Honda trend has change_pct", "change_pct" in trend)
        print(f"    Honda trend: {trend['direction']} ({trend['change_pct']:+.1f}%)")

        all_trends = analyzer.get_all_trends(days=30)
        check("All trends returns list", len(all_trends) >= 0)
        print(f"    Total tracked combos: {len(all_trends)}")

        # --- Test undervalued detection ---
        print("\n--- Undervalued Detection ---")

        # Manually insert older snapshots with higher avg to create undervalued condition
        cur = analyzer._conn.cursor()
        old_date = (datetime.now(timezone.utc) - timedelta(days=15)).strftime("%Y-%m-%d")
        now = datetime.now(timezone.utc).isoformat()
        cur.execute("""
            INSERT OR IGNORE INTO market_snapshots
            (snapshot_date, make, model, avg_price, min_price, max_price,
             listing_count, location, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (old_date, "ducati", "monster 620", 3000, 2500, 3500, 3, "Portland", now))
        analyzer._conn.commit()

        undervalued = analyzer.detect_undervalued(threshold_pct=20.0)
        check("Undervalued detection runs", isinstance(undervalued, list))
        if undervalued:
            check("Undervalued has discount_pct", "discount_pct" in undervalued[0])
            print(f"    Found {len(undervalued)} undervalued listings")
            for u in undervalued:
                print(f"    - {u['title']}: ${u['price']} ({u['discount_pct']:.0f}% below avg)")

        # --- Test location summary ---
        print("\n--- Location Summary ---")
        locations = analyzer.get_location_summary()
        check("Location summary returns list", len(locations) > 0)
        if locations:
            check("Location has count field", "count" in locations[0])
            for loc in locations:
                print(f"    {loc['location']}: {loc['count']} listings, avg ${loc['avg_price']:,.0f}")

        # --- Test dashboard summary ---
        print("\n--- Dashboard Summary ---")
        summary = analyzer.get_summary()
        check("Summary has tracked_makes", "tracked_makes" in summary)
        check("Summary has total_snapshots", "total_snapshots" in summary)
        check("Summary has trends", "trends" in summary)
        print(f"    Tracked makes: {summary['tracked_makes']}")
        print(f"    Total snapshots: {summary['total_snapshots']}")

        # --- Test report generation (without router — fallback) ---
        print("\n--- Report Generation (fallback) ---")

        async def test_report():
            report = await analyzer.generate_report("weekly")
            check("Report has report_id", "report_id" in report)
            check("Report has summary_text", len(report["summary_text"]) > 0)
            print(f"    Report: {report['title']}")
            print(f"    Summary length: {len(report['summary_text'])} chars")
            return report

        report = asyncio.run(test_report())

        # --- Test report retrieval ---
        print("\n--- Report Retrieval ---")
        latest = analyzer.get_latest_report()
        check("Latest report found", latest is not None)
        check("Latest matches generated", latest["report_id"] == report["report_id"])

        fetched = analyzer.get_report(report["report_id"])
        check("Get report by ID works", fetched is not None)

        reports_list = analyzer.list_reports()
        check("List reports returns entries", len(reports_list) > 0)

        # --- Test export ---
        print("\n--- Export ---")
        md = analyzer.export_report_markdown(report["report_id"])
        check("Export returns markdown", md is not None and len(md) > 0)

        bad_export = analyzer.export_report_markdown("nonexistent-id")
        check("Export returns None for bad ID", bad_export is None)

        # --- Cleanup ---
        analyzer.close()
        store.close()

        print("\n" + "=" * 60)
        print(f"Results: {passed} passed, {failed} failed")
        print("=" * 60)

    finally:
        os.unlink(tmp_market.name)
        os.unlink(tmp_scout.name)

    if failed > 0:
        sys.exit(1)
