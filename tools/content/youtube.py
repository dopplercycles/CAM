"""
CAM YouTube Publishing Pipeline

Manages videos through draft → queued → published workflow with SEO
metadata generation via the model router and consistent Doppler Cycles
branding in descriptions.

Usage:
    from tools.content.youtube import YouTubeManager

    yt = YouTubeManager(
        db_path="data/youtube.db",
        router=orchestrator.router,
        on_change=broadcast_youtube_status,
    )
    video = yt.create_video("Valve Adjustment", topic="Harley M-8 valve adjustment")
"""

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.youtube")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_CATEGORIES = ("diagnostics", "history", "barn_find", "how_to", "project_build", "general")
VIDEO_STATUSES = ("draft", "queued", "published", "failed")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class YouTubeVideo:
    """Represents a single YouTube video in the publishing pipeline."""

    video_id: str
    title: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    category: str = "general"
    topic: str = ""
    keywords: list[str] = field(default_factory=list)
    thumbnail_path: str | None = None
    video_path: str | None = None
    scheduled_date: str | None = None
    status: str = "draft"
    published_url: str | None = None
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    chapters: list[dict] = field(default_factory=list)
    pipeline_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 chars of the UUID for display."""
        return self.video_id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON/WS transport."""
        return {
            "video_id": self.video_id,
            "short_id": self.short_id,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "category": self.category,
            "topic": self.topic,
            "keywords": self.keywords,
            "thumbnail_path": self.thumbnail_path,
            "video_path": self.video_path,
            "scheduled_date": self.scheduled_date,
            "status": self.status,
            "published_url": self.published_url,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "comment_count": self.comment_count,
            "chapters": self.chapters,
            "pipeline_id": self.pipeline_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "YouTubeVideo":
        """Build a YouTubeVideo from a sqlite3.Row."""
        d = dict(row)
        return cls(
            video_id=d["video_id"],
            title=d["title"],
            description=d.get("description", ""),
            tags=_parse_json_list(d.get("tags")),
            category=d.get("category", "general"),
            topic=d.get("topic", ""),
            keywords=_parse_json_list(d.get("keywords")),
            thumbnail_path=d.get("thumbnail_path"),
            video_path=d.get("video_path"),
            scheduled_date=d.get("scheduled_date"),
            status=d.get("status", "draft"),
            published_url=d.get("published_url"),
            view_count=d.get("view_count", 0),
            like_count=d.get("like_count", 0),
            comment_count=d.get("comment_count", 0),
            chapters=_parse_json_list(d.get("chapters")),
            pipeline_id=d.get("pipeline_id"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            metadata=_parse_json_dict(d.get("metadata")),
        )


def _parse_json_list(val: str | None) -> list:
    """Safely parse a JSON string to a list."""
    if not val:
        return []
    try:
        result = json.loads(val)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_json_dict(val: str | None) -> dict:
    """Safely parse a JSON string to a dict."""
    if not val:
        return {}
    try:
        result = json.loads(val)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# YouTubeManager
# ---------------------------------------------------------------------------

class YouTubeManager:
    """Manages the YouTube video publishing pipeline with SQLite storage.

    Args:
        db_path:      Path to SQLite database file.
        on_change:    Async callback fired after any state change (for WS broadcast).
        router:       ModelRouter instance for SEO generation.
        channel_url:  YouTube channel URL (filled in when channel is live).
        website_url:  Business website URL (filled in when site is live).
    """

    # Motorcycle-focused seed tags for SEO generation
    MOTO_TAG_BASES = [
        "harley davidson", "ducati", "yamaha", "honda", "kawasaki", "suzuki",
        "triumph", "bmw", "indian", "ktm", "aprilia", "moto guzzi",
        "sportbike", "cruiser", "adventure", "touring", "cafe racer",
        "bobber", "chopper", "dual sport", "enduro", "naked bike",
        "motorcycle repair", "motorcycle diagnostics", "motorcycle maintenance",
        "valve adjustment", "carburetor", "fuel injection", "electrical",
        "engine rebuild", "clutch", "brakes", "suspension", "exhaust",
        "barn find", "restoration", "motorcycle history", "how to",
        "diy motorcycle", "motorcycle tech", "shop talk", "wrench time",
        "portland motorcycle", "mobile mechanic", "doppler cycles",
    ]

    DESCRIPTION_TEMPLATE = """{body}

{chapters_block}
---
Doppler Cycles — Mobile Motorcycle Diagnostics | Portland Metro
George brings 20+ years of wrench time, AMI certified, factory trained.

{channel_url}
{website_url}

#DopplerCycles #MotorcycleDiagnostics {hashtags}"""

    def __init__(
        self,
        db_path: str = "data/youtube.db",
        on_change: Callable[[], Awaitable[None]] | None = None,
        router: Any = None,
        channel_url: str = "",
        website_url: str = "",
    ):
        self._db_path = db_path
        self._on_change = on_change
        self._router = router
        self._channel_url = channel_url
        self._website_url = website_url

        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

        logger.info("YouTubeManager initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self):
        """Create tables and indexes if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS youtube_videos (
                video_id        TEXT PRIMARY KEY,
                title           TEXT NOT NULL,
                description     TEXT DEFAULT '',
                tags            TEXT DEFAULT '[]',
                category        TEXT DEFAULT 'general',
                topic           TEXT DEFAULT '',
                keywords        TEXT DEFAULT '[]',
                thumbnail_path  TEXT,
                video_path      TEXT,
                scheduled_date  TEXT,
                status          TEXT NOT NULL DEFAULT 'draft',
                published_url   TEXT,
                view_count      INTEGER DEFAULT 0,
                like_count      INTEGER DEFAULT 0,
                comment_count   INTEGER DEFAULT 0,
                chapters        TEXT DEFAULT '[]',
                pipeline_id     TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                metadata        TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_yt_status ON youtube_videos(status);
            CREATE INDEX IF NOT EXISTS idx_yt_category ON youtube_videos(category);
            CREATE INDEX IF NOT EXISTS idx_yt_scheduled ON youtube_videos(scheduled_date);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _now_iso(self) -> str:
        """UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    async def _notify_change(self):
        """Fire the on_change callback if set."""
        if self._on_change is not None:
            try:
                await self._on_change()
            except Exception:
                logger.debug("YouTube on_change callback error", exc_info=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_video(
        self,
        title: str,
        topic: str = "",
        category: str = "general",
        keywords: list[str] | None = None,
        scheduled_date: str | None = None,
        thumbnail_path: str | None = None,
        video_path: str | None = None,
        chapters: list[dict] | None = None,
        pipeline_id: str | None = None,
        metadata: dict | None = None,
    ) -> YouTubeVideo:
        """Create a new video in draft status.

        Args:
            title:          Video title.
            topic:          Core subject for SEO generation.
            category:       One of VIDEO_CATEGORIES.
            keywords:       Target keywords for SEO.
            scheduled_date: Planned publish date (YYYY-MM-DD).
            thumbnail_path: Path to thumbnail image.
            video_path:     Path to video file.
            chapters:       List of {time, label} dicts.
            pipeline_id:    Optional link to content pipeline.
            metadata:       Extra metadata dict.

        Returns:
            The newly created YouTubeVideo.
        """
        if category not in VIDEO_CATEGORIES:
            category = "general"

        video_id = str(uuid.uuid4())
        now = self._now_iso()

        self._conn.execute(
            """INSERT INTO youtube_videos
               (video_id, title, topic, category, keywords, scheduled_date,
                thumbnail_path, video_path, chapters, pipeline_id, metadata,
                status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)""",
            (
                video_id, title, topic, category,
                json.dumps(keywords or []),
                scheduled_date, thumbnail_path, video_path,
                json.dumps(chapters or []),
                pipeline_id,
                json.dumps(metadata or {}),
                now, now,
            ),
        )
        self._conn.commit()

        video = self.get_video(video_id)
        logger.info("Created YouTube video %s: %s", video.short_id, title)
        return video

    def update_video(self, video_id: str, **kwargs) -> YouTubeVideo | None:
        """Update fields on an existing video.

        Accepted kwargs match YouTubeVideo fields. JSON fields (tags, keywords,
        chapters, metadata) are serialized automatically.

        Returns:
            Updated YouTubeVideo, or None if not found.
        """
        video = self.get_video(video_id)
        if video is None:
            return None

        # Fields that need JSON serialization
        json_fields = {"tags", "keywords", "chapters", "metadata"}
        allowed = {
            "title", "description", "tags", "category", "topic", "keywords",
            "thumbnail_path", "video_path", "scheduled_date", "status",
            "published_url", "view_count", "like_count", "comment_count",
            "chapters", "pipeline_id", "metadata",
        }

        sets = []
        vals = []
        for key, val in kwargs.items():
            if key not in allowed:
                continue
            if key in json_fields:
                val = json.dumps(val)
            sets.append(f"{key} = ?")
            vals.append(val)

        if not sets:
            return video

        sets.append("updated_at = ?")
        vals.append(self._now_iso())
        vals.append(video_id)

        self._conn.execute(
            f"UPDATE youtube_videos SET {', '.join(sets)} WHERE video_id = ?",
            vals,
        )
        self._conn.commit()

        return self.get_video(video_id)

    def delete_video(self, video_id: str) -> bool:
        """Delete a video from the database.

        Returns:
            True if a row was deleted, False if not found.
        """
        cur = self._conn.execute(
            "DELETE FROM youtube_videos WHERE video_id = ?", (video_id,)
        )
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Deleted YouTube video %s", video_id[:8])
        return deleted

    def get_video(self, video_id: str) -> YouTubeVideo | None:
        """Fetch a single video by ID.

        Returns:
            YouTubeVideo or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM youtube_videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        return YouTubeVideo.from_row(row) if row else None

    def list_videos(self, status: str | None = None, limit: int = 50) -> list[YouTubeVideo]:
        """List videos, optionally filtered by status.

        Args:
            status: Filter by status (draft/queued/published/failed), or None for all.
            limit:  Max results to return.

        Returns:
            List of YouTubeVideo sorted by updated_at descending.
        """
        if status and status in VIDEO_STATUSES:
            rows = self._conn.execute(
                "SELECT * FROM youtube_videos WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM youtube_videos ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [YouTubeVideo.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def queue_video(self, video_id: str) -> YouTubeVideo | None:
        """Move a draft video to queued status.

        Validates that title and description exist before queuing.

        Returns:
            Updated YouTubeVideo, or None if validation fails or not found.
        """
        video = self.get_video(video_id)
        if video is None:
            logger.warning("queue_video: video %s not found", video_id[:8])
            return None

        if video.status != "draft":
            logger.warning("queue_video: video %s is %s, not draft", video.short_id, video.status)
            return None

        if not video.title or not video.description:
            logger.warning("queue_video: video %s missing title or description", video.short_id)
            return None

        return self.update_video(video_id, status="queued")

    def mark_published(self, video_id: str, published_url: str = "") -> YouTubeVideo | None:
        """Move a queued video to published status.

        Args:
            video_id:      The video to publish.
            published_url: The YouTube URL of the published video.

        Returns:
            Updated YouTubeVideo, or None if not found or wrong status.
        """
        video = self.get_video(video_id)
        if video is None:
            logger.warning("mark_published: video %s not found", video_id[:8])
            return None

        if video.status != "queued":
            logger.warning("mark_published: video %s is %s, not queued", video.short_id, video.status)
            return None

        kwargs = {"status": "published"}
        if published_url:
            kwargs["published_url"] = published_url
        return self.update_video(video_id, **kwargs)

    def mark_failed(self, video_id: str, error: str = "") -> YouTubeVideo | None:
        """Move any video to failed status with an error message.

        Args:
            video_id: The video that failed.
            error:    Description of what went wrong.

        Returns:
            Updated YouTubeVideo, or None if not found.
        """
        video = self.get_video(video_id)
        if video is None:
            return None

        meta = video.metadata.copy()
        meta["last_error"] = error
        meta["failed_at"] = self._now_iso()
        return self.update_video(video_id, status="failed", metadata=meta)

    # ------------------------------------------------------------------
    # SEO generation
    # ------------------------------------------------------------------

    async def generate_seo(self, video_id: str) -> dict | None:
        """Generate title, description, and tags for a video using the model router.

        Orchestrates generate_title(), generate_description(), and generate_tags(),
        then updates the video record in the database.

        Returns:
            Dict with generated {title, description, tags}, or None on failure.
        """
        video = self.get_video(video_id)
        if video is None:
            return None

        topic = video.topic or video.title
        keywords = video.keywords or []

        try:
            title = await self.generate_title(topic, keywords)
            description = await self.generate_description(topic, keywords, video.chapters)
            tags = self.generate_tags(topic, keywords)

            self.update_video(video_id, title=title, description=description, tags=tags)
            logger.info("Generated SEO for video %s", video_id[:8])

            return {"title": title, "description": description, "tags": tags}
        except Exception as e:
            logger.error("SEO generation failed for %s: %s", video_id[:8], e)
            return None

    async def generate_title(self, topic: str, keywords: list[str]) -> str:
        """Generate a YouTube-optimized title using the model router.

        Uses 'routine' complexity to hit local Ollama (free).

        Args:
            topic:    Core subject of the video.
            keywords: Target SEO keywords.

        Returns:
            Generated title string (max ~60 chars).
        """
        if not self._router:
            return topic

        kw_str = ", ".join(keywords) if keywords else "motorcycle"
        prompt = (
            f"Generate a YouTube video title for a motorcycle diagnostics channel.\n"
            f"Topic: {topic}\n"
            f"Keywords: {kw_str}\n"
            f"Requirements:\n"
            f"- Under 60 characters\n"
            f"- Engaging and click-worthy but not clickbait\n"
            f"- Include relevant motorcycle terminology\n"
            f"- Written for motorcycle enthusiasts\n"
            f"Return ONLY the title, no quotes or explanation."
        )
        response = await self._router.route(
            prompt,
            task_complexity="routine",
            system_prompt="You are a motorcycle content specialist writing YouTube titles for Doppler Cycles, a mobile motorcycle diagnostics channel in Portland.",
        )
        # Clean up the response — just the title text
        title = response.text.strip().strip('"').strip("'")
        return title[:60] if len(title) > 60 else title

    async def generate_description(
        self, topic: str, keywords: list[str], chapters: list[dict] | None = None
    ) -> str:
        """Generate a YouTube description and merge it into the branding template.

        Uses 'routine' complexity to hit local Ollama (free).

        Args:
            topic:    Core subject of the video.
            keywords: Target SEO keywords.
            chapters: Optional list of {time, label} chapter markers.

        Returns:
            Full description with Doppler Cycles branding template applied.
        """
        if not self._router:
            body = f"In this video, we cover {topic}."
        else:
            kw_str = ", ".join(keywords) if keywords else "motorcycle"
            prompt = (
                f"Write a YouTube video description body for a motorcycle diagnostics channel.\n"
                f"Topic: {topic}\n"
                f"Keywords: {kw_str}\n"
                f"Requirements:\n"
                f"- 2-3 paragraphs, conversational tone\n"
                f"- Include relevant keywords naturally\n"
                f"- Written for motorcycle enthusiasts\n"
                f"- Don't include hashtags or channel info (that's added separately)\n"
                f"Return ONLY the description body text."
            )
            response = await self._router.route(
                prompt,
                task_complexity="routine",
                system_prompt="You are a motorcycle content writer for Doppler Cycles, a mobile diagnostics channel. George has 20+ years of wrench time.",
            )
            body = response.text.strip()

        # Build chapters block
        chapters_block = ""
        if chapters:
            lines = [f"{ch.get('time', '0:00')} {ch.get('label', '')}" for ch in chapters]
            chapters_block = "Chapters:\n" + "\n".join(lines) + "\n"

        # Build hashtags from keywords
        hashtags = " ".join(f"#{kw.replace(' ', '')}" for kw in (keywords or [])[:5])

        return self.DESCRIPTION_TEMPLATE.format(
            body=body,
            chapters_block=chapters_block,
            channel_url=self._channel_url,
            website_url=self._website_url,
            hashtags=hashtags,
        )

    def generate_tags(self, topic: str, keywords: list[str]) -> list[str]:
        """Generate relevant tags by matching topic/keywords against motorcycle vocabulary.

        This is sync — no API calls, just keyword matching. Free.

        Args:
            topic:    Core subject of the video.
            keywords: User-provided target keywords.

        Returns:
            Deduplicated list of up to 15 tags.
        """
        topic_lower = topic.lower()
        keyword_set = {kw.lower() for kw in (keywords or [])}

        tags = list(keywords or [])

        # Add matching base tags
        for base in self.MOTO_TAG_BASES:
            if base in topic_lower or any(kw in base or base in kw for kw in keyword_set):
                tags.append(base)

        # Always include channel tag
        tags.append("doppler cycles")

        # Deduplicate while preserving order, limit to 15
        seen = set()
        unique_tags = []
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                seen.add(tag_lower)
                unique_tags.append(tag)
            if len(unique_tags) >= 15:
                break

        return unique_tags

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def update_stats(
        self, video_id: str, view_count: int, like_count: int, comment_count: int
    ) -> YouTubeVideo | None:
        """Update engagement stats for a video.

        Args:
            video_id:      Video to update.
            view_count:    Current view count.
            like_count:    Current like count.
            comment_count: Current comment count.

        Returns:
            Updated YouTubeVideo, or None if not found.
        """
        return self.update_video(
            video_id,
            view_count=view_count,
            like_count=like_count,
            comment_count=comment_count,
        )

    def get_channel_stats(self) -> dict[str, Any]:
        """Aggregate channel-level stats across all videos.

        Returns:
            Dict with total_videos, published, queued, drafts, total_views,
            total_likes, avg_views.
        """
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM youtube_videos GROUP BY status"
        ).fetchall()
        counts = {r["status"]: r["cnt"] for r in rows}

        totals = self._conn.execute(
            "SELECT COALESCE(SUM(view_count), 0) as views, "
            "COALESCE(SUM(like_count), 0) as likes "
            "FROM youtube_videos WHERE status = 'published'"
        ).fetchone()

        total_videos = sum(counts.values())
        published = counts.get("published", 0)

        return {
            "total_videos": total_videos,
            "published": published,
            "queued": counts.get("queued", 0),
            "drafts": counts.get("draft", 0),
            "failed": counts.get("failed", 0),
            "total_views": totals["views"],
            "total_likes": totals["likes"],
            "avg_views": round(totals["views"] / published, 1) if published else 0,
        }

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, int]:
        """Summary counts for the status bar."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM youtube_videos GROUP BY status"
        ).fetchall()
        counts = {r["status"]: r["cnt"] for r in rows}
        return {
            "total": sum(counts.values()),
            "drafts": counts.get("draft", 0),
            "queued": counts.get("queued", 0),
            "published": counts.get("published", 0),
            "failed": counts.get("failed", 0),
        }

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Full state for WS broadcast to dashboard clients.

        Returns:
            Dict with videos list, status counts, and channel stats.
        """
        videos = self.list_videos()
        return {
            "videos": [v.to_dict() for v in videos],
            "status": self.get_status(),
            "channel_stats": self.get_channel_stats(),
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("YouTubeManager closed")
