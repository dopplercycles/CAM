"""
CAM Social Media Content Generator

Cross-platform social media content generator that creates copy-ready posts
from a single topic brief.  Each platform gets tailored content: short for X,
visual-focused for Instagram, conversational for YouTube Community, detailed
for Facebook.  Actual API posting is future — this generates draft content
for manual posting.

Usage:
    from tools.content.social_media import SocialMediaManager

    sm = SocialMediaManager(
        db_path="data/social_media.db",
        router=orchestrator.router,
        on_change=broadcast_social_media_status,
    )
    posts = await sm.draft_batch("Harley M-8 valve adjustment", platforms=["x_twitter", "instagram"])
"""

import json
import logging
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.social_media")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLATFORMS = ("youtube_community", "instagram", "facebook", "x_twitter")
POST_STATUSES = ("draft", "scheduled", "posted", "failed")

PLATFORM_CONSTRAINTS = {
    "x_twitter":         {"max_chars": 280,  "max_hashtags": 5,  "style": "short_punchy"},
    "instagram":         {"max_chars": 2200, "max_hashtags": 30, "style": "visual_caption"},
    "youtube_community": {"max_chars": 1500, "max_hashtags": 5,  "style": "conversational"},
    "facebook":          {"max_chars": 5000, "max_hashtags": 10, "style": "detailed"},
}

DEFAULT_FREQUENCY = {
    "youtube_community": 1,
    "instagram": 3,
    "facebook": 2,
    "x_twitter": 5,
}

# Per-platform prompt templates
_PLATFORM_PROMPTS = {
    "x_twitter": (
        "Write a tweet for a motorcycle diagnostics business.\n"
        "Topic: {topic}\n"
        "{extra}\n"
        "Requirements:\n"
        "- MUST be under 280 characters total (including hashtags)\n"
        "- Punchy, engaging, gets attention fast\n"
        "- Include 3-5 relevant hashtags at the end\n"
        "- Written for motorcycle enthusiasts\n"
        "Return ONLY the tweet text with hashtags. No quotes, no explanation."
    ),
    "instagram": (
        "Write an Instagram caption for a motorcycle diagnostics business.\n"
        "Topic: {topic}\n"
        "{extra}\n"
        "Requirements:\n"
        "- Visual-focused caption, describe what the viewer would see\n"
        "- Conversational, enthusiast tone\n"
        "- Include 15-20 relevant motorcycle hashtags at the end\n"
        "- Under 2200 characters total\n"
        "Return ONLY the caption text with hashtags. No quotes, no explanation."
    ),
    "youtube_community": (
        "Write a YouTube Community post for a motorcycle diagnostics channel.\n"
        "Topic: {topic}\n"
        "{extra}\n"
        "Requirements:\n"
        "- Conversational, like talking to fellow riders\n"
        "- Ask a question or invite discussion\n"
        "- Under 1500 characters\n"
        "- Include 3-5 relevant hashtags at the end\n"
        "Return ONLY the post text with hashtags. No quotes, no explanation."
    ),
    "facebook": (
        "Write a Facebook post for a mobile motorcycle diagnostics business.\n"
        "Topic: {topic}\n"
        "{extra}\n"
        "Requirements:\n"
        "- Detailed and informative\n"
        "- Conversational but professional\n"
        "- Include a call-to-action (comment, share, or book a service)\n"
        "- Under 5000 characters\n"
        "- Include 5-8 relevant hashtags at the end\n"
        "Return ONLY the post text with hashtags. No quotes, no explanation."
    ),
}

_SYSTEM_PROMPT = (
    "You are a social media content writer for Doppler Cycles, a mobile "
    "motorcycle diagnostics and repair service in Portland, Oregon. "
    "George has 20+ years of wrench time, AMI certified, factory trained. "
    "Write with enthusiasm for motorcycles and genuine technical knowledge. "
    "The brand voice is knowledgeable, conversational, and rider-to-rider."
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SocialPost:
    """Represents a single social media post."""

    post_id: str
    platform: str
    content: str = ""
    media_paths: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    scheduled_time: str | None = None
    status: str = "draft"
    post_url: str | None = None
    topic: str = ""
    pipeline_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 chars of the UUID for display."""
        return self.post_id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON/WS transport."""
        return {
            "post_id": self.post_id,
            "short_id": self.short_id,
            "platform": self.platform,
            "content": self.content,
            "media_paths": self.media_paths,
            "hashtags": self.hashtags,
            "scheduled_time": self.scheduled_time,
            "status": self.status,
            "post_url": self.post_url,
            "topic": self.topic,
            "pipeline_id": self.pipeline_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SocialPost":
        """Build a SocialPost from a sqlite3.Row."""
        d = dict(row)
        return cls(
            post_id=d["post_id"],
            platform=d["platform"],
            content=d.get("content", ""),
            media_paths=_parse_json_list(d.get("media_paths")),
            hashtags=_parse_json_list(d.get("hashtags")),
            scheduled_time=d.get("scheduled_time"),
            status=d.get("status", "draft"),
            post_url=d.get("post_url"),
            topic=d.get("topic", ""),
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


def _extract_hashtags(text: str) -> tuple[str, list[str]]:
    """Split hashtags out of generated text.

    Returns:
        (clean_text, hashtag_list) where clean_text has hashtags removed
        and hashtag_list contains the tags without the '#' prefix.
    """
    hashtag_pattern = re.compile(r"#(\w+)")
    hashtags = hashtag_pattern.findall(text)
    # Remove hashtags from content
    clean = hashtag_pattern.sub("", text).strip()
    # Clean up extra whitespace / trailing newlines
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, hashtags


# ---------------------------------------------------------------------------
# SocialMediaManager
# ---------------------------------------------------------------------------

class SocialMediaManager:
    """Cross-platform social media content generator with SQLite storage.

    Args:
        db_path:   Path to the SQLite database file.
        on_change: Async callback fired after any state mutation (for WS broadcast).
        router:    Model router instance for AI content generation.
    """

    def __init__(
        self,
        db_path: str = "data/social_media.db",
        on_change: Callable[[], Awaitable[None]] | None = None,
        router: Any = None,
    ):
        self._db_path = db_path
        self._on_change = on_change
        self._router = router

        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

        logger.info("SocialMediaManager initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self):
        """Create tables and indexes if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS social_posts (
                post_id         TEXT PRIMARY KEY,
                platform        TEXT NOT NULL,
                content         TEXT DEFAULT '',
                media_paths     TEXT DEFAULT '[]',
                hashtags        TEXT DEFAULT '[]',
                scheduled_time  TEXT,
                status          TEXT NOT NULL DEFAULT 'draft',
                post_url        TEXT,
                topic           TEXT DEFAULT '',
                pipeline_id     TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                metadata        TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_sm_platform ON social_posts(platform);
            CREATE INDEX IF NOT EXISTS idx_sm_status ON social_posts(status);
            CREATE INDEX IF NOT EXISTS idx_sm_scheduled ON social_posts(scheduled_time);
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
                logger.debug("Social media on_change callback error", exc_info=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_post(
        self,
        platform: str,
        topic: str = "",
        content: str = "",
        hashtags: list[str] | None = None,
        media_paths: list[str] | None = None,
        scheduled_time: str | None = None,
        pipeline_id: str | None = None,
        metadata: dict | None = None,
    ) -> SocialPost:
        """Create a new social media post in draft status.

        Args:
            platform:       Target platform (x_twitter, instagram, etc.).
            topic:          Subject of the post.
            content:        Post body text.
            hashtags:       List of hashtag strings (without '#').
            media_paths:    Paths to attached media files.
            scheduled_time: ISO datetime for scheduled posting.
            pipeline_id:    Optional link to content pipeline.
            metadata:       Extra metadata dict.

        Returns:
            The newly created SocialPost.
        """
        if platform not in PLATFORMS:
            logger.warning("create_post: unknown platform %r, defaulting to x_twitter", platform)
            platform = "x_twitter"

        post_id = str(uuid.uuid4())
        now = self._now_iso()

        self._conn.execute(
            """INSERT INTO social_posts
               (post_id, platform, topic, content, hashtags, media_paths,
                scheduled_time, pipeline_id, metadata, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)""",
            (
                post_id, platform, topic, content,
                json.dumps(hashtags or []),
                json.dumps(media_paths or []),
                scheduled_time, pipeline_id,
                json.dumps(metadata or {}),
                now, now,
            ),
        )
        self._conn.commit()

        post = self.get_post(post_id)
        logger.info("Created %s post %s: %s", platform, post.short_id, topic)
        return post

    def update_post(self, post_id: str, **kwargs) -> SocialPost | None:
        """Update fields on an existing post.

        Accepted kwargs match SocialPost fields.  JSON fields (hashtags,
        media_paths, metadata) are serialized automatically.

        Returns:
            Updated SocialPost, or None if not found.
        """
        post = self.get_post(post_id)
        if post is None:
            return None

        json_fields = {"hashtags", "media_paths", "metadata"}
        allowed = {
            "platform", "content", "media_paths", "hashtags",
            "scheduled_time", "status", "post_url", "topic",
            "pipeline_id", "metadata",
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
            return post

        sets.append("updated_at = ?")
        vals.append(self._now_iso())
        vals.append(post_id)

        self._conn.execute(
            f"UPDATE social_posts SET {', '.join(sets)} WHERE post_id = ?",
            vals,
        )
        self._conn.commit()

        return self.get_post(post_id)

    def delete_post(self, post_id: str) -> bool:
        """Delete a post from the database.

        Returns:
            True if a row was deleted, False if not found.
        """
        cur = self._conn.execute(
            "DELETE FROM social_posts WHERE post_id = ?", (post_id,)
        )
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Deleted social post %s", post_id[:8])
        return deleted

    def get_post(self, post_id: str) -> SocialPost | None:
        """Fetch a single post by ID.

        Returns:
            SocialPost or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM social_posts WHERE post_id = ?", (post_id,)
        ).fetchone()
        return SocialPost.from_row(row) if row else None

    def list_posts(
        self,
        platform: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[SocialPost]:
        """List posts, optionally filtered by platform and/or status.

        Args:
            platform: Filter by platform, or None for all.
            status:   Filter by status (draft/scheduled/posted/failed), or None for all.
            limit:    Max results to return.

        Returns:
            List of SocialPost sorted by updated_at descending.
        """
        where_parts = []
        params: list[Any] = []

        if platform and platform in PLATFORMS:
            where_parts.append("platform = ?")
            params.append(platform)
        if status and status in POST_STATUSES:
            where_parts.append("status = ?")
            params.append(status)

        where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM social_posts{where_clause} ORDER BY updated_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [SocialPost.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def schedule_post(self, post_id: str) -> SocialPost | None:
        """Move a draft post to scheduled status.

        Validates that content is non-empty before scheduling.

        Returns:
            Updated SocialPost, or None if validation fails or not found.
        """
        post = self.get_post(post_id)
        if post is None:
            logger.warning("schedule_post: post %s not found", post_id[:8])
            return None

        if post.status != "draft":
            logger.warning("schedule_post: post %s is %s, not draft", post.short_id, post.status)
            return None

        if not post.content.strip():
            logger.warning("schedule_post: post %s has no content", post.short_id)
            return None

        return self.update_post(post_id, status="scheduled")

    def mark_posted(self, post_id: str, post_url: str = "") -> SocialPost | None:
        """Move a scheduled post to posted status.

        Args:
            post_id:  The post to mark as posted.
            post_url: The URL where the post was published.

        Returns:
            Updated SocialPost, or None if not found or wrong status.
        """
        post = self.get_post(post_id)
        if post is None:
            logger.warning("mark_posted: post %s not found", post_id[:8])
            return None

        if post.status != "scheduled":
            logger.warning("mark_posted: post %s is %s, not scheduled", post.short_id, post.status)
            return None

        kwargs = {"status": "posted"}
        if post_url:
            kwargs["post_url"] = post_url
        return self.update_post(post_id, **kwargs)

    def mark_failed(self, post_id: str, error: str = "") -> SocialPost | None:
        """Move any post to failed status with an error message.

        Args:
            post_id: The post that failed.
            error:   Description of what went wrong.

        Returns:
            Updated SocialPost, or None if not found.
        """
        post = self.get_post(post_id)
        if post is None:
            return None

        meta = post.metadata.copy()
        meta["last_error"] = error
        meta["failed_at"] = self._now_iso()
        return self.update_post(post_id, status="failed", metadata=meta)

    # ------------------------------------------------------------------
    # Content generation (async, router)
    # ------------------------------------------------------------------

    async def draft_post(
        self,
        topic: str,
        platform: str,
        extra_context: str = "",
    ) -> SocialPost:
        """Generate a platform-appropriate post via the model router.

        Uses 'routine' complexity to hit local Ollama (free).

        Args:
            topic:         Subject of the post.
            platform:      Target platform.
            extra_context: Additional context to include in the prompt.

        Returns:
            Newly created SocialPost with generated content and hashtags.
        """
        if platform not in PLATFORMS:
            platform = "x_twitter"

        constraints = PLATFORM_CONSTRAINTS[platform]
        content = ""
        hashtags = []

        if self._router:
            template = _PLATFORM_PROMPTS.get(platform, _PLATFORM_PROMPTS["x_twitter"])
            extra_line = f"Extra context: {extra_context}" if extra_context else ""
            prompt = template.format(topic=topic, extra=extra_line)

            try:
                response = await self._router.route(
                    prompt,
                    task_complexity="routine",
                    system_prompt=_SYSTEM_PROMPT,
                )
                raw_text = response.text.strip()
                content, hashtags = _extract_hashtags(raw_text)

                # Enforce character limits
                max_chars = constraints["max_chars"]
                if len(content) > max_chars:
                    content = content[:max_chars - 3] + "..."

                # Enforce hashtag limits
                max_tags = constraints["max_hashtags"]
                hashtags = hashtags[:max_tags]

            except Exception as e:
                logger.error("draft_post failed for %s/%s: %s", platform, topic, e)
                content = f"[Draft generation failed: {e}]"
        else:
            content = f"Draft post about: {topic}"

        return self.create_post(
            platform=platform,
            topic=topic,
            content=content,
            hashtags=hashtags,
            metadata={"generated": True, "extra_context": extra_context},
        )

    async def draft_batch(
        self,
        topic: str,
        platforms: list[str] | None = None,
        extra_context: str = "",
    ) -> list[SocialPost]:
        """Generate posts for multiple platforms from a single topic.

        Calls draft_post sequentially (local Ollama is single-threaded).

        Args:
            topic:         Subject of the posts.
            platforms:      List of target platforms, or None for all.
            extra_context: Additional context for prompts.

        Returns:
            List of newly created SocialPost objects.
        """
        if platforms is None:
            platforms = list(PLATFORMS)

        posts = []
        for platform in platforms:
            if platform in PLATFORMS:
                post = await self.draft_post(topic, platform, extra_context)
                posts.append(post)

        return posts

    # ------------------------------------------------------------------
    # Schedule generation
    # ------------------------------------------------------------------

    def generate_weekly_schedule(
        self,
        frequency: dict[str, int] | None = None,
        start_date: str | None = None,
    ) -> list[dict]:
        """Generate a weekly posting schedule with staggered times.

        Returns time slots only — does NOT auto-create posts.
        Respects Tier 2: George decides when to create content.

        Args:
            frequency:  Posts per week per platform.  Defaults to DEFAULT_FREQUENCY.
            start_date: ISO date string (YYYY-MM-DD) for week start.  Defaults to next Monday.

        Returns:
            List of {platform, day, time, datetime} slot dicts, sorted by datetime.
        """
        freq = frequency or DEFAULT_FREQUENCY.copy()

        # Determine start date (next Monday)
        if start_date:
            start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        else:
            now = datetime.now(timezone.utc)
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            start = (now + timedelta(days=days_until_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        # Staggered posting hours per platform
        platform_hours = {
            "x_twitter": [9, 12, 15, 17, 19],
            "instagram": [10, 14, 18],
            "facebook": [11, 15, 19],
            "youtube_community": [12, 16],
        }

        slots = []
        for platform, count in freq.items():
            if platform not in PLATFORMS or count <= 0:
                continue
            hours = platform_hours.get(platform, [12])
            days = list(range(7))
            # Spread posts across the week
            day_step = max(1, 7 // count)
            for i in range(count):
                day_idx = (i * day_step) % 7
                hour = hours[i % len(hours)]
                slot_dt = start + timedelta(days=days[day_idx], hours=hour)
                slots.append({
                    "platform": platform,
                    "day": slot_dt.strftime("%A"),
                    "time": slot_dt.strftime("%H:%M"),
                    "datetime": slot_dt.isoformat(),
                })

        # Sort by datetime
        slots.sort(key=lambda s: s["datetime"])
        return slots

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, int]:
        """Summary counts for the status bar."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM social_posts GROUP BY status"
        ).fetchall()
        counts = {r["status"]: r["cnt"] for r in rows}
        return {
            "total": sum(counts.values()),
            "drafts": counts.get("draft", 0),
            "scheduled": counts.get("scheduled", 0),
            "posted": counts.get("posted", 0),
            "failed": counts.get("failed", 0),
        }

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Full state for WS broadcast to dashboard clients.

        Returns:
            Dict with posts list and status counts.
        """
        posts = self.list_posts()
        return {
            "posts": [p.to_dict() for p in posts],
            "status": self.get_status(),
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("SocialMediaManager closed")
