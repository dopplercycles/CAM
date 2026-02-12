"""
CAM Video Processing Pipeline

Handles raw footage import, metadata extraction via ffprobe, thumbnail
generation via ffmpeg, Pillow-based thumbnail quality scoring, and format
conversion.  Sits alongside the content pipeline and YouTube manager as
the media preparation layer.

ffmpeg / ffprobe are **soft dependencies** — detected at init via
``shutil.which()``.  All operations that need them degrade gracefully
when the tools are absent.

Usage:
    from tools.content.video_processor import VideoProcessor

    vp = VideoProcessor(
        db_path="data/media.db",
        on_change=broadcast_media_status,
        router=orchestrator.router,
    )
    new_files = vp.scan_footage()
"""

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.video_processor")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".flv", ".wmv", ".m4v", ".mpg", ".mpeg",
}

MEDIA_STATUSES = ("pending", "scanning", "processing", "ready", "failed")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class MediaFile:
    """Represents a single media file tracked by the video processor."""

    media_id: str
    filename: str
    source_path: str
    file_size: int = 0
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    fps: float | None = None
    audio_codec: str | None = None
    format: str = ""
    thumbnail_dir: str | None = None
    best_thumbnail: str | None = None
    transcoded_path: str | None = None
    status: str = "pending"
    pipeline_id: str | None = None
    youtube_video_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """First 8 chars of the UUID for display."""
        return self.media_id[:8]

    @property
    def resolution(self) -> str:
        """Human-readable resolution string."""
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON / WS transport."""
        return {
            "media_id": self.media_id,
            "short_id": self.short_id,
            "filename": self.filename,
            "source_path": self.source_path,
            "file_size": self.file_size,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "fps": self.fps,
            "audio_codec": self.audio_codec,
            "format": self.format,
            "resolution": self.resolution,
            "thumbnail_dir": self.thumbnail_dir,
            "best_thumbnail": self.best_thumbnail,
            "transcoded_path": self.transcoded_path,
            "status": self.status,
            "pipeline_id": self.pipeline_id,
            "youtube_video_id": self.youtube_video_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "MediaFile":
        """Build a MediaFile from a sqlite3.Row."""
        d = dict(row)
        return cls(
            media_id=d["media_id"],
            filename=d["filename"],
            source_path=d["source_path"],
            file_size=d.get("file_size", 0),
            duration=d.get("duration"),
            width=d.get("width"),
            height=d.get("height"),
            codec=d.get("codec"),
            fps=d.get("fps"),
            audio_codec=d.get("audio_codec"),
            format=d.get("format", ""),
            thumbnail_dir=d.get("thumbnail_dir"),
            best_thumbnail=d.get("best_thumbnail"),
            transcoded_path=d.get("transcoded_path"),
            status=d.get("status", "pending"),
            pipeline_id=d.get("pipeline_id"),
            youtube_video_id=d.get("youtube_video_id"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            metadata=_parse_json_dict(d.get("metadata")),
        )


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
# VideoProcessor
# ---------------------------------------------------------------------------

class VideoProcessor:
    """Manages video file import, metadata extraction, thumbnails, and transcoding.

    Args:
        db_path:        Path to SQLite database file.
        watch_dir:      Directory to scan for new video files.
        thumbnail_dir:  Base directory for generated thumbnails.
        processed_dir:  Directory for transcoded output files.
        on_change:      Async callback fired after any state change (for WS broadcast).
        router:         ModelRouter instance for thumbnail suggestion.
    """

    def __init__(
        self,
        db_path: str = "data/media.db",
        watch_dir: str = "data/media/import",
        thumbnail_dir: str = "data/media/thumbnails",
        processed_dir: str = "data/media/processed",
        on_change: Callable[[], Awaitable[None]] | None = None,
        router: Any = None,
    ):
        self._db_path = db_path
        self._watch_dir = watch_dir
        self._thumbnail_dir = thumbnail_dir
        self._processed_dir = processed_dir
        self._on_change = on_change
        self._router = router

        # Detect external tools at init
        self._ffprobe = shutil.which("ffprobe")
        self._ffmpeg = shutil.which("ffmpeg")

        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(watch_dir, exist_ok=True)
        os.makedirs(thumbnail_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        # Database
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

        logger.info(
            "VideoProcessor initialized (db=%s, ffmpeg=%s, ffprobe=%s)",
            db_path,
            self._ffmpeg is not None,
            self._ffprobe is not None,
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self):
        """Create tables and indexes if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS media_files (
                media_id        TEXT PRIMARY KEY,
                filename        TEXT NOT NULL,
                source_path     TEXT NOT NULL,
                file_size       INTEGER DEFAULT 0,
                duration        REAL,
                width           INTEGER,
                height          INTEGER,
                codec           TEXT,
                fps             REAL,
                audio_codec     TEXT,
                format          TEXT DEFAULT '',
                thumbnail_dir   TEXT,
                best_thumbnail  TEXT,
                transcoded_path TEXT,
                status          TEXT NOT NULL DEFAULT 'pending',
                pipeline_id     TEXT,
                youtube_video_id TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                metadata        TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_media_status
                ON media_files(status);
            CREATE INDEX IF NOT EXISTS idx_media_source
                ON media_files(source_path);
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
                logger.debug("VideoProcessor on_change callback error", exc_info=True)

    def _update_field(self, media_id: str, **kwargs) -> MediaFile | None:
        """Update arbitrary fields on a media record.

        JSON fields (metadata) are serialized automatically.
        Returns the updated MediaFile, or None if not found.
        """
        media = self.get_media(media_id)
        if media is None:
            return None

        sets = []
        vals = []
        for key, val in kwargs.items():
            if key == "metadata":
                val = json.dumps(val)
            sets.append(f"{key} = ?")
            vals.append(val)

        if not sets:
            return media

        sets.append("updated_at = ?")
        vals.append(self._now_iso())
        vals.append(media_id)

        self._conn.execute(
            f"UPDATE media_files SET {', '.join(sets)} WHERE media_id = ?",
            vals,
        )
        self._conn.commit()
        return self.get_media(media_id)

    # ------------------------------------------------------------------
    # 1. Scan — find new footage in watch_dir
    # ------------------------------------------------------------------

    def scan_footage(self) -> list[MediaFile]:
        """Walk watch_dir for video files not already tracked in the DB.

        Creates DB entries with status='pending' for each new file found.

        Returns:
            List of newly created MediaFile records.
        """
        new_files: list[MediaFile] = []
        watch = Path(self._watch_dir)

        if not watch.exists():
            logger.warning("Watch directory does not exist: %s", self._watch_dir)
            return new_files

        # Collect existing source_paths for fast lookup
        rows = self._conn.execute("SELECT source_path FROM media_files").fetchall()
        existing_paths = {r["source_path"] for r in rows}

        for root, _dirs, files in os.walk(watch):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VIDEO_EXTENSIONS:
                    continue

                full_path = os.path.join(root, fname)
                if full_path in existing_paths:
                    continue

                # Create new entry
                media_id = str(uuid.uuid4())
                now = self._now_iso()
                try:
                    file_size = os.path.getsize(full_path)
                except OSError:
                    file_size = 0

                # Detect container format from extension
                fmt = ext.lstrip(".")

                self._conn.execute(
                    """INSERT INTO media_files
                       (media_id, filename, source_path, file_size, format,
                        status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
                    (media_id, fname, full_path, file_size, fmt, now, now),
                )
                existing_paths.add(full_path)

                media = MediaFile(
                    media_id=media_id,
                    filename=fname,
                    source_path=full_path,
                    file_size=file_size,
                    format=fmt,
                    status="pending",
                    created_at=now,
                    updated_at=now,
                )
                new_files.append(media)
                logger.info("Scanned new file: %s (%s)", fname, media.short_id)

        self._conn.commit()
        logger.info("Scan complete — %d new file(s) found", len(new_files))
        return new_files

    # ------------------------------------------------------------------
    # 2. Metadata extraction via ffprobe
    # ------------------------------------------------------------------

    def extract_metadata(self, media_id: str) -> MediaFile | None:
        """Run ffprobe to extract duration, resolution, codecs, fps.

        Returns:
            Updated MediaFile, or None if not found or ffprobe unavailable.
        """
        media = self.get_media(media_id)
        if media is None:
            logger.warning("extract_metadata: media %s not found", media_id[:8])
            return None

        if not self._ffprobe:
            logger.warning("ffprobe not available — cannot extract metadata for %s", media.short_id)
            meta = media.metadata.copy()
            meta["error"] = "ffprobe not installed"
            return self._update_field(media_id, status="failed", metadata=meta)

        self._update_field(media_id, status="scanning")

        try:
            proc = subprocess.run(
                [
                    self._ffprobe,
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    media.source_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if proc.returncode != 0:
                meta = media.metadata.copy()
                meta["ffprobe_error"] = proc.stderr.strip()
                logger.error("ffprobe failed for %s: %s", media.short_id, proc.stderr.strip())
                return self._update_field(media_id, status="failed", metadata=meta)

            probe = json.loads(proc.stdout)
        except subprocess.TimeoutExpired:
            logger.error("ffprobe timed out for %s", media.short_id)
            meta = media.metadata.copy()
            meta["error"] = "ffprobe timeout"
            return self._update_field(media_id, status="failed", metadata=meta)
        except json.JSONDecodeError as e:
            logger.error("ffprobe output not valid JSON for %s: %s", media.short_id, e)
            meta = media.metadata.copy()
            meta["error"] = f"ffprobe JSON parse error: {e}"
            return self._update_field(media_id, status="failed", metadata=meta)

        # Parse format info
        fmt_info = probe.get("format", {})
        duration = None
        if fmt_info.get("duration"):
            try:
                duration = float(fmt_info["duration"])
            except (ValueError, TypeError):
                pass

        container = fmt_info.get("format_name", "").split(",")[0] or media.format

        # Parse video / audio stream info
        width = None
        height = None
        codec = None
        fps = None
        audio_codec = None

        for stream in probe.get("streams", []):
            codec_type = stream.get("codec_type", "")
            if codec_type == "video" and codec is None:
                width = stream.get("width")
                height = stream.get("height")
                codec = stream.get("codec_name")
                # Parse fps from r_frame_rate (e.g. "30000/1001")
                r_fps = stream.get("r_frame_rate", "")
                if "/" in r_fps:
                    parts = r_fps.split("/")
                    try:
                        fps = round(float(parts[0]) / float(parts[1]), 2)
                    except (ValueError, ZeroDivisionError):
                        pass
                elif r_fps:
                    try:
                        fps = float(r_fps)
                    except ValueError:
                        pass
            elif codec_type == "audio" and audio_codec is None:
                audio_codec = stream.get("codec_name")

        updated = self._update_field(
            media_id,
            duration=duration,
            width=width,
            height=height,
            codec=codec,
            fps=fps,
            audio_codec=audio_codec,
            format=container,
            status="pending",
        )

        logger.info(
            "Extracted metadata for %s: %s %s %.1fs",
            media.short_id,
            codec or "?",
            f"{width}x{height}" if width and height else "?",
            duration or 0,
        )
        return updated

    # ------------------------------------------------------------------
    # 3. Thumbnail generation via ffmpeg
    # ------------------------------------------------------------------

    def generate_thumbnails(self, media_id: str, count: int = 5) -> list[str]:
        """Generate evenly-spaced thumbnail images from a video.

        Args:
            media_id: The media file to extract thumbnails from.
            count:    Number of thumbnails to generate (default 5).

        Returns:
            List of paths to generated thumbnail images.
        """
        media = self.get_media(media_id)
        if media is None:
            logger.warning("generate_thumbnails: media %s not found", media_id[:8])
            return []

        if not self._ffmpeg:
            logger.warning("ffmpeg not available — cannot generate thumbnails for %s", media.short_id)
            return []

        if not media.duration or media.duration <= 0:
            logger.warning("No duration info for %s — cannot generate thumbnails", media.short_id)
            return []

        # Create thumbnail output directory
        thumb_dir = os.path.join(self._thumbnail_dir, media_id)
        os.makedirs(thumb_dir, exist_ok=True)

        self._update_field(media_id, status="processing", thumbnail_dir=thumb_dir)

        # Calculate evenly-spaced timestamps (skip first/last 5%)
        start = media.duration * 0.05
        end = media.duration * 0.95
        interval = (end - start) / max(count - 1, 1)
        timestamps = [start + (i * interval) for i in range(count)]

        generated: list[str] = []
        for i, ts in enumerate(timestamps):
            output = os.path.join(thumb_dir, f"thumb_{i:03d}.jpg")
            try:
                proc = subprocess.run(
                    [
                        self._ffmpeg,
                        "-ss", f"{ts:.2f}",
                        "-i", media.source_path,
                        "-frames:v", "1",
                        "-q:v", "2",
                        output,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if proc.returncode == 0 and os.path.exists(output):
                    generated.append(output)
                else:
                    logger.warning("Thumbnail %d failed for %s: %s", i, media.short_id, proc.stderr.strip())
            except subprocess.TimeoutExpired:
                logger.warning("Thumbnail %d timed out for %s", i, media.short_id)
            except Exception as e:
                logger.warning("Thumbnail %d error for %s: %s", i, media.short_id, e)

        # Restore status to pending (processing done, not fully "ready" yet)
        self._update_field(media_id, status="pending")

        logger.info("Generated %d/%d thumbnails for %s", len(generated), count, media.short_id)
        return generated

    # ------------------------------------------------------------------
    # 4. Thumbnail quality scoring + suggestion
    # ------------------------------------------------------------------

    def _score_thumbnail(self, path: str) -> dict[str, float]:
        """Compute quality metrics for a single thumbnail using Pillow.

        Metrics:
            brightness — mean of RGB channel means (0-255)
            contrast   — mean of RGB channel standard deviations
            sharpness  — mean of edge-detected grayscale image

        Returns:
            Dict with brightness, contrast, sharpness, composite scores.
        """
        try:
            from PIL import Image, ImageFilter, ImageStat
        except ImportError:
            logger.warning("Pillow not available — cannot score thumbnails")
            return {"brightness": 0, "contrast": 0, "sharpness": 0, "composite": 0}

        try:
            img = Image.open(path).convert("RGB")
            stat = ImageStat.Stat(img)

            # Brightness: mean of channel means (0-255)
            brightness = sum(stat.mean) / 3.0

            # Contrast: mean of channel stddevs
            contrast = sum(stat.stddev) / 3.0

            # Sharpness: mean pixel value of edge-detected grayscale
            edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            sharpness = edge_stat.mean[0]

            # Composite score (weighted)
            composite = (brightness * 0.2) + (contrast * 0.4) + (sharpness * 0.4)

            return {
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "sharpness": round(sharpness, 2),
                "composite": round(composite, 2),
            }
        except Exception as e:
            logger.warning("Failed to score thumbnail %s: %s", path, e)
            return {"brightness": 0, "contrast": 0, "sharpness": 0, "composite": 0}

    async def suggest_best_thumbnail(self, media_id: str) -> str | None:
        """Analyze candidate thumbnails and pick the best one.

        Uses Pillow ImageStat + edge detection to compute quality metrics.
        If a model router is available, sends computed metrics as text for
        selection (router is text-only). Otherwise picks by highest
        composite score.

        Returns:
            Path to the selected best thumbnail, or None if no candidates.
        """
        media = self.get_media(media_id)
        if media is None or not media.thumbnail_dir:
            return None

        # Gather candidate thumbnails
        thumb_dir = Path(media.thumbnail_dir)
        if not thumb_dir.exists():
            return None

        candidates = sorted(thumb_dir.glob("thumb_*.jpg"))
        if not candidates:
            return None

        # Score each candidate
        scored: list[tuple[str, dict[str, float]]] = []
        for cpath in candidates:
            scores = self._score_thumbnail(str(cpath))
            scored.append((str(cpath), scores))

        # Try model router if available
        best_path = None
        if self._router and len(scored) > 1:
            try:
                # Build text summary of scores for the router
                lines = ["Pick the best thumbnail from these candidates based on their quality metrics."]
                lines.append("Higher contrast and sharpness are preferred. Avoid very dark or blown-out images.")
                lines.append("")
                for i, (path, sc) in enumerate(scored):
                    lines.append(
                        f"Candidate {i}: brightness={sc['brightness']}, "
                        f"contrast={sc['contrast']}, sharpness={sc['sharpness']}, "
                        f"composite={sc['composite']}"
                    )
                lines.append("")
                lines.append("Reply with ONLY the candidate number (e.g. '2').")

                response = await self._router.route(
                    "\n".join(lines),
                    task_complexity="routine",
                    system_prompt="You are a video thumbnail quality analyst. Pick the best candidate.",
                )

                # Parse the response — look for a number
                text = response.text.strip()
                for token in text.split():
                    token = token.strip(".,;:()[]")
                    if token.isdigit():
                        idx = int(token)
                        if 0 <= idx < len(scored):
                            best_path = scored[idx][0]
                            logger.info("Router selected thumbnail candidate %d for %s", idx, media.short_id)
                            break
            except Exception as e:
                logger.debug("Router thumbnail suggestion failed, falling back to composite: %s", e)

        # Fallback: pick highest composite score
        if best_path is None:
            best_path = max(scored, key=lambda x: x[1]["composite"])[0]
            logger.info("Selected thumbnail by composite score for %s", media.short_id)

        self._update_field(media_id, best_thumbnail=best_path)
        return best_path

    # ------------------------------------------------------------------
    # 5. Transcode via ffmpeg
    # ------------------------------------------------------------------

    def transcode(
        self,
        media_id: str,
        target_format: str = "mp4",
        target_resolution: str | None = None,
    ) -> MediaFile | None:
        """Transcode a video to a different format or resolution.

        Args:
            media_id:          The media file to transcode.
            target_format:     Target container format (default 'mp4').
            target_resolution: Optional resolution like '1920x1080'.

        Returns:
            Updated MediaFile with transcoded_path set, or None on failure.
        """
        media = self.get_media(media_id)
        if media is None:
            logger.warning("transcode: media %s not found", media_id[:8])
            return None

        if not self._ffmpeg:
            logger.warning("ffmpeg not available — cannot transcode %s", media.short_id)
            meta = media.metadata.copy()
            meta["error"] = "ffmpeg not installed"
            return self._update_field(media_id, metadata=meta)

        # Skip if already same format and no resolution change requested
        if media.format == target_format and not target_resolution:
            logger.info("Skipping transcode for %s — already %s", media.short_id, target_format)
            return media

        self._update_field(media_id, status="processing")

        output_path = os.path.join(self._processed_dir, f"{media_id}.{target_format}")

        cmd = [
            self._ffmpeg,
            "-i", media.source_path,
            "-y",  # overwrite output
        ]

        if target_resolution:
            cmd.extend(["-vf", f"scale={target_resolution.replace('x', ':')}"])

        cmd.append(output_path)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for large files
            )

            if proc.returncode != 0:
                logger.error("Transcode failed for %s: %s", media.short_id, proc.stderr[-500:] if proc.stderr else "")
                meta = media.metadata.copy()
                meta["transcode_error"] = proc.stderr[-500:] if proc.stderr else "unknown"
                return self._update_field(media_id, status="failed", metadata=meta)

        except subprocess.TimeoutExpired:
            logger.error("Transcode timed out for %s", media.short_id)
            meta = media.metadata.copy()
            meta["error"] = "transcode timeout"
            return self._update_field(media_id, status="failed", metadata=meta)

        updated = self._update_field(media_id, transcoded_path=output_path, status="pending")
        logger.info("Transcoded %s → %s", media.short_id, output_path)
        return updated

    # ------------------------------------------------------------------
    # 6–10. CRUD / Status / Linking
    # ------------------------------------------------------------------

    def get_media(self, media_id: str) -> MediaFile | None:
        """Fetch a single media file by ID.

        Returns:
            MediaFile or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM media_files WHERE media_id = ?", (media_id,)
        ).fetchone()
        return MediaFile.from_row(row) if row else None

    def list_media(self, status: str | None = None, limit: int = 50) -> list[MediaFile]:
        """List media files, optionally filtered by status.

        Args:
            status: Filter by status, or None for all.
            limit:  Max results to return.

        Returns:
            List of MediaFile sorted by updated_at descending.
        """
        if status and status in MEDIA_STATUSES:
            rows = self._conn.execute(
                "SELECT * FROM media_files WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM media_files ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [MediaFile.from_row(r) for r in rows]

    def delete_media(self, media_id: str) -> bool:
        """Delete a media file from the database and remove its thumbnail directory.

        Returns:
            True if a row was deleted, False if not found.
        """
        media = self.get_media(media_id)
        if media is None:
            return False

        # Remove thumbnail directory if it exists
        if media.thumbnail_dir and os.path.isdir(media.thumbnail_dir):
            try:
                shutil.rmtree(media.thumbnail_dir)
                logger.info("Removed thumbnail dir for %s", media.short_id)
            except OSError as e:
                logger.warning("Failed to remove thumbnail dir for %s: %s", media.short_id, e)

        cur = self._conn.execute(
            "DELETE FROM media_files WHERE media_id = ?", (media_id,)
        )
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Deleted media %s", media_id[:8])
        return deleted

    def link_pipeline(self, media_id: str, pipeline_id: str) -> MediaFile | None:
        """Link a media file to a content pipeline entry.

        Returns:
            Updated MediaFile, or None if not found.
        """
        return self._update_field(media_id, pipeline_id=pipeline_id)

    def link_youtube(self, media_id: str, youtube_video_id: str) -> MediaFile | None:
        """Link a media file to a YouTube video entry.

        Returns:
            Updated MediaFile, or None if not found.
        """
        return self._update_field(media_id, youtube_video_id=youtube_video_id)

    # ------------------------------------------------------------------
    # 11. Process all — convenience method
    # ------------------------------------------------------------------

    async def process_media(self, media_id: str) -> MediaFile | None:
        """Run the full processing pipeline for a single media file.

        Steps: extract_metadata → generate_thumbnails → suggest_best_thumbnail → mark ready.
        Catches errors at each step; marks failed on unrecoverable error.

        Returns:
            Updated MediaFile, or None if not found.
        """
        media = self.get_media(media_id)
        if media is None:
            logger.warning("process_media: media %s not found", media_id[:8])
            return None

        logger.info("Starting full processing for %s (%s)", media.short_id, media.filename)

        # Step 1: Extract metadata
        try:
            result = self.extract_metadata(media_id)
            if result is None or result.status == "failed":
                logger.error("Metadata extraction failed for %s", media.short_id)
                return self.get_media(media_id)
        except Exception as e:
            logger.error("Metadata extraction error for %s: %s", media.short_id, e, exc_info=True)
            meta = media.metadata.copy()
            meta["error"] = f"metadata extraction: {e}"
            return self._update_field(media_id, status="failed", metadata=meta)

        # Step 2: Generate thumbnails (requires duration from step 1)
        try:
            thumbs = self.generate_thumbnails(media_id)
            if not thumbs:
                logger.warning("No thumbnails generated for %s (ffmpeg missing or no duration)", media.short_id)
        except Exception as e:
            logger.error("Thumbnail generation error for %s: %s", media.short_id, e, exc_info=True)
            # Non-fatal — continue

        # Step 3: Suggest best thumbnail
        try:
            if thumbs:
                await self.suggest_best_thumbnail(media_id)
        except Exception as e:
            logger.error("Thumbnail suggestion error for %s: %s", media.short_id, e, exc_info=True)
            # Non-fatal — continue

        # Mark ready
        updated = self._update_field(media_id, status="ready")
        logger.info("Processing complete for %s — status: ready", media.short_id)
        return updated

    # ------------------------------------------------------------------
    # 12–14. Dashboard integration
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, int]:
        """Summary counts for the status bar.

        Returns:
            Dict with total, pending, processing, ready, failed counts.
        """
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM media_files GROUP BY status"
        ).fetchall()
        counts = {r["status"]: r["cnt"] for r in rows}
        return {
            "total": sum(counts.values()),
            "pending": counts.get("pending", 0),
            "scanning": counts.get("scanning", 0),
            "processing": counts.get("processing", 0),
            "ready": counts.get("ready", 0),
            "failed": counts.get("failed", 0),
        }

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Full state for WS broadcast to dashboard clients.

        Returns:
            Dict with media list, status counts, and tool availability.
        """
        media = self.list_media()
        return {
            "media": [m.to_dict() for m in media],
            "status": self.get_status(),
            "tools": {
                "ffmpeg": self._ffmpeg is not None,
                "ffprobe": self._ffprobe is not None,
            },
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("VideoProcessor closed")
