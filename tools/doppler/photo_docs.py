"""
Photo Documentation Pipeline for Doppler Cycles.

Provides before/during/after service photography management with:
- Watched folder import (data/photos/import/)
- Automatic thumbnail generation via Pillow (300x300)
- EXIF metadata extraction (date, camera, GPS)
- Service record tagging (customer, vehicle, stage)
- Before/after comparison composite generation
- Customer-ready gallery HTML builder
- Content-worthy flagging for the content pipeline

SQLite-backed, single-file module -- same pattern as invoicing.py.
"""

import base64
import json
import logging
import os
import shutil
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHOTO_STAGES = ("before", "during", "after")
PHOTO_STATUSES = ("imported", "tagged", "processed", "archived")
THUMB_MAX_SIZE = (300, 300)
COMPOSITE_WIDTH = 1200   # px, side-by-side comparison
IMPORT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


# ---------------------------------------------------------------------------
# PhotoRecord dataclass
# ---------------------------------------------------------------------------

@dataclass
class PhotoRecord:
    """A single photo in the documentation system."""
    photo_id: str = ""
    filename: str = ""
    original_path: str = ""
    thumbnail_path: str = ""
    file_size: int = 0
    width: int = 0
    height: int = 0
    stage: str = ""                    # before/during/after
    service_record_id: str = ""
    customer_id: str = ""
    customer_name: str = ""
    vehicle_id: str = ""
    vehicle_name: str = ""
    description: str = ""
    content_worthy: int = 0            # boolean flag (0/1)
    status: str = "imported"
    taken_at: str = ""                 # EXIF date or import date
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.photo_id[:8] if self.photo_id else ""

    def to_dict(self) -> dict:
        return {
            "photo_id": self.photo_id,
            "filename": self.filename,
            "original_path": self.original_path,
            "thumbnail_path": self.thumbnail_path,
            "file_size": self.file_size,
            "width": self.width,
            "height": self.height,
            "stage": self.stage,
            "service_record_id": self.service_record_id,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "vehicle_id": self.vehicle_id,
            "vehicle_name": self.vehicle_name,
            "description": self.description,
            "content_worthy": self.content_worthy,
            "status": self.status,
            "taken_at": self.taken_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_row(row) -> "PhotoRecord":
        """Build a PhotoRecord from a sqlite3.Row."""
        r = dict(row)
        return PhotoRecord(
            photo_id=r["photo_id"],
            filename=r.get("filename", ""),
            original_path=r.get("original_path", ""),
            thumbnail_path=r.get("thumbnail_path", ""),
            file_size=int(r.get("file_size", 0)),
            width=int(r.get("width", 0)),
            height=int(r.get("height", 0)),
            stage=r.get("stage", ""),
            service_record_id=r.get("service_record_id", ""),
            customer_id=r.get("customer_id", ""),
            customer_name=r.get("customer_name", ""),
            vehicle_id=r.get("vehicle_id", ""),
            vehicle_name=r.get("vehicle_name", ""),
            description=r.get("description", ""),
            content_worthy=int(r.get("content_worthy", 0)),
            status=r.get("status", "imported"),
            taken_at=r.get("taken_at", ""),
            created_at=r.get("created_at", ""),
            updated_at=r.get("updated_at", ""),
            metadata=json.loads(r.get("metadata", "{}")),
        )


# ---------------------------------------------------------------------------
# PhotoDocManager
# ---------------------------------------------------------------------------

class PhotoDocManager:
    """SQLite-backed photo documentation manager for Doppler Cycles."""

    def __init__(
        self,
        db_path: str = "data/photo_docs.db",
        photos_dir: str = "data/photos",
        on_change: Optional[Callable[[], Coroutine]] = None,
        service_store: Any = None,
        crm_store: Any = None,
    ):
        self._db_path = db_path
        self._photos_dir = Path(photos_dir)
        self._on_change = on_change
        self._service_store = service_store
        self._crm_store = crm_store

        # Create directory structure
        for subdir in ("import", "originals", "thumbnails", "composites", "galleries"):
            (self._photos_dir / subdir).mkdir(parents=True, exist_ok=True)

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("PhotoDocManager initialized (db=%s, photos=%s)", db_path, photos_dir)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS photos (
                photo_id          TEXT PRIMARY KEY,
                filename          TEXT NOT NULL,
                original_path     TEXT NOT NULL,
                thumbnail_path    TEXT DEFAULT '',
                file_size         INTEGER DEFAULT 0,
                width             INTEGER DEFAULT 0,
                height            INTEGER DEFAULT 0,
                stage             TEXT DEFAULT '',
                service_record_id TEXT DEFAULT '',
                customer_id       TEXT DEFAULT '',
                customer_name     TEXT DEFAULT '',
                vehicle_id        TEXT DEFAULT '',
                vehicle_name      TEXT DEFAULT '',
                description       TEXT DEFAULT '',
                content_worthy    INTEGER DEFAULT 0,
                status            TEXT DEFAULT 'imported',
                taken_at          TEXT DEFAULT '',
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                metadata          TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_photo_service ON photos(service_record_id);
            CREATE INDEX IF NOT EXISTS idx_photo_customer ON photos(customer_id);
            CREATE INDEX IF NOT EXISTS idx_photo_stage ON photos(stage);
            CREATE INDEX IF NOT EXISTS idx_photo_status ON photos(status);
            CREATE INDEX IF NOT EXISTS idx_photo_content ON photos(content_worthy);
            CREATE INDEX IF NOT EXISTS idx_photo_created ON photos(created_at);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_photos(self) -> list[PhotoRecord]:
        """Scan import/ folder and ingest any image files found.

        For each image:
        - Generate UUID
        - Read dimensions with Pillow
        - Extract EXIF date
        - Move original to originals/YYYY-MM-DD/{filename}
        - Generate thumbnail
        - Insert DB row with status='imported'

        Returns list of newly imported PhotoRecords.
        """
        import_dir = self._photos_dir / "import"
        imported = []

        for entry in sorted(import_dir.iterdir()):
            if not entry.is_file():
                continue
            ext = entry.suffix.lower()
            if ext not in IMPORT_EXTENSIONS:
                continue

            try:
                photo = self._import_single(entry)
                if photo:
                    imported.append(photo)
            except Exception as exc:
                logger.error("Failed to import %s: %s", entry.name, exc)

        if imported:
            logger.info("Imported %d photo(s) from %s", len(imported), import_dir)
        return imported

    def _import_single(self, file_path: Path) -> Optional[PhotoRecord]:
        """Import a single image file."""
        photo_id = str(uuid.uuid4())
        now = datetime.now().isoformat(timespec="seconds")

        # Read image dimensions and EXIF
        try:
            with Image.open(file_path) as img:
                # Handle EXIF rotation before reading size
                img = ImageOps.exif_transpose(img)
                width, height = img.size
        except Exception as exc:
            logger.warning("Cannot open image %s: %s", file_path.name, exc)
            return None

        exif_data = self._extract_exif(file_path)
        taken_at = exif_data.get("taken_at", now)
        file_size = file_path.stat().st_size

        # Move original to organized directory
        date_folder = datetime.now().strftime("%Y-%m-%d")
        dest_dir = self._photos_dir / "originals" / date_folder
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Handle filename collisions
        dest_path = dest_dir / file_path.name
        if dest_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            dest_path = dest_dir / f"{stem}_{photo_id[:8]}{suffix}"

        shutil.move(str(file_path), str(dest_path))

        # Generate thumbnail
        thumb_path = self._generate_thumbnail(dest_path, photo_id)

        rec = PhotoRecord(
            photo_id=photo_id,
            filename=file_path.name,
            original_path=str(dest_path),
            thumbnail_path=thumb_path,
            file_size=file_size,
            width=width,
            height=height,
            status="imported",
            taken_at=taken_at,
            created_at=now,
            updated_at=now,
            metadata=exif_data,
        )

        self._conn.execute(
            """INSERT INTO photos (
                photo_id, filename, original_path, thumbnail_path,
                file_size, width, height, stage, service_record_id,
                customer_id, customer_name, vehicle_id, vehicle_name,
                description, content_worthy, status, taken_at,
                created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.photo_id, rec.filename, rec.original_path, rec.thumbnail_path,
                rec.file_size, rec.width, rec.height, rec.stage, rec.service_record_id,
                rec.customer_id, rec.customer_name, rec.vehicle_id, rec.vehicle_name,
                rec.description, rec.content_worthy, rec.status, rec.taken_at,
                rec.created_at, rec.updated_at, json.dumps(rec.metadata),
            ),
        )
        self._conn.commit()
        logger.info("Photo imported: %s (%s) %dx%d", rec.filename, rec.short_id, width, height)
        return rec

    # ------------------------------------------------------------------
    # Thumbnail generation
    # ------------------------------------------------------------------

    def _generate_thumbnail(self, source_path: Path, photo_id: str) -> str:
        """Generate a 300x300 max thumbnail, saved as JPEG.

        Handles EXIF rotation via ImageOps.exif_transpose().
        Returns the thumbnail file path as a string.
        """
        thumb_path = self._photos_dir / "thumbnails" / f"{photo_id}.jpg"
        try:
            with Image.open(source_path) as img:
                img = ImageOps.exif_transpose(img)
                # Convert to RGB if needed (e.g. RGBA PNGs, palette images)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.thumbnail(THUMB_MAX_SIZE)
                img.save(str(thumb_path), "JPEG", quality=85)
        except Exception as exc:
            logger.warning("Thumbnail generation failed for %s: %s", photo_id[:8], exc)
            return ""
        return str(thumb_path)

    # ------------------------------------------------------------------
    # EXIF extraction
    # ------------------------------------------------------------------

    def _extract_exif(self, path: Path) -> dict:
        """Extract useful EXIF data from an image.

        Returns dict with taken_at, camera, and any GPS info found.
        Stored in the photo's metadata JSON field.
        """
        result = {}
        try:
            with Image.open(path) as img:
                exif_raw = img._getexif()
                if not exif_raw:
                    return result

                exif = {}
                for tag_id, value in exif_raw.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    # Skip binary/large data
                    if isinstance(value, bytes) and len(value) > 100:
                        continue
                    try:
                        # Ensure JSON-serializable
                        json.dumps(value)
                        exif[tag_name] = value
                    except (TypeError, ValueError):
                        exif[tag_name] = str(value)

                # Extract date taken
                for date_tag in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                    if date_tag in exif:
                        try:
                            dt = datetime.strptime(str(exif[date_tag]), "%Y:%m:%d %H:%M:%S")
                            result["taken_at"] = dt.isoformat(timespec="seconds")
                        except (ValueError, TypeError):
                            pass
                        break

                # Camera model
                if "Model" in exif:
                    result["camera"] = str(exif["Model"]).strip()
                if "Make" in exif:
                    result["camera_make"] = str(exif["Make"]).strip()

                # GPS info (if present)
                if "GPSInfo" in exif:
                    result["has_gps"] = True

        except Exception as exc:
            logger.debug("EXIF extraction failed for %s: %s", path.name, exc)

        return result

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def tag_photo(
        self,
        photo_id: str,
        service_record_id: str = "",
        stage: str = "",
        description: str = "",
        content_worthy: int = 0,
    ) -> Optional[PhotoRecord]:
        """Tag a photo with service record, stage, and description.

        Validates stage, looks up service record for customer/vehicle info,
        updates status to 'tagged', and logs to CRM notes if customer found.
        """
        photo = self.get_photo(photo_id)
        if photo is None:
            return None

        # Validate stage
        if stage and stage not in PHOTO_STAGES:
            logger.warning("Invalid stage '%s' — must be one of %s", stage, PHOTO_STAGES)
            return None

        now = datetime.now().isoformat(timespec="seconds")
        updates = {
            "stage": stage or photo.stage,
            "description": description or photo.description,
            "content_worthy": content_worthy,
            "status": "tagged",
            "updated_at": now,
        }

        # Look up service record for customer/vehicle info
        if service_record_id and self._service_store:
            record = self._service_store.get_record(service_record_id)
            if record:
                updates["service_record_id"] = service_record_id
                updates["customer_id"] = getattr(record, "customer_id", "")
                updates["customer_name"] = getattr(record, "customer_name", "")
                updates["vehicle_id"] = getattr(record, "vehicle_id", "")
                updates["vehicle_name"] = getattr(record, "vehicle_name", "")
        elif service_record_id:
            updates["service_record_id"] = service_record_id

        # Apply updates
        set_clauses = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [photo_id]
        self._conn.execute(
            f"UPDATE photos SET {set_clauses} WHERE photo_id=?", values
        )
        self._conn.commit()

        # Log to CRM notes if customer found
        customer_id = updates.get("customer_id", photo.customer_id)
        if customer_id and self._crm_store:
            try:
                self._crm_store.add_note(
                    customer_id,
                    f"Photo documented: {photo.filename} — stage: {stage or 'unset'}, "
                    f"description: {description or 'none'}",
                    category="documentation",
                )
            except Exception as exc:
                logger.debug("CRM note failed: %s", exc)

        updated = self.get_photo(photo_id)
        logger.info("Photo tagged: %s → stage=%s, service=%s",
                     photo.short_id, stage, service_record_id[:8] if service_record_id else "none")
        return updated

    def tag_photos_batch(
        self,
        photo_ids: list[str],
        service_record_id: str = "",
        stage: str = "",
    ) -> list[PhotoRecord]:
        """Batch version of tag_photo — tags multiple photos with same service record and stage."""
        results = []
        for pid in photo_ids:
            result = self.tag_photo(pid, service_record_id=service_record_id, stage=stage)
            if result:
                results.append(result)
        return results

    # ------------------------------------------------------------------
    # Comparison composites
    # ------------------------------------------------------------------

    def generate_comparison(self, service_record_id: str) -> Optional[str]:
        """Create a side-by-side before/after composite for a service record.

        Finds the first 'before' and first 'after' photo, resizes both to
        the same height, concatenates horizontally, and adds stage labels.

        Returns the composite image path, or None if photos are missing.
        """
        before_photos = self._conn.execute(
            "SELECT * FROM photos WHERE service_record_id=? AND stage='before' ORDER BY created_at LIMIT 1",
            (service_record_id,),
        ).fetchone()
        after_photos = self._conn.execute(
            "SELECT * FROM photos WHERE service_record_id=? AND stage='after' ORDER BY created_at LIMIT 1",
            (service_record_id,),
        ).fetchone()

        if not before_photos or not after_photos:
            logger.warning("Cannot generate comparison for %s — need both before and after photos",
                         service_record_id[:8])
            return None

        before_rec = PhotoRecord.from_row(before_photos)
        after_rec = PhotoRecord.from_row(after_photos)

        try:
            before_img = Image.open(before_rec.original_path)
            before_img = ImageOps.exif_transpose(before_img)
            after_img = Image.open(after_rec.original_path)
            after_img = ImageOps.exif_transpose(after_img)
        except Exception as exc:
            logger.error("Failed to open images for comparison: %s", exc)
            return None

        # Convert to RGB
        if before_img.mode != "RGB":
            before_img = before_img.convert("RGB")
        if after_img.mode != "RGB":
            after_img = after_img.convert("RGB")

        # Resize both to same height, fitting within COMPOSITE_WIDTH
        half_width = COMPOSITE_WIDTH // 2
        target_height = 600  # reasonable comparison height

        before_img = self._fit_image(before_img, half_width, target_height)
        after_img = self._fit_image(after_img, half_width, target_height)

        # Ensure same height (pad shorter one)
        max_h = max(before_img.height, after_img.height)
        if before_img.height < max_h:
            before_img = self._pad_to_height(before_img, max_h)
        if after_img.height < max_h:
            after_img = self._pad_to_height(after_img, max_h)

        # Concatenate horizontally
        composite = Image.new("RGB", (before_img.width + after_img.width, max_h), (30, 30, 30))
        composite.paste(before_img, (0, 0))
        composite.paste(after_img, (before_img.width, 0))

        # Add labels
        draw = ImageDraw.Draw(composite)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except (OSError, IOError):
            font = ImageFont.load_default()

        label_y = max_h - 50
        # BEFORE label (white text on dark semi-transparent bar)
        draw.rectangle([(0, label_y - 5), (before_img.width, max_h)], fill=(0, 0, 0, 180))
        draw.text((20, label_y), "BEFORE", fill=(255, 255, 255), font=font)
        # AFTER label
        draw.rectangle([(before_img.width, label_y - 5), (composite.width, max_h)], fill=(0, 0, 0, 180))
        draw.text((before_img.width + 20, label_y), "AFTER", fill=(255, 255, 255), font=font)

        # Save composite
        composite_path = self._photos_dir / "composites" / f"{service_record_id}.jpg"
        composite.save(str(composite_path), "JPEG", quality=90)

        before_img.close()
        after_img.close()

        logger.info("Comparison composite generated for service %s", service_record_id[:8])
        return str(composite_path)

    @staticmethod
    def _fit_image(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
        """Resize image to fit within max_width x max_height, maintaining aspect ratio."""
        ratio = min(max_width / img.width, max_height / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        return img.resize(new_size, Image.LANCZOS)

    @staticmethod
    def _pad_to_height(img: Image.Image, target_height: int) -> Image.Image:
        """Pad an image vertically to reach target_height, centering content."""
        if img.height >= target_height:
            return img
        padded = Image.new("RGB", (img.width, target_height), (30, 30, 30))
        y_offset = (target_height - img.height) // 2
        padded.paste(img, (0, y_offset))
        return padded

    # ------------------------------------------------------------------
    # Gallery builder
    # ------------------------------------------------------------------

    def build_gallery(self, service_record_id: str) -> Optional[str]:
        """Generate a standalone HTML gallery for a service record.

        Includes all photos grouped by stage, with base64-encoded thumbnails
        for portability. Includes service details header.

        Returns the gallery HTML file path, or None if no photos found.
        """
        photos = self.get_service_photos(service_record_id)
        if not photos:
            logger.warning("No photos found for gallery: %s", service_record_id[:8])
            return None

        # Group by stage
        by_stage = {"before": [], "during": [], "after": [], "untagged": []}
        for p in photos:
            stage_key = p.stage if p.stage in PHOTO_STAGES else "untagged"
            by_stage[stage_key].append(p)

        # Get service record details
        service_info = ""
        if self._service_store:
            record = self._service_store.get_record(service_record_id)
            if record:
                customer = getattr(record, "customer_name", "Customer")
                vehicle = getattr(record, "vehicle_name", "Vehicle")
                svc_date = getattr(record, "date", "")
                services = getattr(record, "services_performed", []) or []
                service_info = f"""
                <div class="service-header">
                    <h2>{customer} — {vehicle}</h2>
                    <p>Date: {svc_date}</p>
                    <p>Services: {', '.join(services) if services else 'N/A'}</p>
                </div>"""

        # Build HTML with inline CSS and base64 thumbnails
        html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Service Photo Gallery — Doppler Cycles</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #1a1a2e; color: #e0e0e0; padding: 2rem; }}
    .header {{ text-align: center; margin-bottom: 2rem; }}
    .header h1 {{ color: #00d4ff; font-size: 1.8rem; }}
    .header p {{ color: #888; margin-top: 0.5rem; }}
    .service-header {{ background: #16213e; padding: 1rem 1.5rem; border-radius: 8px;
                       margin-bottom: 2rem; border-left: 3px solid #00d4ff; }}
    .service-header h2 {{ color: #fff; font-size: 1.2rem; }}
    .service-header p {{ color: #aaa; margin-top: 0.3rem; font-size: 0.9rem; }}
    .stage-section {{ margin-bottom: 2rem; }}
    .stage-title {{ font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;
                    padding: 0.5rem 1rem; border-radius: 6px; display: inline-block; }}
    .stage-before {{ background: #1e3a5f; color: #5b9bd5; }}
    .stage-during {{ background: #3d2e00; color: #f0ad4e; }}
    .stage-after {{ background: #1a3e1a; color: #5cb85c; }}
    .stage-untagged {{ background: #333; color: #999; }}
    .photo-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                   gap: 1rem; }}
    .photo-card {{ background: #16213e; border-radius: 8px; overflow: hidden;
                   border: 1px solid #2a2a4a; }}
    .photo-card img {{ width: 100%; height: 200px; object-fit: cover; }}
    .photo-info {{ padding: 0.75rem; }}
    .photo-info .filename {{ font-weight: 600; font-size: 0.85rem; color: #ddd; }}
    .photo-info .desc {{ color: #888; font-size: 0.8rem; margin-top: 0.3rem; }}
    .photo-info .meta {{ color: #666; font-size: 0.75rem; margin-top: 0.3rem; }}
    .footer {{ text-align: center; margin-top: 3rem; color: #555; font-size: 0.8rem; }}
</style>
</head>
<body>
<div class="header">
    <h1>DOPPLER CYCLES</h1>
    <p>Service Photo Documentation</p>
</div>
{service_info}
"""]

        stage_labels = {
            "before": ("Before Service", "stage-before"),
            "during": ("During Service", "stage-during"),
            "after": ("After Service", "stage-after"),
            "untagged": ("Untagged", "stage-untagged"),
        }

        for stage_key in ("before", "during", "after", "untagged"):
            stage_photos = by_stage[stage_key]
            if not stage_photos:
                continue

            label, css_class = stage_labels[stage_key]
            html_parts.append(f"""
<div class="stage-section">
    <span class="stage-title {css_class}">{label}</span>
    <div class="photo-grid">""")

            for p in stage_photos:
                # Encode thumbnail as base64
                thumb_b64 = ""
                thumb_file = Path(p.thumbnail_path) if p.thumbnail_path else None
                if thumb_file and thumb_file.exists():
                    try:
                        thumb_b64 = base64.b64encode(thumb_file.read_bytes()).decode("ascii")
                    except Exception:
                        pass

                img_src = f"data:image/jpeg;base64,{thumb_b64}" if thumb_b64 else ""
                desc_html = f'<div class="desc">{p.description}</div>' if p.description else ""
                meta_parts = []
                if p.width and p.height:
                    meta_parts.append(f"{p.width}x{p.height}")
                if p.file_size:
                    size_kb = p.file_size / 1024
                    meta_parts.append(f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB")
                if p.taken_at:
                    meta_parts.append(p.taken_at[:10])
                meta_html = f'<div class="meta">{" · ".join(meta_parts)}</div>' if meta_parts else ""

                html_parts.append(f"""
        <div class="photo-card">
            {'<img src="' + img_src + '" alt="' + p.filename + '">' if img_src else '<div style="height:200px;background:#222;display:flex;align-items:center;justify-content:center;color:#555;">No thumbnail</div>'}
            <div class="photo-info">
                <div class="filename">{p.filename}</div>
                {desc_html}
                {meta_html}
            </div>
        </div>""")

            html_parts.append("""
    </div>
</div>""")

        html_parts.append(f"""
<div class="footer">
    Generated by Doppler Cycles — {datetime.now().strftime("%Y-%m-%d %H:%M")}
</div>
</body>
</html>""")

        gallery_path = self._photos_dir / "galleries" / f"{service_record_id}.html"
        gallery_path.write_text("\n".join(html_parts), encoding="utf-8")

        logger.info("Gallery built for service %s (%d photos)", service_record_id[:8], len(photos))
        return str(gallery_path)

    # ------------------------------------------------------------------
    # CRUD + queries
    # ------------------------------------------------------------------

    def get_photo(self, photo_id: str) -> Optional[PhotoRecord]:
        """Fetch a single photo by ID."""
        row = self._conn.execute(
            "SELECT * FROM photos WHERE photo_id = ?", (photo_id,)
        ).fetchone()
        return PhotoRecord.from_row(row) if row else None

    def delete_photo(self, photo_id: str) -> bool:
        """Delete a photo — removes files (original + thumbnail) and DB row."""
        photo = self.get_photo(photo_id)
        if photo is None:
            return False

        # Delete original file
        if photo.original_path and Path(photo.original_path).exists():
            try:
                os.remove(photo.original_path)
            except OSError as exc:
                logger.warning("Could not delete original %s: %s", photo.original_path, exc)

        # Delete thumbnail
        if photo.thumbnail_path and Path(photo.thumbnail_path).exists():
            try:
                os.remove(photo.thumbnail_path)
            except OSError as exc:
                logger.warning("Could not delete thumbnail %s: %s", photo.thumbnail_path, exc)

        self._conn.execute("DELETE FROM photos WHERE photo_id = ?", (photo_id,))
        self._conn.commit()
        logger.info("Photo deleted: %s (%s)", photo.filename, photo.short_id)
        return True

    def update_photo(self, photo_id: str, **kwargs) -> Optional[PhotoRecord]:
        """Update arbitrary fields on a photo record."""
        photo = self.get_photo(photo_id)
        if photo is None:
            return None

        # Filter to valid fields only
        valid_fields = {
            "filename", "stage", "service_record_id", "customer_id", "customer_name",
            "vehicle_id", "vehicle_name", "description", "content_worthy", "status",
            "taken_at", "metadata",
        }
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if not updates:
            return photo

        updates["updated_at"] = datetime.now().isoformat(timespec="seconds")

        # Serialize metadata if dict
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])

        set_clauses = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [photo_id]
        self._conn.execute(f"UPDATE photos SET {set_clauses} WHERE photo_id=?", values)
        self._conn.commit()

        logger.info("Photo updated: %s", photo_id[:8])
        return self.get_photo(photo_id)

    def list_photos(
        self,
        customer_id: str = "",
        service_record_id: str = "",
        stage: str = "",
        status: str = "",
        content_worthy: Optional[int] = None,
        limit: int = 200,
    ) -> list[PhotoRecord]:
        """List photos with optional filters."""
        clauses = []
        params: list = []

        if customer_id:
            clauses.append("customer_id = ?")
            params.append(customer_id)
        if service_record_id:
            clauses.append("service_record_id = ?")
            params.append(service_record_id)
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if content_worthy is not None:
            clauses.append("content_worthy = ?")
            params.append(content_worthy)

        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM photos WHERE {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [PhotoRecord.from_row(r) for r in rows]

    def get_content_worthy(self) -> list[PhotoRecord]:
        """Get all photos flagged as content-worthy."""
        rows = self._conn.execute(
            "SELECT * FROM photos WHERE content_worthy = 1 ORDER BY created_at DESC"
        ).fetchall()
        return [PhotoRecord.from_row(r) for r in rows]

    def get_service_photos(self, service_record_id: str) -> list[PhotoRecord]:
        """Get all photos for a specific service record."""
        rows = self._conn.execute(
            "SELECT * FROM photos WHERE service_record_id = ? ORDER BY stage, created_at",
            (service_record_id,),
        ).fetchall()
        return [PhotoRecord.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Summary status dict for dashboard stat cards."""
        total_row = self._conn.execute("SELECT COUNT(*) as c FROM photos").fetchone()

        counts = {}
        for s in PHOTO_STATUSES:
            row = self._conn.execute(
                "SELECT COUNT(*) as c FROM photos WHERE status = ?", (s,)
            ).fetchone()
            counts[s] = int(row["c"])

        stage_counts = {}
        for st in PHOTO_STAGES:
            row = self._conn.execute(
                "SELECT COUNT(*) as c FROM photos WHERE stage = ?", (st,)
            ).fetchone()
            stage_counts[st] = int(row["c"])

        cw_row = self._conn.execute(
            "SELECT COUNT(*) as c FROM photos WHERE content_worthy = 1"
        ).fetchone()

        return {
            "total": int(total_row["c"]),
            "imported": counts.get("imported", 0),
            "tagged": counts.get("tagged", 0),
            "processed": counts.get("processed", 0),
            "content_worthy": int(cw_row["c"]),
            "by_stage": stage_counts,
        }

    def to_broadcast_dict(self) -> dict:
        """Full state for dashboard WS broadcast."""
        return {
            "photos": [p.to_dict() for p in self.list_photos(limit=200)],
            "status": self.get_status(),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close the database connection."""
        self._conn.close()
        logger.info("PhotoDocManager closed")
