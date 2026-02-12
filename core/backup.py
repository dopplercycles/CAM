"""
CAM Backup & Recovery System

Creates timestamped tar.gz archives of all critical data:
- SQLite databases (using safe online backup API)
- ChromaDB vector store
- Config files and knowledge docs
- State files (working memory, schedules)

Supports rotation (keeps last N backups), restore from archive,
and status reporting for the dashboard.

No external dependencies — uses stdlib tarfile, sqlite3, shutil, tempfile.
"""

import logging
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("cam.backup")

# Files to back up, relative to project root
_SQLITE_DBS = [
    "data/analytics.db",
    "data/memory/episodic.db",
    "data/business.db",
    "data/security_audit.db",
    "data/scout.db",
    "data/content_calendar.db",
    "data/research.db",
]

_CHROMADB_DIR = "data/memory/chromadb"

_CONFIG_FILES = [
    "config/settings.toml",
    "config/persona.yaml",
]

_KNOWLEDGE_FILES = [
    "CAM_BRAIN.md",
]

_STATE_FILES = [
    "data/tasks/working_memory.json",
    "data/schedules.json",
]


class BackupManager:
    """Manages backup creation, rotation, restore, and status reporting.

    Args:
        config: CAMConfig instance (reads config.backup.* settings).
        event_logger: EventLogger for recording backup events.
    """

    def __init__(self, config, event_logger):
        self._config = config
        self._event_logger = event_logger
        self._project_root = Path(__file__).resolve().parent.parent
        self._backup_dir = self._project_root / config.backup.backup_dir
        self._max_backups = config.backup.max_backups
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._last_backup: dict | None = None

    def backup(self) -> dict:
        """Create a timestamped backup archive of all critical data.

        Steps:
        1. Create temp directory
        2. Safe-copy SQLite databases using sqlite3 backup API
        3. Copy ChromaDB directory tree
        4. Copy config, knowledge, and state files
        5. Create tar.gz archive
        6. Clean up temp dir
        7. Rotate old backups

        Returns:
            {"ok": True, "filename": ..., "size": ..., "file_count": ..., "timestamp": ...}
            or {"ok": False, "error": ...} on failure.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_name = f"cam_backup_{timestamp}.tar.gz"
        archive_path = self._backup_dir / archive_name
        file_count = 0

        try:
            with tempfile.TemporaryDirectory(prefix="cam_backup_") as tmp:
                tmp_path = Path(tmp)

                # --- SQLite databases: safe online backup ---
                for rel in _SQLITE_DBS:
                    src = self._project_root / rel
                    if not src.exists():
                        logger.debug("Skipping missing DB: %s", rel)
                        continue
                    dst = tmp_path / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        src_conn = sqlite3.connect(str(src))
                        dst_conn = sqlite3.connect(str(dst))
                        src_conn.backup(dst_conn)
                        dst_conn.close()
                        src_conn.close()
                        file_count += 1
                    except Exception as e:
                        logger.warning("Failed to backup DB %s: %s", rel, e)

                # --- ChromaDB vector store: copy entire directory ---
                chromadb_src = self._project_root / _CHROMADB_DIR
                if chromadb_src.exists():
                    chromadb_dst = tmp_path / _CHROMADB_DIR
                    shutil.copytree(chromadb_src, chromadb_dst)
                    file_count += sum(1 for _ in chromadb_dst.rglob("*") if _.is_file())

                # --- Config, knowledge, and state files ---
                for rel in _CONFIG_FILES + _KNOWLEDGE_FILES + _STATE_FILES:
                    src = self._project_root / rel
                    if not src.exists():
                        logger.debug("Skipping missing file: %s", rel)
                        continue
                    dst = tmp_path / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    file_count += 1

                # --- Create tar.gz archive ---
                with tarfile.open(archive_path, "w:gz") as tar:
                    for item in tmp_path.iterdir():
                        tar.add(item, arcname=item.name)

            # --- Rotate old backups ---
            self._rotate()

            size = archive_path.stat().st_size
            result = {
                "ok": True,
                "filename": archive_name,
                "size": size,
                "file_count": file_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._last_backup = result

            self._event_logger.info(
                "backup",
                f"Backup created: {archive_name} ({self._human_size(size)}, {file_count} files)",
            )
            logger.info("Backup created: %s (%s, %d files)", archive_name, self._human_size(size), file_count)
            return result

        except Exception as e:
            logger.error("Backup failed: %s", e)
            self._event_logger.error("backup", f"Backup failed: {e}")
            return {"ok": False, "error": str(e)}

    def restore(self, filename: str) -> dict:
        """Restore data from a backup archive.

        Extracts the archive and copies files back to their original
        locations. A server restart is needed after restore to reload
        databases.

        Args:
            filename: Name of the backup archive (e.g. cam_backup_20260211_030000.tar.gz)

        Returns:
            {"ok": True, "filename": ..., "files_restored": ...}
            or {"ok": False, "error": ...} on failure.
        """
        archive_path = self._backup_dir / filename
        if not archive_path.exists():
            return {"ok": False, "error": f"Backup not found: {filename}"}

        # Safety: only allow restoring .tar.gz files from the backup dir
        if not filename.endswith(".tar.gz") or "/" in filename or "\\" in filename:
            return {"ok": False, "error": "Invalid backup filename"}

        files_restored = 0
        try:
            with tempfile.TemporaryDirectory(prefix="cam_restore_") as tmp:
                tmp_path = Path(tmp)

                # Extract archive
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=tmp_path)

                # Copy extracted files back to project root
                for item in tmp_path.rglob("*"):
                    if item.is_file():
                        rel = item.relative_to(tmp_path)
                        dst = self._project_root / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dst)
                        files_restored += 1

            result = {
                "ok": True,
                "filename": filename,
                "files_restored": files_restored,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Restore complete. Restart the server to reload databases.",
            }
            self._event_logger.info(
                "backup",
                f"Restored from {filename} ({files_restored} files). Restart needed.",
            )
            logger.info("Restored from %s (%d files)", filename, files_restored)
            return result

        except Exception as e:
            logger.error("Restore failed: %s", e)
            self._event_logger.error("backup", f"Restore from {filename} failed: {e}")
            return {"ok": False, "error": str(e)}

    def list_backups(self) -> list[dict]:
        """List all backup archives, newest first.

        Returns:
            [{"filename": ..., "size": ..., "created_at": ...}, ...]
        """
        backups = []
        for f in sorted(self._backup_dir.glob("cam_backup_*.tar.gz"), reverse=True):
            stat = f.stat()
            backups.append({
                "filename": f.name,
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            })
        return backups

    def get_status(self) -> dict:
        """Return backup system status for the dashboard.

        Returns:
            {"last_backup": {...} | None, "count": int, "total_size": int, "backups": [...]}
        """
        backups = self.list_backups()
        total_size = sum(b["size"] for b in backups)
        return {
            "last_backup": self._last_backup or (backups[0] if backups else None),
            "count": len(backups),
            "total_size": total_size,
            "backups": backups,
        }

    def _rotate(self):
        """Delete oldest backups, keeping only the last max_backups."""
        archives = sorted(self._backup_dir.glob("cam_backup_*.tar.gz"))
        while len(archives) > self._max_backups:
            oldest = archives.pop(0)
            oldest.unlink()
            logger.info("Rotated old backup: %s", oldest.name)

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Format byte count as human-readable string."""
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
