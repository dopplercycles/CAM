"""
CAM Knowledge Ingestion System

Bulk document ingestion for service manuals, technical bulletins, parts
catalogs, and motorcycle specs. Chunks documents into semantically
meaningful sections and stores them in the existing ChromaDB vector store
via LongTermMemory.store().

Supports: .md, .txt, .pdf, .csv
Sources: manual upload via dashboard, automatic scan of inbox folder.

Usage:
    from core.memory.knowledge_ingest import KnowledgeIngest

    ki = KnowledgeIngest(ltm=long_term_memory, ...)
    doc = ki.ingest_bytes(data, "DR650-manual.pdf", source="upload")
    print(f"Ingested {doc.chunk_count} chunks")
"""

import csv
import hashlib
import io
import logging
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger("cam.knowledge_ingest")


# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".csv"}


# ---------------------------------------------------------------------------
# KnowledgeDocument dataclass
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeDocument:
    """Tracks a single ingested document.

    Attributes:
        doc_id:      Unique identifier (timestamp-based)
        filename:    Original filename
        file_type:   Extension (md/txt/pdf/csv)
        status:      pending / processing / completed / failed
        chunk_count: Number of chunks stored in ChromaDB
        source:      upload / inbox
        file_hash:   SHA-256 hex digest for dedup
        created_at:  ISO timestamp
        error:       Error message if failed, else empty
    """
    doc_id: str
    filename: str
    file_type: str
    status: str = "pending"
    chunk_count: int = 0
    source: str = "upload"
    file_hash: str = ""
    created_at: str = ""
    error: str = ""

    @property
    def short_id(self) -> str:
        """First 8 chars of doc_id for display."""
        return self.doc_id[:8]

    def to_dict(self) -> dict:
        """Serialize for JSON / dashboard use."""
        return {
            "doc_id": self.doc_id,
            "short_id": self.short_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "source": self.source,
            "file_hash": self.file_hash,
            "created_at": self.created_at,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# KnowledgeIngest
# ---------------------------------------------------------------------------

class KnowledgeIngest:
    """Ingests documents into CAM's long-term memory.

    Chunks files by type, stores each chunk via ltm.store(), and tracks
    ingestion history in a local SQLite database.

    Args:
        ltm:               LongTermMemory instance (from server.py)
        db_path:           Path to SQLite tracking database
        inbox_dir:         Directory to watch for auto-ingest
        processed_dir:     Where successfully ingested inbox files are moved
        max_file_size:     Maximum file size in bytes (default 10 MB)
        chunk_target_size: Target chunk size in characters
        chunk_overlap:     Character overlap between consecutive text chunks
        on_change:         Async callback fired after ingestion state changes
    """

    def __init__(
        self,
        ltm,
        db_path: str = "data/knowledge_ingest.db",
        inbox_dir: str = "data/knowledge/inbox",
        processed_dir: str = "data/knowledge/processed",
        max_file_size: int = 10_485_760,
        chunk_target_size: int = 1000,
        chunk_overlap: int = 100,
        on_change: Callable | None = None,
    ):
        self._ltm = ltm
        self._db_path = db_path
        self._inbox_dir = Path(inbox_dir)
        self._processed_dir = Path(processed_dir)
        self._max_file_size = max_file_size
        self._chunk_target_size = chunk_target_size
        self._chunk_overlap = chunk_overlap
        self._on_change = on_change

        # Ensure directories exist
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        logger.info(
            "KnowledgeIngest initialized (db=%s, inbox=%s)",
            db_path, inbox_dir,
        )

    def _init_db(self):
        """Create the documents table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id       TEXT PRIMARY KEY,
                filename     TEXT NOT NULL,
                file_type    TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                chunk_count  INTEGER DEFAULT 0,
                source       TEXT DEFAULT 'upload',
                file_hash    TEXT DEFAULT '',
                created_at   TEXT NOT NULL,
                error        TEXT DEFAULT ''
            )
        """)
        self._conn.commit()

    # -------------------------------------------------------------------
    # Hashing / dedup
    # -------------------------------------------------------------------

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        """SHA-256 hex digest of raw file bytes."""
        return hashlib.sha256(data).hexdigest()

    def _is_duplicate(self, file_hash: str) -> bool:
        """Check if a completed document with this hash already exists."""
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE file_hash = ? AND status = 'completed' LIMIT 1",
            (file_hash,),
        ).fetchone()
        return row is not None

    # -------------------------------------------------------------------
    # Chunking strategies
    # -------------------------------------------------------------------

    def _chunk_markdown(self, text: str) -> list[str]:
        """Split markdown on ## headings (reuses seed_from_file logic).

        Falls back to _chunk_text if no headings found.
        """
        sections = []
        current_title = ""
        current_lines: list[str] = []

        for line in text.splitlines():
            if line.startswith("## "):
                if current_title and current_lines:
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append(f"{current_title}\n\n{body}")
                current_title = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Last section
        if current_title and current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append(f"{current_title}\n\n{body}")

        if not sections:
            # No ## headings found — fall back to text chunking
            return self._chunk_text(text)

        # Further split any oversized sections
        result = []
        for section in sections:
            if len(section) > self._chunk_target_size * 2:
                result.extend(self._chunk_text(section))
            else:
                result.append(section)

        return result

    def _chunk_text(self, text: str) -> list[str]:
        """Split text on paragraph boundaries, further split oversized blocks.

        Chunks target ~chunk_target_size characters with chunk_overlap overlap.
        """
        # Split on double newlines (paragraph boundaries)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: list[str] = []
        current_chunk = ""

        for para in paragraphs:
            # If adding this paragraph would exceed target, save current and start new
            if current_chunk and len(current_chunk) + len(para) + 2 > self._chunk_target_size:
                chunks.append(current_chunk)
                # Start new chunk with overlap from end of previous
                if self._chunk_overlap > 0 and len(current_chunk) > self._chunk_overlap:
                    current_chunk = current_chunk[-self._chunk_overlap:] + "\n\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk)

        # Handle any chunks that are still way too large (single huge paragraphs)
        final: list[str] = []
        for chunk in chunks:
            if len(chunk) > self._chunk_target_size * 3:
                # Force-split at target size with overlap
                pos = 0
                while pos < len(chunk):
                    end = pos + self._chunk_target_size
                    final.append(chunk[pos:end])
                    pos = end - self._chunk_overlap
            else:
                final.append(chunk)

        return [c for c in final if c.strip()]

    def _chunk_pdf(self, data: bytes) -> list[str]:
        """Extract text from PDF pages and chunk the combined text."""
        import pdfplumber

        text_parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        combined = "\n\n".join(text_parts)
        if not combined.strip():
            raise ValueError("No text could be extracted from PDF")

        return self._chunk_text(combined)

    def _chunk_csv(self, text: str) -> list[str]:
        """Convert CSV rows to individual knowledge entries.

        Each row becomes: "Col1: val1 | Col2: val2 | ..."
        """
        reader = csv.DictReader(io.StringIO(text))
        chunks = []
        for row in reader:
            parts = [f"{k}: {v}" for k, v in row.items() if v]
            if parts:
                chunks.append(" | ".join(parts))
        return chunks

    # -------------------------------------------------------------------
    # Core ingestion
    # -------------------------------------------------------------------

    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
        source: str = "upload",
    ) -> KnowledgeDocument:
        """Ingest a file's raw bytes into long-term memory.

        1. Validate size and extension
        2. Compute hash, check for duplicates
        3. Chunk by file type
        4. Store each chunk via ltm.store()
        5. Track in SQLite

        Args:
            data:     Raw file bytes
            filename: Original filename (used for extension detection)
            source:   "upload" or "inbox"

        Returns:
            KnowledgeDocument with final status
        """
        now = datetime.now(timezone.utc)
        doc_id = now.strftime("%Y%m%d%H%M%S") + "-" + hashlib.md5(
            (filename + now.isoformat()).encode()
        ).hexdigest()[:8]

        ext = Path(filename).suffix.lower()
        doc = KnowledgeDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=ext.lstrip("."),
            status="processing",
            source=source,
            created_at=now.isoformat(),
        )

        # Validate extension
        if ext not in SUPPORTED_EXTENSIONS:
            doc.status = "failed"
            doc.error = f"Unsupported file type: {ext}"
            self._save_document(doc)
            logger.warning("Rejected %s: unsupported type %s", filename, ext)
            return doc

        # Validate size
        if len(data) > self._max_file_size:
            doc.status = "failed"
            doc.error = f"File too large ({len(data)} bytes, max {self._max_file_size})"
            self._save_document(doc)
            logger.warning("Rejected %s: too large (%d bytes)", filename, len(data))
            return doc

        if len(data) == 0:
            doc.status = "failed"
            doc.error = "Empty file"
            self._save_document(doc)
            return doc

        # Compute hash and check for duplicates
        file_hash = self._compute_hash(data)
        doc.file_hash = file_hash

        if self._is_duplicate(file_hash):
            doc.status = "failed"
            doc.error = "Duplicate: file already ingested"
            self._save_document(doc)
            logger.info("Skipped duplicate %s (hash=%s)", filename, file_hash[:12])
            return doc

        # Insert as processing
        self._save_document(doc)

        try:
            # Chunk by type
            if ext == ".pdf":
                chunks = self._chunk_pdf(data)
            elif ext == ".csv":
                text = data.decode("utf-8", errors="replace")
                chunks = self._chunk_csv(text)
            elif ext == ".md":
                text = data.decode("utf-8", errors="replace")
                chunks = self._chunk_markdown(text)
            else:  # .txt
                text = data.decode("utf-8", errors="replace")
                chunks = self._chunk_text(text)

            if not chunks:
                raise ValueError("No content could be extracted from file")

            # Store each chunk in long-term memory
            stored = 0
            for i, chunk in enumerate(chunks):
                entry = self._ltm.store(
                    content=chunk,
                    category="knowledge",
                    metadata={
                        "source_file": filename,
                        "chunk_index": i,
                        "file_type": doc.file_type,
                        "doc_id": doc_id,
                    },
                )
                if entry is not None:
                    stored += 1

            doc.chunk_count = stored
            doc.status = "completed"
            self._update_document(doc)

            logger.info(
                "Ingested %s: %d chunks stored (source=%s, hash=%s)",
                filename, stored, source, file_hash[:12],
            )

        except Exception as e:
            doc.status = "failed"
            doc.error = str(e)
            self._update_document(doc)
            logger.error("Ingestion failed for %s: %s", filename, e)

        # Fire change callback
        if self._on_change:
            try:
                self._on_change()
            except Exception:
                pass  # Don't let callback errors break ingestion

        return doc

    # -------------------------------------------------------------------
    # Inbox scanning
    # -------------------------------------------------------------------

    def scan_inbox(self) -> list[KnowledgeDocument]:
        """Scan the inbox directory for supported files and ingest them.

        Successfully ingested files are moved to the processed directory.
        Returns list of KnowledgeDocument results.
        """
        results = []

        for ext in SUPPORTED_EXTENSIONS:
            for filepath in self._inbox_dir.glob(f"*{ext}"):
                if not filepath.is_file():
                    continue

                try:
                    data = filepath.read_bytes()
                    doc = self.ingest_bytes(data, filepath.name, source="inbox")
                    results.append(doc)

                    # Move to processed on success or duplicate
                    if doc.status == "completed" or (
                        doc.status == "failed" and doc.error
                        and doc.error.startswith("Duplicate")
                    ):
                        dest = self._processed_dir / filepath.name
                        # Handle name collision in processed dir
                        if dest.exists():
                            stem = dest.stem
                            suffix = dest.suffix
                            counter = 1
                            while dest.exists():
                                dest = self._processed_dir / f"{stem}_{counter}{suffix}"
                                counter += 1
                        shutil.move(str(filepath), str(dest))
                        logger.info("Moved %s to processed/ (%s)", filepath.name, doc.status)

                except Exception as e:
                    logger.error("Failed to process inbox file %s: %s", filepath.name, e)

        return results

    # -------------------------------------------------------------------
    # History / status
    # -------------------------------------------------------------------

    def get_history(self, limit: int = 100) -> list[dict]:
        """Return ingestion history, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_status(self) -> dict:
        """Summary stats for the dashboard."""
        row = self._conn.execute("""
            SELECT
                COUNT(*) as total_documents,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'completed' THEN chunk_count ELSE 0 END) as total_chunks
            FROM documents
        """).fetchone()

        # Count files in inbox
        inbox_pending = 0
        try:
            for ext in SUPPORTED_EXTENSIONS:
                inbox_pending += len(list(self._inbox_dir.glob(f"*{ext}")))
        except Exception:
            pass

        return {
            "total_documents": row["total_documents"] or 0,
            "completed": row["completed"] or 0,
            "failed": row["failed"] or 0,
            "total_chunks": row["total_chunks"] or 0,
            "inbox_pending": inbox_pending,
        }

    def to_broadcast_dict(self) -> dict:
        """Full state for dashboard WebSocket broadcast."""
        return {
            "documents": self.get_history(),
            "status": self.get_status(),
        }

    # -------------------------------------------------------------------
    # SQLite helpers
    # -------------------------------------------------------------------

    def _save_document(self, doc: KnowledgeDocument):
        """Insert a new document row."""
        self._conn.execute("""
            INSERT OR REPLACE INTO documents
                (doc_id, filename, file_type, status, chunk_count, source, file_hash, created_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc.doc_id, doc.filename, doc.file_type, doc.status,
            doc.chunk_count, doc.source, doc.file_hash, doc.created_at, doc.error,
        ))
        self._conn.commit()

    def _update_document(self, doc: KnowledgeDocument):
        """Update an existing document row."""
        self._conn.execute("""
            UPDATE documents SET
                status = ?, chunk_count = ?, file_hash = ?, error = ?
            WHERE doc_id = ?
        """, (doc.status, doc.chunk_count, doc.file_hash, doc.error, doc.doc_id))
        self._conn.commit()

    def close(self):
        """Close the SQLite connection."""
        try:
            self._conn.close()
            logger.info("KnowledgeIngest database closed")
        except Exception:
            pass
