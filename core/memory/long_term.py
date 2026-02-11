"""
CAM Long-Term Memory

Persistent knowledge base backed by ChromaDB vector store. Stores task
results, user preferences, system learnings, and domain knowledge as
vector embeddings so CAM can retrieve semantically relevant memories
during the THINK phase.

Think of this as the shop manual and experience logbook — accumulated
knowledge that persists across sessions and helps CAM give better
answers over time.

Usage:
    from core.memory.long_term import LongTermMemory

    ltm = LongTermMemory()
    ltm.store("Harley M-8 oil pump recall affects 2017-2020 models",
              category="knowledge")

    results = ltm.query("oil pump issues on Milwaukee-Eight")
    for entry in results:
        print(f"[{entry.category}] {entry.content} (score={entry.score:.2f})")
"""

import logging
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.memory.long_term")


# ---------------------------------------------------------------------------
# Valid memory categories
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {"task_result", "user_preference", "system_learning", "knowledge"}


# ---------------------------------------------------------------------------
# MemoryEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single entry in long-term memory.

    Attributes:
        entry_id:   Unique identifier for this memory
        content:    The text content stored
        category:   One of VALID_CATEGORIES
        score:      Similarity score (0-1) from a query, or 0.0 if stored directly
        timestamp:  When the memory was created (UTC ISO string)
        metadata:   Additional key-value data (task_id, source, etc.)
    """
    entry_id: str
    content: str
    category: str
    score: float = 0.0
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for JSON/dashboard use."""
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "category": self.category,
            "score": self.score,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------

class LongTermMemory:
    """ChromaDB-backed persistent vector store for CAM's long-term knowledge.

    Stores text content as vector embeddings using ChromaDB's default
    embedding model (all-MiniLM-L6-v2). Supports semantic similarity
    search, category filtering, and graceful degradation if ChromaDB
    is not available.

    Args:
        persist_directory: Path where ChromaDB stores its data on disk.
        collection_name:   Name of the ChromaDB collection to use.
    """

    def __init__(
        self,
        persist_directory: str = "data/memory/chromadb",
        collection_name: str = "cam_long_term",
    ):
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._initialized = False
        self._error: str | None = None

        logger.info(
            "LongTermMemory created (persist_dir=%s, collection=%s)",
            persist_directory, collection_name,
        )

    # -------------------------------------------------------------------
    # Lazy initialization — ChromaDB is loaded on first use
    # -------------------------------------------------------------------

    def _ensure_initialized(self) -> bool:
        """Initialize ChromaDB client and collection on first use.

        Returns True if ready, False if initialization failed.
        Subsequent calls are no-ops if already initialized.
        """
        if self._initialized:
            return True

        if self._error is not None:
            # Already tried and failed — don't retry every call
            return False

        try:
            import chromadb

            # Ensure persist directory exists
            Path(self._persist_directory).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=self._persist_directory,
            )

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            self._initialized = True
            count = self._collection.count()
            logger.info(
                "LongTermMemory initialized — %d entries in collection '%s'",
                count, self._collection_name,
            )
            return True

        except ImportError:
            self._error = "chromadb package not installed"
            logger.warning(
                "LongTermMemory degraded mode: %s. "
                "Install with: pip install chromadb",
                self._error,
            )
            return False

        except Exception as e:
            self._error = str(e)
            logger.warning(
                "LongTermMemory degraded mode: %s", self._error,
            )
            return False

    # -------------------------------------------------------------------
    # Store
    # -------------------------------------------------------------------

    def store(
        self,
        content: str,
        category: str,
        metadata: dict[str, Any] | None = None,
        entry_id: str | None = None,
    ) -> MemoryEntry | None:
        """Store a new memory entry with vector embedding.

        Args:
            content:   The text to store and embed.
            category:  One of VALID_CATEGORIES.
            metadata:  Optional extra key-value data.
            entry_id:  Optional custom ID (auto-generated if not provided).

        Returns:
            The created MemoryEntry, or None if storage failed.
        """
        if not content or not content.strip():
            logger.warning("Rejecting empty content for LTM store")
            return None

        if category not in VALID_CATEGORIES:
            logger.warning(
                "Invalid LTM category '%s' — must be one of %s",
                category, VALID_CATEGORIES,
            )
            return None

        if not self._ensure_initialized():
            return None

        entry_id = entry_id or str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Build ChromaDB metadata — values must be str, int, float, or bool
        chroma_meta = {
            "category": category,
            "timestamp": timestamp,
        }
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    chroma_meta[key] = value
                else:
                    chroma_meta[key] = str(value)

        try:
            self._collection.upsert(
                ids=[entry_id],
                documents=[content],
                metadatas=[chroma_meta],
            )

            entry = MemoryEntry(
                entry_id=entry_id,
                content=content,
                category=category,
                timestamp=timestamp,
                metadata=metadata or {},
            )

            logger.info(
                "LTM stored: [%s] %.80s (id=%s)",
                category, content, entry_id[:8],
            )
            return entry

        except Exception as e:
            logger.error("LTM store failed: %s", e)
            return None

    # -------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[MemoryEntry]:
        """Search for semantically similar memories.

        Args:
            text:     The query text to find similar entries for.
            top_k:    Maximum number of results to return.
            category: Optional category filter (only return this category).

        Returns:
            List of MemoryEntry objects sorted by similarity (highest first).
            Scores are normalized to 0-1 (1 = perfect match).
        """
        if not self._ensure_initialized():
            return []

        if not text or not text.strip():
            return []

        try:
            query_params = {
                "query_texts": [text],
                "n_results": top_k,
            }
            if category and category in VALID_CATEGORIES:
                query_params["where"] = {"category": category}

            results = self._collection.query(**query_params)

            entries = []
            if results and results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                documents = results["documents"][0] if results["documents"] else []
                distances = results["distances"][0] if results["distances"] else []
                metadatas = results["metadatas"][0] if results["metadatas"] else []

                for i, entry_id in enumerate(ids):
                    doc = documents[i] if i < len(documents) else ""
                    dist = distances[i] if i < len(distances) else 1.0
                    meta = metadatas[i] if i < len(metadatas) else {}

                    # ChromaDB cosine distance is in [0, 2].
                    # Convert to similarity score: 1 - (distance / 2) → [0, 1]
                    score = max(0.0, 1.0 - (dist / 2.0))

                    entries.append(MemoryEntry(
                        entry_id=entry_id,
                        content=doc,
                        category=meta.get("category", "unknown"),
                        score=round(score, 4),
                        timestamp=meta.get("timestamp", ""),
                        metadata={k: v for k, v in meta.items()
                                  if k not in ("category", "timestamp")},
                    ))

            logger.debug(
                "LTM query '%.60s' → %d results", text, len(entries),
            )
            return entries

        except Exception as e:
            logger.error("LTM query failed: %s", e)
            return []

    # -------------------------------------------------------------------
    # Forget
    # -------------------------------------------------------------------

    def forget(self, entry_id: str) -> bool:
        """Delete a specific memory entry by ID.

        Args:
            entry_id: The entry's unique identifier.

        Returns:
            True if the entry was found and deleted, False otherwise.
        """
        if not self._ensure_initialized():
            return False

        try:
            # Check if it exists first
            existing = self._collection.get(ids=[entry_id])
            if not existing["ids"]:
                return False

            self._collection.delete(ids=[entry_id])
            logger.info("LTM forgot entry %s", entry_id[:8])
            return True

        except Exception as e:
            logger.error("LTM forget failed: %s", e)
            return False

    # -------------------------------------------------------------------
    # Recent entries
    # -------------------------------------------------------------------

    def get_recent(self, count: int = 10) -> list[MemoryEntry]:
        """Return the most recent memory entries.

        Fetches all entries and sorts by timestamp descending.
        Used by the dashboard for display purposes.

        Args:
            count: Maximum number of entries to return.

        Returns:
            List of MemoryEntry objects, newest first.
        """
        if not self._ensure_initialized():
            return []

        try:
            total = self._collection.count()
            if total == 0:
                return []

            result = self._collection.get(
                limit=total,
                include=["documents", "metadatas"],
            )

            entries = []
            for i, entry_id in enumerate(result["ids"]):
                doc = result["documents"][i] if result["documents"] else ""
                meta = result["metadatas"][i] if result["metadatas"] else {}

                entries.append(MemoryEntry(
                    entry_id=entry_id,
                    content=doc,
                    category=meta.get("category", "unknown"),
                    timestamp=meta.get("timestamp", ""),
                    metadata={k: v for k, v in meta.items()
                              if k not in ("category", "timestamp")},
                ))

            # Sort by timestamp descending (newest first)
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            return entries[:count]

        except Exception as e:
            logger.error("LTM get_recent failed: %s", e)
            return []

    # -------------------------------------------------------------------
    # Seed from file
    # -------------------------------------------------------------------

    def seed_from_file(self, file_path: str) -> int:
        """Load seed knowledge from a markdown file on first startup.

        Splits the file on '## ' headings. Each section becomes a
        separate "knowledge" entry in the vector store. Only runs if
        the collection is currently empty — safe to call every startup.

        Args:
            file_path: Path to the markdown seed file (e.g. CAM_BRAIN.md).

        Returns:
            Number of entries seeded (0 if collection already had data).
        """
        if not self._ensure_initialized():
            return 0

        # Only seed into an empty collection
        if self._collection.count() > 0:
            logger.debug("LTM already has data, skipping seed from %s", file_path)
            return 0

        path = Path(file_path)
        if not path.exists():
            logger.warning("LTM seed file not found: %s", file_path)
            return 0

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Failed to read seed file %s: %s", file_path, e)
            return 0

        # Split on ## headings
        sections = []
        current_title = ""
        current_lines = []

        for line in text.splitlines():
            if line.startswith("## "):
                # Save previous section
                if current_title and current_lines:
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append((current_title, body))
                current_title = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Don't forget the last section
        if current_title and current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_title, body))

        seeded = 0
        for title, body in sections:
            content = f"{title}\n\n{body}"
            entry = self.store(
                content=content,
                category="knowledge",
                metadata={"source": "seed", "section": title},
            )
            if entry is not None:
                seeded += 1

        if seeded > 0:
            logger.info(
                "LTM seeded %d entries from %s", seeded, file_path,
            )
        return seeded

    # -------------------------------------------------------------------
    # Clear
    # -------------------------------------------------------------------

    def clear(self):
        """Delete all entries and recreate the collection.

        Use with caution — all memories are permanently lost.
        """
        if not self._ensure_initialized():
            return

        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("LTM cleared — collection '%s' recreated", self._collection_name)
        except Exception as e:
            logger.error("LTM clear failed: %s", e)

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a snapshot of long-term memory state.

        Used by the dashboard memory panel and orchestrator status.
        """
        if not self._ensure_initialized():
            return {
                "initialized": False,
                "error": self._error,
                "total_count": 0,
                "counts_by_category": {},
                "persist_directory": self._persist_directory,
                "collection_name": self._collection_name,
            }

        try:
            total = self._collection.count()

            # Count per category
            counts = {}
            for cat in VALID_CATEGORIES:
                try:
                    result = self._collection.get(
                        where={"category": cat},
                        include=[],
                    )
                    counts[cat] = len(result["ids"])
                except Exception:
                    counts[cat] = 0

            return {
                "initialized": True,
                "error": None,
                "total_count": total,
                "counts_by_category": counts,
                "persist_directory": self._persist_directory,
                "collection_name": self._collection_name,
            }

        except Exception as e:
            return {
                "initialized": True,
                "error": str(e),
                "total_count": 0,
                "counts_by_category": {},
                "persist_directory": self._persist_directory,
                "collection_name": self._collection_name,
            }

    def __repr__(self) -> str:
        if self._initialized:
            count = self._collection.count() if self._collection else 0
            return (
                f"LongTermMemory(entries={count}, "
                f"collection='{self._collection_name}')"
            )
        return f"LongTermMemory(initialized=False, error={self._error})"


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import shutil

    # Use a temp directory so we don't clobber real data
    tmp_dir = tempfile.mkdtemp(prefix="cam_ltm_test_")

    try:
        ltm = LongTermMemory(
            persist_directory=tmp_dir,
            collection_name="test_collection",
        )

        print(f"\n{ltm}")
        print(f"Status: {ltm.get_status()}")

        # Store some memories
        ltm.store(
            "Harley-Davidson M-8 oil pump recall affects 2017-2020 Touring models",
            category="knowledge",
            metadata={"source": "research"},
        )
        ltm.store(
            "George prefers torque specs in ft-lbs, not Nm",
            category="user_preference",
        )
        ltm.store(
            "Ducati Desmo valve adjustment requires special tools and shims",
            category="knowledge",
            metadata={"source": "experience"},
        )
        ltm.store(
            "Task completed: researched CB750 barn find restoration costs",
            category="task_result",
        )

        print(f"\nAfter storing: {ltm}")
        print(f"Status: {ltm.get_status()}")

        # Query — semantic search
        print("\n--- Query: 'Harley oil issues' ---")
        results = ltm.query("Harley oil issues", top_k=3)
        for entry in results:
            print(f"  [{entry.category}] (score={entry.score:.3f}) {entry.content[:80]}")

        # Query with category filter
        print("\n--- Query: 'preferences' (category=user_preference) ---")
        results = ltm.query("preferences", top_k=3, category="user_preference")
        for entry in results:
            print(f"  [{entry.category}] (score={entry.score:.3f}) {entry.content[:80]}")

        # Forget
        print("\n--- Recent entries ---")
        recent = ltm.get_recent(5)
        for entry in recent:
            print(f"  [{entry.category}] {entry.content[:60]}...")

        if recent:
            forgotten = ltm.forget(recent[0].entry_id)
            print(f"\nForgot most recent entry: {forgotten}")
            print(f"After forget: {ltm}")

    finally:
        shutil.rmtree(tmp_dir)
        print(f"\nCleaned up temp dir: {tmp_dir}")
