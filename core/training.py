"""
Training and Learning Mode for CAM / Doppler Cycles.

Captures George's real-time diagnostic expertise as structured training
episodes: symptoms → tests → readings → conclusions → notes.  Over time
these episodes build a queryable knowledge base that makes decades of
wrench-time persistent, searchable, and available for pattern matching.

SQLite-backed, single-file module — same pattern as photo_docs.py and
invoicing.py.  Integrates with:
  - EpisodicMemory  (chronological audit trail for every step)
  - LongTermMemory  (ChromaDB vector store for semantic search)
  - ModelRouter      (local Ollama for pattern extraction)
  - DiagnosticEngine (link sessions to decision trees)
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_TYPES = ("symptom", "test", "reading", "conclusion", "note")
SESSION_STATUSES = ("active", "completed", "archived")
CATEGORIES = (
    "electrical",
    "engine",
    "fuel",
    "suspension",
    "drivetrain",
    "brakes",
    "general",
)

PATTERN_EXTRACTION_PROMPT = """\
You are a motorcycle diagnostic expert assistant.  Analyze the following
training session recorded by a master technician.  Extract any reusable
diagnostic patterns.

For each pattern found, return a JSON array of objects with these keys:
  - symptom:  the presenting symptom or complaint
  - cause:    the root cause identified
  - diagnostic_path:  brief description of how the tech got from symptom to cause
  - confidence:  "high", "medium", or "low"

Return ONLY the JSON array — no markdown fences, no commentary.

--- SESSION ---
Title: {title}
Vehicle: {vehicle_info}
Category: {category}

Steps:
{steps_text}
--- END SESSION ---
"""

EXPERIENCE_QUERY_PROMPT = """\
You are a motorcycle diagnostic expert assistant.  A technician is seeing
the symptoms below and wants to know what past experience suggests.

Symptoms: {symptoms}
Category: {category}

Here are relevant past patterns and sessions from the knowledge base:

{context}

Based on this experience, provide:
1. Most likely causes (ranked by confidence)
2. Recommended diagnostic steps
3. Things to watch out for

Be concise and practical — this is shop-floor advice.
"""

TREE_UPDATE_PROMPT = """\
You are a motorcycle diagnostic expert assistant.  Based on the training
patterns below, suggest updates to the diagnostic decision trees for the
"{category}" category.

Current tree nodes:
{tree_nodes}

Training patterns (from real sessions):
{patterns_text}

Suggest specific new branches, questions, or findings that should be added
to the trees.  Return a JSON array of suggestion objects:
  - tree_id:  which tree to update (or "new" for a new tree)
  - node_type:  "question" or "finding"
  - text:  the question or finding text
  - rationale:  why this should be added
  - parent_node:  suggested parent node id (or "start" for root-level)

Return ONLY the JSON array.
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingSession:
    """A single training/learning session capturing diagnostic expertise."""
    session_id: str = ""
    title: str = ""
    vehicle_info: str = ""
    category: str = "general"
    status: str = "active"
    steps: list = field(default_factory=list)
    findings_summary: str = ""
    extracted_patterns: list = field(default_factory=list)
    linked_tree_id: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    completed_at: str = ""

    @property
    def short_id(self) -> str:
        return self.session_id[:8] if self.session_id else ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "vehicle_info": self.vehicle_info,
            "category": self.category,
            "status": self.status,
            "steps": self.steps,
            "findings_summary": self.findings_summary,
            "extracted_patterns": self.extracted_patterns,
            "linked_tree_id": self.linked_tree_id,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }

    @staticmethod
    def from_row(row) -> "TrainingSession":
        """Build a TrainingSession from a sqlite3.Row."""
        r = dict(row)
        return TrainingSession(
            session_id=r["session_id"],
            title=r.get("title", ""),
            vehicle_info=r.get("vehicle_info", ""),
            category=r.get("category", "general"),
            status=r.get("status", "active"),
            steps=json.loads(r.get("steps", "[]")),
            findings_summary=r.get("findings_summary", ""),
            extracted_patterns=json.loads(r.get("extracted_patterns", "[]")),
            linked_tree_id=r.get("linked_tree_id", ""),
            notes=r.get("notes", ""),
            created_at=r.get("created_at", ""),
            updated_at=r.get("updated_at", ""),
            completed_at=r.get("completed_at", ""),
        )


@dataclass
class TrainingPattern:
    """A reusable diagnostic pattern extracted from training sessions."""
    pattern_id: str = ""
    session_id: str = ""
    category: str = ""
    symptom: str = ""
    cause: str = ""
    diagnostic_path: str = ""
    confidence: str = "medium"
    frequency: int = 1
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.pattern_id[:8] if self.pattern_id else ""

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "session_id": self.session_id,
            "category": self.category,
            "symptom": self.symptom,
            "cause": self.cause,
            "diagnostic_path": self.diagnostic_path,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_row(row) -> "TrainingPattern":
        """Build a TrainingPattern from a sqlite3.Row."""
        r = dict(row)
        return TrainingPattern(
            pattern_id=r["pattern_id"],
            session_id=r.get("session_id", ""),
            category=r.get("category", ""),
            symptom=r.get("symptom", ""),
            cause=r.get("cause", ""),
            diagnostic_path=r.get("diagnostic_path", ""),
            confidence=r.get("confidence", "medium"),
            frequency=int(r.get("frequency", 1)),
            created_at=r.get("created_at", ""),
            updated_at=r.get("updated_at", ""),
            metadata=json.loads(r.get("metadata", "{}")),
        )


# ---------------------------------------------------------------------------
# TrainingManager
# ---------------------------------------------------------------------------

class TrainingManager:
    """SQLite-backed training and learning manager for Doppler Cycles.

    Captures diagnostic sessions step-by-step, extracts reusable patterns
    via the model router, stores them in ChromaDB for semantic search, and
    can suggest diagnostic tree updates based on accumulated experience.

    Args:
        db_path:            Path to the SQLite database file.
        router:             Reference to ModelRouter for pattern extraction.
        episodic_memory:    Reference to EpisodicMemory for audit trail.
        long_term_memory:   Reference to LongTermMemory (ChromaDB) for
                            semantic storage and retrieval.
        diagnostic_engine:  Reference to DiagnosticEngine for tree queries.
        on_change:          Async callback fired after any state mutation
                            (for broadcasting updates to the dashboard).
    """

    def __init__(
        self,
        db_path: str = "data/training.db",
        router: Any = None,
        episodic_memory: Any = None,
        long_term_memory: Any = None,
        diagnostic_engine: Any = None,
        on_change: Optional[Callable[[], Coroutine]] = None,
    ):
        self._db_path = db_path
        self._router = router
        self._episodic = episodic_memory
        self._ltm = long_term_memory
        self._diag_engine = diagnostic_engine
        self._on_change = on_change

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("TrainingManager initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        """Create the training tables if they don't exist."""
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                session_id       TEXT PRIMARY KEY,
                title            TEXT NOT NULL,
                vehicle_info     TEXT DEFAULT '',
                category         TEXT DEFAULT 'general',
                status           TEXT DEFAULT 'active',
                steps            TEXT DEFAULT '[]',
                findings_summary TEXT DEFAULT '',
                extracted_patterns TEXT DEFAULT '[]',
                linked_tree_id   TEXT DEFAULT '',
                notes            TEXT DEFAULT '',
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                completed_at     TEXT DEFAULT ''
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_patterns (
                pattern_id       TEXT PRIMARY KEY,
                session_id       TEXT NOT NULL,
                category         TEXT DEFAULT '',
                symptom          TEXT DEFAULT '',
                cause            TEXT DEFAULT '',
                diagnostic_path  TEXT DEFAULT '',
                confidence       TEXT DEFAULT 'medium',
                frequency        INTEGER DEFAULT 1,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                metadata         TEXT DEFAULT '{}'
            )
        """)
        # Indexes for common queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ts_status
            ON training_sessions(status)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ts_category
            ON training_sessions(category)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_tp_session
            ON training_patterns(session_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_tp_category
            ON training_patterns(category)
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now(self) -> str:
        """ISO-8601 timestamp."""
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _fire_change(self):
        """Schedule the on_change callback if one was provided."""
        if self._on_change:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._on_change())
            except RuntimeError:
                pass  # no running loop — skip broadcast

    def _record_episode(self, content: str, tags: list[str] | None = None):
        """Write an entry to episodic memory if available."""
        if self._episodic:
            try:
                self._episodic.record(
                    participant="cam",
                    content=content,
                    context_tags=tags or ["training"],
                )
            except Exception as exc:
                logger.warning("Episodic record failed: %s", exc)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_training_session(
        self,
        title: str,
        vehicle_info: str = "",
        category: str = "general",
        linked_tree_id: str = "",
    ) -> TrainingSession:
        """Create a new training session.

        Args:
            title:          Short description of what's being diagnosed.
            vehicle_info:   Year/make/model or free text.
            category:       One of CATEGORIES.
            linked_tree_id: Optional diagnostic tree to link to.

        Returns:
            The newly created TrainingSession.
        """
        if category not in CATEGORIES:
            category = "general"

        now = self._now()
        session = TrainingSession(
            session_id=str(uuid.uuid4()),
            title=title,
            vehicle_info=vehicle_info,
            category=category,
            status="active",
            steps=[],
            linked_tree_id=linked_tree_id,
            created_at=now,
            updated_at=now,
        )

        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO training_sessions
               (session_id, title, vehicle_info, category, status, steps,
                findings_summary, extracted_patterns, linked_tree_id, notes,
                created_at, updated_at, completed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session.session_id,
                session.title,
                session.vehicle_info,
                session.category,
                session.status,
                json.dumps(session.steps),
                session.findings_summary,
                json.dumps(session.extracted_patterns),
                session.linked_tree_id,
                session.notes,
                session.created_at,
                session.updated_at,
                session.completed_at,
            ),
        )
        self._conn.commit()

        self._record_episode(
            f"Training session started: {title} ({category}) — {vehicle_info}",
            tags=["training", "session_start"],
        )
        logger.info("Training session started: %s [%s]", session.short_id, title)
        self._fire_change()
        return session

    def record_step(
        self,
        session_id: str,
        step_type: str,
        content: str,
        metadata: dict | None = None,
    ) -> TrainingSession | None:
        """Record a diagnostic step in an active session.

        Args:
            session_id: The session to add to.
            step_type:  One of STEP_TYPES (symptom, test, reading, conclusion, note).
            content:    Free-text description of what happened / was observed.
            metadata:   Optional extra data (readings, units, etc.).

        Returns:
            Updated TrainingSession, or None if session not found.
        """
        if step_type not in STEP_TYPES:
            logger.warning("Invalid step type: %s", step_type)
            return None

        session = self.get_session(session_id)
        if not session or session.status != "active":
            logger.warning("Cannot record step — session %s not active", session_id[:8])
            return None

        step = {
            "step_num": len(session.steps) + 1,
            "type": step_type,
            "content": content,
            "timestamp": self._now(),
            "metadata": metadata or {},
        }
        session.steps.append(step)
        now = self._now()

        cur = self._conn.cursor()
        cur.execute(
            """UPDATE training_sessions
               SET steps = ?, updated_at = ?
               WHERE session_id = ?""",
            (json.dumps(session.steps), now, session_id),
        )
        self._conn.commit()
        session.updated_at = now

        self._record_episode(
            f"Training step [{step_type}]: {content[:120]}",
            tags=["training", "step", step_type],
        )
        logger.info(
            "Step #%d (%s) recorded on session %s",
            step["step_num"], step_type, session.short_id,
        )
        self._fire_change()
        return session

    def complete_session(self, session_id: str) -> TrainingSession | None:
        """Mark a training session as completed.

        Returns:
            Updated TrainingSession, or None if not found.
        """
        session = self.get_session(session_id)
        if not session:
            return None
        if session.status != "active":
            logger.warning("Session %s already %s", session.short_id, session.status)
            return session

        now = self._now()
        cur = self._conn.cursor()
        cur.execute(
            """UPDATE training_sessions
               SET status = 'completed', completed_at = ?, updated_at = ?
               WHERE session_id = ?""",
            (now, now, session_id),
        )
        self._conn.commit()

        session.status = "completed"
        session.completed_at = now
        session.updated_at = now

        self._record_episode(
            f"Training session completed: {session.title} ({len(session.steps)} steps)",
            tags=["training", "session_complete"],
        )
        logger.info("Training session completed: %s", session.short_id)
        self._fire_change()
        return session

    # ------------------------------------------------------------------
    # Pattern extraction (async — uses model router)
    # ------------------------------------------------------------------

    async def extract_patterns(self, session_id: str) -> list[dict]:
        """Use the model router to extract diagnostic patterns from a session.

        Routes at 'routine' complexity — uses local Ollama (free).

        Returns:
            List of extracted pattern dicts.
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning("extract_patterns: session %s not found", session_id[:8])
            return []

        if not self._router:
            logger.warning("extract_patterns: no model router available")
            return []

        # Build the steps text block
        steps_text = ""
        for s in session.steps:
            steps_text += f"  {s['step_num']}. [{s['type'].upper()}] {s['content']}\n"

        prompt = PATTERN_EXTRACTION_PROMPT.format(
            title=session.title,
            vehicle_info=session.vehicle_info,
            category=session.category,
            steps_text=steps_text or "  (no steps recorded)",
        )

        try:
            response = await self._router.route(
                prompt,
                task_complexity="routine",
                system_prompt="You are a motorcycle diagnostic pattern extractor.",
            )
            raw = response.text.strip()

            # Try to parse JSON from the response
            patterns_data = self._parse_json_array(raw)
        except Exception as exc:
            logger.error("Pattern extraction failed: %s", exc)
            return []

        # Store each pattern
        now = self._now()
        stored_patterns = []
        for p in patterns_data:
            pattern = TrainingPattern(
                pattern_id=str(uuid.uuid4()),
                session_id=session_id,
                category=session.category,
                symptom=str(p.get("symptom", "")),
                cause=str(p.get("cause", "")),
                diagnostic_path=str(p.get("diagnostic_path", "")),
                confidence=str(p.get("confidence", "medium")),
                frequency=1,
                created_at=now,
                updated_at=now,
                metadata={},
            )

            # Check for existing similar pattern — bump frequency
            existing = self._find_similar_pattern(pattern.symptom, pattern.cause)
            if existing:
                self._bump_pattern_frequency(existing.pattern_id)
                stored_patterns.append(existing.to_dict())
                continue

            # Insert new pattern
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO training_patterns
                   (pattern_id, session_id, category, symptom, cause,
                    diagnostic_path, confidence, frequency,
                    created_at, updated_at, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pattern.pattern_id,
                    pattern.session_id,
                    pattern.category,
                    pattern.symptom,
                    pattern.cause,
                    pattern.diagnostic_path,
                    pattern.confidence,
                    pattern.frequency,
                    pattern.created_at,
                    pattern.updated_at,
                    json.dumps(pattern.metadata),
                ),
            )
            self._conn.commit()

            # Store in LTM (ChromaDB) for semantic search
            if self._ltm:
                try:
                    ltm_content = (
                        f"Diagnostic pattern: {pattern.symptom} → {pattern.cause}. "
                        f"Path: {pattern.diagnostic_path}"
                    )
                    self._ltm.store(
                        content=ltm_content,
                        category="system_learning",
                        metadata={
                            "pattern_id": pattern.pattern_id,
                            "session_id": session_id,
                            "vehicle_category": pattern.category,
                            "confidence": pattern.confidence,
                        },
                    )
                except Exception as exc:
                    logger.warning("LTM store failed for pattern %s: %s", pattern.short_id, exc)

            stored_patterns.append(pattern.to_dict())

        # Build a findings summary from the patterns
        summary_parts = []
        for p in stored_patterns:
            summary_parts.append(f"• {p.get('symptom', '?')} → {p.get('cause', '?')}")
        findings_summary = "\n".join(summary_parts) if summary_parts else "No patterns extracted."

        # Update the session with extracted data
        cur = self._conn.cursor()
        cur.execute(
            """UPDATE training_sessions
               SET findings_summary = ?, extracted_patterns = ?, updated_at = ?
               WHERE session_id = ?""",
            (findings_summary, json.dumps(stored_patterns), self._now(), session_id),
        )
        self._conn.commit()

        self._record_episode(
            f"Extracted {len(stored_patterns)} pattern(s) from session {session.short_id}",
            tags=["training", "pattern_extraction"],
        )
        logger.info("Extracted %d patterns from session %s", len(stored_patterns), session.short_id)
        self._fire_change()
        return stored_patterns

    def _parse_json_array(self, text: str) -> list[dict]:
        """Try to extract a JSON array from model output.

        Handles common issues: markdown fences, trailing text, etc.
        """
        # Strip markdown code fences if present
        cleaned = text
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith("["):
                    cleaned = stripped
                    break

        # Try direct parse
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try to find array in text
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(cleaned[start:end + 1])
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON array from model output")
        return []

    def _find_similar_pattern(
        self, symptom: str, cause: str
    ) -> TrainingPattern | None:
        """Check if a pattern with similar symptom+cause already exists."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT * FROM training_patterns
               WHERE symptom = ? AND cause = ?
               LIMIT 1""",
            (symptom, cause),
        )
        row = cur.fetchone()
        return TrainingPattern.from_row(row) if row else None

    def _bump_pattern_frequency(self, pattern_id: str):
        """Increment the frequency counter for an existing pattern."""
        cur = self._conn.cursor()
        cur.execute(
            """UPDATE training_patterns
               SET frequency = frequency + 1, updated_at = ?
               WHERE pattern_id = ?""",
            (self._now(), pattern_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Experience-based suggestions (async)
    # ------------------------------------------------------------------

    async def suggest_from_experience(
        self, symptoms: str, category: str = ""
    ) -> dict:
        """Query accumulated experience for diagnostic suggestions.

        Combines ChromaDB semantic search with SQL pattern lookup, then
        routes through the model router for synthesis.

        Args:
            symptoms: Description of current symptoms.
            category: Optional category filter.

        Returns:
            Dict with ok, suggestions, related_sessions, related_patterns.
        """
        if not self._router:
            return {"ok": False, "error": "No model router available"}

        context_parts = []

        # 1. Query ChromaDB LTM for semantically similar patterns
        related_sessions = []
        if self._ltm:
            try:
                results = self._ltm.query(
                    text=symptoms,
                    top_k=5,
                    category="system_learning",
                )
                for entry in results:
                    context_parts.append(
                        f"[LTM match] {entry.content} "
                        f"(similarity: {getattr(entry, 'distance', 'n/a')})"
                    )
                    # Track related session IDs from metadata
                    sid = getattr(entry, "metadata", {}).get("session_id", "")
                    if sid and sid not in related_sessions:
                        related_sessions.append(sid)
            except Exception as exc:
                logger.warning("LTM query failed: %s", exc)

        # 2. Query training_patterns table with LIKE search
        related_patterns = []
        cur = self._conn.cursor()
        if category:
            cur.execute(
                """SELECT * FROM training_patterns
                   WHERE category = ? AND (symptom LIKE ? OR cause LIKE ?)
                   ORDER BY frequency DESC LIMIT 10""",
                (category, f"%{symptoms[:60]}%", f"%{symptoms[:60]}%"),
            )
        else:
            cur.execute(
                """SELECT * FROM training_patterns
                   WHERE symptom LIKE ? OR cause LIKE ?
                   ORDER BY frequency DESC LIMIT 10""",
                (f"%{symptoms[:60]}%", f"%{symptoms[:60]}%"),
            )
        for row in cur.fetchall():
            p = TrainingPattern.from_row(row)
            related_patterns.append(p.to_dict())
            context_parts.append(
                f"[Pattern] {p.symptom} → {p.cause} "
                f"(confidence: {p.confidence}, seen {p.frequency}x)"
            )

        context = "\n".join(context_parts) if context_parts else "(No prior experience found.)"

        prompt = EXPERIENCE_QUERY_PROMPT.format(
            symptoms=symptoms,
            category=category or "any",
            context=context,
        )

        try:
            response = await self._router.route(
                prompt,
                task_complexity="routine",
                system_prompt="You are a motorcycle diagnostic advisor drawing on past experience.",
            )
            suggestions = response.text.strip()
        except Exception as exc:
            logger.error("Experience suggestion failed: %s", exc)
            return {"ok": False, "error": str(exc)}

        return {
            "ok": True,
            "suggestions": suggestions,
            "related_sessions": related_sessions,
            "related_patterns": related_patterns,
        }

    # ------------------------------------------------------------------
    # Tree growth suggestions (async)
    # ------------------------------------------------------------------

    async def suggest_tree_updates(self, category: str = "") -> dict:
        """Suggest diagnostic tree updates based on accumulated patterns.

        Loads patterns for the category plus current tree nodes, then
        routes through the model router to suggest new branches.

        Args:
            category: Filter patterns/trees by this category.

        Returns:
            Dict with ok, suggestions, patterns_analyzed.
        """
        if not self._router:
            return {"ok": False, "error": "No model router available"}

        # Load patterns for this category
        patterns = self.list_patterns(category=category, limit=50)
        if not patterns:
            return {
                "ok": True,
                "suggestions": [],
                "patterns_analyzed": 0,
            }

        patterns_text = ""
        for p in patterns:
            patterns_text += (
                f"  • {p.symptom} → {p.cause} "
                f"(path: {p.diagnostic_path}, confidence: {p.confidence}, "
                f"seen {p.frequency}x)\n"
            )

        # Load current tree nodes if diagnostic engine is available
        tree_nodes = ""
        if self._diag_engine:
            try:
                trees = self._diag_engine.list_trees()
                relevant = [t for t in trees if not category or t.get("category") == category]
                for t in relevant:
                    tree_nodes += f"\n  Tree: {t['name']} ({t['tree_id']})\n"
                    full_tree = self._diag_engine.get_tree(t["tree_id"])
                    if full_tree and "nodes" in full_tree:
                        for nid, node in full_tree["nodes"].items():
                            ntype = node.get("type", "?")
                            ntext = node.get("text", node.get("title", ""))
                            tree_nodes += f"    [{nid}] ({ntype}) {ntext}\n"
            except Exception as exc:
                logger.warning("Could not load trees: %s", exc)
                tree_nodes = "(Could not load current trees)"

        if not tree_nodes.strip():
            tree_nodes = "(No trees loaded for this category)"

        prompt = TREE_UPDATE_PROMPT.format(
            category=category or "all",
            tree_nodes=tree_nodes,
            patterns_text=patterns_text,
        )

        try:
            response = await self._router.route(
                prompt,
                task_complexity="routine",
                system_prompt="You are a diagnostic decision tree architect.",
            )
            raw = response.text.strip()
            suggestions = self._parse_json_array(raw)
        except Exception as exc:
            logger.error("Tree update suggestion failed: %s", exc)
            return {"ok": False, "error": str(exc)}

        return {
            "ok": True,
            "suggestions": suggestions,
            "patterns_analyzed": len(patterns),
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> TrainingSession | None:
        """Fetch a single training session by ID."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM training_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        return TrainingSession.from_row(row) if row else None

    def list_sessions(
        self,
        status: str = "",
        category: str = "",
        limit: int = 50,
    ) -> list[TrainingSession]:
        """List training sessions with optional filters.

        Args:
            status:   Filter by status (active, completed, archived).
            category: Filter by category.
            limit:    Max results.

        Returns:
            List of TrainingSession objects, newest first.
        """
        query = "SELECT * FROM training_sessions WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cur = self._conn.cursor()
        cur.execute(query, params)
        return [TrainingSession.from_row(r) for r in cur.fetchall()]

    def delete_session(self, session_id: str) -> bool:
        """Delete a training session and its associated patterns.

        Returns:
            True if the session was deleted, False if not found.
        """
        session = self.get_session(session_id)
        if not session:
            return False

        cur = self._conn.cursor()
        cur.execute("DELETE FROM training_patterns WHERE session_id = ?", (session_id,))
        cur.execute("DELETE FROM training_sessions WHERE session_id = ?", (session_id,))
        self._conn.commit()

        self._record_episode(
            f"Training session deleted: {session.title} ({session.short_id})",
            tags=["training", "session_deleted"],
        )
        logger.info("Training session deleted: %s", session.short_id)
        self._fire_change()
        return True

    def list_patterns(
        self,
        category: str = "",
        limit: int = 50,
    ) -> list[TrainingPattern]:
        """List training patterns with optional category filter."""
        query = "SELECT * FROM training_patterns WHERE 1=1"
        params: list = []
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY frequency DESC, created_at DESC LIMIT ?"
        params.append(limit)

        cur = self._conn.cursor()
        cur.execute(query, params)
        return [TrainingPattern.from_row(r) for r in cur.fetchall()]

    def get_growth_metrics(self) -> dict:
        """Return metrics about knowledge growth over time.

        Returns:
            Dict with sessions_this_month, patterns_this_month,
            top_categories, highest_frequency_patterns.
        """
        cur = self._conn.cursor()

        # Sessions this month
        month_start = datetime.utcnow().strftime("%Y-%m-01")
        cur.execute(
            "SELECT COUNT(*) FROM training_sessions WHERE created_at >= ?",
            (month_start,),
        )
        sessions_this_month = cur.fetchone()[0]

        # Patterns this month
        cur.execute(
            "SELECT COUNT(*) FROM training_patterns WHERE created_at >= ?",
            (month_start,),
        )
        patterns_this_month = cur.fetchone()[0]

        # Top categories by pattern count
        cur.execute(
            """SELECT category, COUNT(*) as cnt
               FROM training_patterns
               GROUP BY category
               ORDER BY cnt DESC LIMIT 5"""
        )
        top_categories = {row["category"]: row["cnt"] for row in cur.fetchall()}

        # Highest frequency patterns
        cur.execute(
            """SELECT symptom, cause, frequency, confidence
               FROM training_patterns
               ORDER BY frequency DESC LIMIT 5"""
        )
        top_patterns = [
            {
                "symptom": row["symptom"],
                "cause": row["cause"],
                "frequency": row["frequency"],
                "confidence": row["confidence"],
            }
            for row in cur.fetchall()
        ]

        return {
            "sessions_this_month": sessions_this_month,
            "patterns_this_month": patterns_this_month,
            "top_categories": top_categories,
            "top_patterns": top_patterns,
        }

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Summary status for the dashboard header cards."""
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) FROM training_sessions")
        total_sessions = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM training_sessions WHERE status = 'completed'"
        )
        completed = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM training_sessions WHERE status = 'active'"
        )
        active = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM training_patterns")
        total_patterns = cur.fetchone()[0]

        # Patterns by category
        cur.execute(
            """SELECT category, COUNT(*) as cnt
               FROM training_patterns GROUP BY category"""
        )
        by_category = {row["category"]: row["cnt"] for row in cur.fetchall()}

        # Patterns this month
        month_start = datetime.utcnow().strftime("%Y-%m-01")
        cur.execute(
            "SELECT COUNT(*) FROM training_patterns WHERE created_at >= ?",
            (month_start,),
        )
        patterns_this_month = cur.fetchone()[0]

        # Average steps per completed session
        cur.execute(
            """SELECT steps FROM training_sessions WHERE status = 'completed'"""
        )
        step_counts = []
        for row in cur.fetchall():
            try:
                steps = json.loads(row["steps"])
                step_counts.append(len(steps))
            except Exception:
                pass
        avg_steps = round(sum(step_counts) / len(step_counts), 1) if step_counts else 0

        return {
            "total_sessions": total_sessions,
            "active": active,
            "completed": completed,
            "total_patterns": total_patterns,
            "by_category": by_category,
            "patterns_this_month": patterns_this_month,
            "avg_steps": avg_steps,
        }

    def to_broadcast_dict(self) -> dict:
        """Full state dict for broadcasting to the dashboard."""
        sessions = self.list_sessions(limit=50)
        patterns = self.list_patterns(limit=50)
        return {
            "sessions": [s.to_dict() for s in sessions],
            "patterns": [p.to_dict() for p in patterns],
            "status": self.get_status(),
        }

    def close(self):
        """Close the SQLite connection. Call on shutdown."""
        try:
            self._conn.close()
            logger.info("TrainingManager database closed")
        except Exception:
            pass
