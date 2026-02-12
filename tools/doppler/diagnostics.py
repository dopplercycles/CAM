"""
CAM Diagnostic Decision Tree Engine

Codifies George's 20+ years of motorcycle diagnostic expertise into
reusable YAML decision trees. Trees walk through symptom → cause → test →
result branching logic. The engine manages interactive sessions, uses the
model router for AI-suggested additional checks, and stores completed
diagnostics as service records.

Trees live in config/diagnostic_trees/ as individual YAML files that George
can edit directly and version-control with git.

Usage:
    from tools.doppler.diagnostics import DiagnosticEngine

    engine = DiagnosticEngine(
        db_path="data/diagnostics.db",
        trees_dir="config/diagnostic_trees",
        service_store=service_store,
        router=router,
        on_change=broadcast_diagnostics_status,
    )
    session = engine.start_session("electrical_no_start", vehicle_id="v-123")
    node = engine.get_current_node(session.session_id)
    result = engine.answer_question(session.session_id, answer_index=0)
"""

import glob
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("cam.diagnostics")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIAG_SYSTEM_PROMPT = """You are an expert motorcycle diagnostic assistant working alongside a certified technician.
Given the diagnostic session history below, suggest additional checks or tests
the technician might have missed. Be specific, practical, and safety-conscious.
Keep suggestions concise — 3-5 bullet points max. Focus on common failure modes
and things that are easy to overlook."""


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticSession:
    """An interactive diagnostic session walking through a decision tree.

    Attributes:
        session_id:      Unique identifier (UUID string).
        tree_id:         Which decision tree is being walked.
        tree_name:       Human-readable tree name.
        category:        Tree category (electrical, engine, suspension).
        vehicle_id:      Optional link to a vehicle in ServiceRecordStore.
        vehicle_summary: Denormalized vehicle display name.
        customer_name:   Customer name for the session.
        status:          active, completed, or abandoned.
        current_node:    ID of the current node in the tree.
        answers:         JSON list of {node_id, question, answer, timestamp}.
        findings:        JSON list of {title, severity, description, actions}.
        ai_suggestions:  AI-generated additional checks (text).
        notes:           Technician's free-text notes.
        started_at:      When the session began (ISO string).
        completed_at:    When the session ended (ISO string or "").
        created_at:      Row creation timestamp.
        updated_at:      Last modification timestamp.
    """
    session_id: str
    tree_id: str = ""
    tree_name: str = ""
    category: str = ""
    vehicle_id: str = ""
    vehicle_summary: str = ""
    customer_name: str = ""
    status: str = "active"
    current_node: str = "start"
    answers: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    ai_suggestions: str = ""
    notes: str = ""
    started_at: str = ""
    completed_at: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def short_id(self) -> str:
        """First 8 characters of the session ID for display."""
        return self.session_id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/dashboard use."""
        return {
            "session_id": self.session_id,
            "short_id": self.short_id,
            "tree_id": self.tree_id,
            "tree_name": self.tree_name,
            "category": self.category,
            "vehicle_id": self.vehicle_id,
            "vehicle_summary": self.vehicle_summary,
            "customer_name": self.customer_name,
            "status": self.status,
            "current_node": self.current_node,
            "answers": self.answers,
            "findings": self.findings,
            "ai_suggestions": self.ai_suggestions,
            "notes": self.notes,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DiagnosticSession":
        """Convert a SQLite row to a DiagnosticSession."""
        def _parse_json(val, default):
            if not val:
                return default
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default

        return cls(
            session_id=row["session_id"],
            tree_id=row["tree_id"],
            tree_name=row["tree_name"],
            category=row["category"],
            vehicle_id=row["vehicle_id"],
            vehicle_summary=row["vehicle_summary"],
            customer_name=row["customer_name"],
            status=row["status"],
            current_node=row["current_node"],
            answers=_parse_json(row["answers"], []),
            findings=_parse_json(row["findings"], []),
            ai_suggestions=row["ai_suggestions"],
            notes=row["notes"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# ---------------------------------------------------------------------------
# Diagnostic Engine
# ---------------------------------------------------------------------------

class DiagnosticEngine:
    """Manages YAML decision trees and interactive diagnostic sessions.

    Loads trees from YAML files, validates node references, and provides
    a session-based interface for walking through diagnostic logic.

    Args:
        db_path:       Path to the SQLite database for session storage.
        trees_dir:     Path to the directory containing YAML tree files.
        service_store: Reference to ServiceRecordStore for saving completed
                       diagnostics as service records.
        router:        Reference to ModelRouter for AI suggestions.
        on_change:     Async callback fired after any state mutation (for
                       broadcasting updates to the dashboard).
    """

    def __init__(
        self,
        db_path: str = "data/diagnostics.db",
        trees_dir: str = "config/diagnostic_trees",
        service_store: Any = None,
        router: Any = None,
        on_change: Callable[[], Coroutine] | None = None,
    ):
        self._db_path = db_path
        self._trees_dir = trees_dir
        self._service_store = service_store
        self._router = router
        self._on_change = on_change

        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        # SQLite setup
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

        # Load trees from YAML
        self._trees: dict[str, dict] = {}
        self._load_trees()

        logger.info(
            "DiagnosticEngine initialized: %d trees loaded from %s",
            len(self._trees), trees_dir,
        )

    # -------------------------------------------------------------------
    # Database setup
    # -------------------------------------------------------------------

    def _create_tables(self):
        """Create the diagnostic_sessions table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnostic_sessions (
                session_id      TEXT PRIMARY KEY,
                tree_id         TEXT NOT NULL,
                tree_name       TEXT NOT NULL DEFAULT '',
                category        TEXT NOT NULL DEFAULT '',
                vehicle_id      TEXT NOT NULL DEFAULT '',
                vehicle_summary TEXT NOT NULL DEFAULT '',
                customer_name   TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'active',
                current_node    TEXT NOT NULL DEFAULT 'start',
                answers         TEXT NOT NULL DEFAULT '[]',
                findings        TEXT NOT NULL DEFAULT '[]',
                ai_suggestions  TEXT NOT NULL DEFAULT '',
                notes           TEXT NOT NULL DEFAULT '',
                started_at      TEXT NOT NULL DEFAULT '',
                completed_at    TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL DEFAULT '',
                updated_at      TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_diag_status
            ON diagnostic_sessions(status)
        """)
        self._conn.commit()

    # -------------------------------------------------------------------
    # Tree loading & validation
    # -------------------------------------------------------------------

    def _load_trees(self):
        """Load all YAML trees from the trees directory.

        Skips files that fail to parse or don't have the required fields.
        Validates that every 'next' reference points to an existing node.
        """
        self._trees = {}
        pattern = os.path.join(self._trees_dir, "*.yaml")
        for filepath in sorted(glob.glob(pattern)):
            try:
                with open(filepath, "r") as f:
                    tree = yaml.safe_load(f)
                if not tree or not isinstance(tree, dict):
                    logger.warning("Skipping empty/invalid YAML: %s", filepath)
                    continue
                tree_id = tree.get("tree_id")
                if not tree_id or "nodes" not in tree:
                    logger.warning(
                        "Skipping YAML without tree_id or nodes: %s", filepath
                    )
                    continue
                # Validate node references
                nodes = tree["nodes"]
                for node_id, node in nodes.items():
                    if node.get("type") == "question":
                        for ans in node.get("answers", []):
                            next_id = ans.get("next", "")
                            if next_id and next_id not in nodes:
                                logger.warning(
                                    "Tree %s: node '%s' answer points to "
                                    "missing node '%s'",
                                    tree_id, node_id, next_id,
                                )
                self._trees[tree_id] = tree
                logger.info(
                    "Loaded tree: %s (%s) — %d nodes",
                    tree_id, tree.get("name", "?"), len(nodes),
                )
            except Exception as e:
                logger.error("Failed to load tree from %s: %s", filepath, e)

    def reload_trees(self):
        """Hot-reload all trees from disk."""
        self._load_trees()
        logger.info("Trees reloaded: %d trees", len(self._trees))

    def list_trees(self) -> list[dict[str, Any]]:
        """Return summary of all loaded trees.

        Returns:
            List of dicts with tree_id, name, category, description, node_count.
        """
        result = []
        for tree_id, tree in sorted(self._trees.items()):
            result.append({
                "tree_id": tree_id,
                "name": tree.get("name", tree_id),
                "category": tree.get("category", ""),
                "description": tree.get("description", ""),
                "node_count": len(tree.get("nodes", {})),
                "safety_warnings": tree.get("safety_warnings", []),
            })
        return result

    def get_tree(self, tree_id: str) -> dict | None:
        """Return the full parsed tree dict, or None if not found."""
        return self._trees.get(tree_id)

    # -------------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------------

    def start_session(
        self,
        tree_id: str,
        vehicle_id: str = "",
        vehicle_summary: str = "",
        customer_name: str = "",
    ) -> DiagnosticSession | None:
        """Start a new diagnostic session with the given tree.

        Args:
            tree_id:         Which decision tree to walk.
            vehicle_id:      Optional vehicle link.
            vehicle_summary: Optional vehicle display name.
            customer_name:   Optional customer name.

        Returns:
            The new DiagnosticSession, or None if tree_id is invalid.
        """
        tree = self._trees.get(tree_id)
        if not tree:
            logger.warning("Cannot start session: unknown tree '%s'", tree_id)
            return None

        now = datetime.now(timezone.utc).isoformat()
        session_id = str(uuid.uuid4())

        # Look up vehicle display name from service store if we have an ID
        if vehicle_id and not vehicle_summary and self._service_store:
            vehicle = self._service_store.get_vehicle(vehicle_id)
            if vehicle:
                vehicle_summary = vehicle.display_name

        session = DiagnosticSession(
            session_id=session_id,
            tree_id=tree_id,
            tree_name=tree.get("name", tree_id),
            category=tree.get("category", ""),
            vehicle_id=vehicle_id,
            vehicle_summary=vehicle_summary,
            customer_name=customer_name,
            status="active",
            current_node="start",
            answers=[],
            findings=[],
            ai_suggestions="",
            notes="",
            started_at=now,
            completed_at="",
            created_at=now,
            updated_at=now,
        )

        self._conn.execute(
            """
            INSERT INTO diagnostic_sessions
                (session_id, tree_id, tree_name, category,
                 vehicle_id, vehicle_summary, customer_name,
                 status, current_node, answers, findings,
                 ai_suggestions, notes, started_at, completed_at,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, session.tree_id, session.tree_name, session.category,
             session.vehicle_id, session.vehicle_summary, session.customer_name,
             session.status, session.current_node,
             json.dumps(session.answers), json.dumps(session.findings),
             session.ai_suggestions, session.notes,
             session.started_at, session.completed_at,
             session.created_at, session.updated_at),
        )
        self._conn.commit()

        logger.info(
            "Diagnostic session started: %s (tree=%s, vehicle=%s)",
            session.short_id, tree_id, vehicle_summary or "none",
        )
        return session

    def get_session(self, session_id: str) -> DiagnosticSession | None:
        """Retrieve a session by ID.

        Returns:
            The DiagnosticSession, or None if not found.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM diagnostic_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        return DiagnosticSession.from_row(row) if row else None

    def list_sessions(
        self, status: str | None = None, limit: int = 50,
    ) -> list[DiagnosticSession]:
        """List sessions, newest first.

        Args:
            status: Optional filter by status (active/completed/abandoned).
            limit:  Max number to return.

        Returns:
            List of DiagnosticSession objects.
        """
        cur = self._conn.cursor()
        if status:
            cur.execute(
                "SELECT * FROM diagnostic_sessions WHERE status = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM diagnostic_sessions "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [DiagnosticSession.from_row(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------
    # Tree walking
    # -------------------------------------------------------------------

    def get_current_node(self, session_id: str) -> dict[str, Any] | None:
        """Get the current node data for a session.

        Returns a dict with the node contents plus the tree's safety warnings
        and the session's breadcrumb trail. Returns None if session or node
        not found.
        """
        session = self.get_session(session_id)
        if not session:
            return None

        tree = self._trees.get(session.tree_id)
        if not tree:
            return None

        node = tree["nodes"].get(session.current_node)
        if not node:
            return None

        # Build breadcrumb trail from answer history
        breadcrumbs = [a["node_id"] for a in session.answers]
        breadcrumbs.append(session.current_node)

        return {
            "node_id": session.current_node,
            "type": node.get("type", "question"),
            "text": node.get("text", ""),
            "title": node.get("title", ""),
            "context": node.get("context", ""),
            "safety": node.get("safety", ""),
            "answers": node.get("answers", []),
            "severity": node.get("severity", ""),
            "description": node.get("description", ""),
            "recommended_actions": node.get("recommended_actions", []),
            "possible_parts": node.get("possible_parts", []),
            "follow_up_trees": node.get("follow_up_trees", []),
            "safety_warnings": tree.get("safety_warnings", []),
            "breadcrumbs": breadcrumbs,
            "session": session.to_dict(),
        }

    def answer_question(
        self, session_id: str, answer_index: int,
    ) -> dict[str, Any]:
        """Record an answer and advance to the next node.

        Args:
            session_id:   The session to advance.
            answer_index: Index into the current node's answers list.

        Returns:
            Dict with ok, node (the new current node data), is_finding,
            finding (if it's a finding node), and answers trail.
        """
        session = self.get_session(session_id)
        if not session or session.status != "active":
            return {"ok": False, "error": "Session not found or not active"}

        tree = self._trees.get(session.tree_id)
        if not tree:
            return {"ok": False, "error": "Tree not found"}

        node = tree["nodes"].get(session.current_node)
        if not node or node.get("type") != "question":
            return {"ok": False, "error": "Current node is not a question"}

        answers_list = node.get("answers", [])
        if answer_index < 0 or answer_index >= len(answers_list):
            return {"ok": False, "error": "Invalid answer index"}

        answer = answers_list[answer_index]
        next_node_id = answer.get("next", "")
        if not next_node_id or next_node_id not in tree["nodes"]:
            return {"ok": False, "error": f"Next node '{next_node_id}' not found"}

        # Record the answer
        now = datetime.now(timezone.utc).isoformat()
        session.answers.append({
            "node_id": session.current_node,
            "question": node.get("text", ""),
            "answer": answer.get("label", ""),
            "timestamp": now,
        })

        # Advance to next node
        session.current_node = next_node_id
        next_node = tree["nodes"][next_node_id]

        # If the next node is a finding, record it
        is_finding = next_node.get("type") == "finding"
        finding = None
        if is_finding:
            finding = {
                "title": next_node.get("title", ""),
                "severity": next_node.get("severity", "medium"),
                "description": next_node.get("description", ""),
                "actions": next_node.get("recommended_actions", []),
                "parts": next_node.get("possible_parts", []),
                "follow_up_trees": next_node.get("follow_up_trees", []),
            }
            session.findings.append(finding)

        # Persist
        self._conn.execute(
            """
            UPDATE diagnostic_sessions
            SET current_node = ?, answers = ?, findings = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (session.current_node, json.dumps(session.answers),
             json.dumps(session.findings), now, session.session_id),
        )
        self._conn.commit()

        logger.info(
            "Session %s: answered '%s' → node '%s'%s",
            session.short_id, answer.get("label", ""),
            next_node_id, " [FINDING]" if is_finding else "",
        )

        # Return the new node data
        new_node_data = self.get_current_node(session_id)
        return {
            "ok": True,
            "node": new_node_data,
            "is_finding": is_finding,
            "finding": finding,
            "answers": session.answers,
        }

    def go_back(self, session_id: str) -> dict[str, Any]:
        """Pop the last answer and return to the previous node.

        Returns:
            Dict with ok and node (the restored node data).
        """
        session = self.get_session(session_id)
        if not session or session.status != "active":
            return {"ok": False, "error": "Session not found or not active"}

        if not session.answers:
            return {"ok": False, "error": "Already at the start"}

        # Pop the last answer
        last = session.answers.pop()
        previous_node = last["node_id"]

        # If current node was a finding, remove the last finding too
        tree = self._trees.get(session.tree_id)
        if tree:
            current = tree["nodes"].get(session.current_node, {})
            if current.get("type") == "finding" and session.findings:
                session.findings.pop()

        session.current_node = previous_node
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            UPDATE diagnostic_sessions
            SET current_node = ?, answers = ?, findings = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (session.current_node, json.dumps(session.answers),
             json.dumps(session.findings), now, session.session_id),
        )
        self._conn.commit()

        logger.info(
            "Session %s: went back to node '%s'",
            session.short_id, previous_node,
        )
        return {
            "ok": True,
            "node": self.get_current_node(session_id),
        }

    # -------------------------------------------------------------------
    # AI suggestions
    # -------------------------------------------------------------------

    async def get_ai_suggestions(self, session_id: str) -> dict[str, Any]:
        """Ask the local model for additional diagnostic suggestions.

        Builds context from the session's answer trail and findings, then
        routes through the model router at 'simple' complexity (free local
        Ollama model).

        Returns:
            Dict with ok and suggestions (text).
        """
        session = self.get_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}

        if not self._router:
            return {"ok": False, "error": "Model router not available"}

        # Build context prompt
        lines = [
            f"Diagnostic: {session.tree_name}",
            f"Vehicle: {session.vehicle_summary or 'Unknown'}",
            "",
            "Steps taken:",
        ]
        for a in session.answers:
            lines.append(f"  Q: {a['question']}")
            lines.append(f"  A: {a['answer']}")
            lines.append("")

        if session.findings:
            lines.append("Findings so far:")
            for f in session.findings:
                lines.append(f"  - [{f.get('severity', '?').upper()}] {f['title']}: {f.get('description', '')}")
            lines.append("")

        if session.notes:
            lines.append(f"Tech notes: {session.notes}")
            lines.append("")

        lines.append(
            "Based on this diagnostic path, what additional checks or tests "
            "should the technician consider? Are there common failure modes "
            "that haven't been checked?"
        )

        prompt = "\n".join(lines)

        try:
            response = await self._router.route(
                prompt=prompt,
                task_complexity="simple",
                system_prompt=DIAG_SYSTEM_PROMPT,
            )
            suggestions = response.text

            # Store on the session
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                """
                UPDATE diagnostic_sessions
                SET ai_suggestions = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (suggestions, now, session.session_id),
            )
            self._conn.commit()

            logger.info(
                "AI suggestions generated for session %s (%d chars)",
                session.short_id, len(suggestions),
            )
            return {"ok": True, "suggestions": suggestions}

        except Exception as e:
            logger.error("AI suggestions failed for session %s: %s",
                         session.short_id, e)
            return {
                "ok": False,
                "error": f"AI suggestion failed: {e}",
                "suggestions": "",
            }

    # -------------------------------------------------------------------
    # Session completion
    # -------------------------------------------------------------------

    def update_notes(self, session_id: str, notes: str) -> bool:
        """Update the technician's notes on a session.

        Args:
            session_id: The session to update.
            notes:      New notes text.

        Returns:
            True if updated, False if session not found.
        """
        session = self.get_session(session_id)
        if not session:
            return False

        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE diagnostic_sessions SET notes = ?, updated_at = ? "
            "WHERE session_id = ?",
            (notes, now, session_id),
        )
        self._conn.commit()
        return True

    def complete_session(self, session_id: str) -> DiagnosticSession | None:
        """Mark a session as completed.

        Returns:
            The updated session, or None if not found.
        """
        return self._finish_session(session_id, "completed")

    def abandon_session(self, session_id: str) -> DiagnosticSession | None:
        """Mark a session as abandoned.

        Returns:
            The updated session, or None if not found.
        """
        return self._finish_session(session_id, "abandoned")

    def _finish_session(
        self, session_id: str, status: str,
    ) -> DiagnosticSession | None:
        """Internal helper to mark a session finished."""
        session = self.get_session(session_id)
        if not session or session.status != "active":
            return None

        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            UPDATE diagnostic_sessions
            SET status = ?, completed_at = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (status, now, now, session_id),
        )
        self._conn.commit()

        logger.info("Session %s marked as %s", session.short_id, status)
        return self.get_session(session_id)

    def complete_and_save_record(
        self, session_id: str,
    ) -> dict[str, Any]:
        """Complete a session AND create a service record from it.

        Calls service_store.add_record() with the diagnostic data formatted
        as a service record for permanent history.

        Returns:
            Dict with ok, session, and record_id (if service record created).
        """
        session = self.complete_session(session_id)
        if not session:
            return {"ok": False, "error": "Session not found or already completed"}

        if not self._service_store:
            return {
                "ok": True,
                "session": session.to_dict(),
                "record_id": None,
                "message": "Session completed but no service store available",
            }

        # Build service record from diagnostic data
        services = [session.tree_name]
        for f in session.findings:
            services.append(f"Finding: {f.get('title', 'Unknown')}")

        # Format the Q&A trail as notes
        notes_parts = [f"=== Diagnostic: {session.tree_name} ===", ""]
        for a in session.answers:
            notes_parts.append(f"Q: {a['question']}")
            notes_parts.append(f"A: {a['answer']}")
            notes_parts.append("")

        if session.findings:
            notes_parts.append("=== Findings ===")
            for f in session.findings:
                notes_parts.append(
                    f"[{f.get('severity', '?').upper()}] {f.get('title', '')}"
                )
                notes_parts.append(f"  {f.get('description', '')}")
                notes_parts.append("")

        if session.ai_suggestions:
            notes_parts.append("=== AI Suggestions ===")
            notes_parts.append(session.ai_suggestions)
            notes_parts.append("")

        if session.notes:
            notes_parts.append("=== Technician Notes ===")
            notes_parts.append(session.notes)

        # Collect all recommended actions as recommendations
        recs = []
        for f in session.findings:
            for action in f.get("actions", []):
                recs.append(action)

        try:
            record = self._service_store.add_record(
                vehicle_id=session.vehicle_id,
                customer_name=session.customer_name,
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                service_type="diagnostic",
                services_performed=services,
                notes="\n".join(notes_parts),
                recommendations="\n".join(recs),
                metadata={"diagnostic_session_id": session.session_id},
            )

            logger.info(
                "Service record %s created from diagnostic session %s",
                record.short_id, session.short_id,
            )
            return {
                "ok": True,
                "session": session.to_dict(),
                "record_id": record.record_id,
            }

        except Exception as e:
            logger.error(
                "Failed to create service record from session %s: %s",
                session.short_id, e,
            )
            return {
                "ok": True,
                "session": session.to_dict(),
                "record_id": None,
                "message": f"Session completed but service record failed: {e}",
            }

    # -------------------------------------------------------------------
    # Dashboard helpers
    # -------------------------------------------------------------------

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Return all diagnostic data for dashboard broadcast.

        Returns:
            Dict with trees, sessions, and status.
        """
        return {
            "trees": self.list_trees(),
            "sessions": [s.to_dict() for s in self.list_sessions()],
            "status": self.get_status(),
        }

    def get_status(self) -> dict[str, Any]:
        """Return summary stats for the dashboard.

        Returns:
            Dict with total_sessions, active, completed, abandoned,
            trees_loaded.
        """
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) FROM diagnostic_sessions")
        total = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM diagnostic_sessions WHERE status = 'active'"
        )
        active = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM diagnostic_sessions WHERE status = 'completed'"
        )
        completed = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM diagnostic_sessions WHERE status = 'abandoned'"
        )
        abandoned = cur.fetchone()[0]

        return {
            "total_sessions": total,
            "active": active,
            "completed": completed,
            "abandoned": abandoned,
            "trees_loaded": len(self._trees),
        }

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------

    def close(self):
        """Close the SQLite connection."""
        self._conn.close()
        logger.info("DiagnosticEngine closed")
