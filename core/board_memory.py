"""
Board Memory — Phase 1.5: Persistent Memory Layer
─────────────────────────────────────────────────────────────────────────────
AI Board of Directors persistent memory for the CAM Dashboard system.

Provides:
  - SQLite-backed session persistence (survives restarts)
  - Per-member long-term memory (key decisions, context, history)
  - Living briefing document (auto-updated after each session)
  - Memory injection into system prompts
  - Memory summarization via Claude Haiku (condenses old memories)

Schema:
  sessions         — board session metadata
  messages         — all messages across all sessions
  board_memory     — per-member persistent memory entries
  decisions        — extracted key decisions/action items
  briefing_history — versioned briefing document snapshots

Usage:
  from core.board_memory import BoardMemory
  memory = BoardMemory(db_path="data/board.db")
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cam.board_memory")

# All 7 board members
ALL_MEMBER_IDS = ["cam", "claude", "claude_code", "deepseek", "grok", "chatgpt", "gemini"]

MEMBER_NAMES = {
    "cam": "Cam",
    "claude": "Claude",
    "claude_code": "Claude Code",
    "deepseek": "DeepSeek",
    "grok": "Grok",
    "chatgpt": "ChatGPT",
    "gemini": "Gemini",
}


class BoardMemory:
    """Persistent memory layer for the AI Board of Directors.

    Thread-safe SQLite backend with per-member memory and session history.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        id          TEXT PRIMARY KEY,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL,
        title       TEXT,
        briefing    TEXT DEFAULT '',
        active_members TEXT DEFAULT '[]',
        summary     TEXT DEFAULT '',
        archived    INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,          -- 'user' | 'assistant'
        speaker     TEXT NOT NULL,          -- 'user' | member_id
        content     TEXT NOT NULL,
        timestamp   TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE TABLE IF NOT EXISTS board_memory (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        member_id   TEXT NOT NULL,          -- 'cam', 'claude', 'deepseek', etc.
        memory_type TEXT NOT NULL,          -- 'fact', 'decision', 'preference', 'context'
        content     TEXT NOT NULL,
        source_session TEXT,               -- which session this came from
        created_at  TEXT NOT NULL,
        weight      REAL DEFAULT 1.0,      -- importance score 0-1
        active      INTEGER DEFAULT 1      -- 0 = archived/superseded
    );

    CREATE TABLE IF NOT EXISTS decisions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        content     TEXT NOT NULL,
        owner       TEXT,                  -- who's responsible
        status      TEXT DEFAULT 'open',   -- 'open' | 'done' | 'dropped'
        created_at  TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE TABLE IF NOT EXISTS briefing_history (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        content     TEXT NOT NULL,
        version     INTEGER NOT NULL,
        created_at  TEXT NOT NULL,
        source      TEXT DEFAULT 'manual'  -- 'manual' | 'auto_update'
    );

    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_board_memory_member ON board_memory(member_id, active);
    CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status);
    """

    def __init__(self, db_path: str = "data/board.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("BoardMemory initialized: %s", self.db_path)

    @contextmanager
    def _conn(self):
        """Thread-safe SQLite connection with WAL mode for better concurrency."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ─── Sessions ─────────────────────────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        title: str = "Board Meeting",
        briefing: str = "",
        active_members: Optional[list] = None,
    ) -> str:
        """Persist a new board session. Returns the session_id."""
        now = self._now()
        members = active_members or ALL_MEMBER_IDS[:]
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO sessions
                   (id, created_at, updated_at, title, briefing, active_members)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, now, now, title, briefing, json.dumps(members)),
            )
        logger.info("Session persisted: %s (%s)", session_id, title)
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """Load a session with all its messages from the DB."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not row:
                return None
            s = dict(row)
            s["active_members"] = json.loads(s["active_members"])
            s["messages"] = self.get_messages(session_id)
            return s

    def update_session(self, session_id: str, **kwargs):
        """Update session fields (title, briefing, active_members, summary, archived)."""
        allowed = {"title", "briefing", "active_members", "summary", "archived"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        if "active_members" in updates:
            updates["active_members"] = json.dumps(updates["active_members"])
        updates["updated_at"] = self._now()
        cols = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [session_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE sessions SET {cols} WHERE id = ?", vals)

    def list_sessions(self, limit: int = 20, include_archived: bool = False) -> list:
        """List past sessions, most recent first, with message counts."""
        with self._conn() as conn:
            q = "SELECT * FROM sessions"
            if not include_archived:
                q += " WHERE archived = 0"
            q += " ORDER BY updated_at DESC LIMIT ?"
            rows = conn.execute(q, (limit,)).fetchall()
            result = []
            for row in rows:
                s = dict(row)
                s["active_members"] = json.loads(s["active_members"])
                count = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                    (s["id"],)
                ).fetchone()[0]
                s["message_count"] = count
                result.append(s)
            return result

    # ─── Messages ─────────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        speaker: str,
        content: str,
    ) -> int:
        """Persist a message. Returns the row ID."""
        now = self._now()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO messages (session_id, role, speaker, content, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, role, speaker, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            return cur.lastrowid

    def get_messages(self, session_id: str) -> list:
        """Get all messages for a session, ordered by ID."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─── Board Memory ──────────────────────────────────────────────────────────

    def add_memory(
        self,
        member_id: str,
        memory_type: str,
        content: str,
        source_session: Optional[str] = None,
        weight: float = 1.0,
    ) -> int:
        """Add a memory entry for a board member."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO board_memory
                   (member_id, memory_type, content, source_session, created_at, weight)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (member_id, memory_type, content, source_session, self._now(), weight),
            )
            return cur.lastrowid

    def get_memories(
        self,
        member_id: str,
        memory_type: Optional[str] = None,
        limit: int = 20,
    ) -> list:
        """Get active memories for a board member, highest weight first."""
        with self._conn() as conn:
            if memory_type:
                rows = conn.execute(
                    """SELECT * FROM board_memory
                       WHERE member_id = ? AND memory_type = ? AND active = 1
                       ORDER BY weight DESC, created_at DESC LIMIT ?""",
                    (member_id, memory_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM board_memory
                       WHERE member_id = ? AND active = 1
                       ORDER BY weight DESC, created_at DESC LIMIT ?""",
                    (member_id, limit),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_memory_count(self, member_id: str) -> int:
        """Count active memories for a member."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM board_memory WHERE member_id = ? AND active = 1",
                (member_id,)
            ).fetchone()[0]

    def build_memory_prompt(self, member_id: str) -> str:
        """Build the memory injection string for a member's system prompt.

        Returns empty string if no memories exist.
        """
        memories = self.get_memories(member_id, limit=15)
        if not memories:
            return ""

        by_type: dict = {}
        for m in memories:
            by_type.setdefault(m["memory_type"], []).append(m["content"])

        sections = ["## Your Board Memory\n"]
        type_labels = {
            "fact":       "Business Facts",
            "decision":   "Past Decisions",
            "preference": "George's Preferences",
            "context":    "Ongoing Context",
        }
        for mtype, label in type_labels.items():
            if mtype in by_type:
                sections.append(f"### {label}")
                for item in by_type[mtype]:
                    sections.append(f"- {item}")
                sections.append("")

        return "\n".join(sections)

    def archive_memory(self, memory_id: int):
        """Mark a memory as archived (inactive)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE board_memory SET active = 0 WHERE id = ?", (memory_id,)
            )

    # ─── Decisions ────────────────────────────────────────────────────────────

    def add_decision(
        self,
        session_id: str,
        content: str,
        owner: Optional[str] = None,
    ) -> int:
        """Record a decision from a board session."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO decisions (session_id, content, owner, created_at)
                   VALUES (?, ?, ?, ?)""",
                (session_id, content, owner, self._now()),
            )
            return cur.lastrowid

    def get_open_decisions(self) -> list:
        """Get all open decisions with their source session titles."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT d.*, s.title as session_title
                   FROM decisions d
                   JOIN sessions s ON d.session_id = s.id
                   WHERE d.status = 'open'
                   ORDER BY d.created_at DESC""",
            ).fetchall()
            return [dict(r) for r in rows]

    def close_decision(self, decision_id: int, status: str = "done"):
        """Mark a decision as done or dropped."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE decisions SET status = ? WHERE id = ?",
                (status, decision_id),
            )

    # ─── Briefing History ─────────────────────────────────────────────────────

    def save_briefing(self, content: str, source: str = "manual") -> int:
        """Save a new briefing version. Returns the version number."""
        with self._conn() as conn:
            version = (conn.execute(
                "SELECT MAX(version) FROM briefing_history"
            ).fetchone()[0] or 0) + 1
            conn.execute(
                """INSERT INTO briefing_history (content, version, created_at, source)
                   VALUES (?, ?, ?, ?)""",
                (content, version, self._now(), source),
            )
            return version

    def get_latest_briefing(self) -> Optional[str]:
        """Get the most recent briefing content."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content FROM briefing_history ORDER BY version DESC LIMIT 1"
            ).fetchone()
            return row["content"] if row else None

    def get_briefing_history(self) -> list:
        """Get all briefing versions (with truncated previews)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, version, created_at, source, SUBSTR(content, 1, 100) as preview "
                "FROM briefing_history ORDER BY version DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    # ─── AI-Powered Memory Extraction ─────────────────────────────────────────

    async def extract_memories_from_session(self, session_id: str) -> dict:
        """After a session ends, use Claude Haiku to extract structured memories.

        Extracts:
          - Key decisions made
          - Business facts revealed
          - George's preferences stated
          - Ongoing context to carry forward

        Returns dict of extracted items per category.
        """
        try:
            import anthropic
        except ImportError:
            logger.error("anthropic SDK not installed — cannot extract memories")
            return {}

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("No ANTHROPIC_API_KEY — skipping memory extraction")
            return {}

        messages = self.get_messages(session_id)
        if not messages:
            return {}

        # Build transcript from session messages
        transcript = "\n".join([
            f"{'USER' if m['role'] == 'user' else m['speaker'].upper()}: {m['content']}"
            for m in messages
        ])

        prompt = f"""Analyze this AI board meeting transcript and extract structured memory items.

TRANSCRIPT:
{transcript}

Return ONLY valid JSON in this exact format:
{{
  "decisions": ["decision 1", "decision 2"],
  "facts": ["business fact 1", "business fact 2"],
  "preferences": ["preference/constraint George stated"],
  "context": ["ongoing situation or thread to carry forward"]
}}

Rules:
- Be specific and concrete, not vague
- decisions: things explicitly agreed upon or resolved
- facts: specific business details revealed (pricing, customers, tools, metrics)
- preferences: how George likes things done, values, constraints
- context: unresolved questions, ongoing projects, things to revisit
- Maximum 5 items per category
- Empty array if nothing found in that category"""

        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

        except Exception as e:
            logger.error("Memory extraction API call failed: %s", e)
            return {}

        try:
            extracted = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Memory extraction returned invalid JSON: %s", text[:200])
            return {}

        # Persist extracted memories for ALL 7 board members
        for member_id in ALL_MEMBER_IDS:
            for item in extracted.get("decisions", []):
                self.add_memory(member_id, "decision", item, session_id, weight=0.9)
            for item in extracted.get("facts", []):
                self.add_memory(member_id, "fact", item, session_id, weight=0.8)
            for item in extracted.get("preferences", []):
                self.add_memory(member_id, "preference", item, session_id, weight=1.0)
            for item in extracted.get("context", []):
                self.add_memory(member_id, "context", item, session_id, weight=0.7)

        # Also add to decisions table
        for item in extracted.get("decisions", []):
            self.add_decision(session_id, item)

        total = sum(len(v) for v in extracted.values())
        logger.info("Extracted %d memories from session %s", total, session_id)
        return extracted

    async def summarize_old_memories(self, member_id: str, keep_recent: int = 10):
        """When a member accumulates >30 memories, condense older ones.

        Uses Claude Haiku to create summary memories, keeping the prompt
        injection lean.
        """
        try:
            import anthropic
        except ImportError:
            return

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return

        with self._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM board_memory WHERE member_id = ? AND active = 1",
                (member_id,)
            ).fetchone()[0]

        if count <= 30:
            return  # not needed yet

        # Get oldest memories beyond keep_recent
        with self._conn() as conn:
            old = conn.execute(
                """SELECT * FROM board_memory
                   WHERE member_id = ? AND active = 1
                   ORDER BY weight ASC, created_at ASC
                   LIMIT ?""",
                (member_id, count - keep_recent),
            ).fetchall()
            old = [dict(r) for r in old]

        if not old:
            return

        items = "\n".join([f"- [{r['memory_type']}] {r['content']}" for r in old])
        prompt = f"""Condense these board memory items into 3-5 concise summary bullets.
Preserve the most important business context. Be specific.

{items}

Return ONLY a JSON array of strings: ["summary 1", "summary 2", ...]"""

        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

        except Exception as e:
            logger.error("Memory summarization failed for %s: %s", member_id, e)
            return

        try:
            summaries = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Memory summarization returned invalid JSON")
            return

        # Archive old, add consolidated summary
        for old_mem in old:
            self.archive_memory(old_mem["id"])

        for summary in summaries:
            self.add_memory(member_id, "context", f"[Condensed] {summary}", weight=0.85)

        logger.info("Condensed %d memories into %d summaries for %s",
                     len(old), len(summaries), member_id)

    # ─── Export ───────────────────────────────────────────────────────────────

    def export_memory_report(self) -> str:
        """Generate a Markdown report of all active board memories."""
        lines = [
            "# AI Board of Directors — Memory Report",
            f"\n_Generated: {self._now()}_\n",
        ]

        # Open decisions
        decisions = self.get_open_decisions()
        if decisions:
            lines.append("## Open Decisions\n")
            for d in decisions:
                lines.append(f"- [ ] {d['content']} _(from: {d['session_title']})_")
            lines.append("")

        # Per-member memories
        lines.append("## Board Member Memories\n")
        for mid in ALL_MEMBER_IDS:
            memories = self.get_memories(mid)
            if not memories:
                continue
            lines.append(f"### {MEMBER_NAMES.get(mid, mid)}\n")
            by_type: dict = {}
            for m in memories:
                by_type.setdefault(m["memory_type"], []).append(m["content"])
            for mtype in ["decision", "fact", "preference", "context"]:
                if mtype in by_type:
                    lines.append(f"**{mtype.title()}s**")
                    for item in by_type[mtype]:
                        lines.append(f"- {item}")
                    lines.append("")

        # Session history
        sessions = self.list_sessions(limit=10)
        if sessions:
            lines.append("## Recent Sessions\n")
            for s in sessions:
                lines.append(
                    f"- **{s['title']}** — {s['created_at'][:10]} "
                    f"({s['message_count']} messages)"
                )

        return "\n".join(lines)
