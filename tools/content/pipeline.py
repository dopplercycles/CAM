"""
CAM Content Pipeline Manager

Manages the full content production workflow:
    Research → Outline → Script → Review → TTS → Package

Stages 1-3 (Research → Outline → Script) run as a TaskChain — the
orchestrator's existing keyword-based dispatch routes each step to the
right agent automatically. Stage 4 (Review) pauses for George's approval.
Stages 5-6 (TTS → Package) are direct tool calls.

Usage:
    from tools.content.pipeline import ContentPipelineManager

    pipeline_mgr = ContentPipelineManager(
        task_queue=task_queue,
        tts_pipeline=tts_pipeline,
        event_logger=event_logger,
    )
    pipeline = pipeline_mgr.create_pipeline("Valve Adjustment", "How to do a valve adjustment on a Harley M-8")
"""

import json
import logging
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from core.task import Task, TaskChain, TaskComplexity, ChainStatus, TaskStatus


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.content_pipeline")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_STAGES = ("research", "outline", "script", "review", "tts", "package", "done")
PIPELINE_STATUSES = ("active", "paused", "completed", "failed", "cancelled")


# ---------------------------------------------------------------------------
# ContentPipelineManager
# ---------------------------------------------------------------------------

class ContentPipelineManager:
    """Manages content production pipelines from topic to packaged output.

    Each pipeline creates a 3-step TaskChain for the AI-driven stages
    (Research → Outline → Script), then pauses for human review before
    running TTS and packaging the final output.

    Args:
        db_path:        Path to the SQLite database file.
        task_queue:     The TaskQueue instance for creating chains.
        tts_pipeline:   The TTSPipeline instance for voice synthesis.
        event_logger:   The EventLogger for audit logging.
        content_calendar: Optional ContentCalendar for linking entries.
        on_change:      Async callback fired after any state change.
    """

    def __init__(
        self,
        db_path: str = "data/content_pipeline.db",
        task_queue: Any = None,
        tts_pipeline: Any = None,
        event_logger: Any = None,
        content_calendar: Any = None,
        on_change: Callable[[], Awaitable[None]] | None = None,
    ):
        self._db_path = db_path
        self._task_queue = task_queue
        self._tts_pipeline = tts_pipeline
        self._event_logger = event_logger
        self._content_calendar = content_calendar
        self._on_change = on_change

        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

        logger.info("ContentPipelineManager initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self):
        """Create tables and indexes if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS content_pipelines (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_id     TEXT UNIQUE NOT NULL,
                title           TEXT NOT NULL,
                topic           TEXT NOT NULL,
                stage           TEXT NOT NULL DEFAULT 'research',
                status          TEXT NOT NULL DEFAULT 'active',
                chain_id        TEXT,
                calendar_entry_id TEXT,
                research_result TEXT,
                outline_result  TEXT,
                script_result   TEXT,
                review_notes    TEXT,
                tts_audio_path  TEXT,
                tts_duration    REAL,
                package_path    TEXT,
                error           TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                completed_at    TEXT,
                metadata        TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_pipeline_stage ON content_pipelines(stage);
            CREATE INDEX IF NOT EXISTS idx_pipeline_status ON content_pipelines(status);
            CREATE INDEX IF NOT EXISTS idx_pipeline_chain ON content_pipelines(chain_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _now_iso(self) -> str:
        """UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a plain dict with parsed metadata."""
        d = dict(row)
        # Parse JSON metadata
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = {}
        return d

    async def _notify_change(self):
        """Fire the on_change callback if set."""
        if self._on_change is not None:
            try:
                await self._on_change()
            except Exception:
                logger.debug("Pipeline on_change callback error", exc_info=True)

    def _log(self, level: str, message: str, **details):
        """Log an event via event_logger if available."""
        if self._event_logger is not None:
            getattr(self._event_logger, level, self._event_logger.info)(
                "content", message, **details
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_pipeline(
        self,
        title: str,
        topic: str,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Create a new content pipeline and its 3-step TaskChain.

        The chain's steps are worded so keyword-based agent dispatch
        routes them correctly:
          Step 1 → ResearchAgent  (keyword: "Research")
          Step 2 → ContentAgent   (keyword: "outline", "content")
          Step 3 → ContentAgent   (keyword: "script", "content")

        Returns:
            The newly created pipeline as a dict.
        """
        pipeline_id = str(uuid.uuid4())
        now = self._now_iso()
        meta_json = json.dumps(metadata or {})

        # Build the 3-step TaskChain
        chain = TaskChain(
            name=f"Content Pipeline: {title}",
            source="pipeline",
            steps=[
                Task(
                    description=(
                        f"Research the following topic thoroughly for a YouTube video: {topic}"
                    ),
                    source="pipeline",
                    complexity=TaskComplexity.HIGH,
                ),
                Task(
                    description=(
                        f"Write a detailed content outline for a YouTube video about: {topic}"
                    ),
                    source="pipeline",
                    complexity=TaskComplexity.HIGH,
                ),
                Task(
                    description=(
                        f"Write a complete YouTube script in Cam's voice about: {topic}"
                    ),
                    source="pipeline",
                    complexity=TaskComplexity.HIGH,
                ),
            ],
        )
        chain.status = ChainStatus.PENDING

        # Register the chain with the task queue
        if self._task_queue is not None:
            self._task_queue.add_chain(chain)

        # Persist the pipeline
        self._conn.execute(
            """INSERT INTO content_pipelines
               (pipeline_id, title, topic, stage, status, chain_id,
                created_at, updated_at, metadata)
               VALUES (?, ?, ?, 'research', 'active', ?, ?, ?, ?)""",
            (pipeline_id, title, topic, chain.chain_id, now, now, meta_json),
        )
        self._conn.commit()

        self._log("info", f"Content pipeline '{title}' created",
                  pipeline_id=pipeline_id[:8], topic=topic[:60])

        return self.get_pipeline(pipeline_id)

    def get_pipeline(self, pipeline_id: str) -> dict[str, Any] | None:
        """Look up a single pipeline by its UUID."""
        row = self._conn.execute(
            "SELECT * FROM content_pipelines WHERE pipeline_id = ?",
            (pipeline_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_all(self, status: str | None = None) -> list[dict[str, Any]]:
        """Return all pipelines, optionally filtered by status.

        Results are ordered newest-first.
        """
        if status:
            rows = self._conn.execute(
                "SELECT * FROM content_pipelines WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM content_pipelines ORDER BY created_at DESC",
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _update_pipeline(self, pipeline_id: str, **fields) -> dict[str, Any] | None:
        """Update arbitrary fields on a pipeline row."""
        fields["updated_at"] = self._now_iso()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [pipeline_id]
        self._conn.execute(
            f"UPDATE content_pipelines SET {set_clause} WHERE pipeline_id = ?",
            values,
        )
        self._conn.commit()
        return self.get_pipeline(pipeline_id)

    # ------------------------------------------------------------------
    # Review actions
    # ------------------------------------------------------------------

    async def approve_review(
        self, pipeline_id: str, notes: str = ""
    ) -> dict[str, Any] | None:
        """Approve a pipeline in review stage — advances to TTS.

        Args:
            pipeline_id: Pipeline UUID.
            notes:       Optional reviewer notes.

        Returns:
            Updated pipeline dict, or None if not found / not in review.
        """
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline or pipeline["stage"] != "review" or pipeline["status"] != "paused":
            return None

        self._update_pipeline(
            pipeline_id,
            stage="tts",
            status="active",
            review_notes=notes,
        )
        self._log("info", f"Pipeline '{pipeline['title']}' approved for TTS",
                  pipeline_id=pipeline_id[:8])
        await self._notify_change()
        return self.get_pipeline(pipeline_id)

    async def reject_review(
        self, pipeline_id: str, notes: str = ""
    ) -> dict[str, Any] | None:
        """Reject a pipeline in review — marks it as failed.

        Args:
            pipeline_id: Pipeline UUID.
            notes:       Reason for rejection.

        Returns:
            Updated pipeline dict, or None if not found / not in review.
        """
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline or pipeline["stage"] != "review" or pipeline["status"] != "paused":
            return None

        self._update_pipeline(
            pipeline_id,
            status="failed",
            review_notes=notes,
            error=f"Rejected: {notes}" if notes else "Rejected by reviewer",
            completed_at=self._now_iso(),
        )
        self._log("warn", f"Pipeline '{pipeline['title']}' rejected",
                  pipeline_id=pipeline_id[:8], notes=notes[:100] if notes else "")
        await self._notify_change()
        return self.get_pipeline(pipeline_id)

    async def cancel_pipeline(self, pipeline_id: str) -> dict[str, Any] | None:
        """Cancel an active or paused pipeline.

        Returns:
            Updated pipeline dict, or None if not found / already done.
        """
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline or pipeline["status"] in ("completed", "cancelled"):
            return None

        self._update_pipeline(
            pipeline_id,
            status="cancelled",
            error="Cancelled by user",
            completed_at=self._now_iso(),
        )
        self._log("info", f"Pipeline '{pipeline['title']}' cancelled",
                  pipeline_id=pipeline_id[:8])
        await self._notify_change()
        return self.get_pipeline(pipeline_id)

    # ------------------------------------------------------------------
    # Background progress checker
    # ------------------------------------------------------------------

    async def check_progress(self):
        """Check all active pipelines and advance stages as needed.

        Called periodically (every ~5s) from the server's background loop.
        Checks TaskChain status for stages 1-3, runs TTS and packaging
        for stages 5-6.
        """
        active = self.list_all(status="active")
        changed = False

        for pipeline in active:
            try:
                stage = pipeline["stage"]
                pid = pipeline["pipeline_id"]

                if stage in ("research", "outline", "script"):
                    changed |= await self._check_chain_progress(pipeline)
                elif stage == "tts":
                    await self._run_tts(pid)
                    changed = True
                elif stage == "package":
                    await self._run_package(pid)
                    changed = True
            except Exception as e:
                logger.error("Pipeline %s check_progress error: %s",
                             pipeline["pipeline_id"][:8], e, exc_info=True)
                self._update_pipeline(
                    pipeline["pipeline_id"],
                    status="failed",
                    error=str(e),
                    completed_at=self._now_iso(),
                )
                changed = True

        if changed:
            await self._notify_change()

    async def _check_chain_progress(self, pipeline: dict) -> bool:
        """Check if the TaskChain has advanced and update pipeline stage.

        Returns True if the pipeline was updated.
        """
        if self._task_queue is None:
            return False

        chain_id = pipeline.get("chain_id")
        if not chain_id:
            return False

        chain = self._task_queue.get_chain(chain_id)
        if chain is None:
            return False

        pid = pipeline["pipeline_id"]

        # Map chain step index to pipeline stage
        step_to_stage = {0: "research", 1: "outline", 2: "script"}

        # Check if chain failed
        if chain.status == ChainStatus.FAILED:
            # Find the error from the failed step
            error_msg = "Chain step failed"
            for step in chain.steps:
                if step.status == TaskStatus.FAILED:
                    error_msg = step.result or "Chain step failed"
                    break
            self._update_pipeline(
                pid, status="failed", error=error_msg,
                completed_at=self._now_iso(),
            )
            self._log("error", f"Pipeline '{pipeline['title']}' failed: {error_msg[:80]}",
                      pipeline_id=pid[:8])
            return True

        # Check if chain completed (all 3 steps done)
        if chain.status == ChainStatus.COMPLETED:
            self._advance_to_review(pipeline, chain)
            return True

        # Update stage based on current step
        current_stage = step_to_stage.get(chain.current_step, pipeline["stage"])
        if current_stage != pipeline["stage"]:
            self._update_pipeline(pid, stage=current_stage)
            self._log("info",
                      f"Pipeline '{pipeline['title']}' advanced to {current_stage}",
                      pipeline_id=pid[:8])
            return True

        return False

    def _advance_to_review(self, pipeline: dict, chain: TaskChain):
        """Extract chain results and move pipeline to review stage."""
        pid = pipeline["pipeline_id"]

        # Extract results from each chain step
        research_result = chain.steps[0].result if len(chain.steps) > 0 else ""
        outline_result = chain.steps[1].result if len(chain.steps) > 1 else ""
        script_result = chain.steps[2].result if len(chain.steps) > 2 else ""

        self._update_pipeline(
            pid,
            stage="review",
            status="paused",
            research_result=research_result or "",
            outline_result=outline_result or "",
            script_result=script_result or "",
        )
        self._log("info",
                  f"Pipeline '{pipeline['title']}' ready for review",
                  pipeline_id=pid[:8])

    async def _run_tts(self, pipeline_id: str):
        """Run TTS synthesis on the script and advance to package stage."""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            return

        script_text = pipeline.get("script_result") or ""
        if not script_text:
            # No script to synthesize — skip TTS, go straight to package
            self._update_pipeline(pipeline_id, stage="package")
            self._log("warn", f"Pipeline '{pipeline['title']}' has no script for TTS, skipping",
                      pipeline_id=pipeline_id[:8])
            return

        if self._tts_pipeline is None:
            # No TTS pipeline available — skip to package
            self._update_pipeline(pipeline_id, stage="package")
            self._log("warn", "TTS pipeline not available, skipping synthesis",
                      pipeline_id=pipeline_id[:8])
            return

        try:
            result = await self._tts_pipeline.synthesize(
                text=script_text,
                filename=f"pipeline_{pipeline_id[:8]}.wav",
            )
            if result.error:
                self._log("warn", f"TTS synthesis warning: {result.error}",
                          pipeline_id=pipeline_id[:8])
            self._update_pipeline(
                pipeline_id,
                stage="package",
                tts_audio_path=result.audio_path or "",
                tts_duration=result.duration_secs,
            )
            self._log("info", f"TTS complete for pipeline '{pipeline['title']}'",
                      pipeline_id=pipeline_id[:8],
                      duration=result.duration_secs)
        except Exception as e:
            logger.error("TTS failed for pipeline %s: %s", pipeline_id[:8], e)
            # Don't fail the whole pipeline — just skip TTS
            self._update_pipeline(pipeline_id, stage="package")
            self._log("warn", f"TTS failed, skipping: {e}",
                      pipeline_id=pipeline_id[:8])

    async def _run_package(self, pipeline_id: str):
        """Assemble the output folder with all pipeline artifacts."""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            return

        # Create output directory
        output_dir = Path(f"data/content/pipelines/{pipeline_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write script
        script_text = pipeline.get("script_result") or ""
        if script_text:
            (output_dir / "script.txt").write_text(script_text, encoding="utf-8")

        # Write outline
        outline_text = pipeline.get("outline_result") or ""
        if outline_text:
            (output_dir / "outline.md").write_text(outline_text, encoding="utf-8")

        # Copy audio file if it exists
        audio_path = pipeline.get("tts_audio_path")
        if audio_path and os.path.isfile(audio_path):
            dest = output_dir / "audio.wav"
            shutil.copy2(audio_path, dest)

        # Write metadata
        meta = {
            "pipeline_id": pipeline_id,
            "title": pipeline["title"],
            "topic": pipeline["topic"],
            "created_at": pipeline["created_at"],
            "completed_at": self._now_iso(),
            "review_notes": pipeline.get("review_notes") or "",
            "tts_duration": pipeline.get("tts_duration"),
            "metadata": pipeline.get("metadata", {}),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        # Mark done
        self._update_pipeline(
            pipeline_id,
            stage="done",
            status="completed",
            package_path=str(output_dir),
            completed_at=self._now_iso(),
        )
        self._log("info", f"Pipeline '{pipeline['title']}' packaged at {output_dir}",
                  pipeline_id=pipeline_id[:8])

    # ------------------------------------------------------------------
    # Status / broadcast
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return aggregate counts by stage and status."""
        stage_counts = {}
        for stage in PIPELINE_STAGES:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM content_pipelines WHERE stage = ?",
                (stage,),
            ).fetchone()
            stage_counts[stage] = row[0] if row else 0

        status_counts = {}
        for status in PIPELINE_STATUSES:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM content_pipelines WHERE status = ?",
                (status,),
            ).fetchone()
            status_counts[status] = row[0] if row else 0

        total = self._conn.execute(
            "SELECT COUNT(*) FROM content_pipelines"
        ).fetchone()

        return {
            "total": total[0] if total else 0,
            "by_stage": stage_counts,
            "by_status": status_counts,
        }

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Full state for WebSocket push: pipelines list + status summary."""
        return {
            "pipelines": self.list_all(),
            "status": self.get_status(),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close the SQLite connection."""
        if self._conn:
            self._conn.close()
            logger.info("ContentPipelineManager closed")


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def test():
        """Quick smoke test for CRUD and status."""
        db_path = "data/content_pipeline_test.db"
        # Clean up any previous test DB
        if os.path.exists(db_path):
            os.remove(db_path)

        mgr = ContentPipelineManager(db_path=db_path)

        # Create
        p = mgr.create_pipeline("Test Video", "How to adjust valves on a Harley M-8")
        print(f"Created pipeline: {p['pipeline_id'][:8]} — {p['title']}")
        assert p["stage"] == "research"
        assert p["status"] == "active"

        # List
        all_pipelines = mgr.list_all()
        assert len(all_pipelines) == 1
        print(f"List all: {len(all_pipelines)} pipeline(s)")

        # Get
        fetched = mgr.get_pipeline(p["pipeline_id"])
        assert fetched is not None
        assert fetched["title"] == "Test Video"
        print(f"Get pipeline: {fetched['title']}")

        # Status
        status = mgr.get_status()
        print(f"Status: {status}")
        assert status["total"] == 1
        assert status["by_stage"]["research"] == 1
        assert status["by_status"]["active"] == 1

        # Broadcast dict
        bd = mgr.to_broadcast_dict()
        assert "pipelines" in bd
        assert "status" in bd
        print(f"Broadcast dict keys: {list(bd.keys())}")

        # Simulate advancing to review
        mgr._update_pipeline(p["pipeline_id"],
                             stage="review", status="paused",
                             research_result="Research data here",
                             outline_result="Outline data here",
                             script_result="Script data here")

        # Approve
        approved = await mgr.approve_review(p["pipeline_id"], notes="Looks good!")
        assert approved is not None
        assert approved["stage"] == "tts"
        assert approved["status"] == "active"
        print(f"Approved: stage={approved['stage']}, status={approved['status']}")

        # Create a second pipeline and cancel it
        p2 = mgr.create_pipeline("Cancelled Video", "This topic was a bad idea")
        cancelled = await mgr.cancel_pipeline(p2["pipeline_id"])
        assert cancelled["status"] == "cancelled"
        print(f"Cancelled: status={cancelled['status']}")

        # Create a third and reject it
        p3 = mgr.create_pipeline("Rejected Video", "Needs rework")
        mgr._update_pipeline(p3["pipeline_id"], stage="review", status="paused",
                             script_result="Draft script")
        rejected = await mgr.reject_review(p3["pipeline_id"], notes="Needs more detail")
        assert rejected["status"] == "failed"
        print(f"Rejected: status={rejected['status']}")

        # Final status
        final_status = mgr.get_status()
        print(f"Final status: {final_status}")

        mgr.close()

        # Cleanup test DB
        os.remove(db_path)
        print("\nAll tests passed!")

    asyncio.run(test())
