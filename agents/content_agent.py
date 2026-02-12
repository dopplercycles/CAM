"""
CAM Content Agent

Local in-process agent that intercepts content-related tasks and handles
them with content-specific prompting. Stores results in long-term memory
and auto-creates content calendar entries for substantial output.

"Local" means this runs inside the server process — no WebSocket, no
network hop. It's checked first in dispatch_to_agent(). If the task
isn't content-related, it returns None and the flow continues to
remote agents unchanged.

Usage:
    from agents.content_agent import ContentAgent

    agent = ContentAgent(
        router=orchestrator.router,
        persona=persona,
        long_term_memory=long_term_memory,
        calendar=content_calendar,
        event_logger=event_logger,
    )

    # Called from dispatch_to_agent() — returns str or None
    result = await agent.try_handle(task, plan)
"""

import logging
from typing import Any, Callable

from core.content_calendar import ContentCalendar, ContentType, ContentStatus


logger = logging.getLogger("cam.content_agent")


# ---------------------------------------------------------------------------
# Content detection keywords — aligned with task_classifier.py
# ---------------------------------------------------------------------------

# Keywords from task_classifier's content_outlines, complex_writing,
# and creative_work categories, plus general content terms.
CONTENT_KEYWORDS = [
    # From content_outlines (Tier 2)
    "outline", "blog outline", "video outline", "content structure",
    "script outline", "episode outline",
    # From complex_writing (Tier 3)
    "write script", "youtube script", "long-form", "technical doc",
    "write article", "blog post", "full script",
    # From creative_work (Tier 3)
    "video concept", "branding", "creative", "narrative",
    "storyline", "episode idea", "thumbnail concept",
    # General content terms
    "content", "script", "episode", "video title", "description draft",
    "show notes", "intro", "outro", "hook",
    # TTS / voice synthesis
    "tts", "synthesize", "voice", "audio", "speech", "narrate", "narration",
]

# Keywords that trigger automatic TTS synthesis after content generation
TTS_TRIGGER_KEYWORDS = ["synthesize", "tts", "voice over", "narrate", "narration", "speak"]

# Task types that warrant a dedicated model call with the content
# system prompt (Tier 3 creative/writing tasks). Other content tasks
# reuse the THINK phase response to respect frugality.
DEDICATED_CALL_TYPES = {"complex_writing", "creative_work"}

# Content-specific system prompt appended to the persona's base prompt
# when making a dedicated content model call.
CONTENT_SYSTEM_PROMPT = """
You are also serving as the Content Agent for Doppler Cycles. When generating
content (scripts, outlines, descriptions, titles), follow these guidelines:

- Write in Cam's voice: knowledgeable, conversational, enthusiast-level
  motorcycle depth. Not corporate, not clickbait.
- Script structure: hook → context → meat → takeaway → call to action.
  The hook should make a rider stop scrolling.
- Audience: experienced riders and motorcycle enthusiasts who can smell
  BS from a mile away. Respect their intelligence.
- Shop counter test: would George say this to a customer standing in
  front of him? If not, rewrite it.
- Always include technical accuracy notes or flag areas that need
  George's verification.
- For video content, include visual cues and B-roll suggestions
  in [brackets].
"""


# ---------------------------------------------------------------------------
# Content type inference — maps keywords to ContentType
# ---------------------------------------------------------------------------

_TYPE_KEYWORDS: dict[str, ContentType] = {
    "script": ContentType.SCRIPT_OUTLINE,
    "outline": ContentType.SCRIPT_OUTLINE,
    "video": ContentType.VIDEO,
    "episode": ContentType.VIDEO,
    "thumbnail": ContentType.VIDEO,
    "research": ContentType.RESEARCH,
    "investigate": ContentType.RESEARCH,
    "deep dive": ContentType.RESEARCH,
    "description": ContentType.EPISODE_DESCRIPTION,
    "show notes": ContentType.EPISODE_DESCRIPTION,
}


# ---------------------------------------------------------------------------
# ContentAgent
# ---------------------------------------------------------------------------

class ContentAgent:
    """Local content agent — intercepts content tasks in dispatch_to_agent().

    Handles content-related tasks with content-specific prompting,
    stores results in long-term memory, and auto-creates calendar
    entries for substantial output.

    Args:
        router:           ModelRouter instance for dedicated content calls.
        persona:          Persona instance for building system prompts.
        long_term_memory: LongTermMemory for storing content results.
        calendar:         ContentCalendar for tracking content pipeline.
        event_logger:     EventLogger for audit trail.
        on_model_call:    Optional callback for model cost tracking.
    """

    def __init__(
        self,
        router,
        persona,
        long_term_memory,
        calendar: ContentCalendar,
        event_logger,
        on_model_call: Callable | None = None,
        tts_pipeline=None,
    ):
        self.router = router
        self.persona = persona
        self.long_term = long_term_memory
        self.calendar = calendar
        self.event_logger = event_logger
        self._on_model_call = on_model_call
        self.tts = tts_pipeline

        logger.info(
            "ContentAgent initialized (tts=%s)",
            "available" if tts_pipeline else "none",
        )

    # -------------------------------------------------------------------
    # Detection
    # -------------------------------------------------------------------

    def is_content_task(self, description: str) -> bool:
        """Check if a task description is content-related.

        Uses lowercase keyword matching against CONTENT_KEYWORDS.

        Args:
            description: The task description text.

        Returns:
            True if the task appears to be content-related.
        """
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in CONTENT_KEYWORDS)

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    async def try_handle(self, task, plan: dict) -> str | None:
        """Try to handle a task as content work.

        Called from dispatch_to_agent() before remote agent routing.
        Returns the content result string if handled, or None to let
        the task fall through to remote agents / model fallback.

        Args:
            task: The Task object being dispatched.
            plan: The plan dict from the THINK phase.

        Returns:
            Content result string, or None if not a content task.
        """
        if not self.is_content_task(task.description):
            return None

        logger.info(
            "Content agent handling task %s: %s",
            task.short_id, task.description[:80],
        )

        # --- Retrieve relevant past content from long-term memory ---
        ltm_context = ""
        try:
            ltm_results = self.long_term.query(task.description, top_k=3)
            relevant = [r for r in ltm_results if r.score > 0.3]
            if relevant:
                ltm_lines = [f"- [{r.category}] {r.content}" for r in relevant]
                ltm_context = (
                    "\n\nRelevant past content from memory:\n"
                    + "\n".join(ltm_lines)
                )
                logger.info(
                    "Content agent retrieved %d relevant memories for %s",
                    len(relevant), task.short_id,
                )
        except Exception:
            logger.debug("Content agent LTM query failed (non-fatal)", exc_info=True)

        # --- Decide whether to make a dedicated model call ---
        task_type = plan.get("task_type", "")
        if task_type in DEDICATED_CALL_TYPES:
            # Tier 3 creative/writing — make a content-specific model call
            result = await self._dedicated_call(task, ltm_context)
        else:
            # Use the THINK phase response — no double-spend
            result = plan.get("model_response", "")
            logger.info(
                "Content agent reusing THINK response for %s (type=%s)",
                task.short_id, task_type,
            )

        # --- Store result in long-term memory ---
        if result and len(result) > 50:
            try:
                ltm_content = f"Content task: {task.description}\nResult: {result[:500]}"
                self.long_term.store(
                    content=ltm_content,
                    category="knowledge",
                    metadata={
                        "task_id": task.task_id,
                        "source": "content_agent",
                        "content_type": self._infer_content_type(task.description).value,
                    },
                )
            except Exception:
                logger.debug("Content agent LTM store failed (non-fatal)", exc_info=True)

        # --- Auto-create calendar entry for substantial output ---
        if result and len(result) > 100:
            try:
                content_type = self._infer_content_type(task.description)
                title = self._infer_title(task.description)
                entry = self.calendar.add_entry(
                    title=title,
                    content_type=content_type.value,
                    description=task.description[:200],
                    body=result[:2000],
                    task_id=task.task_id,
                )
                # Fire the calendar's on_change callback for dashboard broadcast
                if self.calendar._on_change is not None:
                    await self.calendar._notify_change()
                logger.info(
                    "Content agent auto-created calendar entry '%s' (%s)",
                    title, entry.short_id,
                )
            except Exception:
                logger.debug("Content agent calendar entry failed (non-fatal)", exc_info=True)

        # --- Auto-synthesize if task requests TTS ---
        if result and self.tts and self._should_auto_synthesize(task.description):
            try:
                tts_result = await self.tts.synthesize(result)
                if tts_result.error:
                    logger.info(
                        "Content agent auto-TTS failed for %s: %s",
                        task.short_id, tts_result.error,
                    )
                else:
                    result += f"\n\n[Audio synthesized: {tts_result.audio_path} ({tts_result.duration_secs:.1f}s)]"
                    logger.info(
                        "Content agent auto-synthesized audio for %s: %s",
                        task.short_id, tts_result.audio_path,
                    )
            except Exception:
                logger.debug("Content agent auto-TTS failed (non-fatal)", exc_info=True)

        return result

    # -------------------------------------------------------------------
    # Dedicated model call for Tier 3 content tasks
    # -------------------------------------------------------------------

    async def _dedicated_call(self, task, ltm_context: str) -> str:
        """Make a content-specific model call for creative/writing tasks.

        Uses the persona's base system prompt plus the content-specific
        addendum for better content output quality.

        Args:
            task:        The task being handled.
            ltm_context: Relevant long-term memory context string.

        Returns:
            The model's response text.
        """
        system_prompt = self.persona.build_system_prompt() + CONTENT_SYSTEM_PROMPT
        prompt = task.description + ltm_context

        logger.info(
            "Content agent making dedicated model call for %s",
            task.short_id,
        )

        response = await self.router.route(
            prompt=prompt,
            task_complexity="tier3",
            system_prompt=system_prompt,
        )

        # Notify model call listener for cost tracking
        if self._on_model_call is not None:
            try:
                self._on_model_call(
                    model=response.model,
                    backend=response.backend,
                    tokens=response.total_tokens,
                    latency_ms=response.latency_ms,
                    cost_usd=response.cost_usd,
                    task_short_id=task.short_id,
                )
            except Exception:
                logger.debug("Content agent model call callback error", exc_info=True)

        logger.info(
            "Content agent dedicated call complete for %s: model=%s, tokens=%d",
            task.short_id, response.model, response.total_tokens,
        )

        return response.text

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _infer_content_type(self, description: str) -> ContentType:
        """Infer the content type from a task description.

        Scans for keywords and returns the first match.
        Defaults to GENERAL if nothing matches.

        Args:
            description: Task description text.

        Returns:
            The inferred ContentType.
        """
        desc_lower = description.lower()
        for keyword, ctype in _TYPE_KEYWORDS.items():
            if keyword in desc_lower:
                return ctype
        return ContentType.GENERAL

    def _infer_title(self, description: str) -> str:
        """Infer a short title from a task description.

        Takes the first 60 characters, cleaned up to not cut mid-word.

        Args:
            description: Task description text.

        Returns:
            A short title string.
        """
        # Strip common task prefixes
        title = description.strip()
        for prefix in ("write ", "draft ", "create ", "generate "):
            if title.lower().startswith(prefix):
                title = title[len(prefix):]
                break

        # Truncate to 60 chars at a word boundary
        if len(title) > 60:
            title = title[:60].rsplit(" ", 1)[0]

        return title.strip() or "Untitled Content"

    # -------------------------------------------------------------------
    # TTS synthesis
    # -------------------------------------------------------------------

    async def synthesize(self, text: str, voice: str | None = None):
        """Synthesize text to speech using the TTS pipeline.

        Args:
            text:  Text to synthesize.
            voice: Optional voice model name.

        Returns:
            SynthesisResult if TTS is available, None otherwise.
        """
        if not self.tts:
            logger.debug("TTS synthesis requested but no TTS pipeline configured")
            return None
        return await self.tts.synthesize(text=text, voice=voice)

    def _should_auto_synthesize(self, description: str) -> bool:
        """Check if a task description requests TTS synthesis.

        Args:
            description: Task description text.

        Returns:
            True if the task mentions TTS trigger keywords.
        """
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in TTS_TRIGGER_KEYWORDS)
