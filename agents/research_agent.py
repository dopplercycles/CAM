"""
CAM Research Agent

Local in-process agent that intercepts research-related tasks and handles
them with a multi-source web research workflow. Searches DuckDuckGo,
fetches pages, combines findings with long-term memory context, and
synthesizes a structured summary.

Three research tiers:
  - Full:       research_synthesis / business_strategy tasks → search + 5 pages + synthesize
  - Light:      other research tasks → search + 2 pages + synthesize
  - Model-only: no web tools available → LTM context + model synthesis

"Local" means this runs inside the server process — no WebSocket, no
network hop. It's checked after the content agent in dispatch_to_agent().
If the task isn't research-related, it returns None and the flow
continues to remote agents unchanged.

Usage:
    from agents.research_agent import ResearchAgent

    agent = ResearchAgent(
        router=orchestrator.router,
        persona=persona,
        long_term_memory=long_term_memory,
        research_store=research_store,
        event_logger=event_logger,
    )

    # Called from dispatch_to_agent() — returns str or None
    result = await agent.try_handle(task, plan)
"""

import asyncio
import logging
import time
from typing import Any, Callable

from core.research_store import ResearchStore, ResearchStatus


logger = logging.getLogger("cam.research_agent")


# ---------------------------------------------------------------------------
# Research detection keywords — aligned with task_classifier.py
# ---------------------------------------------------------------------------

# Keywords from task_classifier's research_synthesis category, plus
# general research terms that signal an information-gathering task.
RESEARCH_KEYWORDS = [
    # From research_synthesis (Tier 3)
    "research", "synthesize", "combine sources", "literature review",
    "deep dive", "comprehensive analysis",
    # From business_strategy (Tier 3)
    "investigate", "market analysis",
    # General research terms
    "find out", "look up", "what's the current", "survey",
    "industry trend", "compare options", "pros and cons",
    "state of the art", "current market",
]

# Task types from the classifier that get the full web research workflow
# (search + fetch 5 pages + synthesize). Other research tasks get the
# lighter version (search + 2 pages).
FULL_RESEARCH_TYPES = {"research_synthesis", "business_strategy"}

# Research-specific system prompt for the synthesis model call
RESEARCH_SYSTEM_PROMPT = """
You are also serving as the Research Agent for Doppler Cycles. When
synthesizing research findings, follow these guidelines:

Structure your response as:
1. **Overview** — One-paragraph executive summary
2. **Key Findings** — Bulleted list of the most important takeaways
3. **Details** — Deeper analysis organized by theme or source
4. **Gaps & Limitations** — What the research couldn't answer
5. **Sources** — Numbered list of sources used with brief descriptions

Additional guidelines:
- Cite specific sources when making claims (e.g., "According to [Source 2]...")
- Include a confidence assessment (high/medium/low) for key findings
- Focus on the motorcycle industry when relevant — this is for a mobile
  diagnostics and content creation business in Portland, Oregon
- Flag any information that seems outdated or contradictory between sources
- Be direct and actionable — George needs info he can act on, not fluff
"""

# Content budget limits — keep prompts manageable for local models
MAX_PAGES_TO_FETCH = 5
MAX_TEXT_PER_PAGE = 2000
MAX_TOTAL_SOURCE_TEXT = 8000


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """Local research agent — intercepts research tasks in dispatch_to_agent().

    Handles research-related tasks with web searching, page fetching,
    and structured synthesis. Stores results in both long-term memory
    and the research SQLite store.

    Args:
        router:            ModelRouter instance for synthesis calls.
        persona:           Persona instance for building system prompts.
        long_term_memory:  LongTermMemory for context retrieval and result storage.
        research_store:    ResearchStore for persisting research results.
        event_logger:      EventLogger for audit trail.
        web_tool:          Optional WebTool instance (created internally if None).
        on_model_call:     Optional callback for model cost tracking.
    """

    def __init__(
        self,
        router,
        persona,
        long_term_memory,
        research_store: ResearchStore,
        event_logger,
        web_tool=None,
        on_model_call: Callable | None = None,
    ):
        self.router = router
        self.persona = persona
        self.long_term = long_term_memory
        self.store = research_store
        self.event_logger = event_logger
        self._on_model_call = on_model_call

        # Create WebTool — None if import fails (graceful degradation)
        self.web = web_tool
        if self.web is None:
            try:
                from tools.web import WebTool
                self.web = WebTool()
            except Exception:
                logger.warning("WebTool import failed — research agent will use model-only mode")
                self.web = None

        logger.info(
            "ResearchAgent initialized (web=%s)",
            "available" if self.web else "unavailable",
        )

    # -------------------------------------------------------------------
    # Detection
    # -------------------------------------------------------------------

    def is_research_task(self, description: str) -> bool:
        """Check if a task description is research-related.

        Uses lowercase keyword matching against RESEARCH_KEYWORDS.

        Args:
            description: The task description text.

        Returns:
            True if the task appears to be research-related.
        """
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in RESEARCH_KEYWORDS)

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    async def try_handle(self, task, plan: dict) -> str | None:
        """Try to handle a task as research work.

        Called from dispatch_to_agent() after the content agent.
        Returns the research result string if handled, or None to let
        the task fall through to remote agents / model fallback.

        Args:
            task: The Task object being dispatched.
            plan: The plan dict from the THINK phase.

        Returns:
            Research result string, or None if not a research task.
        """
        if not self.is_research_task(task.description):
            return None

        logger.info(
            "Research agent handling task %s: %s",
            task.short_id, task.description[:80],
        )

        # Create store entry (in_progress)
        entry = self.store.add_entry(
            query=task.description,
            task_id=task.task_id,
        )

        try:
            # Route to the appropriate research tier
            task_type = plan.get("task_type", "")

            if self.web is not None and task_type in FULL_RESEARCH_TYPES:
                result = await self._full_research(task, entry)
            elif self.web is not None:
                result = await self._light_research(task, entry)
            else:
                result = await self._model_only_research(task, entry)

            # Update store entry to completed
            self.store.update_entry(
                entry.entry_id,
                status=ResearchStatus.COMPLETED.value,
                summary=result[:2000] if result else "",
            )

            # Store in long-term memory
            if result and len(result) > 50:
                try:
                    ltm_content = f"Research: {task.description}\nSummary: {result[:500]}"
                    self.long_term.store(
                        content=ltm_content,
                        category="knowledge",
                        metadata={
                            "task_id": task.task_id,
                            "source": "research_agent",
                            "research_entry_id": entry.entry_id,
                        },
                    )
                except Exception:
                    logger.debug("Research agent LTM store failed (non-fatal)", exc_info=True)

            # Fire change notification for dashboard
            await self.store._notify_change()

            return result

        except Exception as e:
            logger.warning(
                "Research agent failed for task %s: %s",
                task.short_id, e,
            )
            self.store.update_entry(
                entry.entry_id,
                status=ResearchStatus.FAILED.value,
                summary=f"Research failed: {e}",
            )
            await self.store._notify_change()
            # Return None to let the task fall through to other agents
            return None

    # -------------------------------------------------------------------
    # Research tiers
    # -------------------------------------------------------------------

    async def _full_research(self, task, entry) -> str:
        """Full research: search DDG → fetch top 5 pages → LTM context → synthesize.

        Used for research_synthesis and business_strategy task types.
        """
        logger.info("Research agent: full research for %s", task.short_id)

        # Search
        search_results = await asyncio.to_thread(
            self.web.search, task.description, max_results=8,
        )
        source_dicts = [r.to_dict() for r in search_results]

        # Fetch top pages
        pages_text, pages_fetched = await self._fetch_pages(
            search_results[:MAX_PAGES_TO_FETCH], MAX_TEXT_PER_PAGE,
        )

        # Update store with source info
        self.store.update_entry(
            entry.entry_id,
            sources=source_dicts,
            source_count=len(search_results),
            pages_fetched=pages_fetched,
        )

        # LTM context
        ltm_context = self._get_ltm_context(task.description)

        # Synthesize
        return await self._synthesize(task, pages_text, ltm_context, entry)

    async def _light_research(self, task, entry) -> str:
        """Light research: search DDG → fetch 2 pages → LTM context → synthesize.

        Used for general research keyword matches.
        """
        logger.info("Research agent: light research for %s", task.short_id)

        # Search
        search_results = await asyncio.to_thread(
            self.web.search, task.description, max_results=5,
        )
        source_dicts = [r.to_dict() for r in search_results]

        # Fetch top 2 pages
        pages_text, pages_fetched = await self._fetch_pages(
            search_results[:2], MAX_TEXT_PER_PAGE,
        )

        # Update store with source info
        self.store.update_entry(
            entry.entry_id,
            sources=source_dicts,
            source_count=len(search_results),
            pages_fetched=pages_fetched,
        )

        # LTM context
        ltm_context = self._get_ltm_context(task.description)

        # Synthesize
        return await self._synthesize(task, pages_text, ltm_context, entry)

    async def _model_only_research(self, task, entry) -> str:
        """Model-only research: LTM context only → synthesize (no web).

        Fallback when web tools are unavailable.
        """
        logger.info("Research agent: model-only research for %s (no web)", task.short_id)

        ltm_context = self._get_ltm_context(task.description)

        return await self._synthesize(task, "", ltm_context, entry)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    async def _fetch_pages(
        self, search_results: list, max_text_per_page: int,
    ) -> tuple[str, int]:
        """Fetch pages sequentially with rate limiting.

        Args:
            search_results: List of SearchResult objects.
            max_text_per_page: Max chars to extract per page.

        Returns:
            Tuple of (combined_text, pages_successfully_fetched).
        """
        all_text_parts: list[str] = []
        total_chars = 0
        pages_fetched = 0

        for result in search_results:
            if total_chars >= MAX_TOTAL_SOURCE_TEXT:
                break

            page = await asyncio.to_thread(
                self.web.fetch_page, result.url, max_text_per_page,
            )

            if page.error:
                logger.debug("Skipping %s: %s", result.url, page.error)
                continue

            if page.text:
                # Truncate to fit within total budget
                remaining = MAX_TOTAL_SOURCE_TEXT - total_chars
                text = page.text[:remaining]

                all_text_parts.append(
                    f"[Source: {result.title}]\nURL: {result.url}\n{text}"
                )
                total_chars += len(text)
                pages_fetched += 1

            # Rate limit between requests
            await asyncio.sleep(0.5)

        return "\n\n---\n\n".join(all_text_parts), pages_fetched

    async def _synthesize(self, task, source_text: str, ltm_context: str, entry) -> str:
        """Build prompt and call the model for synthesis.

        Args:
            task:        The task being handled.
            source_text: Combined text from fetched pages.
            ltm_context: Relevant long-term memory context.
            entry:       ResearchEntry for cost tracking.

        Returns:
            The model's synthesized response text.
        """
        system_prompt = self.persona.build_system_prompt() + RESEARCH_SYSTEM_PROMPT

        # Build user prompt
        parts = [f"Research task: {task.description}"]

        if source_text:
            parts.append(f"\n\nWeb research sources:\n{source_text}")

        if ltm_context:
            parts.append(f"\n\n{ltm_context}")

        if not source_text and not ltm_context:
            parts.append(
                "\n\nNo web sources or prior research available. "
                "Synthesize the best answer you can from your training data, "
                "and clearly note the lack of current sources."
            )

        prompt = "".join(parts)

        logger.info(
            "Research agent synthesizing for %s (prompt ~%d chars)",
            task.short_id, len(prompt),
        )

        response = await self.router.route(
            prompt=prompt,
            task_complexity="tier3",
            system_prompt=system_prompt,
        )

        # Track costs
        self.store.update_entry(
            entry.entry_id,
            model_used=response.model,
            tokens_used=response.total_tokens,
            cost_usd=response.cost_usd,
        )

        # Notify model call listener
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
                logger.debug("Research agent model call callback error", exc_info=True)

        logger.info(
            "Research agent synthesis complete for %s: model=%s, tokens=%d",
            task.short_id, response.model, response.total_tokens,
        )

        return response.text

    def _get_ltm_context(self, description: str) -> str:
        """Query long-term memory for relevant past research.

        Args:
            description: Task description for the LTM query.

        Returns:
            Formatted context string, or empty string if nothing relevant.
        """
        try:
            ltm_results = self.long_term.query(description, top_k=3)
            relevant = [r for r in ltm_results if r.score > 0.3]
            if relevant:
                ltm_lines = [f"- [{r.category}] {r.content}" for r in relevant]
                return (
                    "\nRelevant past research from memory:\n"
                    + "\n".join(ltm_lines)
                )
        except Exception:
            logger.debug("Research agent LTM query failed (non-fatal)", exc_info=True)
        return ""
