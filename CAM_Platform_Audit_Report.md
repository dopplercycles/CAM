# CAM Platform Extraction Audit Report

**Date:** 2026-03-02
**Auditor:** Claude Code (Opus 4.6)
**Codebase Version:** HEAD as of audit date
**Scope:** All source files in `/home/george/CAM/` (excluding `.venv/`, `venv/`, `data/`, `__pycache__/`)

---

## 1. Executive Summary

### Files Audited

| Category | Files | Percentage |
|----------|-------|------------|
| **Python source files** | 100 | — |
| **HTML/static files** | 2 | — |
| **Config/YAML/TOML** | 10 | — |
| **Shell scripts** | 4 | — |
| **Documentation (MD)** | 16 | — |
| **Total project files** | ~132 | — |

### Lines of Code

| Type | Lines |
|------|-------|
| Python | 61,841 |
| HTML (index.html + login.html) | ~25,000 |
| Config/YAML/TOML | ~600 |
| Documentation | ~4,000 |
| **Total** | **~91,400** |

### Category Breakdown

| Category | Files | % of Source | Description |
|----------|-------|-------------|-------------|
| **UNIVERSAL** | ~55 | ~50% | Platform core — works for any business |
| **CONFIGURABLE** | ~45 | ~40% | Follows universal patterns, Doppler content swappable |
| **DOPPLER-ONLY** | ~12 | ~10% | George's proprietary IP / competitive moat |

### Overall Assessment

**Platform extraction is highly feasible.** The codebase was architecturally designed for this — whether intentionally or through good engineering instincts. The separation between infrastructure and domain content is remarkably clean:

- **Core infrastructure** (`core/`, `security/`, `agents/connector.py`) is almost entirely domain-agnostic
- **Business logic** (`tools/doppler/`) follows consistent patterns that generalize trivially
- **Domain knowledge** lives in data files (`CAM_BRAIN.md`, `persona.yaml`, `diagnostic_trees/`), not in code
- **The dashboard** has zero Doppler-specific hardcoding in its 24,735-line frontend

### Estimated Extraction Effort

| Phase | Duration | Description |
|-------|----------|-------------|
| Phase 1: Quick wins | 1-2 weeks | Rename/restructure, strip hardcoded references |
| Phase 2: Core extraction | 3-4 weeks | Plugin interfaces, tenant config, onboarding flow |
| Phase 3: Deep refactoring | 4-6 weeks | Multi-tenancy, database isolation, testing |
| **Total** | **8-12 weeks** | Full platform with one vertical (field service) ready |

---

## 2. Architecture Map

### Module Dependency Diagram

```
                         ┌──────────────┐
                         │  settings.   │
                         │    toml      │
                         └──────┬───────┘
                                │ loaded by
                         ┌──────▼───────┐
                         │  core/config │
                         └──────┬───────┘
                                │ used by everything
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
  ┌──────▼──────┐       ┌──────▼──────┐       ┌──────▼──────┐
  │    core/    │       │ interfaces/ │       │   tools/    │
  │orchestrator │◄──────│  dashboard/ │       │             │
  │  (hub)      │       │  server.py  │       │             │
  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
         │                     │                     │
    ┌────┼────┐           ┌────┼────┐           ┌────┼────┐
    │    │    │           │    │    │           │    │    │
  model task memory    WS   REST  auth      doppler ros2 content
  router queue system  hub  API   gate      tools  tools tools
```

### Seam Analysis — Where Universal Meets Domain-Specific

| Seam Location | What's Universal | What's Domain-Specific | Coupling Level |
|---------------|------------------|------------------------|----------------|
| `core/persona.py` ← `config/persona.yaml` | Persona loading/injection logic | Cam's identity, tone, motorcycle knowledge | **Clean** — YAML swap |
| `core/orchestrator.py` ← `tools/doppler/*` | Orchestrator loop, tool dispatch | Tool implementations | **Clean** — tool registry |
| `core/conversation.py` ← persona | Conversation management | Persona system prompt content | **Clean** — config-driven |
| `core/memory/long_term.py` ← `CAM_BRAIN.md` | Vector store, seed loading | Seed file content | **Clean** — file swap |
| `tools/doppler/diagnostics.py` ← `config/diagnostic_trees/` | Diagnostic tree engine | YAML tree content | **Clean** — YAML swap |
| `interfaces/dashboard/server.py` ← all tools | Dashboard WS handlers | Tool-specific message types | **Medium** — new tools need new handlers |
| `interfaces/dashboard/static/index.html` ← all tools | Dashboard UI framework | Business-specific panels | **Medium** — panel architecture |

**Key Finding:** There is no tight coupling between universal and domain-specific code. The architecture uses configuration files, YAML data, and dependency injection throughout. The worst coupling is in `server.py` (2,800+ lines) which acts as a monolithic hub wiring everything together — this is the one file that needs the most extraction work.

---

## 3. Module-by-Module Breakdown

### 3.1 Core Infrastructure (`core/`)

#### core/orchestrator.py
```
Category: UNIVERSAL
Current Function: Main agent loop — observe/think/act/iterate. Manages the
    task queue, dispatches to model router, handles tool calls, manages context.
Platform Potential: Central orchestrator for any AI agent platform.
Dependencies: model_router, task, context_manager, persona, tool_registry,
    memory/*, event_logger, analytics
Extraction Complexity: Low
Notes: 6 Doppler references — all in docstrings/comments. The orchestrator
    itself is business-agnostic.
```

#### core/model_router.py
```
Category: UNIVERSAL
Current Function: Routes prompts to appropriate model (Ollama local, Claude API,
    Moonshot/Kimi, DeepSeek) based on task complexity tier. Tracks costs.
Platform Potential: Multi-model router with cost optimization — core platform value.
Dependencies: config, analytics (for cost tracking)
Extraction Complexity: Low
Notes: 3 Doppler references in docstring examples only. The routing logic,
    cost tracking, and fallback chains are entirely generic. The tier-based
    routing (simple→local, complex→cloud) is universally applicable.
```

#### core/config.py
```
Category: UNIVERSAL
Current Function: Loads settings.toml with environment variable overrides.
    Provides get() with dot-notation paths and defaults.
Platform Potential: Configuration loader — drop-in for any deployment.
Dependencies: None (stdlib only)
Extraction Complexity: Low
Notes: 1 reference ("CAM_BRAIN.md" as default). Clean, self-contained module.
```

#### core/agent_registry.py
```
Category: UNIVERSAL
Current Function: Tracks connected agents — registration, heartbeat monitoring,
    online/offline status, agent metadata.
Platform Potential: Agent/worker registry for any distributed system.
Dependencies: config (heartbeat settings)
Extraction Complexity: Low
Notes: Zero domain-specific references.
```

#### core/analytics.py
```
Category: UNIVERSAL
Current Function: SQLite-backed analytics — model usage counts, token consumption,
    cost tracking, daily/weekly/monthly aggregations.
Platform Potential: Usage analytics and cost dashboard for any AI platform.
Dependencies: config (db_path)
Extraction Complexity: Low
Notes: Zero domain-specific references. Strong platform value — every operator
    needs to track their AI spend.
```

#### core/backup.py
```
Category: UNIVERSAL
Current Function: Scheduled backup of databases and config files to timestamped
    tar.gz archives. Rotation of old backups.
Platform Potential: Backup system for any deployment.
Dependencies: config (backup settings)
Extraction Complexity: Low
Notes: Backs up CAM_BRAIN.md by name — this file path should be configurable.
```

#### core/commands.py
```
Category: CONFIGURABLE
Current Function: Defines available commands that agents can execute (status_report,
    system_info, restart_service, run_diagnostic, capture_sensor_data) with
    timeout and complexity settings.
Platform Potential: Command registry for agent task dispatch. Pattern is universal;
    specific commands are deployment-specific.
Dependencies: config (command timeouts)
Extraction Complexity: Low
Notes: Commands are already config-driven. New verticals add commands via config.
```

#### core/content_calendar.py
```
Category: UNIVERSAL
Current Function: SQLite-backed content pipeline — tracks content pieces through
    stages (idea → research → script → review → production → published).
Platform Potential: Content pipeline management for any content-creating business.
Dependencies: config (db_path)
Extraction Complexity: Low
Notes: 1 Doppler reference in docstring. The pipeline stages and status tracking
    are entirely generic. "Content calendar" is a universal business need.
```

#### core/context_manager.py
```
Category: UNIVERSAL
Current Function: Manages context window budget — tracks token usage across
    memory tiers, triggers rotation at configurable threshold (default 90%),
    generates session summaries on rotation.
Platform Potential: Context window management — critical for any LLM-based platform.
Dependencies: config (context limits per model), memory modules
Extraction Complexity: Low
Notes: Zero domain-specific references. The rotation strategy and per-model
    context limits are pure infrastructure.
```

#### core/conversation.py
```
Category: CONFIGURABLE
Current Function: Manages the conversation flow between George and Cam. Handles
    intent classification, response formatting, check-in frequency control,
    and tool event suppression during multi-step tasks.
Platform Potential: Conversation management layer between operator and AI assistant.
Dependencies: persona, model_router, memory, tool_registry
Extraction Complexity: Medium
Notes: 15 Doppler/George references. The conversation flow, intent classification,
    and check-in frequency control are universal. The persona-specific content
    (Cam's voice) is injected via persona.yaml. The main refactoring needed is
    replacing hardcoded "George" references with a configurable operator name.
```

#### core/deployer.py
```
Category: UNIVERSAL
Current Function: Generates and deploys agent configuration to remote machines
    via SSH. Creates connector scripts and systemd service files.
Platform Potential: Remote agent deployment system — critical for distributed platforms.
Dependencies: config (connector settings), agent_registry
Extraction Complexity: Low
Notes: The deployment targets are config-driven (IPs, usernames). The SSH-based
    deployment pattern is universal for edge computing / IoT scenarios.
```

#### core/event_logger.py
```
Category: UNIVERSAL
Current Function: In-memory ring buffer of system events with severity levels,
    categories, and dashboard broadcast support.
Platform Potential: Event logging backbone for any platform.
Dependencies: config (max_events)
Extraction Complexity: Low
Notes: Zero domain-specific references.
```

#### core/file_transfer.py
```
Category: UNIVERSAL
Current Function: Chunked file transfer between dashboard and agents over
    WebSocket. Base64 encoding, progress tracking, transfer history.
Platform Potential: File distribution system for distributed agent networks.
Dependencies: config (chunk_size, max_file_size)
Extraction Complexity: Low
Notes: Zero domain-specific references.
```

#### core/health_monitor.py
```
Category: UNIVERSAL
Current Function: Monitors agent health via heartbeat tracking, success rate
    calculation, and configurable thresholds.
Platform Potential: Health monitoring for any distributed system.
Dependencies: config (health thresholds), agent_registry
Extraction Complexity: Low
Notes: Zero domain-specific references.
```

#### core/message_bus.py
```
Category: UNIVERSAL
Current Function: In-process pub/sub message bus with per-channel ring buffers.
    Supports async subscribers and message filtering.
Platform Potential: Internal messaging backbone for any agent platform.
Dependencies: config (buffer capacity)
Extraction Complexity: Low
Notes: Zero domain-specific references.
```

#### core/notifications.py
```
Category: UNIVERSAL
Current Function: Alert/notification system — agent disconnect alerts, task
    failure alerts, kill switch alerts, high error rate alerts, cost threshold
    alerts. In-memory history with configurable toggles.
Platform Potential: Notification engine for any platform.
Dependencies: config (notification settings)
Extraction Complexity: Low
Notes: 1 Doppler reference in docstring. All notification types are generic
    (agent health, task status, cost tracking).
```

#### core/offline.py
```
Category: UNIVERSAL
Current Function: Graceful degradation when cloud APIs are unavailable —
    queues tasks, falls back to local models, resumes when connectivity returns.
Platform Potential: Offline resilience for any AI platform with local+cloud models.
Dependencies: config, model_router
Extraction Complexity: Low
Notes: 1 Doppler reference in docstring. The offline strategy is universally applicable.
```

#### core/performance.py
```
Category: UNIVERSAL
Current Function: Performance metrics collection — response times, throughput,
    resource usage (CPU, memory, disk), model latency tracking.
Platform Potential: Performance monitoring for any platform deployment.
Dependencies: config
Extraction Complexity: Low
Notes: 3 Doppler references in docstrings only.
```

#### core/persona.py
```
Category: CONFIGURABLE
Current Function: Loads persona.yaml and injects it as the system prompt.
    Manages persona attributes, tone, knowledge domains.
Platform Potential: Persona engine — every operator gets their own AI assistant identity.
    This is a key platform differentiator.
Dependencies: config (persona.yaml path)
Extraction Complexity: Low
Notes: 5 Doppler references. The loading/injection logic is generic. The persona
    content comes from YAML. For a platform, this module stays as-is; each tenant
    provides their own persona.yaml.
```

#### core/plugins.py
```
Category: UNIVERSAL
Current Function: Plugin architecture — discovers plugins from directory,
    loads manifests, initializes plugin instances, manages lifecycle.
Platform Potential: Plugin system for platform extensibility.
Dependencies: config (plugin directory)
Extraction Complexity: Low
Notes: 1 Doppler reference (author field in hello_world example). The plugin
    architecture is clean and ready for platform use.
```

#### core/reports.py
```
Category: CONFIGURABLE
Current Function: Generates daily/weekly/monthly summary reports. Aggregates
    analytics, task metrics, agent status, content pipeline status.
Platform Potential: Business reporting engine for any operator.
Dependencies: analytics, task, content_calendar, agent_registry
Extraction Complexity: Low
Notes: 3 Doppler references. The report structure (time-based aggregations,
    metric summaries, PDF generation) is universal. Some report sections
    reference business-specific modules that would vary by vertical.
```

#### core/research_store.py
```
Category: UNIVERSAL
Current Function: SQLite-backed storage for web research results — stores queries,
    URLs, extracted content, summaries. Supports search and retrieval.
Platform Potential: Research results database for any AI research workflow.
Dependencies: config (db_path)
Extraction Complexity: Low
Notes: 5 Doppler references in docstrings/examples. The storage schema and
    search logic are entirely generic.
```

#### core/scheduler.py
```
Category: UNIVERSAL
Current Function: Cron-like task scheduler with persistent state. Supports
    recurring tasks (daily, weekly, interval-based). JSON persistence.
Platform Potential: Task scheduling engine for any platform.
Dependencies: config (check_interval, persist_file)
Extraction Complexity: Low
Notes: Zero domain-specific references in the scheduling logic itself.
```

#### core/swarm.py
```
Category: UNIVERSAL
Current Function: Multi-agent swarm coordination — task distribution across
    connected agents, load balancing, capability matching.
Platform Potential: Swarm orchestration for distributed agent deployments.
Dependencies: agent_registry, task
Extraction Complexity: Low
Notes: Zero domain-specific references.
```

#### core/task_classifier.py
```
Category: UNIVERSAL
Current Function: Classifies incoming tasks by complexity tier (Tier 1/2/3)
    to route to appropriate model. Uses keyword matching and heuristics.
Platform Potential: Task complexity classification for any AI routing system.
Dependencies: config (tier definitions)
Extraction Complexity: Low
Notes: Zero domain-specific references in classification logic.
```

#### core/task.py
```
Category: UNIVERSAL
Current Function: Task queue with priority, status tracking, assignment to
    agents, completion handling, and persistence.
Platform Potential: Task management backbone for any platform.
Dependencies: config
Extraction Complexity: Low
Notes: 2 Doppler references in docstrings. Task data model is fully generic.
```

#### core/training.py
```
Category: CONFIGURABLE
Current Function: Training mode — interactive learning sessions where George
    teaches Cam domain knowledge. Stores learned facts in long-term memory.
Platform Potential: Operator knowledge onboarding system. Every operator needs
    to teach their AI assistant domain-specific knowledge.
Dependencies: memory/long_term, model_router
Extraction Complexity: Low
Notes: 8 Doppler references (examples use motorcycle knowledge). The training
    flow is generic — operator teaches AI, AI stores in vector memory.
```

#### core/webhook_manager.py
```
Category: UNIVERSAL
Current Function: Outbound webhooks (HMAC-SHA256 signed, retry with exponential
    backoff) and inbound webhooks (create tasks from external triggers).
    SQLite persistence.
Platform Potential: Webhook integration layer — critical for any platform.
Dependencies: config (webhook settings), task
Extraction Complexity: Low
Notes: Zero domain-specific references. The HMAC signing, retry logic, and
    event filtering are production-quality generic infrastructure.
```

### 3.2 Memory System (`core/memory/`)

**All 6 files are UNIVERSAL with Low extraction complexity.**

| File | Lines | Function | Domain Refs |
|------|-------|----------|-------------|
| `__init__.py` | 17 | Package exports | None |
| `short_term.py` | 290 | Session conversation buffer with summarization | Docstrings/tests only |
| `working.py` | 301 | JSON-backed persistent task state | Docstrings/tests only |
| `episodic.py` | 538 | SQLite conversation history with search | 1 code line ("You are CAM" in summarize prompt) |
| `long_term.py` | 636 | ChromaDB vector store with semantic search | Docstrings/tests only |
| `knowledge_ingest.py` | 552 | Document ingestion pipeline (MD/TXT/PDF/CSV) | Docstrings only |

**Key Finding:** The four-tier memory system (short-term, working, episodic, long-term) is the cleanest subsystem in the entire codebase. Zero domain logic in implementation code. All motorcycle references are confined to docstrings and `if __name__ == "__main__"` test blocks.

The only code-level fix needed: `episodic.py` line 344 hardcodes `"You are CAM"` in the summarize prompt — parameterize to `f"You are {persona_name}"`.

### 3.3 Tools — Content (`tools/content/`)

#### tools/content/tts_pipeline.py
```
Category: UNIVERSAL
Current Function: Piper TTS voice synthesis — text-to-speech with streaming
    sentence-by-sentence synthesis, voice model management, Python API integration.
Platform Potential: Voice synthesis engine for any AI assistant platform.
Dependencies: piper (optional), config (voice settings)
Extraction Complexity: Low
Notes: Zero domain-specific references. The streaming architecture (split
    sentences, synthesize each, stream over WebSocket) is generic.
```

#### tools/content/pipeline.py
```
Category: CONFIGURABLE
Current Function: Content production workflow — orchestrates the pipeline from
    idea through research, scripting, TTS, thumbnail, to scheduling.
Platform Potential: Content workflow engine. The pipeline stages are universal.
Dependencies: content_calendar, tts_pipeline, model_router
Extraction Complexity: Low
Notes: 3 Doppler references. Pipeline stages are generic. Specific content
    types (YouTube videos, social posts) would vary by operator.
```

#### tools/content/youtube.py
```
Category: CONFIGURABLE
Current Function: YouTube API integration — upload, metadata, thumbnails,
    playlist management. Currently stub/planned.
Platform Potential: Video platform publishing module. YouTube is universal.
Dependencies: None (stub)
Extraction Complexity: Low
Notes: 21 Doppler references (channel name, branding). The YouTube API
    integration pattern is generic; channel-specific config is data.
```

#### tools/content/social_media.py
```
Category: CONFIGURABLE
Current Function: Social media content formatting and scheduling —
    multi-platform post generation, thread formatting.
Platform Potential: Social media management module.
Dependencies: content_calendar
Extraction Complexity: Low
Notes: 12 Doppler references in templates/examples. The posting patterns
    (thread formatting, platform-specific length limits) are universal.
```

#### tools/content/video_processor.py
```
Category: CONFIGURABLE
Current Function: Video processing utilities — FFmpeg-based editing, thumbnail
    extraction, format conversion.
Platform Potential: Media processing module.
Dependencies: FFmpeg (external)
Extraction Complexity: Low
Notes: Zero domain-specific references in processing logic.
```

#### tools/content/highway20.py
```
Category: DOPPLER-ONLY
Current Function: Highway 20 documentary project management — route planning,
    stop research, episode outlining for George's cross-country trip.
Platform Potential: None as-is. The "project planning" pattern could generalize
    but the implementation is 100% specific to this one project.
Dependencies: model_router, research_store
Extraction Complexity: N/A — don't extract
Notes: This is George's competitive moat content. Keep in Doppler deployment only.
```

### 3.4 Tools — Business (`tools/doppler/`)

**All 16 business tool files follow the same pattern:** the code implements a universal business function (CRM, invoicing, appointments, etc.) with motorcycle-specific data models and examples embedded in docstrings, seed data, and field names. The architecture is consistently:

- SQLite-backed with clean schema
- Async-compatible with `on_change` callbacks for dashboard broadcast
- `to_broadcast_dict()` / `get_status()` for dashboard integration
- CRUD operations + search + reporting

| File | Lines | Current Function | Generalized Pattern | Motorcycle Refs | Effort |
|------|-------|-----------------|---------------------|-----------------|--------|
| `crm.py` | ~500 | Customer management (riders) | Client/Customer CRM | 3 | Low |
| `diagnostics.py` | ~600 | Diagnostic tree engine | Service workflow engine | 4 | Low |
| `email_templates.py` | ~400 | Email templates (appointment, invoice) | Communication templates | 13 | Medium |
| `feedback.py` | ~350 | Customer feedback/reviews | Client feedback system | 7 | Low |
| `finances.py` | ~550 | Income/expense tracking | Financial management | 2 | Low |
| `inventory.py` | ~450 | Parts/tools tracking | Inventory management | 3 | Low |
| `invoicing.py` | ~650 | Invoice generation (PDF) | Invoice/billing system | 3 | Low |
| `maintenance_scheduler.py` | ~400 | Scheduled maintenance reminders | Recurring service scheduler | 3 | Low |
| `market_analyzer.py` | ~500 | Market analysis (motorcycle market) | Market intelligence | 6 | Medium |
| `market_monitor.py` | ~600 | Craigslist/FB monitoring for bikes | Marketplace monitoring | 20 | Medium |
| `photo_docs.py` | ~350 | Before/after photo documentation | Job photo documentation | 4 | Low |
| `ride_log.py` | ~400 | Ride logging (motorcycle trips) | Asset usage tracking | 3 | Medium |
| `route_planner.py` | ~720 | Mobile service route optimization | Field service routing | 2 | Low |
| `scheduler_appointments.py` | ~600 | Appointment booking & management | Appointment scheduling | 2 | Low |
| `scout.py` | ~550 | Deal finder (Craigslist scraping) | Marketplace deal finder | 15 | Medium |
| `service_records.py` | ~500 | Service history tracking | Job/work order history | 12 | Low |
| `warranty.py` | ~350 | Warranty tracking | Warranty/guarantee tracking | 1 | Low |

**Key Finding:** The `tools/doppler/` directory is misnamed for a platform — it should be `tools/business/` or `tools/vertical/`. Every tool in it implements a pattern any field service business needs. The motorcycle-specific content is almost entirely in:

1. Example data in `if __name__ == "__main__"` blocks
2. Email template text in `email_templates.py`
3. Marketplace search terms in `scout.py` and `market_monitor.py`
4. Data model field names (e.g., "vehicle" instead of generic "asset")

### 3.5 Tools — ROS 2 (`tools/ros2/`)

```
Category: UNIVERSAL
All 5 files | ~1,200 lines total
Current Function: ROS 2 robot fleet integration — node management, message
    bridging, navigation goals, emergency stop, robot registry.
Platform Potential: Robot/IoT fleet management module. Directly applicable to
    any business with physical automation (warehouses, agriculture, security).
Dependencies: rclpy (optional, gracefully degraded)
Extraction Complexity: Low
Notes: 3 Doppler references in robot tool descriptions. The ROS 2 integration
    is entirely generic. Robot definitions come from config (settings.toml).
```

### 3.6 Interfaces

#### interfaces/dashboard/server.py
```
Category: CONFIGURABLE
Current Function: FastAPI WebSocket server — the central hub. Handles all WS
    message types, initializes all subsystems, manages agent connections.
Lines: ~2,800
Platform Potential: Platform dashboard backend. The WS message protocol,
    authentication, and subsystem initialization are universal.
Dependencies: Everything — this is the hub
Extraction Complexity: High
Notes: This is the single most complex file. It's a monolith that wires
    together every subsystem. For platform extraction, it needs:
    1. Subsystem initialization extracted to a registry/factory pattern
    2. WS message handlers registered dynamically (not if/elif chain)
    3. Business-specific handlers (scout, diagnostics, appointments) loaded
       as plugins rather than hardcoded
    Zero Doppler references in actual code — all business specifics come from
    the modules it imports.
```

#### interfaces/dashboard/static/index.html
```
Category: CONFIGURABLE
Current Function: 24,735-line single-page dashboard UI with WebSocket real-time
    updates. Includes: status panel, agent board, task queue, activity log,
    memory panel, content pipeline, cost tracker, kill switch, chat interface,
    voice input, TTS playback, and 15+ business module panels.
Platform Potential: Platform dashboard frontend. The panel architecture,
    WebSocket state management, and UI patterns are universal.
Dependencies: Vanilla JS (no framework), WebSocket to server.py
Extraction Complexity: High
Notes: ZERO Doppler/motorcycle references in the entire 24,735-line file.
    The dashboard is already fully generic — it renders whatever data the
    backend sends. Business-specific panels (scout, diagnostics, appointments)
    are rendered from WS messages without domain assumptions. The main
    extraction challenge is size — consider splitting into components.
```

#### interfaces/dashboard/static/login.html
```
Category: UNIVERSAL
Current Function: Login page with username/password authentication.
Platform Potential: Auth page — drop-in.
Extraction Complexity: Low
Notes: References "CAM Dashboard" in title — trivially parameterized.
```

#### interfaces/cli/terminal.py
```
Category: CONFIGURABLE
Current Function: Command-line interface for direct interaction with Cam.
Platform Potential: CLI interface for any AI assistant platform.
Dependencies: orchestrator, memory, persona
Extraction Complexity: Low
Notes: References "Cam" in prompts — parameterize to persona name.
```

#### interfaces/telegram/bot.py
```
Category: UNIVERSAL
Current Function: Telegram bot integration — receives messages, dispatches
    to orchestrator, returns responses. Chat ID allowlist for security.
Platform Potential: Telegram integration module for any platform.
Dependencies: orchestrator, config
Extraction Complexity: Low
Notes: Zero domain-specific references.
```

#### interfaces/api/routes.py
```
Category: UNIVERSAL
Current Function: REST API endpoints — task CRUD, agent status, system info,
    analytics, webhook management.
Platform Potential: Platform REST API.
Dependencies: Multiple core modules
Extraction Complexity: Low
Notes: Zero domain-specific references. API is resource-based and generic.
```

### 3.7 Security

**All 4 security files are UNIVERSAL with Low extraction complexity.**

| File | Function | Notes |
|------|----------|-------|
| `audit.py` | SQLite audit trail — every action logged | Zero domain refs |
| `auth.py` | Session-based authentication with lockout | Zero domain refs |
| `permissions.py` | Action classification (safe/logged/gated/blocked) | Zero domain refs |
| `set_password.py` | CLI password setup utility | Zero domain refs |

### 3.8 Agents

#### agents/connector.py
```
Category: UNIVERSAL
Current Function: Agent-side WebSocket connector — connects to dashboard,
    handles heartbeat, executes dispatched commands, reports results.
Platform Potential: Agent runtime — deployed on edge devices.
Dependencies: config (connector settings)
Extraction Complexity: Low
Notes: Zero domain-specific references.
```

#### agents/business_agent.py, content_agent.py, research_agent.py
```
Category: CONFIGURABLE
Current Function: Specialized agent implementations with domain-specific
    tool sets and system prompts.
Platform Potential: Agent templates — configurable per vertical.
Dependencies: connector, tool_registry
Extraction Complexity: Medium
Notes: System prompts contain Doppler-specific content. The agent framework
    (connect, receive task, execute tools, return result) is universal.
```

### 3.9 Configuration Files

| File | Category | Extraction | Notes |
|------|----------|------------|-------|
| `config/settings.toml` | CONFIGURABLE | Medium | ~10 Doppler-specific values (IPs, coords, email, scout config) out of ~320 lines. Structure is universal. |
| `config/persona.yaml` | CONFIGURABLE | Low | Content is Doppler; schema is the platform. |
| `config/prompts/model_router.md` | UNIVERSAL | Low | 2 motorcycle examples in an otherwise generic routing prompt. |
| `config/diagnostic_trees/*.yaml` (6) | CONFIGURABLE schema / DOPPLER-ONLY content | Low (schema) | The YAML decision tree schema is industry-agnostic gold. |

### 3.10 Content & Documentation

| File | Category | Notes |
|------|----------|-------|
| `CAM_BRAIN.md` | DOPPLER-ONLY | Seed knowledge — the pattern is universal, content is Doppler IP |
| `content/*.md` (7 files) | DOPPLER-ONLY | George's content strategy and research |
| `docs/cam-commands.md` | CONFIGURABLE | Operations quick-reference — needs IP/name updates |
| `docs/cam-operator-guide.md` | CONFIGURABLE | ~80% universal, ~20% Doppler examples |
| `docs/cam-system-reference.md` | CONFIGURABLE | Technical reference — mostly universal architecture docs |
| `docs/known-issues.md` | UNIVERSAL | Technical debt tracker |
| `CLAUDE.md` | CONFIGURABLE | ~70% universal architecture, ~30% Doppler identity |

### 3.11 Other Files

| File | Category | Notes |
|------|----------|-------|
| `cam_launcher.py` | UNIVERSAL | Entry point |
| `start.sh` / `stop.sh` | UNIVERSAL | Process management scripts |
| `deploy/agent_config.py` | UNIVERSAL | Agent deployment config generator |
| `deploy/install_agent.sh` | UNIVERSAL | Agent installation script |
| `requirements.txt` | UNIVERSAL | Python dependencies |
| `tests/*.py` (4 files) | CONFIGURABLE | Test suites with Doppler-specific validation checks |
| `plugins/hello_world/` | UNIVERSAL | Plugin architecture example |
| `api/export.py` | UNIVERSAL | Data export utilities |

---

## 4. Refactoring Roadmap

### Phase 1: Quick Wins (1-2 weeks, Low effort, High impact)

| # | Task | Impact | Effort | Risk |
|---|------|--------|--------|------|
| 1.1 | Rename `tools/doppler/` → `tools/business/` | Removes brand-specific naming | 2 hours | Low — find/replace imports |
| 1.2 | Parameterize "CAM" / "Cam" in system prompts | Generic persona naming | 4 hours | Low |
| 1.3 | Replace hardcoded "George" with `config.operator_name` | Operator-agnostic | 2 hours | Low |
| 1.4 | Move `CAM_BRAIN.md` content to `config/seed_knowledge.md` | Clearer separation | 1 hour | Low |
| 1.5 | Extract Doppler values from `settings.toml` into a `verticals/motorcycle_repair.toml` overlay | Clean default config | 4 hours | Low |
| 1.6 | Strip motorcycle examples from docstrings/test blocks | Clean codebase | 4 hours | Low |
| 1.7 | Add `[platform]` section to settings.toml: `name`, `operator_name`, `business_type` | Tenant identity | 2 hours | Low |
| 1.8 | Remove `logging.basicConfig()` calls from memory modules | Library-safe imports | 1 hour | Low |

### Phase 2: Core Extraction (3-4 weeks, Medium effort, Critical path)

| # | Task | Impact | Effort | Risk |
|---|------|--------|--------|------|
| 2.1 | Extract `server.py` into modular handler registry | Plugin-based WS handlers | 1 week | Medium — large file, many touchpoints |
| 2.2 | Create business module plugin interface | Verticals as installable modules | 1 week | Medium |
| 2.3 | Build operator onboarding flow | Generate persona.yaml + seed_knowledge.md from questionnaire | 3 days | Low |
| 2.4 | Create config template/overlay system | Base config + vertical overlay + operator overrides | 3 days | Low |
| 2.5 | Split `index.html` into component architecture | Maintainable frontend | 1 week | Medium — 24K lines of vanilla JS |
| 2.6 | Create vertical schema for diagnostic trees | Industry-agnostic tree format with validation | 2 days | Low |
| 2.7 | Abstract email templates into configurable template engine | Per-vertical email content | 2 days | Low |

### Phase 3: Deep Refactoring (4-6 weeks, High effort, Future value)

| # | Task | Impact | Effort | Risk |
|---|------|--------|--------|------|
| 3.1 | Multi-tenant database isolation | Multiple operators on one instance | 2 weeks | High — 37 SQLite databases |
| 3.2 | Agent authentication (shared secrets) | Security for production deployment | 3 days | Medium |
| 3.3 | Platform installer / setup wizard | One-command deployment | 1 week | Medium |
| 3.4 | Vertical marketplace (discover/install industry modules) | Platform ecosystem | 2 weeks | Medium |
| 3.5 | Migration system for SQLite schemas | Safe upgrades across versions | 1 week | Medium |
| 3.6 | Comprehensive test suite with vertical-agnostic fixtures | CI/CD readiness | 1 week | Low |

---

## 5. Interface Design Recommendations

### Business Module Plugin Interface

Each industry vertical provides a module that implements:

```python
class BusinessModule:
    """Base class for industry-specific business modules."""

    # Module metadata
    name: str                          # "motorcycle_repair", "hvac_service", etc.
    display_name: str                  # "Motorcycle Repair & Diagnostics"
    version: str                       # "1.0.0"

    # What this module provides
    tools: list[ToolDefinition]        # Tools registered with the orchestrator
    dashboard_panels: list[PanelDef]   # Dashboard UI panels
    ws_handlers: dict[str, Handler]    # WebSocket message handlers
    scheduled_tasks: list[TaskDef]     # Recurring tasks (e.g., daily reports)

    # Configuration
    config_schema: dict                # JSON Schema for module-specific settings
    seed_knowledge: str                # Path to seed knowledge markdown
    diagnostic_trees: list[str]        # Paths to diagnostic tree YAMLs

    # Lifecycle
    async def initialize(self, config, dependencies) -> None
    async def shutdown(self) -> None
    def get_status(self) -> dict
    def to_broadcast_dict(self) -> dict
```

### What a New Vertical Provides

To add a new industry (e.g., HVAC repair):

```
verticals/hvac_repair/
├── manifest.yaml              # Module metadata
├── config.toml                # Default settings overlay
├── persona.yaml               # AI assistant persona for HVAC
├── seed_knowledge.md          # Domain knowledge (HVAC systems, codes, etc.)
├── diagnostic_trees/
│   ├── furnace_no_heat.yaml   # Troubleshooting trees
│   ├── ac_not_cooling.yaml
│   └── thermostat_issues.yaml
├── email_templates/
│   ├── appointment_confirm.md
│   └── invoice.md
├── tools/
│   ├── __init__.py
│   ├── hvac_codes.py          # EPA refrigerant codes, NFPA references
│   └── equipment_tracker.py   # Equipment-specific tracking
└── dashboard_panels/
    └── service_map.html       # HVAC-specific dashboard panel
```

### Domain Knowledge Injection Points

| Injection Point | Format | How It Works |
|-----------------|--------|--------------|
| Persona | `persona.yaml` | Defines AI assistant identity, tone, knowledge domains |
| Seed Knowledge | `seed_knowledge.md` | Loaded into ChromaDB on first startup |
| Diagnostic Trees | `*.yaml` | Structured troubleshooting workflows |
| Email Templates | `*.md` (Jinja) | Customer-facing communications |
| Tool Configs | `config.toml` overlay | Business-specific settings (rates, areas, etc.) |
| Dashboard Panels | HTML/JS components | Industry-specific UI elements |

---

## 6. Risk Register

### Extraction Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| **server.py monolith** — Breaking changes during handler extraction | High | Medium | Incremental extraction with integration tests; keep backward compat |
| **37 SQLite databases** — Schema coupling between modules | Medium | Medium | Audit cross-database queries; consider consolidation or clear isolation |
| **Dashboard 24K-line HTML** — Regressions during componentization | Medium | Medium | Visual regression testing; extract one panel at a time |
| **Hidden persona assumptions** — Code that assumes "Cam" identity | Low | Low | Grep audit shows only 67 refs in core; most are docstrings |
| **Agent auth gap** — Agents connect with ID only, no secrets | High | High (in production) | Must be resolved before any multi-tenant deployment |
| **No database migrations** — Schema changes require manual intervention | Medium | High (over time) | Add Alembic or custom migration system before platform launch |
| **BusinessAgent duplicate data** — Two data stores for same data | Low | Medium | Resolve before multi-tenancy; single source of truth |

### Hidden Domain Assumptions

Found during audit — places where "universal" code secretly assumes motorcycle context:

| Location | Assumption | Fix |
|----------|------------|-----|
| `episodic.py:344` | Hardcoded "You are CAM" in summarize prompt | Parameterize from persona config |
| `core/conversation.py` | 15 references to "George" as operator name | Replace with `config.operator_name` |
| `core/training.py` | Training examples use motorcycle knowledge | Replace with generic examples |
| `email_templates.py` | Templates mention "motorcycle," "Doppler Cycles" | Move templates to vertical config |
| `scout.py` | Motorcycle makes list hardcoded in settings.toml | Already in config — just needs vertical overlay |
| `backup.py:45` | Backs up `CAM_BRAIN.md` by filename | Use config for seed knowledge path |

### Testing Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| No unit test suite (only integration/validation tests) | Can't verify extraction doesn't break things | High |
| No API contract tests | REST API changes could break clients | Medium |
| No WebSocket protocol tests | WS message format changes invisible | Medium |
| No performance benchmarks | Can't detect regressions | Low |
| v1_validation.py has Doppler-specific checks | Tests need vertical-agnostic mode | Medium |

---

## 7. Platform Config Schema (Draft)

```yaml
# Platform tenant configuration — one per operator deployment

platform:
  name: "SoloOps"                   # or whatever the platform name becomes
  version: "1.0.0"

operator:
  name: "George"                    # Operator's name
  business_name: "Doppler Cycles"   # Business display name
  business_type: "mobile_mechanic"  # Vertical identifier
  timezone: "America/Los_Angeles"
  location:
    address: "Gresham, OR"
    lat: 45.4976
    lon: -122.4302

persona:
  name: "Cam"                       # AI assistant name
  tagline: "Like the camshaft, not the social media thing."
  role: "Research desk to your hands-on expertise"
  tone:
    - knowledgeable
    - conversational
    - professional
  transparency: true                # Always disclose AI nature

services:
  - name: "Mobile Diagnostics"
    duration_default: 60            # minutes
    rate: 75.00                     # hourly
  - name: "Electrical Repair"
    duration_default: 90
    rate: 75.00

service_area:
  center_lat: 45.4976
  center_lon: -122.4302
  radius_miles: 50
  avg_speed_mph: 30.0
  road_factor: 1.4

scheduling:
  business_hours:
    start: "08:00"
    end: "18:00"
  days: ["Mon", "Tue", "Wed", "Thu", "Fri"]
  buffer_minutes: 30               # Between appointments

pricing:
  model: "hourly"                   # or "flat_rate", "per_job"
  default_rate: 75.00
  currency: "USD"
  invoice_prefix: "DC"
  payment_terms_days: 30

domain_knowledge:
  seed_file: "seed_knowledge.md"    # Initial knowledge base
  diagnostic_trees_dir: "diagnostic_trees/"
  # Platform will load all *.yaml from this directory

integrations:
  youtube:
    enabled: false
    channel_id: ""
  telegram:
    enabled: false
    bot_token: ""
  email:
    enabled: false
    from_address: ""
  twilio:
    enabled: false
    phone_number: ""

models:
  # Operator can customize model routing
  simple: "mistral:7b"             # Local, fast, free
  complex: "claude-sonnet-4-6"     # Cloud, capable
  boss: "claude-opus-4-6"          # Operator conversation — best available

content:
  enabled: true
  platforms: ["youtube", "x"]
  calendar_db: "data/content_calendar.db"

marketplace_monitor:
  enabled: true
  keywords: []                     # Vertical-specific search terms
  exclude: ["parts only"]
  price_range: [0, 5000]
  location: "Portland OR"
  radius_miles: 50

robots:
  enabled: false
  # ROS 2 robot definitions go here if applicable
```

---

## 8. Conclusion

CAM is unusually well-positioned for platform extraction. The key factors:

1. **Clean architecture** — Domain knowledge lives in data files, not code
2. **Consistent patterns** — Every business tool follows the same SQLite + async + broadcast pattern
3. **Configuration-driven** — Settings, persona, diagnostic trees are all external files
4. **Dashboard is generic** — 24,735 lines of UI with zero business-specific hardcoding
5. **Memory system is spotless** — Entire four-tier memory stack is domain-agnostic

The biggest blockers are operational, not architectural:
- `server.py` monolith needs modularization
- Agent authentication needs implementation
- Database migration system needs creation
- Multi-tenancy needs design decisions (isolated instances vs shared infrastructure)

The motorcycle repair vertical becomes the first "industry module" that proves the plugin architecture. George retains his competitive moat (diagnostic trees, persona, content strategy, CAM_BRAIN.md knowledge) while the platform serves other solopreneurs.

**Bottom line:** The codebase was built with good engineering instincts that accidentally (or intentionally) created a platform-ready architecture. The extraction is more "organize and formalize" than "rewrite."
