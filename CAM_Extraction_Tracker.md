# CAM Platform Extraction Tracker

**Created:** 2026-03-02
**Status:** Phase 1 not started
**Goal:** Extract CAM into a general-purpose AI operations platform for single-operator businesses

---

## How to Use This Document

- Check off items as you complete them: `- [ ]` → `- [x]`
- Each item has an estimated effort in parentheses
- Items are ordered by dependency — work top to bottom within each phase
- After completing a phase, update the status at the top of this document

---

## Phase 1: Quick Wins (1-2 weeks)

Low effort, high impact changes that immediately decouple Doppler-specific content.

### 1.1 Structural Renames
- [ ] Rename `tools/doppler/` → `tools/business/` (2h)
  - Update all imports across codebase (`from tools.doppler.` → `from tools.business.`)
  - Update `__init__.py` exports
  - Update `CLAUDE.md` directory structure documentation
  - Update `docs/cam-system-reference.md`
- [ ] Rename `CAM_BRAIN.md` → `config/seed_knowledge.md` (1h)
  - Update reference in `config/settings.toml` (`long_term_seed_file`)
  - Update reference in `core/config.py` default
  - Update references in `server.py` and `terminal.py`
  - Update `core/backup.py` backup file list

### 1.2 Parameterize Identity
- [ ] Add `[platform]` section to `settings.toml` (30m)
  ```toml
  [platform]
  name = "CAM"
  operator_name = "George"
  business_name = "Doppler Cycles"
  business_type = "mobile_mechanic"
  ```
- [ ] Replace hardcoded "George" in `core/conversation.py` with `config.get('platform.operator_name')` (1h)
- [ ] Replace "You are CAM" in `core/memory/episodic.py:344` with persona-derived name (30m)
- [ ] Replace "CAM" references in `interfaces/cli/terminal.py` prompts with config value (30m)
- [ ] Replace "CAM Dashboard" in `login.html` title with config value (15m)

### 1.3 Clean Docstrings and Test Data
- [ ] Strip motorcycle examples from `core/memory/short_term.py` docstrings and `__main__` block (30m)
- [ ] Strip motorcycle examples from `core/memory/working.py` docstrings and `__main__` block (30m)
- [ ] Strip motorcycle examples from `core/memory/episodic.py` docstrings and `__main__` block (30m)
- [ ] Strip motorcycle examples from `core/memory/long_term.py` docstrings and `__main__` block (30m)
- [ ] Strip motorcycle references from `core/memory/knowledge_ingest.py` docstring (15m)
- [ ] Replace motorcycle examples in `config/prompts/model_router.md` with generic examples (15m)

### 1.4 Remove Module-Level Side Effects
- [ ] Remove `logging.basicConfig()` from `core/memory/short_term.py` (5m)
- [ ] Remove `logging.basicConfig()` from `core/memory/working.py` (5m)
- [ ] Remove `logging.basicConfig()` from `core/memory/episodic.py` (5m)
- [ ] Remove `logging.basicConfig()` from `core/memory/long_term.py` (5m)

### 1.5 Configuration Separation
- [ ] Create `verticals/motorcycle_repair/` directory structure (30m)
  ```
  verticals/motorcycle_repair/
  ├── config_overlay.toml    # Doppler-specific settings
  ├── persona.yaml           # Cam persona (copy from config/)
  ├── seed_knowledge.md      # CAM_BRAIN.md content (move)
  ├── diagnostic_trees/      # Move from config/diagnostic_trees/
  └── email_templates/       # Extract from tools/business/email_templates.py
  ```
- [ ] Move Doppler-specific values out of `settings.toml` into `config_overlay.toml` (2h)
  - `[scout]` section (motorcycle makes, Portland location)
  - `[business]` section (invoice prefix "DC", labor rate)
  - `[appointments]` section (home coordinates)
  - `[email]` section (dopplercycles.com address)
  - `[auth]` section (username "george")
- [ ] Implement config overlay loading in `core/config.py` (2h)
  - Base config (`settings.toml`) + vertical overlay + env vars

---

## Phase 2: Core Extraction (3-4 weeks)

Medium effort changes that create the plugin/vertical architecture.

### 2.1 Server Modularization
- [ ] Extract WS message handlers from `server.py` into handler modules (1w)
  - Create `interfaces/dashboard/handlers/` directory
  - Move agent handlers → `handlers/agents.py`
  - Move chat handlers → `handlers/chat.py`
  - Move task handlers → `handlers/tasks.py`
  - Move tool-specific handlers → `handlers/tools.py`
  - Move business handlers → `handlers/business.py`
  - Create handler registry that dynamically loads handlers
- [ ] Extract subsystem initialization from `server.py` lifespan function (2d)
  - Create `core/bootstrap.py` — initializes all subsystems based on config
  - `server.py` calls bootstrap, receives initialized subsystems
  - Business modules registered via bootstrap, not hardcoded imports

### 2.2 Business Module Interface
- [ ] Define `BusinessModule` base class in `tools/business/base.py` (1d)
  ```python
  class BusinessModule:
      name: str
      tools: list[ToolDefinition]
      dashboard_panels: list[PanelDef]
      ws_handlers: dict[str, Handler]
      async def initialize(self, config, deps) -> None
      async def shutdown(self) -> None
      def get_status(self) -> dict
      def to_broadcast_dict(self) -> dict
  ```
- [ ] Refactor each `tools/business/*.py` to implement `BusinessModule` interface (3d)
- [ ] Create module discovery/loading in bootstrap (1d)
  - Scan `verticals/{business_type}/tools/` for additional modules
  - Register tools with orchestrator's tool_registry
  - Register WS handlers with dashboard handler registry
  - Register dashboard panels with frontend

### 2.3 Operator Onboarding
- [ ] Create `scripts/setup_operator.py` — interactive questionnaire (2d)
  - Asks: business name, operator name, business type, location, services, rates
  - Generates: `persona.yaml`, `seed_knowledge.md`, `config_overlay.toml`
  - Seeds long-term memory from generated knowledge file
- [ ] Create persona.yaml template with placeholder tokens (1d)
- [ ] Create seed_knowledge.md template (1d)

### 2.4 Dashboard Componentization
- [ ] Identify independent panel sections in `index.html` (1d)
- [ ] Extract each panel into a loadable component (1w)
  - Status panel, agent board, task queue, activity log → core panels (always loaded)
  - Scout, diagnostics, appointments, inventory → business panels (loaded by module)
  - Each panel: HTML template + init function + WS handler registration
- [ ] Create panel loader that reads active modules and injects their panels (1d)

### 2.5 Template Engine for Communications
- [ ] Move email template content from `email_templates.py` to Jinja2 templates (1d)
- [ ] Create template directory per vertical: `verticals/{type}/templates/` (30m)
- [ ] Update email_templates.py to load templates from vertical directory (1d)

---

## Phase 3: Deep Refactoring (4-6 weeks)

High effort changes for production multi-tenancy and ecosystem readiness.

### 3.1 Database Architecture
- [ ] Audit all 37 SQLite databases for cross-references (1d)
- [ ] Decide: isolated instances vs. shared infrastructure with tenant isolation (decision)
- [ ] Implement database migration system (1w)
  - Schema versioning per database
  - Upgrade/downgrade scripts
  - Migration runner in bootstrap
- [ ] Resolve BusinessAgent duplicate data store issue (2d)

### 3.2 Security Hardening
- [ ] Implement agent authentication with shared secrets (2d)
  - Agent provides HMAC-signed handshake on WebSocket connect
  - Dashboard validates against per-agent secrets in config
- [ ] Add API key management for multi-tenant REST API access (1d)
- [ ] Review and strengthen directory jail for multi-tenant deployments (1d)
- [ ] Add rate limiting to REST API and WebSocket connections (1d)

### 3.3 Platform Packaging
- [ ] Create `setup.py` / `pyproject.toml` for pip-installable package (1d)
- [ ] Create Docker container with platform + vertical mount point (2d)
- [ ] Create `soloops init` CLI command that runs onboarding flow (1d)
- [ ] Create `soloops start` / `soloops stop` wrapper scripts (30m)

### 3.4 Vertical Marketplace (Future)
- [ ] Define vertical package format (manifest, required files, validation) (2d)
- [ ] Create vertical installer: `soloops install-vertical hvac_repair` (1w)
- [ ] Create example second vertical (HVAC repair) to validate architecture (1w)
  - Prove the plugin interface works with a non-motorcycle vertical
  - Create HVAC diagnostic trees, persona, seed knowledge
  - Verify dashboard renders HVAC-specific panels correctly

### 3.5 Testing Infrastructure
- [ ] Create unit test suite for core modules (3d)
- [ ] Create integration test suite with generic fixtures (no motorcycle data) (2d)
- [ ] Create API contract tests for REST endpoints (1d)
- [ ] Create WebSocket protocol tests (1d)
- [ ] Add CI pipeline configuration (GitHub Actions) (1d)

---

## Verification Checkpoints

After each phase, verify:

### Phase 1 Checkpoint
- [ ] `grep -r "doppler\|Doppler\|motorcycle\|Harley" core/` returns zero results (excluding comments)
- [ ] CAM still starts and functions normally after all renames
- [ ] Dashboard loads correctly
- [ ] Agents connect and respond to commands
- [ ] Settings load correctly with config overlay system

### Phase 2 Checkpoint
- [ ] A new vertical can be added by dropping files in `verticals/` — no code changes
- [ ] `setup_operator.py` produces working config from questionnaire
- [ ] Dashboard panels load dynamically based on active vertical
- [ ] WS handlers register dynamically — no hardcoded if/elif chain
- [ ] server.py is under 500 lines

### Phase 3 Checkpoint
- [ ] Second vertical (HVAC) works end-to-end without any motorcycle code
- [ ] Agent authentication prevents unauthorized connections
- [ ] Database migrations run cleanly on fresh and existing installations
- [ ] Docker deployment works with `docker run -v verticals/:/app/verticals/`
- [ ] All tests pass with both motorcycle and HVAC vertical fixtures

---

## Decision Log

Track key decisions made during extraction:

| Date | Decision | Rationale | Decided By |
|------|----------|-----------|------------|
| 2026-03-02 | Audit completed | Assess platform extraction feasibility | George |
| | | | |

---

## Notes

- **Don't break Doppler Cycles.** Every extraction step must keep the motorcycle repair vertical fully functional. Extract *from* a working system, never *into* a broken one.
- **One vertical proves the pattern.** The HVAC vertical in Phase 3 is the litmus test. If it works without touching core code, the architecture is right.
- **Frugality principle applies.** Don't build marketplace infrastructure until there's a second customer. Phase 3.4 is optional until demand exists.
- **George is learning software.** Keep the extraction approachable. Comment the new abstractions. Don't add complexity that doesn't earn its keep.
