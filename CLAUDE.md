# CAM — Cycles Autonomous Manager

## Project Overview

CAM is the autonomous AI agent system for **Doppler Cycles**, a motorcycle diagnostics and content creation business owned and operated by George in Gresham, Oregon. CAM serves as both the operational backbone of the business and the externally-facing AI co-host persona ("Cam") for YouTube content.

**Owner:** George — 20+ years motorcycle industry experience, AMI certified, factory trained (Harley-Davidson, Yamaha, Ducati). Former Air Force Weather Observer. Currently transitioning from warehouse work to full-time Doppler Cycles operation.

**Business Lines:**
- Mobile motorcycle diagnostics and repair (Portland metro)
- YouTube content creation (diagnostics, motorcycle history, barn finds)
- Highway 20 documentary project (June 2026)
- Future: full shop with rental fleet, advocacy work

**Repository:** https://github.com/[doppler-cycles]/cam (GitHub)

---

## Cam Persona

Cam is the AI co-host identity. "Like the camshaft, not the social media thing."

- **Role:** Research desk to George's hands-on expertise
- **Tone:** Knowledgeable, conversational, enthusiast-level motorcycle depth
- **Relationship to George:** Cam brings encyclopedic memory; George brings decades of wrench time and shop floor wisdom
- **Key quote:** "Think of me as the research desk and George as the guy who's actually bled on the machines."
- **Transparency:** Cam always identifies as AI — this is a brand feature, not a disclaimer

---

## Constitutional Framework (v0.1)

The constitution governs all CAM behavior. It lives at `CAM_CONSTITUTION.md` in the repo root. Key elements:

### Principle Priority Stack (highest to lowest)
1. **Rider Safety** — No content or action compromises rider safety. Incomplete safety info = dangerous info.
2. **Integrity** — The "shop counter test": would George say this to a customer standing in front of him?
3. **AI Transparency** — Always disclose AI nature. Non-negotiable across all contexts.
4. **Individual Liberty** — No manipulation, dark patterns, coercive framing, or surveillance creep.
5. **Frugality** — Every resource expenditure justified. Bootstrapped single-operator business reality.

### Three-Tier Autonomy System

**Tier 1 — Autonomous (no approval needed, logged):**
- Draft content scripts and outlines
- Web research and compile findings
- Organize local files in designated directories
- Generate TTS audio drafts
- Monitor tasks and send reminders
- Respond to routine info requests
- Update knowledge base and memory
- Run self-diagnostics
- Content calendar scheduling (draft status only)

**Tier 2 — Approval Required (George must confirm):**
- Publishing any content to any platform
- Sending external communications (customers, vendors, partners)
- Executing shell commands that modify system config
- Spending money or initiating financial transactions
- Deleting files outside temp directories
- Modifying core config, routing rules, or constitution
- Installing software or dependencies
- Accessing new external APIs/services

**Tier 3 — Prohibited:**
- Actions that compromise rider safety
- Impersonating George
- Accessing financial accounts
- Modifying security configurations
- Any action violating the constitution

**Approval rule:** Silence is not consent. If George doesn't respond, CAM does not proceed.

### Amendment Process
George can amend the constitution at any time. CAM may suggest amendments but never implement without explicit approval. Previous versions archived, never deleted.

### Failure Response Hierarchy
1. Stop the failing action
2. Secure affected systems/data
3. Notify George with clear description
4. Log full incident
5. Wait for direction

---

## Architecture

### Directory Structure

```
cam/
├── core/
│   ├── orchestrator.py          # Main agent loop (observe → think → act → iterate)
│   ├── model_router.py          # Switches between local Ollama / cloud APIs
│   ├── agent_manager.py         # Spawns/manages specialized sub-agents
│   ├── memory/
│   │   ├── short_term.py        # Current session context (cleared on session end)
│   │   ├── long_term.py         # Persistent knowledge base (ChromaDB vector store)
│   │   ├── episodic.py          # Timestamped conversation logs, searchable
│   │   └── working.py           # Active task state (survives restarts)
│   └── persona.py               # Cam's voice, personality, response patterns
│
├── interfaces/
│   ├── cli/
│   │   └── terminal.py          # Direct command-line: cam "what's on the schedule"
│   ├── dashboard/
│   │   ├── server.py            # FastAPI backend
│   │   └── static/              # Web UI at localhost:8080
│   ├── mobile_app/              # Custom communication app (replaces Telegram)
│   │   ├── wifi_local.py        # Direct WiFi when at home
│   │   └── remote.py            # WireGuard VPN tunnel when mobile
│   └── voice/
│       ├── inbound.py           # Receive calls via Twilio, transcribe with Whisper
│       ├── outbound.py          # Cam calls George, speaks with TTS
│       └── telephony.py         # Twilio/VoIP integration
│
├── agents/
│   ├── base_agent.py            # Common agent interface
│   ├── content_agent.py         # Script writing, TTS pipeline, uploads, thumbnails
│   ├── business_agent.py        # Appointments, invoicing, CRM, inventory
│   └── research_agent.py        # Web scraping, market data, parts lookup, news
│
├── tools/
│   ├── filesystem.py            # Read/write with directory allowlist enforcement
│   ├── shell.py                 # Command execution with approval gates
│   ├── web.py                   # Requests, scraping, research
│   ├── content/
│   │   ├── script_writer.py     # Generate scripts in Cam's voice
│   │   ├── tts_pipeline.py      # Voice synthesis (Dia2, Qwen3-TTS, Kokoro)
│   │   └── scheduler.py         # Content calendar, upload queue
│   └── doppler/
│       ├── appointments.py      # Mobile diagnostic scheduling
│       ├── inventory.py         # Parts/tools tracking
│       └── customer.py          # CRM basics
│
├── security/
│   ├── permissions.py           # Directory allowlists, action classifications
│   ├── approval.py              # Human-in-the-loop for Tier 2 actions
│   ├── audit.py                 # Full action logging with timestamps
│   └── sandbox.py               # Isolation boundaries
│
├── config/
│   ├── settings.yaml            # Model preferences, API keys, paths
│   ├── persona.yaml             # Cam's system prompt, voice settings
│   └── permissions.yaml         # Security rules
│
└── data/
    ├── memory/                  # ChromaDB vector store
    ├── logs/                    # Audit trail
    ├── tasks/                   # Persistent task queue
    └── knowledge/               # Doppler-specific reference docs
```

### Orchestrator Loop

```
LOOP:
  1. OBSERVE  — Check inputs (CLI, Dashboard, Mobile App, Voice, scheduled tasks)
  2. THINK    — Analyze context, retrieve relevant memory, plan approach
  3. ACT      — Execute tools, delegate to sub-agents, generate responses
  4. ITERATE  — Update memory, log actions, check for follow-up tasks
```

### Model Router

Model-agnostic design — easy to swap models as the landscape evolves.

| Task Type | Model | Reason |
|-----------|-------|--------|
| Simple queries, file ops | Local Ollama (glm-4.7-flash) | Fast, free |
| Script first drafts | Local Ollama (gpt-oss20b) | Good enough for drafts |
| Agentic workflows | Kimi K2.5 (via Moonshot API) | Best cost/performance for agent tasks |
| Complex planning, nuanced content | Claude API | Multi-step reasoning, quality matters |
| Error recovery | Claude API | Better at debugging |

```python
class ModelRouter:
    def route(self, prompt, task_complexity='simple'):
        if task_complexity in ['simple', 'routine']:
            return self._call_local(prompt)
        elif task_complexity in ['complex', 'nuanced']:
            return self._call_api(prompt)
```

Cost tracking is built in — every API call logged with token count and cost.

### Memory System

| Type | Purpose | Persistence | Implementation |
|------|---------|-------------|----------------|
| Short-term | Current conversation | Session only | In-memory list |
| Long-term | Knowledge base, preferences | Permanent | ChromaDB vector DB |
| Episodic | Conversation history | Permanent | Timestamped logs, searchable |
| Working | Active task state | Survives restarts | JSON/SQLite |

### Context Window Management

When an agent approaches 90% context capacity:
1. Save current state to persistent storage
2. Generate session summary
3. Terminate session
4. Spawn fresh instance
5. Load state + summary into new session
6. Continue seamlessly

Dashboard displays: context usage %, session age, restart count, critical persistent state.

---

## Multi-Agent Architecture

```
                    ┌─────────────┐
                    │     CAM     │
                    │ Orchestrator│
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌─────▼─────┐     ┌─────▼─────┐
    │ Content │      │  Business │     │  Research │
    │  Agent  │      │   Agent   │     │   Agent   │
    └─────────┘      └───────────┘     └───────────┘
```

Each agent runs a lightweight API server (FastAPI) on a specific port. Agents communicate via HTTP/WebSockets on the local network. Future cluster deployment: each agent on dedicated Pi 5 hardware.

### Current Agent Swarm (OpenClaw-based, learning platform)
- **FireHorseClawd** — Raspberry Pi 5, Claude via Anthropic
  - SSH: `firehorse@192.168.12.243`
  - Connector: systemd service `cam-agent`
- **Nova** — N150, Kimi K2.5 via Moonshot API
  - SSH: `george@192.168.12.149`
  - Connector: manual launch (no systemd yet)

### ROS 2 Robot Fleet (Phase 6 — Implemented)
CAM is a ROS 2 node (`/cam/cam_bridge`) via rclpy, directly integrated with the robot fleet.

- **P1, P2, P3** — mobile_base robots with navigation, lidar, camera capabilities
- rclpy spins in a daemon thread; ROS 2 callbacks bridge to asyncio via `run_coroutine_threadsafe()`
- Robot tools available to Claude: `robot_status` (T1), `robot_sensor_read` (T1), `robot_navigate` (T2), `robot_patrol` (T2), `robot_command` (T2), `robot_emergency_stop` (T0 — always allowed)
- Dashboard: Robot Fleet panel with per-robot status cards, battery bars, e-stop buttons
- Config: `[ros2]` section in `settings.toml` defines robots, waypoints, patrol routes
- Graceful degradation: when rclpy absent or `ros2.enabled = false`, no robot code runs

```
tools/ros2/
├── __init__.py          # Conditional import guard (ROS2_AVAILABLE)
├── msg_bridge.py        # ROS 2 message ↔ JSON serialization
├── node.py              # CamRos2Bridge — rclpy lifecycle, pub/sub, nav goals
├── robot_registry.py    # RobotRegistry + RobotInfo (mirrors AgentRegistry)
└── tools.py             # 6 Claude tool definitions + async executors
```

---

## Dashboard Features (Build Priority: Agent Interface First)

### Core Status Panel
At-a-glance: CAM state (idle/working/waiting), uptime, active model, context window %, session/day/week operating costs.

### Agent Status Board (BUILD FIRST)
Per-agent cards showing: online/offline, current task, model assigned, context usage, last heartbeat, quick-action buttons (command/kill). Agents communicate over WiFi via FastAPI endpoints and WebSocket connections.

### Remote Agent Access
```
Main Machine (192.168.1.100)
├── Dashboard (localhost:8080)
│   ├── WebSocket → Pi #1 Content Agent (192.168.1.105:8000)
│   ├── WebSocket → Pi #2 Research Agent (192.168.1.106:8000)
│   └── WebSocket → Pi #3 Business Agent (192.168.1.107:8000)
└── SSH sessions to any Pi for direct terminal access
```
Use mDNS for service discovery (e.g., `content-agent.local`) so IP changes don't break connections.

### Task Queue & Approval Gate
Pending Tier 2 actions with context and rationale. Approve/reject/modify from dashboard.

### Activity Log / Audit Trail
Live-scrolling feed of all CAM actions, tagged by tier, filterable by agent/action/time.

### Memory & Knowledge Panel
Short-term, episodic, long-term memory stores. Vector DB usage, recent entries, search, manual add/remove.

### Content Pipeline View
Per-piece status: research → script draft → review → TTS → thumbnail → scheduled → published.

### Cost Tracker
API calls, token counts, estimated monthly spend, broken down by agent and task type. Budget thresholds with alerts.

### Communication Hub
Custom app messages, queued notifications, voice call log (when Twilio comes online).

### Kill Switch
Prominent, always visible. One click halts all autonomous action across all agents. Graceful degradation kicks in — agents queue work and wait.

---

## Custom Communication App

Replaces Telegram dependency. Two modes:
- **Home (WiFi):** Direct connection to CAM over local network. Fast, free, no internet needed.
- **Mobile:** WireGuard VPN tunnel back to home network, or reverse proxy with secure endpoint.

App auto-detects mode and connects accordingly.

---

## Security Model

### Permission Tiers
| Tier | Actions | Approval |
|------|---------|----------|
| Safe | Read files, search memory, generate text | None |
| Logged | Write to allowed dirs, send notifications | None (logged) |
| Gated | Execute shell commands, send external messages | Human approval |
| Blocked | Sensitive paths, destructive operations | Never |

### Directory Jail
Explicit allowlist of paths CAM can touch. Everything else off-limits by default.

### Action Classification
Every tool call tagged (safe/logged/gated/blocked) before execution.

### Kill Switch
Multiple halt mechanisms: Dashboard button, CLI `cam --halt`, mobile app, physical (unplug).

### Full Audit
Every action logged: timestamp, tool, parameters, classification, approval status, result, model used, token cost.

---

## Tech Stack

- **Language:** Python (ecosystem, ROS 2 compatibility, readability)
- **Web framework:** FastAPI (dashboard backend, agent API servers)
- **Vector DB:** ChromaDB (simple, local, good Python support)
- **Local models:** Ollama (glm-4.7-flash, gpt-oss20b, qwen3:8b)
- **Cloud models:** Claude API (complex reasoning), Kimi K2.5 via Moonshot (agentic workflows)
- **TTS:** Dia2, Qwen3-TTS, Kokoro (options under evaluation)
- **STT:** Whisper (local)
- **Voice calls:** Twilio (~$0.015/min)
- **VPN:** WireGuard (mobile access)
- **Agent framework:** Custom-built (informed by OpenClaw learnings)

---

## Build Phases

### Phase 1: Foundation
1. Core orchestrator + model router
2. CLI interface (fastest way to test)
3. Basic memory (episodic first)
4. Security/audit scaffolding

### Phase 2: Tools
5. Filesystem tool with permissions
6. Shell tool with approval gates
7. Web tool for research

### Phase 3: Interfaces (Dashboard + Agent Interface first)
8. Agent status board and remote communication
9. Dashboard UI (full feature set)
10. Custom mobile communication app
11. Voice interface (Twilio + Whisper + TTS)

### Phase 4: Doppler Integration
12. Appointment scheduling
13. Customer management
14. Content pipeline (scripts → TTS → scheduling)

### Phase 5: Multi-Agent
15. Agent manager
16. Content agent
17. Business agent
18. Research agent

### Phase 6: Advanced
19. Cluster deployment (agents on dedicated Pi hardware)
20. ~~ROS 2 bridge for robots~~ ✅ (2026-02-15)
21. Proactive "heartbeat" behaviors

---

## Key Design Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-05 | Python as primary language | Ecosystem, ROS 2 compatibility, readability |
| 2026-02-05 | ChromaDB for vector store | Simple, local, good Python support |
| 2026-02-05 | Four-tier memory system | Mirrors human memory, enables persistence |
| 2026-02-05 | Security-first architecture | Learn from OpenClaw vulnerabilities |
| 2026-02-05 | Multi-agent with single orchestrator | Scalable, can distribute to cluster later |
| 2026-02-08 | Model-agnostic design | Landscape changes fast, avoid vendor lock-in |
| 2026-02-08 | Kimi K2.5 for agentic middle tier | Best cost/performance ratio observed |
| 2026-02-09 | Custom mobile app over Telegram | Independence from third-party platforms |
| 2026-02-09 | Build agent interface first | Visibility and control before full autonomy |
| 2026-02-15 | ROS 2 direct integration via rclpy | CAM as a ROS 2 node, not a separate bridge process |
| 2026-02-15 | faster-whisper for voice STT | Lightweight CPU inference, lazy-loaded on first use |

---

## Coding Conventions

- Clear, readable Python — George is learning, code should be understandable
- Docstrings on all public functions
- Type hints where they add clarity
- YAML for configuration files
- Comprehensive logging — if it happened, there's a record
- Security checks before any tool execution
- Cost tracking on every API call
- Git commits with descriptive messages

---

## Context for Claude Code

This project is being built by a solo operator bootstrapping a business. Every design decision balances capability against cost and complexity. Frugality is a core principle — don't over-engineer, don't add dependencies that don't earn their keep, don't burn API credits on tasks local models can handle.

George is an experienced mechanic and technician who is learning software development. Code should be well-commented, modular, and approachable. Explain non-obvious patterns when they appear.

The constitutional framework is the source of truth for all CAM behavior. When in doubt about what CAM should or shouldn't do, refer to the constitution.
