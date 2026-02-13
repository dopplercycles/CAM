"""
CAM Dashboard Server

FastAPI backend serving the agent status dashboard.
Agents connect via WebSocket to report their status.
The dashboard displays real-time agent status to George.

Run with:
    cd ~/CAM
    uvicorn interfaces.dashboard.server:app --host 0.0.0.0 --port 8080 --reload
"""

import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from core.agent_registry import AgentRegistry
from core.config import get_config
from core.event_logger import EventLogger
from core.backup import BackupManager
from core.message_bus import MessageBus
from core.deployer import Deployer
from core.file_transfer import FileTransferManager, chunk_file_data, compute_checksum
from core.health_monitor import HealthMonitor
from core.memory import ShortTermMemory, WorkingMemory, LongTermMemory, EpisodicMemory
from core.notifications import NotificationManager
from core.persona import Persona
from interfaces.telegram.bot import TelegramBot
from core.task import TaskQueue, TaskComplexity, TaskChain, Task, ChainStatus
from core.swarm import SwarmTask, SwarmStatus
from core.orchestrator import Orchestrator
from core.context_manager import ContextManager
from core.analytics import Analytics
from core.commands import CommandLibrary
from core.scheduler import Scheduler, ScheduleType
from core.content_calendar import ContentCalendar, ContentType, ContentStatus
from core.research_store import ResearchStore, ResearchStatus
from tools.doppler.scout import ScoutStore, DopplerScout, SearchCriteria, ListingStatus
from tools.doppler.market_analyzer import MarketAnalyzer
from core.webhook_manager import WebhookManager
from tools.content.pipeline import ContentPipelineManager
from tools.content.youtube import YouTubeManager
from tools.content.video_processor import VideoProcessor
from tools.content.social_media import SocialMediaManager
from tools.doppler.email_templates import EmailTemplateManager
from tools.doppler.photo_docs import PhotoDocManager
from core.training import TrainingManager
from tools.doppler.ride_log import RideLogManager
from tools.doppler.market_monitor import MarketMonitor
from tools.doppler.feedback import FeedbackManager
from tools.doppler.warranty import WarrantyRecallManager
from tools.doppler.maintenance_scheduler import MaintenanceSchedulerManager
from core.plugins import PluginManager
from api.export import ExportManager
from core.offline import OfflineManager
from core.performance import PerformanceMonitor
from core.access_control import AccessControl, Role, WS_PERMISSIONS
from security.audit import SecurityAuditLog
from security.auth import SessionManager
from agents.content_agent import ContentAgent
from tools.content.tts_pipeline import TTSPipeline
from agents.research_agent import ResearchAgent
from agents.business_agent import BusinessAgent, BusinessStore
from tools.doppler.service_records import ServiceRecordStore
from tools.doppler.diagnostics import DiagnosticEngine
from tools.doppler.crm import CRMStore
from tools.doppler.scheduler_appointments import AppointmentScheduler
from tools.doppler.invoicing import InvoiceManager
from tools.doppler.inventory import InventoryManager
from tools.doppler.finances import FinanceTracker
from tools.doppler.route_planner import RoutePlanner
from tools.content.highway20 import Highway20Planner
from core.reports import ReportEngine
from core.memory.knowledge_ingest import KnowledgeIngest
from tests.self_test import SelfTest
from tests.integration_test import LaunchReadiness


# ---------------------------------------------------------------------------
# Logging — CLAUDE.md says "comprehensive logging, if it happened there's a record"
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.dashboard")


# ---------------------------------------------------------------------------
# Configuration — loaded once at startup, hot-reloadable via dashboard
# ---------------------------------------------------------------------------

config = get_config()

# Path to static files directory (index.html lives here)
STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Shared state
#
# The registry is the single source of truth for agent status.
# WebSocket connections and dashboard clients are server-specific.
# ---------------------------------------------------------------------------

# Agent registry — tracks all agents, shared with orchestrator later
# Note: heartbeat_timeout is read at construction — needs restart to change.
registry = AgentRegistry(heartbeat_timeout=config.dashboard.heartbeat_timeout)

# Health monitor — per-agent health metrics, 3-miss heartbeat policy
health_monitor = HealthMonitor(registry=registry, heartbeat_interval=config.health.heartbeat_interval)

# Event logger — centralized audit trail for all CAM activity
# Note: max_events is read at construction — needs restart to change.
event_logger = EventLogger(max_events=config.events.max_events)

# Message bus — inter-agent pub/sub communication
message_bus = MessageBus(
    max_messages=config.message_bus.max_messages,
    event_logger=event_logger,
)

# Analytics — SQLite-backed task history and model cost tracking
# Note: db_path is read at construction — needs restart to change.
analytics = Analytics(db_path=config.analytics.db_path)

# Command library — predefined commands for the dashboard command palette
command_library = CommandLibrary()

# Notification manager — evaluates events against configurable rules
notification_manager = NotificationManager(max_history=config.notifications.max_history)

# Memory systems — session context and persistent task state
short_term_memory = ShortTermMemory(
    max_messages=getattr(getattr(config, 'memory', None), 'short_term_max_messages', 200),
    summary_ratio=getattr(getattr(config, 'memory', None), 'short_term_summary_ratio', 0.5),
)
working_memory = WorkingMemory(
    persist_path=getattr(getattr(config, 'memory', None), 'working_memory_path',
                         'data/tasks/working_memory.json'),
)

# Persona — Cam's identity, voice, and behavioral traits (YAML-driven)
persona = Persona()

# Telegram bot — initialized in lifespan() after activate_kill_switch is defined
telegram_bot: TelegramBot | None = None

# Episodic memory — SQLite-backed conversation history
episodic_memory = EpisodicMemory(
    db_path=getattr(getattr(config, 'memory', None),
                    'episodic_db_path', 'data/memory/episodic.db'),
    retention_days=getattr(getattr(config, 'memory', None),
                           'episodic_retention_days', 365),
)

# Long-term memory — ChromaDB vector store for persistent knowledge
long_term_memory = LongTermMemory(
    persist_directory=getattr(getattr(config, 'memory', None),
                              'long_term_persist_dir', 'data/memory/chromadb'),
    collection_name=getattr(getattr(config, 'memory', None),
                            'long_term_collection', 'cam_long_term'),
)
_seed_file = getattr(getattr(config, 'memory', None),
                     'long_term_seed_file', 'CAM_BRAIN.md')
long_term_memory.seed_from_file(_seed_file)

# Knowledge ingestion — bulk document ingest into LTM
knowledge_ingest = KnowledgeIngest(
    ltm=long_term_memory,
    db_path=getattr(getattr(config, 'knowledge', None), 'db_path', 'data/knowledge_ingest.db'),
    inbox_dir=getattr(getattr(config, 'knowledge', None), 'inbox_dir', 'data/knowledge/inbox'),
    processed_dir=getattr(getattr(config, 'knowledge', None), 'processed_dir', 'data/knowledge/processed'),
    max_file_size=getattr(getattr(config, 'knowledge', None), 'max_file_size', 10_485_760),
    chunk_target_size=getattr(getattr(config, 'knowledge', None), 'chunk_target_size', 1000),
    chunk_overlap=getattr(getattr(config, 'knowledge', None), 'chunk_overlap', 100),
)

# agent_id -> WebSocket (live connections, needed for commands + kill switch)
agent_websockets: dict[str, WebSocket] = {}

# Browser clients watching the dashboard (receive status pushes)
# Value is a dict with mode, username, and role for permission tracking
dashboard_clients: dict[WebSocket, dict] = {}   # ws → {mode, username, role}

# Task queue — shared between dashboard and orchestrator
task_queue = TaskQueue()

# Round-robin counter for fair agent dispatch
_dispatch_counter: int = 0

# Pending task dispatch responses — task_id → asyncio.Future
# Used by dispatch_to_agent() to wait for agent command_responses
pending_task_responses: dict[str, asyncio.Future] = {}

# Kill switch state — when True, all autonomous action is halted
kill_switch_active: bool = False

# Boot time — set by cam_launcher.py before uvicorn starts.
# Falls back to module load time if run directly (not via launcher).
boot_time: str = datetime.now(timezone.utc).isoformat()
boot_duration: float = 0.0

# Last self-test results cache — populated when dashboard runs self-test
last_self_test_results: dict | None = None

# Last launch readiness results cache — populated when dashboard runs integration test
last_launch_readiness_results: dict | None = None

# Background task handle for the orchestrator loop
orchestrator_task: asyncio.Task | None = None

# File transfer manager — tracks active/completed file transfers
ft_manager: FileTransferManager | None = None  # initialized after broadcast_transfer_progress is defined


# ---------------------------------------------------------------------------
# Broadcast helper
# ---------------------------------------------------------------------------

async def broadcast_to_dashboards(message: dict, skip_compact: bool = False):
    """Push a message to all connected dashboard browsers.

    Handles cleanup of any clients that have disconnected.
    Used by both agent status broadcasts and command responses.
    If skip_compact is True, clients in compact mode won't receive this message.
    """
    disconnected = []
    for client, info in list(dashboard_clients.items()):
        if skip_compact and info.get("mode", "full") == "compact":
            continue
        try:
            await client.send_json(message)
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        dashboard_clients.pop(client, None)


async def broadcast_agent_status():
    """Push current agent status to all connected dashboard browsers.

    Reads from the registry — the single source of truth.
    """
    data = registry.to_broadcast_list()
    await broadcast_to_dashboards({"type": "agent_status", "agents": data})


async def broadcast_task_status():
    """Push current task queue state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "task_status",
        "tasks": task_queue.to_broadcast_list(),
        "counts": task_queue.get_status(),
    })


async def broadcast_health_status():
    """Push current agent health metrics to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "health_status",
        "health": health_monitor.to_broadcast_dict(),
    }, skip_compact=True)


async def broadcast_event(event_dict: dict):
    """Push a single new event to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "event",
        "event": event_dict,
    }, skip_compact=True)


async def broadcast_notification(notification_dict: dict):
    """Push a new notification to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "notification",
        "notification": notification_dict,
        "unread_count": notification_manager.get_unread_count(),
    })


# Wire notification manager's broadcast callback
notification_manager.set_broadcast_callback(broadcast_notification)


# Wire the event logger's broadcast callback — also evaluates notification rules
async def broadcast_event_and_evaluate(event_dict: dict):
    """Push event to dashboards and evaluate notification rules."""
    await broadcast_event(event_dict)
    notification_manager.evaluate_event(event_dict)

event_logger.set_broadcast_callback(broadcast_event_and_evaluate)

# Wire the message bus broadcast callback — pushes bus messages to dashboards
async def broadcast_bus_message(message_dict):
    """Push a bus message + stats to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "bus_message",
        "message": message_dict,
        "stats": message_bus.get_stats(),
    })

message_bus.set_broadcast_callback(broadcast_bus_message)


async def broadcast_analytics():
    """Push current analytics summary to all connected dashboard browsers."""
    await broadcast_to_dashboards({"type": "analytics", "data": analytics.get_summary()})


async def broadcast_chain_status():
    """Push current chain state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "chain_status",
        "chains": task_queue.chains_to_broadcast_list(),
    })


async def broadcast_swarm_status():
    """Push current swarm state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "swarm_status",
        "swarms": task_queue.swarms_to_broadcast_list(),
    })


async def broadcast_knowledge_status():
    """Push current knowledge ingestion state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "knowledge_status",
        **knowledge_ingest.to_broadcast_dict(),
    })


async def broadcast_transfer_progress(transfer):
    """Push file transfer progress to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "file_transfer_progress",
        **transfer.to_dict(),
    })


# Now initialize the file transfer manager with the broadcast callback
ft_manager = FileTransferManager(
    on_progress=broadcast_transfer_progress,
    history_size=config.file_transfer.history_size,
    max_active=config.file_transfer.max_active_transfers,
)

# Ensure server-side receive directory exists
Path(config.file_transfer.receive_dir).mkdir(parents=True, exist_ok=True)


async def broadcast_memory_status():
    """Push current memory system state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "memory_status",
        "short_term": short_term_memory.get_status(),
        "working": working_memory.get_status(),
        "long_term": long_term_memory.get_status(),
        "episodic": episodic_memory.get_status(),
    })


async def broadcast_context_status():
    """Push current context window state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "context_status",
        **context_manager.get_status(),
    })


async def broadcast_schedule_status():
    """Push current schedule state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "schedule_status",
        "schedules": scheduler.to_broadcast_list(),
    })


async def broadcast_content_calendar_status():
    """Push current content calendar state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "content_calendar_status",
        "entries": content_calendar.to_broadcast_list(),
    })


async def broadcast_pipeline_status():
    """Push current content pipeline state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "pipeline_status",
        **pipeline_manager.to_broadcast_dict(),
    })


async def broadcast_youtube_status():
    """Push current YouTube publishing state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "youtube_status",
        **youtube_manager.to_broadcast_dict(),
    })


async def broadcast_media_status():
    """Push current media processing state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "media_status",
        **video_processor.to_broadcast_dict(),
    })


async def broadcast_social_media_status():
    """Push current social media posts state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "social_media_status",
        **social_media_manager.to_broadcast_dict(),
    })


async def broadcast_email_status():
    """Push current email template state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "email_status",
        **email_template_manager.to_broadcast_dict(),
    })


async def broadcast_photo_docs_status():
    """Push current photo documentation state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "photo_docs_status",
        **photo_doc_manager.to_broadcast_dict(),
    })


async def broadcast_training_status():
    """Push current training/learning state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "training_status",
        **training_manager.to_broadcast_dict(),
    })


async def broadcast_ride_log_status():
    """Push current ride log state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "ride_log_status",
        **ride_log_manager.to_broadcast_dict(),
    })


async def broadcast_market_monitor_status():
    """Push current market monitor state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "market_monitor_status",
        **market_monitor.to_broadcast_dict(),
    })


async def broadcast_feedback_status():
    """Push current feedback state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "feedback_status",
        **feedback_manager.to_broadcast_dict(),
    })


async def broadcast_warranty_status():
    """Push current warranty/recall state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "warranty_status",
        **warranty_manager.to_broadcast_dict(),
    })


async def broadcast_maintenance_status():
    """Push current maintenance scheduler state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "maintenance_status",
        **maintenance_scheduler.to_broadcast_dict(),
    })


async def broadcast_plugin_status():
    """Push current plugin manager state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "plugin_status",
        **plugin_manager.to_broadcast_dict(),
    })


async def broadcast_research_status():
    """Push current research results to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "research_status",
        "entries": research_store.to_broadcast_list(),
    })


# Content calendar — SQLite-backed content pipeline tracker
content_calendar = ContentCalendar(
    db_path=getattr(getattr(config, 'content_calendar', None),
                    'db_path', 'data/content_calendar.db'),
    on_change=broadcast_content_calendar_status,
)

# Research store — SQLite-backed research result storage
research_store = ResearchStore(
    db_path=getattr(getattr(config, 'research', None),
                    'db_path', 'data/research.db'),
    on_change=broadcast_research_status,
)


async def broadcast_scout_status():
    """Push current scout listings to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "scout_status",
        "entries": scout_store.to_broadcast_list(),
        "status": scout_store.get_status(),
    })


# Scout store — SQLite-backed motorcycle listing storage
scout_store = ScoutStore(
    db_path=getattr(getattr(config, 'scout', None),
                    'db_path', 'data/scout.db'),
    on_change=broadcast_scout_status,
)


async def broadcast_market_analytics():
    """Push current market analytics to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "market_analytics_status",
        **market_analyzer.get_summary(),
    })


# Market analyzer — price tracking and trend analysis (separate DB)
market_analyzer = MarketAnalyzer(
    db_path="data/market_analytics.db",
    scout_store=scout_store,
    on_change=broadcast_market_analytics,
)


async def broadcast_webhook_status():
    """Push current webhook state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "webhook_status",
        **webhook_manager.to_broadcast_dict(),
    })


# Webhook manager — outbound/inbound webhooks with HMAC and retry
_wh_cfg = getattr(config, 'webhooks', None)
webhook_manager = WebhookManager(
    db_path=getattr(_wh_cfg, 'db_path', 'data/webhooks.db'),
    task_queue=task_queue,
    event_logger=event_logger,
    on_change=broadcast_webhook_status,
    max_retries=getattr(_wh_cfg, 'max_retries', 5),
    retry_base_seconds=getattr(_wh_cfg, 'retry_base_seconds', 10),
    retry_max_seconds=getattr(_wh_cfg, 'retry_max_seconds', 3600),
    retry_check_interval=getattr(_wh_cfg, 'retry_check_interval', 15),
    max_delivery_history=getattr(_wh_cfg, 'max_delivery_history', 500),
    inbound_secret=getattr(_wh_cfg, 'inbound_secret', ''),
)

# Re-wire event callback to also evaluate webhooks for outbound delivery
_original_event_broadcast = broadcast_event_and_evaluate

async def _event_broadcast_with_webhooks(event_dict):
    await _original_event_broadcast(event_dict)
    try:
        await webhook_manager.evaluate_event(event_dict)
    except Exception:
        pass

event_logger.set_broadcast_callback(_event_broadcast_with_webhooks)


async def broadcast_business_status():
    """Push current business data to all connected dashboard browsers."""
    data = business_store.to_broadcast_list()
    await broadcast_to_dashboards({
        "type": "business_status",
        **data,
        "status": business_store.get_status(),
    })


# Business store — SQLite-backed customer/appointment/invoice/inventory storage
_biz_cfg = getattr(config, 'business', None)
business_store = BusinessStore(
    db_path=getattr(_biz_cfg, 'db_path', 'data/business.db'),
    on_change=broadcast_business_status,
)
if getattr(_biz_cfg, 'seed_sample_data', True):
    business_store.seed_sample_data()

async def broadcast_service_records_status():
    """Push current service record data to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "service_records_status",
        **service_store.to_broadcast_dict(),
    })

# Service record store — separate SQLite DB for vehicles + service records
service_store = ServiceRecordStore(
    db_path="data/service_records.db",
    on_change=broadcast_service_records_status,
)

async def broadcast_crm_status():
    """Push current CRM data to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "crm_status",
        **crm_store.to_broadcast_dict(),
    })

# CRM store — separate SQLite DB for enriched customer records
crm_store = CRMStore(
    db_path="data/crm.db",
    service_store=service_store,
    on_change=broadcast_crm_status,
)
crm_store.import_from_business_store(business_store)

async def broadcast_appointment_schedule_status():
    """Push current appointment schedule to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "appointment_schedule_status",
        **appointment_scheduler.to_broadcast_dict(),
    })

# Appointment scheduler — separate SQLite DB for enriched scheduling
_appt_cfg = getattr(config, "appointments", None)
appointment_scheduler = AppointmentScheduler(
    db_path=getattr(_appt_cfg, "db_path", "data/appointments.db"),
    on_change=broadcast_appointment_schedule_status,
    home_lat=getattr(_appt_cfg, "home_lat", 45.4976),
    home_lon=getattr(_appt_cfg, "home_lon", -122.4302),
    avg_speed_mph=getattr(_appt_cfg, "avg_speed_mph", 30.0),
    road_factor=getattr(_appt_cfg, "road_factor", 1.4),
)

async def broadcast_invoicing_status():
    """Push current invoicing state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "invoicing_status",
        **invoice_manager.to_broadcast_dict(),
    })

# Invoice manager — separate SQLite DB for professional invoicing
_inv_cfg = getattr(config, "invoicing", None)
_biz_cfg = getattr(config, "business", None)
invoice_manager = InvoiceManager(
    db_path=getattr(_inv_cfg, "db_path", "data/invoices.db"),
    on_change=broadcast_invoicing_status,
    service_store=service_store,
    default_labor_rate=getattr(_biz_cfg, "default_labor_rate", 75.0),
    invoice_prefix=getattr(_biz_cfg, "invoice_prefix", "DC"),
)

# Parts inventory — separate SQLite DB for parts/supplies tracking
async def broadcast_inventory_status():
    """Push current inventory state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "inventory_status",
        **inventory_manager.to_broadcast_dict(),
    })

inventory_manager = InventoryManager(
    db_path="data/inventory.db",
    on_change=broadcast_inventory_status,
    notification_callback=notification_manager.emit,
)

# Financial dashboard — separate SQLite DB for income/expense tracking
async def broadcast_finances_status():
    """Push current finance state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "finances_status",
        **finance_tracker.to_broadcast_dict(current_balance=_finance_state["balance"]),
    })

_finance_state = {"balance": 0.0}  # Manual balance input, stored in memory

finance_tracker = FinanceTracker(
    db_path="data/finances.db",
    on_change=broadcast_finances_status,
    invoice_manager=invoice_manager,
)
finance_tracker.sync_from_invoices()

# Route planner — optimised daily routes for mobile service calls
async def broadcast_route_status():
    """Push current route planner state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "route_status",
        **route_planner.to_broadcast_dict(),
    })

route_planner = RoutePlanner(
    db_path="data/routes.db",
    on_change=broadcast_route_status,
    appointment_scheduler=appointment_scheduler,
)

async def broadcast_hwy20_status():
    """Push current Highway 20 documentary planner state to all dashboards."""
    await broadcast_to_dashboards({
        "type": "hwy20_status",
        **highway20_planner.to_broadcast_dict(),
    })

async def broadcast_diagnostics_status():
    """Push current diagnostic engine state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "diagnostics_status",
        **diagnostic_engine.to_broadcast_dict(),
    })


async def broadcast_reports_status():
    """Push current report engine state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "reports_status",
        **report_engine.to_broadcast_dict(),
    })


async def broadcast_security_audit_status():
    """Push current security audit state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "security_audit_status",
        "entries": security_audit_log.to_broadcast_list(),
        "status": security_audit_log.get_status(),
    })


async def broadcast_export_status():
    """Push current export state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "export_status",
        **export_manager.to_broadcast_dict(),
    })


async def broadcast_offline_status():
    """Push current offline/connectivity state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "offline_status",
        **offline_manager.to_broadcast_dict(),
    })


async def broadcast_performance_status():
    """Push current performance metrics to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "performance_status",
        **performance_monitor.to_broadcast_dict(),
    })


# Security audit log — SQLite-backed persistent audit trail
security_audit_log = SecurityAuditLog(
    db_path=getattr(getattr(config, 'security', None),
                    'audit_db_path', 'data/security_audit.db'),
    on_change=broadcast_security_audit_status,
)

# Session manager — dashboard browser authentication
# When password_hash is empty, auth is disabled (open access, backward compatible)
session_manager = SessionManager(
    username=getattr(getattr(config, 'auth', None), 'username', 'george'),
    password_hash=getattr(getattr(config, 'auth', None), 'password_hash', ''),
    session_timeout=getattr(getattr(config, 'auth', None), 'session_timeout', 3600),
    max_login_attempts=getattr(getattr(config, 'auth', None), 'max_login_attempts', 5),
    lockout_duration=getattr(getattr(config, 'auth', None), 'lockout_duration', 300),
)

# Access control — multi-user role-based permissions
access_control = AccessControl(
    db_path="data/access_control.db",
    audit_log=security_audit_log,
)
access_control.ensure_default_admin(
    username=getattr(getattr(config, 'auth', None), 'username', 'george'),
    password_hash=getattr(getattr(config, 'auth', None), 'password_hash', ''),
)
session_manager.set_access_control(access_control)


# Task scheduler — persists schedules to JSON, submits due tasks to the queue
scheduler = Scheduler(
    task_queue=task_queue,
    persist_path=config.scheduler.persist_file,
    check_interval=config.scheduler.check_interval,
    on_schedule_change=broadcast_schedule_status,
)

# Seed daily backup schedule if it doesn't already exist
if not any(s.name == "Daily Backup" for s in scheduler._schedules.values()):
    scheduler.add_schedule(
        name="Daily Backup",
        description="system_backup",
        schedule_type=ScheduleType.DAILY,
        run_at_time=config.backup.daily_backup_time,
    )

# Seed report schedules — daily, weekly, and monthly
_reports_cfg = getattr(config, "reports", None)
if not any(s.name == "Daily Report" for s in scheduler._schedules.values()):
    scheduler.add_schedule(
        name="Daily Report",
        description="generate_daily_report",
        schedule_type=ScheduleType.DAILY,
        run_at_time=_reports_cfg.daily_time if _reports_cfg else "06:00",
    )
if not any(s.name == "Weekly Report" for s in scheduler._schedules.values()):
    scheduler.add_schedule(
        name="Weekly Report",
        description="generate_weekly_report",
        schedule_type=ScheduleType.DAILY,
        run_at_time=_reports_cfg.weekly_time if _reports_cfg else "07:00",
    )
if not any(s.name == "Monthly Report" for s in scheduler._schedules.values()):
    scheduler.add_schedule(
        name="Monthly Report",
        description="generate_monthly_report",
        schedule_type=ScheduleType.DAILY,
        run_at_time=_reports_cfg.monthly_time if _reports_cfg else "08:00",
    )


async def relay_file_to_agent(transfer, data: bytes, agent_ws):
    """Send file data to an agent in chunks over WebSocket.

    Launched as a background task from the upload endpoint or WS handler.
    Sends file_transfer_start, then iterates chunks with brief yields.
    The agent acks chunks asynchronously; acks update progress via the
    agent WS handler.
    """
    chunk_size = config.file_transfer.chunk_size
    chunks = chunk_file_data(data, chunk_size)

    try:
        # Tell the agent a transfer is starting
        await agent_ws.send_json({
            "type": "file_transfer_start",
            "transfer_id": transfer.transfer_id,
            "filename": transfer.filename,
            "file_size": transfer.file_size,
            "chunk_count": transfer.chunk_count,
            "dest_path": transfer.dest_path,
            "checksum": transfer.checksum,
        })

        # Stream chunks
        for i, b64_chunk in enumerate(chunks):
            await agent_ws.send_json({
                "type": "file_chunk",
                "transfer_id": transfer.transfer_id,
                "chunk_index": i,
                "chunk_count": transfer.chunk_count,
                "data": b64_chunk,
            })
            await asyncio.sleep(0)  # yield to event loop between chunks

    except Exception as e:
        logger.warning("Relay to agent failed for %s: %s", transfer.transfer_id, e)
        await ft_manager.fail_transfer(transfer.transfer_id, f"Relay failed: {e}")


# Agent deployer — orchestrates connector updates via existing file transfer + command systems
deployer = Deployer(
    ft_manager=ft_manager,
    event_logger=event_logger,
    registry=registry,
    agent_websockets=agent_websockets,
    relay_fn=relay_file_to_agent,
    config=config,
)

# Backup manager — creates/restores/rotates backup archives
backup_manager = BackupManager(config=config, event_logger=event_logger)


# ---------------------------------------------------------------------------
# Orchestrator callbacks — push OATI phase changes to dashboards in real time
# ---------------------------------------------------------------------------

async def on_task_phase_change(task, phase, detail):
    """Called by the orchestrator at each OATI phase boundary."""
    await broadcast_to_dashboards({
        "type": "task_phase",
        "task_id": task.task_id,
        "short_id": task.short_id,
        "phase": phase,
        "detail": detail,
    })
    severity = "error" if phase == "FAILED" else "info"
    getattr(event_logger, severity)(
        "task", f"[{phase}] {detail or task.short_id}",
        task_id=task.short_id, phase=phase,
    )

    # Record completed/failed tasks in analytics DB and push to dashboards
    if phase in ("COMPLETED", "FAILED"):
        analytics.record_task(task)
        await broadcast_analytics()
        await message_bus.publish("system", "task_updates", {
            "action": "phase_change", "task_id": task.short_id, "phase": phase,
        })

    # Push updated memory status to dashboards (memory changes during OATI)
    await broadcast_memory_status()
    await broadcast_context_status()


async def on_task_update():
    """Called by the orchestrator when any task's status changes."""
    await broadcast_task_status()


async def on_chain_update(chain):
    """Called by the orchestrator when a chain's status changes."""
    await broadcast_chain_status()
    status_str = chain.status.value if hasattr(chain.status, 'value') else str(chain.status)
    severity = "error" if status_str == "failed" else "info"
    getattr(event_logger, severity)(
        "chain",
        f"Chain '{chain.name}' ({chain.short_id}): {status_str} "
        f"— step {min(chain.current_step + 1, chain.total_steps)}/{chain.total_steps}",
        chain_id=chain.short_id, status=status_str,
    )


async def on_swarm_update(swarm):
    """Called by the orchestrator when a swarm's status changes."""
    await broadcast_swarm_status()
    status_str = swarm.status.value if hasattr(swarm.status, 'value') else str(swarm.status)
    severity = "error" if status_str == "failed" else "info"
    done = len(swarm.completed_subtasks) + len(swarm.failed_subtasks)
    total = len(swarm.subtasks)
    getattr(event_logger, severity)(
        "swarm",
        f"Swarm '{swarm.name}' ({swarm.short_id}): {status_str} "
        f"— {done}/{total} subtasks done ({swarm.progress_pct}%)",
        swarm_id=swarm.short_id, status=status_str,
    )


# ---------------------------------------------------------------------------
# Agent dispatch — sends tasks to remote agents, waits for response
# ---------------------------------------------------------------------------

# Dispatch timeout is read from config at call time (hot-reloadable)


async def dispatch_to_agent(task, plan):
    """Dispatch a task to an available remote agent.

    Routing logic:
    1. If task.assigned_agent is set and that agent is online → use it
    2. Otherwise → pick the first available agent from the registry
    3. If no agents available → return None (orchestrator falls back to model response)

    Sends the task description as a command via the agent's WebSocket,
    then waits for the agent's command_response with a matching task_id.

    Args:
        task: The Task being executed.
        plan: The plan dict from the THINK phase (unused for now, available for future).

    Returns:
        The agent's response string, or None if no agent was available.
    """
    # --- Local agents first — content agent handles content tasks without network overhead ---
    try:
        content_result = await content_agent.try_handle(task, plan)
        if content_result is not None:
            task.assigned_agent = "content_agent"
            event_logger.info("task", f"Content agent handled {task.short_id}",
                              task_id=task.short_id, agent="content_agent")
            await message_bus.publish("content_agent", "task_updates", {
                "action": "task_handled", "task_id": task.short_id,
            })
            return content_result
    except Exception as e:
        logger.warning("Content agent failed for task %s: %s", task.short_id, e)

    # --- Local agents: research agent handles research tasks ---
    try:
        research_result = await research_agent.try_handle(task, plan)
        if research_result is not None:
            task.assigned_agent = "research_agent"
            event_logger.info("task", f"Research agent handled {task.short_id}",
                              task_id=task.short_id, agent="research_agent")
            await message_bus.publish("research_agent", "task_updates", {
                "action": "task_handled", "task_id": task.short_id,
            })
            return research_result
    except Exception as e:
        logger.warning("Research agent failed for task %s: %s", task.short_id, e)

    # --- Local agents: business agent handles business tasks ---
    try:
        business_result = await business_agent.try_handle(task, plan)
        if business_result is not None:
            task.assigned_agent = "business_agent"
            event_logger.info("task", f"Business agent handled {task.short_id}",
                              task_id=task.short_id, agent="business_agent")
            await message_bus.publish("business_agent", "task_updates", {
                "action": "task_handled", "task_id": task.short_id,
            })
            return business_result
    except Exception as e:
        logger.warning("Business agent failed for task %s: %s", task.short_id, e)

    # --- System tasks: backup ---
    if task.description == "system_backup":
        task.assigned_agent = "backup_manager"
        result = backup_manager.backup()
        if result.get("ok"):
            return f"Backup complete: {result['filename']} ({result['file_count']} files)"
        return f"Backup failed: {result.get('error', 'unknown error')}"

    # --- System tasks: scheduled reports ---
    if task.description == "generate_daily_report":
        task.assigned_agent = "report_engine"
        report = report_engine.generate_report("daily")
        await broadcast_reports_status()
        return f"Daily report generated: {report.title}"
    if task.description == "generate_weekly_report":
        task.assigned_agent = "report_engine"
        # Only generate weekly on Mondays
        from datetime import datetime as _dt, timezone as _tz
        if _dt.now(_tz.utc).weekday() != 0:
            return "Skipped — weekly report only runs on Mondays"
        report = report_engine.generate_report("weekly")
        await broadcast_reports_status()
        return f"Weekly report generated: {report.title}"
    if task.description == "generate_monthly_report":
        task.assigned_agent = "report_engine"
        # Only generate monthly on the 1st
        from datetime import datetime as _dt, timezone as _tz
        if _dt.now(_tz.utc).day != 1:
            return "Skipped — monthly report only runs on the 1st"
        report = report_engine.generate_report("monthly")
        await broadcast_reports_status()
        return f"Monthly report generated: {report.title}"

    # --- Routing: find the target agent ---
    agent_id = None
    required_caps = getattr(task, "required_capabilities", []) or []

    if task.assigned_agent:
        # Check if the pre-assigned agent is online, connected, and capable
        agent_info = registry.get_by_id(task.assigned_agent)
        if agent_info and agent_info.status == "online" and task.assigned_agent in agent_websockets:
            if not required_caps or set(required_caps).issubset(set(agent_info.capabilities)):
                agent_id = task.assigned_agent
            else:
                logger.warning(
                    "Assigned agent '%s' lacks required capabilities %s (has %s) — trying others",
                    task.assigned_agent, required_caps, agent_info.capabilities,
                )

    if agent_id is None:
        # Pick a capable agent using round-robin so tasks are
        # distributed fairly across all available agents.
        global _dispatch_counter
        capable = registry.get_capable(required_caps)
        connected = [a for a in capable if a.agent_id in agent_websockets]
        if connected:
            idx = _dispatch_counter % len(connected)
            agent_id = connected[idx].agent_id
            _dispatch_counter += 1

    if agent_id is None:
        caps_msg = f" (required: {required_caps})" if required_caps else ""
        logger.info("No capable agents for task %s%s — falling back to model", task.short_id, caps_msg)
        event_logger.info("task", f"No capable agents for {task.short_id}{caps_msg}, using model fallback",
                          task_id=task.short_id)
        return None

    agent_info = registry.get_by_id(agent_id)
    agent_name = agent_info.name if agent_info else agent_id

    # --- Mark agent busy, set task assignment ---
    task.assigned_agent = agent_name
    if agent_info:
        agent_info.status = "busy"
    health_monitor.on_task_dispatched(agent_id)
    await broadcast_agent_status()

    logger.info(
        "Dispatching task %s to agent '%s' (%s)",
        task.short_id, agent_name, agent_id,
    )
    event_logger.info("task", f"Dispatching {task.short_id} to {agent_name}",
                      task_id=task.short_id, agent=agent_name)
    await broadcast_to_dashboards({
        "type": "task_phase",
        "task_id": task.task_id,
        "short_id": task.short_id,
        "phase": "ACT",
        "detail": f"Dispatching to {agent_name}",
    })

    # --- Send command and wait for response ---
    future = asyncio.get_event_loop().create_future()
    pending_task_responses[task.task_id] = future

    try:
        ws = agent_websockets[agent_id]
        await ws.send_json({
            "type": "command",
            "command": task.description,
            "task_id": task.task_id,
        })

        # Wait for the agent to respond (or timeout)
        dispatch_timeout = config.dashboard.agent_dispatch_timeout
        result = await asyncio.wait_for(future, timeout=dispatch_timeout)
        logger.info(
            "Agent '%s' responded to task %s: %.200s",
            agent_name, task.short_id, result,
        )
        health_monitor.on_task_completed(agent_id)
        event_logger.info("task", f"Agent {agent_name} completed {task.short_id}",
                          task_id=task.short_id, agent=agent_name)
        return result

    except asyncio.TimeoutError:
        logger.warning(
            "Agent '%s' timed out on task %s after %ds",
            agent_name, task.short_id, dispatch_timeout,
        )
        health_monitor.on_task_failed(agent_id)
        event_logger.warn("task", f"Agent {agent_name} timed out on {task.short_id}",
                          task_id=task.short_id, agent=agent_name,
                          timeout=dispatch_timeout)
        return f"[timeout] Agent '{agent_name}' did not respond within {dispatch_timeout}s"

    except Exception as e:
        logger.warning(
            "Dispatch to agent '%s' failed for task %s: %s",
            agent_name, task.short_id, e,
        )
        health_monitor.on_task_failed(agent_id)
        event_logger.error("task", f"Dispatch to {agent_name} failed: {e}",
                           task_id=task.short_id, agent=agent_name, error=str(e))
        return None

    finally:
        pending_task_responses.pop(task.task_id, None)
        # Restore agent to online
        if agent_info and agent_info.status == "busy":
            agent_info.status = "online"
        await broadcast_agent_status()


# ---------------------------------------------------------------------------
# Orchestrator instance — uses the shared task queue and callbacks
# ---------------------------------------------------------------------------

def on_model_call(model, backend, tokens, latency_ms, cost_usd, task_short_id):
    """Called by the orchestrator after each model router call."""
    event_logger.info(
        "model",
        f"Model call: {model} ({backend}) — {tokens} tokens, "
        f"{latency_ms:.0f}ms, ${cost_usd:.6f}",
        model=model, backend=backend, tokens=tokens,
        latency_ms=round(latency_ms, 1), cost_usd=round(cost_usd, 6),
        task_id=task_short_id,
    )
    analytics.record_model_call(
        model=model, backend=backend, tokens=tokens,
        latency_ms=latency_ms, cost_usd=cost_usd,
        task_short_id=task_short_id,
    )


async def on_approval_request(task, perm_result):
    """Called by the orchestrator when a Tier 2 task needs approval.

    Broadcasts the approval request to all connected dashboard browsers
    so George can approve or reject from the Security & Audit panel.
    """
    await broadcast_to_dashboards({
        "type": "approval_request",
        "task_id": task.task_id,
        "short_id": task.short_id,
        "description": task.description,
        "action_type": perm_result.action_type,
        "risk_level": perm_result.risk_level,
        "reason": perm_result.reason,
        "tier": perm_result.tier,
    })
    event_logger.warn(
        "security",
        f"Approval requested for {task.short_id}: {perm_result.action_type} — {task.description[:80]}",
        task_id=task.short_id, action_type=perm_result.action_type,
        risk_level=perm_result.risk_level,
    )


# Context window manager — shares memory instances with orchestrator
context_manager = ContextManager(
    short_term=short_term_memory,
    long_term=long_term_memory,
    episodic=episodic_memory,
    working=working_memory,
    persona=persona,
)

orchestrator = Orchestrator(
    queue=task_queue,
    short_term_memory=short_term_memory,
    working_memory=working_memory,
    long_term_memory=long_term_memory,
    episodic_memory=episodic_memory,
    persona=persona,
    on_phase_change=on_task_phase_change,
    on_task_update=on_task_update,
    on_dispatch_to_agent=dispatch_to_agent,
    on_model_call=on_model_call,
    on_chain_update=on_chain_update,
    on_swarm_update=on_swarm_update,
    on_approval_request=on_approval_request,
    audit_log=security_audit_log,
    context_manager=context_manager,
)

# Diagnostic decision tree engine — YAML-based diagnostic workflows
diagnostic_engine = DiagnosticEngine(
    db_path="data/diagnostics.db",
    trees_dir="config/diagnostic_trees",
    service_store=service_store,
    router=orchestrator.router,
    on_change=broadcast_diagnostics_status,
)

# Highway 20 documentary planner — segment tracking, shots, weather
highway20_planner = Highway20Planner(
    db_path="data/highway20.db",
    router=orchestrator.router,
    ltm=long_term_memory,
    on_change=broadcast_hwy20_status,
)

# Scheduled reporting engine — pulls from all data stores
report_engine = ReportEngine(
    db_path=getattr(config, "reports", None) and config.reports.db_path or "data/reports.db",
    pdf_dir=getattr(config, "reports", None) and config.reports.pdf_dir or "data/reports",
    analytics=analytics,
    business_store=business_store,
    scout_store=scout_store,
    event_logger=event_logger,
    content_calendar=content_calendar,
    service_store=service_store,
    market_analyzer=market_analyzer,
    registry=registry,
    on_change=broadcast_reports_status,
)

# TTS pipeline — Piper TTS with graceful fallback
tts_pipeline = TTSPipeline(config=config)

# Content agent — local in-process agent for content tasks
content_agent = ContentAgent(
    router=orchestrator.router,
    persona=persona,
    long_term_memory=long_term_memory,
    calendar=content_calendar,
    event_logger=event_logger,
    on_model_call=on_model_call,
    tts_pipeline=tts_pipeline,
)

# Content pipeline manager — tracks content through research → TTS → package
pipeline_manager = ContentPipelineManager(
    db_path="data/content_pipeline.db",
    task_queue=task_queue,
    tts_pipeline=tts_pipeline,
    event_logger=event_logger,
    content_calendar=content_calendar,
    on_change=broadcast_pipeline_status,
)

# YouTube publishing pipeline — tracks videos through draft → queued → published
youtube_manager = YouTubeManager(
    db_path="data/youtube.db",
    on_change=broadcast_youtube_status,
    router=orchestrator.router,
)

# Video processor — media file import, metadata, thumbnails, transcoding
video_processor = VideoProcessor(
    db_path="data/media.db",
    on_change=broadcast_media_status,
    router=orchestrator.router,
)

# Social media manager — cross-platform content generator
social_media_manager = SocialMediaManager(
    db_path="data/social_media.db",
    on_change=broadcast_social_media_status,
    router=orchestrator.router,
)

# Email template manager — customer communication generator
email_template_manager = EmailTemplateManager(
    db_path="data/email_communications.db",
    on_change=broadcast_email_status,
    router=orchestrator.router,
    crm_store=crm_store,
    service_store=service_store,
    config=config,
)

# Photo documentation manager — service photography pipeline
photo_doc_manager = PhotoDocManager(
    db_path="data/photo_docs.db",
    photos_dir="data/photos",
    on_change=broadcast_photo_docs_status,
    service_store=service_store,
    crm_store=crm_store,
)

# Training and learning manager — captures diagnostic expertise
training_manager = TrainingManager(
    db_path="data/training.db",
    router=orchestrator.router,
    episodic_memory=episodic_memory,
    long_term_memory=long_term_memory,
    diagnostic_engine=diagnostic_engine,
    on_change=broadcast_training_status,
)

# Ride and service log — mileage tracking for tax documentation
ride_log_manager = RideLogManager(
    db_path="data/ride_log.db",
    on_change=broadcast_ride_log_status,
)

# Competitive market monitor — tracks local service landscape
market_monitor = MarketMonitor(
    db_path="data/market_monitor.db",
    router=orchestrator.router,
    notification_manager=notification_manager,
    long_term_memory=long_term_memory,
    on_change=broadcast_market_monitor_status,
)

# Customer feedback manager — ratings, NPS, sentiment analysis
feedback_manager = FeedbackManager(
    db_path="data/feedback.db",
    router=orchestrator.router,
    crm_store=crm_store,
    service_store=service_store,
    notification_manager=notification_manager,
    on_change=broadcast_feedback_status,
)

# Warranty & Recall Tracker — warranties and safety recall notices
warranty_manager = WarrantyRecallManager(
    db_path="data/warranty.db",
    crm_store=crm_store,
    service_store=service_store,
    notification_manager=notification_manager,
    email_template_manager=email_template_manager,
    on_change=broadcast_warranty_status,
)

# Recurring Maintenance Scheduler — preventive maintenance tracking
maintenance_scheduler = MaintenanceSchedulerManager(
    db_path="data/maintenance_scheduler.db",
    service_store=service_store,
    crm_store=crm_store,
    notification_manager=notification_manager,
    on_change=broadcast_maintenance_status,
)

# Plugin Manager — extensible plugin system
plugin_manager = PluginManager(
    plugins_dir="plugins",
    db_path="data/plugins.db",
    on_change=broadcast_plugin_status,
)

# Export Manager — data export and reporting API
export_manager = ExportManager(
    export_dir="data/exports",
    db_path="data/exports.db",
    on_change=broadcast_export_status,
)

# Offline Manager — connectivity monitoring and graceful degradation
offline_manager = OfflineManager(
    cache_dir="data/cache",
    db_path="data/offline.db",
    ollama_url=getattr(getattr(config, 'models', None), 'ollama_url', 'http://localhost:11434'),
    check_interval=30,
    on_change=broadcast_offline_status,
)

# Performance Monitor — system metrics, alerts, optimization recommendations
performance_monitor = PerformanceMonitor(
    db_path="data/performance.db",
    sample_interval=15,
    analytics_db=getattr(getattr(config, 'analytics', None), 'db_path', 'data/analytics.db'),
    on_change=broadcast_performance_status,
    health_monitor=health_monitor,
    registry=registry,
    task_queue=task_queue if 'task_queue' in dir() else None,
)

# Research agent — local in-process agent for research tasks
research_agent = ResearchAgent(
    router=orchestrator.router,
    persona=persona,
    long_term_memory=long_term_memory,
    research_store=research_store,
    event_logger=event_logger,
    on_model_call=on_model_call,
)

# Business agent — local in-process agent for business tasks
business_agent = BusinessAgent(
    router=orchestrator.router,
    persona=persona,
    long_term_memory=long_term_memory,
    business_store=business_store,
    event_logger=event_logger,
    on_model_call=on_model_call,
    crm_store=crm_store,
)

# Doppler Scout — motorcycle listing scraper with deal scoring
_scout_cfg = getattr(config, 'scout', None)
_scout_criteria = SearchCriteria(
    makes=getattr(_scout_cfg, 'makes', ["honda", "yamaha", "suzuki", "kawasaki", "harley-davidson", "ducati"]),
    models=getattr(_scout_cfg, 'models', []),
    year_min=getattr(_scout_cfg, 'year_min', 0),
    year_max=getattr(_scout_cfg, 'year_max', 0),
    price_min=getattr(_scout_cfg, 'price_min', 0),
    price_max=getattr(_scout_cfg, 'price_max', 5000),
    location=getattr(_scout_cfg, 'location', 'Portland OR'),
    radius_miles=getattr(_scout_cfg, 'radius_miles', 50),
    keywords=getattr(_scout_cfg, 'keywords', []),
    exclude_keywords=getattr(_scout_cfg, 'exclude_keywords', ['parts only']),
)
doppler_scout = DopplerScout(
    store=scout_store,
    router=orchestrator.router,
    criteria=_scout_criteria,
    event_logger=event_logger,
    notification_manager=notification_manager,
    on_model_call=on_model_call,
)

# Wire up market analyzer with the model router now that orchestrator is ready
market_analyzer.router = orchestrator.router
market_analyzer._on_model_call = on_model_call


# ---------------------------------------------------------------------------
# Kill switch — CAM Constitution: "Prominent, always visible. One click
# halts all autonomous action across all agents."
# ---------------------------------------------------------------------------

async def activate_kill_switch():
    """Send shutdown command to every connected agent.

    Per the CAM Constitution, the kill switch must:
    - Halt all autonomous action across all agents
    - Be always visible and one-click
    - Trigger graceful degradation — agents queue work and wait

    This sends a shutdown message to each agent's WebSocket and
    notifies all dashboard browsers of the new state.
    """
    global kill_switch_active
    kill_switch_active = True
    app.state.kill_switch["active"] = True
    logger.critical("KILL SWITCH ACTIVATED — shutting down all agents")
    event_logger.error("system", "KILL SWITCH ACTIVATED — all agents halting")

    # Stop the orchestrator loop
    orchestrator.stop()

    shutdown_msg = {"type": "shutdown", "reason": "kill_switch"}

    # Send shutdown to every connected agent
    disconnected = []
    for agent_id, ws in agent_websockets.items():
        try:
            await ws.send_json(shutdown_msg)
            logger.info("Shutdown sent to agent '%s'", agent_id)
        except Exception:
            disconnected.append(agent_id)

    # Clean up any that were already gone
    for agent_id in disconnected:
        del agent_websockets[agent_id]

    # Mark all agents offline in the registry
    for agent in registry.list_all():
        agent.status = "offline"

    # Notify all dashboard browsers
    await broadcast_agent_status()

    # Also send kill switch state to dashboards
    for client in dashboard_clients:
        try:
            await client.send_json({"type": "kill_switch", "active": True})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Background session cleanup
# ---------------------------------------------------------------------------

async def cleanup_sessions():
    """Background task: purge expired sessions every 60 seconds."""
    while True:
        await asyncio.sleep(60)
        session_manager.cleanup_expired()


# ---------------------------------------------------------------------------
# Background heartbeat checker
# ---------------------------------------------------------------------------

async def check_heartbeats():
    """Background task: check agent heartbeats and broadcast health.

    Uses the HealthMonitor's 3-missed-heartbeats policy as the primary
    offline detection. Also broadcasts health_status every cycle so
    the dashboard indicators stay current.
    """
    while True:
        await asyncio.sleep(config.dashboard.heartbeat_check_interval)
        newly_offline = health_monitor.check_heartbeats()
        if newly_offline:
            for aid in newly_offline:
                event_logger.warn("agent", f"Agent '{aid}' went offline (3 missed heartbeats)",
                                  agent_id=aid)
            await broadcast_agent_status()
        # Always broadcast health so dashboard indicators update
        await broadcast_health_status()


async def pipeline_progress_loop():
    """Check content pipeline progress every 5 seconds."""
    while True:
        try:
            await asyncio.sleep(5)
            await pipeline_manager.check_progress()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.debug("Pipeline progress loop error", exc_info=True)


async def knowledge_inbox_loop():
    """Periodically scan the inbox directory for new documents to ingest."""
    interval = getattr(getattr(config, 'knowledge', None), 'scan_interval', 30)
    while True:
        try:
            await asyncio.sleep(interval)
            results = knowledge_ingest.scan_inbox()
            if results:
                logger.info("Knowledge inbox scan: %d files processed", len(results))
                await broadcast_knowledge_status()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.debug("Knowledge inbox loop error", exc_info=True)


async def appointment_reminder_loop():
    """Periodically check for upcoming appointments and emit reminder notifications."""
    interval = getattr(getattr(config, "appointments", None), "reminder_check_interval", 300)
    while True:
        try:
            await asyncio.sleep(interval)
            pending = appointment_scheduler.get_pending_reminders()
            for item in pending:
                appt = item["appointment"]
                rtype = item["reminder_type"]
                hours = item["hours_until"]
                if rtype == "24hr":
                    title = "Appointment Tomorrow"
                    body = (
                        f"{appt['customer_name']} — {appt['service_type']} "
                        f"at {appt['time_slot']} ({appt['location_address'] or 'no address'})"
                    )
                else:
                    title = "Appointment in ~1 Hour"
                    body = (
                        f"{appt['customer_name']} — {appt['service_type']} "
                        f"at {appt['time_slot']} ({appt['location_address'] or 'no address'})"
                    )
                notification_manager.emit("info", title, body, "appointment")
                appointment_scheduler.mark_reminder_sent(appt["appointment_id"], rtype)
                logger.info("Sent %s reminder for appointment %s", rtype, appt["appointment_id"][:8])
        except asyncio.CancelledError:
            break
        except Exception:
            logger.debug("Appointment reminder loop error", exc_info=True)


async def invoice_overdue_loop():
    """Hourly check for invoices past their due date — auto-mark overdue."""
    while True:
        try:
            await asyncio.sleep(3600)
            newly_overdue = invoice_manager.check_overdue()
            for inv in newly_overdue:
                notification_manager.emit(
                    "warning",
                    "Invoice Overdue",
                    f"{inv.invoice_number} ({inv.customer_name}) — ${inv.total:.2f} due {inv.due_date}",
                    "invoicing",
                )
            if newly_overdue:
                await broadcast_invoicing_status()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.debug("Invoice overdue loop error", exc_info=True)


# ---------------------------------------------------------------------------
# App lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background tasks on startup, clean up on shutdown."""
    global orchestrator_task, telegram_bot
    logger.info("CAM Dashboard starting up")
    event_logger.info("system", "CAM Dashboard starting up")
    heartbeat_task = asyncio.create_task(check_heartbeats())
    session_cleanup_task = asyncio.create_task(cleanup_sessions())
    orchestrator_task = asyncio.create_task(orchestrator.run())
    scheduler_task = asyncio.create_task(scheduler.run())
    webhook_retry_task = asyncio.create_task(webhook_manager.process_retries())
    pipeline_check_task = asyncio.create_task(pipeline_progress_loop())
    knowledge_inbox_task = asyncio.create_task(knowledge_inbox_loop())
    appt_reminder_task = asyncio.create_task(appointment_reminder_loop())
    invoice_overdue_task = asyncio.create_task(invoice_overdue_loop())
    offline_monitor_task = asyncio.create_task(offline_manager.start_monitoring())
    performance_monitor_task = asyncio.create_task(performance_monitor.start_monitoring())
    logger.info("Orchestrator loop started as background task")
    logger.info("Appointment reminder loop started")
    logger.info("Invoice overdue loop started")
    logger.info("Offline monitoring loop started")
    logger.info("Performance monitoring loop started")
    logger.info("Scheduler loop started as background task")
    logger.info("Pipeline progress loop started as background task")
    logger.info("Webhook retry loop started as background task")
    if session_manager.auth_enabled:
        logger.info("Dashboard authentication ENABLED (user: %s)", session_manager.username)
    else:
        logger.info("Dashboard authentication DISABLED (no password_hash configured)")

    # Telegram bot — instantiated here so activate_kill_switch is in scope
    telegram_bot = TelegramBot(
        token=getattr(getattr(config, 'telegram', None), 'bot_token', ''),
        allowed_chat_ids=getattr(getattr(config, 'telegram', None), 'allowed_chat_ids', []),
        task_queue=task_queue,
        registry=registry,
        health_monitor=health_monitor,
        episodic_memory=episodic_memory,
        event_logger=event_logger,
        activate_kill_switch=activate_kill_switch,
        on_status_change=broadcast_to_dashboards,
        poll_interval=getattr(getattr(config, 'telegram', None), 'poll_interval', 0.5),
        task_timeout=getattr(getattr(config, 'telegram', None), 'task_timeout', 120.0),
    )
    await telegram_bot.start()
    app.state.telegram_bot = telegram_bot

    # Initialize plugin system — load enabled plugins and fire on_startup hooks
    plugin_manager.set_app(app)
    plugin_manager.on_startup()

    # Boot complete — log and notify
    event_logger.info("system", f"CAM Online (boot: {boot_duration}s)")
    logger.info("CAM Online — boot completed in %.2fs", boot_duration)

    yield

    # --- Graceful shutdown ---
    logger.info("CAM shutdown initiated")
    event_logger.info("system", "CAM shutting down")

    # 0. Shut down plugins first
    plugin_manager.on_shutdown()

    # 1. Notify connected agents
    shutdown_msg = {"type": "shutdown", "reason": "cam_shutdown"}
    for agent_id, ws in agent_websockets.items():
        try:
            await ws.send_json(shutdown_msg)
            logger.info("Shutdown notification sent to agent '%s'", agent_id)
        except Exception:
            pass

    # 2. Save working memory
    working_memory._save()
    logger.info("Working memory saved (%d active tasks)", len(working_memory))

    # 3. Stop Telegram before orchestrator so in-flight polls end cleanly
    if telegram_bot:
        await telegram_bot.stop()

    # 4. Stop orchestrator and scheduler
    orchestrator.stop()
    scheduler.stop()

    # 5. Create emergency backup
    try:
        result = backup_manager.backup()
        logger.info("Shutdown backup created: %s", result.get("filename", "unknown"))
    except Exception as e:
        logger.warning("Shutdown backup failed: %s", e)

    # 6. Cancel background tasks
    orchestrator_task.cancel()
    heartbeat_task.cancel()
    session_cleanup_task.cancel()
    scheduler_task.cancel()
    webhook_retry_task.cancel()
    pipeline_check_task.cancel()
    knowledge_inbox_task.cancel()
    appt_reminder_task.cancel()
    invoice_overdue_task.cancel()
    offline_manager.stop_monitoring()
    offline_monitor_task.cancel()
    performance_monitor.stop_monitoring()
    performance_monitor_task.cancel()

    # 7. Close all databases
    knowledge_ingest.close()
    episodic_memory.close()
    content_calendar.close()
    pipeline_manager.close()
    youtube_manager.close()
    video_processor.close()
    social_media_manager.close()
    email_template_manager.close()
    photo_doc_manager.close()
    training_manager.close()
    ride_log_manager.close()
    market_monitor.close()
    feedback_manager.close()
    warranty_manager.close()
    maintenance_scheduler.close()
    plugin_manager.close()
    export_manager.close()
    offline_manager.close()
    performance_monitor.close()
    research_store.close()
    scout_store.close()
    market_analyzer.close()
    business_store.close()
    service_store.close()
    crm_store.close()
    appointment_scheduler.close()
    invoice_manager.close()
    inventory_manager.close()
    finance_tracker.close()
    route_planner.close()
    highway20_planner.close()
    diagnostic_engine.close()
    report_engine.close()
    security_audit_log.close()
    webhook_manager.close()
    analytics.close()
    logger.info("CAM shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="CAM Dashboard", lifespan=lifespan)

# Expose shared state on app.state so the REST API router (and future
# routers) can access everything without circular imports.
app.state.registry = registry
app.state.health_monitor = health_monitor
app.state.event_logger = event_logger
app.state.analytics = analytics
app.state.task_queue = task_queue
app.state.scheduler = scheduler
app.state.agent_websockets = agent_websockets
app.state.config = config
app.state.short_term_memory = short_term_memory
app.state.working_memory = working_memory
app.state.long_term_memory = long_term_memory
app.state.episodic_memory = episodic_memory
app.state.content_calendar = content_calendar
app.state.content_agent = content_agent
app.state.tts_pipeline = tts_pipeline
app.state.research_store = research_store
app.state.research_agent = research_agent
app.state.scout_store = scout_store
app.state.doppler_scout = doppler_scout
app.state.market_analyzer = market_analyzer
app.state.business_store = business_store
app.state.business_agent = business_agent
app.state.service_store = service_store
app.state.diagnostic_engine = diagnostic_engine
app.state.report_engine = report_engine
app.state.security_audit_log = security_audit_log
app.state.webhook_manager = webhook_manager
app.state.pipeline_manager = pipeline_manager
app.state.youtube_manager = youtube_manager
app.state.video_processor = video_processor
app.state.social_media_manager = social_media_manager
app.state.email_template_manager = email_template_manager
app.state.photo_doc_manager = photo_doc_manager
app.state.training_manager = training_manager
app.state.ride_log_manager = ride_log_manager
app.state.market_monitor = market_monitor
app.state.feedback_manager = feedback_manager
app.state.warranty_manager = warranty_manager
app.state.maintenance_scheduler = maintenance_scheduler
app.state.plugin_manager = plugin_manager
app.state.export_manager = export_manager
app.state.offline_manager = offline_manager
app.state.session_manager = session_manager
app.state.deployer = deployer
app.state.backup_manager = backup_manager
app.state.message_bus = message_bus
app.state.performance_monitor = performance_monitor
app.state.access_control = access_control
app.state.kill_switch = {"active": False}       # mutable container
app.state.activate_kill_switch = activate_kill_switch

# Wire up export manager with all data sources and register HTTP routes
export_manager.set_managers(
    crm=crm_store,
    service_records=service_store,
    invoices=invoice_manager,
    finances=finance_tracker,
    scout=scout_store,
    ride_log=ride_log_manager,
    content_pipeline=pipeline_manager,
)
export_manager.register_routes(app)

# Mount versioned REST API
from interfaces.api.routes import router as api_router
app.include_router(api_router)

# Mount static files (CSS, JS, images if any)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Authentication middleware
#
# When auth is enabled, all routes except the exempted prefixes require a
# valid cam_session cookie. Unauthenticated browser requests to non-API
# paths get the login page; AJAX/fetch requests get a 401 JSON response.
# ---------------------------------------------------------------------------

# Paths that bypass dashboard cookie auth
_AUTH_EXEMPT_PREFIXES = ("/auth/", "/api/v1/", "/ws/", "/static/")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Check cam_session cookie on dashboard routes when auth is enabled."""
    if not session_manager.auth_enabled:
        return await call_next(request)

    path = request.url.path

    # Exempt paths: auth endpoints, versioned API, agent WS, static files
    for prefix in _AUTH_EXEMPT_PREFIXES:
        if path.startswith(prefix):
            return await call_next(request)

    # The root path and /api/* (non-versioned dashboard endpoints) need auth
    token = request.cookies.get("cam_session")
    if session_manager.validate_session(token):
        return await call_next(request)

    # Not authenticated — for the root path, fall through to serve login.html
    # For API endpoints, return 401 JSON
    if path.startswith("/api/"):
        return JSONResponse(status_code=401, content={"error": "Not authenticated"})

    # For all other paths (including /), serve the login page
    return FileResponse(STATIC_DIR / "login.html")


# ---------------------------------------------------------------------------
# Auth routes — login, lock, status
# ---------------------------------------------------------------------------

@app.post("/auth/login")
async def auth_login(request: Request):
    """Authenticate and set a session cookie."""
    client_ip = request.client.host if request.client else "unknown"

    # Check lockout first
    remaining = session_manager.get_lockout_remaining(client_ip)
    if remaining > 0:
        event_logger.warn("auth", f"Login attempt from locked-out IP {client_ip}",
                          client_ip=client_ip, lockout_remaining=remaining)
        security_audit_log.log_action(
            action_type="login_attempt",
            actor=client_ip,
            target="dashboard",
            result="locked_out",
            risk_level="medium",
            tier=0,
        )
        return JSONResponse(
            status_code=429,
            content={"ok": False, "error": f"Too many attempts. Try again in {remaining}s."},
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Invalid request"})

    username = body.get("username", "")
    password = body.get("password", "")

    token = session_manager.login(username, password, client_ip=client_ip)

    if token is None:
        event_logger.warn("auth", f"Failed login from {client_ip} (user: {username})",
                          client_ip=client_ip, username=username)
        security_audit_log.log_action(
            action_type="login_attempt",
            actor=client_ip,
            target="dashboard",
            result="failed",
            risk_level="medium",
            tier=0,
        )
        return JSONResponse(
            status_code=401,
            content={"ok": False, "error": "Invalid credentials."},
        )

    # Success
    event_logger.info("auth", f"Login successful from {client_ip} (user: {username})",
                      client_ip=client_ip, username=username)
    security_audit_log.log_action(
        action_type="login",
        actor=username,
        target="dashboard",
        result="success",
        risk_level="low",
        tier=0,
    )

    response = JSONResponse(content={"ok": True})
    response.set_cookie(
        key="cam_session",
        value=token,
        httponly=True,
        samesite="strict",
        path="/",
    )
    return response


@app.post("/auth/lock")
async def auth_lock(request: Request):
    """Destroy the current session (lock screen)."""
    token = request.cookies.get("cam_session")
    session_manager.destroy_session(token)

    client_ip = request.client.host if request.client else "unknown"
    event_logger.info("auth", f"Session locked from {client_ip}", client_ip=client_ip)

    response = JSONResponse(content={"ok": True})
    response.delete_cookie("cam_session", path="/")
    return response


@app.get("/auth/status")
async def auth_status(request: Request):
    """Check if the current session is still valid."""
    if not session_manager.auth_enabled:
        default_user = getattr(getattr(config, 'auth', None), 'username', 'george')
        return {
            "authenticated": True,
            "auth_enabled": False,
            **access_control.get_role_info(default_user),
        }

    token = request.cookies.get("cam_session")
    valid = session_manager.validate_session(token)
    result = {"authenticated": valid, "auth_enabled": True}
    if valid:
        session = session_manager.get_session(token)
        if session:
            result.update(access_control.get_role_info(session.username))
    return result


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root(request: Request):
    """Serve the main dashboard page (or login page if auth is enabled and not authenticated)."""
    if session_manager.auth_enabled:
        token = request.cookies.get("cam_session")
        if not session_manager.validate_session(token):
            return FileResponse(STATIC_DIR / "login.html")
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/agents")
async def get_agents():
    """Return current status of all known agents.

    The browser fetches this on page load to get initial state,
    then switches to WebSocket for real-time updates.
    """
    return registry.to_broadcast_list()


@app.get("/api/tasks")
async def get_tasks():
    """Return current task queue state as JSON."""
    return {
        "tasks": task_queue.to_broadcast_list(),
        "counts": task_queue.get_status(),
    }


@app.get("/api/health")
async def get_health():
    """Return current agent health metrics as JSON."""
    return health_monitor.to_broadcast_dict()


@app.get("/api/analytics")
async def get_analytics():
    """Return aggregate analytics summary as JSON."""
    return analytics.get_summary()


@app.get("/api/events")
async def get_events(count: int = 200):
    """Return recent events from the event log."""
    return event_logger.get_recent(count)


@app.get("/api/memory")
async def get_memory():
    """Return current memory system status as JSON."""
    return {
        "short_term": short_term_memory.get_status(),
        "working": working_memory.get_status(),
        "long_term": long_term_memory.get_status(),
        "episodic": episodic_memory.get_status(),
    }


@app.get("/api/schedules")
async def get_schedules():
    """Return current scheduled tasks as JSON."""
    return {"schedules": scheduler.to_broadcast_list()}


@app.get("/api/content-calendar")
async def get_content_calendar():
    """Return current content calendar entries as JSON."""
    return {"entries": content_calendar.to_broadcast_list()}


@app.get("/api/research")
async def get_research():
    """Return current research results as JSON."""
    return {"entries": research_store.to_broadcast_list()}


@app.get("/api/scout")
async def get_scout():
    """Return current scout listings as JSON."""
    return {
        "entries": scout_store.to_broadcast_list(),
        "status": scout_store.get_status(),
    }


@app.post("/api/scout/scan")
async def trigger_scout_scan():
    """Trigger an immediate scout scan for motorcycle listings."""
    try:
        new_listings = await doppler_scout.scan()
        await broadcast_scout_status()
        return {
            "ok": True,
            "new_count": len(new_listings),
            "listings": [l.to_dict() for l in new_listings],
        }
    except Exception as e:
        logger.warning("Scout scan failed: %s", e)
        return {"ok": False, "error": str(e)}


@app.post("/api/scout/{entry_id}/status")
async def update_scout_status(entry_id: str, request: Request):
    """Update a scout listing's status."""
    body = await request.json()
    new_status = body.get("status", "")
    if new_status not in [s.value for s in ListingStatus]:
        return JSONResponse({"ok": False, "error": "Invalid status"}, status_code=400)
    updated = scout_store.update_entry(entry_id, status=new_status)
    if updated:
        await broadcast_scout_status()
        return {"ok": True, "entry": updated.to_dict()}
    return JSONResponse({"ok": False, "error": "Entry not found"}, status_code=404)


@app.post("/api/scout/{entry_id}/rescore")
async def rescore_scout_listing(entry_id: str):
    """Re-score a specific scout listing."""
    try:
        listing = await doppler_scout.score_single(entry_id)
        if listing:
            await broadcast_scout_status()
            return {"ok": True, "entry": listing.to_dict()}
        return JSONResponse({"ok": False, "error": "Entry not found"}, status_code=404)
    except Exception as e:
        logger.warning("Scout rescore failed: %s", e)
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Market Analytics REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/market")
async def get_market_analytics():
    """Return current market analytics summary as JSON."""
    return market_analyzer.get_summary()


@app.post("/api/market/snapshot")
async def take_market_snapshot():
    """Take a price snapshot from current scout listings."""
    try:
        count = market_analyzer.take_snapshot()
        await broadcast_market_analytics()
        return {"ok": True, "inserted": count}
    except Exception as e:
        logger.warning("Market snapshot failed: %s", e)
        return {"ok": False, "error": str(e)}


@app.post("/api/market/report")
async def generate_market_report(request: Request):
    """Generate a market analytics report via model router."""
    try:
        body = await request.json()
        report_type = body.get("report_type", "weekly")
        report = await market_analyzer.generate_report(report_type)
        await broadcast_market_analytics()
        return {"ok": True, **report}
    except Exception as e:
        logger.warning("Market report generation failed: %s", e)
        return {"ok": False, "error": str(e)}


@app.get("/api/market/report/{report_id}/download")
async def download_market_report(report_id: str):
    """Download a market report as a markdown file."""
    # Path traversal protection — report_id must be a valid UUID
    try:
        uuid_obj = __import__("uuid").UUID(report_id)
    except (ValueError, AttributeError):
        return JSONResponse(status_code=400, content={"error": "Invalid report ID"})

    md = market_analyzer.export_report_markdown(str(uuid_obj))
    if md is None:
        return JSONResponse(status_code=404, content={"error": "Report not found"})

    return Response(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="market-report-{report_id[:8]}.md"'},
    )


@app.get("/api/market/trends")
async def get_market_trends():
    """Return price trends for all tracked makes/models."""
    return {"trends": market_analyzer.get_all_trends(days=30)}


@app.get("/api/market/undervalued")
async def get_market_undervalued():
    """Return listings priced significantly below historical averages."""
    return {"undervalued": market_analyzer.detect_undervalued()}


# ---------------------------------------------------------------------------
# Service Records REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/service-report/{record_id}")
async def download_service_report(record_id: str):
    """Download a service record PDF report."""
    # Path traversal protection — record_id must be a valid UUID
    try:
        uuid_obj = __import__("uuid").UUID(record_id)
    except (ValueError, AttributeError):
        return JSONResponse(status_code=400, content={"error": "Invalid record ID"})

    safe_id = str(uuid_obj)
    pdf_path = Path("data/service_reports") / f"{safe_id}.pdf"

    if not pdf_path.exists():
        # Try generating it on the fly
        result = service_store.generate_pdf(safe_id)
        if result is None:
            return JSONResponse(status_code=404, content={"error": "Record not found"})

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"doppler-service-report-{safe_id[:8]}.pdf",
    )


@app.get("/api/invoice-pdf/{invoice_id}")
async def download_invoice_pdf(invoice_id: str):
    """Download an invoice PDF."""
    try:
        uuid_obj = __import__("uuid").UUID(invoice_id)
    except (ValueError, AttributeError):
        return JSONResponse(status_code=400, content={"error": "Invalid invoice ID"})

    safe_id = str(uuid_obj)
    pdf_path = Path("data/invoices") / f"{safe_id}.pdf"

    if not pdf_path.exists():
        result = invoice_manager.generate_pdf(safe_id)
        if result is None:
            return JSONResponse(status_code=404, content={"error": "Invoice not found"})

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"doppler-invoice-{safe_id[:8]}.pdf",
    )


@app.get("/api/report-pdf/{report_id}")
async def download_report_pdf(report_id: str):
    """Download a scheduled report PDF."""
    # Path traversal protection — report_id must be a valid UUID
    try:
        uuid_obj = __import__("uuid").UUID(report_id)
    except (ValueError, AttributeError):
        return JSONResponse(status_code=400, content={"error": "Invalid report ID"})

    safe_id = str(uuid_obj)
    pdf_dir = getattr(config, "reports", None) and config.reports.pdf_dir or "data/reports"
    pdf_path = Path(pdf_dir) / f"{safe_id}.pdf"

    if not pdf_path.exists():
        return JSONResponse(status_code=404, content={"error": "Report PDF not found"})

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"doppler-report-{safe_id[:8]}.pdf",
    )


# ---------------------------------------------------------------------------
# Photo Documentation endpoints
# ---------------------------------------------------------------------------

@app.get("/api/photo/{photo_id}")
async def serve_photo(photo_id: str):
    """Serve original photo file."""
    try:
        uuid_obj = __import__("uuid").UUID(photo_id)
    except (ValueError, AttributeError):
        return JSONResponse(status_code=400, content={"error": "Invalid photo ID"})

    photo = photo_doc_manager.get_photo(str(uuid_obj))
    if not photo or not photo.original_path:
        return JSONResponse(status_code=404, content={"error": "Photo not found"})

    photo_path = Path(photo.original_path)
    if not photo_path.exists():
        return JSONResponse(status_code=404, content={"error": "Photo file not found"})

    # Determine media type from extension
    ext = photo_path.suffix.lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    media_type = media_types.get(ext, "image/jpeg")

    return FileResponse(str(photo_path), media_type=media_type)


@app.get("/api/photo/{photo_id}/thumb")
async def serve_photo_thumb(photo_id: str):
    """Serve photo thumbnail."""
    try:
        uuid_obj = __import__("uuid").UUID(photo_id)
    except (ValueError, AttributeError):
        return JSONResponse(status_code=400, content={"error": "Invalid photo ID"})

    photo = photo_doc_manager.get_photo(str(uuid_obj))
    if not photo or not photo.thumbnail_path:
        return JSONResponse(status_code=404, content={"error": "Thumbnail not found"})

    thumb_path = Path(photo.thumbnail_path)
    if not thumb_path.exists():
        return JSONResponse(status_code=404, content={"error": "Thumbnail file not found"})

    return FileResponse(str(thumb_path), media_type="image/jpeg")


@app.get("/api/photo/composite/{service_record_id}")
async def serve_composite(service_record_id: str):
    """Serve before/after composite image."""
    try:
        uuid_obj = __import__("uuid").UUID(service_record_id)
    except (ValueError, AttributeError):
        return JSONResponse(status_code=400, content={"error": "Invalid service record ID"})

    safe_id = str(uuid_obj)
    composite_path = Path("data/photos/composites") / f"{safe_id}.jpg"
    if not composite_path.exists():
        return JSONResponse(status_code=404, content={"error": "Composite not found"})

    return FileResponse(str(composite_path), media_type="image/jpeg")


@app.get("/api/photo/gallery/{service_record_id}")
async def serve_gallery(service_record_id: str):
    """Serve photo gallery HTML."""
    try:
        uuid_obj = __import__("uuid").UUID(service_record_id)
    except (ValueError, AttributeError):
        return JSONResponse(status_code=400, content={"error": "Invalid service record ID"})

    safe_id = str(uuid_obj)
    gallery_path = Path("data/photos/galleries") / f"{safe_id}.html"
    if not gallery_path.exists():
        return JSONResponse(status_code=404, content={"error": "Gallery not found"})

    return FileResponse(str(gallery_path), media_type="text/html")


# ---------------------------------------------------------------------------
# Webhook REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/webhooks")
async def get_webhooks():
    """Return webhook status and endpoints."""
    return webhook_manager.to_broadcast_dict()


@app.get("/api/webhooks/endpoints")
async def list_webhook_endpoints(direction: str | None = None):
    """List registered webhook endpoints."""
    return {"endpoints": webhook_manager.list_endpoints(direction=direction)}


@app.post("/api/webhooks/endpoints")
async def create_webhook_endpoint(request: Request):
    """Register a new webhook endpoint."""
    try:
        body = await request.json()
        ep = webhook_manager.register_endpoint(
            name=body.get("name", "Unnamed"),
            url=body.get("url", ""),
            direction=body.get("direction", "outbound"),
            secret=body.get("secret", ""),
            event_filters=body.get("event_filters", []),
            severity_filter=body.get("severity_filter", "all"),
            enabled=body.get("enabled", True),
        )
        await broadcast_webhook_status()
        return {"ok": True, "endpoint": ep}
    except Exception as e:
        logger.warning("Webhook endpoint create failed: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.delete("/api/webhooks/endpoints/{endpoint_id}")
async def delete_webhook_endpoint(endpoint_id: str):
    """Delete a webhook endpoint."""
    deleted = webhook_manager.delete_endpoint(endpoint_id)
    if deleted:
        await broadcast_webhook_status()
        return {"ok": True}
    return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)


@app.get("/api/webhooks/deliveries")
async def get_webhook_deliveries(limit: int = 30, endpoint_id: str | None = None):
    """Return recent webhook delivery history."""
    return {
        "deliveries": webhook_manager.get_recent_deliveries(
            limit=limit, endpoint_id=endpoint_id
        )
    }


@app.post("/api/webhooks/incoming")
async def webhook_incoming(request: Request):
    """Inbound webhook endpoint — no API key required.

    External services POST here to create tasks in CAM.
    Validates X-Webhook-Signature if inbound_secret is configured.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)

    source_ip = request.client.host if request.client else ""
    signature = request.headers.get("X-Webhook-Signature", "")

    result = await webhook_manager.process_inbound(
        payload=body, source_ip=source_ip, provided_signature=signature
    )
    if not result.get("ok"):
        return JSONResponse(result, status_code=401)
    return result


@app.get("/api/business")
async def get_business():
    """Return current business data as JSON."""
    return {
        **business_store.to_broadcast_list(),
        "status": business_store.get_status(),
    }


@app.get("/api/security-audit")
async def get_security_audit():
    """Return recent security audit entries as JSON."""
    return {
        "entries": security_audit_log.to_broadcast_list(),
        "status": security_audit_log.get_status(),
    }


@app.get("/api/deploy/status")
async def get_deploy_status():
    """Return deployer status: local version, agent versions, recent history."""
    return deployer.get_status()


@app.get("/api/backup/status")
async def get_backup_status():
    """Return backup system status: last backup, count, total size, list."""
    return backup_manager.get_status()


@app.get("/api/config")
async def get_config_endpoint():
    """Return current configuration as JSON."""
    return {"config": config.to_dict(), "last_loaded": config.last_loaded}


@app.get("/api/tts/status")
async def get_tts_status():
    """Return current TTS pipeline status as JSON."""
    return tts_pipeline.get_status()


@app.get("/api/tts/audio")
async def get_tts_audio_list():
    """Return list of generated audio files with metadata."""
    return {"audio": tts_pipeline.list_audio()}


@app.get("/api/tts/audio/{filename}")
async def get_tts_audio_file(filename: str):
    """Serve a WAV audio file for browser playback.

    Includes path traversal protection — filename must resolve
    to a path within the audio directory.
    """
    audio_dir = Path(tts_pipeline._audio_dir).resolve()
    file_path = (audio_dir / filename).resolve()

    # Path traversal protection
    if not str(file_path).startswith(str(audio_dir)):
        return JSONResponse(status_code=403, content={"error": "Access denied"})

    if not file_path.exists() or not file_path.suffix == ".wav":
        return JSONResponse(status_code=404, content={"error": "File not found"})

    return FileResponse(file_path, media_type="audio/wav", filename=filename)


@app.post("/api/events/export")
async def export_events():
    """Export all events to a JSON file in data/logs/."""
    from datetime import datetime as dt
    filepath = Path("data/logs") / f"events_{dt.now().strftime('%Y%m%d_%H%M%S')}.json"
    num = event_logger.export_json(filepath)
    return {"ok": True, "filepath": str(filepath), "event_count": num}


@app.post("/api/file/upload")
async def upload_file(
    file: UploadFile = File(...),
    agent_id: str = Form(...),
    dest_path: str = Form(""),
):
    """Upload a file and relay it to an agent over WebSocket.

    Accepts multipart form data with:
        file      — the file to send
        agent_id  — target agent's ID
        dest_path — destination path on the agent (optional)
    """
    # Validate agent is online
    if agent_id not in agent_websockets:
        return {"ok": False, "error": "Agent is offline"}

    agent_info = registry.get_by_id(agent_id)
    agent_name = agent_info.name if agent_info else agent_id

    # Read file data
    data = await file.read()
    file_size = len(data)
    max_size = config.file_transfer.max_file_size

    if file_size > max_size:
        return {"ok": False, "error": f"File too large ({file_size} bytes, max {max_size})"}
    if file_size == 0:
        return {"ok": False, "error": "Empty file"}

    filename = file.filename or "upload"
    checksum = compute_checksum(data)
    chunk_size = config.file_transfer.chunk_size
    chunk_count = (file_size + chunk_size - 1) // chunk_size

    try:
        transfer = ft_manager.create_transfer(
            direction="to_agent",
            agent_id=agent_id,
            agent_name=agent_name,
            filename=filename,
            file_size=file_size,
            chunk_count=chunk_count,
            checksum=checksum,
            dest_path=dest_path,
        )
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    event_logger.info(
        "file_transfer",
        f"Uploading {filename} ({file_size} bytes) to {agent_name}",
        transfer_id=transfer.transfer_id, agent=agent_name, filename=filename,
    )

    # Launch relay as background task
    agent_ws = agent_websockets[agent_id]
    asyncio.create_task(relay_file_to_agent(transfer, data, agent_ws))

    return {
        "ok": True,
        "transfer_id": transfer.transfer_id,
        "filename": filename,
        "file_size": file_size,
        "chunk_count": chunk_count,
    }


@app.post("/api/knowledge/upload")
async def knowledge_upload(file: UploadFile = File(...)):
    """Upload a document for knowledge ingestion.

    Accepts any supported file type (.md, .txt, .pdf, .csv).
    Chunks the document and stores in long-term memory.
    """
    data = await file.read()
    filename = file.filename or "upload"

    doc = knowledge_ingest.ingest_bytes(data, filename, source="upload")

    event_logger.info(
        "knowledge",
        f"Knowledge upload: {filename} → {doc.status} ({doc.chunk_count} chunks)",
        filename=filename, status=doc.status, chunks=doc.chunk_count,
    )

    await broadcast_knowledge_status()

    return {
        "ok": doc.status == "completed",
        "document": doc.to_dict(),
    }


# ---------------------------------------------------------------------------
# WebSocket: agents connect here to send heartbeats
# ---------------------------------------------------------------------------

@app.websocket("/ws/agent/{agent_id}")
async def agent_websocket(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for agents to connect and send heartbeats.

    Agents open a connection to /ws/agent/{their_id} and send JSON:
        {"type": "heartbeat", "name": "AgentName", "ip": "192.168.1.x",
         "capabilities": ["research", "content"]}

    On connect: agent is registered in the registry.
    On heartbeat: last_heartbeat timestamp updated.
    On disconnect: agent marked offline in the registry.
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info("Agent '%s' WebSocket opened from %s", agent_id, client_host)

    # Store the WebSocket so we can send commands back (e.g., kill switch)
    agent_websockets[agent_id] = websocket

    # Start health tracking for this connection session
    health_monitor.on_agent_connected(agent_id)
    plugin_manager.on_agent_connected(agent_id)

    try:
        while True:
            data = await websocket.receive_json()

            msg_type = data.get("type")

            if msg_type == "heartbeat":
                name = data.get("name", agent_id)
                ip = data.get("ip", client_host)
                capabilities = data.get("capabilities")

                if registry.get_by_id(agent_id) is None:
                    # First heartbeat — register in the registry
                    registry.register(
                        agent_id=agent_id,
                        name=name,
                        ip_address=ip,
                        capabilities=capabilities,
                    )
                    event_logger.info("agent", f"Agent '{name}' connected from {ip}",
                                      agent_id=agent_id, ip=ip)
                else:
                    # Subsequent heartbeat — update via registry
                    registry.update_heartbeat(
                        agent_id=agent_id,
                        ip_address=ip,
                        capabilities=capabilities,
                    )

                health_monitor.on_heartbeat(agent_id)
                await broadcast_agent_status()

            elif msg_type == "command_response":
                response_text = data.get("response", "")
                task_id = data.get("task_id")

                logger.info(
                    "Agent '%s' response (task_id=%s): %s",
                    agent_id, task_id, response_text,
                )

                # If this response is for a dispatched task, resolve the Future
                if task_id and task_id in pending_task_responses:
                    future = pending_task_responses[task_id]
                    if not future.done():
                        future.set_result(response_text)
                else:
                    # Manual command from dashboard — relay as before
                    await broadcast_to_dashboards({
                        "type": "command_response",
                        "agent_id": agent_id,
                        "command": data.get("command", ""),
                        "response": response_text,
                    })

            elif msg_type == "pong":
                ping_id = data.get("ping_id")
                if ping_id:
                    rtt_ms = health_monitor.resolve_pong(ping_id)
                    await broadcast_to_dashboards({
                        "type": "ping_result",
                        "agent_id": agent_id,
                        "rtt_ms": round(rtt_ms, 1) if rtt_ms is not None else None,
                        "error": None if rtt_ms is not None else "Unknown ping_id",
                    })

            # --- File transfer: agent acking our chunks ---
            elif msg_type == "file_chunk_ack":
                transfer_id = data.get("transfer_id", "")
                chunk_index = data.get("chunk_index", 0)
                ok = data.get("ok", True)

                if not ok:
                    error = data.get("error", "Agent rejected chunk")
                    await ft_manager.fail_transfer(transfer_id, error)
                else:
                    transfer = ft_manager.get_transfer(transfer_id)
                    if transfer:
                        await ft_manager.update_progress(transfer_id, chunk_index + 1)

            elif msg_type == "file_transfer_complete":
                transfer_id = data.get("transfer_id", "")
                ok = data.get("ok", True)

                if ok:
                    await ft_manager.complete_transfer(transfer_id)
                    event_logger.info(
                        "file_transfer",
                        f"Transfer {transfer_id} completed successfully",
                        transfer_id=transfer_id, agent_id=agent_id,
                    )
                else:
                    error = data.get("error", "Agent reported failure")
                    await ft_manager.fail_transfer(transfer_id, error)
                    event_logger.error(
                        "file_transfer",
                        f"Transfer {transfer_id} failed: {error}",
                        transfer_id=transfer_id, agent_id=agent_id,
                    )

            # --- File transfer: agent sending a file to us ---
            elif msg_type == "file_send_start":
                transfer_id = data.get("transfer_id", "")
                ok = data.get("ok", True)

                if not ok:
                    error = data.get("error", "Agent could not read file")
                    await ft_manager.fail_transfer(transfer_id, error)
                else:
                    filename = data.get("filename", "")
                    file_size = data.get("file_size", 0)
                    chunk_count = data.get("chunk_count", 0)
                    checksum = data.get("checksum", "")
                    agent_info = registry.get_by_id(agent_id)
                    agent_name = agent_info.name if agent_info else agent_id

                    transfer = ft_manager.get_transfer(transfer_id)
                    if transfer:
                        # Update the existing transfer with file metadata
                        transfer.filename = filename
                        transfer.file_size = file_size
                        transfer.chunk_count = chunk_count
                        transfer.checksum = checksum
                    else:
                        transfer = ft_manager.create_transfer(
                            direction="from_agent",
                            agent_id=agent_id,
                            agent_name=agent_name,
                            filename=filename,
                            file_size=file_size,
                            chunk_count=chunk_count,
                            checksum=checksum,
                        )
                    ft_manager.init_receive_buffer(transfer.transfer_id)

            elif msg_type == "file_send_chunk":
                transfer_id = data.get("transfer_id", "")
                chunk_index = data.get("chunk_index", 0)
                chunk_count = data.get("chunk_count", 0)
                b64_data = data.get("data", "")

                transfer = ft_manager.get_transfer(transfer_id)
                if not transfer:
                    continue

                try:
                    raw = base64.b64decode(b64_data)
                    ft_manager.store_chunk(transfer_id, chunk_index, raw)
                    await ft_manager.update_progress(transfer_id, chunk_index + 1)

                    # Check if all chunks received
                    if chunk_index + 1 >= chunk_count:
                        assembled = ft_manager.assemble_chunks(transfer_id, chunk_count)
                        if assembled is not None:
                            # Verify checksum
                            actual_checksum = compute_checksum(assembled)
                            if transfer.checksum and actual_checksum != transfer.checksum:
                                await ft_manager.fail_transfer(
                                    transfer_id,
                                    f"Checksum mismatch: expected {transfer.checksum}, got {actual_checksum}",
                                )
                                continue

                            # Write to receive directory
                            receive_dir = Path(config.file_transfer.receive_dir)
                            receive_dir.mkdir(parents=True, exist_ok=True)
                            dest = receive_dir / transfer.filename
                            dest.write_bytes(assembled)
                            transfer.dest_path = str(dest)
                            await ft_manager.complete_transfer(transfer_id)
                            event_logger.info(
                                "file_transfer",
                                f"Received {transfer.filename} from {agent_id} → {dest}",
                                transfer_id=transfer_id, agent_id=agent_id,
                                filename=transfer.filename,
                            )

                except Exception as e:
                    logger.warning("Error receiving chunk %d of %s: %s", chunk_index, transfer_id, e)
                    await ft_manager.fail_transfer(transfer_id, str(e))

            # --- Message Bus: remote agent publishing ---
            elif msg_type == "bus_publish":
                channel = data.get("channel", "")
                payload = data.get("payload", {})
                result = await message_bus.publish(agent_id, channel, payload)
                await websocket.send_json({
                    "type": "bus_publish_result",
                    "ok": result is not None,
                    "message_id": result.message_id if result else None,
                })

    except WebSocketDisconnect:
        logger.info("Agent '%s' disconnected", agent_id)
        event_logger.warn("agent", f"Agent '{agent_id}' disconnected", agent_id=agent_id)
        # Remove from live WebSocket tracking
        agent_websockets.pop(agent_id, None)
        # Accumulate uptime in health monitor before marking offline
        health_monitor.on_agent_disconnected(agent_id)
        plugin_manager.on_agent_disconnected(agent_id)
        # Mark offline in registry (keeps the record for dashboard display)
        registry.deregister(agent_id)
        await broadcast_agent_status()


# ---------------------------------------------------------------------------
# Helper: extract username from WS session cookie
# ---------------------------------------------------------------------------

def get_ws_username(websocket: WebSocket) -> str:
    """Get the authenticated username from a WebSocket's session cookie.

    Returns the default admin username if auth is disabled or session
    lookup fails (backward compatible — single-user mode).
    """
    if not session_manager.auth_enabled:
        return getattr(getattr(config, 'auth', None), 'username', 'george')
    token = websocket.cookies.get("cam_session")
    session = session_manager.get_session(token)
    if session:
        return session.username
    return getattr(getattr(config, 'auth', None), 'username', 'george')


# ---------------------------------------------------------------------------
# WebSocket: browser clients connect here to watch agent status
# ---------------------------------------------------------------------------

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for dashboard browser clients.

    Browsers connect here to receive real-time agent status updates.
    On connect, they immediately get the current state of all agents.
    After that, updates are pushed whenever agent status changes.

    When auth is enabled, the cam_session cookie must be valid or the
    connection is rejected with close code 4401.
    """
    # Validate session cookie before accepting the WebSocket
    if session_manager.auth_enabled:
        token = websocket.cookies.get("cam_session")
        if not session_manager.validate_session(token):
            await websocket.close(code=4401, reason="Not authenticated")
            return

    global last_self_test_results, last_launch_readiness_results

    await websocket.accept()

    # Identify the connected user for permission enforcement
    ws_username = get_ws_username(websocket)
    ws_role = access_control.get_user_role(ws_username)
    role_str = ws_role.value if ws_role else "viewer"
    dashboard_clients[websocket] = {
        "mode": "full",
        "username": ws_username,
        "role": role_str,
    }
    logger.info(
        "Dashboard client connected from %s (user=%s, role=%s)",
        websocket.client.host if websocket.client else "unknown",
        ws_username, role_str,
    )

    # Send role info so the client can enforce panel visibility
    await websocket.send_json({
        "type": "role_info",
        **access_control.get_role_info(ws_username),
    })

    # Send user list to admins for the user management panel
    if ws_role == Role.ADMIN:
        await websocket.send_json({
            "type": "user_list",
            "users": access_control.to_broadcast_list(),
        })

    # Send current state right away so the page doesn't start blank
    await websocket.send_json({
        "type": "agent_status",
        "agents": registry.to_broadcast_list(),
    })

    # Also send current kill switch state
    await websocket.send_json({"type": "kill_switch", "active": kill_switch_active})

    # Send current task queue state
    await websocket.send_json({
        "type": "task_status",
        "tasks": task_queue.to_broadcast_list(),
        "counts": task_queue.get_status(),
    })

    # Send current health metrics
    await websocket.send_json({
        "type": "health_status",
        "health": health_monitor.to_broadcast_dict(),
    })

    # Send recent event log so the panel isn't empty on connect
    await websocket.send_json({
        "type": "event_log",
        "events": event_logger.get_recent(config.dashboard.default_event_count),
    })

    # Send current config so the settings panel renders immediately
    await websocket.send_json({
        "type": "config",
        "config": config.to_dict(),
        "last_loaded": config.last_loaded,
    })

    # Send boot time so the header can show uptime
    await websocket.send_json({
        "type": "boot_info",
        "boot_time": boot_time,
        "boot_duration": boot_duration,
    })

    # Send current analytics summary
    await websocket.send_json({"type": "analytics", "data": analytics.get_summary()})

    # Send command library for the command palette
    await websocket.send_json({
        "type": "command_library",
        "commands": command_library.to_broadcast_list(),
    })

    # Send current chain state
    await websocket.send_json({
        "type": "chain_status",
        "chains": task_queue.chains_to_broadcast_list(),
    })

    # Send current swarm state
    await websocket.send_json({
        "type": "swarm_status",
        "swarms": task_queue.swarms_to_broadcast_list(),
    })

    # Send notification history so the bell icon shows correct state
    await websocket.send_json({
        "type": "notification_history",
        "notifications": notification_manager.get_recent(50),
        "unread_count": notification_manager.get_unread_count(),
    })

    # Send file transfer state (active + history)
    await websocket.send_json({
        "type": "file_transfer_history",
        "transfers": ft_manager.get_all_transfers(),
    })

    # Send deploy status (local version, agent versions, history)
    await websocket.send_json({
        "type": "deploy_status",
        **deployer.get_status(),
    })

    # Send backup status
    await websocket.send_json({
        "type": "backup_status",
        **backup_manager.get_status(),
    })

    # Send message bus state
    await websocket.send_json({
        "type": "bus_status",
        **message_bus.to_broadcast_dict(),
    })

    # Send current schedule state
    await websocket.send_json({
        "type": "schedule_status",
        "schedules": scheduler.to_broadcast_list(),
    })

    # Send current content calendar state
    await websocket.send_json({
        "type": "content_calendar_status",
        "entries": content_calendar.to_broadcast_list(),
    })

    # Send current content pipeline state
    await websocket.send_json({
        "type": "pipeline_status",
        **pipeline_manager.to_broadcast_dict(),
    })

    # Send current YouTube publishing state
    await websocket.send_json({
        "type": "youtube_status",
        **youtube_manager.to_broadcast_dict(),
    })

    # Send current media processing state
    await websocket.send_json({
        "type": "media_status",
        **video_processor.to_broadcast_dict(),
    })

    # Send current social media posts
    await websocket.send_json({
        "type": "social_media_status",
        **social_media_manager.to_broadcast_dict(),
    })

    # Send current email template status
    await websocket.send_json({
        "type": "email_status",
        **email_template_manager.to_broadcast_dict(),
    })

    # Send current photo documentation status
    await websocket.send_json({
        "type": "photo_docs_status",
        **photo_doc_manager.to_broadcast_dict(),
    })

    # Send current training/learning status
    await websocket.send_json({
        "type": "training_status",
        **training_manager.to_broadcast_dict(),
    })

    # Send current ride log status
    await websocket.send_json({
        "type": "ride_log_status",
        **ride_log_manager.to_broadcast_dict(),
    })

    # Send current market monitor status
    await websocket.send_json({
        "type": "market_monitor_status",
        **market_monitor.to_broadcast_dict(),
    })

    # Send current feedback status
    await websocket.send_json({
        "type": "feedback_status",
        **feedback_manager.to_broadcast_dict(),
    })

    # Send current warranty & recall status
    await websocket.send_json({
        "type": "warranty_status",
        **warranty_manager.to_broadcast_dict(),
    })

    # Send current maintenance scheduler status
    await websocket.send_json({
        "type": "maintenance_status",
        **maintenance_scheduler.to_broadcast_dict(),
    })

    # Send current plugin manager status
    await websocket.send_json({
        "type": "plugin_status",
        **plugin_manager.to_broadcast_dict(),
    })

    # Send current export status
    await websocket.send_json({
        "type": "export_status",
        **export_manager.to_broadcast_dict(),
    })

    # Send current offline/connectivity status
    await websocket.send_json({
        "type": "offline_status",
        **offline_manager.to_broadcast_dict(),
    })

    # Send current performance metrics
    await websocket.send_json({
        "type": "performance_status",
        **performance_monitor.to_broadcast_dict(),
    })

    # Send current research results
    await websocket.send_json({
        "type": "research_status",
        "entries": research_store.to_broadcast_list(),
    })

    # Send current scout listings
    await websocket.send_json({
        "type": "scout_status",
        "entries": scout_store.to_broadcast_list(),
        "status": scout_store.get_status(),
    })

    # Send current market analytics
    await websocket.send_json({
        "type": "market_analytics_status",
        **market_analyzer.get_summary(),
    })

    # Send current business data
    await websocket.send_json({
        "type": "business_status",
        **business_store.to_broadcast_list(),
        "status": business_store.get_status(),
    })

    # Send current service records data
    await websocket.send_json({
        "type": "service_records_status",
        **service_store.to_broadcast_dict(),
    })

    # Send current CRM data
    await websocket.send_json({
        "type": "crm_status",
        **crm_store.to_broadcast_dict(),
    })

    # Send current appointment schedule
    await websocket.send_json({
        "type": "appointment_schedule_status",
        **appointment_scheduler.to_broadcast_dict(),
    })

    # Send current invoicing data
    await websocket.send_json({
        "type": "invoicing_status",
        **invoice_manager.to_broadcast_dict(),
    })

    # Send current parts inventory data
    await websocket.send_json({
        "type": "inventory_status",
        **inventory_manager.to_broadcast_dict(),
    })

    # Send current financial dashboard data
    await websocket.send_json({
        "type": "finances_status",
        **finance_tracker.to_broadcast_dict(current_balance=_finance_state["balance"]),
    })

    # Send current route planner state
    await websocket.send_json({
        "type": "route_status",
        **route_planner.to_broadcast_dict(),
    })

    # Send current Highway 20 documentary planner state
    await websocket.send_json({
        "type": "hwy20_status",
        **highway20_planner.to_broadcast_dict(),
    })

    # Send current diagnostic engine state
    await websocket.send_json({
        "type": "diagnostics_status",
        **diagnostic_engine.to_broadcast_dict(),
    })

    # Send current reports state
    await websocket.send_json({
        "type": "reports_status",
        **report_engine.to_broadcast_dict(),
    })

    # Send current knowledge ingestion state
    await websocket.send_json({
        "type": "knowledge_status",
        **knowledge_ingest.to_broadcast_dict(),
    })

    # Send current security audit state
    await websocket.send_json({
        "type": "security_audit_status",
        "entries": security_audit_log.to_broadcast_list(),
        "status": security_audit_log.get_status(),
    })

    # Send current webhook state
    await websocket.send_json({
        "type": "webhook_status",
        **webhook_manager.to_broadcast_dict(),
    })

    # Send current memory system status
    await websocket.send_json({
        "type": "memory_status",
        "short_term": short_term_memory.get_status(),
        "working": working_memory.get_status(),
        "long_term": long_term_memory.get_status(),
        "episodic": episodic_memory.get_status(),
    })

    # Send current context window status
    await websocket.send_json({
        "type": "context_status",
        **context_manager.get_status(),
    })

    # Send TTS pipeline status
    await websocket.send_json({
        "type": "tts_status",
        **tts_pipeline.get_status(),
    })

    # Send TTS audio list
    await websocket.send_json({
        "type": "tts_audio_list",
        "audio": tts_pipeline.list_audio(),
    })

    # Send TTS voice list
    await websocket.send_json({
        "type": "tts_voices",
        "voices": tts_pipeline.list_voices(),
    })

    # Send persona data for the persona preview panel
    await websocket.send_json({"type": "persona", "persona": persona.to_dict()})

    # Send Telegram bot connection status
    if telegram_bot:
        await websocket.send_json({
            "type": "telegram_status",
            "connected": telegram_bot.is_connected,
        })

    # Send cached launch readiness results if available
    if last_launch_readiness_results:
        await websocket.send_json({
            "type": "launch_readiness_results",
            **last_launch_readiness_results,
        })

    try:
        # Listen for dashboard commands (kill switch, task submit, agent controls)
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Permission gate — check role before handling any message
            msg_type = msg.get("type", "")
            if msg_type and not access_control.check_ws_permission(ws_username, msg_type):
                required = WS_PERMISSIONS.get(msg_type, Role.ADMIN)
                await websocket.send_json({
                    "type": "permission_denied",
                    "denied_action": msg_type,
                    "required_role": required.value if isinstance(required, Role) else required,
                    "your_role": role_str,
                })
                security_audit_log.log_action(
                    action_type="permission_denied",
                    actor=ws_username,
                    target=msg_type,
                    result="blocked",
                    risk_level="medium",
                )
                continue

            if msg.get("type") == "kill_switch":
                await activate_kill_switch()

            elif msg.get("type") == "command":
                # Route a command to a specific agent
                target_id = msg.get("agent_id", "")
                command_text = msg.get("command", "")

                if target_id in agent_websockets:
                    try:
                        await agent_websockets[target_id].send_json({
                            "type": "command",
                            "command": command_text,
                        })
                        logger.info(
                            "Command sent to agent '%s': %s",
                            target_id, command_text,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to send command to agent '%s'",
                            target_id,
                        )
                        await broadcast_to_dashboards({
                            "type": "command_response",
                            "agent_id": target_id,
                            "command": command_text,
                            "response": "[error] Agent connection lost",
                        })
                else:
                    logger.warning(
                        "Command for unknown/offline agent '%s'",
                        target_id,
                    )
                    await broadcast_to_dashboards({
                        "type": "command_response",
                        "agent_id": target_id,
                        "command": command_text,
                        "response": "[error] Agent is offline",
                    })

            elif msg.get("type") == "task_submit":
                # Submit a new task to the queue
                description = (msg.get("description") or "").strip()
                complexity_str = (msg.get("complexity") or "auto").lower()
                raw_caps = msg.get("required_capabilities") or []
                required_capabilities = [c.strip() for c in raw_caps if isinstance(c, str) and c.strip()]

                if not description:
                    await websocket.send_json({
                        "type": "task_submitted",
                        "ok": False,
                        "error": "Description is required",
                    })
                    continue

                try:
                    complexity = TaskComplexity(complexity_str)
                except ValueError:
                    complexity = TaskComplexity.AUTO

                task = task_queue.add_task(
                    description=description,
                    source="dashboard",
                    complexity=complexity,
                    required_capabilities=required_capabilities,
                )
                logger.info(
                    "Task %s submitted from dashboard (caps=%s): %s",
                    task.short_id, required_capabilities or "none", description,
                )
                event_logger.info("task", f"Task {task.short_id} submitted: {description[:80]}",
                                  task_id=task.short_id, complexity=complexity.value,
                                  required_capabilities=required_capabilities or None,
                                  source="dashboard")

                await websocket.send_json({
                    "type": "task_submitted",
                    "ok": True,
                    "task_id": task.task_id,
                    "short_id": task.short_id,
                })
                await broadcast_task_status()

            elif msg.get("type") == "task_list":
                # Dashboard requesting a full task list refresh
                await websocket.send_json({
                    "type": "task_status",
                    "tasks": task_queue.to_broadcast_list(),
                    "counts": task_queue.get_status(),
                })

            elif msg.get("type") == "ping_agent":
                target_id = msg.get("agent_id", "")
                if target_id in agent_websockets:
                    try:
                        ping_msg = health_monitor.create_ping(target_id)
                        await agent_websockets[target_id].send_json(ping_msg)
                        logger.info("Ping sent to agent '%s'", target_id)
                    except Exception:
                        logger.warning("Failed to ping agent '%s'", target_id)
                        await broadcast_to_dashboards({
                            "type": "ping_result",
                            "agent_id": target_id,
                            "rtt_ms": None,
                            "error": "Failed to send ping",
                        })
                else:
                    await broadcast_to_dashboards({
                        "type": "ping_result",
                        "agent_id": target_id,
                        "rtt_ms": None,
                        "error": "Agent is offline",
                    })

            elif msg.get("type") == "command_execute":
                # Execute a predefined command from the command palette
                command_name = msg.get("command_name", "")
                target_id = msg.get("agent_id", "")
                params = msg.get("params", {})

                cmd = command_library.get(command_name)
                if not cmd:
                    await websocket.send_json({
                        "type": "command_execute_result",
                        "ok": False,
                        "error": f"Unknown command: {command_name}",
                    })
                    continue

                # Build command string from name + params
                param_parts = [command_name]
                for p in cmd.parameters:
                    value = params.get(p["name"])
                    if value is None and p.get("default") is not None:
                        value = p["default"]
                    if value is not None:
                        param_parts.append(f"{p['name']}={value}")
                command_text = " ".join(param_parts)

                if target_id not in agent_websockets:
                    await websocket.send_json({
                        "type": "command_execute_result",
                        "ok": False,
                        "error": "Agent is offline",
                        "agent_id": target_id,
                        "command_name": command_name,
                    })
                    event_logger.warn(
                        "command",
                        f"Command '{command_name}' failed: agent '{target_id}' offline",
                        command=command_name, agent_id=target_id,
                    )
                    continue

                try:
                    await agent_websockets[target_id].send_json({
                        "type": "command",
                        "command": command_text,
                    })
                    logger.info(
                        "Command palette: sent '%s' to agent '%s'",
                        command_text, target_id,
                    )
                    event_logger.info(
                        "command",
                        f"Executed '{command_name}' on agent '{target_id}'",
                        command=command_name, agent_id=target_id,
                        params=params or None,
                    )
                    await websocket.send_json({
                        "type": "command_execute_result",
                        "ok": True,
                        "agent_id": target_id,
                        "command_name": command_name,
                        "command_text": command_text,
                    })
                except Exception:
                    logger.warning(
                        "Failed to send command '%s' to agent '%s'",
                        command_name, target_id,
                    )
                    await websocket.send_json({
                        "type": "command_execute_result",
                        "ok": False,
                        "error": "Agent connection lost",
                        "agent_id": target_id,
                        "command_name": command_name,
                    })

            elif msg.get("type") == "command_list":
                # Dashboard requesting a command library refresh
                await websocket.send_json({
                    "type": "command_library",
                    "commands": command_library.to_broadcast_list(),
                })

            elif msg.get("type") == "config_reload":
                # Reload config from disk and broadcast to all dashboards
                changes = config.reload()
                event_logger.info(
                    "system",
                    f"Configuration reloaded ({len(changes)} change(s))",
                    changes=changes if changes else None,
                )
                await broadcast_to_dashboards({
                    "type": "config",
                    "config": config.to_dict(),
                    "last_loaded": config.last_loaded,
                })

            elif msg.get("type") == "persona_reload":
                # Reload persona config from disk and broadcast to all dashboards
                persona.reload()
                event_logger.info("system", "Persona configuration reloaded")
                await broadcast_to_dashboards({"type": "persona", "persona": persona.to_dict()})

            elif msg.get("type") == "chain_submit":
                # Submit a multi-step task chain
                chain_name = (msg.get("name") or "").strip()
                raw_steps = msg.get("steps") or []

                if not chain_name or not raw_steps:
                    await websocket.send_json({
                        "type": "chain_submitted",
                        "ok": False,
                        "error": "Chain name and at least one step are required",
                    })
                    continue

                # Build Task objects for each step
                step_tasks = []
                for step_def in raw_steps:
                    desc = (step_def.get("description") or "").strip()
                    if not desc:
                        continue
                    complexity_str = (step_def.get("complexity") or "auto").lower()
                    try:
                        complexity = TaskComplexity(complexity_str)
                    except ValueError:
                        complexity = TaskComplexity.AUTO
                    raw_caps = step_def.get("required_capabilities") or []
                    caps = [c.strip() for c in raw_caps if isinstance(c, str) and c.strip()]

                    step_tasks.append(Task(
                        description=desc,
                        source="chain",
                        complexity=complexity,
                        required_capabilities=caps,
                    ))

                if not step_tasks:
                    await websocket.send_json({
                        "type": "chain_submitted",
                        "ok": False,
                        "error": "No valid steps provided",
                    })
                    continue

                chain = TaskChain(
                    name=chain_name,
                    steps=step_tasks,
                    source="dashboard",
                )
                task_queue.add_chain(chain)

                logger.info(
                    "Chain %s submitted from dashboard (%d steps): %s",
                    chain.short_id, chain.total_steps, chain_name,
                )
                event_logger.info(
                    "chain",
                    f"Chain '{chain_name}' ({chain.short_id}) submitted: {chain.total_steps} steps",
                    chain_id=chain.short_id, steps=chain.total_steps,
                    source="dashboard",
                )

                await websocket.send_json({
                    "type": "chain_submitted",
                    "ok": True,
                    "chain_id": chain.chain_id,
                    "short_id": chain.short_id,
                })
                await broadcast_chain_status()
                await broadcast_task_status()

            elif msg.get("type") == "swarm_submit":
                # Submit a parallel swarm — all subtasks run at once
                swarm_name = (msg.get("name") or "").strip()
                objective = (msg.get("objective") or "").strip()
                raw_subtasks = msg.get("subtasks") or []

                if not swarm_name or not objective or not raw_subtasks:
                    await websocket.send_json({
                        "type": "swarm_submitted",
                        "ok": False,
                        "error": "Swarm name, objective, and at least one subtask are required",
                    })
                    continue

                # Build Task objects for each subtask
                subtask_list = []
                for sub_def in raw_subtasks:
                    desc = (sub_def.get("description") or "").strip()
                    if not desc:
                        continue
                    complexity_str = (sub_def.get("complexity") or "auto").lower()
                    try:
                        complexity = TaskComplexity(complexity_str)
                    except ValueError:
                        complexity = TaskComplexity.AUTO
                    raw_caps = sub_def.get("required_capabilities") or []
                    caps = [c.strip() for c in raw_caps if isinstance(c, str) and c.strip()]

                    subtask_list.append(Task(
                        description=desc,
                        source="swarm",
                        complexity=complexity,
                        required_capabilities=caps,
                    ))

                if not subtask_list:
                    await websocket.send_json({
                        "type": "swarm_submitted",
                        "ok": False,
                        "error": "No valid subtasks provided",
                    })
                    continue

                swarm = SwarmTask(
                    name=swarm_name,
                    objective=objective,
                    subtasks=subtask_list,
                    source="dashboard",
                )
                task_queue.add_swarm(swarm)

                logger.info(
                    "Swarm %s submitted from dashboard (%d subtasks): %s",
                    swarm.short_id, len(swarm.subtasks), swarm_name,
                )
                event_logger.info(
                    "swarm",
                    f"Swarm '{swarm_name}' ({swarm.short_id}) submitted: "
                    f"{len(swarm.subtasks)} subtasks",
                    swarm_id=swarm.short_id, subtasks=len(swarm.subtasks),
                    source="dashboard",
                )

                await websocket.send_json({
                    "type": "swarm_submitted",
                    "ok": True,
                    "swarm_id": swarm.swarm_id,
                    "short_id": swarm.short_id,
                })
                await broadcast_swarm_status()
                await broadcast_task_status()

            elif msg.get("type") == "file_send":
                # Dashboard sending a small file (<1MB) via WebSocket
                target_id = msg.get("agent_id", "")
                filename = msg.get("filename", "")
                dest_path = msg.get("dest_path", "")
                b64_data = msg.get("data", "")

                if target_id not in agent_websockets:
                    await websocket.send_json({
                        "type": "file_transfer_progress",
                        "status": "failed",
                        "error": "Agent is offline",
                        "filename": filename,
                    })
                    continue

                try:
                    data = base64.b64decode(b64_data)
                except Exception:
                    await websocket.send_json({
                        "type": "file_transfer_progress",
                        "status": "failed",
                        "error": "Invalid base64 data",
                        "filename": filename,
                    })
                    continue

                file_size = len(data)
                max_size = config.file_transfer.max_file_size
                if file_size > max_size:
                    await websocket.send_json({
                        "type": "file_transfer_progress",
                        "status": "failed",
                        "error": f"File too large ({file_size} bytes, max {max_size})",
                        "filename": filename,
                    })
                    continue

                agent_info = registry.get_by_id(target_id)
                agent_name = agent_info.name if agent_info else target_id
                checksum = compute_checksum(data)
                chunk_size = config.file_transfer.chunk_size
                chunk_count = (file_size + chunk_size - 1) // chunk_size

                try:
                    transfer = ft_manager.create_transfer(
                        direction="to_agent",
                        agent_id=target_id,
                        agent_name=agent_name,
                        filename=filename,
                        file_size=file_size,
                        chunk_count=chunk_count,
                        checksum=checksum,
                        dest_path=dest_path,
                    )
                except ValueError as e:
                    await websocket.send_json({
                        "type": "file_transfer_progress",
                        "status": "failed",
                        "error": str(e),
                        "filename": filename,
                    })
                    continue

                event_logger.info(
                    "file_transfer",
                    f"Sending {filename} ({file_size} bytes) to {agent_name}",
                    transfer_id=transfer.transfer_id, agent=agent_name,
                )
                agent_ws = agent_websockets[target_id]
                asyncio.create_task(relay_file_to_agent(transfer, data, agent_ws))

            elif msg.get("type") == "file_request_from_agent":
                # Dashboard requesting a file from an agent
                target_id = msg.get("agent_id", "")
                file_path = msg.get("file_path", "")

                if target_id not in agent_websockets:
                    await websocket.send_json({
                        "type": "file_transfer_progress",
                        "status": "failed",
                        "error": "Agent is offline",
                        "filename": file_path,
                    })
                    continue

                agent_info = registry.get_by_id(target_id)
                agent_name = agent_info.name if agent_info else target_id

                # Create a placeholder transfer — metadata filled in by file_send_start
                transfer = ft_manager.create_transfer(
                    direction="from_agent",
                    agent_id=target_id,
                    agent_name=agent_name,
                    filename=os.path.basename(file_path) if file_path else "unknown",
                    file_size=0,
                    chunk_count=0,
                )

                event_logger.info(
                    "file_transfer",
                    f"Requesting {file_path} from {agent_name}",
                    transfer_id=transfer.transfer_id, agent=agent_name,
                )

                await agent_websockets[target_id].send_json({
                    "type": "file_request",
                    "transfer_id": transfer.transfer_id,
                    "file_path": file_path,
                })

            elif msg.get("type") == "file_transfer_cancel":
                transfer_id = msg.get("transfer_id", "")
                transfer = ft_manager.get_transfer(transfer_id)
                if transfer:
                    # Notify agent if it's a to_agent transfer
                    if transfer.agent_id in agent_websockets:
                        try:
                            await agent_websockets[transfer.agent_id].send_json({
                                "type": "file_transfer_cancel",
                                "transfer_id": transfer_id,
                                "reason": "Cancelled by user",
                            })
                        except Exception:
                            pass
                    await ft_manager.cancel_transfer(transfer_id)

            elif msg.get("type") == "deploy_check_version":
                # Check connector version on a single agent
                target_id = msg.get("agent_id", "")
                version_info = await deployer.check_version(target_id)
                local = deployer.get_local_version()
                match = (version_info.get("hash") == local["hash"]) if version_info.get("hash") else None
                await websocket.send_json({
                    "type": "deploy_version_result",
                    "agent_id": target_id,
                    "remote": version_info,
                    "local": local,
                    "match": match,
                })

            elif msg.get("type") == "deploy_agent":
                # Deploy connector to a single agent
                target_id = msg.get("agent_id", "")
                result = await deployer.deploy_connector(target_id)
                await broadcast_to_dashboards({
                    "type": "deploy_result",
                    **result.to_dict(),
                })
                # Push updated deploy status to all dashboards
                await broadcast_to_dashboards({
                    "type": "deploy_status",
                    **deployer.get_status(),
                })

            elif msg.get("type") == "deploy_all":
                # Deploy connector to all online agents sequentially
                await deployer.deploy_all(broadcast_fn=broadcast_to_dashboards)
                # Push final status
                await broadcast_to_dashboards({
                    "type": "deploy_status",
                    **deployer.get_status(),
                })

            elif msg.get("type") == "backup_now":
                # Run backup immediately
                result = backup_manager.backup()
                await broadcast_to_dashboards({
                    "type": "backup_result",
                    **result,
                })
                await broadcast_to_dashboards({
                    "type": "backup_status",
                    **backup_manager.get_status(),
                })

            elif msg.get("type") == "backup_restore":
                # Restore from a specific backup archive
                filename = msg.get("filename", "")
                result = backup_manager.restore(filename)
                await broadcast_to_dashboards({
                    "type": "backup_restore_result",
                    **result,
                })

            elif msg.get("type") == "schedule_create":
                sched_name = (msg.get("name") or "").strip()
                sched_desc = (msg.get("description") or "").strip()
                sched_type_str = (msg.get("schedule_type") or "interval").lower()
                complexity_str = (msg.get("complexity") or "auto").lower()
                interval_seconds = msg.get("interval_seconds", 300)
                run_at_time = msg.get("run_at_time", "09:00")
                raw_caps = msg.get("required_capabilities") or []
                caps = [c.strip() for c in raw_caps if isinstance(c, str) and c.strip()]

                if not sched_name or not sched_desc:
                    await websocket.send_json({
                        "type": "schedule_created",
                        "ok": False,
                        "error": "Name and description are required",
                    })
                    continue

                try:
                    complexity = TaskComplexity(complexity_str)
                except ValueError:
                    complexity = TaskComplexity.AUTO

                try:
                    sched_type = ScheduleType(sched_type_str)
                except ValueError:
                    sched_type = ScheduleType.INTERVAL

                try:
                    interval_seconds = int(interval_seconds)
                except (ValueError, TypeError):
                    interval_seconds = 300

                sched = scheduler.add_schedule(
                    name=sched_name,
                    description=sched_desc,
                    complexity=complexity,
                    schedule_type=sched_type,
                    interval_seconds=interval_seconds,
                    run_at_time=run_at_time,
                    required_capabilities=caps,
                )
                event_logger.info(
                    "scheduler",
                    f"Schedule '{sched_name}' ({sched.short_id}) created: {sched_type_str}",
                    schedule_id=sched.short_id, schedule_type=sched_type_str,
                )
                await websocket.send_json({
                    "type": "schedule_created",
                    "ok": True,
                    "schedule_id": sched.schedule_id,
                    "short_id": sched.short_id,
                })
                await broadcast_schedule_status()

            elif msg.get("type") == "schedule_update":
                schedule_id = msg.get("schedule_id", "")
                update_fields = {}
                for key in ("name", "description", "complexity", "schedule_type",
                            "interval_seconds", "run_at_time", "required_capabilities"):
                    if key in msg:
                        update_fields[key] = msg[key]

                result = scheduler.update_schedule(schedule_id, **update_fields)
                if result:
                    event_logger.info(
                        "scheduler",
                        f"Schedule '{result.name}' ({result.short_id}) updated",
                        schedule_id=result.short_id,
                    )
                    await websocket.send_json({"type": "schedule_updated", "ok": True})
                    await broadcast_schedule_status()
                else:
                    await websocket.send_json({
                        "type": "schedule_updated",
                        "ok": False,
                        "error": "Schedule not found",
                    })

            elif msg.get("type") == "schedule_delete":
                schedule_id = msg.get("schedule_id", "")
                removed = scheduler.remove_schedule(schedule_id)
                if removed:
                    event_logger.info(
                        "scheduler",
                        f"Schedule {schedule_id[:8]} deleted",
                        schedule_id=schedule_id[:8],
                    )
                    await websocket.send_json({"type": "schedule_deleted", "ok": True})
                    await broadcast_schedule_status()
                else:
                    await websocket.send_json({
                        "type": "schedule_deleted",
                        "ok": False,
                        "error": "Schedule not found",
                    })

            elif msg.get("type") == "schedule_toggle":
                schedule_id = msg.get("schedule_id", "")
                result = scheduler.toggle_enable(schedule_id)
                if result:
                    event_logger.info(
                        "scheduler",
                        f"Schedule '{result.name}' ({result.short_id}) {'enabled' if result.enabled else 'disabled'}",
                        schedule_id=result.short_id,
                        enabled=result.enabled,
                    )
                    await websocket.send_json({
                        "type": "schedule_toggled",
                        "ok": True,
                        "enabled": result.enabled,
                    })
                    await broadcast_schedule_status()
                else:
                    await websocket.send_json({
                        "type": "schedule_toggled",
                        "ok": False,
                        "error": "Schedule not found",
                    })

            elif msg.get("type") == "content_entry_create":
                title = (msg.get("title") or "").strip()
                description = (msg.get("description") or "").strip()
                content_type = (msg.get("content_type") or "general").lower()
                scheduled_date = msg.get("scheduled_date") or None

                if not title:
                    await websocket.send_json({
                        "type": "content_entry_created",
                        "ok": False,
                        "error": "Title is required",
                    })
                    continue

                entry = content_calendar.add_entry(
                    title=title,
                    content_type=content_type,
                    description=description,
                    scheduled_date=scheduled_date,
                )
                event_logger.info(
                    "content",
                    f"Content entry '{title}' ({entry.short_id}) created",
                    entry_id=entry.short_id, content_type=content_type,
                )
                await websocket.send_json({
                    "type": "content_entry_created",
                    "ok": True,
                    "entry_id": entry.entry_id,
                    "short_id": entry.short_id,
                })
                await broadcast_content_calendar_status()

            elif msg.get("type") == "content_entry_update":
                entry_id = msg.get("entry_id", "")
                update_fields = {}
                for key in ("title", "description", "content_type", "status",
                            "scheduled_date", "body"):
                    if key in msg:
                        update_fields[key] = msg[key]

                result = content_calendar.update_entry(entry_id, **update_fields)
                if result:
                    event_logger.info(
                        "content",
                        f"Content entry '{result.title}' ({result.short_id}) updated",
                        entry_id=result.short_id,
                    )
                    await websocket.send_json({"type": "content_entry_updated", "ok": True})
                    await broadcast_content_calendar_status()
                else:
                    await websocket.send_json({
                        "type": "content_entry_updated",
                        "ok": False,
                        "error": "Entry not found",
                    })

            elif msg.get("type") == "content_entry_delete":
                entry_id = msg.get("entry_id", "")
                removed = content_calendar.remove_entry(entry_id)
                if removed:
                    event_logger.info(
                        "content",
                        f"Content entry {entry_id[:8]} deleted",
                        entry_id=entry_id[:8],
                    )
                    await websocket.send_json({"type": "content_entry_deleted", "ok": True})
                    await broadcast_content_calendar_status()
                else:
                    await websocket.send_json({
                        "type": "content_entry_deleted",
                        "ok": False,
                        "error": "Entry not found",
                    })

            # --- Content Pipeline handlers ---
            elif msg.get("type") == "pipeline_create":
                title = (msg.get("title") or "").strip()
                topic = (msg.get("topic") or "").strip()
                if not title or not topic:
                    await websocket.send_json({
                        "type": "pipeline_created",
                        "ok": False,
                        "error": "Title and topic are required",
                    })
                    continue
                pipeline = pipeline_manager.create_pipeline(
                    title=title, topic=topic,
                    metadata=msg.get("metadata"),
                )
                await websocket.send_json({
                    "type": "pipeline_created",
                    "ok": True,
                    "pipeline_id": pipeline["pipeline_id"],
                })
                await broadcast_pipeline_status()

            elif msg.get("type") == "pipeline_approve":
                pipeline_id = msg.get("pipeline_id", "")
                notes = (msg.get("notes") or "").strip()
                result = await pipeline_manager.approve_review(pipeline_id, notes)
                if result:
                    await websocket.send_json({"type": "pipeline_approved", "ok": True})
                    await broadcast_pipeline_status()
                else:
                    await websocket.send_json({
                        "type": "pipeline_approved",
                        "ok": False,
                        "error": "Pipeline not found or not in review",
                    })

            elif msg.get("type") == "pipeline_reject":
                pipeline_id = msg.get("pipeline_id", "")
                notes = (msg.get("notes") or "").strip()
                result = await pipeline_manager.reject_review(pipeline_id, notes)
                if result:
                    await websocket.send_json({"type": "pipeline_rejected", "ok": True})
                    await broadcast_pipeline_status()
                else:
                    await websocket.send_json({
                        "type": "pipeline_rejected",
                        "ok": False,
                        "error": "Pipeline not found or not in review",
                    })

            elif msg.get("type") == "pipeline_cancel":
                pipeline_id = msg.get("pipeline_id", "")
                result = await pipeline_manager.cancel_pipeline(pipeline_id)
                if result:
                    await websocket.send_json({"type": "pipeline_cancelled", "ok": True})
                    await broadcast_pipeline_status()
                else:
                    await websocket.send_json({
                        "type": "pipeline_cancelled",
                        "ok": False,
                        "error": "Pipeline not found or already finished",
                    })

            # ── YouTube Publishing ──────────────────────────────
            elif msg.get("type") == "yt_create":
                try:
                    video = youtube_manager.create_video(
                        title=msg.get("title", "Untitled"),
                        topic=msg.get("topic", ""),
                        category=msg.get("category", "general"),
                        keywords=msg.get("keywords", []),
                        scheduled_date=msg.get("scheduled_date"),
                        thumbnail_path=msg.get("thumbnail_path"),
                        video_path=msg.get("video_path"),
                        chapters=msg.get("chapters", []),
                        pipeline_id=msg.get("pipeline_id"),
                    )
                    await websocket.send_json({"type": "yt_created", "ok": True, "video": video.to_dict()})
                    await broadcast_youtube_status()
                except Exception as e:
                    await websocket.send_json({"type": "yt_created", "ok": False, "error": str(e)})

            elif msg.get("type") == "yt_update":
                video_id = msg.get("video_id", "")
                kwargs = {k: v for k, v in msg.items() if k not in ("type", "video_id")}
                video = youtube_manager.update_video(video_id, **kwargs)
                if video:
                    await websocket.send_json({"type": "yt_updated", "ok": True, "video": video.to_dict()})
                    await broadcast_youtube_status()
                else:
                    await websocket.send_json({"type": "yt_updated", "ok": False, "error": "Video not found"})

            elif msg.get("type") == "yt_delete":
                video_id = msg.get("video_id", "")
                deleted = youtube_manager.delete_video(video_id)
                if deleted:
                    await websocket.send_json({"type": "yt_deleted", "ok": True})
                    await broadcast_youtube_status()
                else:
                    await websocket.send_json({"type": "yt_deleted", "ok": False, "error": "Video not found"})

            elif msg.get("type") == "yt_queue":
                video_id = msg.get("video_id", "")
                video = youtube_manager.queue_video(video_id)
                if video:
                    await websocket.send_json({"type": "yt_queued", "ok": True, "video": video.to_dict()})
                    await broadcast_youtube_status()
                else:
                    await websocket.send_json({"type": "yt_queued", "ok": False, "error": "Cannot queue — missing title/description or not in draft status"})

            elif msg.get("type") == "yt_publish":
                video_id = msg.get("video_id", "")
                url = msg.get("published_url", "")
                video = youtube_manager.mark_published(video_id, url)
                if video:
                    await websocket.send_json({"type": "yt_published", "ok": True, "video": video.to_dict()})
                    await broadcast_youtube_status()
                else:
                    await websocket.send_json({"type": "yt_published", "ok": False, "error": "Cannot publish — not in queued status"})

            elif msg.get("type") == "yt_fail":
                video_id = msg.get("video_id", "")
                error = msg.get("error", "")
                video = youtube_manager.mark_failed(video_id, error)
                if video:
                    await websocket.send_json({"type": "yt_failed", "ok": True})
                    await broadcast_youtube_status()
                else:
                    await websocket.send_json({"type": "yt_failed", "ok": False, "error": "Video not found"})

            elif msg.get("type") == "yt_generate_seo":
                video_id = msg.get("video_id", "")
                result = await youtube_manager.generate_seo(video_id)
                if result:
                    await websocket.send_json({"type": "yt_seo_result", "ok": True, "seo": result})
                    await broadcast_youtube_status()
                else:
                    await websocket.send_json({"type": "yt_seo_result", "ok": False, "error": "SEO generation failed"})

            elif msg.get("type") == "yt_update_stats":
                video_id = msg.get("video_id", "")
                video = youtube_manager.update_stats(
                    video_id,
                    view_count=msg.get("view_count", 0),
                    like_count=msg.get("like_count", 0),
                    comment_count=msg.get("comment_count", 0),
                )
                if video:
                    await websocket.send_json({"type": "yt_stats_updated", "ok": True})
                    await broadcast_youtube_status()
                else:
                    await websocket.send_json({"type": "yt_stats_updated", "ok": False, "error": "Video not found"})

            # --- Media Library (Video Processor) ---

            elif msg.get("type") == "media_scan":
                new_files = video_processor.scan_footage()
                await websocket.send_json({"type": "media_scanned", "ok": True, "count": len(new_files)})
                await broadcast_media_status()

            elif msg.get("type") == "media_process":
                media_id = msg.get("media_id", "")
                result = await video_processor.process_media(media_id)
                if result:
                    await websocket.send_json({"type": "media_processed", "ok": True})
                else:
                    await websocket.send_json({"type": "media_processed", "ok": False, "error": "Processing failed"})
                await broadcast_media_status()

            elif msg.get("type") == "media_extract":
                media_id = msg.get("media_id", "")
                result = video_processor.extract_metadata(media_id)
                if result:
                    await websocket.send_json({"type": "media_extracted", "ok": True})
                else:
                    await websocket.send_json({"type": "media_extracted", "ok": False, "error": "Extraction failed"})
                await broadcast_media_status()

            elif msg.get("type") == "media_thumbnails":
                media_id = msg.get("media_id", "")
                thumbs = video_processor.generate_thumbnails(media_id)
                await websocket.send_json({"type": "media_thumbs_generated", "ok": True, "count": len(thumbs)})
                await broadcast_media_status()

            elif msg.get("type") == "media_suggest_thumb":
                media_id = msg.get("media_id", "")
                best = await video_processor.suggest_best_thumbnail(media_id)
                if best:
                    await websocket.send_json({"type": "media_thumb_suggested", "ok": True, "path": best})
                else:
                    await websocket.send_json({"type": "media_thumb_suggested", "ok": False, "error": "No candidates"})
                await broadcast_media_status()

            elif msg.get("type") == "media_transcode":
                media_id = msg.get("media_id", "")
                fmt = msg.get("format", "mp4")
                resolution = msg.get("resolution")
                result = video_processor.transcode(media_id, target_format=fmt, target_resolution=resolution)
                if result:
                    await websocket.send_json({"type": "media_transcoded", "ok": True})
                else:
                    await websocket.send_json({"type": "media_transcoded", "ok": False, "error": "Transcode failed"})
                await broadcast_media_status()

            elif msg.get("type") == "media_link_pipeline":
                media_id = msg.get("media_id", "")
                pipeline_id = msg.get("pipeline_id", "")
                result = video_processor.link_pipeline(media_id, pipeline_id)
                if result:
                    await websocket.send_json({"type": "media_linked_pipeline", "ok": True})
                else:
                    await websocket.send_json({"type": "media_linked_pipeline", "ok": False, "error": "Not found"})
                await broadcast_media_status()

            elif msg.get("type") == "media_link_youtube":
                media_id = msg.get("media_id", "")
                youtube_video_id = msg.get("youtube_video_id", "")
                result = video_processor.link_youtube(media_id, youtube_video_id)
                if result:
                    await websocket.send_json({"type": "media_linked_youtube", "ok": True})
                else:
                    await websocket.send_json({"type": "media_linked_youtube", "ok": False, "error": "Not found"})
                await broadcast_media_status()

            elif msg.get("type") == "media_delete":
                media_id = msg.get("media_id", "")
                deleted = video_processor.delete_media(media_id)
                if deleted:
                    await websocket.send_json({"type": "media_deleted", "ok": True})
                else:
                    await websocket.send_json({"type": "media_deleted", "ok": False, "error": "Not found"})
                await broadcast_media_status()

            # ── Social Media handlers ──────────────────────────────
            elif msg.get("type") == "sm_create":
                post = social_media_manager.create_post(
                    platform=msg.get("platform", "x_twitter"),
                    topic=msg.get("topic", ""),
                    content=msg.get("content", ""),
                    hashtags=msg.get("hashtags", []),
                    media_paths=msg.get("media_paths", []),
                    scheduled_time=msg.get("scheduled_time"),
                    pipeline_id=msg.get("pipeline_id"),
                    metadata=msg.get("metadata", {}),
                )
                await websocket.send_json({"type": "sm_created", "ok": True, "post": post.to_dict()})
                await broadcast_social_media_status()

            elif msg.get("type") == "sm_update":
                post_id = msg.get("post_id", "")
                kwargs = {}
                for k in ("platform", "content", "hashtags", "media_paths",
                          "scheduled_time", "post_url", "topic", "pipeline_id", "metadata"):
                    if k in msg:
                        kwargs[k] = msg[k]
                post = social_media_manager.update_post(post_id, **kwargs)
                if post:
                    await websocket.send_json({"type": "sm_updated", "ok": True, "post": post.to_dict()})
                else:
                    await websocket.send_json({"type": "sm_updated", "ok": False, "error": "Not found"})
                await broadcast_social_media_status()

            elif msg.get("type") == "sm_delete":
                post_id = msg.get("post_id", "")
                deleted = social_media_manager.delete_post(post_id)
                if deleted:
                    await websocket.send_json({"type": "sm_deleted", "ok": True})
                else:
                    await websocket.send_json({"type": "sm_deleted", "ok": False, "error": "Not found"})
                await broadcast_social_media_status()

            elif msg.get("type") == "sm_schedule":
                post_id = msg.get("post_id", "")
                post = social_media_manager.schedule_post(post_id)
                if post:
                    await websocket.send_json({"type": "sm_scheduled", "ok": True, "post": post.to_dict()})
                else:
                    await websocket.send_json({"type": "sm_scheduled", "ok": False, "error": "Cannot schedule (empty content or wrong status)"})
                await broadcast_social_media_status()

            elif msg.get("type") == "sm_mark_posted":
                post_id = msg.get("post_id", "")
                post_url = msg.get("post_url", "")
                post = social_media_manager.mark_posted(post_id, post_url)
                if post:
                    await websocket.send_json({"type": "sm_marked_posted", "ok": True, "post": post.to_dict()})
                else:
                    await websocket.send_json({"type": "sm_marked_posted", "ok": False, "error": "Cannot mark posted (wrong status)"})
                await broadcast_social_media_status()

            elif msg.get("type") == "sm_draft":
                topic = msg.get("topic", "")
                platforms = msg.get("platforms") or ["x_twitter", "instagram", "youtube_community", "facebook"]
                extra_context = msg.get("extra_context", "")
                posts = await social_media_manager.draft_batch(topic, platforms, extra_context)
                await websocket.send_json({
                    "type": "sm_drafted",
                    "ok": True,
                    "posts": [p.to_dict() for p in posts],
                })
                await broadcast_social_media_status()

            elif msg.get("type") == "sm_generate_schedule":
                frequency = msg.get("frequency")
                slots = social_media_manager.generate_weekly_schedule(frequency)
                await websocket.send_json({
                    "type": "sm_schedule_generated",
                    "ok": True,
                    "slots": slots,
                })

            # ── Email Template handlers ────────────────────────────
            elif msg.get("type") == "email_generate":
                cust_id = msg.get("customer_id", "")
                tmpl = msg.get("template_type", "")
                extra = msg.get("extra_data", {})
                try:
                    rec = await email_template_manager.generate_email(cust_id, tmpl, extra)
                    await websocket.send_json({
                        "type": "email_generated", "ok": True, "email": rec.to_dict(),
                    })
                except Exception as exc:
                    await websocket.send_json({
                        "type": "email_generated", "ok": False, "error": str(exc),
                    })
                await broadcast_email_status()

            elif msg.get("type") == "email_queue":
                eid = msg.get("email_id", "")
                ok = email_template_manager.queue_email(eid)
                await websocket.send_json({"type": "email_queued", "ok": ok})
                await broadcast_email_status()

            elif msg.get("type") == "email_send":
                eid = msg.get("email_id", "")
                ok = email_template_manager.send_email(eid)
                await websocket.send_json({"type": "email_sent", "ok": ok})
                await broadcast_email_status()

            elif msg.get("type") == "email_send_queued":
                count = email_template_manager.send_queued()
                await websocket.send_json({"type": "email_send_queued_done", "ok": True, "count": count})
                await broadcast_email_status()

            elif msg.get("type") == "email_delete":
                eid = msg.get("email_id", "")
                ok = email_template_manager.delete_email(eid)
                await websocket.send_json({"type": "email_deleted", "ok": ok})
                await broadcast_email_status()

            elif msg.get("type") == "email_customer_history":
                cust_id = msg.get("customer_id", "")
                history = email_template_manager.get_customer_history(cust_id)
                await websocket.send_json({
                    "type": "email_customer_history_result",
                    "customer_id": cust_id,
                    "emails": history,
                })

            # ── Photo Documentation handlers ──────────────────────
            elif msg.get("type") == "photo_import":
                imported = photo_doc_manager.import_photos()
                await websocket.send_json({
                    "type": "photo_imported",
                    "ok": True,
                    "count": len(imported),
                    "photos": [p.to_dict() for p in imported],
                })
                await broadcast_photo_docs_status()

            elif msg.get("type") == "photo_tag":
                pid = msg.get("photo_id", "")
                rec = photo_doc_manager.tag_photo(
                    pid,
                    service_record_id=msg.get("service_record_id", ""),
                    stage=msg.get("stage", ""),
                    description=msg.get("description", ""),
                    content_worthy=int(msg.get("content_worthy", 0)),
                )
                await websocket.send_json({
                    "type": "photo_tagged",
                    "ok": rec is not None,
                    "photo": rec.to_dict() if rec else None,
                })
                await broadcast_photo_docs_status()

            elif msg.get("type") == "photo_tag_batch":
                pids = msg.get("photo_ids", [])
                results = photo_doc_manager.tag_photos_batch(
                    pids,
                    service_record_id=msg.get("service_record_id", ""),
                    stage=msg.get("stage", ""),
                )
                await websocket.send_json({
                    "type": "photo_batch_tagged",
                    "ok": True,
                    "count": len(results),
                })
                await broadcast_photo_docs_status()

            elif msg.get("type") == "photo_generate_comparison":
                srid = msg.get("service_record_id", "")
                comp_path = photo_doc_manager.generate_comparison(srid)
                await websocket.send_json({
                    "type": "photo_comparison_result",
                    "ok": comp_path is not None,
                    "path": comp_path or "",
                    "url": f"/api/photo/composite/{srid}" if comp_path else "",
                })

            elif msg.get("type") == "photo_build_gallery":
                srid = msg.get("service_record_id", "")
                gal_path = photo_doc_manager.build_gallery(srid)
                await websocket.send_json({
                    "type": "photo_gallery_result",
                    "ok": gal_path is not None,
                    "path": gal_path or "",
                    "url": f"/api/photo/gallery/{srid}" if gal_path else "",
                })

            elif msg.get("type") == "photo_update":
                pid = msg.get("photo_id", "")
                updates = {k: v for k, v in msg.items() if k not in ("type", "photo_id")}
                rec = photo_doc_manager.update_photo(pid, **updates)
                await websocket.send_json({
                    "type": "photo_updated",
                    "ok": rec is not None,
                    "photo": rec.to_dict() if rec else None,
                })
                await broadcast_photo_docs_status()

            elif msg.get("type") == "photo_delete":
                pid = msg.get("photo_id", "")
                ok = photo_doc_manager.delete_photo(pid)
                await websocket.send_json({"type": "photo_deleted", "ok": ok, "photo_id": pid})
                await broadcast_photo_docs_status()

            elif msg.get("type") == "photo_list_service":
                srid = msg.get("service_record_id", "")
                photos = photo_doc_manager.get_service_photos(srid)
                await websocket.send_json({
                    "type": "photo_service_list",
                    "service_record_id": srid,
                    "photos": [p.to_dict() for p in photos],
                })

            elif msg.get("type") == "photo_content_worthy":
                photos = photo_doc_manager.get_content_worthy()
                await websocket.send_json({
                    "type": "photo_content_worthy_list",
                    "photos": [p.to_dict() for p in photos],
                })

            # ── Training / Learning Mode handlers ─────────────────
            elif msg.get("type") == "training_start":
                title = msg.get("title", "Untitled Session")
                vehicle_info = msg.get("vehicle_info", "")
                category = msg.get("category", "general")
                linked_tree_id = msg.get("linked_tree_id", "")
                session = training_manager.start_training_session(
                    title=title,
                    vehicle_info=vehicle_info,
                    category=category,
                    linked_tree_id=linked_tree_id,
                )
                await websocket.send_json({
                    "type": "training_started",
                    "ok": True,
                    "session": session.to_dict(),
                })
                await broadcast_training_status()

            elif msg.get("type") == "training_record_step":
                sid = msg.get("session_id", "")
                step_type = msg.get("step_type", "")
                content = msg.get("content", "")
                meta = msg.get("metadata", {})
                session = training_manager.record_step(sid, step_type, content, meta)
                await websocket.send_json({
                    "type": "training_step_recorded",
                    "ok": session is not None,
                    "session": session.to_dict() if session else None,
                })
                await broadcast_training_status()

            elif msg.get("type") == "training_complete":
                sid = msg.get("session_id", "")
                session = training_manager.complete_session(sid)
                await websocket.send_json({
                    "type": "training_completed",
                    "ok": session is not None,
                    "session": session.to_dict() if session else None,
                })
                await broadcast_training_status()

            elif msg.get("type") == "training_extract":
                sid = msg.get("session_id", "")
                patterns = await training_manager.extract_patterns(sid)
                await websocket.send_json({
                    "type": "training_patterns_extracted",
                    "ok": True,
                    "patterns": patterns,
                    "session_id": sid,
                })
                await broadcast_training_status()

            elif msg.get("type") == "training_suggest":
                symptoms = msg.get("symptoms", "")
                category = msg.get("category", "")
                result = await training_manager.suggest_from_experience(symptoms, category)
                await websocket.send_json({
                    "type": "training_suggestions",
                    **result,
                })

            elif msg.get("type") == "training_tree_suggest":
                category = msg.get("category", "")
                result = await training_manager.suggest_tree_updates(category)
                await websocket.send_json({
                    "type": "training_tree_suggestions",
                    **result,
                })

            elif msg.get("type") == "training_delete":
                sid = msg.get("session_id", "")
                ok = training_manager.delete_session(sid)
                await websocket.send_json({
                    "type": "training_deleted",
                    "ok": ok,
                    "session_id": sid,
                })
                await broadcast_training_status()

            # ================================================================
            # RIDE LOG
            # ================================================================

            elif msg.get("type") == "ride_log":
                entry = ride_log_manager.log_ride(
                    date=msg.get("date", ""),
                    start_time=msg.get("start_time", ""),
                    end_time=msg.get("end_time", ""),
                    start_location=msg.get("start_location", ""),
                    end_location=msg.get("end_location", ""),
                    distance=float(msg.get("distance", 0)),
                    purpose=msg.get("purpose", "personal"),
                    weather_conditions=msg.get("weather_conditions", ""),
                    fuel_used=float(msg.get("fuel_used", 0)),
                    odometer_start=float(msg.get("odometer_start", 0)),
                    odometer_end=float(msg.get("odometer_end", 0)),
                    notes=msg.get("notes", ""),
                )
                await websocket.send_json({
                    "type": "ride_logged",
                    "ok": True,
                    "ride": entry.to_dict(),
                })
                await broadcast_ride_log_status()

            elif msg.get("type") == "ride_update":
                ride_id = msg.get("ride_id", "")
                fields = {k: v for k, v in msg.items() if k not in ("type", "ride_id")}
                # Convert numeric fields
                for num_field in ("distance", "fuel_used", "odometer_start", "odometer_end"):
                    if num_field in fields:
                        fields[num_field] = float(fields[num_field])
                entry = ride_log_manager.update_ride(ride_id, **fields)
                await websocket.send_json({
                    "type": "ride_updated",
                    "ok": entry is not None,
                    "ride": entry.to_dict() if entry else None,
                })
                await broadcast_ride_log_status()

            elif msg.get("type") == "ride_delete":
                ride_id = msg.get("ride_id", "")
                ok = ride_log_manager.delete_ride(ride_id)
                await websocket.send_json({
                    "type": "ride_deleted",
                    "ok": ok,
                    "ride_id": ride_id,
                })
                await broadcast_ride_log_status()

            elif msg.get("type") == "ride_mileage_report":
                year = int(msg.get("year", 0))
                report = ride_log_manager.mileage_report(year)
                await websocket.send_json({
                    "type": "ride_mileage_report_result",
                    **report,
                })

            elif msg.get("type") == "ride_fuel_stats":
                months = int(msg.get("months", 6))
                stats = ride_log_manager.fuel_tracker(months)
                await websocket.send_json({
                    "type": "ride_fuel_stats_result",
                    **stats,
                })

            # ================================================================
            # MARKET MONITOR
            # ================================================================

            elif msg.get("type") == "market_add_competitor":
                comp = market_monitor.add_competitor(
                    name=msg.get("name", ""),
                    comp_type=msg.get("comp_type", "independent_shop"),
                    services=msg.get("services", []),
                    pricing=msg.get("pricing", {}),
                    rating=float(msg.get("rating", 0)),
                    review_count=int(msg.get("review_count", 0)),
                    location=msg.get("location", ""),
                    coverage_area=msg.get("coverage_area", ""),
                    source=msg.get("source", "manual"),
                    source_url=msg.get("source_url", ""),
                    phone=msg.get("phone", ""),
                    website=msg.get("website", ""),
                    notes=msg.get("notes", ""),
                )
                await websocket.send_json({
                    "type": "market_competitor_added",
                    "ok": True,
                    "competitor": comp.to_dict(),
                })
                await broadcast_market_monitor_status()

            elif msg.get("type") == "market_update_competitor":
                cid = msg.get("competitor_id", "")
                fields = {k: v for k, v in msg.items()
                          if k not in ("type", "competitor_id")}
                for num_f in ("rating",):
                    if num_f in fields:
                        fields[num_f] = float(fields[num_f])
                for int_f in ("review_count",):
                    if int_f in fields:
                        fields[int_f] = int(fields[int_f])
                comp = market_monitor.update_competitor(cid, **fields)
                await websocket.send_json({
                    "type": "market_competitor_updated",
                    "ok": comp is not None,
                    "competitor": comp.to_dict() if comp else None,
                })
                await broadcast_market_monitor_status()

            elif msg.get("type") == "market_delete_competitor":
                cid = msg.get("competitor_id", "")
                ok = market_monitor.delete_competitor(cid)
                await websocket.send_json({
                    "type": "market_competitor_deleted",
                    "ok": ok,
                    "competitor_id": cid,
                })
                await broadcast_market_monitor_status()

            elif msg.get("type") == "market_scan":
                source = msg.get("source", "google")
                result = await market_monitor.scan_for_competitors(source)
                await websocket.send_json({
                    "type": "market_scan_result",
                    "ok": True,
                    **result,
                })
                await broadcast_market_monitor_status()

            elif msg.get("type") == "market_analyze":
                analysis_type = msg.get("analysis_type", "competitive")
                result = await market_monitor.generate_analysis(analysis_type)
                await websocket.send_json({
                    "type": "market_analysis_result",
                    "ok": True,
                    **result,
                })
                await broadcast_market_monitor_status()

            elif msg.get("type") == "market_pricing_matrix":
                matrix = market_monitor.pricing_matrix()
                await websocket.send_json({
                    "type": "market_pricing_result",
                    **matrix,
                })

            # ================================================================
            # CUSTOMER FEEDBACK
            # ================================================================

            elif msg.get("type") == "feedback_submit":
                fb = feedback_manager.submit_feedback(
                    customer_id=msg.get("customer_id", ""),
                    customer_name=msg.get("customer_name", ""),
                    service_record_id=msg.get("service_record_id", ""),
                    date=msg.get("date", ""),
                    overall_rating=int(msg.get("overall_rating", 0)),
                    service_quality=int(msg.get("service_quality", 0)),
                    communication=int(msg.get("communication", 0)),
                    timeliness=int(msg.get("timeliness", 0)),
                    value=int(msg.get("value", 0)),
                    comments=msg.get("comments", ""),
                    testimonial_approved=bool(msg.get("testimonial_approved", False)),
                )
                await websocket.send_json({
                    "type": "feedback_submitted",
                    "ok": True,
                    "feedback": fb.to_dict(),
                })
                await broadcast_feedback_status()

            elif msg.get("type") == "feedback_update":
                fid = msg.get("feedback_id", "")
                fields = {k: v for k, v in msg.items()
                          if k not in ("type", "feedback_id")}
                for int_f in ("overall_rating", "service_quality",
                              "communication", "timeliness", "value"):
                    if int_f in fields:
                        fields[int_f] = int(fields[int_f])
                fb = feedback_manager.update_feedback(fid, **fields)
                await websocket.send_json({
                    "type": "feedback_updated",
                    "ok": fb is not None,
                    "feedback": fb.to_dict() if fb else None,
                })
                await broadcast_feedback_status()

            elif msg.get("type") == "feedback_delete":
                fid = msg.get("feedback_id", "")
                ok = feedback_manager.delete_feedback(fid)
                await websocket.send_json({
                    "type": "feedback_deleted",
                    "ok": ok,
                    "feedback_id": fid,
                })
                await broadcast_feedback_status()

            elif msg.get("type") == "feedback_analyze":
                fid = msg.get("feedback_id", "")
                result = await feedback_manager.analyze_sentiment(fid)
                await websocket.send_json({
                    "type": "feedback_analyzed",
                    "ok": "error" not in result,
                    "feedback_id": fid,
                    **result,
                })
                await broadcast_feedback_status()

            elif msg.get("type") == "feedback_request":
                result = feedback_manager.request_feedback(
                    customer_id=msg.get("customer_id", ""),
                    service_record_id=msg.get("service_record_id", ""),
                    service_date=msg.get("service_date", ""),
                    delay_days=int(msg.get("delay_days", 3)),
                )
                await websocket.send_json({
                    "type": "feedback_request_created",
                    "ok": True,
                    **result,
                })

            elif msg.get("type") == "feedback_testimonial":
                fid = msg.get("feedback_id", "")
                fb = feedback_manager.queue_testimonial(fid)
                await websocket.send_json({
                    "type": "feedback_testimonial_queued",
                    "ok": fb is not None,
                    "feedback_id": fid,
                })
                await broadcast_feedback_status()

            # ================================================================
            # WARRANTY & RECALL TRACKER
            # ================================================================

            elif msg.get("type") == "warranty_add":
                w = warranty_manager.add_warranty(
                    vehicle_id=msg.get("vehicle_id", ""),
                    customer_id=msg.get("customer_id", ""),
                    component=msg.get("component", ""),
                    start_date=msg.get("start_date", ""),
                    end_date=msg.get("end_date", ""),
                    coverage_type=msg.get("coverage_type", "factory"),
                    provider=msg.get("provider", ""),
                    documentation_path=msg.get("documentation_path", ""),
                    mileage_limit=int(msg.get("mileage_limit", 0)),
                    notes=msg.get("notes", ""),
                )
                await websocket.send_json({
                    "type": "warranty_added",
                    "ok": True,
                    "warranty": w.to_dict(),
                })
                await broadcast_warranty_status()

            elif msg.get("type") == "warranty_update":
                wid = msg.get("warranty_id", "")
                fields = {k: v for k, v in msg.items()
                          if k not in ("type", "warranty_id")}
                if "mileage_limit" in fields:
                    fields["mileage_limit"] = int(fields["mileage_limit"])
                w = warranty_manager.update_warranty(wid, **fields)
                await websocket.send_json({
                    "type": "warranty_updated",
                    "ok": w is not None,
                    "warranty": w.to_dict() if w else None,
                })
                await broadcast_warranty_status()

            elif msg.get("type") == "warranty_delete":
                wid = msg.get("warranty_id", "")
                ok = warranty_manager.delete_warranty(wid)
                await websocket.send_json({
                    "type": "warranty_deleted",
                    "ok": ok,
                    "warranty_id": wid,
                })
                await broadcast_warranty_status()

            elif msg.get("type") == "recall_add":
                r = warranty_manager.add_recall(
                    nhtsa_id=msg.get("nhtsa_id", ""),
                    make=msg.get("make", ""),
                    model=msg.get("model", ""),
                    year_start=int(msg.get("year_start", 0)),
                    year_end=int(msg.get("year_end", 0)),
                    component=msg.get("component", ""),
                    description=msg.get("description", ""),
                    remedy=msg.get("remedy", ""),
                    date_issued=msg.get("date_issued", ""),
                    severity=msg.get("severity", "safety"),
                    notes=msg.get("notes", ""),
                )
                await websocket.send_json({
                    "type": "recall_added",
                    "ok": True,
                    "recall": r.to_dict(),
                })
                await broadcast_warranty_status()

            elif msg.get("type") == "recall_delete":
                rid = msg.get("recall_id", "")
                ok = warranty_manager.delete_recall(rid)
                await websocket.send_json({
                    "type": "recall_deleted",
                    "ok": ok,
                    "recall_id": rid,
                })
                await broadcast_warranty_status()

            elif msg.get("type") == "warranty_scan_recalls":
                cid = msg.get("customer_id", "")
                matches = warranty_manager.scan_recalls(customer_id=cid)
                count = warranty_manager.notify_recall_matches(matches)
                await websocket.send_json({
                    "type": "warranty_scan_results",
                    "ok": True,
                    "matches": matches,
                    "notifications_sent": count,
                })
                await broadcast_warranty_status()

            elif msg.get("type") == "warranty_check_expiring":
                days = int(msg.get("days", 30))
                count = warranty_manager.notify_expiring_warranties(days)
                await websocket.send_json({
                    "type": "warranty_expiring_checked",
                    "ok": True,
                    "notifications_sent": count,
                })

            # ================================================================
            # MAINTENANCE SCHEDULER
            # ================================================================

            elif msg.get("type") == "maintenance_add_schedule":
                s = maintenance_scheduler.add_schedule(
                    vehicle_id=msg.get("vehicle_id", ""),
                    customer_id=msg.get("customer_id", ""),
                    service_type=msg.get("service_type", "oil_change"),
                    interval_miles=int(msg.get("interval_miles", 0)),
                    interval_months=int(msg.get("interval_months", 0)),
                    last_service_date=msg.get("last_service_date", ""),
                    last_service_miles=int(msg.get("last_service_miles", 0)),
                    current_mileage=int(msg.get("current_mileage", 0)),
                    notes=msg.get("notes", ""),
                )
                await websocket.send_json({
                    "type": "maintenance_schedule_added",
                    "ok": True,
                    "schedule": s.to_dict(),
                })
                await broadcast_maintenance_status()

            elif msg.get("type") == "maintenance_update_schedule":
                sid = msg.get("schedule_id", "")
                fields = {k: v for k, v in msg.items()
                          if k not in ("type", "schedule_id")}
                for int_f in ("interval_miles", "interval_months",
                              "last_service_miles", "current_mileage", "enabled"):
                    if int_f in fields:
                        fields[int_f] = int(fields[int_f])
                s = maintenance_scheduler.update_schedule(sid, **fields)
                await websocket.send_json({
                    "type": "maintenance_schedule_updated",
                    "ok": s is not None,
                    "schedule": s.to_dict() if s else None,
                })
                await broadcast_maintenance_status()

            elif msg.get("type") == "maintenance_delete_schedule":
                sid = msg.get("schedule_id", "")
                ok = maintenance_scheduler.delete_schedule(sid)
                await websocket.send_json({
                    "type": "maintenance_schedule_deleted",
                    "ok": ok,
                    "schedule_id": sid,
                })
                await broadcast_maintenance_status()

            elif msg.get("type") == "maintenance_setup_vehicle":
                created = maintenance_scheduler.setup_vehicle(
                    vehicle_id=msg.get("vehicle_id", ""),
                    customer_id=msg.get("customer_id", ""),
                    make=msg.get("make", ""),
                    model=msg.get("model", ""),
                    current_mileage=int(msg.get("current_mileage", 0)),
                )
                await websocket.send_json({
                    "type": "maintenance_vehicle_setup",
                    "ok": True,
                    "count": len(created),
                    "schedules": [s.to_dict() for s in created],
                })
                await broadcast_maintenance_status()

            elif msg.get("type") == "maintenance_update_mileage":
                vid = msg.get("vehicle_id", "")
                miles = int(msg.get("mileage", 0))
                count = maintenance_scheduler.update_mileage(vid, miles)
                await websocket.send_json({
                    "type": "maintenance_mileage_updated",
                    "ok": count > 0,
                    "vehicle_id": vid,
                    "mileage": miles,
                    "schedules_updated": count,
                })
                await broadcast_maintenance_status()

            elif msg.get("type") == "maintenance_record_service":
                sched = maintenance_scheduler.record_service(
                    vehicle_id=msg.get("vehicle_id", ""),
                    service_type=msg.get("service_type", ""),
                    date=msg.get("date", ""),
                    mileage=int(msg.get("mileage", 0)),
                )
                await websocket.send_json({
                    "type": "maintenance_service_recorded",
                    "ok": sched is not None,
                    "schedule": sched.to_dict() if sched else None,
                })
                await broadcast_maintenance_status()

            elif msg.get("type") == "maintenance_generate_reminders":
                reminders = maintenance_scheduler.generate_reminders()
                await websocket.send_json({
                    "type": "maintenance_reminders_generated",
                    "ok": True,
                    "count": len(reminders),
                    "reminders": reminders,
                })

            elif msg.get("type") == "maintenance_forecast":
                days = int(msg.get("days", 90))
                forecast = maintenance_scheduler.maintenance_forecast(days)
                await websocket.send_json({
                    "type": "maintenance_forecast_result",
                    "ok": True,
                    **forecast,
                })

            # ================================================================
            # PLUGIN MANAGER
            # ================================================================

            elif msg.get("type") == "plugin_enable":
                pid = msg.get("plugin_id", "")
                ok = plugin_manager.enable_plugin(pid)
                await websocket.send_json({
                    "type": "plugin_enabled",
                    "ok": ok,
                    "plugin_id": pid,
                })
                await broadcast_plugin_status()

            elif msg.get("type") == "plugin_disable":
                pid = msg.get("plugin_id", "")
                ok = plugin_manager.disable_plugin(pid)
                await websocket.send_json({
                    "type": "plugin_disabled",
                    "ok": ok,
                    "plugin_id": pid,
                })
                await broadcast_plugin_status()

            elif msg.get("type") == "plugin_load":
                pid = msg.get("plugin_id", "")
                instance = plugin_manager.load_plugin(pid)
                await websocket.send_json({
                    "type": "plugin_loaded",
                    "ok": instance is not None,
                    "plugin_id": pid,
                })
                await broadcast_plugin_status()

            elif msg.get("type") == "plugin_unload":
                pid = msg.get("plugin_id", "")
                ok = plugin_manager.unload_plugin(pid)
                await websocket.send_json({
                    "type": "plugin_unloaded",
                    "ok": ok,
                    "plugin_id": pid,
                })
                await broadcast_plugin_status()

            elif msg.get("type") == "plugin_config":
                pid = msg.get("plugin_id", "")
                config_data = msg.get("config", {})
                ok = plugin_manager.update_config(pid, config_data)
                await websocket.send_json({
                    "type": "plugin_config_updated",
                    "ok": ok,
                    "plugin_id": pid,
                })
                await broadcast_plugin_status()

            elif msg.get("type") == "plugin_status_get":
                pid = msg.get("plugin_id", "")
                status = plugin_manager.plugin_status(pid)
                await websocket.send_json({
                    "type": "plugin_status_result",
                    "ok": status is not None,
                    "plugin": status,
                })

            # ── Export handlers ──────────────────────────────
            elif msg.get("type") == "export_get_status":
                await websocket.send_json({
                    "type": "export_status",
                    **export_manager.to_broadcast_dict(),
                })

            elif msg.get("type") == "export_get_fields":
                dtype = msg.get("data_type", "")
                fields = export_manager.get_available_fields(dtype)
                await websocket.send_json({
                    "type": "export_fields_result",
                    "data_type": dtype,
                    "fields": fields,
                })

            elif msg.get("type") == "export_clear_history":
                count = export_manager.clear_history()
                await websocket.send_json({
                    "type": "export_history_cleared",
                    "count": count,
                })
                await broadcast_export_status()

            # ── Offline mode handlers ────────────────────────
            elif msg.get("type") == "offline_get_status":
                await websocket.send_json({
                    "type": "offline_status",
                    **offline_manager.to_broadcast_dict(),
                })

            elif msg.get("type") == "offline_force_check":
                result = await offline_manager.check_connectivity()
                await websocket.send_json({
                    "type": "offline_check_result",
                    **result,
                })
                await broadcast_offline_status()

            elif msg.get("type") == "offline_process_queue":
                count = await offline_manager.force_process_queue()
                await websocket.send_json({
                    "type": "offline_queue_processed",
                    "count": count,
                })
                await broadcast_offline_status()

            elif msg.get("type") == "offline_clear_queue":
                status_filter = msg.get("status", "")
                count = offline_manager.clear_queue(status=status_filter)
                await websocket.send_json({
                    "type": "offline_queue_cleared",
                    "count": count,
                })
                await broadcast_offline_status()

            elif msg.get("type") == "offline_remove_op":
                queue_id = msg.get("queue_id", "")
                ok = offline_manager.remove_operation(queue_id)
                await websocket.send_json({
                    "type": "offline_op_removed",
                    "ok": ok,
                    "queue_id": queue_id,
                })
                await broadcast_offline_status()

            # ── Performance Monitor handlers ──────────────────────
            elif msg.get("type") == "performance_get_status":
                await websocket.send_json({
                    "type": "performance_status",
                    **performance_monitor.to_broadcast_dict(),
                })

            elif msg.get("type") == "performance_apply_rec":
                rec_id = msg.get("rec_id", "")
                result = performance_monitor.apply_recommendation(rec_id)
                await websocket.send_json({
                    "type": "performance_rec_applied",
                    "rec_id": rec_id,
                    **result,
                })
                await broadcast_performance_status()

            elif msg.get("type") == "performance_dismiss_rec":
                rec_id = msg.get("rec_id", "")
                ok = performance_monitor.dismiss_recommendation(rec_id)
                await websocket.send_json({
                    "type": "performance_rec_dismissed",
                    "rec_id": rec_id,
                    "ok": ok,
                })
                await broadcast_performance_status()

            elif msg.get("type") == "performance_update_threshold":
                key = msg.get("key", "")
                value = float(msg.get("value", 0))
                ok = performance_monitor.set_threshold(key, value)
                await websocket.send_json({
                    "type": "performance_threshold_updated",
                    "key": key,
                    "value": value,
                    "ok": ok,
                })
                await broadcast_performance_status()

            # --- User Management ---
            elif msg.get("type") == "user_list":
                await websocket.send_json({
                    "type": "user_list",
                    "users": access_control.to_broadcast_list(),
                })

            elif msg.get("type") == "user_add":
                from security.auth import hash_password as _hash_pw
                new_username = msg.get("username", "").strip().lower()
                display_name = msg.get("display_name", "").strip()
                password = msg.get("password", "")
                new_role = msg.get("role", "viewer")
                telegram_id = msg.get("telegram_chat_id")
                if not new_username or not display_name:
                    await websocket.send_json({
                        "type": "user_add_result",
                        "ok": False,
                        "error": "Username and display name are required",
                    })
                else:
                    pw_hash = _hash_pw(password) if password else ""
                    user = access_control.add_user(
                        username=new_username,
                        display_name=display_name,
                        role=new_role,
                        password_hash=pw_hash,
                        telegram_chat_id=int(telegram_id) if telegram_id else None,
                        created_by=ws_username,
                    )
                    if user:
                        await websocket.send_json({"type": "user_add_result", "ok": True})
                        await broadcast_to_dashboards({
                            "type": "user_list",
                            "users": access_control.to_broadcast_list(),
                        })
                    else:
                        await websocket.send_json({
                            "type": "user_add_result",
                            "ok": False,
                            "error": f"User '{new_username}' already exists",
                        })

            elif msg.get("type") == "user_set_role":
                target_user = msg.get("username", "")
                new_role = msg.get("role", "")
                ok = access_control.set_role(target_user, new_role, changed_by=ws_username)
                await websocket.send_json({
                    "type": "user_set_role_result",
                    "ok": ok,
                    "username": target_user,
                })
                if ok:
                    await broadcast_to_dashboards({
                        "type": "user_list",
                        "users": access_control.to_broadcast_list(),
                    })

            elif msg.get("type") == "user_revoke":
                target_user = msg.get("username", "")
                ok = access_control.revoke_access(target_user, revoked_by=ws_username)
                await websocket.send_json({
                    "type": "user_revoke_result",
                    "ok": ok,
                    "username": target_user,
                })
                if ok:
                    await broadcast_to_dashboards({
                        "type": "user_list",
                        "users": access_control.to_broadcast_list(),
                    })

            elif msg.get("type") == "user_restore":
                target_user = msg.get("username", "")
                ok = access_control.restore_access(target_user, restored_by=ws_username)
                await websocket.send_json({
                    "type": "user_restore_result",
                    "ok": ok,
                    "username": target_user,
                })
                if ok:
                    await broadcast_to_dashboards({
                        "type": "user_list",
                        "users": access_control.to_broadcast_list(),
                    })

            elif msg.get("type") == "user_delete":
                target_user = msg.get("username", "")
                if target_user == ws_username:
                    await websocket.send_json({
                        "type": "user_delete_result",
                        "ok": False,
                        "error": "Cannot delete your own account",
                    })
                else:
                    ok = access_control.delete_user(target_user, deleted_by=ws_username)
                    await websocket.send_json({
                        "type": "user_delete_result",
                        "ok": ok,
                        "username": target_user,
                    })
                    if ok:
                        await broadcast_to_dashboards({
                            "type": "user_list",
                            "users": access_control.to_broadcast_list(),
                        })

            elif msg.get("type") == "research_result_delete":
                entry_id = msg.get("entry_id", "")
                removed = research_store.remove_entry(entry_id)
                if removed:
                    event_logger.info(
                        "research",
                        f"Research entry {entry_id[:8]} deleted",
                        entry_id=entry_id[:8],
                    )
                    await websocket.send_json({"type": "research_result_deleted", "ok": True})
                    await broadcast_research_status()
                else:
                    await websocket.send_json({
                        "type": "research_result_deleted",
                        "ok": False,
                        "error": "Entry not found",
                    })

            elif msg.get("type") == "research_search":
                keyword = (msg.get("keyword") or "").strip()
                if keyword:
                    results = research_store.search_by_query(keyword, limit=20)
                else:
                    results = research_store.list_all(limit=50)
                await websocket.send_json({
                    "type": "research_search_results",
                    "entries": [e.to_dict() for e in results],
                    "keyword": keyword,
                })

            elif msg.get("type") == "scout_scan":
                # Trigger an immediate scout scan
                try:
                    new_listings = await doppler_scout.scan()
                    await websocket.send_json({
                        "type": "scout_scan_result",
                        "ok": True,
                        "new_count": len(new_listings),
                    })
                    await broadcast_scout_status()
                except Exception as e:
                    await websocket.send_json({
                        "type": "scout_scan_result",
                        "ok": False,
                        "error": str(e),
                    })

            elif msg.get("type") == "scout_update_status":
                entry_id = msg.get("entry_id", "")
                new_status = msg.get("status", "")
                if new_status in [s.value for s in ListingStatus]:
                    updated = scout_store.update_entry(entry_id, status=new_status)
                    if updated:
                        event_logger.info(
                            "scout",
                            f"Scout listing {entry_id[:8]} marked {new_status}",
                            entry_id=entry_id[:8], status=new_status,
                        )
                        await broadcast_scout_status()

            elif msg.get("type") == "scout_delete":
                entry_id = msg.get("entry_id", "")
                removed = scout_store.remove_entry(entry_id)
                if removed:
                    event_logger.info(
                        "scout",
                        f"Scout listing {entry_id[:8]} deleted",
                        entry_id=entry_id[:8],
                    )
                    await websocket.send_json({"type": "scout_deleted", "ok": True})
                    await broadcast_scout_status()
                else:
                    await websocket.send_json({
                        "type": "scout_deleted",
                        "ok": False,
                        "error": "Entry not found",
                    })

            elif msg.get("type") == "scout_rescore":
                entry_id = msg.get("entry_id", "")
                try:
                    listing = await doppler_scout.score_single(entry_id)
                    if listing:
                        await websocket.send_json({
                            "type": "scout_rescored",
                            "ok": True,
                            "entry": listing.to_dict(),
                        })
                        await broadcast_scout_status()
                    else:
                        await websocket.send_json({
                            "type": "scout_rescored",
                            "ok": False,
                            "error": "Entry not found",
                        })
                except Exception as e:
                    await websocket.send_json({
                        "type": "scout_rescored",
                        "ok": False,
                        "error": str(e),
                    })

            # -----------------------------------------------------------
            # Market Analytics handlers
            # -----------------------------------------------------------
            elif msg.get("type") == "market_snapshot":
                try:
                    count = market_analyzer.take_snapshot()
                    await websocket.send_json({
                        "type": "market_snapshot_result",
                        "ok": True,
                        "inserted": count,
                    })
                    await broadcast_market_analytics()
                except Exception as e:
                    await websocket.send_json({
                        "type": "market_snapshot_result",
                        "ok": False,
                        "error": str(e),
                    })

            elif msg.get("type") == "market_generate_report":
                try:
                    report_type = msg.get("report_type", "weekly")
                    report = await market_analyzer.generate_report(report_type)
                    await websocket.send_json({
                        "type": "market_report_result",
                        "ok": True,
                        **report,
                    })
                    await broadcast_market_analytics()
                except Exception as e:
                    await websocket.send_json({
                        "type": "market_report_result",
                        "ok": False,
                        "error": str(e),
                    })

            elif msg.get("type") == "market_get_history":
                try:
                    history = market_analyzer.get_price_history(
                        make=msg.get("make", ""),
                        model=msg.get("model", ""),
                        days=msg.get("days", 30),
                    )
                    await websocket.send_json({
                        "type": "market_history_result",
                        "ok": True,
                        "make": msg.get("make", ""),
                        "model": msg.get("model", ""),
                        "history": history,
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "market_history_result",
                        "ok": False,
                        "error": str(e),
                    })

            # -----------------------------------------------------------
            # Webhook handlers
            # -----------------------------------------------------------
            elif msg.get("type") == "webhook_register":
                try:
                    ep = webhook_manager.register_endpoint(
                        name=msg.get("name", "Unnamed"),
                        url=msg.get("url", ""),
                        direction=msg.get("direction", "outbound"),
                        secret=msg.get("secret", ""),
                        event_filters=msg.get("event_filters", []),
                        severity_filter=msg.get("severity_filter", "all"),
                        enabled=msg.get("enabled", True),
                    )
                    event_logger.info("webhook", f"Webhook endpoint registered: {ep['name']}")
                    await websocket.send_json({"type": "webhook_registered", "ok": True, "endpoint": ep})
                    await broadcast_webhook_status()
                except Exception as e:
                    await websocket.send_json({"type": "webhook_registered", "ok": False, "error": str(e)})

            elif msg.get("type") == "webhook_update":
                eid = msg.get("endpoint_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("name", "url", "secret", "event_filters",
                                    "severity_filter", "enabled", "direction")}
                updated = webhook_manager.update_endpoint(eid, **updates)
                if updated:
                    event_logger.info("webhook", f"Webhook endpoint updated: {updated['name']}")
                    await websocket.send_json({"type": "webhook_updated", "ok": True, "endpoint": updated})
                    await broadcast_webhook_status()
                else:
                    await websocket.send_json({"type": "webhook_updated", "ok": False, "error": "Not found"})

            elif msg.get("type") == "webhook_delete":
                eid = msg.get("endpoint_id", "")
                deleted = webhook_manager.delete_endpoint(eid)
                if deleted:
                    event_logger.info("webhook", f"Webhook endpoint deleted: {eid[:8]}")
                    await websocket.send_json({"type": "webhook_deleted", "ok": True, "endpoint_id": eid})
                    await broadcast_webhook_status()
                else:
                    await websocket.send_json({"type": "webhook_deleted", "ok": False, "error": "Not found"})

            elif msg.get("type") == "webhook_toggle":
                eid = msg.get("endpoint_id", "")
                enabled = msg.get("enabled", True)
                updated = webhook_manager.update_endpoint(eid, enabled=enabled)
                if updated:
                    state = "enabled" if enabled else "disabled"
                    event_logger.info("webhook", f"Webhook endpoint {state}: {updated['name']}")
                    await websocket.send_json({"type": "webhook_toggled", "ok": True, "endpoint": updated})
                    await broadcast_webhook_status()
                else:
                    await websocket.send_json({"type": "webhook_toggled", "ok": False, "error": "Not found"})

            elif msg.get("type") == "webhook_test":
                eid = msg.get("endpoint_id", "")
                try:
                    test_event = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "severity": "info",
                        "category": "test",
                        "message": "Webhook test delivery from CAM dashboard",
                        "details": {"test": True},
                    }
                    await webhook_manager._deliver_to_endpoint(eid, test_event)
                    await websocket.send_json({"type": "webhook_test_result", "ok": True, "endpoint_id": eid})
                except Exception as e:
                    await websocket.send_json({"type": "webhook_test_result", "ok": False, "error": str(e)})

            elif msg.get("type") == "webhook_delivery_history":
                eid = msg.get("endpoint_id")
                limit = msg.get("limit", 30)
                deliveries = webhook_manager.get_recent_deliveries(limit=limit, endpoint_id=eid)
                await websocket.send_json({
                    "type": "webhook_delivery_history_result",
                    "deliveries": deliveries,
                })

            # -----------------------------------------------------------
            # Business CRUD handlers
            # -----------------------------------------------------------
            elif msg.get("type") == "business_customer_create":
                cust = business_store.add_customer(
                    name=msg.get("name", ""),
                    phone=msg.get("phone", ""),
                    email=msg.get("email", ""),
                    bike_info=msg.get("bike_info", ""),
                    notes=msg.get("notes", ""),
                )
                event_logger.info(
                    "business",
                    f"Customer created: {cust.name} ({cust.short_id})",
                    customer_id=cust.short_id,
                )
                await broadcast_business_status()

            elif msg.get("type") == "business_customer_update":
                cust_id = msg.get("customer_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("name", "phone", "email", "bike_info", "notes")}
                updated = business_store.update_customer(cust_id, **updates)
                if updated:
                    event_logger.info(
                        "business",
                        f"Customer updated: {updated.name} ({updated.short_id})",
                        customer_id=updated.short_id,
                    )
                    await broadcast_business_status()

            elif msg.get("type") == "business_customer_delete":
                cust_id = msg.get("customer_id", "")
                removed = business_store.remove_customer(cust_id)
                if removed:
                    event_logger.info(
                        "business",
                        f"Customer deleted: {cust_id[:8]}",
                        customer_id=cust_id[:8],
                    )
                await websocket.send_json({
                    "type": "business_customer_deleted",
                    "ok": removed,
                })
                if removed:
                    await broadcast_business_status()

            elif msg.get("type") == "business_appointment_create":
                appt = business_store.add_appointment(
                    customer_id=msg.get("customer_id", ""),
                    customer_name=msg.get("customer_name", ""),
                    date=msg.get("date", ""),
                    time=msg.get("time", ""),
                    bike=msg.get("bike", ""),
                    service_type=msg.get("service_type", ""),
                    status=msg.get("status", "scheduled"),
                    notes=msg.get("notes", ""),
                    location=msg.get("location", ""),
                    estimated_cost=float(msg.get("estimated_cost", 0)),
                )
                event_logger.info(
                    "business",
                    f"Appointment created: {appt.customer_name} on {appt.date} ({appt.short_id})",
                    appointment_id=appt.short_id,
                )
                await broadcast_business_status()

            elif msg.get("type") == "business_appointment_update":
                appt_id = msg.get("appointment_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("customer_id", "customer_name", "date", "time",
                                    "bike", "service_type", "status", "notes",
                                    "location", "estimated_cost")}
                if "estimated_cost" in updates:
                    updates["estimated_cost"] = float(updates["estimated_cost"])
                updated = business_store.update_appointment(appt_id, **updates)
                if updated:
                    event_logger.info(
                        "business",
                        f"Appointment updated: {updated.short_id}",
                        appointment_id=updated.short_id,
                    )
                    await broadcast_business_status()

            elif msg.get("type") == "business_appointment_delete":
                appt_id = msg.get("appointment_id", "")
                removed = business_store.remove_appointment(appt_id)
                if removed:
                    event_logger.info(
                        "business",
                        f"Appointment deleted: {appt_id[:8]}",
                        appointment_id=appt_id[:8],
                    )
                await websocket.send_json({
                    "type": "business_appointment_deleted",
                    "ok": removed,
                })
                if removed:
                    await broadcast_business_status()

            elif msg.get("type") == "business_invoice_create":
                items = msg.get("items", [])
                inv = business_store.add_invoice(
                    customer_id=msg.get("customer_id", ""),
                    customer_name=msg.get("customer_name", ""),
                    date=msg.get("date", ""),
                    items=items,
                    labor_hours=float(msg.get("labor_hours", 0)),
                    labor_rate=float(msg.get("labor_rate",
                                             getattr(_biz_cfg, 'default_labor_rate', 75.0))),
                    status=msg.get("status", "draft"),
                    notes=msg.get("notes", ""),
                    appointment_id=msg.get("appointment_id", ""),
                    prefix=getattr(_biz_cfg, 'invoice_prefix', 'DC'),
                )
                event_logger.info(
                    "business",
                    f"Invoice created: {inv.invoice_number} — ${inv.total:.2f} ({inv.short_id})",
                    invoice_id=inv.short_id, invoice_number=inv.invoice_number,
                )
                await broadcast_business_status()

            elif msg.get("type") == "business_invoice_update":
                inv_id = msg.get("invoice_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("customer_id", "customer_name", "date", "items",
                                    "labor_hours", "labor_rate", "status", "notes",
                                    "appointment_id")}
                for float_key in ("labor_hours", "labor_rate"):
                    if float_key in updates:
                        updates[float_key] = float(updates[float_key])
                updated = business_store.update_invoice(inv_id, **updates)
                if updated:
                    event_logger.info(
                        "business",
                        f"Invoice updated: {updated.invoice_number} ({updated.short_id})",
                        invoice_id=updated.short_id,
                    )
                    await broadcast_business_status()

            elif msg.get("type") == "business_invoice_delete":
                inv_id = msg.get("invoice_id", "")
                removed = business_store.remove_invoice(inv_id)
                if removed:
                    event_logger.info(
                        "business",
                        f"Invoice deleted: {inv_id[:8]}",
                        invoice_id=inv_id[:8],
                    )
                await websocket.send_json({
                    "type": "business_invoice_deleted",
                    "ok": removed,
                })
                if removed:
                    await broadcast_business_status()

            elif msg.get("type") == "business_inventory_create":
                item = business_store.add_inventory_item(
                    name=msg.get("name", ""),
                    category=msg.get("category", ""),
                    quantity=int(msg.get("quantity", 0)),
                    cost=float(msg.get("cost", 0)),
                    location=msg.get("location", ""),
                    reorder_threshold=int(msg.get("reorder_threshold", 0)),
                    supplier=msg.get("supplier", ""),
                    part_number=msg.get("part_number", ""),
                    notes=msg.get("notes", ""),
                )
                event_logger.info(
                    "business",
                    f"Inventory item created: {item.name} ({item.short_id})",
                    item_id=item.short_id,
                )
                await broadcast_business_status()

            elif msg.get("type") == "business_inventory_update":
                item_id = msg.get("item_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("name", "category", "quantity", "cost", "location",
                                    "reorder_threshold", "supplier", "part_number", "notes")}
                for int_key in ("quantity", "reorder_threshold"):
                    if int_key in updates:
                        updates[int_key] = int(updates[int_key])
                for float_key in ("cost",):
                    if float_key in updates:
                        updates[float_key] = float(updates[float_key])
                updated = business_store.update_inventory_item(item_id, **updates)
                if updated:
                    event_logger.info(
                        "business",
                        f"Inventory item updated: {updated.name} ({updated.short_id})",
                        item_id=updated.short_id,
                    )
                    await broadcast_business_status()

            elif msg.get("type") == "business_inventory_delete":
                item_id = msg.get("item_id", "")
                removed = business_store.remove_inventory_item(item_id)
                if removed:
                    event_logger.info(
                        "business",
                        f"Inventory item deleted: {item_id[:8]}",
                        item_id=item_id[:8],
                    )
                await websocket.send_json({
                    "type": "business_inventory_deleted",
                    "ok": removed,
                })
                if removed:
                    await broadcast_business_status()

            # -----------------------------------------------------------
            # Service Records CRUD handlers
            # -----------------------------------------------------------
            elif msg.get("type") == "service_vehicle_create":
                v = service_store.add_vehicle(
                    year=msg.get("year", ""),
                    make=msg.get("make", ""),
                    model=msg.get("model", ""),
                    vin=msg.get("vin", ""),
                    owner_id=msg.get("owner_id", ""),
                    owner_name=msg.get("owner_name", ""),
                    color=msg.get("color", ""),
                    mileage=int(msg.get("mileage", 0)),
                    notes=msg.get("notes", ""),
                )
                event_logger.info(
                    "business",
                    f"Vehicle created: {v.display_name} ({v.short_id})",
                    vehicle_id=v.short_id,
                )
                await broadcast_service_records_status()

            elif msg.get("type") == "service_vehicle_delete":
                vid = msg.get("vehicle_id", "")
                removed = service_store.remove_vehicle(vid)
                if removed:
                    event_logger.info(
                        "business",
                        f"Vehicle deleted: {vid[:8]}",
                        vehicle_id=vid[:8],
                    )
                await websocket.send_json({
                    "type": "service_vehicle_deleted",
                    "ok": removed,
                })
                if removed:
                    await broadcast_service_records_status()

            elif msg.get("type") == "service_record_create":
                parts = msg.get("parts_used", [])
                rec = service_store.add_record(
                    vehicle_id=msg.get("vehicle_id", ""),
                    customer_id=msg.get("customer_id", ""),
                    customer_name=msg.get("customer_name", ""),
                    date=msg.get("date", ""),
                    service_type=msg.get("service_type", ""),
                    services_performed=msg.get("services_performed", []),
                    parts_used=parts,
                    labor_hours=float(msg.get("labor_hours", 0)),
                    labor_rate=float(msg.get("labor_rate", 75.0)),
                    notes=msg.get("notes", ""),
                    recommendations=msg.get("recommendations", ""),
                    appointment_id=msg.get("appointment_id", ""),
                    invoice_id=msg.get("invoice_id", ""),
                )
                event_logger.info(
                    "business",
                    f"Service record created: {rec.vehicle_summary} — ${rec.total_cost:.2f} ({rec.short_id})",
                    record_id=rec.short_id,
                )
                await broadcast_service_records_status()

                # Auto-decrement inventory for parts used in this service record
                if parts:
                    for p in parts:
                        pn = p.get("part_number", "") if isinstance(p, dict) else ""
                        qty = int(p.get("quantity", 1)) if isinstance(p, dict) else 1
                        if pn:
                            inventory_manager.use_part_by_number(
                                pn, quantity=qty,
                                service_record_id=rec.record_id,
                                reason="service_record",
                            )
                    await broadcast_inventory_status()

            elif msg.get("type") == "service_record_update":
                rec_id = msg.get("record_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("vehicle_id", "customer_id", "customer_name",
                                    "date", "service_type", "services_performed",
                                    "parts_used", "labor_hours", "labor_rate",
                                    "notes", "recommendations")}
                for float_key in ("labor_hours", "labor_rate"):
                    if float_key in updates:
                        updates[float_key] = float(updates[float_key])
                updated = service_store.update_record(rec_id, **updates)
                if updated:
                    event_logger.info(
                        "business",
                        f"Service record updated: {updated.short_id}",
                        record_id=updated.short_id,
                    )
                    await broadcast_service_records_status()

            elif msg.get("type") == "service_record_delete":
                rec_id = msg.get("record_id", "")
                removed = service_store.remove_record(rec_id)
                if removed:
                    event_logger.info(
                        "business",
                        f"Service record deleted: {rec_id[:8]}",
                        record_id=rec_id[:8],
                    )
                await websocket.send_json({
                    "type": "service_record_deleted",
                    "ok": removed,
                })
                if removed:
                    await broadcast_service_records_status()

            elif msg.get("type") == "service_record_pdf":
                rec_id = msg.get("record_id", "")
                try:
                    pdf_path = service_store.generate_pdf(rec_id)
                    if pdf_path:
                        await websocket.send_json({
                            "type": "service_record_pdf_result",
                            "ok": True,
                            "record_id": rec_id,
                            "url": f"/api/service-report/{rec_id}",
                        })
                    else:
                        await websocket.send_json({
                            "type": "service_record_pdf_result",
                            "ok": False,
                            "error": "Record not found",
                        })
                except Exception as e:
                    await websocket.send_json({
                        "type": "service_record_pdf_result",
                        "ok": False,
                        "error": str(e),
                    })

            # -----------------------------------------------------------
            # Diagnostic Decision Tree handlers
            # -----------------------------------------------------------
            elif msg.get("type") == "diag_start_session":
                tree_id = msg.get("tree_id", "")
                vehicle_id = msg.get("vehicle_id", "")
                vehicle_summary = msg.get("vehicle_summary", "")
                customer_name = msg.get("customer_name", "")
                session = diagnostic_engine.start_session(
                    tree_id=tree_id,
                    vehicle_id=vehicle_id,
                    vehicle_summary=vehicle_summary,
                    customer_name=customer_name,
                )
                if session:
                    node = diagnostic_engine.get_current_node(session.session_id)
                    await websocket.send_json({
                        "type": "diag_session_started",
                        "ok": True,
                        "session": session.to_dict(),
                        "node": node,
                    })
                    event_logger.info(
                        "diagnostics",
                        "Diagnostic session started",
                        tree_id=tree_id,
                        session_id=session.short_id,
                    )
                    await broadcast_diagnostics_status()
                else:
                    await websocket.send_json({
                        "type": "diag_session_started",
                        "ok": False,
                        "error": f"Unknown tree: {tree_id}",
                    })

            elif msg.get("type") == "diag_answer":
                session_id = msg.get("session_id", "")
                answer_index = int(msg.get("answer_index", 0))
                result = diagnostic_engine.answer_question(session_id, answer_index)
                await websocket.send_json({
                    "type": "diag_answer_result",
                    **result,
                })
                if result.get("ok"):
                    await broadcast_diagnostics_status()

            elif msg.get("type") == "diag_go_back":
                session_id = msg.get("session_id", "")
                result = diagnostic_engine.go_back(session_id)
                await websocket.send_json({
                    "type": "diag_go_back_result",
                    **result,
                })
                if result.get("ok"):
                    await broadcast_diagnostics_status()

            elif msg.get("type") == "diag_ai_suggest":
                session_id = msg.get("session_id", "")
                result = await diagnostic_engine.get_ai_suggestions(session_id)
                await websocket.send_json({
                    "type": "diag_ai_suggest_result",
                    **result,
                })

            elif msg.get("type") == "diag_complete":
                session_id = msg.get("session_id", "")
                result = diagnostic_engine.complete_and_save_record(session_id)
                await websocket.send_json({
                    "type": "diag_complete_result",
                    **result,
                })
                if result.get("ok"):
                    await broadcast_diagnostics_status()
                    await broadcast_service_records_status()

            elif msg.get("type") == "diag_abandon":
                session_id = msg.get("session_id", "")
                session = diagnostic_engine.abandon_session(session_id)
                await websocket.send_json({
                    "type": "diag_abandon_result",
                    "ok": session is not None,
                    "session": session.to_dict() if session else None,
                })
                if session:
                    await broadcast_diagnostics_status()

            elif msg.get("type") == "diag_reload_trees":
                diagnostic_engine.reload_trees()
                await websocket.send_json({
                    "type": "diag_reload_result",
                    "ok": True,
                    "trees_loaded": len(diagnostic_engine._trees),
                })
                event_logger.info(
                    "diagnostics",
                    "Diagnostic trees reloaded",
                    trees_loaded=len(diagnostic_engine._trees),
                )
                await broadcast_diagnostics_status()

            elif msg.get("type") == "diag_update_notes":
                session_id = msg.get("session_id", "")
                notes = msg.get("notes", "")
                ok = diagnostic_engine.update_notes(session_id, notes)
                await websocket.send_json({
                    "type": "diag_notes_result",
                    "ok": ok,
                })

            # --- CRM handlers ---
            elif msg.get("type") == "crm_customer_create":
                customer = crm_store.add_customer(
                    name=msg.get("name", ""),
                    phone=msg.get("phone", ""),
                    email=msg.get("email", ""),
                    address=msg.get("address", ""),
                    preferred_contact_method=msg.get("preferred_contact_method", "phone"),
                    tags=msg.get("tags", []),
                    notes_summary=msg.get("notes_summary", ""),
                )
                await websocket.send_json({
                    "type": "crm_customer_created",
                    "ok": True,
                    "customer": customer.to_dict(),
                })
                await broadcast_crm_status()

            elif msg.get("type") == "crm_customer_update":
                cid = msg.get("customer_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("name", "phone", "email", "address",
                                    "preferred_contact_method", "tags",
                                    "notes_summary", "last_contact",
                                    "business_customer_id", "metadata")}
                customer = crm_store.update_customer(cid, **updates)
                if customer:
                    await websocket.send_json({
                        "type": "crm_customer_updated",
                        "ok": True,
                        "customer": customer.to_dict(),
                    })
                    await broadcast_crm_status()

            elif msg.get("type") == "crm_customer_delete":
                cid = msg.get("customer_id", "")
                removed = crm_store.remove_customer(cid)
                await websocket.send_json({
                    "type": "crm_customer_deleted",
                    "ok": removed,
                    "customer_id": cid,
                })
                if removed:
                    await broadcast_crm_status()

            elif msg.get("type") == "crm_customer_search":
                query = msg.get("query", "")
                results = crm_store.search_customers(query)
                await websocket.send_json({
                    "type": "crm_search_result",
                    "customers": [c.to_dict() for c in results],
                })

            elif msg.get("type") == "crm_customer_profile":
                cid = msg.get("customer_id", "")
                profile = crm_store.get_customer_profile(cid)
                await websocket.send_json({
                    "type": "crm_profile_result",
                    **profile,
                })

            elif msg.get("type") == "crm_note_create":
                cid = msg.get("customer_id", "")
                content = msg.get("content", "")
                category = msg.get("category", "general")
                note = crm_store.add_note(cid, content, category)
                await websocket.send_json({
                    "type": "crm_note_created",
                    "ok": True,
                    "note": note.to_dict(),
                })
                await broadcast_crm_status()

            elif msg.get("type") == "crm_note_delete":
                nid = msg.get("note_id", "")
                removed = crm_store.remove_note(nid)
                await websocket.send_json({
                    "type": "crm_note_deleted",
                    "ok": removed,
                    "note_id": nid,
                })
                if removed:
                    await broadcast_crm_status()

            # --- Appointment Scheduler handlers ---
            elif msg.get("type") == "appt_schedule_book":
                appt = appointment_scheduler.book(
                    customer_id=msg.get("customer_id", ""),
                    customer_name=msg.get("customer_name", ""),
                    vehicle_id=msg.get("vehicle_id", ""),
                    vehicle_summary=msg.get("vehicle_summary", ""),
                    date=msg.get("date", ""),
                    time_slot=msg.get("time_slot", ""),
                    duration_estimate=msg.get("duration_estimate", 60),
                    location_address=msg.get("location_address", ""),
                    location_lat=msg.get("location_lat", 0.0),
                    location_lon=msg.get("location_lon", 0.0),
                    service_type=msg.get("service_type", "diagnostic"),
                    estimated_cost=msg.get("estimated_cost", 0.0),
                    notes=msg.get("notes", ""),
                    metadata=msg.get("metadata"),
                )
                event_logger.info("appointment", f"Booked: {appt.customer_name} on {appt.date} {appt.time_slot}")
                await websocket.send_json({
                    "type": "appt_schedule_booked",
                    "ok": True,
                    "appointment": appt.to_dict(),
                })
                await broadcast_appointment_schedule_status()
                # Auto-replan today's route when appointments change
                _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if appt.date == _today and route_planner.get_day_plan(_today):
                    try:
                        route_planner.plan_route(_today)
                        await broadcast_route_status()
                    except Exception:
                        logger.exception("Auto-replan failed after appointment book")

            elif msg.get("type") == "appt_schedule_reschedule":
                appt = appointment_scheduler.reschedule(
                    appointment_id=msg.get("appointment_id", ""),
                    new_date=msg.get("new_date"),
                    new_time_slot=msg.get("new_time_slot"),
                    new_duration=msg.get("new_duration"),
                )
                if appt:
                    event_logger.info("appointment", f"Rescheduled: {appt.customer_name} to {appt.date} {appt.time_slot}")
                await websocket.send_json({
                    "type": "appt_schedule_rescheduled",
                    "ok": appt is not None,
                    "appointment": appt.to_dict() if appt else None,
                })
                if appt:
                    await broadcast_appointment_schedule_status()
                    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if appt.date == _today and route_planner.get_day_plan(_today):
                        try:
                            route_planner.plan_route(_today)
                            await broadcast_route_status()
                        except Exception:
                            logger.exception("Auto-replan failed after reschedule")

            elif msg.get("type") == "appt_schedule_cancel":
                appt = appointment_scheduler.cancel(
                    appointment_id=msg.get("appointment_id", ""),
                    reason=msg.get("reason", ""),
                )
                if appt:
                    event_logger.info("appointment", f"Cancelled: {appt.customer_name} on {appt.date}")
                await websocket.send_json({
                    "type": "appt_schedule_cancelled",
                    "ok": appt is not None,
                    "appointment": appt.to_dict() if appt else None,
                })
                if appt:
                    await broadcast_appointment_schedule_status()
                    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if appt.date == _today and route_planner.get_day_plan(_today):
                        try:
                            route_planner.plan_route(_today)
                            await broadcast_route_status()
                        except Exception:
                            logger.exception("Auto-replan failed after cancel")

            elif msg.get("type") == "appt_schedule_update":
                aid = msg.get("appointment_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("customer_name", "vehicle_id", "vehicle_summary",
                                    "date", "time_slot", "duration_estimate",
                                    "location_address", "location_lat", "location_lon",
                                    "service_type", "status", "estimated_cost", "notes")}
                appt = appointment_scheduler.update_appointment(aid, **updates)
                await websocket.send_json({
                    "type": "appt_schedule_updated",
                    "ok": appt is not None,
                    "appointment": appt.to_dict() if appt else None,
                })
                if appt:
                    await broadcast_appointment_schedule_status()
                    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if appt.date == _today and route_planner.get_day_plan(_today):
                        try:
                            route_planner.plan_route(_today)
                            await broadcast_route_status()
                        except Exception:
                            logger.exception("Auto-replan failed after update")

            elif msg.get("type") == "appt_schedule_delete":
                aid = msg.get("appointment_id", "")
                removed = appointment_scheduler.remove_appointment(aid)
                await websocket.send_json({
                    "type": "appt_schedule_deleted",
                    "ok": removed,
                    "appointment_id": aid,
                })
                if removed:
                    await broadcast_appointment_schedule_status()
                    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if route_planner.get_day_plan(_today):
                        try:
                            route_planner.plan_route(_today)
                            await broadcast_route_status()
                        except Exception:
                            logger.exception("Auto-replan failed after delete")

            elif msg.get("type") == "appt_schedule_check_availability":
                result = appointment_scheduler.check_availability(
                    target_date=msg.get("date", ""),
                    time_slot=msg.get("time_slot", ""),
                    duration_minutes=msg.get("duration_minutes", 60),
                )
                await websocket.send_json({
                    "type": "appt_availability_result",
                    **result,
                })

            elif msg.get("type") == "appt_schedule_day":
                day_appts = appointment_scheduler.get_day_schedule(msg.get("date", ""))
                await websocket.send_json({
                    "type": "appt_day_schedule",
                    "date": msg.get("date", ""),
                    "appointments": [a.to_dict() for a in day_appts],
                })

            elif msg.get("type") == "appt_schedule_week":
                week = appointment_scheduler.get_week_view(msg.get("start_date", ""))
                await websocket.send_json({
                    "type": "appt_week_view",
                    "start_date": msg.get("start_date", ""),
                    "week": week,
                })

            # --- Invoicing handlers ---
            elif msg.get("type") == "invoicing_create":
                inv = invoice_manager.create_invoice(
                    customer_name=msg.get("customer_name", ""),
                    customer_id=msg.get("customer_id", ""),
                    customer_email=msg.get("customer_email", ""),
                    customer_phone=msg.get("customer_phone", ""),
                    service_record_id=msg.get("service_record_id", ""),
                    invoice_date=msg.get("date", ""),
                    due_date=msg.get("due_date", ""),
                    line_items=msg.get("line_items", []),
                    labor_hours=float(msg.get("labor_hours", 0)),
                    labor_rate=float(msg.get("labor_rate", 0)) or None,
                    tax_rate=float(msg.get("tax_rate", 0)),
                    notes=msg.get("notes", ""),
                    appointment_id=msg.get("appointment_id", ""),
                )
                event_logger.info("invoicing", f"Created: {inv.invoice_number} for {inv.customer_name} — ${inv.total:.2f}")
                await websocket.send_json({"type": "invoicing_created", "ok": True, "invoice": inv.to_dict()})
                await broadcast_invoicing_status()

            elif msg.get("type") == "invoicing_create_from_service":
                inv = invoice_manager.create_from_service_record(
                    record_id=msg.get("record_id", ""),
                    parts_markup=float(msg.get("parts_markup", 0)),
                )
                if inv:
                    event_logger.info("invoicing", f"Created from service: {inv.invoice_number} — ${inv.total:.2f}")
                await websocket.send_json({
                    "type": "invoicing_created",
                    "ok": inv is not None,
                    "invoice": inv.to_dict() if inv else None,
                })
                if inv:
                    await broadcast_invoicing_status()

            elif msg.get("type") == "invoicing_update":
                inv_id = msg.get("invoice_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("customer_name", "customer_id", "customer_email",
                                    "customer_phone", "date", "due_date", "line_items",
                                    "labor_hours", "labor_rate", "tax_rate", "notes", "status")}
                inv = invoice_manager.update_invoice(inv_id, **updates)
                await websocket.send_json({
                    "type": "invoicing_updated",
                    "ok": inv is not None,
                    "invoice": inv.to_dict() if inv else None,
                })
                if inv:
                    await broadcast_invoicing_status()

            elif msg.get("type") == "invoicing_mark_paid":
                inv = invoice_manager.mark_paid(
                    invoice_id=msg.get("invoice_id", ""),
                    payment_method=msg.get("payment_method", ""),
                    paid_date=msg.get("paid_date", ""),
                )
                if inv:
                    event_logger.info("invoicing", f"Paid: {inv.invoice_number} via {inv.payment_method or 'unspecified'}")
                await websocket.send_json({
                    "type": "invoicing_paid",
                    "ok": inv is not None,
                    "invoice": inv.to_dict() if inv else None,
                })
                if inv:
                    await broadcast_invoicing_status()
                    # Auto-sync finances when invoice paid
                    finance_tracker.sync_from_invoices()
                    await broadcast_finances_status()

            elif msg.get("type") == "invoicing_mark_sent":
                inv = invoice_manager.mark_sent(msg.get("invoice_id", ""))
                if inv:
                    event_logger.info("invoicing", f"Sent: {inv.invoice_number}")
                await websocket.send_json({
                    "type": "invoicing_sent",
                    "ok": inv is not None,
                    "invoice": inv.to_dict() if inv else None,
                })
                if inv:
                    await broadcast_invoicing_status()

            elif msg.get("type") == "invoicing_delete":
                inv_id = msg.get("invoice_id", "")
                ok = invoice_manager.delete_invoice(inv_id)
                await websocket.send_json({
                    "type": "invoicing_deleted",
                    "ok": ok,
                    "invoice_id": inv_id,
                })
                if ok:
                    event_logger.info("invoicing", f"Deleted invoice {inv_id[:8]}")
                    await broadcast_invoicing_status()

            elif msg.get("type") == "invoicing_pdf":
                inv_id = msg.get("invoice_id", "")
                path = invoice_manager.generate_pdf(inv_id)
                await websocket.send_json({
                    "type": "invoicing_pdf_result",
                    "ok": path is not None,
                    "invoice_id": inv_id,
                    "url": f"/api/invoice-pdf/{inv_id}" if path else None,
                })
                if path:
                    await broadcast_invoicing_status()

            elif msg.get("type") == "invoicing_send":
                result = invoice_manager.send_invoice(msg.get("invoice_id", ""))
                await websocket.send_json({
                    "type": "invoicing_send_result",
                    **result,
                })
                await broadcast_invoicing_status()

            # --- Parts Inventory handlers ---
            elif msg.get("type") == "inventory_add_part":
                part = inventory_manager.add_part(
                    name=msg.get("name", ""),
                    part_number=msg.get("part_number", ""),
                    description=msg.get("description", ""),
                    category=msg.get("category", ""),
                    quantity_on_hand=int(msg.get("quantity_on_hand", 0)),
                    reorder_point=int(msg.get("reorder_point", 0)),
                    cost=float(msg.get("cost", 0)),
                    retail_price=float(msg.get("retail_price", 0)),
                    supplier=msg.get("supplier", ""),
                    location=msg.get("location", ""),
                    notes=msg.get("notes", ""),
                )
                event_logger.info(
                    "inventory",
                    f"Part added: {part.name} (P/N: {part.part_number or 'N/A'}) — qty {part.quantity_on_hand}",
                    part_id=part.short_id,
                )
                await websocket.send_json({"type": "inventory_part_added", "ok": True, "part": part.to_dict()})
                await broadcast_inventory_status()

            elif msg.get("type") == "inventory_update_part":
                pid = msg.get("part_id", "")
                updates = {k: v for k, v in msg.items() if k not in ("type", "part_id")}
                # Type coercion for numeric fields
                for int_field in ("quantity_on_hand", "reorder_point"):
                    if int_field in updates:
                        updates[int_field] = int(updates[int_field])
                for float_field in ("cost", "retail_price"):
                    if float_field in updates:
                        updates[float_field] = float(updates[float_field])
                updated = inventory_manager.update_part(pid, **updates)
                await websocket.send_json({
                    "type": "inventory_part_updated", "ok": updated is not None,
                    "part": updated.to_dict() if updated else None,
                })
                if updated:
                    await broadcast_inventory_status()

            elif msg.get("type") == "inventory_use_part":
                pid = msg.get("part_id", "")
                qty = int(msg.get("quantity", 1))
                reason = msg.get("reason", "manual")
                entry = inventory_manager.use_part(pid, quantity=qty, reason=reason)
                if entry:
                    event_logger.info(
                        "inventory",
                        f"Part used: {entry.part_name} x{entry.quantity_used} ({reason})",
                        part_id=entry.part_id[:8],
                    )
                await websocket.send_json({
                    "type": "inventory_part_used", "ok": entry is not None,
                    "usage": entry.to_dict() if entry else None,
                })
                if entry:
                    await broadcast_inventory_status()

            elif msg.get("type") == "inventory_delete_part":
                pid = msg.get("part_id", "")
                removed = inventory_manager.delete_part(pid)
                event_logger.info("inventory", f"Part deleted: {pid[:8]}", part_id=pid[:8])
                await websocket.send_json({"type": "inventory_part_deleted", "ok": removed})
                if removed:
                    await broadcast_inventory_status()

            elif msg.get("type") == "inventory_reorder_check":
                low = inventory_manager.reorder_check()
                await websocket.send_json({
                    "type": "inventory_reorder_result",
                    "parts": [p.to_dict() for p in low],
                })

            elif msg.get("type") == "inventory_cost_report":
                report = inventory_manager.cost_report()
                await websocket.send_json({
                    "type": "inventory_cost_report_result",
                    **report,
                })

            # --- Financial Dashboard handlers ---
            elif msg.get("type") == "finances_add_transaction":
                try:
                    txn = finance_tracker.add_transaction(
                        date_str=msg.get("date", ""),
                        txn_type=msg.get("txn_type", ""),
                        category=msg.get("category", ""),
                        amount=float(msg.get("amount", 0)),
                        description=msg.get("description", ""),
                        reference_id=msg.get("reference_id", ""),
                        metadata=msg.get("metadata"),
                    )
                    event_logger.info("finances", f"Added {txn.type}: ${txn.amount:.2f} ({txn.category})")
                    await websocket.send_json({"type": "finances_txn_added", "ok": True, "transaction": txn.to_dict()})
                    await broadcast_finances_status()
                except (ValueError, Exception) as e:
                    await websocket.send_json({"type": "finances_txn_added", "ok": False, "error": str(e)})

            elif msg.get("type") == "finances_update_transaction":
                kwargs = {k: v for k, v in msg.items() if k not in ("type", "txn_id")}
                txn = finance_tracker.update_transaction(msg.get("txn_id", ""), **kwargs)
                await websocket.send_json({
                    "type": "finances_txn_updated",
                    "ok": txn is not None,
                    "transaction": txn.to_dict() if txn else None,
                })
                if txn:
                    await broadcast_finances_status()

            elif msg.get("type") == "finances_delete_transaction":
                ok = finance_tracker.delete_transaction(msg.get("txn_id", ""))
                if ok:
                    event_logger.info("finances", f"Deleted transaction {msg.get('txn_id', '')[:8]}")
                await websocket.send_json({"type": "finances_txn_deleted", "ok": ok})
                if ok:
                    await broadcast_finances_status()

            elif msg.get("type") == "finances_sync_invoices":
                count = finance_tracker.sync_from_invoices()
                event_logger.info("finances", f"Invoice sync: {count} new transactions")
                await websocket.send_json({"type": "finances_sync_result", "ok": True, "count": count})
                await broadcast_finances_status()

            elif msg.get("type") == "finances_set_balance":
                _finance_state["balance"] = float(msg.get("balance", 0))
                event_logger.info("finances", f"Balance set to ${_finance_state['balance']:.2f}")
                await broadcast_finances_status()

            elif msg.get("type") == "finances_runway":
                balance = float(msg.get("balance", _finance_state["balance"]))
                runway = finance_tracker.runway_calculator(balance)
                await websocket.send_json({"type": "finances_runway_result", **runway})

            # --- Route Planner handlers ---
            elif msg.get("type") == "route_plan_day":
                try:
                    route = route_planner.plan_route(
                        date_str=msg.get("date", ""),
                        depart_time=msg.get("depart_time", "08:00"),
                    )
                    event_logger.info("route", f"Planned route for {route.date}: {route.stop_count} stops, {route.total_distance_miles:.1f} mi")
                    await websocket.send_json({
                        "type": "route_planned",
                        "ok": True,
                        "route": route.to_dict(),
                    })
                    await broadcast_route_status()
                except Exception as e:
                    logger.error("Route planning failed: %s", e)
                    await websocket.send_json({"type": "route_planned", "ok": False, "error": str(e)})

            elif msg.get("type") == "route_start":
                route = route_planner.start_route(msg.get("route_id", ""))
                await websocket.send_json({
                    "type": "route_started",
                    "ok": route is not None,
                    "route": route.to_dict() if route else None,
                })
                if route:
                    event_logger.info("route", f"Route started: {route.date}")
                    await broadcast_route_status()

            elif msg.get("type") == "route_complete_stop":
                stop = route_planner.complete_stop(
                    route_id=msg.get("route_id", ""),
                    stop_order=msg.get("stop_order", 0),
                    actual_arrival=msg.get("actual_arrival", ""),
                    actual_departure=msg.get("actual_departure", ""),
                )
                await websocket.send_json({
                    "type": "route_stop_completed",
                    "ok": stop is not None,
                    "stop": stop.to_dict() if stop else None,
                })
                if stop:
                    event_logger.info("route", f"Stop {stop.stop_order} completed: {stop.customer_name}")
                    await broadcast_route_status()

            elif msg.get("type") == "route_complete":
                route = route_planner.complete_route(
                    route_id=msg.get("route_id", ""),
                    actual_return=msg.get("actual_return", ""),
                )
                await websocket.send_json({
                    "type": "route_completed",
                    "ok": route is not None,
                    "route": route.to_dict() if route else None,
                })
                if route:
                    event_logger.info("route", f"Route completed: {route.date}")
                    await broadcast_route_status()

            elif msg.get("type") == "route_get_day":
                route = route_planner.get_day_plan(msg.get("date", ""))
                await websocket.send_json({
                    "type": "route_day_plan",
                    "ok": True,
                    "route": route.to_dict() if route else None,
                })

            # --- Highway 20 Documentary handlers ---
            elif msg.get("type") == "hwy20_add_segment":
                location = (msg.get("location_name") or "").strip()
                state = (msg.get("state") or "").strip().upper()
                if not location or not state:
                    await websocket.send_json({"type": "hwy20_segment_added", "ok": False, "error": "Location and state required"})
                    continue
                try:
                    seg = highway20_planner.add_segment(
                        location_name=location, state=state,
                        mile_marker_start=float(msg.get("mile_marker_start", 0)),
                        mile_marker_end=float(msg.get("mile_marker_end", 0)),
                        gps_lat=float(msg.get("gps_lat", 0)),
                        gps_lon=float(msg.get("gps_lon", 0)),
                        description=msg.get("description", ""),
                        terrain_type=msg.get("terrain_type", "plains"),
                        points_of_interest=msg.get("points_of_interest", []),
                        best_season=msg.get("best_season", "summer"),
                        priority=int(msg.get("priority", 2)),
                        planned_date=msg.get("planned_date", ""),
                    )
                    event_logger.info("highway20", f"Segment '{location}' ({seg.short_id}) added in {state}")
                    await websocket.send_json({"type": "hwy20_segment_added", "ok": True, "segment_id": seg.segment_id})
                    await broadcast_hwy20_status()
                except Exception as e:
                    logger.error("Highway 20 add segment failed: %s", e)
                    await websocket.send_json({"type": "hwy20_segment_added", "ok": False, "error": str(e)})

            elif msg.get("type") == "hwy20_update_segment":
                seg_id = msg.get("segment_id", "")
                fields = {}
                for key in ("location_name", "state", "mile_marker_start", "mile_marker_end",
                            "gps_lat", "gps_lon", "description", "terrain_type",
                            "points_of_interest", "filming_notes", "weather_conditions",
                            "best_season", "priority", "planned_date"):
                    if key in msg:
                        fields[key] = msg[key]
                seg = highway20_planner.update_segment(seg_id, **fields)
                if seg:
                    event_logger.info("highway20", f"Segment '{seg.location_name}' ({seg.short_id}) updated")
                    await websocket.send_json({"type": "hwy20_segment_updated", "ok": True})
                    await broadcast_hwy20_status()
                else:
                    await websocket.send_json({"type": "hwy20_segment_updated", "ok": False, "error": "Segment not found"})

            elif msg.get("type") == "hwy20_delete_segment":
                seg_id = msg.get("segment_id", "")
                removed = highway20_planner.delete_segment(seg_id)
                if removed:
                    event_logger.info("highway20", f"Segment {seg_id[:8]} deleted")
                    await websocket.send_json({"type": "hwy20_segment_deleted", "ok": True})
                    await broadcast_hwy20_status()
                else:
                    await websocket.send_json({"type": "hwy20_segment_deleted", "ok": False, "error": "Segment not found"})

            elif msg.get("type") == "hwy20_mark_filmed":
                seg_id = msg.get("segment_id", "")
                seg = highway20_planner.mark_filmed(
                    seg_id,
                    filming_notes=msg.get("filming_notes", ""),
                    weather_conditions=msg.get("weather_conditions", ""),
                )
                if seg:
                    event_logger.info("highway20", f"Segment '{seg.location_name}' marked filmed")
                    await websocket.send_json({"type": "hwy20_filmed", "ok": True})
                    await broadcast_hwy20_status()
                else:
                    await websocket.send_json({"type": "hwy20_filmed", "ok": False, "error": "Segment not found"})

            elif msg.get("type") == "hwy20_generate_shots":
                seg_id = msg.get("segment_id", "")
                try:
                    shots = await highway20_planner.shot_list_generator(seg_id)
                    await websocket.send_json({
                        "type": "hwy20_shots_generated",
                        "ok": True,
                        "shots": [s.to_dict() for s in shots],
                        "count": len(shots),
                    })
                    await broadcast_hwy20_status()
                except Exception as e:
                    logger.error("Highway 20 shot generation failed: %s", e)
                    await websocket.send_json({"type": "hwy20_shots_generated", "ok": False, "error": str(e)})

            elif msg.get("type") == "hwy20_add_shot":
                seg_id = msg.get("segment_id", "")
                shot = highway20_planner.add_shot(
                    segment_id=seg_id,
                    shot_type=msg.get("shot_type", "wide"),
                    description=msg.get("description", ""),
                    equipment=msg.get("equipment", ""),
                    time_of_day=msg.get("time_of_day", "any"),
                    duration_sec=int(msg.get("duration_sec", 30)),
                    notes=msg.get("notes", ""),
                )
                if shot:
                    await websocket.send_json({"type": "hwy20_shot_added", "ok": True, "shot_id": shot.shot_id})
                    await broadcast_hwy20_status()
                else:
                    await websocket.send_json({"type": "hwy20_shot_added", "ok": False, "error": "Segment not found"})

            elif msg.get("type") == "hwy20_update_shot":
                shot_id = msg.get("shot_id", "")
                fields = {}
                for key in ("shot_type", "description", "equipment", "time_of_day",
                            "duration_sec", "status", "notes"):
                    if key in msg:
                        fields[key] = msg[key]
                shot = highway20_planner.update_shot(shot_id, **fields)
                if shot:
                    await websocket.send_json({"type": "hwy20_shot_updated", "ok": True})
                    await broadcast_hwy20_status()
                else:
                    await websocket.send_json({"type": "hwy20_shot_updated", "ok": False, "error": "Shot not found"})

            elif msg.get("type") == "hwy20_delete_shot":
                shot_id = msg.get("shot_id", "")
                removed = highway20_planner.delete_shot(shot_id)
                if removed:
                    await websocket.send_json({"type": "hwy20_shot_deleted", "ok": True})
                    await broadcast_hwy20_status()
                else:
                    await websocket.send_json({"type": "hwy20_shot_deleted", "ok": False, "error": "Shot not found"})

            elif msg.get("type") == "hwy20_weather":
                seg_id = msg.get("segment_id", "")
                weather = highway20_planner.weather_planner(segment_id=seg_id)
                await websocket.send_json({"type": "hwy20_weather_result", "ok": True, "weather": weather})

            elif msg.get("type") == "hwy20_scouting_note":
                seg_id = msg.get("segment_id", "")
                note = (msg.get("note") or "").strip()
                if not note:
                    await websocket.send_json({"type": "hwy20_note_stored", "ok": False, "error": "Note is required"})
                    continue
                stored = highway20_planner.store_scouting_note(seg_id, note)
                await websocket.send_json({"type": "hwy20_note_stored", "ok": stored})

            elif msg.get("type") == "hwy20_search_notes":
                query = (msg.get("query") or "").strip()
                results = highway20_planner.search_scouting_notes(query) if query else []
                await websocket.send_json({"type": "hwy20_search_results", "ok": True, "results": results})

            # --- Scheduled Reports handlers ---
            elif msg.get("type") == "reports_generate":
                report_type = msg.get("report_type", "daily")
                try:
                    report = report_engine.generate_report(report_type)
                    await websocket.send_json({
                        "type": "reports_generate_result",
                        "ok": True,
                        "report": report.to_dict(),
                    })
                    await broadcast_reports_status()
                except Exception as e:
                    logger.error("Report generation failed: %s", e)
                    await websocket.send_json({
                        "type": "reports_generate_result",
                        "ok": False,
                        "error": str(e),
                    })

            elif msg.get("type") == "reports_get_html":
                report_id = msg.get("report_id", "")
                report = report_engine.get_report(report_id)
                if report:
                    await websocket.send_json({
                        "type": "reports_html_result",
                        "ok": True,
                        "report_id": report_id,
                        "html_content": report.html_content,
                        "title": report.title,
                    })
                else:
                    await websocket.send_json({
                        "type": "reports_html_result",
                        "ok": False,
                        "error": "Report not found",
                    })

            elif msg.get("type") == "reports_delete":
                report_id = msg.get("report_id", "")
                ok = report_engine.delete_report(report_id)
                await websocket.send_json({
                    "type": "reports_delete_result",
                    "ok": ok,
                })
                if ok:
                    await broadcast_reports_status()

            elif msg.get("type") == "knowledge_scan_inbox":
                results = knowledge_ingest.scan_inbox()
                await websocket.send_json({
                    "type": "knowledge_scan_result",
                    "ok": True,
                    "scanned": len(results),
                    "completed": sum(1 for d in results if d.status == "completed"),
                })
                await broadcast_knowledge_status()

            elif msg.get("type") == "approval_response":
                # Dashboard approving or rejecting a Tier 2 action
                task_id = msg.get("task_id", "")
                approved = msg.get("approved", False)
                orchestrator.resolve_approval(task_id, approved)
                action_str = "approved" if approved else "rejected"
                event_logger.info(
                    "security",
                    f"Tier 2 action {action_str} from dashboard (task {task_id[:8]})",
                    task_id=task_id[:8], action=action_str,
                )
                # Audit log update happens inside orchestrator.act()
                # Broadcast updated audit status after a brief delay
                await asyncio.sleep(0.2)
                await broadcast_security_audit_status()

            elif msg.get("type") == "set_client_mode":
                mode = msg.get("mode", "full")
                if mode in ("compact", "full"):
                    dashboard_clients[websocket]["mode"] = mode
                    logger.info("Dashboard client switched to %s mode", mode)

            elif msg.get("type") == "security_audit_filter":
                # Dashboard requesting filtered audit entries
                risk_level = msg.get("risk_level") or None
                actor = msg.get("actor") or None
                result_filter = msg.get("result") or None
                tier = msg.get("tier")
                limit = min(int(msg.get("limit", 200)), 500)

                if tier is not None:
                    try:
                        tier = int(tier)
                    except (ValueError, TypeError):
                        tier = None

                entries = security_audit_log.filter_entries(
                    risk_level=risk_level,
                    actor=actor,
                    result=result_filter,
                    tier=tier,
                    limit=limit,
                )
                await websocket.send_json({
                    "type": "security_audit_filtered",
                    "entries": [e.to_dict() for e in entries],
                })

            elif msg.get("type") == "notification_dismiss":
                notif_id = msg.get("id", "")
                notification_manager.dismiss(notif_id)
                await broadcast_to_dashboards({
                    "type": "notification_count",
                    "unread_count": notification_manager.get_unread_count(),
                })

            elif msg.get("type") == "notification_dismiss_all":
                notification_manager.dismiss_all()
                await broadcast_to_dashboards({
                    "type": "notification_count",
                    "unread_count": notification_manager.get_unread_count(),
                })

            elif msg.get("type") == "run_self_test":
                # Run the system self-test suite
                self_test = SelfTest(
                    config=config,
                    event_logger=event_logger,
                    router=orchestrator.router,
                    registry=registry,
                    task_queue=task_queue,
                    short_term_memory=short_term_memory,
                    working_memory=working_memory,
                    episodic_memory=episodic_memory,
                    long_term_memory=long_term_memory,
                    analytics=analytics,
                    content_calendar=content_calendar,
                    research_store=research_store,
                    security_audit_log=security_audit_log,
                    agent_websockets=agent_websockets,
                    port=config.dashboard.port,
                )
                results = await self_test.run_all()
                last_self_test_results = results
                await broadcast_to_dashboards({
                    "type": "self_test_results",
                    **results,
                })

            elif msg.get("type") == "run_launch_readiness":
                # Run the launch readiness integration test suite
                async def lr_progress(completed, total, result):
                    """Broadcast real-time progress to all dashboards."""
                    await broadcast_to_dashboards({
                        "type": "launch_readiness_progress",
                        "completed": completed,
                        "total": total,
                        "result": result,
                    })

                lr = LaunchReadiness(
                    config=config,
                    event_logger=event_logger,
                    router=orchestrator.router,
                    registry=registry,
                    task_queue=task_queue,
                    short_term_memory=short_term_memory,
                    working_memory=working_memory,
                    episodic_memory=episodic_memory,
                    long_term_memory=long_term_memory,
                    analytics=analytics,
                    security_audit_log=security_audit_log,
                    agent_websockets=agent_websockets,
                    port=config.dashboard.port,
                    orchestrator=orchestrator,
                    notification_manager=notification_manager,
                    backup_manager=backup_manager,
                    telegram_bot=telegram_bot,
                    content_agent=content_agent,
                    research_agent=research_agent,
                    business_agent=business_agent,
                    self_test_class=SelfTest,
                    on_progress=lr_progress,
                )
                results = await lr.run_all()
                last_launch_readiness_results = results
                await broadcast_to_dashboards({
                    "type": "launch_readiness_results",
                    **results,
                })

            elif msg.get("type") == "tts_synthesize":
                tts_text = (msg.get("text") or "").strip()
                tts_voice = msg.get("voice") or None
                if not tts_text:
                    await websocket.send_json({
                        "type": "tts_result",
                        "ok": False,
                        "error": "Text is required",
                    })
                    continue

                tts_result = await tts_pipeline.synthesize(
                    text=tts_text, voice=tts_voice,
                )
                result_data = {
                    "type": "tts_result",
                    "ok": tts_result.error is None,
                    **tts_result.to_dict(),
                }
                await broadcast_to_dashboards(result_data)
                if not tts_result.error:
                    event_logger.info(
                        "tts",
                        f"Synthesized audio: {tts_result.audio_path} ({tts_result.duration_secs:.1f}s)",
                        voice=tts_result.voice,
                    )

            elif msg.get("type") == "tts_queue":
                tts_items = msg.get("items") or []
                tts_results = await tts_pipeline.queue_synthesis(tts_items)
                await broadcast_to_dashboards({
                    "type": "tts_queue_result",
                    "results": [r.to_dict() for r in tts_results],
                })

            elif msg.get("type") == "tts_list_audio":
                await websocket.send_json({
                    "type": "tts_audio_list",
                    "audio": tts_pipeline.list_audio(),
                })

            elif msg.get("type") == "tts_list_voices":
                await websocket.send_json({
                    "type": "tts_voices",
                    "voices": tts_pipeline.list_voices(),
                })

            elif msg.get("type") == "tts_delete":
                tts_filename = msg.get("filename", "")
                tts_error = tts_pipeline.delete_audio(tts_filename)
                if tts_error:
                    await websocket.send_json({
                        "type": "tts_deleted",
                        "ok": False,
                        "error": tts_error,
                    })
                else:
                    event_logger.info("tts", f"Audio deleted: {tts_filename}")
                    await broadcast_to_dashboards({
                        "type": "tts_deleted",
                        "ok": True,
                        "filename": tts_filename,
                    })
                    # Broadcast updated audio list
                    await broadcast_to_dashboards({
                        "type": "tts_audio_list",
                        "audio": tts_pipeline.list_audio(),
                    })

            elif msg.get("type") == "tts_status":
                await websocket.send_json({
                    "type": "tts_status",
                    **tts_pipeline.get_status(),
                })

            elif msg.get("type") == "episodic_search":
                # Search episodic memory with optional filters
                keyword = msg.get("keyword", "")
                start_time = msg.get("start_time")
                end_time = msg.get("end_time")
                participant = msg.get("participant")
                task_id = msg.get("task_id")
                limit = min(int(msg.get("limit", 50)), 200)
                offset = int(msg.get("offset", 0))

                results = episodic_memory.search(
                    keyword=keyword or None,
                    start_time=start_time,
                    end_time=end_time,
                    participant=participant,
                    task_id=task_id,
                    limit=limit,
                    offset=offset,
                )
                await websocket.send_json({
                    "type": "episodic_search_results",
                    "episodes": [ep.to_dict() for ep in results],
                    "keyword": keyword,
                    "count": len(results),
                })

            # --- Message Bus: publish from dashboard ---
            elif msg.get("type") == "bus_publish":
                sender = msg.get("sender", "dashboard")
                channel = msg.get("channel", "")
                payload = msg.get("payload", {})
                result = await message_bus.publish(sender, channel, payload)
                await websocket.send_json({
                    "type": "bus_publish_result",
                    "ok": result is not None,
                    "message_id": result.message_id if result else None,
                })

            # --- Message Bus: fetch history ---
            elif msg.get("type") == "bus_history":
                channel = msg.get("channel")  # None = all channels
                count = min(int(msg.get("count", 50)), 200)
                await websocket.send_json({
                    "type": "bus_history_result",
                    "messages": message_bus.get_recent(channel=channel, count=count),
                    "stats": message_bus.get_stats(),
                    "matrix": message_bus.get_sender_channel_matrix(),
                })

    except WebSocketDisconnect:
        dashboard_clients.pop(websocket, None)
        logger.info("Dashboard client disconnected")


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.dashboard.host,
        port=config.dashboard.port,
        log_level=config.dashboard.log_level,
    )
