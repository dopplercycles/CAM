"""
CAM Memory System

Four-tier memory architecture:
    - Short-term:  Current session context (in-memory, cleared on restart)
    - Working:     Active task state (JSON-persisted, survives restarts)
    - Episodic:    Timestamped conversation logs (SQLite-backed)
    - Long-term:   Persistent knowledge base via ChromaDB vector store
"""

from core.memory.short_term import ShortTermMemory
from core.memory.working import WorkingMemory
from core.memory.long_term import LongTermMemory
from core.memory.episodic import EpisodicMemory

__all__ = ["ShortTermMemory", "WorkingMemory", "LongTermMemory", "EpisodicMemory"]
