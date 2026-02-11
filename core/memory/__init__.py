"""
CAM Memory System

Four-tier memory architecture:
    - Short-term:  Current session context (in-memory, cleared on restart)
    - Working:     Active task state (JSON-persisted, survives restarts)
    - Episodic:    Timestamped conversation logs (future)
    - Long-term:   Persistent knowledge base via ChromaDB (future)
"""

from core.memory.short_term import ShortTermMemory
from core.memory.working import WorkingMemory

__all__ = ["ShortTermMemory", "WorkingMemory"]
