"""AIRI WebSocket client, protocol modules, and conversation management.

Phase 4: Full voice pipeline — conversation context, event routing, error recovery.
"""

from src.airi.conversation import ConversationContext, Turn, TurnRole, TurnStatus
from src.airi.websocket_client import AIRIClient, AIRIEventType

__all__ = [
    "AIRIClient",
    "AIRIEventType",
    "ConversationContext",
    "Turn",
    "TurnRole",
    "TurnStatus",
]
