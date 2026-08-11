"""
Conversation context manager for AIRI Voice Module.

Maintains multi-turn dialogue state across the voice pipeline,
tracking sessions, utterances, and response correlation.

Phase 4: LLM Integration — conversation layer.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)


class TurnRole(str, Enum):
    """Speaker role in a conversation turn."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TurnStatus(str, Enum):
    """Status of a conversation turn."""
    PENDING = "pending"         # STT done, waiting for LLM
    STREAMING = "streaming"     # LLM is streaming response
    COMPLETE = "complete"       # Response fully received
    INTERRUPTED = "interrupted" # User interrupted (Phase 5)
    ERROR = "error"             # Turn failed


@dataclass
class Turn:
    """A single conversation turn (user utterance + assistant response).

    Attributes:
        turn_id: Unique turn identifier.
        role: Speaker role.
        text: The text content.
        timestamp: When the turn started.
        confidence: STT confidence (user turns only).
        language: Detected language (user turns only).
        status: Processing status.
        response_chunks: Accumulated LLM response text chunks.
        response_timestamp: When the assistant response completed.
        metadata: Arbitrary additional data.
    """
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: TurnRole = TurnRole.USER
    text: str = ""
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.0
    language: str = "zh"
    status: TurnStatus = TurnStatus.PENDING

    # Accumulated response
    response_chunks: list[str] = field(default_factory=list)
    response_timestamp: float | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def response_text(self) -> str:
        """Get complete accumulated response text."""
        return "".join(self.response_chunks)

    @property
    def duration(self) -> float:
        """Turn duration in seconds (from utterance to response complete)."""
        end = self.response_timestamp or time.time()
        return end - self.timestamp

    def append_response(self, chunk: str) -> None:
        """Append a response chunk from streaming LLM output.

        Args:
            chunk: Text chunk from LLM.
        """
        self.response_chunks.append(chunk)
        if self.status == TurnStatus.PENDING:
            self.status = TurnStatus.STREAMING

    def mark_complete(self) -> None:
        """Mark the turn as complete (response fully received)."""
        self.status = TurnStatus.COMPLETE
        self.response_timestamp = time.time()
        logger.debug(
            "Turn {} complete: user=\"{}\" → assistant=\"{}\" ({:.1f}s)",
            self.turn_id,
            self.text[:50],
            self.response_text[:50],
            self.duration,
        )

    def mark_error(self, error: str) -> None:
        """Mark the turn as errored.

        Args:
            error: Error description.
        """
        self.status = TurnStatus.ERROR
        self.metadata["error"] = error
        logger.warning("Turn {} error: {}", self.turn_id, error)

    def mark_interrupted(self) -> None:
        """Mark the turn as interrupted by user (Phase 5)."""
        self.status = TurnStatus.INTERRUPTED
        self.response_timestamp = time.time()
        logger.debug("Turn {} interrupted", self.turn_id)


@dataclass
class ConversationContext:
    """Multi-turn conversation state manager.

    Tracks all turns in the current conversation session,
    provides history for LLM context, and handles interruptions.

    Attributes:
        session_id: Unique session identifier.
        history_limit: Maximum turns to retain in history.
        turns: Ordered list of conversation turns.
        created_at: Session creation timestamp.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    history_limit: int = 20
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # ── Turn lifecycle ──────────────────────────────────────────

    def start_user_turn(
        self,
        text: str,
        confidence: float = 0.0,
        language: str = "zh",
    ) -> Turn:
        """Begin a new user turn.

        Args:
            text: Transcribed user utterance.
            confidence: STT confidence score.
            language: Detected language.

        Returns:
            The newly created Turn.
        """
        turn = Turn(
            role=TurnRole.USER,
            text=text,
            confidence=confidence,
            language=language,
            status=TurnStatus.PENDING,
        )
        self.turns.append(turn)

        # Trim history
        while len(self.turns) > self.history_limit:
            removed = self.turns.pop(0)
            logger.debug("Trimmed old turn {}", removed.turn_id)

        logger.info(
            "Turn {} — user: \"{}\" (conf={:.2f}, {} chars)",
            turn.turn_id, text[:60], confidence, len(text),
        )
        return turn

    def append_response(self, chunk: str) -> Turn | None:
        """Append a response chunk from LLM to the current turn.

        Args:
            chunk: Text chunk from streaming LLM output.

        Returns:
            The current (last) turn, or None if no active turn.
        """
        if not self.turns:
            logger.warning("append_response called with no turns")
            return None

        current = self.turns[-1]
        current.append_response(chunk)
        return current

    def complete_current_turn(self) -> Turn | None:
        """Mark the current (last user) turn as complete.

        Returns:
            The completed turn, or None.
        """
        if not self.turns:
            return None
        current = self.turns[-1]
        current.mark_complete()
        return current

    def error_current_turn(self, error: str) -> Turn | None:
        """Mark the current turn as errored.

        Args:
            error: Error description.

        Returns:
            The errored turn, or None.
        """
        if not self.turns:
            return None
        current = self.turns[-1]
        current.mark_error(error)
        return current

    # ── History utilities ───────────────────────────────────────

    @property
    def recent_history(self) -> list[dict[str, str]]:
        """Get recent turns as alternating user/assistant pairs for LLM context.

        Returns:
            List of {"role": "user"|"assistant", "content": "..."} dicts,
            suitable for injection into LLM conversation history.
        """
        history: list[dict[str, str]] = []
        for t in self.turns:
            if t.status not in (TurnStatus.COMPLETE, TurnStatus.STREAMING):
                continue
            # User utterance
            if t.text:
                history.append({"role": "user", "content": t.text})
            # Assistant response
            if t.response_text:
                history.append({"role": "assistant", "content": t.response_text})
        return history

    @property
    def last_user_text(self) -> str | None:
        """Get the most recent user utterance."""
        for t in reversed(self.turns):
            if t.role == TurnRole.USER:
                return t.text
        return None

    @property
    def turn_count(self) -> int:
        """Number of completed turns."""
        return sum(1 for t in self.turns if t.status == TurnStatus.COMPLETE)

    @property
    def active_turn(self) -> Turn | None:
        """Get the currently active (pending/streaming) turn."""
        if not self.turns:
            return None
        last = self.turns[-1]
        if last.status in (TurnStatus.PENDING, TurnStatus.STREAMING):
            return last
        return None

    # ── Session management ──────────────────────────────────────

    def clear(self) -> None:
        """Reset the conversation (start a new session)."""
        self.session_id = uuid.uuid4().hex[:8]
        self.turns.clear()
        logger.info("Conversation cleared, new session: {}", self.session_id)

    def summary(self) -> dict[str, Any]:
        """Get a summary of the conversation state.

        Returns:
            Dictionary with session info and statistics.
        """
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "total_turns": len(self.turns),
            "duration_s": time.time() - self.created_at,
            "last_user_text": self.last_user_text,
            "active_turn": self.active_turn.turn_id if self.active_turn else None,
        }
