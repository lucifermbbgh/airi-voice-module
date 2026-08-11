"""Unit tests for ConversationContext and Turn.

Phase 4: LLM Integration — conversation management tests.
"""

from __future__ import annotations

import pytest

from src.airi.conversation import (
    ConversationContext,
    Turn,
    TurnRole,
    TurnStatus,
)


# ═══════════════════════════════════════════════════════════════
# Turn tests
# ═══════════════════════════════════════════════════════════════

class TestTurn:
    """Tests for the Turn dataclass."""

    def test_default_turn_is_user(self) -> None:
        """New turn defaults to USER role and PENDING status."""
        turn = Turn()
        assert turn.role == TurnRole.USER
        assert turn.status == TurnStatus.PENDING
        assert turn.text == ""
        assert len(turn.turn_id) == 12

    def test_turn_has_unique_id(self) -> None:
        """Each turn gets a unique identifier."""
        t1 = Turn()
        t2 = Turn()
        assert t1.turn_id != t2.turn_id

    def test_append_response_updates_status(self) -> None:
        """Appending a response chunk transitions from pending to streaming."""
        turn = Turn()
        assert turn.status == TurnStatus.PENDING
        turn.append_response("Hello")
        assert turn.status == TurnStatus.STREAMING
        assert turn.response_text == "Hello"

    def test_append_response_accumulates(self) -> None:
        """Multiple append_response calls accumulate text."""
        turn = Turn()
        turn.append_response("Hello, ")
        turn.append_response("world!")
        assert turn.response_text == "Hello, world!"

    def test_mark_complete(self) -> None:
        """mark_complete sets status and timestamp."""
        turn = Turn(text="What is AI?")
        turn.append_response("AI is artificial intelligence.")
        turn.mark_complete()

        assert turn.status == TurnStatus.COMPLETE
        assert turn.response_timestamp is not None
        assert turn.duration >= 0

    def test_mark_error(self) -> None:
        """mark_error sets status and records error message."""
        turn = Turn()
        turn.mark_error("STT timeout")
        assert turn.status == TurnStatus.ERROR
        assert turn.metadata["error"] == "STT timeout"

    def test_mark_interrupted(self) -> None:
        """mark_interrupted sets status and timestamp."""
        turn = Turn(text="Long question...")
        turn.append_response("The answer is...")
        turn.mark_interrupted()

        assert turn.status == TurnStatus.INTERRUPTED
        assert turn.response_timestamp is not None


# ═══════════════════════════════════════════════════════════════
# ConversationContext tests
# ═══════════════════════════════════════════════════════════════

class TestConversationContext:
    """Tests for ConversationContext."""

    def test_new_context_is_empty(self) -> None:
        """Fresh context has no turns."""
        ctx = ConversationContext()
        assert ctx.turn_count == 0
        assert len(ctx.turns) == 0
        assert ctx.active_turn is None
        assert ctx.last_user_text is None

    def test_start_user_turn(self) -> None:
        """start_user_turn creates a user turn."""
        ctx = ConversationContext()
        turn = ctx.start_user_turn("Hello AIRI", confidence=0.95)

        assert turn.role == TurnRole.USER
        assert turn.text == "Hello AIRI"
        assert turn.confidence == 0.95
        assert turn.status == TurnStatus.PENDING
        assert len(ctx.turns) == 1
        assert ctx.active_turn is turn
        assert ctx.last_user_text == "Hello AIRI"

    def test_append_response_to_active_turn(self) -> None:
        """append_response appends to the last (user) turn."""
        ctx = ConversationContext()
        ctx.start_user_turn("What is Python?")

        turn = ctx.append_response("Python is a ")
        assert turn is not None
        assert turn.status == TurnStatus.STREAMING

        ctx.append_response("programming language.")
        assert turn.response_text == "Python is a programming language."

    def test_append_response_with_no_turns(self) -> None:
        """append_response on empty context returns None."""
        ctx = ConversationContext()
        result = ctx.append_response("orphan chunk")
        assert result is None

    def test_complete_current_turn(self) -> None:
        """complete_current_turn marks the active turn as done."""
        ctx = ConversationContext()
        ctx.start_user_turn("How are you?")
        ctx.append_response("I'm doing great!")

        turn = ctx.complete_current_turn()
        assert turn is not None
        assert turn.status == TurnStatus.COMPLETE
        assert ctx.turn_count == 1

    def test_error_current_turn(self) -> None:
        """error_current_turn marks the active turn as errored."""
        ctx = ConversationContext()
        ctx.start_user_turn("Test")
        turn = ctx.error_current_turn("STT failed")
        assert turn is not None
        assert turn.status == TurnStatus.ERROR
        assert turn.metadata["error"] == "STT failed"

    def test_recent_history_format(self) -> None:
        """recent_history returns LLM-compatible dicts."""
        ctx = ConversationContext()
        t1 = ctx.start_user_turn("Who are you?")
        t1.append_response("I am AIRI.")
        t1.mark_complete()

        t2 = ctx.start_user_turn("What can you do?")
        t2.append_response("I can chat with you.")
        t2.mark_complete()

        history = ctx.recent_history
        assert len(history) == 4  # 2 turns → 4 entries: user/assistant/user/assistant
        assert history[0] == {"role": "user", "content": "Who are you?"}
        assert history[1] == {"role": "assistant", "content": "I am AIRI."}
        assert history[2] == {"role": "user", "content": "What can you do?"}
        assert history[3] == {"role": "assistant", "content": "I can chat with you."}

    def test_history_limit_enforced(self) -> None:
        """Context trims turns beyond history_limit."""
        ctx = ConversationContext(history_limit=3)

        for i in range(5):
            t = ctx.start_user_turn(f"Msg {i}")
            t.mark_complete()

        assert len(ctx.turns) == 3
        assert ctx.turns[0].text == "Msg 2"  # Oldest remaining
        assert ctx.turns[-1].text == "Msg 4"  # Newest

    def test_session_isolation(self) -> None:
        """Each context gets a unique session ID."""
        ctx1 = ConversationContext()
        ctx2 = ConversationContext()
        assert ctx1.session_id != ctx2.session_id

    def test_clear_resets_state(self) -> None:
        """clear() wipes turns and generates new session."""
        ctx = ConversationContext()
        old_id = ctx.session_id
        ctx.start_user_turn("Test")
        assert len(ctx.turns) == 1

        ctx.clear()
        assert len(ctx.turns) == 0
        assert ctx.session_id != old_id
        assert ctx.turn_count == 0

    def test_summary(self) -> None:
        """summary() returns session statistics."""
        ctx = ConversationContext()
        ctx.start_user_turn("Hello")
        ctx.append_response("Hi!")
        ctx.complete_current_turn()

        summary = ctx.summary()
        assert "session_id" in summary
        assert summary["turn_count"] == 1
        assert summary["total_turns"] == 1
        assert summary["duration_s"] >= 0

    def test_multiple_streaming_chunks(self) -> None:
        """Streaming responses with many small chunks."""
        ctx = ConversationContext()
        ctx.start_user_turn("Tell me a story")

        chunks = ["Once ", "upon ", "a ", "time, ", "there ", "was ", "a ", "robot."]
        for chunk in chunks:
            ctx.append_response(chunk)

        ctx.complete_current_turn()
        assert ctx.turns[-1].response_text == "Once upon a time, there was a robot."

    def test_active_turn_none_after_complete(self) -> None:
        """After completing a turn, active_turn returns None."""
        ctx = ConversationContext()
        ctx.start_user_turn("Hi")
        ctx.complete_current_turn()
        assert ctx.active_turn is None

    def test_start_turn_while_previous_active(self) -> None:
        """Starting a new user turn while previous is streaming — 
        the previous remains incomplete (Phase 5 interruption pattern)."""
        ctx = ConversationContext()
        t1 = ctx.start_user_turn("First question")
        ctx.append_response("Answering first...")

        # Start a new turn without completing the first
        t2 = ctx.start_user_turn("Second question")
        assert t1.status == TurnStatus.STREAMING  # Not forcibly closed
        assert t2.status == TurnStatus.PENDING
        assert ctx.active_turn is t2
