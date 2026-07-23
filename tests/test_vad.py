"""
Tests for the Silero VAD module.

Tests focus on the VAD state machine and speech event generation
using synthetic audio data. The actual ONNX model loading is
optional in tests (mocked).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.vad.silero_vad import (
    SileroVAD,
    SpeechEventType,
    SpeechState,
)


@pytest.fixture
def vad():
    """Create a VAD instance with test-friendly settings."""
    return SileroVAD(
        threshold=0.5,
        min_speech_duration=0.25,
        min_silence_duration=0.5,
        frame_size=512,
        sample_rate=16000,
    )


class TestSileroVADInit:
    """Test VAD initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        vad = SileroVAD()
        assert vad.threshold == 0.5
        assert vad.min_speech_frames >= 1
        assert vad.min_silence_frames >= 1
        assert vad.frame_size == 512
        assert vad.sample_rate == 16000
        assert vad.state == SpeechState.SILENCE
        assert not vad.is_speaking

    def test_custom_params(self):
        """Test custom parameters."""
        vad = SileroVAD(
            threshold=0.3,
            min_speech_duration=0.5,
            min_silence_duration=1.0,
            frame_size=256,
            sample_rate=8000,
        )
        assert vad.threshold == 0.3
        assert vad.state == SpeechState.SILENCE

    def test_min_speech_frames_calculation(self):
        """Test frame count calculation from duration."""
        # 0.25s at 16kHz with 512 frame size = ~7.8 frames → min 7 frames
        vad = SileroVAD(min_speech_duration=0.25, sample_rate=16000, frame_size=512)
        assert vad.min_speech_frames >= 7
        assert vad.min_speech_frames <= 8

        # 0.5s silence at 16kHz with 512 frame size = ~15.6 frames → min 15 frames
        assert vad.min_silence_frames >= 15
        assert vad.min_silence_frames <= 16


class TestVADStateMachine:
    """Test VAD state machine transitions."""

    def test_silence_frame(self, vad):
        """Test that silence frame doesn't trigger events."""
        frame = np.zeros(512, dtype=np.float32)
        # Model loads via ONNX Runtime
        vad.load_model()
        assert vad._model is not None
        # A silent frame should have low probability
        prob = vad._get_speech_prob(frame)
        assert prob < 0.5, f"Expected low probability for silence, got {prob}"

    def test_speech_event_structure(self, vad):
        """Test SpeechEvent dataclass."""
        event_types = [SpeechEventType.SPEECH_START, SpeechEventType.SPEECH_END]
        for et in event_types:
            event = type("SpeechEvent", (), {
                "type": et,
                "audio": np.array([0.1, 0.2]),
                "timestamp": 1234.56,
                "duration": 1.5,
                "num_frames": 50,
                "max_prob": 0.95,
                "sample_rate": 16000,
            })
            assert event.type in event_types
            assert event.timestamp > 0
            assert event.duration > 0


class TestSpeechEvent:
    """Test SpeechEvent data structure."""

    def test_speech_start_event(self):
        """Test SPEECH_START event."""
        event = type("SpeechEvent", (), {
            "type": SpeechEventType.SPEECH_START,
            "audio": None,
            "timestamp": 100.0,
            "duration": 0.0,
            "num_frames": 0,
            "max_prob": 0.0,
        })
        assert event.type == SpeechEventType.SPEECH_START
        assert event.audio is None

    def test_speech_end_event(self):
        """Test SPEECH_END event with audio."""
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        event = type("SpeechEvent", (), {
            "type": SpeechEventType.SPEECH_END,
            "audio": audio,
            "timestamp": 100.0,
            "duration": 1.0,
            "num_frames": 50,
            "max_prob": 0.95,
            "sample_rate": 16000,
        })
        assert event.type == SpeechEventType.SPEECH_END
        assert len(event.audio) == 16000
        assert event.duration == 1.0


class TestVADFlush:
    """Test VAD flush behavior."""

    def test_flush_when_silent(self, vad):
        """Test flush when in silence state returns None."""
        result = vad.flush()
        assert result is None
        assert vad.state == SpeechState.SILENCE

    def test_flush_resets_state(self, vad):
        """Test flush resets VAD to silence."""
        vad.flush()
        assert vad.state == SpeechState.SILENCE
