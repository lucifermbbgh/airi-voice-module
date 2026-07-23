"""
Tests for the STT (Speech-to-Text) module.

Tests focus on:
- STTResult dataclass integrity
- FastWhisperSTT initialization and configuration
- Audio preprocessing (sample rate validation, silence detection)
- Mocked inference to verify async behavior

Actual model inference tests are in test_stt_integration.py
to keep unit tests fast and dependency-free.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.stt.faster_whisper_stt import FasterWhisperSTT, STTResult


@pytest.fixture
def stt() -> FasterWhisperSTT:
    """Create a STT instance with test-friendly settings (no model loaded)."""
    return FasterWhisperSTT(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        language="zh",
        model_dir="/tmp/stt_test_models",
    )


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate a synthetic 1-second audio segment at 16kHz."""
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440Hz sine wave + noise (simulates speech-like signal)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio += 0.05 * np.random.randn(len(t))
    return audio.astype(np.float32)


# ─── STTResult Tests ──────────────────────────────────────────────


class TestSTTResult:
    """STTResult dataclass integrity tests."""

    def test_minimal_result(self):
        """STTResult can be created with minimal fields."""
        result = STTResult(
            text="你好世界",
            confidence=0.95,
            language="zh",
            language_probability=0.99,
            duration=2.0,
            inference_time=0.5,
        )
        assert result.text == "你好世界"
        assert result.confidence == 0.95

    def test_result_with_segments(self):
        """STTResult supports optional segments."""
        result = STTResult(
            text="今天天气怎么样",
            confidence=0.92,
            language="zh",
            language_probability=0.98,
            duration=2.5,
            inference_time=0.3,
            segments=[
                {"start": 0.0, "end": 1.0, "text": "今天"},
                {"start": 1.0, "end": 2.5, "text": "天气怎么样"},
            ],
        )
        assert len(result.segments) == 2
        assert result.segments[0]["text"] == "今天"

    def test_result_empty_text(self):
        """STTResult can represent failed/no-speech detection."""
        result = STTResult(
            text="",
            confidence=0.0,
            language="zh",
            language_probability=0.0,
            duration=0.0,
            inference_time=0.0,
        )
        assert result.text == ""
        assert result.confidence == 0.0


# ─── FasterWhisperSTT Initialization Tests ───────────────────────


class TestFasterWhisperSTTInit:
    """STT engine initialization tests."""

    def test_default_initialization(self, stt: FasterWhisperSTT):
        """Engine initializes with default parameters."""
        assert stt.model_size == "tiny"
        assert stt.device == "cpu"
        assert stt.compute_type == "int8"
        assert stt.language == "zh"

    def test_model_not_loaded_by_default(self, stt: FasterWhisperSTT):
        """Model should not be loaded on init (lazy loading)."""
        assert not stt.is_loaded

    def test_valid_model_sizes(self):
        """All predefined model sizes are recognized."""
        for size in FasterWhisperSTT.MODELS:
            stt = FasterWhisperSTT(model_size=size)
            assert stt.model_size == size

    @pytest.mark.parametrize("invalid_size", ["invalid", "xlarge", ""])
    def test_invalid_model_size(self, invalid_size: str):
        """Invalid model size raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            FasterWhisperSTT(model_size=invalid_size)

    def test_hotwords(self):
        """Hotwords are stored and accessible."""
        stt = FasterWhisperSTT(hotwords=["Claude", "AIRI", "Silero"])
        assert len(stt.hotwords) == 3
        assert "Claude" in stt.hotwords


# ─── Audio Preprocessing Tests ──────────────────────────────────


class TestAudioPreprocessing:
    """Audio validation and preprocessing tests."""

    def test_valid_audio_format(self, stt: FasterWhisperSTT, sample_audio):
        """STT validates correct audio format."""
        assert stt._validate_audio(sample_audio, 16000) is True

    def test_invalid_sample_rate(self, stt: FasterWhisperSTT, sample_audio):
        """Non-16kHz audio is flagged."""
        assert stt._validate_audio(sample_audio, 48000) is False

    def test_invalid_dtype(self, stt: FasterWhisperSTT):
        """Non-float32 audio is flagged."""
        audio_int16 = np.zeros(16000, dtype=np.int16)
        assert stt._validate_audio(audio_int16, 16000) is False

    def test_empty_audio(self, stt: FasterWhisperSTT):
        """Empty audio is flagged."""
        empty = np.array([], dtype=np.float32)
        assert stt._validate_audio(empty, 16000) is False

    def test_too_short_audio(self, stt: FasterWhisperSTT):
        """Very short audio (<0.1s) is flagged."""
        short = np.zeros(100, dtype=np.float32)
        assert stt._validate_audio(short, 16000) is False

    def test_too_long_audio(self, stt: FasterWhisperSTT):
        """Very long audio (>30s) is flagged for real-time use."""
        long_audio = np.zeros(16000 * 60, dtype=np.float32)  # 60s
        assert stt._validate_audio(long_audio, 16000) is False

    def test_silence_detection(self, stt: FasterWhisperSTT):
        """Near-silence audio returns empty result through fast path."""
        silence = np.zeros(16000, dtype=np.float32)  # 1s silence
        assert stt._is_silence(silence)

    def test_speech_detection(self, stt: FasterWhisperSTT, sample_audio):
        """Audio with signal is not silence."""
        assert not stt._is_silence(sample_audio)


# ─── Async Transcribe Tests ─────────────────────────────────────


class TestTranscribe:
    """Transcribe method tests (mocked, no real model)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("side_effect,expected_match", [
        (ImportError("faster-whisper not installed"), r"faster-whisper"),
    ])
    async def test_transcribe_raises_before_load(
        self, stt: FasterWhisperSTT, sample_audio,
        side_effect: Exception, expected_match: str,
    ):
        """Transcribe raises error when model lazy-load fails."""
        import unittest.mock as mock

        with mock.patch.object(stt, 'load_model', side_effect=side_effect):
            with pytest.raises(Exception, match=expected_match):
                await stt.transcribe(sample_audio)
        # After the error, model should still not be loaded
        assert not stt.is_loaded

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio(self, stt: FasterWhisperSTT):
        """Transcribe with empty audio returns empty result."""
        empty = np.array([], dtype=np.float32)
        result = await stt.transcribe(empty)
        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_silence(self, stt: FasterWhisperSTT):
        """Transcribe near-silence returns empty result."""
        silence = np.zeros(16000, dtype=np.float32)
        result = await stt.transcribe(silence)
        assert result.text == ""
        assert result.confidence == 0.0
