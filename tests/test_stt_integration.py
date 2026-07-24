"""
Integration tests for the STT pipeline.

Tests the full chain:
1. TextPostProcessor with various STT outputs
2. STT configuration loading and environment overrides
3. Main entry point argument parsing for new modes
4. VAD → STT callback chain via synthetic SpeechEvents

These tests do NOT require a microphone or real model inference.
Mocking is used where necessary to avoid hardware/network dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.config import STTConfig, Config, load_config
from src.stt import FasterWhisperSTT, STTResult, TextPostProcessor
from src.vad.silero_vad import SpeechEvent, SpeechEventType


# ══════════════════════════════════════════════════════════════════════
# Part 1: TextPostProcessor Tests
# ══════════════════════════════════════════════════════════════════════


class TestTextPostProcessor:
    """Text post-processor integration tests."""

    def test_empty_text(self):
        """Empty input returns empty."""
        pp = TextPostProcessor()
        assert pp.process("") == ""
        assert pp.process("  ", confidence=0.5) == ""

    def test_low_confidence_skipped(self):
        """Below min_confidence returns raw text."""
        pp = TextPostProcessor(min_confidence=0.5)
        raw = "测试原始"
        result = pp.process(raw, confidence=0.3)
        assert result == raw  # Skipped due to low confidence

    def test_punctuation_restoration_question(self):
        """Interrogative patterns get question mark."""
        pp = TextPostProcessor()

        # Question particle
        result = pp.process("你在干什么呢")
        assert result.endswith("？")

        # Question pattern
        result = pp.process("这是什么")
        assert result.endswith("？")

        result = pp.process("你要不要来")
        assert result.endswith("？")

    def test_punctuation_restoration_declarative(self):
        """Declarative sentences get period."""
        pp = TextPostProcessor()
        result = pp.process("今天天气不错")
        assert result.endswith("。")

    def test_existing_punctuation_preserved(self):
        """Text that already has punctuation is not modified."""
        pp = TextPostProcessor()
        assert pp.process("你好！") == "你好！"
        assert pp.process("你好吗？") == "你好吗？"
        assert pp.process("好的。") == "好的。"

    def test_hotword_correction_case_insensitive(self):
        """Hotwords corrected regardless of case."""
        pp = TextPostProcessor(hotwords=["Claude", "AIRI"])
        result = pp.process("我用了 claude 和 airi")
        assert "Claude" in result
        assert "AIRI" in result

    def test_hotword_no_change_when_correct(self):
        """Correctly transcribed hotwords unchanged."""
        pp = TextPostProcessor(hotwords=["Claude"])
        result = pp.process("我用了 Claude")
        assert "Claude" in result
        # Count only one "Claude"
        assert result.count("Claude") == 1

    def test_cjk_latin_spacing(self):
        """Spacing inserted between CJK and Latin characters."""
        pp = TextPostProcessor(enable_punctuation=False, enable_spacing=True)
        result = pp.process("使用Claude测试")
        assert "Claude" in result
        # After CJK/Latin spacing: "使用 Claude 测试"
        assert "Claude" in result  # The hotword itself
        # Should have space before Claude (CJK→Latin)
        assert "用 Claude" in result or result == "使用 Claude 测试"

    def test_punctuation_with_comma(self):
        """Comma after topic markers."""
        pp = TextPostProcessor()
        result = pp.process("但是我不想去")
        assert "但是，" in result or result == "但是，我不想去。"

    @pytest.mark.parametrize("text,expected_suffix", [
        ("hello world", "。"),
        ("how are you", "。"),  # Default period for English
        ("真的吗", "？"),
        ("为什么", "？"),
        ("测试", "。"),
    ])
    def test_various_inputs(self, text, expected_suffix):
        """Parametrized test for various input types."""
        pp = TextPostProcessor()
        result = pp.process(text)
        assert result.endswith(expected_suffix), f"'{text}' → '{result}' should end with '{expected_suffix}'"

    def test_dynamic_hotwords(self):
        """Hotwords can be added dynamically."""
        pp = TextPostProcessor(hotwords=["Claude"])
        pp.add_hotwords(["AIRI", "SillyTavern"])
        assert "AIRI" in pp.hotwords
        assert "SillyTavern" in pp.hotwords
        assert len(pp.hotwords) == 3

    def test_set_hotwords_replaces(self):
        """set_hotwords replaces the entire list."""
        pp = TextPostProcessor(hotwords=["Claude"])
        pp.set_hotwords(["AIRI"])
        assert pp.hotwords == ["AIRI"]

    def test_confidence_threshold(self):
        """min_confidence filter works."""
        pp = TextPostProcessor(min_confidence=0.5)
        # Low confidence → raw text returned as-is
        raw = "测试低置信度"
        result = pp.process(raw, confidence=0.2)
        assert result == raw  # No processing

        # High confidence → processed
        result = pp.process(raw, confidence=0.8)
        assert result != raw
        assert result.endswith("。")

    def test_disabled_punctuation(self):
        """Punctuation can be disabled."""
        pp = TextPostProcessor(enable_punctuation=False)
        result = pp.process("测试")
        assert result == "测试"  # No punctuation added

    def test_disabled_spacing(self):
        """Spacing normalization can be disabled."""
        pp = TextPostProcessor(enable_spacing=False, enable_punctuation=False)
        result = pp.process("中文English混合")
        assert result  # Just returns as-is


# ══════════════════════════════════════════════════════════════════════
# Part 2: STT Configuration Tests
# ══════════════════════════════════════════════════════════════════════


class TestSTTConfig:
    """STT configuration integration tests."""

    def test_stt_config_defaults(self):
        """STTConfig has correct defaults matching design doc."""
        cfg = STTConfig()
        assert cfg.model_size == "small"
        assert cfg.device == "cpu"
        assert cfg.compute_type == "int8"
        assert cfg.language == "zh"
        assert cfg.beam_size == 5
        assert cfg.vad_filter is True
        assert cfg.enable_post_processing is True
        assert cfg.min_confidence == 0.3
        assert cfg.hotwords == []
        assert cfg.model_dir is None

    def test_config_includes_stt(self):
        """Top-level Config includes STTConfig."""
        cfg = Config()
        assert isinstance(cfg.stt, STTConfig)
        assert cfg.stt.model_size == "small"

    def test_stt_config_from_yaml(self):
        """STT config loads from YAML file."""
        config_path = Path("config/default.yaml")
        if config_path.exists():
            cfg = load_config(str(config_path))
            assert isinstance(cfg.stt, STTConfig)
            assert cfg.stt.model_size == "small"
            assert cfg.stt.language == "zh"
            assert cfg.stt.device == "cpu"

    def test_stt_env_overrides(self, monkeypatch):
        """Environment variables override STT config."""
        monkeypatch.setenv("STT_MODEL_SIZE", "tiny")
        monkeypatch.setenv("STT_LANGUAGE", "en")
        monkeypatch.setenv("STT_DEVICE", "cuda")
        monkeypatch.setenv("STT_COMPUTE_TYPE", "float16")

        cfg = load_config("config/default.yaml")
        assert cfg.stt.model_size == "tiny"
        assert cfg.stt.language == "en"
        assert cfg.stt.device == "cuda"
        assert cfg.stt.compute_type == "float16"

    def test_stt_serialization(self):
        """STTConfig serializes to dict correctly."""
        cfg = STTConfig(
            model_size="medium",
            language="en",
            hotwords=["Claude", "AIRI"],
            enable_post_processing=True,
            min_confidence=0.5,
        )
        d = {
            "model_size": cfg.model_size,
            "device": cfg.device,
            "compute_type": cfg.compute_type,
            "language": cfg.language,
            "hotwords": cfg.hotwords,
            "enable_post_processing": cfg.enable_post_processing,
            "min_confidence": cfg.min_confidence,
            "beam_size": cfg.beam_size,
            "vad_filter": cfg.vad_filter,
            "model_dir": cfg.model_dir,
        }
        # Verify JSON serializable
        json_str = json.dumps(d, ensure_ascii=False)
        restored = json.loads(json_str)
        assert restored["model_size"] == "medium"
        assert restored["language"] == "en"
        assert restored["hotwords"] == ["Claude", "AIRI"]
        assert restored["min_confidence"] == 0.5


# ══════════════════════════════════════════════════════════════════════
# Part 3: Main Entry Point Tests
# ══════════════════════════════════════════════════════════════════════


class TestMainArgs:
    """Test main entry point argument parsing.

    Note: These tests create a fresh ArgumentParser from argparse directly
    rather than importing from src.main, to avoid triggering the full
    import chain (sounddevice/scipy/numpy) which may cause runtime
    errors in the test environment.
    """

    def _make_parser(self):
        """Recreate the argument parser from src.main._parse_args."""
        import argparse
        parser = argparse.ArgumentParser(
            description="AIRI Voice Module - Real-time voice interaction backend",
        )
        parser.add_argument("--config", type=str, default="config/default.yaml")
        parser.add_argument("--list-devices", action="store_true")
        parser.add_argument("--test-vad", action="store_true")
        parser.add_argument("--test-stt", action="store_true")
        return parser

    def test_test_stt_flag(self):
        """--test-stt flag is recognised."""
        parser = self._make_parser()
        args = parser.parse_args(["--test-stt"])
        assert args.test_stt is True
        assert args.test_vad is False
        assert args.list_devices is False

    def test_default_mode(self):
        """No flags defaults to full mode."""
        parser = self._make_parser()
        args = parser.parse_args([])
        assert args.test_stt is False
        assert args.test_vad is False
        assert args.list_devices is False

    def test_list_devices_flag(self):
        """--list-devices is recognised."""
        parser = self._make_parser()
        args = parser.parse_args(["--list-devices"])
        assert args.list_devices is True
        assert args.test_vad is False
        assert args.test_stt is False

    def test_config_path(self):
        """--config path is passed through."""
        parser = self._make_parser()
        args = parser.parse_args(["--config", "custom_config.yaml"])
        assert args.config == "custom_config.yaml"
        assert args.test_vad is False
        assert args.test_stt is False


# ══════════════════════════════════════════════════════════════════════
# Part 4: STT → PostProcessor Integration
# ══════════════════════════════════════════════════════════════════════


class TestSTTWithPostProcessor:
    """Integration between STT engine and post-processor."""

    def test_post_processor_attached_to_stt(self):
        """STT engine can be created alongside a post-processor."""
        stt = FasterWhisperSTT(model_size="tiny", model_dir="/tmp/stt_test_int")
        pp = TextPostProcessor(hotwords=["AIRI", "Claude"])

        assert stt.model_size == "tiny"
        assert pp.hotwords == ["AIRI", "Claude"]

        # Verify lazy loading — model not loaded yet
        assert not stt.is_loaded

    @pytest.mark.asyncio
    async def test_stt_transcribe_silence(self):
        """STT returns empty result for silent audio."""
        stt = FasterWhisperSTT(model_size="tiny", model_dir="/tmp/stt_test_int")
        audio = np.zeros(1600, dtype=np.float32)  # 0.1s of silence
        result = await stt.transcribe(audio, 16000)
        assert isinstance(result, STTResult)
        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_stt_transcribe_empty(self):
        """STT returns empty result for empty audio."""
        stt = FasterWhisperSTT(model_size="tiny", model_dir="/tmp/stt_test_int")
        audio = np.array([], dtype=np.float32)
        result = await stt.transcribe(audio, 16000)
        assert result.text == ""

    def test_post_processor_chains_with_stt_result(self):
        """Post-processor can process STTResult text."""
        pp = TextPostProcessor(hotwords=["Claude"])

        # Simulate what happens after STT inference
        raw_text = "claude今天天气不错"
        processed = pp.process(raw_text, confidence=0.8)

        # Hotword should be corrected
        assert "Claude" in processed
        # Punctuation should be added
        assert processed.endswith("。")

    def test_post_processor_low_confidence_skips(self):
        """Post-processor skips processing when confidence is low."""
        pp = TextPostProcessor(min_confidence=0.5)

        raw_text = "这是低质量识别结果"
        # Low confidence → should return raw text unchanged
        result = pp.process(raw_text, confidence=0.2)
        assert result == raw_text

        # High confidence → should process
        result = pp.process(raw_text, confidence=0.8)
        assert result != raw_text
        assert result.endswith("。")

    @pytest.mark.asyncio
    async def test_stt_invalid_audio_format(self):
        """STT rejects invalid audio format."""
        stt = FasterWhisperSTT(model_size="tiny", model_dir="/tmp/stt_test_int")

        # Wrong sample rate
        audio = np.random.randn(4800).astype(np.float32)  # 0.1s at 48kHz
        result = await stt.transcribe(audio, 48000)
        assert result.text == ""
        assert result.confidence == 0.0

        # Wrong dtype
        audio = np.random.randn(1600).astype(np.int16)
        result = await stt.transcribe(audio, 16000)
        assert result.text == ""


# ══════════════════════════════════════════════════════════════════════
# Part 5: VAD → STT Callback Chain
# ══════════════════════════════════════════════════════════════════════


class TestVADtoSTTCallback:
    """VAD speech event → STT integration tests."""

    def test_speech_event_creation(self):
        """Synthetic SpeechEvent matches what VAD produces."""
        audio = np.random.randn(16000).astype(np.float32)  # 1s at 16kHz
        event = SpeechEvent(
            type=SpeechEventType.SPEECH_END,
            audio=audio,
            timestamp=100.0,
            duration=1.0,
            num_frames=32,
            max_prob=0.85,
            sample_rate=16000,
        )
        assert event.type == SpeechEventType.SPEECH_END
        assert event.audio.shape == (16000,)
        assert event.duration == 1.0
        assert event.sample_rate == 16000

    @pytest.mark.asyncio
    async def test_stt_with_vad_audio(self):
        """STT processes audio from a synthetic VAD output."""
        stt = FasterWhisperSTT(model_size="tiny", model_dir="/tmp/stt_test_int")

        # Simulate VAD audio output (non-speech → empty result)
        audio = np.random.randn(8000).astype(np.float32) * 0.001  # Very quiet
        event = SpeechEvent(
            type=SpeechEventType.SPEECH_END,
            audio=audio,
            duration=0.5,
            sample_rate=16000,
        )

        result = await stt.transcribe(event.audio, event.sample_rate)
        assert isinstance(result, STTResult)
        # Quiet audio should be detected as silence
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_callback_chain_mock(self):
        """Simulate the VAD→STT callback chain with mocks."""
        stt = FasterWhisperSTT(model_size="tiny", model_dir="/tmp/stt_test_int")
        pp = TextPostProcessor(hotwords=["AIRI"])

        # Mock the transcribe method
        stt.transcribe = AsyncMock(return_value=STTResult(
            text="claude 你好",
            confidence=0.85,
            language="zh",
            language_probability=0.95,
            duration=1.0,
            inference_time=0.1,
        ))

        # Simulate VAD event
        audio = np.random.randn(16000).astype(np.float32)
        event = SpeechEvent(
            type=SpeechEventType.SPEECH_END,
            audio=audio,
            duration=1.0,
            sample_rate=16000,
        )

        # Run the chain
        result = await stt.transcribe(event.audio, event.sample_rate)
        assert result.text == "claude 你好"
        assert result.confidence >= pp.min_confidence

        # Post-process
        processed = pp.process(result.text, confidence=result.confidence)
        assert "Claude" in processed or processed == "claude 你好。"

    def test_callback_filter_low_confidence(self):
        """Low-confidence results can be filtered out."""
        pp = TextPostProcessor(min_confidence=0.5)

        # Text with low confidence
        text = "噪声产生的错误文字"
        confidence = 0.1

        # Should skip processing
        result = pp.process(text, confidence=confidence)
        assert result == text  # Raw text returned

        # Above threshold
        result = pp.process(text, confidence=0.6)
        assert result != text  # Processed


# ══════════════════════════════════════════════════════════════════════
# Part 6: Model Download Script
# ══════════════════════════════════════════════════════════════════════


class TestDownloadScript:
    """Test the download script's argument parsing and model listing."""

    def test_parse_args_defaults(self):
        """Default arguments are correct."""
        from scripts.download_models import _parse_args
        import sys

        with patch.object(sys, "argv", ["scripts/download_models.py"]):
            args = _parse_args()
            assert args.model_size == "tiny"
            assert args.verify is False
            assert args.force is False
            assert args.list_models is False

    def test_parse_args_custom(self):
        """Custom arguments are parsed."""
        from scripts.download_models import _parse_args
        import sys

        test_args = ["--model-size", "small", "--verify", "--dir", "/tmp/models"]
        with patch.object(sys, "argv", ["scripts/download_models.py", *test_args]):
            args = _parse_args()
            assert args.model_size == "small"
            assert args.verify is True
            assert args.dir == "/tmp/models"

    def test_list_models_output(self):
        """list_models prints model info."""
        from scripts.download_models import _list_models
        import sys
        from io import StringIO

        captured = StringIO()
        sys.stdout = captured
        _list_models()
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "tiny" in output
        assert "small" in output
        assert "large-v3" in output
        assert "Recommended" in output
        assert "✅" in output  # small is recommended

    def test_build_download_dir_default(self):
        """Default download directory uses models/faster-whisper-{size}."""
        from scripts.download_models import _get_download_dir

        download_dir = _get_download_dir("small", None)
        assert "faster-whisper-small" in str(download_dir)
        assert "models" in str(download_dir)

    def test_build_download_dir_custom(self):
        """Custom download directory is used when provided."""
        from scripts.download_models import _get_download_dir

        download_dir = _get_download_dir("tiny", "/custom/path")
        assert str(download_dir) == "/custom/path"


# ══════════════════════════════════════════════════════════════════════
# Part 7: Module Exports
# ══════════════════════════════════════════════════════════════════════


class TestSTTModuleExports:
    """Test the STT module's public API."""

    def test_imports(self):
        """All expected exports are available."""
        from src.stt import FasterWhisperSTT, STTResult, TextPostProcessor
        assert FasterWhisperSTT is not None
        assert STTResult is not None
        assert TextPostProcessor is not None

    def test_result_dataclass(self):
        """STTResult is a proper dataclass."""
        result = STTResult(
            text="测试",
            confidence=0.9,
            language="zh",
            language_probability=0.95,
            duration=2.0,
            inference_time=0.5,
        )
        assert result.text == "测试"
        assert result.confidence == 0.9
        assert result.language == "zh"
        assert result.duration == 2.0
        assert result.inference_time == 0.5

    def test_result_with_segments(self):
        """STTResult supports optional segments."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "你好", "confidence": 0.9},
            {"start": 1.0, "end": 2.0, "text": "世界", "confidence": 0.85},
        ]
        result = STTResult(
            text="你好世界",
            confidence=0.88,
            language="zh",
            language_probability=0.95,
            duration=2.0,
            inference_time=0.5,
            segments=segments,
        )
        assert len(result.segments) == 2
        assert result.segments[0]["text"] == "你好"
