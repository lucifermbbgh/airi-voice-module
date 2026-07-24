"""
Tests for the TTS (Text-to-Speech) module.

Tests cover:
- TTSResult dataclass integrity
- TTSBase abstract interface
- CosyVoiceTTS initialization and configuration
- CosyVoiceTTS edge cases (empty text, invalid params)
- TTS utility functions (volume, resample, sentence splitting)
- TTSConfig loading and env overrides

Actual model inference tests require cosyvoice installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import TTSConfig, Config, load_config
from src.tts import CosyVoiceTTS, TTSBase, TTSResult


# ══════════════════════════════════════════════════════════════════════
# Part 1: TTSResult Tests
# ══════════════════════════════════════════════════════════════════════


class TestTTSResult:
    """TTSResult dataclass integrity tests."""

    def test_minimal_result(self):
        """TTSResult with all fields."""
        audio = np.zeros(24000, dtype=np.float32)  # 1s at 24kHz
        result = TTSResult(
            audio=audio,
            sample_rate=24000,
            duration=1.0,
            text="你好",
            synthesis_time=0.5,
        )
        assert result.audio.shape == (24000,)
        assert result.sample_rate == 24000
        assert result.duration == 1.0
        assert result.text == "你好"
        assert result.synthesis_time == 0.5

    def test_auto_duration(self):
        """Duration auto-calculated from audio length."""
        audio = np.zeros(48000, dtype=np.float32)  # 2s at 24kHz
        result = TTSResult(
            audio=audio,
            sample_rate=24000,
            duration=0.0,  # Will be auto-calculated
            text="测试",
            synthesis_time=0.3,
        )
        assert result.duration == 2.0  # 48000 / 24000

    def test_auto_duration_empty(self):
        """Empty audio has zero duration."""
        result = TTSResult(
            audio=np.array([], dtype=np.float32),
            sample_rate=24000,
            duration=0.0,
            text="",
            synthesis_time=0.0,
        )
        assert result.duration == 0.0

    def test_duration_preserved(self):
        """Explicit duration is preserved."""
        audio = np.zeros(48000, dtype=np.float32)
        result = TTSResult(
            audio=audio,
            sample_rate=24000,
            duration=5.0,  # Explicit, even though audio is only 2s
            text="测试",
            synthesis_time=0.3,
        )
        assert result.duration == 5.0  # Not overridden


# ══════════════════════════════════════════════════════════════════════
# Part 2: CosyVoiceTTS Init Tests
# ══════════════════════════════════════════════════════════════════════


class TestCosyVoiceTTSInit:
    """CosyVoiceTTS initialization tests."""

    def test_default_initialization(self):
        """Default parameters are correct."""
        tts = CosyVoiceTTS()
        assert tts.model_size == "base"
        assert tts.device == "cpu"
        assert tts.sample_rate == 24000
        assert tts.default_voice == "default"
        assert tts.default_speed == 1.0
        assert tts.name == "cosyvoice"

    def test_model_not_loaded_by_default(self):
        """Model is lazy-loaded (not loaded on init)."""
        tts = CosyVoiceTTS()
        assert not tts.is_loaded

    def test_custom_parameters(self):
        """Custom parameters are accepted."""
        tts = CosyVoiceTTS(
            model_size="small",
            device="cuda",
            model_dir="/custom/path",
            sample_rate=16000,
            default_voice="中文男声",
            default_speed=1.5,
        )
        assert tts.model_size == "small"
        assert tts.device == "cuda"
        assert tts.model_dir == "/custom/path"
        assert tts.sample_rate == 16000
        assert tts.default_voice == "中文男声"
        assert tts.default_speed == 1.5

    def test_valid_model_sizes(self):
        """All valid model sizes are accepted."""
        for size in ["base", "small"]:
            tts = CosyVoiceTTS(model_size=size)
            assert tts.model_size == size

    def test_invalid_model_size(self):
        """Invalid model size raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model_size"):
            CosyVoiceTTS(model_size="invalid")
        with pytest.raises(ValueError, match="Invalid model_size"):
            CosyVoiceTTS(model_size="large")

    def test_voices_property(self):
        """Voices property returns preset list."""
        tts = CosyVoiceTTS()
        voices = tts.voices
        assert len(voices) >= 8
        assert voices[0]["id"] == "default"
        assert voices[1]["id"] == "中文男声"

    def test_model_info_property(self):
        """Model info has correct structure."""
        tts = CosyVoiceTTS(model_size="base")
        info = tts.model_info
        assert info["engine"] == "cosyvoice"
        assert info["model_size"] == "base"
        assert info["device"] == "cpu"
        assert info["loaded"] is False

    def test_cleanup(self):
        """Cleanup sets model to None."""
        import asyncio
        tts = CosyVoiceTTS()
        asyncio.run(tts.cleanup())
        assert not tts.is_loaded
        assert tts._executor is None

    def test_unload_model(self):
        """Unload sets model to None."""
        import asyncio
        tts = CosyVoiceTTS()
        asyncio.run(tts.unload_model())
        assert not tts.is_loaded


# ══════════════════════════════════════════════════════════════════════
# Part 3: CosyVoiceTTS Edge Cases
# ══════════════════════════════════════════════════════════════════════


class TestCosyVoiceTTSEdgeCases:
    """Edge case tests for CosyVoiceTTS."""

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self):
        """Empty text returns empty TTSResult."""
        tts = CosyVoiceTTS()
        result = await tts.synthesize("")
        assert len(result.audio) == 0
        assert result.duration == 0.0
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_synthesize_whitespace_text(self):
        """Whitespace-only text returns empty TTSResult."""
        tts = CosyVoiceTTS()
        result = await tts.synthesize("   ")
        assert len(result.audio) == 0

    def test_set_voice_valid(self):
        """Setting valid voice works."""
        tts = CosyVoiceTTS()
        tts.set_voice("中文男声")
        assert tts.default_voice == "中文男声"
        tts.set_voice("default")
        assert tts.default_voice == "default"

    def test_set_voice_invalid(self):
        """Setting invalid voice raises ValueError."""
        tts = CosyVoiceTTS()
        with pytest.raises(ValueError, match="Unknown voice"):
            tts.set_voice("nonexistent_voice")

    def test_set_speed_valid(self):
        """Setting valid speed works."""
        tts = CosyVoiceTTS()
        tts.set_speed(0.5)
        assert tts.default_speed == 0.5
        tts.set_speed(2.0)
        assert tts.default_speed == 2.0
        tts.set_speed(1.0)
        assert tts.default_speed == 1.0

    def test_set_speed_invalid_low(self):
        """Speed below 0.5 raises ValueError."""
        tts = CosyVoiceTTS()
        with pytest.raises(ValueError, match="Speed must be between"):
            tts.set_speed(0.1)

    def test_set_speed_invalid_high(self):
        """Speed above 2.0 raises ValueError."""
        tts = CosyVoiceTTS()
        with pytest.raises(ValueError, match="Speed must be between"):
            tts.set_speed(5.0)


# ══════════════════════════════════════════════════════════════════════
# Part 4: Sentence Splitting
# ══════════════════════════════════════════════════════════════════════


class TestSentenceSplitting:
    """Text sentence splitting for streaming TTS."""

    def test_single_sentence(self):
        """Single sentence returns one chunk."""
        result = CosyVoiceTTS._split_sentences("你好吗？")
        assert len(result) == 1
        assert result[0] == "你好吗？"

    def test_multiple_sentences(self):
        """Multiple sentences are split correctly."""
        result = CosyVoiceTTS._split_sentences("你好。今天天气不错！你吃饭了吗？是的。")
        assert len(result) == 4
        assert result[0] == "你好。"
        assert result[1] == "今天天气不错！"
        assert result[2] == "你吃饭了吗？"
        assert result[3] == "是的。"

    def test_no_punctuation(self):
        """Text without punctuation returns as single chunk."""
        result = CosyVoiceTTS._split_sentences("今天天气不错")
        assert len(result) == 1
        assert result[0] == "今天天气不错"

    def test_mixed(self):
        """Mixed punctuation and trailing text."""
        result = CosyVoiceTTS._split_sentences("好的。继续")
        assert len(result) == 2
        assert result[0] == "好的。"
        assert result[1] == "继续"

    def test_newline_separator(self):
        """Newline splits sentences."""
        result = CosyVoiceTTS._split_sentences("第一行\n第二行")
        assert len(result) == 2
        assert result[0] == "第一行"
        assert result[1] == "第二行"

    def test_empty_string(self):
        """Empty string returns list with empty string."""
        result = CosyVoiceTTS._split_sentences("")
        assert len(result) == 1
        assert result[0] == ""


# ══════════════════════════════════════════════════════════════════════
# Part 5: TTS Utility Functions
# ══════════════════════════════════════════════════════════════════════


class TestTTSUtilities:
    """TTS utility function tests."""

    def test_normalize_volume_empty(self):
        """Empty audio returns unchanged."""
        result = TTSBase.normalize_volume(np.array([], dtype=np.float32))
        assert len(result) == 0

    def test_normalize_volume_silence(self):
        """Silent audio (all zeros) returns unchanged."""
        audio = np.zeros(1000, dtype=np.float32)
        result = TTSBase.normalize_volume(audio)
        assert np.all(result == 0.0)

    def test_normalize_volume_scales(self):
        """Volume is scaled to target peak."""
        audio = np.array([0.5, -0.8, 0.3], dtype=np.float32)
        result = TTSBase.normalize_volume(audio, target_peak=0.9)
        # After scaling: -0.8 -> -0.9 max, so peak should be ≤ 0.9
        assert float(np.max(np.abs(result))) <= 0.901
        # Verify scaling direction preserved
        assert result[1] < 0  # Negative stays negative
        assert result[0] > 0  # Positive stays positive

    def test_normalize_volume_clips(self):
        """Volume clipped to target peak."""
        audio = np.array([2.0, -1.5, 0.5], dtype=np.float32)
        result = TTSBase.normalize_volume(audio, target_peak=0.95)
        assert float(np.max(np.abs(result))) <= 0.951

    def test_resample_same_rate(self):
        """Resampling at same rate returns unchanged."""
        audio = np.zeros(1000, dtype=np.float32)
        result = TTSBase.resample(audio, 16000, 16000)
        assert len(result) == 1000

    def test_resample_up(self):
        """Upsampling increases sample count."""
        audio = np.zeros(16000, dtype=np.float32)  # 1s at 16kHz
        result = TTSBase.resample(audio, 16000, 24000)
        assert len(result) == 24000  # 1s at 24kHz

    def test_resample_down(self):
        """Downsampling decreases sample count."""
        audio = np.zeros(24000, dtype=np.float32)  # 1s at 24kHz
        result = TTSBase.resample(audio, 24000, 16000)
        assert len(result) == 16000  # 1s at 16kHz

    def test_resample_empty(self):
        """Empty audio resample returns empty."""
        result = TTSBase.resample(np.array([], dtype=np.float32), 16000, 24000)
        assert len(result) == 0


# ══════════════════════════════════════════════════════════════════════
# Part 6: TTSConfig Tests
# ══════════════════════════════════════════════════════════════════════


class TestTTSConfig:
    """TTS configuration integration tests."""

    def test_tts_config_defaults(self):
        """TTSConfig has correct defaults from design doc."""
        cfg = TTSConfig()
        assert cfg.engine == "cosyvoice"
        assert cfg.model_size == "base"
        assert cfg.voice_id == "default"
        assert cfg.speed == 1.0
        assert cfg.sample_rate == 24000
        assert cfg.device == "cpu"
        assert cfg.streaming is True
        assert cfg.max_text_length == 500
        assert cfg.enable_cache is True
        assert cfg.cache_size == 128

    def test_config_includes_tts(self):
        """Top-level Config includes TTSConfig."""
        cfg = Config()
        assert isinstance(cfg.tts, TTSConfig)
        assert cfg.tts.engine == "cosyvoice"
        assert cfg.tts.sample_rate == 24000

    def test_tts_config_from_yaml(self):
        """TTS config loads from YAML file."""
        config_path = "config/default.yaml"
        if Path(config_path).exists():
            cfg = load_config(config_path)
            assert isinstance(cfg.tts, TTSConfig)
            assert cfg.tts.engine == "cosyvoice"
            assert cfg.tts.voice_id == "default"
            assert cfg.tts.streaming is True

    def test_tts_env_overrides(self, monkeypatch):
        """Environment variables override TTS config."""
        monkeypatch.setenv("TTS_ENGINE", "edge_tts")
        monkeypatch.setenv("TTS_VOICE_ID", "en_female")
        monkeypatch.setenv("TTS_SPEED", "1.5")
        monkeypatch.setenv("TTS_DEVICE", "cuda")

        cfg = load_config("config/default.yaml")
        assert cfg.tts.engine == "edge_tts"
        assert cfg.tts.voice_id == "en_female"
        assert cfg.tts.speed == 1.5
        assert cfg.tts.device == "cuda"

    def test_tts_serialization(self):
        """TTSConfig can be serialized to dict."""
        cfg = TTSConfig(
            engine="edge_tts",
            voice_id="en_male",
            speed=1.2,
            enable_cache=False,
        )
        d = {
            "engine": cfg.engine,
            "voice_id": cfg.voice_id,
            "speed": cfg.speed,
            "enable_cache": cfg.enable_cache,
        }
        import json
        json_str = json.dumps(d, ensure_ascii=False)
        restored = json.loads(json_str)
        assert restored["engine"] == "edge_tts"
        assert restored["voice_id"] == "en_male"
        assert restored["speed"] == 1.2
        assert restored["enable_cache"] is False


# Helper import
from pathlib import Path
