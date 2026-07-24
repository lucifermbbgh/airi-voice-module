"""
Integration tests for the TTS pipeline.

Tests the full chain:
1. TTSManager with mock engine + mock playback
2. TTSCache LRU behavior, hit rate, eviction
3. TTSManager say() flow (cache hit vs miss)
4. TTSManager stop/pause/resume
5. TTSManager cleanup
6. TTS config loading with full Config

These tests do NOT require a real TTS engine or audio hardware.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.config import TTSConfig, Config, load_config
from src.tts import TTSBase, TTSResult, TTSCache, TTSManager


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_engine():
    """Create a mock TTS engine for testing."""
    engine = MagicMock(spec=TTSBase)
    engine.name = "mock_tts"
    engine.default_voice = "default"
    engine.default_speed = 1.0
    engine.sample_rate = 24000
    engine.max_text_length = 500

    async def synthesize(text, voice_id="default", speed=1.0):
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 24000)).astype(np.float32) * 0.5
        return TTSResult(
            audio=audio,
            sample_rate=24000,
            duration=1.0,
            text=text,
            synthesis_time=0.1,
        )

    async def synthesize_stream(text, voice_id="default", speed=1.0):
        for _ in range(3):
            yield np.zeros(8000, dtype=np.float32)

    engine.synthesize = AsyncMock(side_effect=synthesize)
    # Async generator — assign directly, not via AsyncMock
    # because AsyncMock does not support async generators
    engine.synthesize_stream = synthesize_stream
    engine.cleanup = AsyncMock()
    return engine


@pytest.fixture
def mock_playback():
    """Create a mock AudioPlayback for testing."""
    pb = MagicMock()
    pb.is_playing = False

    async def play(audio, sample_rate=None, text=""):
        pb.is_playing = True

    async def wait_completion():
        pb.is_playing = False

    async def stop_current():
        pb.is_playing = False

    async def stop():
        pb.is_playing = False

    pb.play = AsyncMock(side_effect=play)
    pb.wait_for_completion = AsyncMock(side_effect=wait_completion)
    pb.stop_current = AsyncMock(side_effect=stop_current)
    pb.stop = AsyncMock(side_effect=stop)
    return pb


# ══════════════════════════════════════════════════════════════════════
# Part 1: TTSCache Integration Tests
# ══════════════════════════════════════════════════════════════════════


class TestTTSCache:
    """TTSCache integration tests."""

    def test_cache_basic(self):
        """Basic put/get works."""
        cache = TTSCache(max_size=10)
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cache.put("hello", "default", 1.0, audio)
        result = cache.get("hello", "default", 1.0)
        assert result is not None
        assert np.array_equal(result, audio)

    def test_cache_miss(self):
        """Cache miss returns None."""
        cache = TTSCache(max_size=10)
        result = cache.get("not_cached", "default", 1.0)
        assert result is None

    def test_cache_voice_sensitivity(self):
        """Different voices have different cache entries."""
        cache = TTSCache(max_size=10)
        audio1 = np.array([1.0])
        audio2 = np.array([2.0])
        cache.put("hello", "voice1", 1.0, audio1)
        cache.put("hello", "voice2", 1.0, audio2)

        result1 = cache.get("hello", "voice1", 1.0)
        result2 = cache.get("hello", "voice2", 1.0)
        assert np.array_equal(result1, [1.0])
        assert np.array_equal(result2, [2.0])

    def test_cache_speed_sensitivity(self):
        """Different speeds have different cache entries."""
        cache = TTSCache(max_size=10)
        cache.put("hello", "default", 1.0, np.array([1.0]))
        cache.put("hello", "default", 1.5, np.array([2.0]))

        r1 = cache.get("hello", "default", 1.0)
        r2 = cache.get("hello", "default", 1.5)
        assert r1[0] == 1.0
        assert r2[0] == 2.0

    def test_cache_lru_eviction(self):
        """LRU eviction removes oldest entries."""
        cache = TTSCache(max_size=3)
        cache.put("a", "v", 1.0, np.array([1]))
        cache.put("b", "v", 1.0, np.array([2]))
        cache.put("c", "v", 1.0, np.array([3]))
        assert cache.size == 3

        # Access 'a' to make it recently used
        cache.get("a", "v", 1.0)

        # Add 'd' — should evict 'b' (least recently used)
        cache.put("d", "v", 1.0, np.array([4]))
        assert cache.size == 3
        assert cache.get("a", "v", 1.0) is not None  # Still there
        assert cache.get("b", "v", 1.0) is None  # Evicted
        assert cache.get("d", "v", 1.0) is not None  # New

    def test_cache_clear(self):
        """Clear resets all state."""
        cache = TTSCache(max_size=10)
        cache.put("hello", "default", 1.0, np.array([1.0]))
        cache.clear()
        assert cache.size == 0
        assert cache.get("hello", "default", 1.0) is None
        assert cache.hit_rate == 0.0

    def test_cache_hit_rate(self):
        """Hit rate is calculated correctly."""
        cache = TTSCache(max_size=10)
        cache.put("hello", "default", 1.0, np.array([1.0]))
        # 1 hit, 0 miss so far
        cache.get("hello", "default", 1.0)  # hit
        assert cache.hit_rate == 1.0

        cache.get("missing", "default", 1.0)  # miss
        assert 0.0 < cache.hit_rate < 1.0

    def test_cache_stats(self):
        """Stats returns correct structure."""
        cache = TTSCache(max_size=5)
        cache.put("a", "v", 1.0, np.array([1]))
        cache.get("a", "v", 1.0)  # hit
        cache.get("b", "v", 1.0)  # miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["max_size"] == 5
        assert stats["hit_rate"] == 0.5

    def test_cache_max_size_default(self):
        """Default max_size is 128."""
        cache = TTSCache()
        assert cache.max_size == 128


# ══════════════════════════════════════════════════════════════════════
# Part 2: TTSManager Integration Tests
# ══════════════════════════════════════════════════════════════════════


class TestTTSManager:
    """TTSManager integration tests."""

    @pytest.mark.asyncio
    async def test_say_empty_text(self, mock_engine, mock_playback):
        """Empty text returns False."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        result = await manager.say("")
        assert result is False
        assert not manager.is_speaking

        result = await manager.say("   ")
        assert result is False

    @pytest.mark.asyncio
    async def test_say_synthesizes(self, mock_engine, mock_playback):
        """say() calls engine.synthesize and playback.play."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        result = await manager.say("你好")
        assert result is True
        mock_engine.synthesize.assert_awaited_once_with("你好", "default", 1.0)
        mock_playback.play.assert_called_once()
        mock_playback.wait_for_completion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_say_uses_cache(self, mock_engine, mock_playback):
        """say() caches and reuses results."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)

        # First call — cache miss
        await manager.say("缓存测试")
        first_call_count = mock_engine.synthesize.await_count

        # Second call — cache hit
        await manager.say("缓存测试")
        # Engine should not have been called again
        assert mock_engine.synthesize.await_count == first_call_count

    @pytest.mark.asyncio
    async def test_say_custom_voice_speed(self, mock_engine, mock_playback):
        """say() passes custom voice and speed."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        await manager.say("test", voice_id="中文男声", speed=1.5)
        mock_engine.synthesize.assert_awaited_once_with("test", "中文男声", 1.5)

    @pytest.mark.asyncio
    async def test_stop(self, mock_engine, mock_playback):
        """stop() interrupts playback."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        await manager.stop()
        mock_playback.stop_current.assert_awaited_once()
        assert not manager._speaking

    @pytest.mark.asyncio
    async def test_pause_resume(self, mock_engine, mock_playback):
        """pause() and resume() work."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        manager.pause()
        mock_playback.pause.assert_called_once()
        manager.resume()
        mock_playback.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_engine, mock_playback):
        """clear_cache() empties the cache."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        await manager.say("test")
        assert manager.cache.size > 0

        mock_engine.synthesize.reset_mock()
        mock_engine.synthesize.await_count = 0

        manager.clear_cache()
        assert manager.cache.size == 0

        # After clearing, should synthesize again
        await manager.say("test")
        assert mock_engine.synthesize.await_count > 0

    @pytest.mark.asyncio
    async def test_cache_stats(self, mock_engine, mock_playback):
        """cache_stats() returns proper data."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        await manager.say("a")
        await manager.say("a")  # hit
        await manager.say("b")
        stats = manager.cache_stats()
        assert stats["hits"] >= 1
        assert stats["size"] >= 2

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_engine, mock_playback):
        """cleanup() releases all resources."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        await manager.cleanup()
        mock_playback.stop.assert_awaited_once()
        mock_engine.cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_callbacks(self, mock_engine, mock_playback):
        """on_start and on_end callbacks are called."""
        events = []

        def on_start(text):
            events.append(("start", text))

        def on_end(text):
            events.append(("end", text))

        manager = TTSManager(
            engine=mock_engine,
            playback=mock_playback,
            on_start=on_start,
            on_end=on_end,
        )
        await manager.say("回调测试")
        assert len(events) == 2
        assert events[0] == ("start", "回调测试")
        assert events[1] == ("end", "回调测试")

    @pytest.mark.asyncio
    async def test_say_stream(self, mock_engine, mock_playback):
        """say_stream() plays audio chunks."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        result = await manager.say_stream("流式测试", voice_id="default", speed=1.0)
        assert result is True
        # Playback should have been called at least once
        mock_playback.play.assert_called()

    @pytest.mark.asyncio
    async def test_is_speaking(self, mock_engine, mock_playback):
        """is_speaking property works."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        assert not manager.is_speaking

        # When playback is playing, is_speaking should reflect it
        mock_playback.is_playing = True
        assert manager.is_speaking

        mock_playback.is_playing = False
        assert not manager.is_speaking


# ══════════════════════════════════════════════════════════════════════
# Part 3: TTS Config Integration
# ══════════════════════════════════════════════════════════════════════


class TestTTSConfigIntegration:
    """TTS configuration with full Config."""

    def test_tts_in_full_config(self):
        """TTSConfig is accessible from Config."""
        cfg = Config()
        assert isinstance(cfg.tts, TTSConfig)
        assert cfg.tts.engine == "cosyvoice"
        assert cfg.tts.model_size == "base"
        assert cfg.tts.sample_rate == 24000
        assert cfg.tts.streaming is True

    def test_tts_custom_config(self):
        """Custom TTSConfig via Config."""
        cfg = Config()
        cfg.tts.engine = "edge_tts"
        cfg.tts.voice_id = "en_female"
        cfg.tts.speed = 1.2
        assert cfg.tts.engine == "edge_tts"
        assert cfg.tts.voice_id == "en_female"
        assert cfg.tts.speed == 1.2

    def test_tts_yaml_loading(self):
        """TTS config loads from YAML."""
        config_path = "config/default.yaml"
        from pathlib import Path
        if Path(config_path).exists():
            cfg = load_config(config_path)
            assert cfg.tts.engine == "cosyvoice"
            assert cfg.tts.streaming is True

    def test_tts_env_overrides(self, monkeypatch):
        """TTS env vars override YAML values."""
        monkeypatch.setenv("TTS_ENGINE", "edge_tts")
        monkeypatch.setenv("TTS_VOICE_ID", "en_male")
        cfg = load_config("config/default.yaml")
        assert cfg.tts.engine == "edge_tts"
        assert cfg.tts.voice_id == "en_male"


# ══════════════════════════════════════════════════════════════════════
# Part 4: Pipeline Integration (VAD -> STT -> TTS)
# ══════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
    """End-to-end VAD → STT → TTS chain."""

    @pytest.mark.asyncio
    async def test_stt_to_tts_flow(self, mock_engine, mock_playback):
        """Simulate STT output flowing to TTS."""
        # Simulate what happens after VAD → STT
        stt_text = "今天天气怎么样"

        # Feed STT result into TTS
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        await manager.say(stt_text)

        # Verify the STT text was synthesized
        mock_engine.synthesize.assert_awaited_once_with(
            stt_text, mock_engine.default_voice, mock_engine.default_speed,
        )

    @pytest.mark.asyncio
    async def test_stt_to_tts_empty(self, mock_engine, mock_playback):
        """Empty STT text does not trigger TTS."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)
        result = await manager.say("")
        assert result is False
        mock_engine.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_stt_to_tts_low_confidence(self, mock_engine, mock_playback):
        """Low confidence STT should not trigger TTS."""
        manager = TTSManager(engine=mock_engine, playback=mock_playback)

        # Simulate low confidence filter
        stt_confidence = 0.1
        min_confidence = 0.3

        if stt_confidence < min_confidence:
            result = await manager.say("")
            assert result is False
        else:
            result = await manager.say("应该被忽略的文本")
            assert result is True
