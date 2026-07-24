"""
TTS Manager — orchestrates TTS engine, caching, and playback.

Acts as the central coordinator between:
- TTS engine (CosyVoiceTTS, etc.)
- Voice cache (frequent phrases)
- AudioPlayback (speaker output)

Usage:
    tts_mgr = TTSManager(engine, playback)
    await tts_mgr.say("你好，我是 AIRI")
    await tts_mgr.stop()  # Interrupt current playback
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Callable

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


class TTSCache:
    """LRU cache for TTS audio results.

    Caches frequent phrases to avoid repeated synthesis.
    Thread-safe for concurrent access.

    Attributes:
        max_size: Maximum number of cached entries.
    """

    def __init__(self, max_size: int = 128):
        """Initialize LRU cache.

        Args:
            max_size: Maximum cache entries (default: 128).
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str, voice_id: str, speed: float) -> str:
        """Generate cache key from synthesis parameters.

        Args:
            text: Input text.
            voice_id: Voice identifier.
            speed: Speaking speed.

        Returns:
            Cache key string.
        """
        raw = f"{text}||{voice_id}||{speed:.2f}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, voice_id: str, speed: float) -> np.ndarray | None:
        """Get cached audio if available.

        Args:
            text: Input text.
            voice_id: Voice identifier.
            speed: Speaking speed.

        Returns:
            Cached audio array, or None if not found.
        """
        key = self._make_key(text, voice_id, speed)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, voice_id: str, speed: float,
            audio: np.ndarray) -> None:
        """Cache audio for future use.

        Args:
            text: Input text.
            voice_id: Voice identifier.
            speed: Speaking speed.
            audio: Audio data to cache.
        """
        key = self._make_key(text, voice_id, speed)
        self._cache[key] = audio
        self._cache.move_to_end(key)

        # Evict oldest if over limit
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, size, hit_rate.
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
        }


class TTSManager:
    """TTS Manager — orchestrates synthesis, caching, and playback.

    Coordinates the full TTS pipeline:
    text → [cache check] → [synthesize] → [cache store] → [playback]

    Attributes:
        engine: TTS engine instance (TTSBase subclass).
        playback: AudioPlayback instance for output.
        cache: Voice cache instance.
    """

    def __init__(
        self,
        engine,
        playback,
        cache: TTSCache | None = None,
        on_start: Callable | None = None,
        on_end: Callable | None = None,
    ):
        """Initialize TTS Manager.

        Args:
            engine: TTS engine (must implement TTSBase interface).
            playback: AudioPlayback instance.
            cache: Optional TTSCache instance (creates default if None).
            on_start: Optional callback when TTS playback starts.
            on_end: Optional callback when TTS playback ends.
        """
        self.engine = engine
        self.playback = playback
        self.cache = cache or TTSCache(max_size=128)

        self._on_start = on_start
        self._on_end = on_end
        self._speaking = False

        logger.info(
            "TTSManager initialized: engine={}, cache={}",
            engine.name, self.cache.max_size,
        )

    @property
    def is_speaking(self) -> bool:
        """Check if TTS is currently speaking."""
        return self._speaking or self.playback.is_playing

    async def say(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float | None = None,
    ) -> bool:
        """Synthesise and play text (blocking — waits for playback).

        Full pipeline: cache check → synthesis → cache store → play → wait.

        Args:
            text: Text to speak.
            voice_id: Voice override (uses engine default if None).
            speed: Speed override (uses engine default if None).

        Returns:
            True if speech was synthesised and played.
        """
        if not text or not text.strip():
            return False

        voice_id = voice_id or self.engine.default_voice
        speed = speed or self.engine.default_speed
        text = text.strip()

        # Truncate if too long
        max_len = getattr(self.engine, "max_text_length", 500)
        if len(text) > max_len:
            text = text[:max_len]
            logger.debug("TTS text truncated to {} chars", max_len)

        # Check cache
        cached_audio = self.cache.get(text, voice_id, speed)
        if cached_audio is not None:
            logger.debug("TTS cache hit: \"{}\"", text[:40])
            self._speaking = True
            if self._on_start:
                self._on_start(text)

            await self.playback.play(
                cached_audio,
                self.engine.sample_rate,
                text=text,
            )
            await self.playback.wait_for_completion()

            self._speaking = False
            if self._on_end:
                self._on_end(text)
            return True

        # Cache miss — synthesise
        logger.debug("TTS cache miss: \"{}\"", text[:40])
        self._speaking = True
        if self._on_start:
            self._on_start(text)

        result = await self.engine.synthesize(text, voice_id, speed)

        if len(result.audio) > 0:
            # Cache the result
            self.cache.put(text, voice_id, speed, result.audio)

            # Play
            await self.playback.play(
                result.audio,
                self.engine.sample_rate,
                text=text,
            )
            await self.playback.wait_for_completion()

        self._speaking = False
        if self._on_end:
            self._on_end(text)

        return len(result.audio) > 0

    async def say_stream(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float | None = None,
    ) -> bool:
        """Synthesise and play text (streaming — lower latency).

        Uses streaming synthesis to start playback before full
        synthesis completes. Sentences are played as they arrive.

        Args:
            text: Text to speak.
            voice_id: Voice override.
            speed: Speed override.

        Returns:
            True if any audio was played.
        """
        if not text or not text.strip():
            return False

        voice_id = voice_id or self.engine.default_voice
        speed = speed or self.engine.default_speed
        text = text.strip()

        self._speaking = True
        if self._on_start:
            self._on_start(text)

        played = False
        try:
            async for chunk in self.engine.synthesize_stream(
                text, voice_id, speed,
            ):
                if len(chunk) > 0:
                    await self.playback.play(
                        chunk,
                        self.engine.sample_rate,
                        text=text,
                    )
                    played = True

            # Wait for last chunk to finish
            if played:
                await self.playback.wait_for_completion()
        except Exception as e:
            logger.error("TTS stream error: {}", e)

        self._speaking = False
        if self._on_end:
            self._on_end(text)

        return played

    async def stop(self) -> None:
        """Stop current TTS playback immediately.

        Interrupts playback and resets speaking state.
        Used for Phase 5 (interruption mechanism).
        """
        self._speaking = False
        await self.playback.stop_current()
        logger.debug("TTS playback stopped")

    def pause(self) -> None:
        """Pause TTS playback."""
        self.playback.pause()

    def resume(self) -> None:
        """Resume TTS playback."""
        self.playback.resume()

    def clear_cache(self) -> None:
        """Clear the voice cache."""
        self.cache.clear()
        logger.debug("TTS cache cleared")

    def cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, size, hit_rate.
        """
        return self.cache.stats()

    async def cleanup(self) -> None:
        """Release all resources."""
        await self.playback.stop()
        await self.engine.cleanup()
        self.cache.clear()
        logger.info("TTSManager resources released")
