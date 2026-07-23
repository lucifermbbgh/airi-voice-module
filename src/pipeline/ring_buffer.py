"""
Audio ring buffer with asyncio support.

Provides thread-safe buffering between the sounddevice callback
(which runs in a C thread) and the asyncio pipeline consumers.

Uses a two-stage architecture:
  Stage 1: deques for raw audio from sounddevice callback
  Stage 2: asyncio.Queue for resampled audio consumed by VAD
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import AsyncIterator

import numpy as np


class AudioRingBuffer:
    """Thread-safe audio ring buffer bridging sounddevice and asyncio.

    Stage 1 (thread-safe): sounddevice callback writes raw frames.
    Stage 2 (asyncio): VAD consumer reads resampled frames.

    Attributes:
        max_history: Maximum number of raw frames to keep.
        sample_rate: Sample rate of audio stored.
    """

    def __init__(self, max_history: int = 100):
        """Initialize ring buffer.

        Args:
            max_history: Maximum raw frames to keep in history.
        """
        self._raw_frames: deque[np.ndarray] = deque(maxlen=max_history)
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._running = False
        self._sample_rate: int = 0

    def write_raw(self, frame: np.ndarray, sample_rate: int) -> None:
        """Write raw audio frame from sounddevice callback.

        Thread-safe: called from sounddevice callback thread.

        Args:
            frame: Audio frame as numpy array (float32).
            sample_rate: Sample rate of the frame.
        """
        self._sample_rate = sample_rate
        self._raw_frames.append(frame)

    async def write_processed(self, frame: np.ndarray) -> None:
        """Write processed (resampled) audio for VAD consumption.

        Called from asyncio context after resampling.

        Args:
            frame: Processed audio frame as numpy array.
        """
        if self._running:
            await self._queue.put(frame)

    async def read(self) -> AsyncIterator[np.ndarray]:
        """Async iterator yielding processed audio frames.

        Yields:
            Audio frames as numpy arrays.
        """
        self._running = True
        try:
            while self._running:
                try:
                    frame = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                    yield frame
                except asyncio.TimeoutError:
                    continue
        finally:
            self._running = False

    def get_raw_history(self) -> list[np.ndarray]:
        """Get all raw frames in history buffer.

        Returns:
            List of raw audio frames (oldest first).
        """
        return list(self._raw_frames)

    def get_raw_concatenated(self) -> np.ndarray:
        """Get all raw frames concatenated into one array.

        Returns:
            Concatenated raw audio as a 1D numpy array.
        """
        frames = self.get_raw_history()
        if not frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(frames)

    @property
    def sample_rate(self) -> int:
        """Get sample rate of stored audio."""
        return self._sample_rate

    @property
    def qsize(self) -> int:
        """Get number of frames waiting in the processed queue."""
        return self._queue.qsize()

    def stop(self) -> None:
        """Stop the read iterator."""
        self._running = False

    def clear(self) -> None:
        """Clear all buffers."""
        self._raw_frames.clear()
        # Drain the asyncio queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
