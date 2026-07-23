"""
Audio playback module.

Uses sounddevice.OutputStream to play audio to the default (or specified)
speaker output device. Supports queue-based playback with stop/interrupt.

Designed for low-latency streaming from TTS output.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PlaybackSegment:
    """A single audio segment for playback.

    Attributes:
        audio: Audio data as float32 numpy array.
        sample_rate: Sample rate of the audio.
        text: Associated text (for TTS).
        sequence: Sequence number for ordering.
    """
    audio: np.ndarray
    sample_rate: int
    text: str = ""
    sequence: int = 0


class AudioPlayback:
    """Audio playback engine with queue support.

    Attributes:
        device_id: Sounddevice device ID (None = default).
        sample_rate: Output sample rate in Hz.
        channels: Number of output channels.
    """

    def __init__(
        self,
        device_id: int | None = None,
        sample_rate: int = 24000,
        channels: int = 1,
        blocksize: int = 512,
    ):
        """Initialize audio playback.

        Args:
            device_id: Sounddevice device ID (None = default).
            sample_rate: Output sample rate in Hz.
            channels: Number of output channels.
            blocksize: Stream blocksize.
        """
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize

        self._stream: sd.OutputStream | None = None
        self._queue: deque[PlaybackSegment] = deque()
        self._current: PlaybackSegment | None = None
        self._current_pos = 0
        self._running = False
        self._paused = False
        self._event = asyncio.Event()

    def _callback(self, outdata: np.ndarray, frames: int,
                  _time_info, _status: sd.CallbackFlags) -> None:
        """Sounddevice output callback.

        Args:
            outdata: Output buffer to fill.
            frames: Number of frames requested.
            _time_info: Time info (unused).
            _status: Status flags.
        """
        if _status:
            logger.warning("Playback callback status: {}", _status)

        if self._paused or not self._current:
            outdata.fill(0)
            return

        segment = self._current
        remaining = len(segment.audio) - self._current_pos
        to_copy = min(frames, remaining)

        if to_copy > 0:
            chunk = segment.audio[self._current_pos:self._current_pos + to_copy]
            if chunk.ndim == 1:
                outdata[:to_copy, 0] = chunk
            else:
                outdata[:to_copy] = chunk
            self._current_pos += to_copy

        # Fill remainder with silence
        if to_copy < frames:
            outdata[to_copy:].fill(0)

        # If segment finished, move to next
        if self._current_pos >= len(segment.audio):
            self._advance_to_next()

    def _advance_to_next(self) -> None:
        """Advance to the next queued segment."""
        if self._queue:
            self._current = self._queue.popleft()
            self._current_pos = 0
            logger.debug("Playing next segment (seq={}, {} samples)",
                         self._current.sequence, len(self._current.audio))
        else:
            self._current = None
            self._current_pos = 0
            self._event.set()

    async def start(self) -> None:
        """Start playback stream."""
        if self._running:
            logger.warning("Playback already running")
            return

        logger.info(
            "Starting playback: device={}, rate={}, channels={}",
            self.device_id or "default",
            self.sample_rate,
            self.channels,
        )

        self._stream = sd.OutputStream(
            device=self.device_id,
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
            latency="low",
        )

        self._running = True
        self._stream.start()
        logger.info("Playback started")

    async def stop(self) -> None:
        """Stop playback stream and clear queue."""
        self._running = False
        self._queue.clear()
        self._current = None
        self._current_pos = 0
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._event.set()
        logger.info("Playback stopped")

    async def play(self, audio: np.ndarray,
                   sample_rate: int | None = None,
                   text: str = "") -> None:
        """Queue audio for playback.

        Args:
            audio: Audio data as float32 numpy array.
            sample_rate: Sample rate of the audio (uses default if None).
            text: Optional associated text.
        """
        if not self._running:
            logger.warning("Playback not running, ignoring play request")
            return

        segment = PlaybackSegment(
            audio=audio.copy(),
            sample_rate=sample_rate or self.sample_rate,
            text=text,
            sequence=len(self._queue),
        )

        self._queue.append(segment)

        if self._current is None:
            self._advance_to_next()

    async def stop_current(self) -> None:
        """Stop current playback immediately."""
        self._current = None
        self._current_pos = 0
        self._queue.clear()
        self._event.set()

    def pause(self) -> None:
        """Pause playback."""
        self._paused = True

    def resume(self) -> None:
        """Resume playback."""
        self._paused = False

    async def wait_for_completion(self) -> None:
        """Wait for all queued audio to finish playing."""
        if self._current is None and not self._queue:
            return
        self._event.clear()
        await self._event.wait()

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._current is not None

    @property
    def queue_size(self) -> int:
        """Get number of segments waiting in queue."""
        return len(self._queue)

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio output devices.

        Returns:
            List of device info dicts.
        """
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] > 0:
                devices.append({
                    "id": i,
                    "name": dev["name"],
                    "channels": dev["max_output_channels"],
                    "default_samplerate": dev["default_samplerate"],
                    "host_api": sd.query_hostapis(dev["hostapi"])["name"],
                })
        return devices
