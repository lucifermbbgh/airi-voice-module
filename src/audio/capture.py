"""
Microphone audio capture module.

Uses sounddevice.InputStream to capture real-time audio from the
default (or specified) microphone. Manages the audio ring buffer
and coordinates with the resampler for VAD-ready output.

Designed for low-latency streaming: the sounddevice callback runs
in a C thread and writes raw frames to the ring buffer. An asyncio
task then reads, resamples, and pushes to the VAD pipeline.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np
import sounddevice as sd

from src.audio.resampler import Resampler
from src.pipeline.ring_buffer import AudioRingBuffer
from src.logger import get_logger

logger = get_logger(__name__)


class AudioCapture:
    """Microphone audio capture with resampling pipeline.

    Manages sounddevice InputStream, ring buffer, and async iteration
    of VAD-ready audio frames.

    Attributes:
        device_id: Sounddevice device ID (None = default).
        sample_rate: Capture sample rate in Hz.
        frames_per_buffer: Samples per callback frame.
        channels: Number of audio channels (1 = mono).
    """

    def __init__(
        self,
        device_id: int | None = None,
        sample_rate: int = 48000,
        frames_per_buffer: int = 512,
        channels: int = 1,
        target_sample_rate: int = 16000,
    ):
        """Initialize audio capture.

        Args:
            device_id: Sounddevice device ID (None = default).
            sample_rate: Capture sample rate in Hz.
            frames_per_buffer: Samples per callback frame.
            channels: Number of audio channels.
            target_sample_rate: Target sample rate for VAD.
        """
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.frames_per_buffer = frames_per_buffer
        self.channels = channels
        self.target_sample_rate = target_sample_rate

        self._stream: sd.InputStream | None = None
        self._running = False
        self._buffer = AudioRingBuffer()
        self._resampler = Resampler(sample_rate, target_sample_rate)

    def _callback(self, indata: np.ndarray, frames: int,
                  _time_info, _status: sd.CallbackFlags) -> None:
        """Sounddevice callback - called from C thread.

        Args:
            indata: Input audio data (frames, channels).
            frames: Number of frames.
            _time_info: Time info (unused).
            _status: Status flags (unused).
        """
        if _status:
            logger.warning("Sounddevice callback status: {}", _status)

        # Write raw audio to ring buffer (thread-safe)
        # indata[:, 0] always returns 1D array regardless of mono/stereo
        audio = indata[:, 0].copy()
        self._buffer.write_raw(audio.astype(np.float32), self.sample_rate)

    async def start(self) -> None:
        """Start audio capture stream."""
        if self._running:
            logger.warning("Capture already running")
            return

        logger.info(
            "Starting capture: device={}, rate={}, buffer={}, channels={}",
            self.device_id or "default",
            self.sample_rate,
            self.frames_per_buffer,
            self.channels,
        )

        self._stream = sd.InputStream(
            device=self.device_id,
            samplerate=self.sample_rate,
            blocksize=self.frames_per_buffer,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
            latency="low",
        )

        self._running = True
        self._stream.start()
        logger.info("Capture started")

    async def stop(self) -> None:
        """Stop audio capture stream."""
        self._running = False
        self._buffer.stop()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Capture stopped")

    async def read_frames(self) -> AsyncIterator[np.ndarray]:
        """Async iterator yielding VAD-ready audio frames.

        Each frame is a 1D float32 numpy array at target_sample_rate (16kHz),
        with the standard VAD frame size of 512 samples (32ms per frame).

        Raw frames from the sounddevice C callback are collected from the
        ring buffer deque, resampled (48kHz → 16kHz), and accumulated until
        a full VAD frame is ready.
        """
        resampler = self._resampler
        frame_buffer = np.array([], dtype=np.float32)
        vad_frame_size = 512  # VAD expects 512 samples @ 16kHz (32ms)

        try:
            while self._running or len(frame_buffer) >= vad_frame_size:
                # Collect raw frames written by sounddevice C callback
                raw_frames = self._buffer.get_raw_history()
                if raw_frames:
                    self._buffer._raw_frames.clear()

                    # Concatenate all pending raw frames
                    if len(raw_frames) == 1:
                        raw_audio = raw_frames[0]
                    else:
                        raw_audio = np.concatenate(raw_frames)

                    # Resample 48kHz → 16kHz
                    resampled = resampler.resample(raw_audio)

                    if len(resampled) > 0:
                        frame_buffer = np.concatenate([frame_buffer, resampled])

                # Yield complete VAD frames as they become available
                while len(frame_buffer) >= vad_frame_size:
                    yield frame_buffer[:vad_frame_size]
                    frame_buffer = frame_buffer[vad_frame_size:]

                if not raw_frames:
                    await asyncio.sleep(0.005)  # No data, brief yield
        finally:
            # Flush any remaining audio on stop
            if len(frame_buffer) > 0:
                logger.debug("Flushing {} remaining resampled samples", len(frame_buffer))
                # Pad with silence to make a full VAD frame
                if len(frame_buffer) > 0:
                    yield frame_buffer

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    @property
    def buffer(self) -> AudioRingBuffer:
        """Get the internal ring buffer."""
        return self._buffer

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices.

        Returns:
            List of device info dicts with id, name, channels, rate.
        """
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append({
                    "id": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_samplerate": dev["default_samplerate"],
                    "host_api": sd.query_hostapis(dev["hostapi"])["name"],
                })
        return devices
