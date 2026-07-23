"""
Audio pipeline orchestration module.

Manages the three concurrent asyncio tasks (capture, VAD, playback)
and coordinates the flow of audio data between them.

Architecture:
    _capture_loop ──raw──→  _vad_loop  ──speech_events──→  consumer
         │                                                     │
         │                   ┌─────────────────────────────────┘
         │                   ▼
         │             _playback_loop  ←── external audio (TTS)
         │
    sounddevice InputStream    sounddevice OutputStream
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np

from src.audio.capture import AudioCapture
from src.audio.playback import AudioPlayback
from src.audio.resampler import Resampler
from src.config import Config
from src.logger import get_logger
from src.vad.silero_vad import SileroVAD, SpeechEvent, SpeechEventType

logger = get_logger(__name__)

# Type alias for speech event callbacks
SpeechCallback = Callable[[SpeechEvent], Any]


class AudioPipeline:
    """Audio pipeline orchestrator.

    Manages capture → resample → VAD → event dispatch → playback
    as concurrent asyncio tasks.

    Attributes:
        capture: Audio capture instance.
        playback: Audio playback instance.
        vad: VAD detector instance.
        resampler: Audio resampler instance.
        config: Application configuration.
        speech_callbacks: Registered speech event callbacks.
    """

    def __init__(self, config: Config):
        """Initialize pipeline from config.

        Args:
            config: Application configuration.
        """
        self.config = config
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._speech_callbacks: list[SpeechCallback] = []

        # Initialize components
        self.resampler = Resampler(
            source_rate=config.audio.sample_rate,
            target_rate=config.audio.target_sample_rate,
        )

        self.capture = AudioCapture(
            device_id=config.audio.input_device,
            sample_rate=config.audio.sample_rate,
            frames_per_buffer=config.audio.frames_per_buffer,
            target_sample_rate=config.audio.target_sample_rate,
        )

        self.playback = AudioPlayback(
            device_id=config.audio.output_device,
            sample_rate=config.audio.output_sample_rate,
        )

        self.vad = SileroVAD(
            model_path=config.vad.model_path,
            threshold=config.vad.threshold,
            min_speech_duration=config.vad.min_speech_duration,
            min_silence_duration=config.vad.min_silence_duration,
            frame_size=config.vad.frame_size,
            sample_rate=config.audio.target_sample_rate,
        )

    def on_speech_event(self, callback: SpeechCallback) -> None:
        """Register a callback for speech events.

        Args:
            callback: Function to call on speech events.
        """
        self._speech_callbacks.append(callback)

    def _dispatch_speech_event(self, event: SpeechEvent) -> None:
        """Dispatch speech event to all registered callbacks.

        Args:
            event: Speech event to dispatch.
        """
        for callback in self._speech_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("Speech callback error: {}", e)

    async def _capture_loop(self) -> None:
        """Capture audio from microphone and push to VAD.

        This task reads VAD-ready frames from AudioCapture
        and pushes them to the VAD detector.
        """
        logger.info("Capture loop started")

        frame_count = 0
        try:
            async for frame in self.capture.read_frames():
                if not self._running:
                    break

                frame_count += 1
                event = await self.vad.process_frame(frame)

                if event is not None:
                    self._dispatch_speech_event(event)

        except asyncio.CancelledError:
            logger.debug("Capture loop cancelled")
        except Exception as e:
            logger.error("Capture loop error: {}", e)
        finally:
            logger.info("Capture loop stopped (processed {} frames)", frame_count)

    async def _playback_loop(self) -> None:
        """Playback loop - processes queued audio.

        Currently idle; playback is driven by external calls to
        self.playback.play(). In future phases, this will also
        handle TTS audio from AIRI.
        """
        logger.info("Playback loop started")
        try:
            # Keep alive - playback is callback-driven
            while self._running:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.debug("Playback loop cancelled")
        except Exception as e:
            logger.error("Playback loop error: {}", e)
        finally:
            logger.info("Playback loop stopped")

    async def start(self) -> None:
        """Start the audio pipeline.

        Launches capture, VAD dispatch, and playback as concurrent tasks.
        """
        if self._running:
            logger.warning("Pipeline already running")
            return

        self._running = True

        # Start physical devices
        await self.capture.start()
        await self.playback.start()

        # Load VAD model
        self.vad.load_model()

        # Launch concurrent tasks
        self._tasks = [
            asyncio.create_task(self._capture_loop(), name="capture"),
            asyncio.create_task(self._playback_loop(), name="playback"),
        ]

        logger.info("Pipeline started with {} tasks", len(self._tasks))

    async def stop(self) -> None:
        """Stop the audio pipeline gracefully."""
        if not self._running:
            return

        logger.info("Stopping pipeline...")

        self._running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to finish
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

        # Flush VAD (capture remaining speech)
        event = self.vad.flush()
        if event is not None:
            self._dispatch_speech_event(event)

        # Stop physical devices
        await self.capture.stop()
        await self.playback.stop()

        logger.info("Pipeline stopped")

    async def play_audio(self, audio: np.ndarray,
                         sample_rate: int | None = None,
                         text: str = "") -> None:
        """Queue audio for playback.

        Args:
            audio: Audio data as float32 numpy array.
            sample_rate: Sample rate (uses default if None).
            text: Optional associated text.
        """
        await self.playback.play(audio, sample_rate, text)

    def list_audio_devices(self) -> tuple[list[dict], list[dict]]:
        """List available audio devices.

        Returns:
            Tuple of (input_devices, output_devices).
        """
        return (AudioCapture.list_devices(), AudioPlayback.list_devices())

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running

    @property
    def vad_state(self):
        """Get current VAD state."""
        return self.vad.state

    @property
    def is_speaking(self) -> bool:
        """Check if VAD currently detects speech."""
        return self.vad.is_speaking
