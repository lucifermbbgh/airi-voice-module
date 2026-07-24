"""
TTS (Text-to-Speech) engine abstraction layer.

Defines the base interface for all TTS engines in the AIRI Voice Module.
All engine implementations must inherit from TTSBase.

Usage:
    class MyTTS(TTSBase):
        async def synthesize(self, text, voice_id="default", speed=1.0):
            ...
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np


@dataclass
class TTSResult:
    """TTS synthesis result.

    Attributes:
        audio: Synthesized audio data as float32 numpy array.
            Values typically in range [-1.0, 1.0].
        sample_rate: Sample rate of the audio in Hz.
        duration: Duration of the audio in seconds.
        text: Text that was synthesised.
        synthesis_time: Time taken for synthesis in seconds.
    """
    audio: np.ndarray
    sample_rate: int
    duration: float
    text: str
    synthesis_time: float

    def __post_init__(self) -> None:
        """Auto-calculate duration if not provided."""
        if self.duration <= 0 and len(self.audio) > 0:
            self.duration = len(self.audio) / self.sample_rate


class TTSBase(abc.ABC):
    """Abstract base class for TTS engines.

    All TTS engine implementations must subclass this and implement
    the abstract methods.

    Key design decisions:
        - Lazy loading: model loaded on first synthesize call
        - Async-first: all public methods are async
        - Engine-agnostic: TTSResult is the universal output format
        - Streaming support: synthesize_stream for low-latency playback

    Subclasses must define:
        - synthesize()     - single text → complete audio
        - synthesize_stream() - single text → audio chunks
        - load_model()     - load model into memory
        - unload_model()   - release model memory
        - cleanup()        - release all resources
        - voices property  - available voice list
        - name property    - engine identifier
        - is_loaded property - model load state
    """

    @abc.abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str = "default",
        speed: float = 1.0,
    ) -> TTSResult:
        """Synthesise text to speech (complete audio).

        Generates the complete audio for the given text.
        Suitable for short to medium-length utterances.

        Args:
            text: Text to synthesise.
            voice_id: Voice/ speaker identifier.
            speed: Speaking speed (0.5-2.0, 1.0 = normal).

        Returns:
            TTSResult with the complete synthesised audio.
        """
        ...

    @abc.abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice_id: str = "default",
        speed: float = 1.0,
    ) -> AsyncIterator[np.ndarray]:
        """Synthesise text to speech (streaming chunks).

        Generates audio in chunks for low-latency playback.
        Each chunk is ~200ms of audio. The caller can start
        playing as soon as the first chunk arrives.

        Args:
            text: Text to synthesise.
            voice_id: Voice/ speaker identifier.
            speed: Speaking speed (0.5-2.0, 1.0 = normal).

        Yields:
            Audio chunks as float32 numpy arrays.
            Each chunk is ~200ms at the engine's sample rate.
        """
        if False:  # pragma: no cover (async generator stub)
            yield np.array([], dtype=np.float32)

    # ── Model Lifecycle ─────────────────────────────────────────────

    @abc.abstractmethod
    async def load_model(self) -> None:
        """Load the TTS model into memory.

        Called lazily on first synthesize() if not already loaded.
        """
        ...

    @abc.abstractmethod
    async def unload_model(self) -> None:
        """Unload the TTS model to free memory."""
        ...

    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Release all resources (model, threads, etc.)."""
        ...

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        ...

    @property
    @abc.abstractmethod
    def voices(self) -> list[dict]:
        """Get list of available voices.

        Each voice is a dict with:
            - id: Voice identifier used in synthesize()
            - name: Human-readable name
            - description: Optional description

        Returns:
            List of voice info dicts.
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get engine name (e.g., 'cosyvoice', 'edge_tts')."""
        ...

    # ── Utility Helpers ─────────────────────────────────────────────

    @staticmethod
    def normalize_volume(audio: np.ndarray,
                         target_peak: float = 0.95) -> np.ndarray:
        """Normalise audio volume to target peak amplitude.

        Args:
            audio: Input audio array (float32).
            target_peak: Target peak amplitude (0.0-1.0).

        Returns:
            Volume-normalised audio.
        """
        if len(audio) == 0:
            return audio

        current_peak = float(np.max(np.abs(audio)))
        if current_peak > 0:
            gain = target_peak / current_peak
            scaled = audio * gain
            # Clip to target peak to avoid floating-point overshoot
            return np.clip(scaled, -target_peak, target_peak)
        return audio

    @staticmethod
    def resample(audio: np.ndarray,
                 source_rate: int,
                 target_rate: int) -> np.ndarray:
        """Resample audio to a different sample rate.

        Uses simple linear interpolation for speed.
        For production use, consider scipy.signal.resample.

        Args:
            audio: Input audio array.
            source_rate: Current sample rate.
            target_rate: Desired sample rate.

        Returns:
            Resampled audio array.
        """
        if source_rate == target_rate or len(audio) == 0:
            return audio

        # Calculate new length
        num_samples = int(len(audio) * target_rate / source_rate)

        # Linear interpolation
        indices = np.linspace(0, len(audio) - 1, num_samples)
        left = indices.astype(np.int64)
        right = np.clip(left + 1, 0, len(audio) - 1)
        frac = indices - left

        return audio[left] * (1 - frac) + audio[right] * frac
