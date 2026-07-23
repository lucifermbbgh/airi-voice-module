"""
Silero VAD (Voice Activity Detection) module.

Integrates Silero VAD v5 ONNX model for real-time speech detection.
Implements a state machine that tracks speech/silence segments and
emits structured SpeechEvent objects.

The VAD processes 512-sample frames at 16kHz (32ms per frame) with
a configurable sliding window for smoothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


class SpeechState(Enum):
    """VAD state machine states."""
    SILENCE = "silence"
    SPEECH = "speech"
    PENDING_START = "pending_start"  # Waiting for min_speech_duration confirmation
    PENDING_END = "pending_end"     # Waiting for min_silence_duration confirmation


class SpeechEventType(Enum):
    """Types of speech events emitted by the VAD."""
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    UTTERANCE = "utterance"  # Complete utterance with full audio


@dataclass
class SpeechEvent:
    """Event emitted when speech starts or ends.

    Attributes:
        type: Event type (START, END, or UTTERANCE).
        audio: Audio data for the speech segment (only for END/UTTERANCE).
        timestamp: Monotonic timestamp of the event.
        duration: Duration of the speech segment in seconds (only for END).
        num_frames: Number of VAD frames in the segment.
        max_prob: Maximum VAD probability in the segment.
    """
    type: SpeechEventType
    audio: np.ndarray | None = None
    timestamp: float = 0.0
    duration: float = 0.0
    num_frames: int = 0
    max_prob: float = 0.0
    sample_rate: int = 16000


class SileroVAD:
    """Silero VAD detector with state machine.

    Processes streaming audio frames and emits speech events based on
    configurable thresholds and timing constraints.

    Attributes:
        model_path: Path to Silero VAD ONNX model.
        threshold: Speech probability threshold (0.0-1.0).
        min_speech_duration: Minimum speech duration in seconds.
        min_silence_duration: Minimum silence to end utterance in seconds.
        frame_size: Frame size in samples (at target sample rate).
        sample_rate: Expected input sample rate.
    """

    # Standard Silero VAD configuration
    _SILERO_SAMPLE_RATE = 16000
    _SILERO_FRAME_SIZE = 512  # 32ms at 16kHz

    def __init__(
        self,
        model_path: str | Path = "models/silero_vad.onnx",
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.5,
        frame_size: int = 512,
        sample_rate: int = 16000,
    ):
        """Initialize VAD detector.

        Args:
            model_path: Path to ONNX model file.
            threshold: Speech probability threshold.
            min_speech_duration: Minimum speech to confirm utterance (s).
            min_silence_duration: Silence to end utterance (s).
            frame_size: Audio frame size in samples.
            sample_rate: Expected audio sample rate.
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.min_speech_frames = max(1, int(min_speech_duration * sample_rate / frame_size))
        self.min_silence_frames = max(1, int(min_silence_duration * sample_rate / frame_size))
        self.frame_size = frame_size
        self.sample_rate = sample_rate

        self._model = None
        self._get_speech_timestamps = None
        self._collect_audio = None
        self._hn = None  # Hidden state for stateful ONNX model

        # State machine
        self._state = SpeechState.SILENCE
        self._speech_frames: list[np.ndarray] = []
        self._silence_frames = 0
        self._speech_frame_count = 0
        self._max_prob: float = 0.0
        self._start_time: float = 0.0
        self._frame_duration = frame_size / sample_rate

        logger.info(
            "VAD initialized: threshold={}, min_speech={}s ({} frames), "
            "min_silence={}s ({} frames)",
            threshold, min_speech_duration, self.min_speech_frames,
            min_silence_duration, self.min_silence_frames,
        )

    def load_model(self) -> None:
        """Load Silero VAD ONNX model.

        Uses two strategies in order:
        1. Find the bundled ONNX model from the installed silero-vad pip package
           (v4+) using importlib.resources.
        2. Fall back to the configured model_path on disk.

        Call this before processing any audio.
        """
        if self._model is not None:
            return

        # Strategy 1: bundled model from the silero-vad pip package
        try:
            import importlib.resources as ir

            model_file = ir.files("silero_vad.data").joinpath("silero_vad.onnx")
            if model_file.exists():
                self.model_path = Path(str(model_file))
                logger.info("Found bundled VAD model: {}", self.model_path)
                self._load_onnx_direct()
                return
        except (ImportError, KeyError, TypeError, FileNotFoundError, RuntimeError) as e:
            logger.debug("Bundled model lookup failed: {}", e)

        # Strategy 2: explicit model_path on disk
        logger.info("Falling back to configured model path: {}", self.model_path)
        self._load_onnx_direct()

    def _load_onnx_direct(self) -> None:
        """Load VAD model directly via ONNX Runtime.

        Fallback if silero_vad package is not available.
        """
        import onnxruntime

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"VAD model not found: {self.model_path}\n"
                "The model is bundled in the silero-vad pip package.\n"
                "Make sure it is installed: pip install silero-vad"
            )

        providers = [
            # CUDAExecutionProvider is optional; only use if available
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        # Filter to only available providers to avoid warnings
        available = onnxruntime.get_available_providers()
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        self._model = onnxruntime.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        logger.info(
            "VAD model loaded via ONNX Runtime: {}",
            self._model.get_modelmeta().description,
        )

    def _get_speech_prob(self, audio_frame: np.ndarray) -> float:
        """Get speech probability for a single audio frame.

        Args:
            audio_frame: 1D float32 array of length frame_size at 16kHz.

        Returns:
            Speech probability (0.0 to 1.0).
        """
        if len(audio_frame) != self.frame_size:
            raise ValueError(
                f"Expected frame size {self.frame_size}, got {len(audio_frame)}"
            )

        # Direct ONNX Runtime inference
        # Dynamically build input feed from model's expected inputs
        input_feed = {}
        for inp in self._model.get_inputs():
            name = inp.name

            if name == 'input':
                input_feed[name] = audio_frame.reshape(1, -1).astype(np.float32)
            elif name == 'sr':
                input_feed[name] = np.array([self.sample_rate], dtype=np.int64)
            elif 'state' in name.lower() or 'hn' in name.lower():
                # Stateful input: initialize or reuse hidden state
                shape = inp.shape
                # Handle dynamic dimensions (None in shape means dynamic)
                state_shape = tuple(
                    1 if d is None or (isinstance(d, int) and d <= 0) else d
                    for d in shape
                )
                if self._hn is None or self._hn.shape != state_shape:
                    self._hn = np.zeros(state_shape, dtype=np.float32)
                input_feed[name] = self._hn
            else:
                # Unknown input: try zeros with expected shape
                shape = inp.shape
                default_shape = tuple(
                    1 if d is None or (isinstance(d, int) and d <= 0) else d
                    for d in shape
                )
                input_feed[name] = np.zeros(default_shape, dtype=np.float32)

        # Run inference
        output_names = [o.name for o in self._model.get_outputs()]
        outputs = self._model.run(output_names, input_feed)

        # Update hidden state if model returned it (stateful models)
        if self._hn is not None and len(outputs) > 1:
            for i, o in enumerate(self._model.get_outputs()):
                if 'state' in o.name.lower() or 'hn' in o.name.lower():
                    self._hn = outputs[i]
                    break

        return float(outputs[0][0][0])

    async def process_frame(self, audio_frame: np.ndarray) -> SpeechEvent | None:
        """Process a single audio frame and emit speech events.

        Args:
            audio_frame: 1D float32 array, 512 samples @ 16kHz.

        Returns:
            SpeechEvent if state transition occurred, else None.
        """
        if self._model is None:
            self.load_model()

        prob = self._get_speech_prob(audio_frame)
        is_speech = prob >= self.threshold
        now = time.monotonic()

        return self._update_state(prob, is_speech, audio_frame, now)

    def _update_state(
        self,
        prob: float,
        is_speech: bool,
        audio_frame: np.ndarray,
        now: float,
    ) -> SpeechEvent | None:
        """Update VAD state machine.

        Args:
            prob: Speech probability.
            is_speech: Whether probability exceeds threshold.
            audio_frame: Current audio frame.
            now: Current monotonic time.

        Returns:
            SpeechEvent if state changed, else None.
        """
        event = None

        if self._state == SpeechState.SILENCE:
            if is_speech:
                # Potential speech start - enter pending
                self._state = SpeechState.PENDING_START
                self._speech_frames = [audio_frame.copy()]
                self._speech_frame_count = 1
                self._max_prob = prob
                self._start_time = now
                logger.debug("VAD: pending start (prob={:.3f})", prob)
            # else: remain silent

        elif self._state == SpeechState.PENDING_START:
            if is_speech:
                self._speech_frames.append(audio_frame.copy())
                self._speech_frame_count += 1
                self._max_prob = max(self._max_prob, prob)

                if self._speech_frame_count >= self.min_speech_frames:
                    # Confirmed speech!
                    self._state = SpeechState.SPEECH
                    event = SpeechEvent(
                        type=SpeechEventType.SPEECH_START,
                        timestamp=self._start_time,
                    )
                    logger.debug(
                        "VAD: speech START (confirmed after {} frames)",
                        self._speech_frame_count,
                    )
            else:
                # False start - noise spike
                self._state = SpeechState.SILENCE
                self._speech_frames.clear()
                self._speech_frame_count = 0
                logger.debug("VAD: pending cancelled (false start)")

        elif self._state == SpeechState.SPEECH:
            if is_speech:
                self._speech_frames.append(audio_frame.copy())
                self._speech_frame_count += 1
                self._max_prob = max(self._max_prob, prob)
                self._silence_frames = 0
            else:
                # Silence detected during speech
                self._silence_frames += 1
                self._speech_frames.append(audio_frame.copy())
                self._speech_frame_count += 1

                if self._silence_frames >= self.min_silence_frames:
                    # End of utterance
                    event = self._create_speech_end_event(now)
                    self._reset()
                    logger.debug(
                        "VAD: speech END (silence={} frames, dur={:.2f}s)",
                        self._silence_frames,
                        event.duration if event else 0,
                    )

        return event

    def flush(self) -> SpeechEvent | None:
        """Force end current speech segment.

        Call this when the pipeline is stopping to capture any
        remaining speech audio.

        Returns:
            SpeechEvent for the pending utterance, or None.
        """
        if self._state in (SpeechState.SPEECH, SpeechState.PENDING_START):
            event = self._create_speech_end_event(time.monotonic())
            self._reset()
            logger.debug("VAD: flushed speech segment (dur={:.2f}s)",
                         event.duration if event else 0)
            return event
        self._reset()
        return None

    def _create_speech_end_event(self, now: float) -> SpeechEvent | None:
        """Create a SPEECH_END event from accumulated frames.

        Args:
            now: Current time.

        Returns:
            SpeechEvent or None if no valid audio.
        """
        if not self._speech_frames:
            return None

        audio = np.concatenate(self._speech_frames)
        duration = len(audio) / self.sample_rate

        return SpeechEvent(
            type=SpeechEventType.SPEECH_END,
            audio=audio,
            timestamp=self._start_time,
            duration=duration,
            num_frames=self._speech_frame_count,
            max_prob=self._max_prob,
            sample_rate=self.sample_rate,
        )

    def _reset(self) -> None:
        """Reset VAD state to silence."""
        self._state = SpeechState.SILENCE
        self._speech_frames.clear()
        self._speech_frame_count = 0
        self._silence_frames = 0
        self._max_prob = 0.0
        self._start_time = 0.0

    async def process_stream(
        self,
        frames: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[SpeechEvent]:
        """Process a stream of audio frames.

        Args:
            frames: Async iterator of audio frames (float32, 512 @ 16kHz).

        Yields:
            SpeechEvent on state transitions.
        """
        async for frame in frames:
            event = await self.process_frame(frame)
            if event is not None:
                yield event

    @property
    def state(self) -> SpeechState:
        """Get current VAD state."""
        return self._state

    @property
    def is_speaking(self) -> bool:
        """Check if VAD currently detects speech."""
        return self._state in (SpeechState.SPEECH, SpeechState.PENDING_START)
