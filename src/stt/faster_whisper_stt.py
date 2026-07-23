"""
Faster-Whisper STT (Speech-to-Text) engine.

Converts speech audio segments from VAD into text using OpenAI's
Whisper model via the CTranslate2 inference backend (faster-whisper).

Key design decisions:
- int8 quantization for CPU-friendly inference
- Lazy model loading (model loaded on first transcribe call)
- Async wrapper around sync Faster-Whisper via run_in_executor
- Language-preferenced inference for better Chinese accuracy
- Hotword support for domain-specific terminology

Usage:
    stt = FasterWhisperSTT(model_size="small", device="cpu")
    result = await stt.transcribe(audio_array)
    print(result.text)  # "今天天气怎么样"
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────

_STANDARD_SAMPLE_RATE = 16000  # Whisper expects 16kHz
_MIN_AUDIO_DURATION = 0.1     # Minimum audio duration in seconds
_MAX_AUDIO_DURATION = 30.0    # Maximum for real-time use (30s per segment)
_SILENCE_THRESHOLD_RMS = 0.01  # RMS below this is considered silence


# ── Data Classes ───────────────────────────────────────────────────


@dataclass
class STTResult:
    """STT inference result.

    Attributes:
        text: Recognized text.
        confidence: Overall confidence score (0.0 to 1.0).
        language: Detected or specified language code.
        language_probability: Confidence in language detection.
        duration: Duration of input audio in seconds.
        inference_time: Time taken for inference in seconds.
        segments: Optional list of segment dicts with timestamps.
    """
    text: str
    confidence: float
    language: str
    language_probability: float
    duration: float
    inference_time: float
    segments: list[dict] | None = None


# ── STT Engine ─────────────────────────────────────────────────────


class FasterWhisperSTT:
    """Faster-Whisper STT engine with async wrapper.

    Provides speech-to-text conversion using CTranslate2-optimized
    Whisper models. Designed for integration with the VAD pipeline's
    SpeechEvent.SPEECH_END callback.

    Attributes:
        model_size: Whisper model size identifier.
        device: Computation device ('cpu' or 'cuda').
        compute_type: Quantization type ('int8', 'float16', 'float32').
        language: Preferred language code (None = auto-detect).
        beam_size: Beam search width for decoding.
        vad_filter: Enable built-in VAD filtering in Whisper.
        hotwords: List of hotwords to boost recognition.
    """

    # Predefined model specifications: size → (approx RAM, typical RTF)
    MODELS: dict[str, dict] = {
        "tiny":   {"ram_mb": 400,  "rtf": 0.3},
        "base":   {"ram_mb": 600,  "rtf": 0.2},
        "small":  {"ram_mb": 1200, "rtf": 0.1},
        "medium": {"ram_mb": 2500, "rtf": 0.08},
        "large-v3": {"ram_mb": 3200, "rtf": 0.05},
    }

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        model_dir: str | None = None,
        language: str | None = "zh",
        beam_size: int = 5,
        vad_filter: bool = True,
        hotwords: list[str] | None = None,
    ):
        """Initialize STT engine.

        Args:
            model_size: Model size identifier.
                One of: tiny, base, small (default), medium, large-v3.
            device: Computation device ('cpu' or 'cuda').
            compute_type: Quantization type.
                'int8' for CPU (fastest), 'float16'/'float32' for GPU.
            model_dir: Directory for model storage.
                If None, uses huggingface_hub default cache.
            language: Preferred language code.
                'zh' for Chinese, 'en' for English, None for auto-detect.
            beam_size: Beam search width (1-10). Larger = better but slower.
            vad_filter: Enable Whisper's built-in VAD filtering.
                Removes silent segments before inference.
            hotwords: List of hotwords/phrases to boost recognition.
        """
        if model_size not in self.MODELS:
            valid = ", ".join(self.MODELS.keys())
            raise ValueError(f"Invalid model_size '{model_size}'. "
                             f"Choose from: {valid}")

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model_dir = model_dir
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.hotwords = list(hotwords) if hotwords else []

        # Internal state
        self._model = None      # Lazy-loaded FasterWhisper model
        self._executor = None   # ThreadPoolExecutor for async inference

        spec = self.MODELS[model_size]
        logger.info(
            "STT initialized: model={}, device={}, compute={}, "
            "language={}, ram≈{}MB, RTF≈{}",
            model_size, device, compute_type,
            language or "auto", spec["ram_mb"], spec["rtf"],
        )

    # ── Model Loading ──────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the Faster-Whisper model (lazy, called on first use).

        Uses CTranslate2 backend with specified quantization.
        Model is downloaded on first use via huggingface_hub.
        """
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed.\n"
                "Install it with: pip install faster-whisper\n"
                "Or add it to requirements.txt"
            )

        model_name = f"guillaumeklay/faster-whisper-{self.model_size}"

        logger.info(
            "Loading STT model: {} (device={}, compute={})",
            model_name, self.device, self.compute_type,
        )

        load_start = time.monotonic()
        self._model = WhisperModel(
            model_size_or_path=model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.model_dir,
        )
        load_time = time.monotonic() - load_start

        logger.info(
            "STT model loaded in {:.1f}s: {}",
            load_time, model_name,
        )

    def unload_model(self) -> None:
        """Unload model to free memory."""
        self._model = None
        logger.info("STT model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        return self._model is not None

    @property
    def model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "loaded": self.is_loaded,
            **self.MODELS.get(self.model_size, {}),
        }

    # ── Hotword Management ─────────────────────────────────────────

    def set_language(self, language: str | None) -> None:
        """Set or clear language preference.

        Args:
            language: Language code ('zh', 'en') or None for auto-detect.
        """
        old = self.language
        self.language = language
        logger.debug("STT language: {} → {}", old, language or "auto")

    def add_hotwords(self, hotwords: list[str]) -> None:
        """Add hotwords to boost recognition.

        Args:
            hotwords: List of words/phrases to prioritize.
        """
        for word in hotwords:
            if word and word not in self.hotwords:
                self.hotwords.append(word)
        logger.debug("STT hotwords: {} (total={})", hotwords, len(self.hotwords))

    # ── Audio Validation ──────────────────────────────────────────

    def _validate_audio(
        self, audio: np.ndarray, sample_rate: int,
    ) -> bool:
        """Validate audio format before inference.

        Args:
            audio: Audio array to validate.
            sample_rate: Audio sample rate.

        Returns:
            True if audio format is valid.
        """
        if audio.dtype != np.float32:
            logger.warning("Audio dtype is {}, expected float32", audio.dtype)
            return False

        if sample_rate != _STANDARD_SAMPLE_RATE:
            logger.warning(
                "Audio sample rate is {}Hz, expected {}Hz",
                sample_rate, _STANDARD_SAMPLE_RATE,
            )
            return False

        duration = len(audio) / sample_rate
        if duration < _MIN_AUDIO_DURATION:
            logger.debug("Audio too short: {:.3f}s", duration)
            return False

        if duration > _MAX_AUDIO_DURATION:
            logger.warning(
                "Audio too long for real-time: {:.1f}s (max={}s)",
                duration, _MAX_AUDIO_DURATION,
            )
            return False

        return True

    @staticmethod
    def _is_silence(audio: np.ndarray) -> bool:
        """Check if audio is effectively silence.

        Args:
            audio: Audio array (float32).

        Returns:
            True if audio amplitude is below silence threshold.
        """
        if len(audio) == 0:
            return True
        rms = np.sqrt(np.mean(audio ** 2))
        return rms < _SILENCE_THRESHOLD_RMS

    # ── Core Inference ─────────────────────────────────────────────

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribe speech audio to text.

        Args:
            audio: Float32 numpy array of speech audio.
            sample_rate: Sample rate (must be 16000).

        Returns:
            STTResult with recognized text and metadata.

        Performance (small + int8 + CPU):
            1s audio → ~100ms inference
            5s audio → ~500ms inference
            RTF ≈ 0.1
        """
        # Fast path: empty or silence
        if len(audio) == 0 or self._is_silence(audio):
            return STTResult(
                text="",
                confidence=0.0,
                language=self.language or "unknown",
                language_probability=0.0,
                duration=len(audio) / sample_rate,
                inference_time=0.0,
            )

        # Validate audio format
        if not self._validate_audio(audio, sample_rate):
            logger.warning("Audio validation failed, skipping transcription")
            return STTResult(
                text="",
                confidence=0.0,
                language=self.language or "unknown",
                language_probability=0.0,
                duration=len(audio) / sample_rate,
                inference_time=0.0,
            )

        # Lazy-load model on first use
        if self._model is None:
            self.load_model()

        # Run inference in thread pool to avoid blocking event loop
        loop = asyncio.get_running_loop()
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="stt",
            )

        logger.debug(
            "STT inference: {:.2f}s audio, language={}",
            len(audio) / sample_rate,
            self.language or "auto",
        )

        infer_start = time.monotonic()

        try:
            result = await loop.run_in_executor(
                self._executor,
                self._infer_sync,
                audio,
                sample_rate,
            )
        except Exception as e:
            logger.error("STT inference error: {}", e)
            return STTResult(
                text="",
                confidence=0.0,
                language=self.language or "unknown",
                language_probability=0.0,
                duration=len(audio) / sample_rate,
                inference_time=time.monotonic() - infer_start,
            )

        inference_time = time.monotonic() - infer_start

        logger.info(
            "STT: \"{}\" (conf={:.2f}, lang={}, {:.1f}s audio in {:.1f}s, RTF={:.2f})",
            result.text[:60],
            result.confidence,
            result.language,
            result.duration,
            inference_time,
            inference_time / result.duration if result.duration > 0 else 0,
        )

        return result

    def _infer_sync(self, audio: np.ndarray,
                    sample_rate: int) -> STTResult:
        """Synchronous inference (runs in thread pool).

        Args:
            audio: Audio array.
            sample_rate: Sample rate.

        Returns:
            STTResult.
        """
        duration = len(audio) / sample_rate

        segments, info = self._model.transcribe(
            audio=audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            hotwords=" ".join(self.hotwords) if self.hotwords else None,
            condition_on_previous_text=True,
        )

        # Collect all segments
        text_parts = []
        all_segments = []
        total_confidence = 0.0
        segment_count = 0

        for seg in segments:
            text_parts.append(seg.text)
            all_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "avg_logprob": getattr(seg, "avg_logprob", 0),
                "no_speech_prob": getattr(seg, "no_speech_prob", 0),
            })
            # Weight confidence by segment duration
            seg_duration = seg.end - seg.start
            if seg_duration > 0:
                # Convert avg_logprob to a 0-1 confidence-like score
                # Typical avg_logprob: -0.5 to 0.0 (higher is better)
                seg_confidence = max(0.0, min(1.0, 1.0 + seg.avg_logprob))
                total_confidence += seg_confidence * seg_duration
                segment_count += seg_duration

        text = " ".join(text_parts).strip()
        avg_confidence = (
            total_confidence / segment_count
            if segment_count > 0
            else 0.0
        )

        return STTResult(
            text=text,
            confidence=avg_confidence,
            language=info.language,
            language_probability=info.language_probability,
            duration=duration,
            inference_time=0.0,  # Set by caller
            segments=all_segments if all_segments else None,
        )

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[tuple[np.ndarray, int]],
    ) -> AsyncIterator[STTResult]:
        """Transcribe a stream of audio segments sequentially.

        Each VAD speech segment is transcribed independently,
        and results are yielded as they complete.

        Args:
            audio_chunks: Async iterator of (audio, sample_rate) tuples.

        Yields:
            STTResult for each audio chunk.

        Example:
            async for audio, sr in vad_speech_segments():
                result = await stt.transcribe(audio, sr)
                if result.text:
                    print(result.text)
        """
        async for audio, sample_rate in audio_chunks:
            result = await self.transcribe(audio, sample_rate)
            if result.text:  # Only yield non-empty results
                yield result

    # ── Cleanup ────────────────────────────────────────────────────

    async def cleanup(self) -> None:
        """Release resources (model, thread pool)."""
        self.unload_model()
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        logger.info("STT resources released")
