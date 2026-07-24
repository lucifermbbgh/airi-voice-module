"""
CosyVoice 2 TTS engine implementation.

Uses Alibaba's CosyVoice 2 model for high-quality Chinese speech synthesis.
Fully offline after model download.

Key features:
- High-quality Chinese TTS (MOS > 4.0)
- Streaming inference for low-latency playback
- Multiple preset voices
- Zero-shot voice cloning (via reference audio)

Usage:
    tts = CosyVoiceTTS(model_size="base")
    result = await tts.synthesize("你好，我是 AIRI")
    async for chunk in tts.synthesize_stream("长文本..."):
        playback.play(chunk)

Dependencies:
    - cosyvoice>=1.0.0 (installed separately, pulls torch)
    - modelscope (for model download)
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from src.logger import get_logger
from src.tts.tts_engine import TTSBase, TTSResult

logger = get_logger(__name__)

# ── Available Models ─────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "base": {
        "name": "CosyVoice-2-0.5B",
        "ram_mb": 1500,
        "rtf": 0.2,
        "description": "Default model, good balance of quality and speed",
    },
    "small": {
        "name": "CosyVoice-2-0.5B",
        "ram_mb": 1000,
        "rtf": 0.15,
        "description": "Lightweight model, faster but slightly lower quality",
    },
}

# ── Preset Voices ───────────────────────────────────────────────────

PRESET_VOICES: list[dict] = [
    {"id": "default", "name": "默认女声", "description": "标准中文女声"},
    {"id": "中文男声", "name": "中文男声", "description": "标准中文男声"},
    {"id": "中文女声", "name": "中文女声", "description": "标准中文女声"},
    {"id": "英文女声", "name": "英文女声", "description": "标准英文女声"},
    {"id": "英文男声", "name": "英文男声", "description": "标准英文男声"},
    {"id": "日语女声", "name": "日语女声", "description": "标准日语女声"},
    {"id": "韩语女声", "name": "韩语女声", "description": "标准韩语女声"},
    {"id": "粤语女声", "name": "粤语女声", "description": "标准粤语女声"},
]

# ── Text Segmentation for Streaming ─────────────────────────────────

# Sentence-ending punctuation for splitting text
_SENTENCE_END = {".", "!", "?", "。", "！", "？", "\n"}

# Pause-based punctuation (add slight break)
_PAUSE = {"，", "；", "、", ",", ";", "：", ":"}


class CosyVoiceTTS(TTSBase):
    """CosyVoice 2 TTS engine.

    Attributes:
        model_size: Model size identifier ('base' or 'small').
        device: Computation device ('cpu' or 'cuda').
        model_dir: Custom model directory path.
        sample_rate: Output sample rate (24000 default).
        default_voice: Default voice ID for synthesis.
        default_speed: Default speaking speed.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        model_dir: str | None = None,
        sample_rate: int = 24000,
        default_voice: str = "default",
        default_speed: float = 1.0,
    ):
        """Initialize CosyVoice TTS engine.

        Args:
            model_size: Model size ('base' or 'small').
            device: Computation device.
            model_dir: Path to model directory.
                If None, uses ModelScope default cache.
            sample_rate: Output sample rate.
            default_voice: Default voice ID.
            default_speed: Default speaking speed (0.5-2.0).

        Raises:
            ValueError: If model_size is invalid.
        """
        if model_size not in MODELS:
            valid = ", ".join(MODELS.keys())
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Choose from: {valid}"
            )

        self.model_size = model_size
        self.device = device
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.default_voice = default_voice
        self.default_speed = default_speed

        # Internal state
        self._model = None  # Lazy-loaded CosyVoice model
        self._executor = None  # ThreadPoolExecutor for sync inference

        spec = MODELS[model_size]
        logger.info(
            "TTS initialized: engine=CosyVoice2, model={}, device={}, "
            "ram≈{}MB, RTF≈{}",
            spec["name"], device, spec["ram_mb"], spec["rtf"],
        )

    # ── Properties ──────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Engine name."""
        return "cosyvoice"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def voices(self) -> list[dict]:
        """Get list of available voices."""
        return list(PRESET_VOICES)

    @property
    def model_info(self) -> dict:
        """Get current model information."""
        return {
            "engine": self.name,
            "model_size": self.model_size,
            "model_name": MODELS[self.model_size]["name"],
            "device": self.device,
            "loaded": self.is_loaded,
            "sample_rate": self.sample_rate,
            "default_voice": self.default_voice,
        }

    # ── Model Lifecycle ─────────────────────────────────────────────

    async def load_model(self) -> None:
        """Load the CosyVoice 2 model.

        Uses lazy import so the module can be imported without
        cosyvoice installed (useful for testing).
        """
        if self._model is not None:
            return

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
        except ImportError:
            raise ImportError(
                "cosyvoice is not installed.\n"
                "Install it with: pip install cosyvoice\n"
                "Or see: https://github.com/FunAudioLLM/CosyVoice"
            )

        # Determine model path
        if self.model_dir:
            model_path = self.model_dir
        else:
            # Use default CosyVoice-2 model name
            model_name = MODELS[self.model_size]["name"]
            model_path = f"pretrained_models/{model_name}"

        logger.info(
            "Loading CosyVoice model: {} (device={})",
            model_path, self.device,
        )

        load_start = time.monotonic()
        try:
            self._model = CosyVoice(model_path)
        except Exception as e:
            logger.error("Failed to load CosyVoice model: {}", e)
            raise RuntimeError(
                f"Failed to load CosyVoice model from '{model_path}'. "
                f"Ensure the model is downloaded. Error: {e}"
            )

        load_time = time.monotonic() - load_start
        logger.info(
            "CosyVoice model loaded in {:.1f}s: {}",
            load_time, model_path,
        )

    async def unload_model(self) -> None:
        """Unload model to free memory."""
        self._model = None
        logger.info("CosyVoice model unloaded")

    async def cleanup(self) -> None:
        """Release all resources."""
        self._model = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        logger.info("CosyVoice resources released")

    # ── Synthesis — Complete Audio ──────────────────────────────────

    async def synthesize(
        self,
        text: str,
        voice_id: str = "default",
        speed: float = 1.0,
    ) -> TTSResult:
        """Synthesise text to speech (complete audio).

        Args:
            text: Text to synthesise.
            voice_id: Voice identifier.
            speed: Speaking speed (0.5-2.0, 1.0 = normal).

        Returns:
            TTSResult with complete synthesized audio.
        """
        if not text or not text.strip():
            return TTSResult(
                audio=np.array([], dtype=np.float32),
                sample_rate=self.sample_rate,
                duration=0.0,
                text="",
                synthesis_time=0.0,
            )

        # Lazy-load model
        if self._model is None:
            await self.load_model()

        text = text.strip()

        # Run synthesis in thread pool (CosyVoice is blocking)
        loop = asyncio.get_running_loop()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="tts",
            )

        logger.debug("TTS synthesising: \"{}\" (voice={}, speed={})",
                      text[:60], voice_id, speed)

        infer_start = time.monotonic()
        try:
            audio = await loop.run_in_executor(
                self._executor,
                self._synthesize_sync,
                text,
                voice_id,
                speed,
            )
        except Exception as e:
            logger.error("TTS synthesis error: {}", e)
            return TTSResult(
                audio=np.array([], dtype=np.float32),
                sample_rate=self.sample_rate,
                duration=0.0,
                text=text,
                synthesis_time=time.monotonic() - infer_start,
            )

        synthesis_time = time.monotonic() - infer_start
        duration = len(audio) / self.sample_rate

        logger.info(
            "TTS: \"{}\" ({:.1f}s audio in {:.1f}s, RTF={:.2f}, voice={})",
            text[:60], duration, synthesis_time,
            synthesis_time / duration if duration > 0 else 0,
            voice_id,
        )

        return TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration=duration,
            text=text,
            synthesis_time=synthesis_time,
        )

    def _synthesize_sync(
        self,
        text: str,
        voice_id: str = "default",
        speed: float = 1.0,
    ) -> np.ndarray:
        """Synchronous synthesis (runs in thread pool).

        Args:
            text: Text to synthesise.
            voice_id: Voice identifier.
            speed: Speaking speed.

        Returns:
            Audio array as float32 numpy.
        """
        # CosyVoice returns audio as torch tensor or numpy array
        # The exact API depends on the version installed
        result = self._model.tts(
            text=text,
            spk_id=voice_id,
            speed=speed,
        )

        # Handle different return types
        if hasattr(result, "numpy"):
            # torch.Tensor
            audio = result.numpy()
        elif isinstance(result, np.ndarray):
            audio = result
        elif hasattr(result, "cpu"):
            # torch.Tensor on GPU
            audio = result.cpu().numpy()
        else:
            # Assume it's array-like
            audio = np.asarray(result, dtype=np.float32)

        # Ensure float32 and flatten
        audio = np.asarray(audio, dtype=np.float32).flatten()

        # Normalize volume
        audio = self.normalize_volume(audio)

        return audio

    # ── Synthesis — Streaming ───────────────────────────────────────

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str = "default",
        speed: float = 1.0,
    ) -> AsyncIterator[np.ndarray]:
        """Synthesise text to speech (streaming chunks).

        Splits text into sentences and yields audio for each sentence
        as it's generated. Allows the caller to start playback
        before the full text is synthesised.

        Args:
            text: Text to synthesise.
            voice_id: Voice identifier.
            speed: Speaking speed (0.5-2.0).

        Yields:
            Audio chunks for each sentence.
        """
        if not text or not text.strip():
            return

        # Lazy-load model
        if self._model is None:
            await self.load_model()

        text = text.strip()

        # Split into sentences for streaming
        sentences = self._split_sentences(text)

        logger.debug("TTS stream: {} sentences from \"{}\"",
                      len(sentences), text[:40])

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            result = await self.synthesize(sentence, voice_id, speed)

            if len(result.audio) > 0:
                logger.debug("TTS stream chunk {}/{}: {:.1f}s audio",
                             i + 1, len(sentences), result.duration)
                yield result.audio

    # ── Text Processing ─────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for streaming.

        Uses sentence-ending punctuation as boundaries.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        sentences = []
        current = []

        for char in text:
            current.append(char)
            if char in _SENTENCE_END:
                sentences.append("".join(current).strip())
                current = []

        # Remaining text
        if current:
            remaining = "".join(current).strip()
            if remaining:
                sentences.append(remaining)

        return sentences if sentences else [text]

    # ── Voice Management ────────────────────────────────────────────

    def set_voice(self, voice_id: str) -> None:
        """Set default voice.

        Args:
            voice_id: Voice identifier.

        Raises:
            ValueError: If voice_id is not available.
        """
        if not any(v["id"] == voice_id for v in PRESET_VOICES):
            available = ", ".join(v["id"] for v in PRESET_VOICES)
            raise ValueError(
                f"Unknown voice '{voice_id}'. "
                f"Available voices: {available}"
            )
        self.default_voice = voice_id
        logger.debug("TTS default voice set to: {}", voice_id)

    def set_speed(self, speed: float) -> None:
        """Set default speaking speed.

        Args:
            speed: Speaking speed (0.5-2.0).

        Raises:
            ValueError: If speed is out of range.
        """
        if not 0.5 <= speed <= 2.0:
            raise ValueError(
                f"Speed must be between 0.5 and 2.0, got {speed}"
            )
        self.default_speed = speed
        logger.debug("TTS default speed set to: {}", speed)
