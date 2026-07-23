"""
STT (Speech-to-Text) module for AIRI Voice Module.

Phase 2: Converts speech audio segments from VAD into text,
which is then sent to AIRI via WebSocket as input:text:voice events.

Primary engine: Faster-Whisper (int8 quantized, CPU-friendly).
Fallback: SenseVoice (if Chinese accuracy requirements exceed Whisper).
"""

from __future__ import annotations

from src.stt.faster_whisper_stt import FasterWhisperSTT, STTResult

__all__ = [
    "FasterWhisperSTT",
    "STTResult",
]
