"""
TTS (Text-to-Speech) module for AIRI Voice Module.

Phase 3: Converts text responses from AIRI into speech audio,
which is played through the speaker output.

Primary engine: CosyVoice 2 (high-quality Chinese TTS, fully offline).
Fallback: Edge-TTS (zero-install, online API).
"""

from __future__ import annotations

from src.tts.tts_engine import TTSBase, TTSResult

__all__ = [
    "TTSBase",
    "TTSResult",
]
