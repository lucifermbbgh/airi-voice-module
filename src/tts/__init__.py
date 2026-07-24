"""
TTS (Text-to-Speech) module for AIRI Voice Module.

Phase 3: Converts text responses from AIRI into speech audio,
which is played through the speaker output.

Primary engine: CosyVoice 2 (high-quality Chinese TTS, fully offline).
Fallback: Edge-TTS (zero-install, online API).
"""

from __future__ import annotations

from src.tts.cosyvoice_tts import CosyVoiceTTS
from src.tts.tts_engine import TTSBase, TTSResult
from src.tts.tts_manager import TTSCache, TTSManager

__all__ = [
    "CosyVoiceTTS",
    "TTSBase",
    "TTSResult",
    "TTSCache",
    "TTSManager",
]
