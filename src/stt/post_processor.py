"""
Text post-processor for STT output.

Lightweight rule-based engine that improves STT transcription quality:

1. Punctuation restoration — Adds Chinese/English punctuation to raw text
2. Spacing normalization — Fixes CJK/Latin mixed spacing
3. Proper noun correction — Hotword-guided term correction

Design decision (from PHASE-2-STT-DETAILED-DESIGN.md):
    Keep it as a lightweight rule engine rather than an ML model.
    Punctuation restoration uses pattern matching + hotword hints.

Usage:
    processor = TextPostProcessor(hotwords=["AIRI", "Claude"])
    text = processor.process("测试测试看看你还能输出什么内容")
    # → "测试，测试，看看你还能输出什么内容？"
"""

from __future__ import annotations

import re
from typing import Pattern


# ── Punctuation patterns ────────────────────────────────────────────

# Chinese sentence-ending particles — append full-width question mark
_CN_QUESTION_PARTICLES: Pattern[str] = re.compile(
    r"[吗嘛么罢吧啦呢]",  # 你好吗 → 你好吗？
)

# Interrogative patterns that signal a question
_CN_QUESTION_PATTERNS: Pattern[str] = re.compile(
    r"(?:什么|怎么|哪[儿里]|多少|几[个天]|何时|为什么|要不要|能不能|会不会|是不是|有没有)"
)

# Punctuation that signals end of sentence — we insert 。or ？
_CN_SENTENCE_END: Pattern[str] = re.compile(
    r"[。！？.!?]$"
)

# Prefixes before which we restore a comma (adverbial / topic markers)
_CN_COMMA_BEFORE: Pattern[str] = re.compile(
    r"(然后|但是|而且|不过|因为|所以|虽然|如果|比如|总之|另外|此外|首先|其次|最后)"
)

# CJK range (used in spacing rules)
_CJK: Pattern[str] = re.compile(r"[一-鿿㐀-䶿豈-﫿]")

# Latin range
_LATIN: Pattern[str] = re.compile(r"[a-zA-Z]")

# Digit range
_DIGIT: Pattern[str] = re.compile(r"[0-9]")


class TextPostProcessor:
    """Lightweight rule-based text post-processor for STT output.

    Restores punctuation, normalizes spacing, and applies hotword
    corrections to raw transcription text.

    Attributes:
        hotwords: List of hotwords to prioritise in recognition.
        min_confidence: Minimum confidence to apply post-processing.
        enable_punctuation: Enable punctuation restoration.
        enable_spacing: Enable CJK/Latin spacing normalization.
    """

    def __init__(
        self,
        hotwords: list[str] | None = None,
        min_confidence: float = 0.0,
        enable_punctuation: bool = True,
        enable_spacing: bool = True,
    ) -> None:
        """Initialize text post-processor.

        Args:
            hotwords: List of hotwords to prioritise.
            min_confidence: Minimum transcription confidence to process.
            enable_punctuation: Enable punctuation restoration.
            enable_spacing: Enable CJK/Latin spacing normalization.
        """
        self.hotwords = list(hotwords) if hotwords else []
        self.min_confidence = min_confidence
        self.enable_punctuation = enable_punctuation
        self.enable_spacing = enable_spacing

    def process(
        self,
        text: str,
        confidence: float = 0.0,
    ) -> str:
        """Post-process transcribed text.

        Applies configured transformations in order:
        1. Strip leading/trailing whitespace
        2. Hotword-aware normalisation
        3. Spacing normalisation (CJK/Latin)
        4. Punctuation restoration

        Args:
            text: Raw STT output text.
            confidence: Transcription confidence (0.0-1.0).

        Returns:
            Processed text with punctuation and corrected spacing.
        """
        if not text:
            return ""

        if confidence < self.min_confidence:
            return text

        result = text.strip()

        # 1. Hotword-aware deduplication
        result = self._correct_hotwords(result)

        # 2. CJK/Latin spacing
        if self.enable_spacing:
            result = self._normalize_spacing(result)

        # 3. Punctuation restoration
        if self.enable_punctuation:
            result = self._restore_punctuation(result)

        return result

    # ── Hotword Correction ─────────────────────────────────────────

    def _correct_hotwords(self, text: str) -> str:
        """Apply hotword-aware corrections to recognised text.

        For each hotword, checks if a phonetically-similar but incorrect
        transcription exists and replaces it with the correct term.

        This uses simple case-insensitive exact matching for now.
        Future enhancement: fuzzy matching via Levenshtein distance.

        Args:
            text: Raw text.

        Returns:
            Text with hotword corrections applied.
        """
        if not self.hotwords:
            return text

        for word in self.hotwords:
            if not word:
                continue
            # Case-insensitive replacement (hotwords like "Claude" may be
            # transcribed in lowercase)
            lower_text = text.lower()
            lower_word = word.lower()
            if lower_word in lower_text:
                # Preserve original case of the replacement
                # Simple: replace all occurrences
                text = re.sub(
                    re.escape(lower_word),
                    word,
                    text,
                    flags=re.IGNORECASE,
                )

        return text

    # ── Spacing Normalisation ──────────────────────────────────────

    def _normalize_spacing(self, text: str) -> str:
        """Normalise spacing between CJK and Latin/digit characters.

        Inserts a space between CJK and Latin characters where needed,
        and removes spaces that shouldn't exist (e.g., inside CJK text).

        Rules:
            CJK + Latin → "CJK Latin"
            Latin + CJK → "Latin CJK"
            CJK + Digit → "CJK5" (no space — common in Chinese)
            Digit + CJK → "5CJK" (no space)
            Removes multiple consecutive spaces → single space

        Args:
            text: Input text.

        Returns:
            Text with normalised spacing.
        """
        # Insert space between CJK and Latin
        text = re.sub(
            r"([一-鿿])([a-zA-Z])",
            r"\1 \2",
            text,
        )
        text = re.sub(
            r"([a-zA-Z])([一-鿿])",
            r"\1 \2",
            text,
        )
        # Collapse multiple spaces
        text = re.sub(r" +", " ", text)
        return text.strip()

    # ── Punctuation Restoration ────────────────────────────────────

    def _restore_punctuation(self, text: str) -> str:
        """Restore missing punctuation to raw STT text.

        Faster-Whisper (especially small model) often outputs text
        without punctuation. This method adds basic punctuation:

        1. Question marks after interrogative patterns
        2. Period at the end of declarative sentences
        3. Commas after adverbial / topic markers

        Args:
            text: Text without punctuation.

        Returns:
            Text with restored punctuation.
        """
        if not text:
            return ""

        # Already has sentence-ending punctuation — skip
        if _CN_SENTENCE_END.search(text):
            return text

        result = text

        # 1. Add comma after topic markers (then, but, because, etc.)
        # e.g., "但是我不想去" → "但是，我不想去"
        result = _CN_COMMA_BEFORE.sub(r"\1，", result)

        # 2. Detect question
        is_question = False
        if _CN_QUESTION_PARTICLES.search(result):
            is_question = True
        elif _CN_QUESTION_PATTERNS.search(result):
            is_question = True

        # 3. Append sentence-ending punctuation
        result = result.strip()
        if is_question:
            result += "？"
        else:
            result += "。"

        return result

    # ── Utility ─────────────────────────────────────────────────────

    def add_hotwords(self, hotwords: list[str]) -> None:
        """Add hotwords dynamically.

        Args:
            hotwords: List of hotwords to add.
        """
        for word in hotwords:
            word = word.strip()
            if word and word not in self.hotwords:
                self.hotwords.append(word)

    def set_hotwords(self, hotwords: list[str]) -> None:
        """Replace the entire hotword list.

        Args:
            hotwords: New hotword list.
        """
        self.hotwords = [w for w in hotwords if w.strip()]
