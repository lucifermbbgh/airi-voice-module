"""
Audio resampling module.

Converts between sample rates using scipy.signal.resample_poly
for high-quality polyphase resampling with minimal latency.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


class Resampler:
    """Audio resampler with polyphase filtering.

    Converts audio between source and target sample rates.
    Optimized for the common case of 48kHz → 16kHz downsampling.

    Attributes:
        source_rate: Source sample rate in Hz.
        target_rate: Target sample rate in Hz.
        ratio: Resampling ratio (target / source).
    """

    def __init__(self, source_rate: int = 48000, target_rate: int = 16000):
        """Initialize resampler.

        Args:
            source_rate: Source sample rate in Hz.
            target_rate: Target sample rate in Hz.
        """
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("Sample rates must be positive")

        self.source_rate = source_rate
        self.target_rate = target_rate
        self.ratio = target_rate / source_rate

    def resample(self, data: np.ndarray) -> np.ndarray:
        """Resample audio data to target sample rate.

        Args:
            data: Input audio array of shape (N,) or (N, channels).
                  Values should be float32 in range [-1.0, 1.0].

        Returns:
            Resampled audio array at target sample rate.
        """
        if len(data) == 0:
            return np.array([], dtype=np.float32)

        if self.source_rate == self.target_rate:
            return data.astype(np.float32, copy=False)

        # Preserve dtype for processing
        orig_dtype = data.dtype
        data_f = data.astype(np.float64, copy=False)

        # Handle multi-channel: resample each channel independently
        if data_f.ndim == 1:
            resampled = signal.resample_poly(
                data_f,
                up=self.target_rate,
                down=self.source_rate,
                padtype="constant",
            )
        else:
            channels = []
            for ch in range(data_f.shape[1]):
                ch_data = signal.resample_poly(
                    data_f[:, ch],
                    up=self.target_rate,
                    down=self.source_rate,
                    padtype="constant",
                )
                channels.append(ch_data)
            resampled = np.column_stack(channels)

        return resampled.astype(np.float32)

    def resample_buffer(
        self,
        data: np.ndarray,
        source_rate: int | None = None,
    ) -> np.ndarray:
        """Resample with optional per-call source rate override.

        Args:
            data: Input audio array.
            source_rate: Override source rate for this call (optional).

        Returns:
            Resampled audio array.
        """
        if source_rate is not None and source_rate != self.source_rate:
            # Create temporary resampler with overridden source rate
            return Resampler(source_rate, self.target_rate).resample(data)
        return self.resample(data)

    @staticmethod
    def to_float32(data: np.ndarray) -> np.ndarray:
        """Convert int16 PCM data to float32 [-1.0, 1.0].

        Args:
            data: int16 PCM audio array.

        Returns:
            float32 audio array.
        """
        if data.dtype == np.float32:
            return data
        if data.dtype == np.int16:
            return (data / 32768.0).astype(np.float32)
        if data.dtype == np.int32:
            return (data / 2147483648.0).astype(np.float32)
        return data.astype(np.float32)

    @staticmethod
    def to_int16(data: np.ndarray) -> np.ndarray:
        """Convert float32 audio to int16 PCM.

        Args:
            data: float32 audio array in range [-1.0, 1.0].

        Returns:
            int16 PCM audio array.
        """
        if data.dtype == np.int16:
            return data
        # Clip to prevent overflow
        clipped = np.clip(data, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)
