"""
Tests for the audio capture module.

Tests focus on the resampler and ring buffer since actual
microphone capture requires hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.resampler import Resampler


class TestResampler:
    """Test audio resampler functionality."""

    def test_init(self):
        """Test resampler initialization."""
        r = Resampler(48000, 16000)
        assert r.source_rate == 48000
        assert r.target_rate == 16000
        assert r.ratio == 1 / 3

    def test_downsample(self):
        """Test downsampling 48kHz to 16kHz."""
        r = Resampler(48000, 16000)
        # 1 second of audio at 48kHz
        data = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000).astype(np.float32)
        result = r.resample(data)
        assert len(result) == pytest.approx(16000, rel=0.01)
        assert result.dtype == np.float32

    def test_upsample(self):
        """Test upsampling 16kHz to 48kHz."""
        r = Resampler(16000, 48000)
        data = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        result = r.resample(data)
        assert len(result) == pytest.approx(48000, rel=0.01)
        assert result.dtype == np.float32

    def test_same_rate(self):
        """Test no-op when rates match."""
        r = Resampler(16000, 16000)
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = r.resample(data)
        assert len(result) == 3
        assert np.allclose(result, data)

    def test_empty_data(self):
        """Test empty input."""
        r = Resampler(48000, 16000)
        result = r.resample(np.array([], dtype=np.float32))
        assert len(result) == 0

    def test_to_float32_from_int16(self):
        """Test int16 to float32 conversion."""
        data = np.array([0, 16384, 32767, -32768], dtype=np.int16)
        result = Resampler.to_float32(data)
        assert result.dtype == np.float32
        assert np.allclose(result, [0.0, 0.5, 1.0, -1.0], atol=1e-4)

    def test_to_int16_from_float32(self):
        """Test float32 to int16 conversion."""
        data = np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float32)
        result = Resampler.to_int16(data)
        assert result.dtype == np.int16
        assert np.allclose(result, [0, 16384, 32767, -32768], atol=1)

    def test_invalid_rate(self):
        """Test invalid sample rate."""
        with pytest.raises(ValueError):
            Resampler(0, 16000)

    def test_multichannel(self):
        """Test resampling multi-channel audio."""
        r = Resampler(48000, 16000)
        data = np.random.randn(4800, 2).astype(np.float32)
        result = r.resample(data)
        assert result.shape[1] == 2
        assert len(result) == pytest.approx(1600, rel=0.01)


class TestAudioCapture:
    """Test AudioCapture static methods."""

    def test_list_devices(self):
        """Test device listing (should not raise)."""
        try:
            from src.audio.capture import AudioCapture
        except OSError as e:
            if "PortAudio" in str(e):
                pytest.skip("PortAudio library not available")
            raise
        # Just verify it runs and returns a list
        devices = AudioCapture.list_devices()
        assert isinstance(devices, list)
        # May be empty in CI, but should have valid structure
        for dev in devices:
            assert "id" in dev
            assert "name" in dev
            assert "channels" in dev
