"""
Tests for the audio pipeline orchestration module.

Tests focus on pipeline initialization, component wiring,
and event routing. Full end-to-end tests require hardware.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config, load_config
from src.pipeline.audio_pipeline import AudioPipeline


class TestConfig:
    """Test configuration loading."""

    def test_default_config(self):
        """Test loading default configuration."""
        config = Config()
        assert config.audio.sample_rate == 48000
        assert config.audio.target_sample_rate == 16000
        assert config.vad.threshold == 0.5
        assert config.airi.host == "localhost"
        assert config.airi.port == 10443
        assert config.logging.level == "DEBUG"

    def test_config_from_yaml(self):
        """Test loading from YAML file."""
        config_path = Path("config/default.yaml")
        if config_path.exists():
            config = load_config(str(config_path))
            assert isinstance(config, Config)
            assert config.audio.sample_rate == 48000

    def test_env_overrides(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("AIRI_HOST", "192.168.1.100")
        monkeypatch.setenv("VAD_THRESHOLD", "0.3")
        monkeypatch.setenv("LOG_LEVEL", "INFO")

        config = load_config("config/default.yaml")
        assert config.airi.host == "192.168.1.100"
        assert config.vad.threshold == 0.3
        assert config.logging.level == "INFO"


class TestPipelineInit:
    """Test pipeline initialization."""

    def test_pipeline_creation(self):
        """Test pipeline object creation."""
        config = Config()
        pipeline = AudioPipeline(config)
        assert pipeline.config is config
        assert not pipeline.is_running
        assert pipeline.capture is not None
        assert pipeline.playback is not None
        assert pipeline.vad is not None
        assert pipeline.resampler is not None

    def test_speech_callback_registration(self):
        """Test speech event callback registration."""
        config = Config()
        pipeline = AudioPipeline(config)

        events = []
        pipeline.on_speech_event(lambda e: events.append(e))
        assert len(pipeline._speech_callbacks) == 1

    def test_audio_device_listing(self):
        """Test device listing (should not raise)."""
        config = Config()
        pipeline = AudioPipeline(config)
        inputs, outputs = pipeline.list_audio_devices()
        assert isinstance(inputs, list)
        assert isinstance(outputs, list)


class TestRingBuffer:
    """Test audio ring buffer."""

    def test_ring_buffer_creation(self):
        """Test ring buffer initialization."""
        from src.pipeline.ring_buffer import AudioRingBuffer
        buf = AudioRingBuffer(max_history=10)
        assert buf.qsize == 0
        assert buf.sample_rate == 0

    def test_raw_write_and_retrieve(self):
        """Test writing and retrieving raw frames."""
        from src.pipeline.ring_buffer import AudioRingBuffer
        import numpy as np
        buf = AudioRingBuffer(max_history=100)

        frame = np.random.randn(512).astype(np.float32)
        buf.write_raw(frame, 48000)

        history = buf.get_raw_history()
        assert len(history) == 1
        assert np.allclose(history[0], frame)
        assert buf.sample_rate == 48000

    def test_raw_concatenation(self):
        """Test concatenating multiple raw frames."""
        from src.pipeline.ring_buffer import AudioRingBuffer
        import numpy as np
        buf = AudioRingBuffer(max_history=100)

        for _ in range(5):
            buf.write_raw(np.random.randn(512).astype(np.float32), 48000)

        result = buf.get_raw_concatenated()
        assert len(result) == 5 * 512

    def test_clear(self):
        """Test clearing the buffer."""
        from src.pipeline.ring_buffer import AudioRingBuffer
        import numpy as np
        buf = AudioRingBuffer(max_history=100)

        buf.write_raw(np.random.randn(512).astype(np.float32), 48000)
        buf.clear()

        assert len(buf.get_raw_history()) == 0
