"""
Configuration loader for AIRI Voice Module.

Loads YAML configuration with environment variable overrides.
Environment variables take precedence over YAML values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    """Audio I/O configuration."""
    input_device: int | None = None
    sample_rate: int = 48000
    frames_per_buffer: int = 512
    output_device: int | None = None
    output_sample_rate: int = 24000
    target_sample_rate: int = 16000


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    model_path: str = "models/silero_vad.onnx"
    threshold: float = 0.5
    min_speech_duration: float = 0.25
    min_silence_duration: float = 0.5
    frame_size: int = 512


@dataclass
class AIRIConfig:
    """AIRI WebSocket connection configuration."""
    host: str = "localhost"
    port: int = 10443
    token: str = ""
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 0

    @property
    def url(self) -> str:
        """Get WebSocket URL."""
        return f"ws://{self.host}:{self.port}"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "DEBUG"
    format: str = "{time:HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} | {message}"
    file: str = "logs/voice-module.log"
    rotation: str = "10 MB"


@dataclass
class PipelineConfig:
    """Audio pipeline configuration."""
    speech_buffer_max_duration: float = 10.0


@dataclass
class Config:
    """Application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    airi: AIRIConfig = field(default_factory=AIRIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def __post_init__(self):
        # Convert dicts to dataclass instances if loaded from YAML
        if isinstance(self.audio, dict):
            self.audio = AudioConfig(**self.audio)
        if isinstance(self.vad, dict):
            self.vad = VADConfig(**self.vad)
        if isinstance(self.airi, dict):
            self.airi = AIRIConfig(**self.airi)
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
        if isinstance(self.pipeline, dict):
            self.pipeline = PipelineConfig(**self.pipeline)


# Environment variable mapping for config overrides
ENV_MAP: dict[str, str] = {
    "AIRI_HOST": "airi.host",
    "AIRI_PORT": "airi.port",
    "AIRI_TOKEN": "airi.token",
    "AUDIO_INPUT_DEVICE": "audio.input_device",
    "AUDIO_OUTPUT_DEVICE": "audio.output_device",
    "VAD_THRESHOLD": "vad.threshold",
    "LOG_LEVEL": "logging.level",
}


def _apply_env_overrides(cfg: dict) -> dict:
    """Apply environment variable overrides to config dict."""
    for env_var, config_path in ENV_MAP.items():
        value = os.environ.get(env_var)
        if value is None:
            continue

        parts = config_path.split(".")
        target = cfg
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Type cast
        key = parts[-1]
        existing = target.get(key)
        if isinstance(existing, bool):
            target[key] = value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            target[key] = int(value)
        elif isinstance(existing, float):
            target[key] = float(value)
        elif existing is None:
            # None defaults: try to infer numeric type from the value
            try:
                target[key] = int(value)
            except ValueError:
                try:
                    target[key] = float(value)
                except ValueError:
                    target[key] = value
        else:
            target[key] = value

    return cfg


def load_config(config_path: str | Path = "config/default.yaml") -> Config:
    """Load configuration from YAML file with env overrides.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Config dataclass instance.
    """
    config_path = Path(config_path)

    # Default config
    cfg: dict = {}

    # Load from file if exists
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        # Use built-in defaults
        cfg = _default_dict()

    # Apply environment variable overrides
    cfg = _apply_env_overrides(cfg)

    return Config(**cfg)


def _default_dict() -> dict:
    """Get default configuration as dict."""
    return {
        "audio": {
            "input_device": None,
            "sample_rate": 48000,
            "frames_per_buffer": 512,
            "output_device": None,
            "output_sample_rate": 24000,
            "target_sample_rate": 16000,
        },
        "vad": {
            "model_path": "models/silero_vad.onnx",
            "threshold": 0.5,
            "min_speech_duration": 0.25,
            "min_silence_duration": 0.5,
            "frame_size": 512,
        },
        "airi": {
            "host": "localhost",
            "port": 10443,
            "token": "",
            "reconnect_interval": 5,
            "max_reconnect_attempts": 0,
        },
        "logging": {
            "level": "DEBUG",
            "format": "{time:HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} | {message}",
            "file": "logs/voice-module.log",
            "rotation": "10 MB",
        },
        "pipeline": {
            "speech_buffer_max_duration": 10.0,
        },
    }
