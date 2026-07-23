"""
Logging configuration for AIRI Voice Module.

Uses loguru for structured logging with file rotation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "DEBUG",
    fmt: str | None = None,
    log_file: str | None = None,
    rotation: str = "10 MB",
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        fmt: Log format string.
        log_file: Path to log file (optional).
        rotation: Log file rotation size.
    """
    # Remove default handler
    logger.remove()

    # Default format
    if fmt is None:
        fmt = (
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level:<7}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Console handler
    logger.add(
        sys.stderr,
        format=fmt,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format=fmt.replace("<", "").replace(">", ""),
            level=level,
            rotation=rotation,
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True,
        )

    logger.debug("Logging initialized (level={})", level)


def get_logger(name: str):
    """Get a named logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger bound with the given name.
    """
    return logger.bind(name=name)
