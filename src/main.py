"""
AIRI Voice Module - Main entry point.

Usage:
    python -m src.main                  # Start voice pipeline
    python -m src.main --list-devices   # List audio devices
    python -m src.main --test-vad       # Test VAD only (no AIRI)
    python -m src.main --config path    # Custom config path
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from src.audio.capture import AudioCapture
from src.audio.playback import AudioPlayback
from src.config import load_config
from src.logger import get_logger, setup_logging
from src.pipeline.audio_pipeline import AudioPipeline
from src.vad.silero_vad import SpeechEventType

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="AIRI Voice Module - Real-time voice interaction backend",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input/output devices",
    )
    parser.add_argument(
        "--test-vad",
        action="store_true",
        help="Run in VAD test mode (capture → VAD → log, no AIRI connection)",
    )
    return parser.parse_args()


def _list_devices() -> None:
    """List all available audio devices."""
    print("\n=== Available Audio Input Devices ===")
    for dev in AudioCapture.list_devices():
        print(f"  [{dev['id']}] {dev['name']}")
        print(f"         Channels: {dev['channels']}, "
              f"Rate: {dev['default_samplerate']:.0f} Hz, "
              f"API: {dev['host_api']}")
        print()

    print("\n=== Available Audio Output Devices ===")
    for dev in AudioPlayback.list_devices():
        print(f"  [{dev['id']}] {dev['name']}")
        print(f"         Channels: {dev['channels']}, "
              f"Rate: {dev['default_samplerate']:.0f} Hz, "
              f"API: {dev['host_api']}")
        print()


def _speech_event_callback(event) -> None:
    """Callback for speech events in test mode.

    Args:
        event: SpeechEvent from VAD.
    """
    if event.type == SpeechEventType.SPEECH_START:
        print(f"\n🗣️  [SPEECH START] {event.timestamp:.3f}")
    elif event.type == SpeechEventType.SPEECH_END:
        print(f"🤫 [SPEECH END] dur={event.duration:.2f}s, "
              f"frames={event.num_frames}, max_prob={event.max_prob:.3f}")
        # In test mode, just log - don't send to STT yet


async def _run_test_vad(pipeline: AudioPipeline) -> None:
    """Run pipeline in VAD test mode.

    Captures audio, runs VAD, and prints speech events.
    Press Ctrl+C to stop.

    Args:
        pipeline: Configured AudioPipeline instance.
    """
    pipeline.on_speech_event(_speech_event_callback)

    print("\n🎤 VAD Test Mode - Listening... (Ctrl+C to stop)")
    print("=" * 60)

    try:
        await pipeline.start()
        # Keep running until interrupted
        while pipeline.is_running:
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    finally:
        await pipeline.stop()
        print("\nVAD test complete.")


async def _run_full(pipeline: AudioPipeline) -> None:
    """Run full voice pipeline with AIRI connection.

    Args:
        pipeline: Configured AudioPipeline instance.
    """
    # Register speech callback for AIRI integration (Phase 2+)
    pipeline.on_speech_event(_speech_event_callback)

    print("\n🎤 AIRI Voice Module - Full mode")
    print("=" * 60)

    try:
        await pipeline.start()
        # Keep running
        while pipeline.is_running:
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    finally:
        await pipeline.stop()
        print("\nVoice module stopped.")


async def _async_main(args: argparse.Namespace) -> None:
    """Async main entry point.

    Args:
        args: Parsed command line arguments.
    """
    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(
        level=config.logging.level,
        fmt=config.logging.format,
        log_file=config.logging.file,
        rotation=config.logging.rotation,
    )

    logger.info("AIRI Voice Module starting...")
    logger.info("Config: {}", args.config)

    if args.list_devices:
        _list_devices()
        return

    # Create pipeline
    pipeline = AudioPipeline(config)

    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)
    except NotImplementedError:
        # Windows: add_signal_handler is not supported.
        # Default Ctrl+C → KeyboardInterrupt → asyncio.run() handles cleanup.
        logger.info("Signal handlers not supported on this platform")
        logger.info("Use Ctrl+C to gracefully shut down")

    # Run selected mode
    try:
        if args.test_vad:
            task = asyncio.create_task(_run_test_vad(pipeline))
        else:
            task = asyncio.create_task(_run_full(pipeline))

        # Create a coroutine that waits for the shutdown event
        async def _wait_shutdown():
            await shutdown_event.wait()

        shutdown_task = asyncio.create_task(_wait_shutdown())

        # Wait for either shutdown signal OR pipeline task completion
        done, pending = await asyncio.wait(
            [task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel whichever is still pending
        for t in pending:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        # If pipeline task failed, log the exception
        if task.done() and task.exception():
            logger.error("Pipeline error: {}", task.exception())
    finally:
        await pipeline.stop()
        logger.info("AIRI Voice Module stopped.")


def main() -> None:
    """Main entry point."""
    args = _parse_args()
    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
