"""
AIRI Voice Module - Main entry point.

Usage:
    python -m src.main                       # Full mode (VAD → STT → AIRI)
    python -m src.main --list-devices        # List audio devices
    python -m src.main --test-vad            # Test VAD only (no STT, no AIRI)
    python -m src.main --test-stt            # Test VAD → STT (no AIRI)
    python -m src.main --test-tts            # Test TTS (type text → hear speech)
    python -m src.main --test-tts-no-play    # TTS → WAV file (no playback)
    python -m src.main --config path         # Custom config path
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

import numpy as np

from src.audio.capture import AudioCapture
from src.audio.playback import AudioPlayback
from src.config import load_config
from src.logger import get_logger, setup_logging
from src.pipeline.audio_pipeline import AudioPipeline
from src.stt import FasterWhisperSTT, TextPostProcessor
from src.tts import CosyVoiceTTS, TTSManager
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
    parser.add_argument(
        "--test-stt",
        action="store_true",
        help="Run in STT test mode (capture → VAD → STT → log transcriptions)",
    )
    parser.add_argument(
        "--test-tts",
        action="store_true",
        help="Run in TTS test mode (type text → hear speech output)",
    )
    parser.add_argument(
        "--test-tts-no-play",
        type=str,
        nargs="?",
        const="output/tts_test.wav",
        metavar="OUTPUT_WAV",
        help="Run TTS synthesis only (no playback), save to WAV file. "
             "Default: output/tts_test.wav. Useful for headless verification.",
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


async def _run_test_stt(
    pipeline: AudioPipeline,
    stt: FasterWhisperSTT,
    post_processor: TextPostProcessor | None = None,
) -> None:
    """Run pipeline in STT test mode.

    Captures audio, runs VAD → STT for each speech segment,
    and prints transcriptions. No AIRI connection needed.

    Args:
        pipeline: Configured AudioPipeline instance.
        stt: Configured STT engine instance.
        post_processor: Optional text post-processor.
    """
    async def on_speech(event) -> None:
        """Speech event → STT callback handler.

        Args:
            event: SpeechEvent from VAD.
        """
        if event.type == SpeechEventType.SPEECH_START:
            print(f"\n🗣️  [SPEECH START] {event.timestamp:.3f}")
        elif event.type == SpeechEventType.SPEECH_END:
            print(f"🤫 [SPEECH END] dur={event.duration:.2f}s, "
                  f"frames={event.num_frames}, max_prob={event.max_prob:.3f}")

            # Run STT on the speech segment
            if event.audio is not None:
                print(f"   📝 Transcribing {len(event.audio)} samples...")
                result = await stt.transcribe(event.audio, event.sample_rate)

                if result.text:
                    # Apply post-processing
                    display_text = result.text
                    if post_processor:
                        display_text = post_processor.process(
                            result.text, confidence=result.confidence,
                        )

                    print(f"   ✅ [{result.language}] "
                          f"\"{display_text}\" "
                          f"(conf={result.confidence:.2f}, "
                          f"{result.inference_time:.2f}s)")
                else:
                    print(f"   ⏭️  Empty result (silence or low confidence)")

    pipeline.on_speech_event(on_speech)

    print("\n🎤 STT Test Mode - Voice → Text... (Ctrl+C to stop)")
    print("=" * 60)

    try:
        await pipeline.start()
        while pipeline.is_running:
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    finally:
        await pipeline.stop()
        print("\nSTT test complete.")


async def _run_test_tts(
    tts_config,
    playback: AudioPlayback | None = None,
) -> None:
    """Run TTS interactive test mode.

    Users type text at the prompt and hear the synthesised speech.
    Type 'exit' or Ctrl+C to quit.

    Args:
        tts_config: TTSConfig instance.
        playback: Optional pre-existing AudioPlayback (creates one if None).
    """
    own_playback = playback is None
    pb = playback or AudioPlayback(
        device_id=None,
        sample_rate=tts_config.sample_rate,
    )

    try:
        await pb.start()
        print("\n🔊 TTS Test Mode - Interactive Text-to-Speech")
        print("=" * 60)
        print(f"   Engine: {tts_config.engine} ({tts_config.model_size})")
        print(f"   Voice: {tts_config.voice_id}, Speed: {tts_config.speed}")
        print(f"   Device: {tts_config.device}")
        print(f"   Streaming: {tts_config.streaming}")
        print("=" * 60)
        print("   Type text and press Enter to hear it spoken.")
        print("   Type 'exit' or 'quit' to stop.\n")

        # Initialize TTS engine
        if tts_config.engine == "cosyvoice":
            engine = CosyVoiceTTS(
                model_size=tts_config.model_size,
                device=tts_config.device,
                model_dir=tts_config.model_dir,
                sample_rate=tts_config.sample_rate,
                default_voice=tts_config.voice_id,
                default_speed=tts_config.speed,
            )
        else:
            print(f"❌ Unsupported engine: {tts_config.engine}")
            print("   Currently supported: cosyvoice")
            return

        tts_mgr = TTSManager(engine=engine, playback=pb)

        # Interactive loop
        while True:
            try:
                text = input("📝 TTS > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                break

            if not text:
                continue
            if text.lower() in ("exit", "quit", "q"):
                break

            print(f"   🔊 Synthesising {len(text)} chars...")
            if tts_config.streaming and len(text) > 20:
                result = await tts_mgr.say_stream(text)
            else:
                result = await tts_mgr.say(text)
            print(f"   ✅ Done (played={result})")

    except ImportError as e:
        print(f"\n❌ Engine import error: {e}")
        print("   Install the required package:\n"
              f"     pip install {tts_config.engine}")
    except Exception as e:
        print(f"\n❌ TTS error: {e}")
        logger.error("TTS test error: {}", e)
    finally:
        if own_playback:
            await pb.stop()


async def _run_test_tts_no_play(
    tts_config,
    output_path: str = "output/tts_test.wav",
) -> None:
    """Run TTS synthesis test without playback (save to WAV).

    Useful for headless verification (e.g. Windows without speaker).

    Args:
        tts_config: TTSConfig instance.
        output_path: Path to save the WAV file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("\n🔊 TTS No-Playback Test Mode - Text → WAV File")
    print("=" * 60)
    print(f"   Engine: {tts_config.engine} ({tts_config.model_size})")
    print(f"   Voice: {tts_config.voice_id}, Speed: {tts_config.speed}")
    print(f"   Output: {output_file}")
    print("=" * 60)
    print("   Type text and press Enter to synthesise.")
    print("   Type 'exit' or 'quit' to stop.\n")

    # Initialize TTS engine
    try:
        if tts_config.engine == "cosyvoice":
            engine = CosyVoiceTTS(
                model_size=tts_config.model_size,
                device=tts_config.device,
                model_dir=tts_config.model_dir,
                sample_rate=tts_config.sample_rate,
                default_voice=tts_config.voice_id,
                default_speed=tts_config.speed,
            )
        else:
            print(f"❌ Unsupported engine: {tts_config.engine}")
            return

        test_texts = [
            "你好，我是 AIRI，你的智能语音助手。",
            "今天天气真不错，适合出去散步。",
            "欢迎使用语音对话模块，Text-to-Speech 功能已就绪。",
        ]

        for i, text in enumerate(test_texts):
            print(f"\n   [{i + 1}/{len(test_texts)}] Synthesising {len(text)} chars...")
            print(f"       Text: \"{text}\"")
            result = await engine.synthesize(
                text,
                voice_id=tts_config.voice_id,
                speed=tts_config.speed,
            )

            if len(result.audio) > 0:
                wav_path = output_file.parent / f"tts_test_{i + 1}.wav"
                from scipy.io import wavfile
                wavfile.write(
                    str(wav_path),
                    result.sample_rate,
                    (result.audio * 32767).astype(np.int16),
                )
                print(f"       ✅ Saved: {wav_path}")
                print(f"          Duration: {result.duration:.2f}s, "
                      f"RTF: {result.synthesis_time / result.duration:.2f}"
                      if result.duration > 0 else "")
            else:
                print(f"       ⚠️  Empty audio (synthesis_time={result.synthesis_time:.2f}s)")

        # Interactive mode after preset tests
        print("\n" + "=" * 60)
        print("   Preset tests done. Enter custom text ('exit' to quit):")
        while True:
            try:
                text = input("📝 TTS > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                break

            if not text:
                continue
            if text.lower() in ("exit", "quit", "q"):
                break

            result = await engine.synthesize(
                text, voice_id=tts_config.voice_id, speed=tts_config.speed,
            )
            if len(result.audio) > 0:
                wav_path = output_file.parent / f"tts_custom.wav"
                from scipy.io import wavfile
                wavfile.write(
                    str(wav_path),
                    result.sample_rate,
                    (result.audio * 32767).astype(np.int16),
                )
                print(f"   ✅ Saved: {wav_path} ({result.duration:.2f}s)")
            else:
                print(f"   ⚠️  Empty audio")

    except ImportError as e:
        print(f"\n❌ Engine import error: {e}")
    except Exception as e:
        print(f"\n❌ TTS error: {e}")
        logger.error("TTS no-playback test error: {}", e)
    finally:
        await engine.cleanup()
        print(f"\n✅ All WAV files saved to: {output_file.parent}/")


async def _run_full(
    pipeline: AudioPipeline,
    stt: FasterWhisperSTT,
    post_processor: TextPostProcessor | None = None,
) -> None:
    """Run full voice pipeline with STT, AIRI, TTS, and conversation context.

    Full chain: VAD → STT → ConversationContext → AIRI → TTS → AudioPlayback.

    Phase 4 improvements over the original skeleton:
    - ConversationContext: per-turn tracking, multi-turn history
    - Error recovery: STT exceptions caught and logged
    - AIRI disconnect buffering: unsent STT results queued
    - output:gen-ai:chat:complete: marks turns complete
    - Graceful TTS degradation: logs text when TTS unavailable

    Args:
        pipeline: Configured AudioPipeline instance.
        stt: Configured STT engine instance.
        post_processor: Optional text post-processor.
    """
    from src.airi.websocket_client import AIRIClient
    from src.airi.conversation import ConversationContext, TurnStatus

    # ── Phase 4: Conversation context ──────────────────────────
    ctx = ConversationContext(history_limit=20)

    # ── AIRI client ────────────────────────────────────────────
    airi = AIRIClient(
        host=pipeline.config.airi.host,
        port=pipeline.config.airi.port,
        path=pipeline.config.airi.path,
        token=pipeline.config.airi.token,
        reconnect_interval=pipeline.config.airi.reconnect_interval,
        max_attempts=pipeline.config.airi.max_reconnect_attempts,
    )

    # ── Audio playback ─────────────────────────────────────────
    playback = AudioPlayback(
        device_id=pipeline.config.audio.output_device,
        sample_rate=pipeline.config.audio.output_sample_rate,
    )

    # ── Phase 4: Buffered send queue (for AIRI disconnect) ────
    pending_sends: asyncio.Queue[dict] = asyncio.Queue(maxsize=32)

    async def _flush_pending_sends() -> None:
        """Drain buffered STT results when AIRI reconnects."""
        while not pending_sends.empty():
            try:
                msg = pending_sends.get_nowait()
                await airi.send(msg)
                pending_sends.task_done()
                logger.debug("Flushed buffered STT: \"{}\"", msg["data"]["text"][:60])
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.warning("Flush send error: {}", e)
                break

    # ── TTS initialization ─────────────────────────────────────
    tts_mgr = None
    tts_available = False
    try:
        tts_cfg = pipeline.config.tts
        if tts_cfg.engine == "cosyvoice":
            tts_engine = CosyVoiceTTS(
                model_size=tts_cfg.model_size,
                device=tts_cfg.device,
                model_dir=tts_cfg.model_dir,
                sample_rate=tts_cfg.sample_rate,
                default_voice=tts_cfg.voice_id,
                default_speed=tts_cfg.speed,
            )
        else:
            logger.warning("Unsupported TTS engine '{}', TTS disabled",
                           tts_cfg.engine)
            tts_engine = None

        if tts_engine is not None:
            tts_mgr = TTSManager(engine=tts_engine, playback=playback)
            tts_available = True
            logger.info("TTS initialized: engine={}, voice={}, speed={}",
                        tts_cfg.engine, tts_cfg.voice_id, tts_cfg.speed)
    except ImportError as e:
        logger.warning("TTS engine import error (TTS disabled): {}", e)
    except Exception as e:
        logger.warning("TTS initialization error (TTS disabled): {}", e)

    # ── AIRI → TTS handler ─────────────────────────────────────
    async def _on_airi_message(data: dict) -> None:
        """Handle AIRI streaming chat message → TTS playback + context.

        Accumulates response chunks into the active conversation turn.
        Falls back to text-only logging when TTS is unavailable.

        Args:
            data: Event data dict from AIRI.
        """
        text = (data.get("text") or data.get("message")
                or data.get("content") or "")
        if not text:
            return

        # Phase 4: Track in conversation context
        turn = ctx.append_response(text)
        if turn is not None:
            logger.debug("Turn {} chunk: \"{}\"", turn.turn_id, text[:60])

        # Phase 4: Play via TTS (or log when unavailable)
        if tts_available and tts_mgr is not None:
            logger.debug("AIRI→TTS: \"{}\"", text[:80])
            try:
                await tts_mgr.say(text)
            except Exception as e:
                logger.error("TTS playback error: {}", e)
        else:
            # Graceful degradation: log the text
            logger.info("AIRI reply (no TTS): \"{}\"", text[:120])

    async def _on_airi_complete(data: dict) -> None:
        """Handle AIRI chat complete → mark turn as done.

        Args:
            data: Event data dict from AIRI.
        """
        turn = ctx.complete_current_turn()
        if turn is not None:
            logger.info(
                "Turn {} complete: {} chars, {:.1f}s",
                turn.turn_id, len(turn.response_text), turn.duration,
            )
        print(f"   ✅ [AIRI complete]")

    airi.on("output:gen-ai:chat:message", _on_airi_message)
    airi.on("output:gen-ai:chat:complete", _on_airi_complete)
    logger.info("AIRI event handlers registered")

    # ── Speech → STT → AIRI callback ───────────────────────────
    # Track connection status (nonlocal in closure)
    airi_connected = False

    async def on_speech(event) -> None:
        """Speech event → STT → ConversationContext → AIRI.

        Phase 4 additions:
        - STT exceptions are caught and logged (no silent crashes)
        - Low-confidence results are tracked in context
        - During AIRI disconnect, results are buffered in pending_sends queue

        Args:
            event: SpeechEvent from VAD.
        """
        nonlocal airi_connected

        if event.type == SpeechEventType.SPEECH_START:
            print(f"\n🗣️  [SPEECH START]")
            # Phase 4: Interrupt current TTS playback if user speaks (Phase 5 prep)
            if tts_available and tts_mgr is not None:
                try:
                    await tts_mgr.stop()
                except Exception:
                    pass

        elif event.type == SpeechEventType.SPEECH_END:
            print(f"🤫 [END] dur={event.duration:.2f}s", end="")

            if event.audio is None:
                print()
                return

            # Phase 4: STT with error recovery
            try:
                result = await stt.transcribe(event.audio, event.sample_rate)
            except Exception as e:
                logger.error("STT transcription error: {}", e)
                print(f" ❌ STT error: {e}")
                return

            if result.text and result.confidence >= pipeline.config.stt.min_confidence:
                # Apply post-processing
                output_text = result.text
                if post_processor:
                    output_text = post_processor.process(
                        result.text, confidence=result.confidence,
                    )

                # Phase 4: Start conversation turn
                turn = ctx.start_user_turn(
                    text=output_text,
                    confidence=result.confidence,
                    language=result.language or "zh",
                )

                # Phase 4: Send to AIRI (or buffer if disconnected)
                message = {
                    "type": "input:text:voice",
                    "data": {
                        "text": output_text,
                        "language": result.language or "zh",
                        "turn_id": turn.turn_id,
                    },
                }

                if airi_connected:
                    success = await airi.send(message)
                    if success:
                        print(f" → \"{output_text}\" (conf={result.confidence:.2f})")
                        logger.info("STT→AIRI: \"{}\" (conf={:.2f})",
                                    output_text, result.confidence)
                    else:
                        print(f" ❌ AIRI send failed")
                        # Phase 4: Buffer for retry
                        await _safe_put_pending(message, output_text)
                else:
                    print(f" ⏳ (AIRI not connected, buffered)")
                    await _safe_put_pending(message, output_text)

            elif result.text:
                print(f" (low conf={result.confidence:.2f}, dropped)")
                logger.debug("Low confidence STT dropped: \"{}\"", result.text[:60])
            else:
                print(f" (silent)")

    async def _safe_put_pending(message: dict, text: str) -> None:
        """Safely enqueue a message for retry on reconnect.

        Args:
            message: The message dict to buffer.
            text: Display text for logging.
        """
        try:
            pending_sends.put_nowait(message)
            logger.debug("Buffered STT for retry: \"{}\"", text[:60])
        except asyncio.QueueFull:
            logger.warning("Pending send queue full, dropping: \"{}\"", text[:60])

    pipeline.on_speech_event(on_speech)

    print("\n🎤 AIRI Voice Module - Full Mode (VAD → STT → AIRI → TTS)")
    print("=" * 60)
    print(f"   AIRI:     {pipeline.config.airi.url}")
    print(f"   TTS:      {'✅ ' + pipeline.config.tts.engine if tts_available else '❌ disabled'}")
    print(f"   Session:  {ctx.session_id}")
    print("=" * 60)

    try:
        # Start audio playback
        await playback.start()
        logger.info("AudioPlayback started")

        # Start AIRI client in background
        airi_task = asyncio.create_task(airi.run(), name="airi")

        # Start pipeline (capture + VAD)
        await pipeline.start()

        # Wait for AIRI connection
        for _ in range(10):
            if airi.is_connected:
                airi_connected = True
                print("   ✅ Connected to AIRI")
                # Flush any buffered messages from previous sessions
                await _flush_pending_sends()
                break
            await asyncio.sleep(0.5)

        if not airi_connected:
            print("   ⚠️  AIRI not available — voice input will be buffered")

        # ── Main event loop ────────────────────────────────────
        while pipeline.is_running:
            # Phase 4: Connection state transitions
            if airi.is_connected and not airi_connected:
                airi_connected = True
                print("\n   🔄 AIRI reconnected")
                await _flush_pending_sends()
            elif not airi.is_connected and airi_connected:
                airi_connected = False
                print("\n   🔌 AIRI disconnected, buffering...")

            await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        pass
    finally:
        # Phase 4: Log final conversation summary
        summary = ctx.summary()
        logger.info(
            "Conversation ended: session={}, turns={}, duration={:.0f}s",
            summary["session_id"], summary["total_turns"],
            summary["duration_s"],
        )

        await pipeline.stop()
        airi_task.cancel()
        try:
            await airi_task
        except asyncio.CancelledError:
            pass
        await airi.stop()
        if tts_mgr is not None:
            await tts_mgr.cleanup()
        await playback.stop()
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

    # Initialize STT if needed (test-stt or full mode)
    stt = None
    post_processor = None
    if args.test_stt or not (args.test_vad or args.list_devices):
        stt = FasterWhisperSTT(
            model_size=config.stt.model_size,
            device=config.stt.device,
            compute_type=config.stt.compute_type,
            model_dir=config.stt.model_dir,
            language=config.stt.language,
            beam_size=config.stt.beam_size,
            vad_filter=config.stt.vad_filter,
            hotwords=config.stt.hotwords,
        )

        if config.stt.enable_post_processing:
            post_processor = TextPostProcessor(
                hotwords=config.stt.hotwords,
                min_confidence=config.stt.min_confidence,
            )

    # Run selected mode
    try:
        if args.test_vad:
            task = asyncio.create_task(_run_test_vad(pipeline))
        elif args.test_stt:
            task = asyncio.create_task(
                _run_test_stt(pipeline, stt, post_processor)
            )
        elif args.test_tts:
            task = asyncio.create_task(
                _run_test_tts(config.tts)
            )
        elif args.test_tts_no_play:
            task = asyncio.create_task(
                _run_test_tts_no_play(config.tts, args.test_tts_no_play)
            )
        else:
            task = asyncio.create_task(
                _run_full(pipeline, stt, post_processor)
            )

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
        # Cleanup STT if initialized
        if stt is not None:
            await stt.cleanup()
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
