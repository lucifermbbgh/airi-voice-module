#!/usr/bin/env python3
"""
AIRI Voice Module — Phase 3 Step 7: Windows TTS Verification Script.

Tests TTS engine installation, model loading, synthesis, and playback
on the target Windows machine.

Usage:
    python scripts/test_tts_windows.py [--mode MODE] [--text TEXT]

Modes:
    check       Check environment (Python, CUDA, deps) — default
    synthesize  Test text-to-speech synthesis (saves WAV file)
    play        Test synthesis + playback through speakers
    all         Run all checks sequentially

Examples:
    python scripts/test_tts_windows.py
    python scripts/test_tts_windows.py --mode synthesize
    python scripts/test_tts_windows.py --mode play --text "你好，世界"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, status: bool, detail: str = "") -> None:
    """Print a check result."""
    icon = "✅" if status else "❌"
    if detail:
        print(f"  {icon} {name}: {detail}")
    else:
        print(f"  {icon} {name}")


def _check_import(module_name: str, pip_name: str | None = None,
                  import_as: str | None = None) -> tuple[bool, str]:
    """Check if a Python module is importable.

    Args:
        module_name: Module name to import.
        pip_name: Name used in pip install (defaults to module_name).
        import_as: Import with this name if different from module_name.

    Returns:
        Tuple of (importable, version_or_error).
    """
    pip_name = pip_name or module_name
    try:
        mod = __import__(import_as or module_name)
        ver = getattr(mod, "__version__", str(getattr(mod, "version", "installed")))
        return True, ver
    except ImportError:
        return False, f"Not installed. Run: pip install {pip_name}"
    except Exception as e:
        return False, str(e)


async def check_environment() -> dict:
    """Check Python, PyTorch, CUDA, and audio device status.

    Returns:
        Dict with environment info.
    """
    print_header("Step 7.1: Environment Check")

    info: dict = {
        "python": {},
        "cuda": {},
        "audio": {},
        "tts": {},
        "project": {},
        "all_ok": True,
    }

    # ── Python ────────────────────────────────────────────────────
    py_ver = sys.version
    info["python"]["version"] = py_ver
    print_result("Python", True, py_ver.split()[0])

    # ── Core Project Dependencies ─────────────────────────────────
    # These are needed BEFORE any src/ import can work
    print_header("Project Dependencies")
    core_deps = [
        ("loguru", "loguru"),
        ("numpy", "numpy"),
        ("websockets", "websockets"),
        ("yaml", "pyyaml"),
        ("onnxruntime", "onnxruntime"),
    ]
    all_core_ok = True
    for mod_name, pip_name in core_deps:
        ok, detail = _check_import(mod_name, pip_name)
        print_result(mod_name.capitalize(), ok, detail)
        info["project"][mod_name] = ok
        if not ok:
            all_core_ok = False

    info["project"]["core_ok"] = all_core_ok
    if not all_core_ok:
        info["all_ok"] = False

    # ── PyTorch / CUDA ────────────────────────────────────────────
    print_header("PyTorch / CUDA")
    try:
        import torch
        info["cuda"]["torch_version"] = torch.__version__
        info["cuda"]["cuda_available"] = torch.cuda.is_available()
        print_result("PyTorch", True, torch.__version__)

        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda or "unknown"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            info["cuda"]["cuda_version"] = cuda_ver
            info["cuda"]["gpu_name"] = gpu_name
            info["cuda"]["gpu_count"] = gpu_count
            print_result("CUDA", True, f"{cuda_ver} | {gpu_name} (x{gpu_count})")
        else:
            print_result("CUDA", False, "CPU mode (CUDA not available)")
            info["all_ok"] = False
    except ImportError:
        print_result("PyTorch", False, "Not installed — needed for CosyVoice 2")
        info["cuda"]["cuda_available"] = False
        info["all_ok"] = False
    except Exception as e:
        print_result("PyTorch check", False, str(e))
        info["all_ok"] = False

    # ── Audio Devices ─────────────────────────────────────────────
    print_header("Audio Devices")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        output_devices = [d for d in devices if d["max_output_channels"] > 0]
        input_devices = [d for d in devices if d["max_input_channels"] > 0]

        default_out = sd.default.device[1]
        default_in = sd.default.device[0]

        info["audio"]["output_devices"] = len(output_devices)
        info["audio"]["input_devices"] = len(input_devices)
        info["audio"]["default_output"] = default_out

        print_result("sounddevice", True,
                     f"{len(output_devices)} outputs, {len(input_devices)} inputs")
        print_result("Default Output", True,
                     f"{output_devices[default_out]['name']}"
                     if default_out >= 0 and output_devices else "None")

        # Show output devices
        for i, dev in enumerate(output_devices):
            ch = dev["max_output_channels"]
            rate = dev["default_samplerate"]
            marker = " ← DEFAULT" if (
                default_out >= 0 and i == default_out
            ) else ""
            print(f"       [{i}] {dev['name']} ({ch} ch, {rate:.0f} Hz){marker}")
    except ImportError:
        print_result("sounddevice", False, "Not installed. Run: pip install sounddevice")
        info["all_ok"] = False
    except Exception as e:
        print_result("Audio devices", False, str(e))

    # ── TTS & Synthesis Dependencies ──────────────────────────────
    print_header("TTS & Synthesis Dependencies")
    synth_deps = [
        ("scipy", "scipy"),
        ("cosyvoice", "cosyvoice"),
        ("edge_tts", "edge-tts"),
    ]
    for mod_name, pip_name in synth_deps:
        ok, detail = _check_import(mod_name, pip_name)
        print_result(mod_name.capitalize(), ok, detail)
        info["tts"][mod_name] = ok
        if mod_name in ("cosyvoice", "scipy") and not ok:
            info["all_ok"] = False

    return info


# ── Test: Synthesis (without playback, save WAV) ──────────────

async def test_synthesize(text: str | None = None,
                          output_dir: str = "output",
                          model_dir: str | None = None) -> bool:
    """Test TTS synthesis with CosyVoice 2 engine.

    All project imports are wrapped in try/except to handle
    missing dependencies gracefully.

    Args:
        text: Text to synthesise (default: preset test phrases).
        output_dir: Directory to save WAV files.
        model_dir: Path to CosyVoice model directory.

    Returns:
        True if all tests passed.
    """
    print_header("Step 7.2: TTS Synthesis Test")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    test_texts = [
        text or "你好，我是 AIRI，你的智能语音助手。",
        text or "今天天气真不错，适合出去散步。",
        text or "Welcome to the AIRI Voice Module, Text-to-Speech is ready.",
    ]

    # Lazy imports — may fail if loguru / cosyvoice not installed
    try:
        from src.config import TTSConfig
        from src.tts import CosyVoiceTTS, TTSResult
    except ImportError as e:
        print_result("Project Import", False,
                     f"Cannot import TTS module: {e}\n"
                     "       Run check mode first to see what's missing.")
        return False

    cfg = TTSConfig(
        engine="cosyvoice",
        model_size="base",
        device="cuda",
        sample_rate=24000,
        voice_id="default",
        speed=1.0,
        model_dir=model_dir,
    )

    try:
        tts = CosyVoiceTTS(
            model_size=cfg.model_size,
            device=cfg.device,
            model_dir=cfg.model_dir,
            sample_rate=cfg.sample_rate,
            default_voice=cfg.voice_id,
            default_speed=cfg.speed,
        )

        # May need CPU fallback if CUDA unavailable
        try:
            await tts.load_model()
        except Exception:
            if cfg.device == "cuda":
                print("  ⚠️  CUDA model load failed, retrying on CPU...")
                cfg.device = "cpu"
                tts = CosyVoiceTTS(
                    model_size=cfg.model_size,
                    device=cfg.device,
                    model_dir=cfg.model_dir,
                    sample_rate=cfg.sample_rate,
                    default_voice=cfg.voice_id,
                    default_speed=cfg.speed,
                )
                await tts.load_model()
            else:
                raise

        print_result("Model Load", True, f"device={cfg.device}")

        # Synthesize each test text
        all_passed = True
        for i, t in enumerate(test_texts):
            print(f"\n  [{i + 1}/{len(test_texts)}] Synthesising {len(t)} chars...")
            print(f"       Text: \"{t}\"")

            synth_start = time.monotonic()
            result: TTSResult = await tts.synthesize(
                t, voice_id=cfg.voice_id, speed=cfg.speed,
            )
            synth_time = time.monotonic() - synth_start

            if len(result.audio) > 0:
                duration = result.duration
                rtf = result.synthesis_time / duration if duration > 0 else 0

                wav_path = output_path / f"tts_windows_test_{i + 1}.wav"
                try:
                    from scipy.io import wavfile
                    wavfile.write(
                        str(wav_path),
                        result.sample_rate,
                        (result.audio * 32767).astype("int16"),
                    )
                    saved_msg = f" | Saved: {wav_path}"
                except ImportError:
                    saved_msg = " | (scipy not installed, WAV not saved)"

                print_result("Synthesis", True,
                             f"{duration:.2f}s audio in {synth_time:.2f}s "
                             f"(RTF={rtf:.2f}){saved_msg}")

                if rtf > 0.3:
                    print(f"       ⚠️  RTF {rtf:.2f} > target 0.3 — "
                          f"consider int8 quantization or CUDA")
            else:
                print_result("Synthesis", False, "Empty audio output")
                all_passed = False

        # Test streaming synthesis
        print(f"\n  Streaming synthesis test...")
        stream_text = "这是一个流式合成测试。我们会分段输出音频。每一段都可以单独播放。"
        chunk_count = 0
        async for chunk in tts.synthesize_stream(
            stream_text, voice_id=cfg.voice_id, speed=cfg.speed,
        ):
            if len(chunk) > 0:
                chunk_count += 1

        print_result("Streaming", chunk_count > 0,
                     f"{chunk_count} chunks from {len(stream_text)} chars")

        # Voice switching test
        print(f"\n  Voice switching test...")
        for voice_id in ["default", "中文男声"]:
            try:
                result = await tts.synthesize(
                    "语音切换测试。", voice_id=voice_id,
                )
                print_result(f"Voice '{voice_id}'", len(result.audio) > 0,
                             f"{result.duration:.2f}s")
            except Exception as e:
                print_result(f"Voice '{voice_id}'", False, str(e))

        return all_passed

    except ImportError as e:
        print_result("CosyVoice 2", False,
                     f"Import error: {e}\n"
                     "       Install with: pip install cosyvoice")
        return False
    except Exception as e:
        print_result("Synthesis Test", False, str(e))
        return False
    finally:
        await tts.cleanup()


# ── Test: Synthesis + Playback (speaker output) ───────────────

async def test_playback(text: str = "你好，欢迎使用AIRI语音模块，语音合成与播放测试。",
                        model_dir: str | None = None) -> bool:
    """Test TTS synthesis + playback through speakers.

    All project imports are wrapped in try/except to handle
    missing dependencies gracefully.

    Args:
        text: Text to speak.

    Returns:
        True if playback was successful.
    """
    print_header("Step 7.3: TTS Playback Test (Speaker Output)")

    # Lazy imports — may fail if loguru / cosyvoice / sounddevice not installed
    try:
        from src.config import TTSConfig
        from src.tts import CosyVoiceTTS, TTSManager
        from src.audio.playback import AudioPlayback
    except ImportError as e:
        print_result("Project Import", False,
                     f"Cannot import playback modules: {e}\n"
                     "       Install: pip install sounddevice loguru")
        return False

    cfg = TTSConfig(
        engine="cosyvoice",
        model_size="base",
        device="cuda",
        voice_id="default",
        speed=1.0,
        sample_rate=24000,
        model_dir=model_dir,
    )

    try:
        import sounddevice as sd
        # Use the output device's native sample rate (e.g. 44100 for Realtek),
        # NOT the TTS sample rate (24000) — the playback stream resamples.
        try:
            device_rate = int(sd.query_devices(kind="output")["default_samplerate"])
        except Exception:
            device_rate = 44100

        playback = AudioPlayback(
            device_id=None,
            sample_rate=device_rate,
        )

        tts = CosyVoiceTTS(
            model_size=cfg.model_size,
            device=cfg.device,
            model_dir=cfg.model_dir,
            sample_rate=cfg.sample_rate,
            default_voice=cfg.voice_id,
            default_speed=cfg.speed,
        )

        tts_mgr = TTSManager(engine=tts, playback=playback)

        await playback.start()
        print_result("Playback Started", True, "Audio output stream ready")

        print(f"\n  Speaking: \"{text}\"")
        print("  🔊 Listen to the speaker...")

        result = await tts_mgr.say(text)
        print_result("Playback", result, "" if result else "Failed")

        # Test pause/resume
        print(f"\n  Testing controls...")
        await tts_mgr.pause()
        print_result("Pause", True, "Playback paused")
        await asyncio.sleep(0.5)
        await tts_mgr.resume()
        print_result("Resume", True, "Playback resumed")

        return result

    except Exception as e:
        print_result("Playback Test", False, str(e))
        return False
    finally:
        try:
            await tts_mgr.cleanup()
        except Exception:
            pass
        try:
            await playback.stop()
        except Exception:
            pass


# ── Main entry point ──────────────────────────────────────────

async def main() -> int:
    """Run Windows TTS verification."""
    parser = argparse.ArgumentParser(
        description="AIRI Phase 3 Step 7: Windows TTS Verification",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["check", "synthesize", "play", "all"],
        default="check",
        help="Verification mode (default: check)",
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Custom text for synthesis/playback tests",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Output directory for WAV files (default: output/)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to CosyVoice model directory "
             "(default: $TTS_MODEL_DIR or pretrained_models/CosyVoice2-0.5B)",
    )

    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AIRI Voice Module — Phase 3 Step 7                    ║")
    print("║  Windows TTS Verification                              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    all_passed = True

    # ── Check mode (always runs first) ────────────────────────────
    env_info = await check_environment()
    cuda_ok = env_info.get("cuda", {}).get("cuda_available", False)
    cosy_ok = env_info.get("tts", {}).get("cosyvoice", False)
    core_ok = env_info.get("project", {}).get("core_ok", False)

    # ── Pre-check before running heavier tests ────────────────────
    if args.mode in ("synthesize", "play", "all"):
        missing = []
        if not core_ok:
            missing.append("core project dependencies (loguru, etc.)")
        if not cosy_ok:
            missing.append("CosyVoice 2 (pip install cosyvoice)")

        if missing:
            print()
            print("─" * 60)
            print("  ⚠️  Cannot continue: missing dependencies")
            for m in missing:
                print(f"     • {m}")
            print()
            print("  Run 'pip install' commands above, then retry.")
            print("─" * 60)
            all_passed = False

    # ── Run selected modes ────────────────────────────────────────
    if args.mode == "check":
        all_passed = cuda_ok and cosy_ok and core_ok

    elif args.mode == "synthesize" and all_passed:
        ok = await test_synthesize(args.text, args.output_dir, args.model_dir)
        all_passed = all_passed and ok

    elif args.mode == "play" and all_passed:
        if args.mode == "all":
            print("\n" + "─" * 60)
        ok = await test_playback(
            args.text or "你好，欢迎使用AIRI语音模块，所有测试已完成。",
            args.model_dir,
        )
        all_passed = all_passed and ok

    elif args.mode == "all" and all_passed:
        ok = await test_synthesize(args.text, args.output_dir, args.model_dir)
        all_passed = all_passed and ok

        if all_passed:
            print("\n" + "─" * 60)
            ok = await test_playback(
                args.text or "你好，欢迎使用AIRI语音模块，所有测试已完成。",
                args.model_dir,
            )
            all_passed = all_passed and ok

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  {'✅ All tests passed!' if all_passed else '❌ Some tests failed.'}")
    print("  Phase 3 Step 7 Windows Verification "
          f"{'COMPLETE' if all_passed else 'INCOMPLETE'}")
    print("=" * 60)
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
