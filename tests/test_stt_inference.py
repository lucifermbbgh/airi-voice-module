"""
STT 实时推理测试工具

通过麦克风录制音频并运行 faster-whisper 语音识别，实时输出转写结果。
支持多语言、音频文件输入等模式。

用法:
    python -m tests.test_stt_inference                          # 录音 5 秒并转写
    python -m tests.test_stt_inference --duration 10            # 录音 10 秒
    python -m tests.test_stt_inference --model small            # 使用 small 模型
    python -m tests.test_stt_inference --language en            # 英文识别
    python -m tests.test_stt_inference --file test.wav          # 从 WAV 文件转写
    python -m tests.test_stt_inference --device 9               # 指定麦克风设备
    python -m tests.test_stt_inference --list-devices           # 列出音频设备
    python -m tests.test_stt_inference --save output.wav        # 保存录音到文件
    python -m tests.test_stt_inference --hotwords "AIRI,Claude" # 热词增强
"""

from __future__ import annotations

import argparse
import math
import time
import sys
from pathlib import Path

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────

_STANDARD_SAMPLE_RATE = 16000
_CHANNELS = 1
_DTYPE = "float32"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STT 实时推理测试 - 麦克风录音并转写为文字",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=5,
        help="录音时长（秒），默认 5 秒",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper 模型大小（默认 tiny）",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="zh",
        help="识别语言（zh/en/ja/auto），默认 zh",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="WAV 文件路径（指定后从文件读取而非麦克风）",
    )
    parser.add_argument(
        "--device", "-dev",
        type=int,
        default=None,
        help="音频输入设备编号",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出可用音频输入设备",
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="保存录音到 WAV 文件",
    )
    parser.add_argument(
        "--hotwords", "-hw",
        type=str,
        default=None,
        help="热词列表（逗号分隔），如 'AIRI,Claude,Silero'",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细推理信息（时间戳、置信度等）",
    )
    parser.add_argument(
        "--compute",
        type=str,
        default="int8",
        choices=["int8", "float16", "float32"],
        help="量化精度（默认 int8，CPU 推荐）",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="模型缓存目录",
    )
    return parser.parse_args()


def _list_devices() -> None:
    """列出所有音频输入设备。"""
    import sounddevice as sd

    print("\n=== 可用音频输入设备 ===")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']}")
            print(f"         Channels: {dev['max_input_channels']}, "
                  f"Rate: {int(dev['default_samplerate'])} Hz")
    print()


def _record_microphone(duration: int, device: int | None = None,
                       samplerate: int = _STANDARD_SAMPLE_RATE) -> np.ndarray:
    """从麦克风录制音频。

    Args:
        duration: 录音时长（秒）。
        device: 音频设备编号。
        samplerate: 采样率。

    Returns:
        float32 音频数组，shape=(samples,)。
    """
    import sounddevice as sd

    total_samples = int(duration * samplerate)
    recorded: list[np.ndarray] = []
    max_val = 0.0

    def callback(indata: np.ndarray, frames: int, _time_info, status) -> None:
        nonlocal max_val
        if status:
            print(f"      ⚠️  {status}")
        audio = indata[:, 0].copy()
        recorded.append(audio)
        level = float(np.max(np.abs(audio)))
        max_val = max(max_val, level)
        # 实时显示音量条
        bar_len = min(int(level * 50), 50)
        bar = "█" * bar_len
        pct = level * 100
        print(f"      🎤 录制中... {pct:5.1f}% |{bar:<51}", end="\r")

    stream = sd.InputStream(
        device=device,
        samplerate=samplerate,
        channels=_CHANNELS,
        dtype=_DTYPE,
        blocksize=int(samplerate * 0.1),  # 100ms 每帧
        callback=callback,
    )

    print(f"\n🎤 正在录音 ({duration} 秒)，请说话...")
    print(f"   {'=' * 60}")

    try:
        stream.start()
        time.sleep(duration)
        stream.stop()
    except KeyboardInterrupt:
        print("\n\n   ⏹️  用户中断")
    except Exception as e:
        print(f"\n   ❌ 录音错误: {e}")
        raise
    finally:
        stream.close()

    if not recorded:
        raise RuntimeError("未录制到任何音频")

    audio = np.concatenate(recorded)
    print(f"\n   {'=' * 60}")
    print(f"   ✅ 录音完成: {len(audio) / samplerate:.1f}s, 峰值={max_val:.3f}")

    return audio


def _load_wav_file(file_path: str) -> tuple[np.ndarray, int]:
    """从 WAV 文件加载音频。

    Args:
        file_path: WAV 文件路径。

    Returns:
        (audio_array, sample_rate) 元组。
    """
    import wave

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"WAV 文件不存在: {file_path}")

    with wave.open(str(path), "rb") as wf:
        samplerate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()

        raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        # 混音到单声道
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        print(f"\n📂 从文件加载: {file_path}")
        print(f"   采样率: {samplerate} Hz")
        print(f"   声道数: {n_channels}")
        print(f"   时长: {len(audio) / samplerate:.1f}s")
        print(f"   峰值: {float(np.max(np.abs(audio))):.3f}")

        return audio, samplerate


def _save_wav_file(audio: np.ndarray, file_path: str,
                   samplerate: int = _STANDARD_SAMPLE_RATE) -> None:
    """保存音频到 WAV 文件。

    Args:
        audio: float32 音频数组。
        file_path: 输出路径。
        samplerate: 采样率。
    """
    import wave

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # float32 → int16
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())

    print(f"\n💾 录音已保存: {path}")
    print(f"   大小: {path.stat().st_size / 1024:.1f} KB")


def _run_inference(audio: np.ndarray, samplerate: int,
                   args: argparse.Namespace) -> None:
    """运行 STT 推理并显示结果。

    Args:
        audio: 音频数据。
        samplerate: 音频采样率。
        args: 命令行参数。
    """
    from faster_whisper import WhisperModel

    model_name = f"Systran/faster-whisper-{args.model}"
    print(f"\n🧠 加载模型: {model_name}")
    print(f"   设备: cpu, 量化: {args.compute}")
    print(f"   语言: {args.language}")

    load_start = time.monotonic()
    model = WhisperModel(
        model_size_or_path=model_name,
        device="cpu",
        compute_type=args.compute,
        download_root=args.model_dir,
    )
    load_time = time.monotonic() - load_start
    print(f"   加载耗时: {load_time:.1f}s")

    # 重采样到 16kHz（如果不是）
    if samplerate != _STANDARD_SAMPLE_RATE:
        from scipy import signal
        resample_ratio = _STANDARD_SAMPLE_RATE / samplerate
        new_len = int(len(audio) * resample_ratio)
        audio = signal.resample(audio, new_len).astype(np.float32)
        print(f"   重采样: {samplerate}Hz → {_STANDARD_SAMPLE_RATE}Hz")

    # 配置热词
    hotwords = None
    if args.hotwords:
        hotwords = [w.strip() for w in args.hotwords.split(",")]
        print(f"   热词: {hotwords}")

    print(f"\n🔊 音频信息: {len(audio) / _STANDARD_SAMPLE_RATE:.1f}s @ {_STANDARD_SAMPLE_RATE}Hz")
    print(f"\n🔄 正在转写...")
    print(f"   {'=' * 60}")

    infer_start = time.monotonic()
    segments, info = model.transcribe(
        audio,
        language=args.language if args.language != "auto" else None,
        beam_size=5,
        hotwords=hotwords,
    )
    infer_time = time.monotonic() - infer_start

    # 收集所有片段
    segment_list = list(segments)
    full_text = "".join(seg.text for seg in segment_list)

    print(f"\n📊 推理统计:")
    print(f"   模型: {model_name}")
    print(f"   推理耗时: {infer_time:.2f}s")
    print(f"   音频时长: {len(audio) / _STANDARD_SAMPLE_RATE:.1f}s")
    print(f"   实时率: {infer_time / (len(audio) / _STANDARD_SAMPLE_RATE):.2f}x")
    print(f"   检测语言: {info.language} (p={info.language_probability:.2f})")

    print(f"\n📝 转写结果:")
    print(f"   {'=' * 60}")
    if full_text.strip():
        if args.verbose and segment_list:
            for seg in segment_list:
                confidence_display = f"[conf={seg.avg_logprob:.2f}]" if hasattr(seg, 'avg_logprob') else ""
                print(f"   [{seg.start:.1f}s-{seg.end:.1f}s] {confidence_display} {seg.text}")
        else:
            print(f"   {full_text}")
    else:
        print(f"   ⚠️  未识别到文字（麦克风音量不足或环境噪声过多？）")
        print(f"   提示: 可以试试 test_mic_level 工具检查麦克风电平")

    print(f"   {'=' * 60}")

    # 推理速度评估
    rtf = infer_time / (len(audio) / _STANDARD_SAMPLE_RATE)
    if rtf < 0.5:
        speed_rating = "⚡ 极快 — 适合实时流式"
    elif rtf < 1.0:
        speed_rating = "✅ 实时 — 可以使用"
    elif rtf < 2.0:
        speed_rating = "🐢 较慢 — 适合非实时场景或换更大模型"
    else:
        speed_rating = "🐌 极慢 — 建议用 int8 量化或换 tiny/base 模型"

    print(f"\n📈 性能评估:")
    print(f"   实时率 (RTF): {rtf:.2f}x — {speed_rating}")


def main() -> None:
    args = _parse_args()

    if args.list_devices:
        _list_devices()
        return

    try:
        # 加载音频
        if args.file:
            audio, samplerate = _load_wav_file(args.file)
        else:
            audio = _record_microphone(args.duration, args.device)
            samplerate = _STANDARD_SAMPLE_RATE

            # 可选保存录音
            if args.save:
                _save_wav_file(audio, args.save, samplerate)

        # 运行推理
        _run_inference(audio, samplerate, args)

    except ImportError as e:
        print(f"\n❌ 依赖未安装: {e}")
        if "faster-whisper" in str(e):
            print(f"   安装: pip install faster-whisper")
        elif "sounddevice" in str(e):
            print(f"   安装: pip install sounddevice")
        else:
            print(f"   检查依赖: pip install faster-whisper sounddevice scipy")
    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
