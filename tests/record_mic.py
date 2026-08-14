"""
麦克风录音验证脚本

采集指定设备的音频，保存为 WAV 文件，用于客观确认"采集到的信号到底是什么"。
配合 faster-whisper 转写，可判断采集到的是语音、噪声还是静音。

用法:
    python -m tests.record_mic                     # 默认设备，5 秒
    python -m tests.record_mic --device 5          # DirectSound 设备，5 秒
    python -m tests.record_mic --device 5 --duration 8 --rate 48000

输出:
    - 保存 WAV 到 output/record_mic_<timestamp>.wav
    - 打印 Peak / RMS 统计（RMS 更能反映"实际响度"）
"""

from __future__ import annotations

import argparse
import time
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="麦克风录音验证 - 采集音频并保存 WAV",
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=None,
        help="输入设备编号（默认系统默认设备）",
    )
    parser.add_argument(
        "--duration", "-t",
        type=int,
        default=5,
        help="录音时长（秒），默认 5 秒",
    )
    parser.add_argument(
        "--rate", "-r",
        type=int,
        default=48000,
        help="采样率（Hz），默认 48000",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出 WAV 路径（默认 output/record_mic_<时间戳>.wav）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    device_info = (
        sd.query_devices(args.device, kind="input")
        if args.device is not None
        else sd.query_devices(kind="input")
    )

    print(f"\n🎙️  麦克风录音验证")
    print(f"   {'=' * 50}")
    print(f"   📍 设备: {args.device if args.device is not None else '默认'}")
    print(f"   🆔 设备名: {device_info['name']}")
    print(f"   📊 采样率: {args.rate} Hz")
    print(f"   ⏱️  时长: {args.duration} 秒")
    print(f"   {'=' * 50}\n")
    print(f"   🗣️  请大声、清晰地说一句话（如：\"今天天气很好\"）...\n")

    # 采集音频（阻塞模式，一次性采集完整时长）
    audio = sd.rec(
        int(args.duration * args.rate),
        samplerate=args.rate,
        channels=1,
        dtype="float32",
        device=args.device,
    )
    sd.wait()  # 等待采集完成

    # 统计
    mono = audio[:, 0]
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    print(f"\n   📊 统计:")
    print(f"   Peak: {peak:.4f} ({peak * 100:.1f}%)")
    print(f"   RMS:  {rms:.4f} ({rms * 100:.1f}%)  ← RMS 反映实际响度")
    print(f"   样本数: {len(mono)}")

    # 保存 WAV
    import os
    output_path = args.output or os.path.join(
        "output", f"record_mic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    int16_data = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 位
        wf.setframerate(args.rate)
        wf.writeframes(int16_data.tobytes())

    print(f"\n   💾 已保存: {output_path}")
    print(f"   → 转写命令: python -m tests.transcribe_wavs {output_path}")
    print(f"   → 或把 WAV 发给我，我用 faster-whisper 转写\n")


if __name__ == "__main__":
    main()
