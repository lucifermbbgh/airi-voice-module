"""
麦克风音量测试工具

检测 Windows 默认麦克风的实际输入音量峰值，用于诊断 VAD 检测失效问题。

用法:
    python -m tests.test_mic_level
    python -m tests.test_mic_level --duration 10      # 监听 10 秒
    python -m tests.test_mic_level --device 1          # 指定设备编号
    python -m tests.test_mic_level --device 9 --rate 48000  # WASAPI 设备
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="麦克风音量测试 - 检测音频输入峰值电平",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=5,
        help="监听时长（秒），默认 5 秒",
    )
    parser.add_argument(
        "--device", "-dev",
        type=int,
        default=None,
        help="音频输入设备编号（默认使用系统默认设备）",
    )
    parser.add_argument(
        "--rate", "-r",
        type=int,
        default=48000,
        help="采样率（Hz），默认 48000",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    device_info = sd.query_devices(args.device, kind="input") if args.device is not None else sd.query_devices(kind="input")
    print(f"\n🎤 麦克风音量测试")
    print(f"   {"=" * 50}")
    print(f"   📍 设备: {args.device if args.device is not None else '默认'}")
    print(f"   🆔 设备名: {device_info['name']}")
    print(f"   📊 采样率: {args.rate} Hz")
    print(f"   ⏱️  时长: {args.duration} 秒")
    print(f"   {"=" * 50}\n")

    print(f"   🗣️  请对着麦克风说话...")
    print()

    max_level = 0.0
    sample_count = 0

    def callback(indata: np.ndarray, frames: int, _time_info, _status) -> None:
        nonlocal max_level, sample_count
        level = float(np.max(np.abs(indata)))
        max_level = max(max_level, level)
        sample_count += 1

        # 每 5 帧打印一次当前峰值
        if sample_count % 5 == 0:
            bar_len = int(level * 50)
            bar = "█" * min(bar_len, 50)
            pct = level * 100
            print(f"      Peak: {level:.6f} ({pct:.1f}%)  {'|' + bar:<51}", end="\r")

    stream = sd.InputStream(
        device=args.device,
        samplerate=args.rate,
        channels=1,
        dtype="float32",
        callback=callback,
    )

    try:
        stream.start()
        time.sleep(args.duration)
        stream.stop()
    except KeyboardInterrupt:
        print("\n\n   ⏹️  用户中断")
    except Exception as e:
        print(f"\n\n   ❌ 错误: {e}")
        return

    stream.close()

    print(f"\n\n   {"=" * 50}")
    print(f"   📊 测试结果:")
    print(f"   采样次数: {sample_count}")
    print(f"   最大峰值: {max_level:.6f} ({max_level * 100:.1f}%)")

    # 评估音量水平
    if max_level < 0.01:
        suggestion = "🔇 音量极低! 请检查麦克风是否开启、音量是否调高、Windows隐私权限"
    elif max_level < 0.05:
        suggestion = "🔈 音量偏低，VAD 可能无法触发。建议提高麦克风增益或靠近麦克风"
    elif max_level < 0.2:
        suggestion = "🔉 音量尚可，但 VAD 阈值 0.5 可能略高。可尝试降低 VAD_THRESHOLD=0.3"
    elif max_level < 0.5:
        suggestion = "🔊 音量正常。如 VAD 仍未触发，建议尝试降低 VAD_THRESHOLD=0.3"
    else:
        suggestion = "📢 音量充足! VAD 应能正常工作"

    print(f"   建议: {suggestion}")
    print(f"   {"=" * 50}\n")


if __name__ == "__main__":
    main()
