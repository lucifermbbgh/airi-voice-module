"""
VAD 实时诊断工具

实时捕获麦克风音频，对每一帧运行 Silero VAD 推理并显示概率值。
用于诊断 VAD 检测失效问题：查看实际语音概率曲线、阈值匹配情况。

用法:
    python -m tests.test_vad_diagnostic              # 默认监听 10 秒
    python -m tests.test_vad_diagnostic --duration 5  # 监听 5 秒
    python -m tests.test_vad_diagnostic --threshold 0.5  # 用不同阈值
    python -m tests.test_vad_diagnostic --device 1 --rate 48000
    python -m tests.test_vad_diagnostic --model models/silero_vad.onnx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from src.audio.resampler import Resampler
from src.vad.silero_vad import SileroVAD


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAD 实时诊断 - 显示每帧语音概率和阈值对比",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=10,
        help="监听时长（秒），默认 10 秒",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="VAD 阈值（覆盖 config/default.yaml），默认 0.3",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/silero_vad.onnx",
        help="ONNX 模型路径，默认 models/silero_vad.onnx",
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
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示每一帧的概率（默认每 5 帧显示一次）",
    )
    return parser.parse_args()


def _make_signal_bar(prob: float, threshold: float) -> str:
    """生成概率可视化条。

    Args:
        prob: VAD 概率值 (0.0-1.0)。
        threshold: 当前阈值。

    Returns:
        带颜色指示的字符串条。
    """
    bar_len = int(prob * 50)
    bar = "█" * min(bar_len, 50)

    if prob >= threshold:
        flag = "🔊 SPEECH"
    elif prob >= threshold * 0.5:
        flag = "🔉"
    elif prob >= 0.05:
        flag = "🔈"
    else:
        flag = "🔇"

    return f"{flag}  |{bar:<51}"


def main() -> None:
    args = _parse_args()

    # 获取使用配置的阈值（默认 0.3，可被命令行覆盖）
    threshold = args.threshold if args.threshold is not None else 0.3

    # 加载 VAD 模型
    vad = SileroVAD(model_path=args.model, threshold=threshold)
    vad.load_model()

    resampler = Resampler(args.rate, 16000)

    # 获取设备信息
    if args.device is not None:
        device_info = sd.query_devices(args.device, kind="input")
    else:
        device_info = sd.query_devices(kind="input")

    print(f"\n🎤 VAD 实时诊断")
    print(f"   {'=' * 55}")
    print(f"   📍 设备: {args.device if args.device is not None else '默认'}")
    print(f"   🆔 设备名: {device_info['name']}")
    print(f"   📊 采样率: {args.rate} Hz → 16 kHz")
    print(f"   🎯 VAD 阈值: {threshold}")
    print(f"   ⏱️  时长: {args.duration} 秒")
    print(f"   {'=' * 55}")
    print(f"   🗣️  请对着麦克风说话...")
    print()

    # 统计数据
    total_frames = 0
    above_threshold = 0
    max_prob = 0.0
    max_prob_frame = 0
    last_display_time = 0.0

    # 帧缓冲：48kHz raw → resample → 16kHz → 积累到 512 → VAD
    frame_buffer: list[float] = []
    _frame_count_local = 0  # 用于闭包里的计数

    def callback(indata: np.ndarray, frames: int, _time_info, _status) -> None:
        nonlocal total_frames, above_threshold, max_prob, max_prob_frame, frame_buffer, _frame_count_local, last_display_time

        if _status:
            print(f"      ⚠️  Status: {_status}")

        # 重采样麦克风捕获的 48kHz → 16kHz
        audio_48k = indata[:, 0].copy()
        resampled = resampler.resample(audio_48k)

        if len(resampled) == 0:
            return

        frame_buffer.extend(resampled.tolist())

        # 积累满 512 samples 就送 VAD
        while len(frame_buffer) >= 512:
            frame = np.array(frame_buffer[:512], dtype=np.float32)
            frame_buffer = frame_buffer[512:]

            prob = vad._get_speech_prob(frame)
            total_frames += 1
            _frame_count_local += 1

            if prob > max_prob:
                max_prob = prob
                max_prob_frame = total_frames
            if prob >= threshold:
                above_threshold += 1

            # 显示频率控制
            now = time.monotonic()
            show = args.verbose
            if not show and (total_frames % 5 == 0 or prob >= threshold * 0.8):
                show = True
            if not show and (now - last_display_time) >= 1.0:
                show = True

            if show:
                bar = _make_signal_bar(prob, threshold)
                elapsed = now - _time_info.current_time if hasattr(_time_info, 'current_time') else 0
                print(f"      frame {total_frames:4d}  prob={prob:.4f}  {bar}")
                last_display_time = now

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
        import traceback
        traceback.print_exc()
        return
    finally:
        stream.close()

    # 刷新剩余帧
    while len(frame_buffer) >= 512:
        frame = np.array(frame_buffer[:512], dtype=np.float32)
        frame_buffer = frame_buffer[512:]
        prob = vad._get_speech_prob(frame)
        total_frames += 1
        if prob > max_prob:
            max_prob = prob
            max_prob_frame = total_frames
        if prob >= threshold:
            above_threshold += 1

    # 打印统计摘要
    print(f"\n\n   {'=' * 55}")
    print(f"   📊 诊断统计:")
    print(f"   总帧数:      {total_frames}")
    print(f"   最高概率:    {max_prob:.4f} (第 {max_prob_frame} 帧)")
    print(f"   超过阈值 {threshold}: {above_threshold}/{total_frames} ({above_threshold/max(total_frames,1)*100:.1f}%)")
    print()

    # 诊断建议
    if max_prob < 0.05:
        suggestion = "🔇 VAD 概率极低! 检查麦克风是否工作、重采样是否正确"
    elif max_prob < threshold * 0.8:
        suggestion = (
            f"🔉 VAD 最大概率 {max_prob:.3f} 低于阈值 {threshold}。"
            f"建议降低阈值到 {max_prob * 0.8:.2f} 或检查重采样"
        )
    elif max_prob >= threshold:
        suggestion = f"📢 VAD 可触发! 阈值 {threshold} 下 {above_threshold} 帧达标。如仍无事件，检查 pipeline 集成"

    print(f"   建议: {suggestion}")
    print(f"   {'=' * 55}\n")


if __name__ == "__main__":
    main()
