"""
VAD 模型推理方式对比工具

对比两种 VAD 推理方式，诊断 ONNX Runtime 直加载 vs silero-vad 包自带 API 的差异。

用法:
    python -m tests.test_vad_model_compare                     # 默认 10 秒
    python -m tests.test_vad_model_compare --duration 5         # 监听 5 秒
    python -m tests.test_vad_model_compare --method both        # 两种方式都跑
    python -m tests.test_vad_model_compare --method onnx        # 只用 ONNX Runtime
    python -m tests.test_vad_model_compare --method package     # 只用 silero-vad 包
    python -m tests.test_vad_model_compare --device 1 --rate 48000
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

from src.vad.silero_vad import SileroVAD


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAD 模型推理方式对比 - 诊断 ONNX Runtime vs silero-vad 包 API 差异",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=10,
        help="监听时长（秒），默认 10 秒",
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        default="both",
        choices=["both", "onnx", "package"],
        help="推理方式: both（两种都跑）, onnx（仅 ONNX Runtime）, package（仅 silero-vad 包 API）",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="VAD 比较阈值，默认 0.3",
    )
    parser.add_argument(
        "--device", "-dev",
        type=int,
        default=None,
        help="音频输入设备编号（默认使用系统默认设备）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示每一帧的概率（默认每 5 帧显示一次）",
    )
    return parser.parse_args()


def _make_signal_bar(prob: float, threshold: float) -> str:
    """生成概率可视化条。"""
    bar_len = int(prob * 50)
    bar = "█" * min(bar_len, 50)
    if prob >= threshold:
        flag = "🔊"
    elif prob >= 0.05:
        flag = "🔉"
    else:
        flag = "🔇"
    return f"{flag}  |{bar:<51}"


class VADModelCompare:
    """VAD 推理方式对比器。

    同时或分别运行 ONNX Runtime 和 silero-vad 包两种推理方式，
    比较每帧概率输出，用于诊断推理 API 差异。

    Attributes:
        duration: 监听时长（秒）。
        method: 推理方式 ("both", "onnx", "package")。
        threshold: VAD 阈值（用于显示标记）。
        device: 音频设备编号。
        verbose: 是否显示每帧概率。
    """

    def __init__(
        self,
        duration: int = 10,
        method: str = "both",
        threshold: float = 0.3,
        device: int | None = None,
        verbose: bool = False,
    ):
        """初始化对比器。

        Args:
            duration: 监听时长（秒）。
            method: 推理方式。
            threshold: VAD 阈值。
            device: 音频设备编号。
            verbose: 是否显示每帧概率。
        """
        self.duration = duration
        self.method = method
        self.threshold = threshold
        self.device = device
        self.verbose = verbose

        # 推理器
        self._onnx_vad: SileroVAD | None = None
        self._package_model = None

        # 统计
        self._stats = {
            "onnx": {"total": 0, "above": 0, "max_prob": 0.0},
            "package": {"total": 0, "above": 0, "max_prob": 0.0},
        }

    def _load_onnx(self) -> None:
        """加载 ONNX Runtime 直加载的 VAD。"""
        self._onnx_vad = SileroVAD(threshold=self.threshold)
        self._onnx_vad.load_model()
        print(f"   ✅ ONNX Runtime VAD 已加载")

    def _load_package(self) -> None:
        """加载 silero-vad 包自带的 VAD 模型。"""
        try:
            from silero_vad import load_silero_vad
            self._package_model = load_silero_vad(onnx=True)
            print(f"   ✅ silero-vad 包 API VAD 已加载")
        except ImportError:
            print(f"   ❌ silero-vad 包未安装，跳过 package 方式")
            self._package_model = None
        except Exception as e:
            print(f"   ❌ silero-vad 包加载失败: {e}")
            self._package_model = None

    def _get_onnx_prob(self, frame: np.ndarray) -> float:
        """ONNX Runtime 方式推理。

        Args:
            frame: 512 样本 float32 数组 @ 16kHz。

        Returns:
            VAD 概率 (0.0-1.0)。
        """
        if self._onnx_vad is None:
            return 0.0
        return self._onnx_vad._get_speech_prob(frame)

    def _get_package_prob(self, frame: np.ndarray) -> float:
        """silero-vad 包 API 方式推理。

        Args:
            frame: 512 样本 float32 数组 @ 16kHz。

        Returns:
            VAD 概率 (0.0-1.0)。
        """
        if self._package_model is None:
            return 0.0
        try:
            return float(self._package_model(frame, 16000).item())
        except Exception as e:
            print(f"      ⚠️  package 推理错误: {e}")
            return 0.0

    def run(self) -> None:
        """运行对比测试。"""
        # 获取设备信息
        if self.device is not None:
            device_info = sd.query_devices(self.device, kind="input")
        else:
            device_info = sd.query_devices(kind="input")

        # 加载模型
        use_onnx = self.method in ("both", "onnx")
        use_package = self.method in ("both", "package")

        print(f"\n🔬 VAD 模型推理方式对比")
        print(f"   {'=' * 60}")
        print(f"   📍 设备: {self.device if self.device is not None else '默认'}")
        print(f"   🆔 设备名: {device_info['name']}")
        print(f"   📊 采样率: 16000 Hz（直捕）")
        print(f"   🎯 阈值: {self.threshold}")
        print(f"   ⏱️  时长: {self.duration} 秒")
        print(f"   🔬 方法: {self.method}")
        print(f"   {'=' * 60}\n")

        if use_onnx:
            self._load_onnx()
        if use_package:
            self._load_package()

        print(f"\n   🗣️  请对着麦克风说话...\n")

        # 帧缓冲（直捕 16kHz）
        frame_buf: list[float] = []
        last_display = 0.0

        def callback(indata: np.ndarray, frames: int, _time_info, _status) -> None:
            nonlocal frame_buf, last_display

            if _status:
                print(f"      ⚠️  Status: {_status}")

            audio = indata[:, 0].copy()
            if len(audio) == 0:
                return

            frame_buf.extend(audio.tolist())

            while len(frame_buf) >= 512:
                frame = np.array(frame_buf[:512], dtype=np.float32)
                frame_buf = frame_buf[512:]

                probs = {}
                if use_onnx:
                    p = self._get_onnx_prob(frame)
                    probs["onnx"] = p
                    self._stats["onnx"]["total"] += 1
                    if p > self._stats["onnx"]["max_prob"]:
                        self._stats["onnx"]["max_prob"] = p
                    if p >= self.threshold:
                        self._stats["onnx"]["above"] += 1

                if use_package:
                    p = self._get_package_prob(frame)
                    probs["package"] = p
                    self._stats["package"]["total"] += 1
                    if p > self._stats["package"]["max_prob"]:
                        self._stats["package"]["max_prob"] = p
                    if p >= self.threshold:
                        self._stats["package"]["above"] += 1

                now = time.monotonic()
                show = self.verbose
                if not show and (self._stats["onnx"]["total"] % 5 == 0):
                    show = True
                if not show and (now - last_display) >= 1.0:
                    show = True

                if show:
                    total = self._stats["onnx"]["total"]
                    if use_onnx and use_package:
                        p_onnx = probs.get("onnx", 0)
                        p_pkg = probs.get("package", 0)
                        bar_onnx = _make_signal_bar(p_onnx, self.threshold)
                        bar_pkg = _make_signal_bar(p_pkg, self.threshold)
                        diff = p_onnx - p_pkg
                        print(f"      frame {total:4d}")
                        print(f"        ONNX:    prob={p_onnx:.4f}  {bar_onnx}")
                        print(f"        PACKAGE: prob={p_pkg:.4f}  {bar_pkg}")
                        if abs(diff) > 0.01:
                            print(f"        ⚠️  diff={diff:+.4f}")
                    elif use_onnx:
                        p = probs.get("onnx", 0)
                        bar = _make_signal_bar(p, self.threshold)
                        print(f"      frame {total:4d}  prob={p:.4f}  {bar}")
                    elif use_package:
                        p = probs.get("package", 0)
                        bar = _make_signal_bar(p, self.threshold)
                        print(f"      frame {total:4d}  prob={p:.4f}  {bar}")

                    last_display = now

        # 启动音频流
        stream = sd.InputStream(
            device=self.device,
            samplerate=16000,
            channels=1,
            dtype="float32",
            blocksize=512,
            callback=callback,
        )

        try:
            stream.start()
            time.sleep(self.duration)
            stream.stop()
        except KeyboardInterrupt:
            print("\n\n   ⏹️  用户中断")
        except Exception as e:
            print(f"\n\n   ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            stream.close()

        # 输出统计
        self._print_summary()

    def _print_summary(self) -> None:
        """打印统计摘要。"""
        print(f"\n\n   {'=' * 60}")

        if self.method in ("both", "onnx"):
            s = self._stats["onnx"]
            print(f"   📊 ONNX Runtime:")
            print(f"     总帧数:      {s['total']}")
            print(f"     最高概率:    {s['max_prob']:.4f}")
            print(f"     超过 {self.threshold}: {s['above']}/{s['total']} ({s['above']/max(s['total'],1)*100:.1f}%)")

        if self.method in ("both", "package"):
            s = self._stats["package"]
            print(f"   📊 silero-vad 包 API:")
            print(f"     总帧数:      {s['total']}")
            print(f"     最高概率:    {s['max_prob']:.4f}")
            print(f"     超过 {self.threshold}: {s['above']}/{s['total']} ({s['above']/max(s['total'],1)*100:.1f}%)")

        if self.method == "both":
            onnx_max = self._stats["onnx"]["max_prob"]
            pkg_max = self._stats["package"]["max_prob"]
            print(f"\n   💡 对比结论:")
            if onnx_max < 0.01 and pkg_max < 0.01:
                print(f"     两种方式概率都极低 → 问题不在推理 API，在音频源/麦克风驱动层")
            elif onnx_max < 0.01 and pkg_max >= 0.1:
                print(f"     silero-vad 包 API 显著更高 → ONNX Runtime 调用方式有问题")
            elif onnx_max >= 0.1 and pkg_max < 0.01:
                print(f"     ONNX Runtime 显著更高 → silero-vad 包 API 调用方式有问题")
            else:
                diff = abs(onnx_max - pkg_max)
                if diff < 0.05:
                    print(f"     两种方式结果一致 (diff={diff:.4f}) → 推理 API 正常")
                else:
                    print(f"     两种方式有差异 (diff={diff:.4f}) → 需进一步分析")

        print(f"   {'=' * 60}\n")


def main() -> None:
    args = _parse_args()
    compare = VADModelCompare(
        duration=args.duration,
        method=args.method,
        threshold=args.threshold,
        device=args.device,
        verbose=args.verbose,
    )
    compare.run()


if __name__ == "__main__":
    main()
