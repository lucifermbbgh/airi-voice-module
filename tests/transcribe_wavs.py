#!/usr/bin/env python3
"""用 faster-whisper 识别 Windows 端生成的 TTS 音频，判断内容是否正常。

用途
====
AIRI 语音模块调试期间，Windows 端 CosyVoice2 合成的音频"无法识别人声"。
本脚本用本地 faster-whisper（Hermes venv 环境）转写这些 WAV 文件，
客观判断合成内容是否正常。

用法
====
    /home/elysia/.hermes/hermes-agent/venv/bin/python3 tests/transcribe_wavs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from faster_whisper import WhisperModel

# 要识别的 WAV 文件（Windows 端生成）
TARGETS = [
    "diagnose_output.wav",        # 官方 AutoModel 合成（长文本 20 字）
    "zs.wav",                     # 官方 AutoModel 合成（短文本 9 字）
    "output/tts_windows_test_1.wav",  # 项目代码合成（中文 20 字）
    "output/tts_windows_test_2.wav",  # 项目代码合成（中文 15 字）
    "output/tts_windows_test_3.wav",  # 项目代码合成（英文）
]


def main() -> int:
    print("下载 base 模型（约 145MB，首次需走代理）...")
    # base 模型中文识别精度足够判断"是否正常中文"
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("模型加载完成\n")

    for f in TARGETS:
        p = Path(f)
        if not p.exists():
            print(f"[缺失] {f}")
            continue
        print(f"[识别] {f}")
        segments, info = model.transcribe(
            str(p), language="zh", beam_size=5,
        )
        text = "".join(seg.text for seg in segments).strip()
        print(f"    语言概率: {info.language_probability:.3f}")
        print(f"    转写结果: {text!r}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
