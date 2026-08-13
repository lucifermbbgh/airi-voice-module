#!/usr/bin/env python3
"""CosyVoice2 合成诊断脚本（Windows 端）。

用途
====
用 CosyVoice 官方 ``AutoModel`` API 直接合成一段测试音频，用于诊断
"TTS 合成声音异常（非正常语言）"问题。

此脚本**完全绕过**项目自研的 TTS 模块（``src/tts/cosyvoice_tts.py``），
直接调用官方 API。用于二分定位问题根源：

* 若本脚本合成的音频正常 → 问题在项目自研 TTS 模块
* 若本脚本合成的音频也乱 → 问题在模型/环境（依赖版本等）

用法
====
.. code-block:: powershell

    python tests/diagnose_cosyvoice.py
    python tests/diagnose_cosyvoice.py --text "你好，我是AIRI。"
    python tests/diagnose_cosyvoice.py --model-dir "D:\\DevProject\\PythonProject\\CosyVoice\\pretrained_models\\CosyVoice2-0.5B"

输出
====
在项目根目录生成 ``diagnose_output.wav``，用系统播放器双击即可听内容。

判断标准
--------
* 听到**正常中文** → 项目自研 TTS 模块有 bug
* 听到**奇怪语言/无法识别的人声** → 模型或依赖版本问题
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 官方 requirements.txt 要求的关键依赖版本（用于对比）
# 来源：CosyVoice 仓库根目录 requirements.txt
# ---------------------------------------------------------------------------
OFFICIAL_VERSIONS = {
    "x-transformers": "2.11.24",   # flow 模块 DiT 的 RotaryEmbedding 核心，版本敏感
    "wetext": "0.0.4",             # 中文文本归一化前端，版本敏感
    "transformers": "4.51.3",      # Qwen2 LLM 加载，版本敏感
    "diffusers": "0.29.0",         # Matcha-TTS 依赖
    "lightning": "2.2.4",          # Matcha-TTS 依赖
    "librosa": "0.10.2",           # Matcha-TTS 音频处理
    "onnxruntime": "1.18.0",       # speech_tokenizer / campplus 推理
    "openai-whisper": "20231117",  # prompt_wav 的 mel 谱提取
    "conformer": "0.3.2",          # Matcha-TTS 依赖
}


def check_dependencies() -> None:
    """检查关键依赖的实际版本，并与官方要求对比。"""
    print("=" * 60)
    print(" 依赖版本检查")
    print("=" * 60)

    mismatches = []
    for pkg, official in OFFICIAL_VERSIONS.items():
        try:
            actual = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            print(f"  ❌ {pkg:<18} 未安装（官方要求 {official}）")
            mismatches.append(pkg)
            continue

        # 对比主版本（去掉 +cu124 这类本地后缀）
        actual_clean = actual.split("+")[0]
        status = "✅" if actual_clean == official else "⚠️"
        if actual_clean != official:
            mismatches.append(pkg)
        print(f"  {status} {pkg:<18} 实际 {actual:<12} 官方要求 {official}")

    if mismatches:
        print()
        print(f"  ⚠️  有 {len(mismatches)} 个依赖版本不匹配，可能导致合成异常：")
        print(f"     {', '.join(mismatches)}")
        print("     建议用 pip install <包名>==<官方版本> 逐个降级验证。")
    else:
        print()
        print("  ✅ 所有关键依赖版本与官方要求一致。")
    print()


def synthesize_test(model_dir: str, text: str,
                    prompt_wav: str, prompt_text: str,
                    output_path: str) -> bool:
    """用官方 AutoModel 合成一段测试音频。

    Args:
        model_dir: CosyVoice2 模型目录。
        text: 要合成的文本。
        prompt_wav: 参考音频路径（zero_shot_prompt.wav）。
        prompt_text: 参考音频对应的文字。
        output_path: 输出 WAV 文件路径。

    Returns:
        True 表示合成成功。
    """
    print("=" * 60)
    print(" 官方 API 合成测试")
    print("=" * 60)
    print(f"  文本: {text}")
    print(f"  参考音频: {prompt_wav}")
    print(f"  模型目录: {model_dir}")
    print()

    # 延迟导入：CosyVoice 依赖较重，且需要 .pth 路径注入
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
        import torchaudio
    except ImportError as e:
        print(f"  ❌ 导入 CosyVoice 失败: {e}")
        print("     请确认 cosyvoice.pth 已注入 site-packages。")
        return False

    print("  加载模型（首次约 15-30 秒）...")
    cosyvoice = AutoModel(model_dir=model_dir)

    # 用官方 API 合成（generator，逐个 chunk 保存）
    chunks = []
    for output in cosyvoice.inference_zero_shot(
        text,
        prompt_text=prompt_text,
        prompt_wav=prompt_wav,
    ):
        speech = output["tts_speech"]
        chunks.append(speech)
        duration = speech.shape[1] / cosyvoice.sample_rate
        print(f"    合成 chunk: {duration:.2f}s audio, "
              f"sample_rate={cosyvoice.sample_rate}")

    if not chunks:
        print("  ❌ 未生成任何音频。")
        return False

    # 拼接所有 chunk 并保存
    import torch
    full = torch.cat(chunks, dim=1)
    total = full.shape[1] / cosyvoice.sample_rate
    torchaudio.save(output_path, full, cosyvoice.sample_rate)
    print()
    print(f"  ✅ 已保存: {output_path}")
    print(f"     总时长: {total:.2f}s")
    print(f"     采样率: {cosyvoice.sample_rate} Hz")
    print()
    print("  ▶  请用系统播放器双击该 WAV 文件，判断声音是否正常：")
    print("     - 正常中文 → 项目自研 TTS 模块有 bug")
    print("     - 奇怪语言 → 模型/依赖版本问题")
    return True


def main() -> int:
    """主流程：检查依赖 → 官方 API 合成测试。"""
    parser = argparse.ArgumentParser(
        description="CosyVoice2 合成诊断脚本（绕过项目 TTS 模块，直接用官方 API）",
    )
    parser.add_argument(
        "--model-dir", "-m",
        type=str,
        default=os.environ.get(
            "TTS_MODEL_DIR",
            r"D:\DevProject\PythonProject\CosyVoice\pretrained_models\CosyVoice2-0.5B",
        ),
        help="CosyVoice2 模型目录（默认读 TTS_MODEL_DIR 环境变量）",
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default="你好，我是AIRI，你的智能语音助手。",
        help="要合成的测试文本",
    )
    parser.add_argument(
        "--prompt-wav",
        type=str,
        default=r"D:\DevProject\PythonProject\CosyVoice\asset\zero_shot_prompt.wav",
        help="零样本参考音频路径",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="希望你以后能够做的比我还好呦。",
        help="参考音频对应的文字（必须与 prompt-wav 内容一致）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="diagnose_output.wav",
        help="输出 WAV 文件路径",
    )
    args = parser.parse_args()

    check_dependencies()

    ok = synthesize_test(
        model_dir=args.model_dir,
        text=args.text,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        output_path=args.output,
    )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
