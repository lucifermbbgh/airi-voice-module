"""
STT 模型下载与验证工具

下载并验证 faster-whisper CTranslate2 模型，确认模型文件完整且推理正常。

用法:
    python -m tests.test_stt_download                          # 下载 tiny 模型（默认）
    python -m tests.test_stt_download --model small             # 下载 small 模型
    python -m tests.test_stt_download --model tiny --verify     # 下载并验证推理
    python -m tests.test_stt_download --model tiny --dir D:/models/whisper  # 指定下载目录
    python -m tests.test_stt_download --list-models             # 列出可用模型
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.logger import get_logger, setup_logging

logger = get_logger(__name__)

# 可用模型列表（大小和推荐用途）
AVAILABLE_MODELS = {
    "tiny": {
        "size_mb": 75,
        "ram_mb": 400,
        "rtf": 0.3,
        "description": "最小最快，适合快速验证和测试",
    },
    "base": {
        "size_mb": 150,
        "ram_mb": 500,
        "rtf": 0.3,
        "description": "基础模型，精度略好于 tiny",
    },
    "small": {
        "size_mb": 460,
        "ram_mb": 1000,
        "rtf": 0.3,
        "description": "推荐用于正式使用，精度/速度平衡",
    },
    "medium": {
        "size_mb": 1500,
        "ram_mb": 2500,
        "rtf": 0.4,
        "description": "高精度，需要 ~2.5GB 内存",
    },
    "large-v3": {
        "size_mb": 3000,
        "ram_mb": 3500,
        "rtf": 0.5,
        "description": "最高精度，需要 ~3.5GB 内存",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STT 模型下载与验证 - 下载 faster-whisper 模型并验证推理",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="tiny",
        choices=list(AVAILABLE_MODELS.keys()),
        help="模型大小（默认 tiny）",
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="下载后运行推理验证（使用合成音频）",
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="模型下载目录（默认使用 huggingface 缓存）",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出可用模型信息",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重新下载（即使已缓存）",
    )
    return parser.parse_args()


def _list_models() -> None:
    """打印可用模型列表。"""
    print("\n📦 可用 STT 模型")
    print(f"   {'=' * 60}")
    print(f"   {'名称':<12} {'大小':<8} {'内存':<8} {'推荐用途'}")
    print(f"   {'=' * 60}")
    for name, info in AVAILABLE_MODELS.items():
        size = f"{info['size_mb']}MB"
        ram = f"{info['ram_mb']}MB"
        print(f"   {name:<12} {size:<8} {ram:<8} {info['description']}")
    print(f"   {'=' * 60}")
    print(f"   💡 首次验证推荐: --model tiny（最快）")
    print(f"   💡 正式使用推荐: --model small（精度/速度平衡）")
    print()


def download_model(model_size: str, download_root: str | None = None,
                   force: bool = False) -> str:
    """下载 faster-whisper 模型到本地缓存。

    Args:
        model_size: 模型大小（tiny/base/small/medium/large-v3）。
        download_root: 下载目录，None 使用 huggingface 默认缓存。
        force: 强制重新下载。

    Returns:
        模型缓存路径。

    Raises:
        ImportError: faster-whisper 未安装。
        Exception: 下载失败。
    """
    from faster_whisper import WhisperModel

    model_name = f"Systran/faster-whisper-{model_size}"
    info = AVAILABLE_MODELS[model_size]

    print(f"\n📥 下载模型: {model_name}")
    print(f"   大小约 {info['size_mb']}MB，内存约 {info['ram_mb']}MB")
    if download_root:
        print(f"   下载到: {download_root}")
    else:
        print(f"   下载到: huggingface 默认缓存目录")
    print()

    if force:
        import shutil
        from huggingface_hub import snapshot_download
        # 清除缓存（如果存在）
        try:
            cache_path = snapshot_download(model_name, local_files_only=True)
            shutil.rmtree(cache_path, ignore_errors=True)
            print(f"   🧹 已清除缓存: {cache_path}")
        except Exception:
            pass

    start = time.monotonic()
    model = WhisperModel(
        model_size_or_path=model_name,
        device="cpu",
        compute_type="int8",
        download_root=download_root,
    )
    elapsed = time.monotonic() - start

    print(f"\n✅ 模型下载并加载成功!")
    print(f"   模型: {model_name}")
    print(f"   耗时: {elapsed:.1f} 秒")

    # 获取实际路径
    actual_path = str(model.model_path) if hasattr(model, 'model_path') else model_name
    return actual_path


def verify_inference(model_size: str, download_root: str | None = None) -> None:
    """用合成音频验证模型推理。

    Args:
        model_size: 模型大小。
        download_root: 模型缓存目录。
    """
    import numpy as np
    from faster_whisper import WhisperModel

    model_name = f"Systran/faster-whisper-{model_size}"

    print(f"\n🔬 验证推理: {model_name}")
    print(f"   生成 3 秒合成语音信号...")

    # 生成合成音频（模拟语音的调频信号）
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 200Hz-800Hz 扫频 + 谐波，模拟语音
    audio = 0.3 * np.sin(2 * np.pi * (200 + 600 * t / duration) * t)
    audio += 0.15 * np.sin(2 * np.pi * (400 + 1200 * t / duration) * t)
    audio += 0.05 * np.random.randn(len(t))
    audio = audio.astype(np.float32)

    # 加载模型
    print(f"   加载模型...")
    load_start = time.monotonic()
    model = WhisperModel(
        model_size_or_path=model_name,
        device="cpu",
        compute_type="int8",
        download_root=download_root,
    )
    load_time = time.monotonic() - load_start
    print(f"   模型加载耗时: {load_time:.1f}s")

    # 推理（合成音频应该检测到能量，但不会转成有意义的文字）
    print(f"   推理中...")
    infer_start = time.monotonic()
    segments, info = model.transcribe(audio, language="zh")
    infer_time = time.monotonic() - infer_start

    print(f"\n📊 推理结果:")
    print(f"   推理耗时: {infer_time:.2f}s")
    print(f"   音频时长: {duration:.1f}s")
    print(f"   实时率: {infer_time / duration:.2f}x")
    print(f"   检测到语言: {info.language} (p={info.language_probability:.2f})")

    segments_list = list(segments)
    if segments_list:
        print(f"   识别到 {len(segments_list)} 个片段:")
        for seg in segments_list:
            print(f"     [{seg.start:.1f}s-{seg.end:.1f}s] {seg.text}")
    else:
        print(f"   未识别到文字（合成音频无实际语音内容，属于正常现象）")

    print(f"\n✅ 模型推理验证通过!")


def main() -> None:
    args = _parse_args()

    if args.list_models:
        _list_models()
        return

    # 显示模型信息
    info = AVAILABLE_MODELS[args.model]
    print(f"\n🎯 模型: {args.model}")
    print(f"   大小: ~{info['size_mb']}MB | 内存: ~{info['ram_mb']}MB")
    print(f"   描述: {info['description']}")
    print()

    try:
        # 下载模型
        download_model(args.model, args.dir, args.force)

        # 可选验证
        if args.verify:
            verify_inference(args.model, args.dir)

        print(f"\n✨ 完成！")
        print(f"   下载命令: python -m tests.test_stt_download --model {args.model}")
        if args.verify:
            print(f"   验证命令: python -m tests.test_stt_download --model {args.model} --verify")

    except ImportError as e:
        print(f"\n❌ faster-whisper 未安装: {e}")
        print(f"   安装: pip install faster-whisper")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
