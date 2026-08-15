#!/usr/bin/env python3
"""
CosyVoice 模型下载脚本（HF 镜像优先，断点续传可靠）。

背景：modelscope 的 snapshot_download 下载大文件（llm.pt ~2GB）时连接
中断后重试不可靠，导致文件截断（"File is not a zip file"）。改用
huggingface_hub（配 hf-mirror 镜像）下载，它支持断点续传 + 下载后校验。

同时补下载参考音频 zero_shot_prompt.wav（位于 CosyVoice 源码仓库 asset/，
ModelScope 模型仓库不含它）。

用法（Windows PowerShell，项目 .venv 内）：
    # 先设 HF 镜像（国内加速 + 断点续传）
    $env:HF_ENDPOINT="https://hf-mirror.com"
    python scripts/download_cosyvoice.py

    # 指定目录
    python scripts/download_cosyvoice.py --dir D:/models/CosyVoice2-0.5B
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

HF_REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"
PROMPT_WAV_URL = (
    "https://github.com/FunAudioLLM/CosyVoice/raw/main/asset/zero_shot_prompt.wav"
)
DEFAULT_DIR = _PROJECT_ROOT / "pretrained_models" / "CosyVoice2-0.5B"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CosyVoice2-0.5B model (HuggingFace mirror)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DEFAULT_DIR),
        help=f"Download directory (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    return parser.parse_args()


def _download_prompt_wav(download_dir: Path) -> bool:
    """Download the zero-shot reference audio into the model asset dir."""
    prompt = download_dir / "asset" / "zero_shot_prompt.wav"
    if prompt.exists() and prompt.stat().st_size > 100_000:
        print(f"  ✅ 参考音频已存在：{prompt}")
        return True
    prompt.parent.mkdir(parents=True, exist_ok=True)
    print(f"  📥 下载参考音频 zero_shot_prompt.wav ...")
    try:
        urllib.request.urlretrieve(PROMPT_WAV_URL, str(prompt))
    except Exception as e:
        print(f"  ⚠️  参考音频下载失败：{e}")
        print(f"     请手动从 Linux 复制：/home/elysia/project/CosyVoice/asset/zero_shot_prompt.wav")
        print(f"     或浏览器打开：{PROMPT_WAV_URL}")
        return False
    size = prompt.stat().st_size
    print(f"  ✅ 参考音频就绪：{prompt}（{size} 字节）")
    return size > 100_000


def main() -> None:
    args = _parse_args()
    download_dir = Path(args.dir)

    print("\n" + "=" * 60)
    print("  CosyVoice 模型下载器（HF 镜像）")
    print("=" * 60)
    print(f"  模型: {HF_REPO_ID}")
    print(f"  目标: {download_dir}")
    print()

    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "  ❌ huggingface_hub 未安装。\n"
            "     安装：pip install huggingface_hub"
        )
        sys.exit(1)

    endpoint = __import__("os").environ.get("HF_ENDPOINT", "")
    if not endpoint:
        print("  ⚠️  未设置 HF_ENDPOINT，直连 huggingface.co 可能很慢/失败。")
        print("     建议先执行：$env:HF_ENDPOINT=\"https://hf-mirror.com\"\n")
    else:
        print(f"  🔗 使用 HF 镜像：{endpoint}\n")

    print(f"  📦 开始下载 {HF_REPO_ID}（约 3.5GB，支持断点续传）...\n")
    start = time.monotonic()
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,  # Windows 兼容
        )
    except Exception as e:
        print(f"\n  ❌ 下载失败：{e}")
        sys.exit(1)

    elapsed = time.monotonic() - start
    print(f"\n  ✅ 模型下载完成，耗时 {elapsed:.0f}s（{elapsed / 60:.1f} min）")

    # 参考音频（模型仓库不含，需单独下）
    print()
    _download_prompt_wav(download_dir)

    print()
    print("  下一步验证完整性：")
    print("    python scripts/verify_cosyvoice_model.py")
    print()


if __name__ == "__main__":
    main()
