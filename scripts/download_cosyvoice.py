#!/usr/bin/env python3
"""
CosyVoice 模型下载脚本（HF 镜像，断点续传可靠）。

背景：modelscope 的 snapshot_download 下载大文件（llm.pt ~2GB）时连接
中断后重试不可靠，导致文件截断（"File is not a zip file"）。改用
huggingface_hub（配 hf-mirror 镜像）下载，它支持断点续传 + 下载后校验。

镜像地址硬编码在脚本里，无需手动设 HF_ENDPOINT（避免 PowerShell
复制粘贴时带入 @url:` 等格式污染）。

同时补下载参考音频 zero_shot_prompt.wav（位于 CosyVoice 源码仓库 asset/，
模型仓库不含它）。

用法（Windows PowerShell，项目 .venv 内）：
    python scripts/download_cosyvoice.py

    # 指定目录 / 强制重下
    python scripts/download_cosyvoice.py --dir D:/models/CosyVoice2-0.5B --force
"""

from __future__ import annotations

import os

# 关键：必须在 import huggingface_hub 之前设置，因为它在 import 时读取
# 端点常量。同时设新旧两个变量，兼容 huggingface_hub 0.x 与 1.x。
_HF_MIRROR = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = _HF_MIRROR          # 旧版变量名
os.environ["HF_HUB_ENDPOINT"] = _HF_MIRROR      # 1.x 版变量名

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

# 损坏/需重下的核心权重（--force 时删除，保证重新下载而非复用截断文件）
_CORE_PT_FILES = ("llm.pt", "flow.pt", "hift.pt", "spk2info.pt", "flow.cache.pt")


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
        help="Delete existing .pt weight files before re-downloading",
    )
    return parser.parse_args()


def _purge_core_files(download_dir: Path) -> None:
    """删除核心权重文件，确保 --force 时重新下载而非复用截断文件。"""
    for name in _CORE_PT_FILES:
        p = download_dir / name
        if p.exists():
            p.unlink()
            print(f"  🗑️  已删除旧文件：{name}")


def _download_prompt_wav(download_dir: Path) -> bool:
    """Download the zero-shot reference audio into the model asset dir."""
    prompt = download_dir / "asset" / "zero_shot_prompt.wav"
    if prompt.exists() and prompt.stat().st_size > 100_000:
        print(f"  ✅ 参考音频已存在：{prompt}")
        return True
    prompt.parent.mkdir(parents=True, exist_ok=True)
    print("  📥 下载参考音频 zero_shot_prompt.wav ...")
    try:
        urllib.request.urlretrieve(PROMPT_WAV_URL, str(prompt))
    except Exception as e:
        print(f"  ⚠️  参考音频下载失败：{e}")
        print(
            "     请手动从 Linux 复制：/home/elysia/project/CosyVoice/asset/zero_shot_prompt.wav\n"
            f"     或浏览器打开：{PROMPT_WAV_URL}"
        )
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
    print(f"  镜像: {_HF_MIRROR}")
    print()

    download_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        _purge_core_files(download_dir)
        print()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "  ❌ huggingface_hub 未安装。\n"
            "     安装：pip install huggingface_hub"
        )
        sys.exit(1)

    print(f"  📦 开始下载 {HF_REPO_ID}（约 3.5GB，支持断点续传）...\n")
    start = time.monotonic()
    try:
        # 注意：huggingface_hub 1.x 已废弃 local_dir_use_symlinks，
        # 只传 local_dir 即可（自动按本地目录模式下载，不用 symlink）。
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=str(download_dir),
        )
    except Exception as e:
        print(f"\n  ❌ 下载失败：{e}")
        print("\n  若仍连不上镜像，可尝试用 git 克隆（LFS 续传）：")
        print("    git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git "
              "pretrained_models\\CosyVoice2-0.5B")
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
