#!/usr/bin/env python3
"""
CosyVoice 模型下载脚本。

下载 CosyVoice2-0.5B 模型（ModelScope `iic/CosyVoice2-0.5B`）到本地
`pretrained_models/CosyVoice2-0.5B`，供离线 TTS 合成使用。

背景：cosyvoice_tts.load_model 在 model_dir 为空时会回退到
`pretrained_models/CosyVoice2-0.5B`，但该目录不存在时 CosyVoice 会把这个
「本地路径字符串」当成 ModelScope model_id 去在线下载，得到 404
（正确的 model_id 是 `iic/CosyVoice2-0.5B`）。本脚本把模型正确下载到
本地目录，之后在 config/default.yaml 里把 tts.model_dir 指向该目录即可。

用法（Windows PowerShell，项目 .venv 内）：
    python scripts/download_cosyvoice.py

    # 下载到自定义目录
    python scripts/download_cosyvoice.py --dir D:/models/CosyVoice2-0.5B

    # 强制重新下载
    python scripts/download_cosyvoice.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

MODEL_ID = "iic/CosyVoice2-0.5B"
DEFAULT_DIR = _PROJECT_ROOT / "pretrained_models" / "CosyVoice2-0.5B"
# 模型完整性判断的关键文件（缺任一个都视为下载不完整）。
# CosyVoice2 加载需要（见 cosyvoice/cli/cosyvoice.py CosyVoice2.__init__）：
#   cosyvoice2.yaml + llm.pt/flow.pt/hift.pt + campplus.onnx
#   + speech_tokenizer_v2.onnx + spk2info.pt + CosyVoice-BlankEN/ 目录
_REQUIRED_FILES = [
    "cosyvoice2.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v2.onnx",
    "spk2info.pt",
    "CosyVoice-BlankEN",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CosyVoice2-0.5B model from ModelScope",
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
        help="Force re-download even if the model already exists",
    )
    return parser.parse_args()


def _is_complete(download_dir: Path) -> bool:
    """Return True if the model directory looks complete."""
    if not download_dir.exists():
        return False
    present = [f for f in _REQUIRED_FILES if (download_dir / f).exists()]
    return len(present) == len(_REQUIRED_FILES)


def main() -> None:
    args = _parse_args()
    download_dir = Path(args.dir)

    print("\n" + "=" * 60)
    print("  CosyVoice 模型下载器")
    print("=" * 60)
    print(f"  模型: {MODEL_ID}")
    print(f"  目标: {download_dir}")
    print()

    if _is_complete(download_dir) and not args.force:
        print(f"  ✅ 模型已存在且完整：{download_dir}")
        print(f"     （用 --force 重新下载）")
        print()
        return

    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ImportError:
        print(
            "  ❌ modelscope 未安装。\n"
            "     安装：pip install modelscope\n"
            "     （cosyvoice 依赖它，通常已随 cosyvoice 安装）"
        )
        sys.exit(1)

    print(f"  📦 开始下载 {MODEL_ID}（约 2GB，请耐心等待）...")
    print(f"     ModelScope 国内直连，一般不需要代理。\n")

    start = time.monotonic()
    try:
        snapshot_download(
            MODEL_ID,
            local_dir=str(download_dir),
        )
    except Exception as e:
        print(f"\n  ❌ 下载失败：{e}")
        print(f"     提示：若因网络问题失败，可尝试手动：")
        print(f"     git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git \"{download_dir}\"")
        sys.exit(1)

    elapsed = time.monotonic() - start
    print(f"\n  ✅ 下载完成，耗时 {elapsed:.0f}s（{elapsed / 60:.1f} min）")

    if not _is_complete(download_dir):
        print(f"  ⚠️  警告：关键文件缺失，下载可能不完整。请用 --force 重试。")
        sys.exit(1)

    print(f"  ✅ 模型就绪：{download_dir}")
    print()
    print("  接下来在 config/default.yaml 里设置：")
    print(f"    tts:")
    print(f"      model_dir: \"{download_dir.as_posix()}\"")
    print()


if __name__ == "__main__":
    main()
