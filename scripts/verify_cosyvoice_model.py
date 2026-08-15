#!/usr/bin/env python3
"""
CosyVoice 模型完整性验证脚本。

CosyVoice2 加载需要（见 cosyvoice/cli/cosyvoice.py CosyVoice2.__init__）：
  cosyvoice2.yaml + llm.pt/flow.pt/hift.pt + campplus.onnx
  + speech_tokenizer_v2.onnx + spk2info.pt + CosyVoice-BlankEN/ 目录

.pt 文件是 zip 归档，用 zipfile 检查 central directory 是否完整，
不需要把 2GB 权重加载进内存，速度很快。配合「删除损坏文件后重跑
download_cosyvoice.py」即可增量重下缺失/损坏的文件。

用法（Windows PowerShell，项目 .venv 内）：
    python scripts/verify_cosyvoice_model.py [模型目录]

    # 默认目录
    python scripts/verify_cosyvoice_model.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
DEFAULT_DIR = _PROJECT_ROOT / "pretrained_models" / "CosyVoice2-0.5B"

# 关键文件：CosyVoice2 加载必需
_PT_FILES = ["llm.pt", "flow.pt", "hift.pt", "spk2info.pt"]
_OTHER_FILES = ["cosyvoice2.yaml", "campplus.onnx", "speech_tokenizer_v2.onnx"]
_DIRS = ["CosyVoice-BlankEN"]


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    print(f"\n验证 CosyVoice 模型目录：{model_dir}\n")
    if not model_dir.is_dir():
        print(f"  ❌ 目录不存在：{model_dir}")
        sys.exit(1)

    all_ok = True

    print("── .pt 权重文件（zip 完整性）──")
    for name in _PT_FILES:
        p = model_dir / name
        if not p.exists():
            print(f"  ❌ {name}: 缺失")
            all_ok = False
            continue
        try:
            with zipfile.ZipFile(p) as zf:
                n = len(zf.namelist())
            print(f"  ✅ {name}: 完整 ({p.stat().st_size / 1e9:.2f}GB, {n} 条目)")
        except Exception as e:
            print(f"  ❌ {name}: 损坏 — {e}")
            all_ok = False

    print("\n── 其他关键文件 ──")
    for name in _OTHER_FILES:
        p = model_dir / name
        if p.exists():
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}: 缺失")
            all_ok = False

    for name in _DIRS:
        p = model_dir / name
        if p.is_dir():
            n = len(list(p.rglob("*")))
            print(f"  ✅ {name}/ ({n} 项)")
        else:
            print(f"  ❌ {name}/: 缺失")
            all_ok = False

    # 参考音频（zero-shot 合成需要）
    prompt = model_dir / "asset" / "zero_shot_prompt.wav"
    print("\n── 参考音频 ──")
    if prompt.exists():
        print(f"  ✅ asset/zero_shot_prompt.wav ({prompt.stat().st_size} 字节)")
    else:
        print(f"  ❌ asset/zero_shot_prompt.wav: 缺失（zero-shot 合成会失败）")
        all_ok = False

    print()
    if all_ok:
        print("  ✅ 模型完整，可以直接运行。")
    else:
        print("  ❌ 存在缺失/损坏文件。修复方法：")
        print("     1. 删除上面标记 ❌ 的文件")
        print("     2. 重跑 python scripts/download_cosyvoice.py（会增量重下缺失文件）")
    print()


if __name__ == "__main__":
    main()
