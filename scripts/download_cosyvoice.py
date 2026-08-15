#!/usr/bin/env python3
"""
CosyVoice 模型下载脚本（HTTP Range 断点续传直下）。

不走 huggingface_hub / modelscope 库（那些库对大文件断连后重试不可靠，
导致 llm.pt 截断成 "File is not a zip file"）。直接用 urllib + HTTP
Range 从 modelscope 下载文件，支持断点续传：中断后重跑本脚本，会从
已下载的字节处继续，不会从头重下，也不会截断。

说明：
- spk2info.pt 非必需：cosyvoice/cli/frontend.py 里文件不存在时置空字典
  self.spk2info = {}，运行期 CosyVoice 首次 zero-shot 后会自动 torch.save
  生成。因此本脚本不下载它（modelscope 仓库本身也没有它）。
- 增量下载：已完整文件自动跳过；损坏的 .pt 会自动重下。

用法（Windows PowerShell，项目 .venv 内）：
    python scripts/download_cosyvoice.py

    # 指定目录
    python scripts/download_cosyvoice.py --dir D:/models/CosyVoice2-0.5B
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
DEFAULT_DIR = _PROJECT_ROOT / "pretrained_models" / "CosyVoice2-0.5B"

BASE_URL = "https://modelscope.cn/models/iic/CosyVoice2-0.5B/resolve/master"

# CosyVoice2.__init__ 加载必需的文件（见 cosyvoice/cli/cosyvoice.py）
REQUIRED_FILES = [
    "cosyvoice2.yaml",
    "campplus.onnx",
    "speech_tokenizer_v2.onnx",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "CosyVoice-BlankEN/config.json",
    "CosyVoice-BlankEN/generation_config.json",
    "CosyVoice-BlankEN/merges.txt",
    "CosyVoice-BlankEN/model.safetensors",
    "CosyVoice-BlankEN/tokenizer_config.json",
    "CosyVoice-BlankEN/vocab.json",
]

# 可选加速文件（缺了也能跑，只是慢一点 / 无法走 jit/trt）
OPTIONAL_FILES = [
    "flow.cache.pt",
    "flow.decoder.estimator.fp32.onnx",
    "flow.encoder.fp16.zip",
    "flow.encoder.fp32.zip",
    "speech_tokenizer_v2.batch.onnx",
]

# 需要 zip 完整性校验的 .pt 文件
PT_FILES = {"llm.pt", "flow.pt", "hift.pt", "flow.cache.pt"}

# 参考音频（zero-shot 合成需要；模型仓库不含，从 CosyVoice 源码仓库拿）
PROMPT_WAV_URLS = [
    "https://cdn.jsdelivr.net/gh/FunAudioLLM/CosyVoice@main/asset/zero_shot_prompt.wav",
    "https://github.com/FunAudioLLM/CosyVoice/raw/main/asset/zero_shot_prompt.wav",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download CosyVoice2-0.5B via HTTP Range (resumeable)",
    )
    p.add_argument("--dir", type=str, default=str(DEFAULT_DIR),
                   help=f"下载目录（默认 {DEFAULT_DIR}）")
    p.add_argument("--include-optional", action="store_true",
                   help="同时下载可选加速文件（多 ~1.5GB）")
    return p.parse_args()


def _file_url(path: str) -> str:
    return f"{BASE_URL}/{path}"


def _pt_ok(path: Path) -> bool:
    """用 zipfile 校验 .pt 文件 central directory 是否完整。"""
    try:
        with zipfile.ZipFile(path) as zf:
            zf.namelist()
        return True
    except Exception:
        return False


def _fetch_total_size(url: str, timeout: int = 60) -> int | None:
    """通过 Range 探测文件总大小（bytes=0-0 → Content-Range total）。"""
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cr = resp.headers.get("Content-Range", "")
            if "/" in cr:
                return int(cr.split("/")[-1])
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def _download_file(path: str, dest: Path, is_pt: bool) -> bool:
    """断点续传下载单个文件，返回是否完整。"""
    url = _file_url(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    offset = dest.stat().st_size if dest.exists() else 0

    # 先确认目标总大小
    total = _fetch_total_size(url)
    if total is None:
        print(f"  ⚠️ 无法获取 {path} 的远程大小，跳过（可稍后重跑）")
        return False
    if offset >= total and (not is_pt or _pt_ok(dest)):
        print(f"  ✅ {path}: 已完整，跳过")
        return True

    print(f"  📥 {path}: {total / 1e9:.2f}GB，从 {offset / 1e6:.0f}MB 续传")

    last_report = 0.0
    while offset < total:
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                with open(dest, "ab") as f:
                    while True:
                        chunk = resp.read(1024 * 256)  # 256KB
                        if not chunk:
                            break
                        f.write(chunk)
                        offset += len(chunk)
                        # 每 ~5s 或每 100MB 报一次进度
                        now = time.monotonic()
                        if now - last_report > 5:
                            print(f"      {offset / 1e9:.2f}/{total / 1e9:.2f}GB "
                                  f"({offset * 100 // total}%)")
                            last_report = now
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ 中断：{e}")
            print(f"     已下 {offset / 1e9:.2f}GB，3 秒后从断点续传...")
            time.sleep(3)
            offset = dest.stat().st_size if dest.exists() else 0
            continue

    # 下载完成，校验
    if is_pt and not _pt_ok(dest):
        print(f"  ❌ {path}: 下载完成但 zip 校验失败，删除后重试")
        dest.unlink(missing_ok=True)
        return False
    print(f"  ✅ {path}: 完整（{dest.stat().st_size} 字节）")
    return True


def _download_prompt_wav(dest_dir: Path) -> bool:
    """下载参考音频 zero_shot_prompt.wav。"""
    dest = dest_dir / "asset" / "zero_shot_prompt.wav"
    if dest.exists() and dest.stat().st_size > 300_000:
        print(f"  ✅ 参考音频已存在（{dest.stat().st_size} 字节）")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in PROMPT_WAV_URLS:
        print(f"  📥 下载参考音频：{url}")
        try:
            urllib.request.urlretrieve(url, str(dest))
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ 失败：{e}")
            continue
        if dest.exists() and dest.stat().st_size > 300_000:
            print(f"  ✅ 参考音频就绪（{dest.stat().st_size} 字节）")
            return True
    print("  ❌ 参考音频下载失败，请手动从 Linux 复制：")
    print("     /home/elysia/project/CosyVoice/asset/zero_shot_prompt.wav")
    return False


def main() -> None:
    args = _parse_args()
    dest_dir = Path(args.dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = list(REQUIRED_FILES)
    if args.include_optional:
        files += OPTIONAL_FILES

    print("\n" + "=" * 60)
    print("  CosyVoice2-0.5B 下载（HTTP Range 断点续传）")
    print("=" * 60)
    print(f"  源: {BASE_URL}")
    print(f"  目标: {dest_dir}")
    print(f"  文件数: {len(files)}")
    print()

    start = time.monotonic()
    ok = True
    for f in files:
        if not _download_file(f, dest_dir / f, f in PT_FILES):
            ok = False

    # 参考音频
    print()
    _download_prompt_wav(dest_dir)

    elapsed = time.monotonic() - start
    print("\n" + "=" * 60)
    if ok:
        print(f"  ✅ 全部文件下载完成，耗时 {elapsed / 60:.1f} min")
    else:
        print(f"  ⚠️ 部分文件未完成（{elapsed / 60:.1f} min），直接重跑本脚本续传即可")
    print("=" * 60)
    print("  下一步验证：python scripts/verify_cosyvoice_model.py")
    print()


if __name__ == "__main__":
    main()
