#!/usr/bin/env python3
"""诊断 CosyVoice 模型下载失败根因：HF 端点是否生效 + 镜像可达性。

在 Windows 项目 .venv 内运行：
    python tests/probe_hf_endpoint.py
"""
from __future__ import annotations

import os
import socket
import urllib.request

# 1) 先设环境变量（与 download_cosyvoice.py 一致）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"


def _check_url(name: str, url: str, proxy: str | None = None) -> None:
    """用 urllib 直接探 URL，可选走代理。"""
    print(f"\n[探] {name}: {url}")
    if proxy:
        print(f"     经代理 {proxy}")
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            if proxy
            else urllib.request.ProxyHandler({})
        )
        req = urllib.request.Request(url, method="GET")
        resp = opener.open(req, timeout=15)
        # 只读前 512 字节，避免拉大文件
        body = resp.read(512)
        print(f"     ✅ 可达  HTTP {resp.status}  ({len(body)} bytes 预览)")
    except Exception as e:  # noqa: BLE001
        print(f"     ❌ 失败  {type(e).__name__}: {e}")


def main() -> None:
    print("=" * 60)
    print("  HF 端点诊断")
    print("=" * 60)

    # 2) 检查 huggingface_hub 版本与端点常量
    try:
        import huggingface_hub
        print(f"\nhuggingface_hub 版本: {huggingface_hub.__version__}")
        print(f"constants.ENDPOINT   : {huggingface_hub.constants.ENDPOINT}")
        print(f"constants.HF_HUB_*   : "
              f"{getattr(huggingface_hub.constants, 'HF_HUB_ENDPOINT', 'N/A')}")
    except ImportError:
        print("\n❌ huggingface_hub 未安装")
        return

    # 3) 探测 hf-mirror.com 直连（不代理）
    _check_url(
        "hf-mirror 根路径",
        "https://hf-mirror.com",
    )
    _check_url(
        "hf-mirror API（模型元数据）",
        "https://hf-mirror.com/api/models/FunAudioLLM/CosyVoice2-0.5B",
    )

    # 4) 探测 hf-mirror.com 走本地代理（xray 10809）
    _check_url(
        "hf-mirror 根路径（代理 10809）",
        "https://hf-mirror.com",
        proxy="http://127.0.0.1:10809",
    )

    # 5) 对照：huggingface.co 直连（国内通常失败）
    _check_url(
        "huggingface.co 官方（对照）",
        "https://huggingface.co/api/models/FunAudioLLM/CosyVoice2-0.5B",
    )

    print("\n" + "=" * 60)
    print("  诊断结论提示：")
    print("  - 若 ENDPOINT 不是 https://hf-mirror.com → 环境变量没吃进去")
    print("  - 若 hf-mirror 直连 ❌ 但代理 ✅ → 下载需走代理，")
    print("    脚本里加 proxies 或设 HTTP(S)_PROXY 环境变量")
    print("  - 若 hf-mirror 直连 ✅ → 端点配置问题，改用 endpoint= 参数")
    print("=" * 60)


if __name__ == "__main__":
    main()
