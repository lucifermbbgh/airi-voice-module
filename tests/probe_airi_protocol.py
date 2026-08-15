"""
诊断 AIRI WebSocket 插件协议：连接 → 发 module:authenticate → 打印所有原始响应。

用途：确定 AIRI 服务器对 module:authenticate 的真实响应，以及消息是否为
superjson 编码（这决定了客户端接收端的解析逻辑）。

用法（Windows PowerShell，在项目 .venv 内）：
    python tests/probe_airi_protocol.py [token]

    # 或者用已设置的环境变量 AIRI_TOKEN：
    python tests/probe_airi_protocol.py

预期观察：
    1. 连接后服务器是否「主动」发 module:authenticated（说明服务器未配 token，
       在 open 阶段就放行）
    2. 发出 module:authenticate 后，服务器回 module:authenticated 还是 error
    3. 原始消息是否是 superjson 包装（{"json": ..., "meta": ...}）
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import websockets


def _dump(label: str, raw: str | bytes) -> None:
    """Print a raw message and attempt to parse it."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    print(f"\n[{label}] 原始消息（前 600 字符）:")
    print(f"  {raw[:600]}")

    try:
        parsed = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        print(f"  [JSON 解析失败] {e}")
        return

    # superjson 包装检测
    if isinstance(parsed, dict) and "json" in parsed and "meta" in parsed:
        print(f"  [superjson 包装] 内层 type={parsed['json'].get('type')}")
        print(f"  [superjson 内层] data={parsed['json'].get('data')}")
        print(f"  [superjson meta] values keys={list(parsed.get('meta', {}).get('values', {}).keys())}")
    else:
        print(f"  [标准 JSON] type={parsed.get('type')}")
        print(f"  [标准 JSON] data={parsed.get('data')}")
        print(f"  [标准 JSON] metadata={parsed.get('metadata')}")


async def main() -> None:
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AIRI_TOKEN", "")
    url = "ws://localhost:6121/ws"

    print(f"连接 {url}")
    print(f"token: {'已提供 (长度 ' + str(len(token)) + ')' if token else '空'}")

    async with websockets.connect(url) as ws:
        print("✓ 已连接，等待 1.5s 观察服务器 open 阶段是否主动发消息...")

        # 先观察 open 阶段服务器是否主动发 authenticated（服务器未配 token 时会）
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
                _dump("open 阶段收到", raw)
        except asyncio.TimeoutError:
            print("  （1.5s 内服务器未主动发消息 → 服务器配了 token，等待客户端认证）")

        # 发送 module:authenticate
        msg = {
            "type": "module:authenticate",
            "data": {"token": token},
            "metadata": {
                "source": {"kind": "plugin", "plugin": {"id": "probe"}, "id": "probe-1"},
                "event": {"id": "probe-event-1"},
            },
        }
        await ws.send(json.dumps(msg, ensure_ascii=False))
        print(f"\n已发送 module:authenticate（token={'***' if token else '空'}）")

        # 接收响应（8 秒）
        print("等待服务器响应（8s）...")
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=8)
                _dump("收到响应", raw)
        except asyncio.TimeoutError:
            print("\n8 秒内无更多消息。")


if __name__ == "__main__":
    asyncio.run(main())
