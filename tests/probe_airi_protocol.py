"""
诊断 AIRI WebSocket 插件协议（完整握手流程）。

连 6121 → module:authenticate → 收到 authenticated 后发 module:announce
→ 打印服务器回的所有响应，重点观察 registry:modules:sync 里自己模块的
identity 结构（决定客户端如何认领 ready）。

用法（Windows PowerShell，项目 .venv 内）：
    python tests/probe_airi_protocol.py [token]

    # 或用环境变量 AIRI_TOKEN
    python tests/probe_airi_protocol.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid

import websockets


def _pretty(label: str, raw: str | bytes) -> None:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    print(f"\n=== [{label}] ===")
    print(raw[:1500])


async def main() -> None:
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AIRI_TOKEN", "")
    url = "ws://localhost:6121/ws"
    instance_id = f"probe-{uuid.uuid4().hex[:8]}"

    print(f"连接 {url} | token={'***' if token else '空'} | instance_id={instance_id}")

    async with websockets.connect(url) as ws:
        # ① 发 authenticate
        await ws.send(json.dumps({
            "type": "module:authenticate",
            "data": {"token": token},
            "metadata": {
                "source": {"kind": "plugin", "plugin": {"id": "probe"}, "id": instance_id},
                "event": {"id": "evt-auth"},
            },
        }, ensure_ascii=False))
        print("已发送 module:authenticate")

        # ② 收 authenticated
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            _pretty("收到(1)", raw)
        except asyncio.TimeoutError:
            print("!! 5s 未收到 authenticated 响应")
            return

        # ③ 发 announce
        await ws.send(json.dumps({
            "type": "module:announce",
            "data": {
                "name": "probe",
                "identity": {"kind": "plugin", "plugin": {"id": "probe"}, "id": instance_id},
                "possibleEvents": [],
                "dependencies": [],
            },
            "metadata": {
                "source": {"kind": "plugin", "plugin": {"id": "probe"}, "id": instance_id},
                "event": {"id": "evt-announce"},
            },
        }, ensure_ascii=False))
        print("已发送 module:announce")

        # ④ 收所有响应（8 秒）
        print("等待响应（8s）...")
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=8)
                _pretty("收到", raw)
        except asyncio.TimeoutError:
            print("\n8 秒无更多消息。")


if __name__ == "__main__":
    asyncio.run(main())
