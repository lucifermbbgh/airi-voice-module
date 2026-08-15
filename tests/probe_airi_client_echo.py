"""
用 Voice Module 自己的 AIRIClient 复现「发 input:text → 收 output」。

probe_airi_consumer.py（原始 websockets）已确认能收到 output:gen-ai:chat:message，
但 Voice Module 全模式收不到。本脚本用项目自带的 AIRIClient 走完整流程
（connect + recv loop + handshake + 发 input），逐帧打印收到的每个事件，
定位是 AIRIClient 实现有 bug，还是 main.py 的时序/环境问题。

用法（Windows PowerShell，项目 .venv 内）：
    python tests/probe_airi_client_echo.py [token]

    # 或用环境变量 AIRI_TOKEN
    python tests/probe_airi_client_echo.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from src.airi.websocket_client import AIRIClient


async def main() -> None:
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AIRI_TOKEN", "")
    client = AIRIClient(token=token, name="echo-probe")

    received: list[tuple[str, dict]] = []

    def on_message(data: dict) -> None:
        received.append(("output:gen-ai:chat:message", data))
        msg = data.get("message", {})
        content = msg.get("content") if isinstance(msg, dict) else data.get("content")
        print(f"\n✅ 收到 output:gen-ai:chat:message")
        print(f"   content = {content!r}")

    def on_complete(data: dict) -> None:
        received.append(("output:gen-ai:chat:complete", data))
        print(f"\n✅ 收到 output:gen-ai:chat:complete")

    client.on("output:gen-ai:chat:message", on_message)
    client.on("output:gen-ai:chat:complete", on_complete)

    # 连接 + 启动 recv loop + 握手（复现 run() 的核心流程）
    ok = await client.connect()
    print(f"connect={ok}")
    if not ok:
        return

    recv_task = asyncio.create_task(client._recv_loop())
    await client._start_handshake()

    # 等 ready（复现 main.py 的 20 次 x 0.5s 等待）
    for _ in range(20):
        if client.is_ready:
            break
        await asyncio.sleep(0.5)
    print(f"ready={client.is_ready}")

    # 发 input:text（复现 main.py 的发送路径；AIRIClient.send 只透传
    # type/data/metadata，route 等附加字段会被丢弃——这一点本身也值得注意）
    sent = await client.send({
        "type": "input:text",
        "data": {"text": "你好，请简单介绍一下你自己"},
    })
    print(f"发送 input:text: {sent}")

    # 观察 15s
    for _ in range(15):
        await asyncio.sleep(1)
        if received:
            break

    print(f"\n共收到 output 事件数: {len(received)}")
    recv_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
