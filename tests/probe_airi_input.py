"""
诊断 AIRI 输入事件：对比 input:text（打字）与 input:text:voice（语音）两条路径。

完成握手后依次发送两种输入事件，观察哪个能触发 output:gen-ai:chat:message，
从而定位「语音输入不触发回复」是 transcription 字段问题还是路由问题。

用法（Windows PowerShell，项目 .venv 内）：
    python tests/probe_airi_input.py [token]

    # 或用环境变量 AIRI_TOKEN
    python tests/probe_airi_input.py
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
    # 只看 type 和 data 的 message 部分，避免刷屏
    try:
        parsed = json.loads(raw)
        inner = parsed.get("json", parsed) if isinstance(parsed, dict) else parsed
        t = inner.get("type")
        d = inner.get("data", {})
        if t == "output:gen-ai:chat:message":
            msg = d.get("message", {})
            content = msg.get("content") if isinstance(msg, dict) else None
            print(f"\n[{label}] 收到 output:gen-ai:chat:message")
            print(f"  content = {content!r}")
        elif t == "output:gen-ai:chat:complete":
            print(f"\n[{label}] 收到 output:gen-ai:chat:complete")
        else:
            print(f"\n[{label}] 收到 type={t} data_keys={list(d.keys()) if isinstance(d, dict) else None}")
    except Exception:
        print(f"\n[{label}] 原始: {raw[:300]}")


async def main() -> None:
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AIRI_TOKEN", "")
    url = "ws://localhost:6121/ws"
    name = "probe-input"
    instance_id = f"{name}-{uuid.uuid4().hex[:8]}"
    identity = {"id": instance_id, "extension": {"id": name}}

    print(f"连接 {url} | token={'***' if token else '空'}")

    async with websockets.connect(url) as ws:
        async def send(payload: dict) -> None:
            payload.setdefault("metadata", {
                "source": {"kind": "plugin", "id": instance_id,
                           "extension": {"id": name}, "plugin": {"id": name}},
                "event": {"id": uuid.uuid4().hex[:16]},
            })
            await ws.send(json.dumps(payload, ensure_ascii=False))

        # ① authenticate
        await send({"type": "module:authenticate", "data": {"token": token}})
        print("已发 module:authenticate")

        # ② 等 authenticated
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            _pretty("认证响应", raw)
        except asyncio.TimeoutError:
            print("!! 认证无响应")
            return

        # ③ announce
        await send({
            "type": "extension:module:announce",
            "data": {"name": name, "identity": identity,
                     "possibleEvents": [], "dependencies": []},
        })
        print("已发 extension:module:announce")

        # ④ 等 announced
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            _pretty("announce 响应", raw)
        except asyncio.TimeoutError:
            print("!! announce 无响应")

        # ⑤ 发 input:text（模拟打字）
        print("\n===== 测试 1: input:text（打字） =====")
        await send({
            "type": "input:text",
            "data": {"text": "你好，请简单介绍一下你自己"},
        })
        print("已发 input:text，等待 12s 观察回复...")
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=12)
                _pretty("打字路径", raw)
        except asyncio.TimeoutError:
            print("（12s 无更多消息）")

        # ⑥ 发 input:text:voice（模拟语音）
        print("\n===== 测试 2: input:text:voice（语音） =====")
        await send({
            "type": "input:text:voice",
            "data": {"transcription": "你好，请简单介绍一下你自己"},
        })
        print("已发 input:text:voice，等待 12s 观察回复...")
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=12)
                _pretty("语音路径", raw)
        except asyncio.TimeoutError:
            print("（12s 无更多消息）")


if __name__ == "__main__":
    asyncio.run(main())
