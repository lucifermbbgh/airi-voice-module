"""
诊断 AIRI chat-ingestion consumer 是否注册。

核心原理（AIRI 0.11.3 源码逆向结论）：
- `input:text` / `input:text:voice` 事件的 delivery 是
  `consumer-group: chat-ingestion, selection: first`（plugin-protocol events.ts）。
- server-runtime 对这类事件：无 consumer 时若 `delivery.required` 未设则静默丢弃
  （server/index.ts 第 899-905 行），设了 required 则回 error `noConsumerRegistered`。
- 只有 AIRI **主窗口**（虚拟形象窗口，路由 `/`）会初始化 context-bridge 并
  注册 chat-ingestion consumer；**聊天窗口**（路由 `/chat`，isAuxiliaryChatRoute）
  特意不注册（apps/stage-tamagotchi/src/renderer/App.vue 第 142/244 行）。

本脚本给 input:text 附加 `route.delivery.required=true`，强制 server 在无 consumer
时回 error，从而明确区分三种结果：
  1. 收到 error（noConsumerRegistered）→ 没有窗口注册 consumer
     （多半是打开了 /chat 聊天窗口，而不是主窗口/虚拟形象窗口）
  2. 收到 output:gen-ai:chat:message → consumer 已注册且正常回复 ✅
  3. 无任何响应 → consumer 已注册但未回复（activeProvider/activeModel 为空等）

用法（Windows PowerShell，项目 .venv 内）：
    python tests/probe_airi_consumer.py [token]

    # 或用环境变量 AIRI_TOKEN
    python tests/probe_airi_consumer.py
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
    try:
        parsed = json.loads(raw)
        inner = parsed.get("json", parsed) if isinstance(parsed, dict) else parsed
        t = inner.get("type")
        d = inner.get("data", {})
        if t == "error":
            print(f"\n[{label}] ⛔ 收到 error: {d!r}")
        elif t == "output:gen-ai:chat:message":
            msg = d.get("message", {})
            content = msg.get("content") if isinstance(msg, dict) else d.get("content")
            print(f"\n[{label}] ✅ 收到 output:gen-ai:chat:message")
            print(f"  content = {content!r}")
        elif t == "output:gen-ai:chat:complete":
            print(f"\n[{label}] ✅ 收到 output:gen-ai:chat:complete")
        else:
            print(f"\n[{label}] 收到 type={t} data_keys={list(d.keys()) if isinstance(d, dict) else None}")
    except Exception:
        print(f"\n[{label}] 原始: {raw[:300]}")


async def main() -> None:
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AIRI_TOKEN", "")
    url = "ws://localhost:6121/ws"
    name = "probe-consumer"
    instance_id = f"{name}-{uuid.uuid4().hex[:8]}"
    identity = {"id": instance_id, "extension": {"id": name}}

    print(f"连接 {url} | token={'***' if token else '空'}")
    print("（请先确认 AIRI 桌面已启动）\n")

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
            _pretty("认证", raw)
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

        # ④ 等 announced / sync（任取一帧，握手即完成）
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            _pretty("announce", raw)
        except asyncio.TimeoutError:
            print("!! announce 无响应")

        # ⑤ 发 input:text，带 route.delivery.required=true（关键差异）
        print("\n===== 发 input:text（required=true）=====")
        await send({
            "type": "input:text",
            "data": {"text": "你好，请简单介绍一下你自己"},
            "route": {"delivery": {"required": True}},
        })
        print("已发 input:text（required=true），等待 12s 观察...\n")
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=12)
                _pretty("响应", raw)
        except asyncio.TimeoutError:
            print("\n（12s 无更多消息）")

    print("\n========== 结论判定 ==========")
    print("若上面收到 error（noConsumerRegistered）→ 没有窗口注册 consumer：")
    print("   请打开 AIRI 的【主窗口/虚拟形象窗口】（不是 /chat 聊天窗口）再测。")
    print("若收到 output:gen-ai:chat:message → consumer 正常，链路已通 ✅")
    print("若无任何响应 → consumer 已注册但未回复（检查 LLM provider/model 是否激活）。")


if __name__ == "__main__":
    asyncio.run(main())
