# AIRI 语音对话模块 — Phase 4 LLM 对话集成详细实现设计

> **日期**: 2026-08-13
> **版本**: 1.0 (实现完成，Linux 201 测试通过，Windows 单元测试验证通过)
> **对应代码**: `src/airi/conversation.py`、`src/main.py`（`_run_full()`）、`src/airi/websocket_client.py`
> **单元测试**: `tests/test_conversation.py` (21 项)
> **提交**: `2e2e6ae`
> **总体设计**: 见 `PHASE-4-LLM.md`

---

## 目录

1. [总体架构](#一总体架构)
2. [核心数据模型](#二核心数据模型)
3. [Turn 生命周期状态机](#三turn-生命周期状态机)
4. [recent_history — LLM 上下文格式](#四recent_history--llm-上下文格式)
5. [_run_full() 全链路编排](#五_run_full-全链路编排)
6. [六项增强设计](#六六项增强设计)
7. [容错设计](#七容错设计)
8. [测试设计](#八测试设计)

---

## 一、总体架构

Phase 4 在语音链路中桥接「STT 输出」与「AIRI 大脑」：

```
用户语音 → VAD → STT → [Phase 4: ConversationContext + AIRI WS] → LLM → TTS → 扬声器
```

**核心原则**：Voice Module **不直接调用 LLM**，而是通过 AIRI 插件协议（WebSocket）
发送 `input:text:voice` 事件（STT 文字），接收 `output:gen-ai:chat:message` /
`output:gen-ai:chat:complete` 事件（LLM 回复），再喂给 TTS。

---

## 二、核心数据模型

### 2.1 枚举

```python
class TurnRole(str, Enum):
    USER = "user"            # 用户
    ASSISTANT = "assistant"  # AIRI/LLM
    SYSTEM = "system"        # 系统

class TurnStatus(str, Enum):
    PENDING = "pending"          # STT 完成，等待 LLM
    STREAMING = "streaming"      # LLM 正在流式回复
    COMPLETE = "complete"        # 回复完整接收
    INTERRUPTED = "interrupted"  # 用户打断（Phase 5 预埋）
    ERROR = "error"              # 轮次失败
```

### 2.2 `Turn` — 单轮对话

```python
@dataclass
class Turn:
    turn_id: str                       # uuid4().hex[:12]
    role: TurnRole = TurnRole.USER
    text: str = ""                     # 用户话语（或 assistant 回复）
    timestamp: float                   # 轮次开始时间
    confidence: float = 0.0            # STT 置信度（仅 user turn）
    language: str = "zh"               # 检测语言（仅 user turn）
    status: TurnStatus = PENDING
    response_chunks: list[str]         # LLM 流式回复累积
    response_timestamp: float | None   # 回复完成时间
    metadata: dict[str, Any]           # 扩展数据
```

关键属性/方法：

| 成员 | 说明 |
|:-----|:-----|
| `response_text` | 拼接 `response_chunks` 得到完整回复 |
| `duration` | 从话语到回复完成的时长 |
| `append_response(chunk)` | 追加流式 chunk，PENDING→STREAMING |
| `mark_complete()` | STREAMING→COMPLETE，记录 response_timestamp |
| `mark_error(msg)` | →ERROR，记录错误到 metadata |
| `mark_interrupted()` | →INTERRUPTED（Phase 5 预埋） |

### 2.3 `ConversationContext` — 会话管理

```python
@dataclass
class ConversationContext:
    session_id: str        # uuid4().hex[:8]，每次启动生成
    history_limit: int = 20  # 保留最近 20 轮
    turns: list[Turn]      # 有序轮次列表
    created_at: float
```

核心方法：

| 方法 | 说明 |
|:-----|:-----|
| `start_user_turn(text, confidence, language)` | 创建新 user Turn，超限时淘汰最旧轮次 |
| `append_response(chunk)` | 追加到当前活跃 Turn |
| `complete_current_turn()` | 标记当前 Turn 完成 |
| `error_current_turn(msg)` | 标记当前 Turn 错误 |
| `recent_history()` | 返回 LLM 格式的上下文（见下） |
| `last_user_text()` / `turn_count()` / `active_turn()` | 查询接口 |
| `clear()` | 清空会话 |

---

## 三、Turn 生命周期状态机

```
                 start_user_turn()
        ┌────────────────────────────┐
        │         PENDING            │
        └──────────────┬─────────────┘
                       │ append_response(chunk)  ← 收到 output:gen-ai:chat:message
                       ▼
        ┌────────────────────────────┐
        │        STREAMING           │
        └──────┬─────────────┬───────┘
               │             │
   mark_complete()           mark_interrupted()   ← 用户打断（Phase 5）
               │             │
               ▼             ▼
        ┌──────────┐   ┌──────────────┐
        │ COMPLETE │   │ INTERRUPTED  │
        └──────────┘   └──────────────┘

        任意状态 ── mark_error(msg) ──► ERROR
```

状态转换由 `output:gen-ai:chat:message`（流式 chunk）和 `output:gen-ai:chat:complete`
（完成信号）两个 AIRI 事件驱动。

---

## 四、recent_history — LLM 上下文格式

`recent_history()` 返回交替的 user/assistant 消息列表，供 LLM 上下文注入：

```python
def recent_history(self) -> list[dict[str, str]]:
    history = []
    for turn in self.turns[-self.history_limit:]:
        if turn.status == TurnStatus.COMPLETE:
            history.append({"role": "user", "content": turn.text})
            if turn.response_text:
                history.append({"role": "assistant", "content": turn.response_text})
    return history
```

**设计要点**：
- 只包含 COMPLETE 轮次（PENDING/ERROR/INTERRUPTED 不注入）
- 严格交替 user → assistant 顺序
- 最多 `history_limit`（20）轮

---

## 五、_run_full() 全链路编排

`src/main.py` 的 `_run_full()` 是 Phase 4 的核心入口，串联 VAD→STT→AIRI→TTS：

```
_run_full()
  ├─ 初始化 ConversationContext（新 session）
  ├─ 启动 VAD 管线（capture + vad + playback 三协程）
  ├─ 连接 AIRI WebSocket（ws://localhost:10443）
  └─ 事件循环：
       SPEECH_START → 预备打断（tts_mgr.stop()）
       SPEECH_END → STT 转写 → start_user_turn()
                  → pending_sends 队列 → 发送 input:text:voice
       output:gen-ai:chat:message → append_response() → TTS 逐块播放
       output:gen-ai:chat:complete → complete_current_turn()
```

---

## 六、六项增强设计

Phase 4 对 `_run_full()` 的六项增强：

| # | 增强 | 设计 |
|:-:|:-----|:-----|
| 1 | **上下文追踪** | 每次 STT → `start_user_turn()`；LLM 回复 → `append_response()`/`complete_current_turn()` |
| 2 | **STT 异常恢复** | STT 转写包 try/except，失败仅记日志，不中断管线 |
| 3 | **断连缓冲** | STT 结果放入 `asyncio.Queue`（pending_sends），AIRI 重连后自动刷新发送 |
| 4 | **complete 事件** | 处理 `output:gen-ai:chat:complete`，标记 Turn 完成 |
| 5 | **TTS 降级** | TTS 不可用/失败时降级为日志输出，语音模块不崩溃 |
| 6 | **打断预备** | `SPEECH_START` 事件触发 `tts_mgr.stop()`（Phase 5 完整打断） |

---

## 七、容错设计

| 故障场景 | 处理策略 |
|:---------|:---------|
| STT 识别失败 | 捕获异常，记日志，不中断管线 |
| AIRI 断连 | STT 结果入 `pending_sends` 队列，重连后刷新 |
| TTS 不可用 | 降级为日志输出，仍正常运行 |
| TTS 播放错误 | 捕获异常，不阻断后续消息 |

---

## 八、测试设计

### 8.1 测试架构

| 文件 | 测试数 | 说明 |
|:-----|:------:|:-----|
| `tests/test_conversation.py` | 21 | 单元测试（Turn + ConversationContext 全方法覆盖） |

### 8.2 覆盖点

- Turn 生命周期：append_response / mark_complete / mark_error / mark_interrupted
- ConversationContext：start_user_turn / recent_history / 历史淘汰 / clear
- recent_history 格式：严格交替 user/assistant，只含 COMPLETE 轮次
- 上下文独立性验证：`Turns: 2, History: 4`

### 8.3 平台验证

| 平台 | 结果 |
|:-----|:-----|
| Linux | 201 passed, 2 skipped（含 conftest.py 自动修复 PYTHONPATH 污染） |
| Windows | conversation 21/21 + 全量 193 + STT 67/67 + VAD 9/9 |

全链路端到端（`_run_full()` 真实 AIRI 交互）待 Phase 1 VAD 阻塞解决后执行。
