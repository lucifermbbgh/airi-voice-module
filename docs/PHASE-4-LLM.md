# AIRI 语音对话模块 — Phase 4 设计：LLM 对话集成

> **日期**: 2026-08-11  
> **状态**: 实现阶段  
> **依赖**: Phase 1 (VAD) + Phase 2 (STT) + Phase 3 (TTS)  
> **目标平台**: Windows 11 (AIRI 主机) / Linux (开发测试)

---

## 一、架构概览

### Phase 4 在全链路中的位置

```
[ 麦克风 ] ─→ [ VAD ] ─→ [ STT ] ─→ [ AIRI WS ] ─→ [ LLM ] ─→ [ TTS ] ─→ [ 扬声器 ]
   Phase 1      Phase 1     Phase 2       Phase 4       Phase 4    Phase 3      Phase 1
                                           ↑
                                     您在这里 👈
```

### 数据流

```
用户语音 → VAD检测 → STT转文字 → ConversationContext → AIRI WebSocket
                                                              │
                                              ┌───────────────┘
                                              ▼
                                     AIRI LLM (DeepSeek)
                                              │
                                              ▼
                                   output:gen-ai:chat:message
                                              │
                              ┌───────────────┼───────────────┐
                              ▼                               ▼
                     ConversationContext              TTS → 扬声器
                     (记录响应文本)                   (语音播放)
                              │
                              ▼
                   output:gen-ai:chat:complete
                              │
                              ▼
                     Turn.mark_complete()
```

---

## 二、设计原则

### 2.1 Voice Module 不直接调用 LLM

Voice Module 通过 **AIRI 插件协议**（WebSocket）与 AIRI 通信，LLM 调用由 AIRI 内部完成：

| 方向 | 事件类型 | 内容 | 使用场景 |
|------|---------|------|----------|
| Voice→AIRI | `input:text:voice` | STT 转文字结果 + turn_id | 每次用户说完话 |
| AIRI→Voice | `output:gen-ai:chat:message` | LLM 流式回复文本块 | TTS 逐块播放 |
| AIRI→Voice | `output:gen-ai:chat:complete` | LLM 回复完成信号 | 标记对话轮次完成 |

### 2.2 对话上下文管理

新增 `ConversationContext` 模块，维护多轮对话状态：

- **Turn 粒度追踪**：每次用户说话 → 创建 Turn，LLM 回复 → 追加 response_chunks，complete 事件 → 标记完成
- **多轮历史**：保留最近 20 轮对话，支持 LLM 上下文注入（Phase 5 打断后复用）
- **Session 隔离**：每次启动生成新 session_id

### 2.3 容错设计

| 故障场景 | 处理策略 |
|---------|---------|
| STT 识别失败 | 捕获异常，日志记录，不中断管线 |
| AIRI 断连 | STT 结果放入 `pending_sends` 队列，重连后自动刷新 |
| TTS 不可用 | 降级为日志输出，语音模块仍正常运行 |
| TTS 播放错误 | 捕获异常，日志记录，不阻断后续消息 |

---

## 三、模块设计

### 3.1 ConversationContext (`src/airi/conversation.py`)

核心数据结构：

```python
ConversationContext
├── session_id: str          # 会话唯一标识
├── history_limit: int = 20  # 最大保留轮次
├── turns: list[Turn]        # 对话轮次列表
│
├── start_user_turn(text, confidence, language) → Turn
├── append_response(chunk) → Turn | None
├── complete_current_turn() → Turn | None
├── error_current_turn(error) → Turn | None
│
├── recent_history → list[dict]  # LLM 上下文格式
├── last_user_text → str | None
├── active_turn → Turn | None
└── summary() → dict

Turn
├── turn_id: str             # 轮次唯一标识
├── role: TurnRole           # user / assistant / system
├── text: str                # 用户话语或助手回复
├── confidence: float        # STT 置信度（用户轮次）
├── status: TurnStatus       # pending / streaming / complete / interrupted / error
├── response_chunks: list    # LLM 流式回复片段
└── metadata: dict           # 扩展元数据
```

### 3.2 增强的 `_run_full()` (`src/main.py`)

Phase 4 在原有骨架上的 6 项改进：

| # | 改进 | 原代码 | Phase 4 |
|---|------|--------|---------|
| 1 | **对话上下文** | 无 | `ConversationContext` 追踪每轮对话 |
| 2 | **STT 异常恢复** | 异常导致回调崩溃 | `try/except` + 日志 |
| 3 | **断连缓冲** | STT 结果直接丢弃 | `asyncio.Queue` 缓冲，重连后刷新 |
| 4 | **complete 事件** | 未处理 | 标记 Turn 完成，记录统计 |
| 5 | **TTS 降级** | TTS 不可用时无反馈 | 日志输出 AIRI 回复文本 |
| 6 | **打断预备** | 无 | SPEECH_START 时调用 `tts_mgr.stop()` |

---

## 四、配置扩展

### `config/default.yaml` 无需变更

Phase 4 复用现有 `airi:` 配置段，无需新增配置项。会话参数（history_limit=20）硬编码为合理默认值，后续可按需暴露为配置。

---

## 五、新增文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/airi/conversation.py` | ~220 | 对话上下文管理（Turn + ConversationContext） |
| `src/airi/__init__.py` | 更新 | 导出 ConversationContext 等新类 |
| `src/main.py` | 更新 | `_run_full()` 集成上下文 + 错误恢复 |
| `tests/test_conversation.py` | 新增 | 对话上下文单元测试 |
| `docs/PHASE-4-LLM.md` | 本文件 | Phase 4 设计文档 |

---

## 六、测试策略

### 单元测试 (`tests/test_conversation.py`)

| 测试用例 | 覆盖 |
|---------|------|
| `test_start_user_turn` | 创建用户轮次 |
| `test_append_response` | 流式追加 LLM 回复 |
| `test_complete_turn` | 标记轮次完成 |
| `test_recent_history` | 生成 LLM 上下文格式 |
| `test_history_limit` | 超出限制的裁剪 |
| `test_error_turn` | 错误标记 |
| `test_session_isolation` | session_id 唯一性 |
| `test_active_turn` | 活跃轮次追踪 |
| `test_clear` | 会话重置 |

### 集成测试（Phase 4 Step 2，需 Windows 环境）

- 端到端 `_run_full()` 全链路测试
- AIRI 实际 WebSocket 事件格式验证
- 断连重连 + 缓冲刷新测试

---

## 七、与后续 Phase 的关系

```
Phase 4 (本阶段)
  │
  ├─→ Phase 5 (打断机制)
  │     ├─ ConversationContext.mark_interrupted()
  │     ├─ VAD SPEECH_START → tts_mgr.stop() (已在 Phase 4 预埋)
  │     └─ 中断后保留上下文重新请求 LLM
  │
  └─→ Phase 6 (产品化)
        ├─ 流式优化
        ├─ 对话历史持久化
        └─ 性能监控
```

---

## 八、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| AIRI 消息格式与猜测不符 | 🟡 中 | `_on_airi_message` 提取不到文本 | Windows 端抓包验证真协议格式 |
| 大量对话导致内存增长 | 🟢 低 | history_limit=20 限制 | LRU 裁剪 + summary() 监控 |
| TTS 阻塞 asyncio 事件循环 | 🟡 中 | 音频播放期间无法处理新语音 | 已在 `say()` 中使用 `await` 非阻塞模式 |

---

> **下一步**: 完成单元测试，提交到 GitHub，等待 Windows 环境跑端到端验证。
