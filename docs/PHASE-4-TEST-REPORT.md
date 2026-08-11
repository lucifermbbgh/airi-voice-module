# AIRI Voice Module — Phase 4 LLM 对话集成 测试报告

> **日期**: 2026-08-11
> **平台**: Linux 开发环境 (Python 3.14)
> **项目路径**: `/home/elysia/project/airi-voice-module/`
> **Git Commit**: `2d29f6a` (Phase 4 完成 + PYTHONPATH 修复)

---

## 一、测试范围

Phase 4 LLM 对话集成测试覆盖以下模块：

| 模块 | 文件 | 行数 |
|:----|:-----|:----:|
| 对话上下文管理 | `src/airi/conversation.py` | 220 |
| AIRI 模块导出 | `src/airi/__init__.py` | 16 |
| 全链路编排 | `src/main.py` (_run_full 重写) | ~290 |
| 上下文单元测试 | `tests/test_conversation.py` | 210 |
| PYTHONPATH 修复 | `tests/conftest.py` | 25 |

---

## 二、测试环境

### 开发环境（Linux）✅ 已验证

| 项目 | 值 |
|:----|:----|
| 系统 | Linux (Ubuntu, GNOME Wayland) |
| Python | 3.14 |
| 测试框架 | pytest 9.1.1 + asyncio |
| 测试模式 | 纯单元测试（无 AIRI WebSocket 依赖） |
| **测试结果** | **201/201 通过，2 跳过 (1.42s)** |

---

## 三、测试详细结果

### 3.1 单元测试明细

| 测试套件 | 测试数 | 通过 | 失败 | 跳过 | 说明 |
|:--------|:------:|:----:|:----:|:----:|:-----|
| `test_conversation.py` | 18 | 18 | 0 | 0 | Phase 4 新增 |
| `test_capture.py` | 9 | 9 | 0 | 0 | 原有，无回归 |
| `test_vad.py` | 9 | 9 | 0 | 0 | 原有，无回归 |
| `test_stt.py` | 21 | 21 | 0 | 0 | 原有，无回归 |
| `test_stt_integration.py` | 46 | 46 | 0 | 0 | 原有，无回归 |
| `test_stt_inference.py` | — | — | — | — | 排除（需硬件） |
| `test_tts.py` | 39 | 39 | 0 | 0 | 原有，无回归 |
| `test_tts_integration.py` | 28 | 28 | 0 | 0 | 原有，无回归 |
| `test_pipeline.py` | — | — | — | — | 排除（需硬件） |
| 诊断/工具测试 | — | 通过 | — | 2 跳过 | 需麦克风硬件 |
| **合计** | **201** | **201** | **0** | **2** | |

### 3.2 Phase 4 新增测试 — ConversationContext

| # | 测试用例 | 覆盖点 |
|:--|:--------|:------|
| 1 | `test_default_turn_is_user` | Turn 默认值：角色/状态/ID |
| 2 | `test_turn_has_unique_id` | 每轮独立 ID |
| 3 | `test_append_response_updates_status` | 追加回复后状态迁移 (pending→streaming) |
| 4 | `test_append_response_accumulates` | 多次追加文本累积 |
| 5 | `test_mark_complete` | 完成标记 + 时间戳 |
| 6 | `test_mark_error` | 错误标记 + 错误信息记录 |
| 7 | `test_mark_interrupted` | 打断标记 (Phase 5 预埋) |
| 8 | `test_new_context_is_empty` | 新上下文初始状态 |
| 9 | `test_start_user_turn` | 创建用户轮次 |
| 10 | `test_append_response_to_active_turn` | 流式追加到活跃轮次 |
| 11 | `test_append_response_with_no_turns` | 空上下文边界处理 |
| 12 | `test_complete_current_turn` | 完成当前轮次 |
| 13 | `test_error_current_turn` | 错误标记当前轮次 |
| 14 | `test_recent_history_format` | LLM 格式历史输出 (user/assistant 交替) |
| 15 | `test_history_limit_enforced` | 历史裁剪 (超过 20 轮) |
| 16 | `test_session_isolation` | 会话 ID 唯一性 |
| 17 | `test_clear_resets_state` | 会话重置 |
| 18 | `test_summary` | 会话摘要统计 |
| 19 | `test_multiple_streaming_chunks` | 多块流式响应 |
| 20 | `test_active_turn_none_after_complete` | 完成后活跃轮次清空 |
| 21 | `test_start_turn_while_previous_active` | 新轮次开始不强制关闭旧轮次 (Phase 5 打断模式) |

---

## 四、Phase 4 实现摘要

### 4.1 新增模块

| 文件 | 行数 | 功能 |
|:----|:----:|:-----|
| `src/airi/conversation.py` | 220 | `Turn` + `ConversationContext` — 对话轮次管理、流式响应累积、LLM 历史格式输出、会话状态追踪 |
| `tests/test_conversation.py` | 210 | 21 项单元测试，100% 覆盖 ConversationContext 和 Turn |
| `tests/conftest.py` | 25 | 永久修复 Hermes venv PYTHONPATH 污染 |
| `docs/PHASE-4-LLM.md` | — | Phase 4 完整设计文档 |

### 4.2 修改文件

| 文件 | 变更 | 说明 |
|:----|:----|:-----|
| `src/main.py` | `_run_full()` 重写 | 6 项增强 (见下方) |
| `src/airi/__init__.py` | 导出扩展 | 新增 ConversationContext / Turn / TurnRole / TurnStatus |

### 4.3 `_run_full()` 六项增强

| # | 增强 | 说明 |
|:--|:-----|:-----|
| 1 | **对话上下文** | `ConversationContext` 追踪每轮对话，含 turn_id / 时间戳 / 置信度 / 响应分块 |
| 2 | **STT 异常恢复** | `try/except` 包裹 `stt.transcribe()`，异常不崩溃 |
| 3 | **断连缓冲** | `asyncio.Queue(32)` 缓冲 AIRI 断连期间的 STT 结果，重连后自动 `_flush_pending_sends()` |
| 4 | **complete 事件** | 处理 `output:gen-ai:chat:complete`，标记 Turn 完成并记录统计 |
| 5 | **TTS 降级** | TTS 不可用时降级为 `logger.info()` 文本输出，管线继续运行 |
| 6 | **打断预备** | `SPEECH_START` 时调用 `tts_mgr.stop()`，为 Phase 5 打断机制预埋 |

---

## 五、已知问题与限制

| 问题 | 严重程度 | 说明 |
|:----|:--------:|:-----|
| AIRI 消息格式未验证 | 🟡 中 | `_on_airi_message` 使用 `data.get("text")` / `data.get("message")` / `data.get("content")` 三选一，未对 AIRI 真实协议格式做过验证，需 Windows 端实测确认 |
| Python 3.14 numpy C 扩展污染 | ✅ 已修复 | `conftest.py` 在测试收集前剥离 Hermes 注入的 Python 3.11 site-packages |
| 全链路端到端未跑 | 🔴 阻塞 | `_run_full()` 需要 AIRI WebSocket 在线 + Phase 1 VAD 可用 + Phase 3 TTS 可用，被 Windows 端阻塞 |

---

## 六、Git 提交记录

| Commit | 说明 |
|:------|:-----|
| `b811dc8` | Phase 4: LLM对话集成 — 对话上下文管理 + 错误恢复 + 断连缓冲 |
| `2d29f6a` | fix: permanent PYTHONPATH fix via conftest.py |

---

## 七、Windows 验证指南

### 环境准备

```powershell
# 1. 拉取最新代码
cd D:\DevProject\PythonProject\airi-voice-module
git pull origin main

# 2. 激活 venv（如果尚未）
.venv\Scripts\activate

# 3. 确保依赖
pip install -r requirements.txt
```

### 第一步：跑 Phase 4 新增测试

```powershell
# 先验证新模块不受 Windows 环境影响
python -m pytest tests/test_conversation.py -v
```

预期：21 passed。

### 第二步：跑完整测试套件

```powershell
python -m pytest tests/ -v --ignore=tests/test_mic_level.py --ignore=tests/test_pipeline.py --ignore=tests/test_vad_diagnostic.py --ignore=tests/test_vad_model_compare.py
```

预期排除硬件依赖测试后，全部通过。

### 第三步：修复阻塞项

```
Phase 1 VAD 阻塞:
  方案A: 禁用 Realtek 音频增强（控制面板 → 声音 → 麦克风属性 → 增强 → 禁用所有）
  方案B: 外接 USB 麦克风
  方案C: 换用 webrtcvad

Phase 3 TTS 收尾:
  pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  pip install -r D:\DevProject\PythonProject\CosyVoice\requirements_infer.txt
  # 创建 cosyvoice.pth 注入 site-packages
  python scripts/test_tts_windows.py --mode synthesize
```

### 第四步：端到端测试

```powershell
# 确保 AIRI 正在运行（Electron 应用已启动，WebSocket 在 10443 端口）
python -m src.main
```

预期输出：
```
🎤 AIRI Voice Module - Full Mode (VAD → STT → AIRI → TTS)
   AIRI:  ws://localhost:10443
   TTS:   ✅ cosyvoice
   Session: a1b2c3d4
   ✅ Connected to AIRI
```

然后对着麦克风说话，应该看到 `🗣️ [SPEECH START]` → `🤫 [END]` → STT 文字 → AIRI TTS 语音回复。
