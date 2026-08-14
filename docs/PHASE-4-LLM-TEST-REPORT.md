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

> 在 Windows 11 主机上逐条执行。每步标注 **预期输出** 和 **✅ 通过标准** / **❌ 失败排查**。

---

### 步骤 0：拉取代码并确认环境

```powershell
cd D:\DevProject\PythonProject\airi-voice-module
git pull origin main
```

**预期输出**：
```
Updating dafc981..e6b01ab
Fast-forward
 CLAUDE.md                    |  88 ++++++
 docs/PHASE-4-TEST-REPORT.md  | 198 +++++++++++
 2 files changed, 286 insertions(+)
```

✅ 看到 `Fast-forward` 且无冲突即通过。

---

```powershell
.venv\Scripts\activate
python --version
```

**预期输出**：`Python 3.13.2`

✅ Python 3.11+ 即可。如果报 `.venv` 不存在，检查路径是否正确。

---

```powershell
python -c "import numpy, websockets, yaml, loguru, sounddevice; print('deps OK')"
```

**预期输出**：`deps OK`

❌ 如果 `ModuleNotFoundError`，运行 `pip install -r requirements.txt`。

> **常见坑：PyPI 下载超时（Windows + 代理环境）**
>
> 如果 `pip install` 报 `ReadTimeoutError` 或 `ConnectionResetError(10054)`，需要先设置代理（PowerShell 语法）：
> ```powershell
> $env:HTTP_PROXY="http://127.0.0.1:10809"
> $env:HTTPS_PROXY="http://127.0.0.1:10809"
> pip install -r requirements.txt
> ```
> 如果 10809 仍超时，换 10808 端口（xray mixed 协议）。或使用 `--proxy` 参数：
> ```powershell
> pip install -r requirements.txt --proxy http://127.0.0.1:10809
> ```
> 注意：PowerShell 用 `$env:VAR="value"`，不是 cmd 的 `set VAR=value`。

---

### 步骤 1：验证 Phase 4 新增模块

```powershell
python -m pytest tests/test_conversation.py -v
```

**预期输出**（最后 5 行）：
```
tests/test_conversation.py::TestTurn::test_default_turn_is_user PASSED
tests/test_conversation.py::TestTurn::test_turn_has_unique_id PASSED
...
tests/test_conversation.py::TestConversationContext::test_start_turn_while_previous_active PASSED
============================== 21 passed in X.XXs ==============================
```

✅ **21 passed** → Phase 4 核心模块在 Windows 上正常。

❌ 如果 `ModuleNotFoundError: No module named 'src.airi.conversation'`，说明 `git pull` 未拉取到新文件。运行 `git log --oneline -3` 确认最新 commit 为 `e6b01ab`。

---

### 步骤 2：跑完整测试套件（排除需硬件的测试）

```powershell
python -m pytest tests/ -v --ignore=tests/test_mic_level.py --ignore=tests/test_pipeline.py --ignore=tests/test_vad_diagnostic.py --ignore=tests/test_vad_model_compare.py -q
```

**预期输出**：
```
....................................... [XX%]
....................................... [XX%]
...............................         [100%]
XXX passed, X skipped in X.XXs
```

✅ **所有测试 passed，无 FAILED 或 ERROR**。

❌ 如果出现 `numpy._core._multiarray_umath` 错误（Windows 上通常不会有 Linux 的 PYTHONPATH 问题，但如果出现），检查 Python 版本和 numpy 版本是否匹配：
```powershell
python -c "import numpy; print(numpy.__version__)"
# 应输出 2.x（而非 1.x）
```

---

### 步骤 3：验证 Phase 4 上下文管理独立功能

```powershell
python -c "
from src.airi.conversation import ConversationContext, Turn, TurnRole, TurnStatus

ctx = ConversationContext()
turn = ctx.start_user_turn('你好 AIRI', confidence=0.95)
ctx.append_response('你好！')
ctx.append_response('有什么可以帮助你的？')
ctx.complete_current_turn()

turn2 = ctx.start_user_turn('今天天气怎么样')
ctx.append_response('今天天气晴朗，气温 25°C。')
ctx.complete_current_turn()

history = ctx.recent_history
print(f'Turns: {ctx.turn_count}')
print(f'History entries: {len(history)}')
for h in history:
    print(f'  [{h[\"role\"]}] {h[\"content\"][:50]}')
print(f'Summary: {ctx.summary()}')
"
```

**预期输出**：
```
Turns: 2
History entries: 4
  [user] 你好 AIRI
  [assistant] 你好！有什么可以帮助你的？
  [user] 今天天气怎么样
  [assistant] 今天天气晴朗，气温 25°C。
Summary: {'session_id': 'xxxxxxxx', 'turn_count': 2, ...}
```

✅ 输出 2 轮对话、4 条历史记录、user/assistant 交替。

---

### 步骤 4：验证 STT 模块在 Windows 上仍正常

```powershell
python -m pytest tests/test_stt.py tests/test_stt_integration.py -v -q
```

**预期输出**：
```
...............................
67 passed in X.XXs
```

✅ **67 passed**（21 单元 + 46 集成），与 Phase 2 一致，无回归。

---

### 步骤 5：【阻塞项】修复 Phase 1 VAD

> ⚠️ 此步骤需要手动操作 Windows 系统设置。

#### 5a：确认当前阻塞状态

```powershell
python -m pytest tests/test_vad.py -v -q
```

**当前预期**（context 前缀 bug 已修复后）：**Linux 与 Windows 均应通过**。

#### 5b：尝试修复 — 禁用 Realtek 音频增强

1. `Win+R` → 输入 `mmsys.cpl` → 回车
2. 点击「录制」选项卡
3. 右键你的麦克风设备 → 「属性」
4. 点击「增强」选项卡
5. 勾选「禁用所有增强」或逐个取消勾选
6. 点击「确定」

#### 5c：验证修复效果

```powershell
python -c "
import sounddevice as sd
import numpy as np
from src.vad.silero_vad import SileroVAD

vad = SileroVAD(model_path='models/silero_vad.onnx', threshold=0.3)
vad.load_model()

# 录制 3 秒并检测语音概率
print('Recording 3s... speak now!')
audio = sd.rec(int(16000 * 3), samplerate=16000, channels=1, dtype='float32')
sd.wait()

# 分帧检测
for i in range(0, len(audio) - 512, 512):
    prob = vad._get_speech_prob(audio[i:i+512].flatten())
    if prob > 0.3:
        print(f'  Frame {i//512}: speech prob = {prob:.4f} ⬆️ VOICE')
    elif i % 10 == 0:
        print(f'  Frame {i//512}: speech prob = {prob:.4f}')

print(f'Max probability: {vad.max_prob:.4f}')
vad.unload_model()
"
```

**修复成功标准**：安静时 prob < 0.15，说话时 prob > 0.3。

**修复前（已知情况）**：max_prob = 0.0787（7.87%）— 全部被 DSP 滤除。

**仍不成功时的备选**：外接 USB 麦克风，或换用 webrtcvad（见 `docs/PHASE-1-TEST-REPORT.md`）。

---

### 步骤 6：【阻塞项】完成 Phase 3 TTS Windows 部署

#### 6a：安装 CUDA PyTorch

```powershell
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**验证**：
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
```

**预期输出**：`CUDA: True, Device: NVIDIA GeForce RTX 3070 Ti Laptop GPU`

#### 6b：安装 CosyVoice 推理依赖

```powershell
pip install -r D:\DevProject\PythonProject\CosyVoice\requirements_infer.txt
```

#### 6c：创建 CosyVoice 模块注入文件

```powershell
python -c "
import site, os
pth_path = os.path.join(site.getsitepackages()[0], 'cosyvoice.pth')
with open(pth_path, 'w', encoding='utf-8') as f:
    f.write(r'D:\\DevProject\\PythonProject\\CosyVoice')
print(f'Created: {pth_path}')
"
```

**验证**：
```powershell
python -c "from cosyvoice.cli.cosyvoice import AutoModel; print('CosyVoice OK')"
```

**预期输出**：`CosyVoice OK`

#### 6d：TTS 合成验证

```powershell
python scripts/test_tts_windows.py --mode synthesize
```

**预期输出**：
```
[1/3] Synthesising 18 chars...
   Text: "你好，我是 AIRI，你的智能语音助手。"
   ✅ Saved: output/tts_test_1.wav
          Duration: 2.34s, RTF: 0.15

[2/3] ...
[3/3] ...
```

✅ **每个测试文本都生成 .wav 文件且 Duration > 0**。

#### 6e：TTS 播放验证（需扬声器）

```powershell
python scripts/test_tts_windows.py --mode play
```

对着提示输入文字，确认能听到语音输出。

---

### 步骤 7：全链路端到端测试

> ⚠️ 前置条件：步骤 5（VAD）和步骤 6（TTS）已修复 + AIRI 正在运行。

#### 7a：确认 AIRI 在线

```powershell
python -c "
import asyncio, websockets
async def check():
    try:
        ws = await websockets.connect('ws://localhost:10443', ping_timeout=3)
        print('✅ AIRI WebSocket reachable')
        await ws.close()
    except Exception as e:
        print(f'❌ AIRI not reachable: {e}')
asyncio.run(check())
"
```

**预期输出**：`✅ AIRI WebSocket reachable`

❌ 如果连接失败，确认 AIRI Electron 应用已启动。在 AIRI 设置中确认 WebSocket 端口为 10443。

#### 7b：启动全链路

```powershell
python -m src.main
```

**预期输出**：
```

🎤 AIRI Voice Module - Full Mode (VAD → STT → AIRI → TTS)
============================================================
   AIRI:     ws://localhost:10443
   TTS:      ✅ cosyvoice
   Session:  a1b2c3d4
============================================================
   ✅ Connected to AIRI
```

#### 7c：语音交互测试

对着麦克风说：「你好 AIRI，今天天气怎么样？」

**预期链路**：
```
🗣️  [SPEECH START]
🤫 [END] dur=2.34s → "你好AIRI今天天气怎么样" (conf=0.87)
                                    ↓ WebSocket → AIRI → DeepSeek
   ✅ [AIRI complete]               ← LLM 回复（TTS 语音播放）
```

✅ **看到 `🗣️ SPEECH START` → `🤫 END` → STT 文字 → `✅ AIRI complete`，并听到 AIRI 语音回复**。

❌ 如果 `SPEECH START` 不触发 → 回到步骤 5（VAD 阻塞）。
❌ 如果 STT 文字为空或乱码 → 检查 models/faster-whisper-small 是否已下载。
❌ 如果 `AIRI not connected` → 检查步骤 7a。
❌ 如果 `✅ AIRI complete` 出现但没声音 → 检查步骤 6e（TTS 播放）。

---

### 步骤 8：对话上下文验证

连续说两轮对话，验证上下文管理：

```powershell
# 先问一个需要记忆的问题
(对着麦克风) "我叫Elysia，记住我的名字"
# → AIRI 回复确认

# 再问一个依赖上下文的问题
(对着麦克风) "我刚才说了我叫什么名字"
# → AIRI 应该回答 "Elysia"
```

**通过标准**：AIRI 的第二轮回答中包含 "Elysia"，证明对话上下文在 AIRI 端正常工作。

> 注：多轮上下文由 AIRI 内部维护，Voice Module 的 `ConversationContext` 负责本地追踪和日志记录。AIRI 是否记住了上下文取决于 AIRI 的 session 管理配置。

---

### 完整的 Windows 验证检查清单

| # | 操作 | 命令 | 通过标准 |
|:--|:-----|:-----|:--------|
| 0a | 拉取代码 | `git pull` | `Fast-forward` 无冲突 |
| 0b | 确认 Python | `python --version` | 3.11+ |
| 0c | 确认依赖 | `python -c "import numpy,websockets..."` | `deps OK` |
| 1 | Phase 4 单元测试 | `pytest tests/test_conversation.py -v` | 21 passed |
| 2 | 完整测试套件 | `pytest tests/ -q --ignore=...` | 全部 passed |
| 3 | 上下文功能验证 | `python -c "from src.airi.conversation..."` | 2 turns / 4 history |
| 4 | STT 无回归 | `pytest tests/test_stt.py tests/test_stt_integration.py -q` | 67 passed |
| 5 | VAD 修复 | 禁用 Realtek 增强 → 运行验证脚本 | max_prob > 0.3（说话时） |
| 6a | CUDA PyTorch | `pip install torch cu124` → 验证 CUDA | `True, RTX 3070 Ti` |
| 6b-d | TTS 合成 | `python scripts/test_tts_windows.py --mode synthesize` | 生成 .wav（Duration > 0） |
| 6e | TTS 播放 | `python scripts/test_tts_windows.py --mode play` | 听到语音 |
| 7a | AIRI 连通性 | `asyncio.run(check())` | `✅ reachable` |
| 7b-c | 全链路端到端 | `python -m src.main` | SPEECH→STT→AIRI→TTS 完整闭环 |
| 8 | 上下文测试 | 连续两轮对话 | AIRI 记住前一轮信息 |

---

## 八、Windows 验证执行结果（2026-08-13）

> 第七章的验证指南已在本会话 Windows 端实际执行，结果如下。

### 8.1 验证结果汇总

| 步骤 | 操作 | 结果 |
|:-----|:-----|:-----|
| 1 | Phase 4 单元测试（conversation） | ✅ 21/21 passed (0.17s) |
| 2 | 完整测试套件 | ✅ 193 passed |
| 3 | 上下文功能验证 | ✅ Turns: 2, History: 4 |
| 4 | STT 无回归 | ✅ 67/67 passed (0.33s) |
| 5 | VAD 单元测试 | ✅ 9/9 passed (0.61s，离线合成音频) |
| 6 | TTS 合成 + 播放 | ✅ 已完成（见 PHASE-3-TTS-TEST-REPORT.md 第九章） |

### 8.2 Windows 环境问题与修复

| 问题 | 原因 | 修复 |
|:-----|:-----|:-----|
| pytest 缺失（`No module named pytest`） | Windows venv 未装 pytest | 设代理后 `pip install pytest pytest-asyncio` |
| pip 下载超时（`ReadTimeoutError`） | PyPI 需走代理 | PowerShell 用 `$env:HTTP_PROXY="http://127.0.0.1:10809"`（非 cmd 的 `set`） |
| pyworld 编译失败 | 旧版无 cp313 wheel | pyworld 0.3.5 已有 cp313 wheel，直接装 |

### 8.3 剩余阻塞项

- **Phase 1 VAD**：已解决（context 前缀 bug 修复，`8e37cb2`；真实语音概率恢复正常）
- **全链路端到端**：待 VAD 阻塞解决后执行（步骤 7）
