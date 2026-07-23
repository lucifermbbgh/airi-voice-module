# AIRI 语音模块 — 完整项目架构

> **项目名称：** AIRI Voice Module
> **目标平台：** Windows 11（主机运行） + Linux（开发环境）
> **Python 版本：** 3.12+
> **文档版本：** v1.0 · 2026-07-22

---

## 目录

1. [项目全景](#1-项目全景)
2. [技术栈总览](#2-技术栈总览)
3. [系统架构图](#3-系统架构图)
4. [数据流设计](#4-数据流设计)
5. [Phase 1：基础音频管道](#5-phase-1基础音频管道当前阶段)
6. [Phase 2：STT 集成](#6-phase-2stt-集成待规划)
7. [Phase 3：TTS 集成](#7-phase-3tts-集成待规划)
8. [Phase 4：AIRI 对话集成](#8-phase-4airi-对话集成待规划)
9. [Phase 5：打断机制](#9-phase-5打断机制待规划)
10. [Phase 6：产品级体验](#10-phase-6产品级体验待规划)
11. [项目结构](#11-项目结构)
12. [部署指南](#12-部署指南)
13. [模型资源清单](#13-模型资源清单)

---

## 1. 项目全景

### 1.1 项目目标

为 [Project AIRI](https://github.com/moeru-ai/airi) 系统添加**实时流式语音对话**能力，使用户能够通过麦克风与 AIRI 进行自然语音交互，AIRI 以语音回复。

### 1.2 分阶段实施路线

```
Phase 1 ── Capture + VAD + Playback (音频管道基础)    ✅ 已完成 (2026-07-21)
           │
Phase 2 ── STT 集成 (Faster-Whisper large-v3)         📋 待规划
           │  └─ 备选: SenseVoice / Paraformer
           │
Phase 3 ── TTS 集成 (CosyVoice 2)                     📋 待规划
           │  └─ 备选: ChatTTS
           │
Phase 4 ── AIRI WebSocket 对话集成                     📋 待规划
           │  └─ input:text / input:text:voice 事件
           │
Phase 5 ── 打断机制                                    📋 待规划
           │  └─ VAD 检测 → 暂停 TTS → 保留上下文 → 重新请求 LLM
           │
Phase 6 ── 产品级体验                                  📋 待规划
              └─ 流式优化 / Speculative Decoding / 零拷贝格式转换
```

**预计总工时：** 约 8~10 天（6 个 Phase）

### 1.3 设计原则

| 原则 | 说明 |
|------|------|
| **低延迟** | 端到端语音延迟 < 500ms 为目标 |
| **流式处理** | 边接收边处理，不等待完整音频 |
| **可打断** | 用户可随时中断 AIRI 的回复 |
| **模块化** | 各 Phase 可独立开发、测试、替换 |
| **Windows 优先** | 目标运行平台为 Windows 11 |
| **GPU 加速可选** | CUDA 加速 STT/TTS，无 GPU 可降级 |

---

## 2. 技术栈总览

### 2.1 核心依赖

| 层级 | 技术选型 | 版本 | 用途 |
|------|---------|------|------|
| **语言/运行时** | Python 3.12+ | ≥3.12 | 异步并发、低延迟流水线 |
| **异步框架** | asyncio（标准库） | — | 三协程并发编排 |
| **音频 I/O** | sounddevice (PortAudio) | ≥0.5.0 | 麦克风输入 + 扬声器输出 |
| **音频处理** | numpy + scipy | ≥1.26.0 / ≥1.14.0 | 重采样、格式转换、信号处理 |
| **VAD** | Silero VAD v6 + ONNX Runtime | ≥1.17.0 | 语音活动检测、状态机 |
| **STT（Phase 2）** | Faster-Whisper large-v3 | — | 语音→文字（中英双语） |
| **TTS（Phase 3）** | CosyVoice 2 / ChatTTS | — | 文字→语音、声音克隆、情感控制 |
| **WebSocket** | websockets | ≥12.0 | 与 AIRI 通信 |
| **配置管理** | PyYAML + 环境变量覆盖 | ≥6.0 | 可配置化参数 |
| **日志** | loguru | ≥0.7.0 | DEBUG 级别→文件+控制台 |
| **测试** | pytest + pytest-asyncio | ≥8.0 | 单元测试、异步测试 |

### 2.2 硬件需求

| 配置 | VRAM | 模型方案 |
|------|------|---------|
| **完整配置** | ~4-5 GB | Faster-Whisper large-v3 (2.5GB) + CosyVoice 2 (1.5GB) + VAD (0.1GB) |
| **轻量配置** | ~3 GB | Faster-Whisper medium + ChatTTS |
| **纯 CPU** | 0 GB (RAM ≥ 8GB) | ONNX Runtime CPU 推理 |

### 2.3 STT 方案对比（Phase 2 选型依据）

| 方案 | 识别质量 | 推理速度 | 中英双语 | 资源占用 |
|------|---------|---------|---------|---------|
| **Faster-Whisper large-v3** ★ | ⭐⭐⭐⭐⭐ | 4x 原版 Whisper | ✅ 极佳 | ~2.5GB VRAM |
| SenseVoice | ⭐⭐⭐⭐ | 快 | ✅ 优秀 | ~1GB VRAM |
| Paraformer | ⭐⭐⭐⭐ | 极快 | ✅ 良好 | ~1GB VRAM |

### 2.4 TTS 方案对比（Phase 3 选型依据）

| 方案 | 自然度 | 中英双语 | 声音克隆 | 情感控制 | Windows 兼容 |
|------|--------|---------|---------|---------|-------------|
| **CosyVoice 2** ★ | ⭐⭐⭐⭐⭐ | ✅ | ✅ | ✅ | ⚠️ 需验证 |
| ChatTTS | ⭐⭐⭐⭐ | ✅ 优秀 | ❌ | ✅ | ✅ 良好 |

---

## 3. 系统架构图

### 3.1 六层流水线架构

```
                           ┌──────────────────────────────────────────────────────────────┐
                           │                     AIRI Voice Module                         │
                           │                                                                │
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────┐
│          │   │          │   │          │   │              │   │           │   │          │
│ 麦克风   │──→│   VAD    │──→│   STT    │──→│  WebSocket   │──→│ AIRI LLM  │──→│   TTS    │──→  扬声器
│ (48kHz)  │   │ 检测语音  │   │ 语音→文字 │   │  ←→ AIRI    │   │ (DeepSeek)│   │ 文字→语音 │
│          │   │          │   │          │   │              │   │           │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────────┘   └───────────┘   └──────────┘
    P1             P1             P2                P4               P4             P3
   ✅ done       ✅ done        📋 plan           📋 plan         ✅ 已有        📋 plan

                                    ↑ 打断机制 (Phase 5)
                                      用户说话 → 暂停 TTS → 保留上下文 → 重新请求 LLM
```

### 3.2 Phase 1 当前实现：三协程流水线

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                               AudioPipeline                                           │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │  COROUTINE 1: _capture_loop()                                                │     │
│  │                                                                               │     │
│  │  ┌─────────────────────────────────────┐                                     │     │
│  │  │  sounddevice.InputStream            │  ← C 线程回调                        │     │
│  │  │  device=default, rate=48000,        │                                     │     │
│  │  │  blocksize=512, channels=1          │                                     │     │
│  │  └──────────────┬──────────────────────┘                                     │     │
│  │                 │ raw audio @ 48000 Hz, float32                               │     │
│  │                 ▼                                                             │     │
│  │  ┌──────────────────────────────┐                                            │     │
│  │  │ AudioRingBuffer.write_raw()  │  ← 线程安全 deque                           │     │
│  │  │ (C 线程 → asyncio bridge)    │                                            │     │
│  │  └──────────────┬───────────────┘                                            │     │
│  │                 │ asyncio.Queue.get()                                         │     │
│  │                 ▼                                                             │     │
│  │  ┌──────────────────────────────┐                                            │     │
│  │  │ Resampler.resample()         │  ← scipy.signal.resample_poly               │     │
│  │  │ 48kHz → 16kHz 多相滤波器      │                                            │     │
│  │  └──────────────┬───────────────┘                                            │     │
│  │                 │ 512-sample frame @ 16kHz (≈32ms)                           │     │
│  │                 ▼                                                             │     │
│  │  ┌──────────────────────────────┐                                            │     │
│  │  │ SileroVAD.process_frame()    │  ← ONNX Runtime 推理                        │     │
│  │  │ VAD 4 态状态机                │                                            │     │
│  │  └──────────────┬───────────────┘                                            │     │
│  │                 │ SpeechEvent?                                               │     │
│  │                 ▼                                                             │     │
│  │  ┌──────────────────────────────┐                                            │     │
│  │  │ _dispatch_speech_event()     │  → print / 回调 / 后续处理                  │     │
│  │  └──────────────────────────────┘                                            │     │
│  └──────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │  COROUTINE 2: _playback_loop()       ← 当前空闲，等待 TTS Phase 3 集成        │     │
│  │                                                                               │     │
│  │  ┌─────────────────────────────────────┐                                     │     │
│  │  │  sounddevice.OutputStream           │  ← 后台保持活跃                      │     │
│  │  │  device=default, rate=24000,        │                                     │     │
│  │  │  blocksize=512, channels=1          │                                     │     │
│  │  └─────────────────────────────────────┘                                     │     │
│  │                                                                               │     │
│  │  play_audio() 被调用 → PlaybackSegment 入队                                    │     │
│  │  C 线程回调 -> 消费 -> 播放                                                  │     │
│  └──────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │  MAIN ENTRY: main.py                                                         │     │
│  │                                                                               │     │
│  │  python -m src.main              → 完整模式（含 AIRI）                        │     │
│  │  python -m src.main --test-vad   → VAD 测试模式                               │     │
│  │  python -m src.main --list-devices → 列出音频设备                              │     │
│  └──────────────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 模块依赖关系

```
main.py
   │
   ├── config.py ─────────────── config/default.yaml
   │
   ├── pipeline/audio_pipeline.py
   │      │
   │      ├── audio/capture.py ──── pipeline/ring_buffer.py
   │      │                              │
   │      │                              └── audio/resampler.py
   │      │
   │      ├── vad/silero_vad.py ──── models/silero_vad.onnx
   │      │
   │      └── audio/playback.py
   │
   └── airi/websocket_client.py (Phase 4)
```

---

## 4. 数据流设计

### 4.1 音频数据流

```
[48,000 samples/sec]   →  sounddevice 回调 (C 线程)
[float32, 1 channel]   →  AudioRingBuffer.write_raw()   (线程安全)
       ↓
[asyncio.Queue]        →  Resampler.resample()           (scipy 多相滤波)
       ↓
[16,000 samples/sec]   →  SileroVAD.process_frame()     (ONNX Runtime)
[512 samples/frame]    →  每 32ms 一次推理
       ↓
[SpeechEvent]          →  _dispatch_speech_event()
                            ├── SPEECH_START: 时间戳
                            └── SPEECH_END:   audio + duration + max_prob
```

### 4.2 事件数据流 (Phase 4+)

```
[麦克风] → VAD → SPEECH_END(audio)
                   ↓
            STT Pipeline (Phase 2)
                   ↓
            "用户说了什么" (text)
                   ↓
            AIRI WebSocket (Phase 4)
                   ↓
            AIRI LLM 回复 (text)
                   ↓
            TTS Pipeline (Phase 3)
                   ↓
            AudioPlayback.play()
                   ↓
            [扬声器] 播放语音
```

### 4.3 控制流：Graceful Shutdown

```
Ctrl+C / SIGINT
    │
    ▼
shutdown_event.set()
    │
    ├── task.cancel()
    │       ├── _capture_loop()  → 中断循环 → 停止 capture
    │       └── _playback_loop() → 中断循环 → 停止 playback
    │
    ├── pipeline.stop()
    │       ├── capture.stop()   → 关闭 InputStream
    │       ├── playback.stop()  → 关闭 OutputStream + 清空队列
    │       ├── vad.flush()      → 强制结束当前语音段
    │       └── gather tasks     → 等待协程结束
    │
    └── logger.info("Pipeline stopped")
```

---

## 5. Phase 1：基础音频管道（当前阶段）

### 5.1 状态：✅ 已完成 (2026-07-21)

### 5.2 VAD 状态机

```
        ┌──────────┐
        │ SILENCE  │  ← 初始状态，无语音
        └────┬─────┘
             │ prob ≥ threshold (0.5)
             ▼
        ┌──────────────┐
        │ PENDING_START│  ← 等待确认（防噪声误触发）
        └────┬─────┬───┘
             │     │ prob < threshold → 噪声尖峰，丢弃
             │     ▼           回到 SILENCE
             │
             │ 连续 ≥7 帧 (≥224ms) prob ≥ threshold
             ▼
        ┌──────────┐
        │  SPEECH  │  ← 输出 SPEECH_START 事件
        └────┬─────┘    持续累积音频帧 _speech_frames[]
             │
             │ 连续 ≥15 帧 (≥480ms) prob < threshold
             ▼
        ┌──────────────┐
        │  PENDING_END │  → 输出 SPEECH_END 事件
        └──────┬───────┘      包含完整语音片段
               │
               ▼
          回到 SILENCE (_reset())
```

### 5.3 VAD 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `threshold` | 0.5 | 语音/非语音判定概率阈值 |
| `min_speech_duration` | 0.25s | 最短语音片段（≈7帧@32ms） |
| `min_silence_duration` | 0.5s | 沉默判定为结束（≈15帧@32ms） |
| `frame_size` | 512 | 每帧采样数 @ 16kHz |
| `sample_rate` | 16000 | VAD 模型期望采样率 |

### 5.4 音频缓冲架构（环形缓冲）

```
sounddevice 回调 (C 线程)
     │
     ▼  write_raw()
┌───────────────────────────────────────┐
│  Stage 1: deque[np.ndarray]          │  ← 线程安全，无锁写入
│  maxlen=100 帧                        │
└───────────────────────────────────────┘
     │
     ▼  asyncio 消费者读取 → 重采样 → write_processed()
┌───────────────────────────────────────┐
│  Stage 2: asyncio.Queue[np.ndarray]  │  ← 协程安全，供 VAD 消费
│  无界队列                              │
└───────────────────────────────────────┘
     │
     ▼  read() → async generator
VAD process_frame()
```

### 5.5 音频设备支持

| 平台 | API | 支持状态 |
|------|-----|---------|
| Windows | MME / WASAPI | ✅ sounddevice 原生支持 |
| Windows | ASIO | ⚠️ 需额外配置 |
| Linux | ALSA / PulseAudio | ✅ 开发环境使用 |
| macOS | CoreAudio | ✅ 理论支持（未验证） |

---

## 6. Phase 2：STT 集成（待规划）

### 6.1 目标

将 Phase 1 VAD 输出的语音片段（`SpeechEvent.audio`）转换为文字。

### 6.2 方案选型

| 方案 | 优先级 | 理由 |
|------|--------|------|
| **Faster-Whisper large-v3** | ★ 首选 | 中英识别极佳，4x 原版 Whisper，社区成熟 |
| SenseVoice | 备选 | 阿里达摩院，中英优秀，资源占用低 |
| Paraformer | 备选 | 阿里达摩院，推理极快 |

### 6.3 处理流程

```
SpeechEvent.audio (float32, 16kHz)
       │
       ▼
Faster-Whisper 推理
       │
       ├── 语言自动检测 (中/英)
       └── 带时间戳的字级别识别
       │
       ▼
"用户语音转换后的文字"
       │
       ▼
→ AIRI WebSocket output_text 事件
```

### 6.4 流式优化方案

- **Voice Activity 分段 STT**：只在 SPEECH_END 时触发，避免中间片段识别
- **增量识别**：长语音段落可边接收边输出 intermediate 结果
- **Speculative Decoding**：推测性解码加速推理

---

## 7. Phase 3：TTS 集成（待规划）

### 7.1 目标

将 AIRI LLM 的文字回复转换为语音，通过扬声器播放。

### 7.2 方案选型

| 方案 | 优先级 | 理由 |
|------|--------|------|
| **CosyVoice 2** | ★ 首选 | 中英双语、声音克隆、情感控制、自然度高 |
| ChatTTS | 备选 | Windows 兼容性更好，中英优秀 |

### 7.3 处理流程

```
AIRI LLM 回复 (text)
       │
       ▼
TTS 模型推理
       │
       ├── CosyVoice 2: 声音克隆 + 情感参数
       └── ChatTTS:  标准 TTS
       │
       ▼
audio (float32, 24000Hz)
       │
       ▼
AudioPlayback.play()
       │
       ▼
[扬声器]
```

### 7.4 流式优化方案

- **Token 级 TTS 合成**：LLM 逐 token 输出时即可开始合成，不必等待完整回复
- **流式播放**：TTS 音频块边合成边入队播放
- **零拷贝格式转换**：避免不必要的音频数据拷贝

---

## 8. Phase 4：AIRI 对话集成（待规划）

### 8.1 目标

通过 WebSocket 连接 AIRI 现有系统，实现 STT→LLM→TTS 闭环。

### 8.2 通信协议

```
AIRI WebSocket (ws://localhost:10443)
       │
       ├── 发送: input:text (STT 文字)
       │    └── { "type": "input:text", "text": "..." }
       │
       ├── 接收: output_text (LLM 回复)
       │    └── { "type": "output_text", "text": "..." }
       │
       └── (Phase 5) 发送: input:text:voice (打断后上下文)
            └── { "type": "input:text:voice", "text": "...", "interrupted": true }
```

### 8.3 示例交互

```
用户:  "[麦克风] 今天天气怎么样？"
       │
       ▼
VAD:  SPEECH_START → SPEECH_END
       │
       ▼
STT:  "今天天气怎么样？"
       │
       ▼
WebSocket → AIRI → LLM(DeepSeek)
       │
       ▼
AIRI → WebSocket: "今天天气晴朗，气温25°C..."
       │
       ▼
TTS:  CosyVoice 2 → 音频流
       │
       ▼
Playback: [扬声器播放语音]
```

---

## 9. Phase 5：打断机制（待规划）

### 9.1 目标

用户能在 AIRI 说话时随时打断，AIRI 停止播放、保留对话上下文、等待用户新输入。

### 9.2 打断流程

```
[AIRI 正在 TTS 播放中...]
       │
用户:  "[麦克风] 等等，我说的是..."
       │
       ▼
VAD:  检测到语音 (speech_start)
       │
       ├── playback.stop_current()    ← 立即停止 TTS 播放
       ├── TTS 引擎清空队列
       │
       ▼
VAD:  SPEECH_END
       │
       ├── STT 识别新语音
       │
       ▼
WebSocket → AIRI (保留历史上下文)
       │
       ▼
AIRI 基于对话历史 + 新输入重新回复
       │
       ▼
TTS + Playback 播放新回复
```

### 9.3 中断标志

```
{
    "type": "input:text:voice",
    "text": "等等，我说的是...",
    "interrupted": true,
    "interrupted_llm_output": "今天天气晴朗..."  // 被中断的 LLM 回复
}
```

---

## 10. Phase 6：产品级体验（待规划）

### 10.1 优化方向

| 优化项 | 说明 | 预期效果 |
|--------|------|---------|
| Speculative Decoding | STT/TTS 推测性解码 | 减少推理延迟 30-50% |
| 零拷贝音频格式转换 | 避免 numpy 数据拷贝 | 减少内存带宽消耗 |
| 动态 VAD 阈值 | 根据环境噪声自动调整 threshold | 降低误触发率 |
| 自动重连 | WebSocket 断开自动重连 | 提高稳定性 |
| 错误恢复 | 模型加载失败自动降级 | 提高鲁棒性 |
| 流式 Token 合成 | TTS 边接收 LLM token 边合成 | 降低首字节延迟 |
| 音量归一化 | 自动调整输入/输出音量 | 一致的录音/播放体验 |

### 10.2 监控与调试

```python
# 性能指标收集（Phase 6 实现）
metrics = {
    "vad_latency_ms": [],        # VAD 推理延迟
    "stt_latency_ms": [],        # STT 推理延迟
    "tts_latency_ms": [],        # TTS 推理延迟
    "pipeline_roundtrip_ms": [], # 端到端延迟
    "vad_trigger_rate": 0.0,    # VAD 触发率
    "interruption_count": 0,     # 打断次数
    "audio_device_status": "",   # 音频设备状态
}
```

---

## 11. 项目结构

```
airi-voice-module/
│
├── src/                              # 源代码
│   ├── __init__.py
│   ├── main.py                       # 入口（CLI 参数解析）
│   ├── config.py                     # YAML 配置 + 环境变量覆盖
│   ├── logger.py                     # loguru 日志配置
│   │
│   ├── audio/                        # 音频模块
│   │   ├── __init__.py
│   │   ├── capture.py                # AudioCapture（麦克风输入）
│   │   ├── playback.py               # AudioPlayback（扬声器输出）
│   │   └── resampler.py              # Resampler（重采样 48k↔16k）
│   │
│   ├── vad/                          # VAD 模块
│   │   ├── __init__.py
│   │   └── silero_vad.py             # SileroVAD（ONNX + 状态机）
│   │
│   ├── pipeline/                     # 流水线编排
│   │   ├── __init__.py
│   │   ├── audio_pipeline.py         # AudioPipeline（三协程编排）
│   │   └── ring_buffer.py            # AudioRingBuffer（C线程↔asyncio桥接）
│   │
│   └── airi/                         # AIRI 集成
│       ├── __init__.py
│       └── websocket_client.py       # AIRI WebSocket 客户端
│
├── models/                           # AI 模型文件
│   └── silero_vad.onnx               # Silero VAD ONNX 模型 (2.3MB)
│
├── config/                           # 配置文件
│   └── default.yaml                  # 默认配置参数
│
├── tests/                            # 测试
│   ├── __init__.py
│   ├── test_vad.py                   # VAD 状态机单元测试
│   ├── test_capture.py               # 捕获模块测试
│   └── test_pipeline.py              # 流水线集成测试
│
├── docs/                             # 文档
│   ├── PHASE-1-DESIGN.md             # Phase 1 设计文档
│   └── ARCHITECTURE.md               # 完整架构文档 (本文)
│
├── logs/                             # 运行时日志输出
│
├── requirements.txt                  # Python 依赖
├── pyproject.toml                    # 项目元数据
├── README.md                         # 项目说明
└── .gitignore                        # Git 忽略规则
```

---

## 12. 部署指南

### 12.1 环境要求

- **操作系统：** Windows 11（目标）/ Linux（开发）
- **Python：** 3.12+
- **GPU（可选）：** NVIDIA GPU + CUDA 12.x 加速 STT/TTS
- **音频设备：** 麦克风 + 扬声器（或耳机）

### 12.2 安装步骤（Windows）

```powershell
# 1. 创建虚拟环境
cd D:\DevProject\PythonProject\airi-voice-module
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. 安装核心依赖
pip install onnxruntime sounddevice numpy scipy pyyaml loguru websockets

# 3. 获取 VAD 模型（二选一）
# 选项A: 通过 pip 获取
pip install silero-vad --no-deps
python -c "import sys,pathlib,shutil; p=next(x for x in sys.path if 'site-packages' in x); s=pathlib.Path(p)/'silero_vad'/'data'/'silero_vad.onnx'; d=pathlib.Path('models/silero_vad.onnx'); shutil.copy(str(s),str(d))"

# 选项B: 从 GitHub Release 下载
# Invoke-WebRequest -Uri "https://github.com/snakers4/silero-vad/releases/download/v6.2/silero_vad.onnx" -OutFile "models\silero_vad.onnx"

# 4. 验证安装
python -m src.main --list-devices
python -m src.main --test-vad
```

### 12.3 运行模式

| 命令 | 模式 | 说明 |
|------|------|------|
| `python -m src.main --list-devices` | 设备列表 | 列出所有音频输入/输出设备 |
| `python -m src.main --test-vad` | VAD 测试 | 仅 Capture→VAD，无 AIRI 连接 |
| `python -m src.main` | 完整模式 | 全流水线（Phase 4 后可用） |

### 12.4 配置覆盖

可通过环境变量覆盖 YAML 配置：

```powershell
# 修改 VAD 阈值
$env:VAD_THRESHOLD = "0.3"
python -m src.main --test-vad

# 指定音频设备
$env:AUDIO_INPUT_DEVICE = "1"
python -m src.main --test-vad
```

---

## 13. 模型资源清单

| 模型 | 版本 | 大小 | 来源 | 用途 | Phase |
|------|------|------|------|------|-------|
| `silero_vad.onnx` | v6.2.1 | ~2.3 MB | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | 语音活动检测 | P1 ✅ |
| Faster-Whisper large-v3 | latest | ~2.5 GB | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 语音→文字 | P2 📋 |
| CosyVoice 2 | latest | ~1.5 GB | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | 文字→语音 | P3 📋 |
| ChatTTS (备选) | latest | ~500 MB | [2noise/ChatTTS](https://github.com/2noise/ChatTTS) | 文字→语音 | P3 📋 |

---

## 附录 A：Phase 1 部署排查记录

### A.1 Windows 部署问题清单

| 问题 | 原因 | 解决 |
|------|------|------|
| `SSLEOFError` | 代理/防火墙导致 pip SSL 连接中断 | 添加 `--trusted-host pypi.org --trusted-host files.pythonhosted.org` |
| `silero-vad` 模型加载失败 | v6.x pip 包依赖 PyTorch（~2-3GB），`importlib.resources` 触发 torch import | 方案A: ONNX 模型文件放入 `models/`，纯 onnxruntime 加载 |
| Windows venv 路径错误 | venv 误用了 WSL Python 解释器 | 检查 `where python`，确保指向 Windows Python |
| PyCharm pip 兼容性 | pip 20.3.4 与 Python 3.13 distutils 移除冲突 | 升级 pip 至最新版 |

### A.2 VAD 模型加载策略

```python
# 最终采用的方案（silero_vad.py）
def load_model(self):
    # 策略1: 从 silero-vad pip 包内嵌路径加载（依赖 torch → 通常失败）
    # 策略2: 从本地 models/silero_vad.onnx 加载 ← 默认命中
    #         纯 onnxruntime，无 PyTorch 依赖
    self._load_onnx_direct()
```

---

## 附录 B：性能指标参考

| 指标 | Phase 1 | Phase 2+ | Phase 6 目标 |
|------|---------|----------|-------------|
| VAD 推理延迟 | ~2-5ms/帧 | ~2-5ms/帧 | < 2ms/帧 |
| STT 推理延迟 | — | ~500-2000ms/句 | < 300ms (流式) |
| TTS 推理延迟 | — | ~200-500ms/句 | < 100ms (流式) |
| 端到端延迟 | — | ~1-3s | < 500ms |
| 麦克风捕获延迟 | ~10ms | ~10ms | < 5ms |
| 音频输出延迟 | ~20ms | ~20ms | < 10ms |
| VAD 误触发率 | — | — | < 5% |

---

> **文档维护者：** Claude Code (via MCP 对话自动记忆系统)
> **最后更新：** 2026-07-22
> **关联线程：** AIRI 语音模块设计 (TTS/STT)
