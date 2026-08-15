# AIRI Voice Module — Project Brief

为 AIRI 系统添加实时语音对话能力。Voice Module 作为 AIRI 插件运行，通过 WebSocket 协议桥接「耳朵（VAD→STT）」和「嘴巴（TTS）」到 AIRI 的大脑（LLM）。

## 当前状态

| Phase | 内容 | 完成度 | 平台 |
|-------|------|:------:|------|
| Phase 1 | VAD 语音检测 | ✅ 100% | 已解决（context 前缀 bug） |
| Phase 2 | STT 语音识别 | ✅ 100% | Linux + Windows 双平台 67/67 |
| Phase 3 | TTS 语音合成 | 🔶 95% | Linux 67/67，Windows 待 CUDA PyTorch |
| Phase 4 | LLM 对话集成 | ✅ 100% Linux | Windows 端到端待验证 |
| Phase 5 | 打断机制 | ⬜ 0% | 已预埋 tts_mgr.stop() |
| Phase 6 | 产品化 | ⬜ 0% | — |

**阻塞项**：Phase 3 Windows CUDA PyTorch 安装。

## 架构

```
Mic → VAD → STT → AIRI WebSocket (ws://localhost:6121/ws) → TTS → Speaker
 P1     P1    P2              P4                            P3     P1
```

Voice Module 不直接调用 LLM。它通过 AIRI 插件协议发送 `input:text:voice` 事件（STT 文字），接收 `output:gen-ai:chat:message` 事件（LLM 回复），再喂给 TTS 播放。

## 代码地图

```
src/
├── main.py              # 入口：CLI 模式分发 + _run_full() 全链路
├── config.py            # 配置加载（YAML + 环境变量覆盖）
├── audio/
│   ├── capture.py       # 麦克风输入（sounddevice InputStream）
│   ├── playback.py      # 扬声器输出（sounddevice OutputStream）
│   └── resampler.py     # 48kHz→16kHz 重采样
├── vad/
│   └── silero_vad.py    # Silero VAD + 状态机
├── stt/
│   ├── faster_whisper_stt.py  # Faster-Whisper 引擎
│   └── post_processor.py      # 文本后处理（标点/热词）
├── tts/
│   ├── tts_engine.py         # TTS 接口抽象
│   ├── cosyvoice_tts.py      # CosyVoice 2 引擎
│   └── tts_manager.py        # 合成→缓存→播放 编排
├── airi/
│   ├── websocket_client.py   # AIRI WS 客户端（连接/心跳/收发）
│   └── conversation.py       # Phase 4 对话上下文管理
└── pipeline/
    ├── audio_pipeline.py     # 三协程编排（capture/VAD/playback）
    └── ring_buffer.py        # 线程安全音频缓冲
```

## 关键设计决策

- **直连 AIRI WebSocket**：Voice Module 通过 `ws://localhost:6121/ws` 与 AIRI 插件协议通信，不绕任何中间层
- **三协程流水线**：capture_loop / vad_loop / playback_loop 并发运行
- **TTS 引擎**：CosyVoice 2（首选），完全离线，中文极优
- **STT 引擎**：Faster-Whisper small int8，CTranslate2 后端

## 开发命令

```bash
# 测试（Linux）
cd /home/elysia/project/airi-voice-module
.venv/bin/python3 -m pytest tests/ -q
# → 201 passed, 2 skipped

# Windows 路径
# D:\DevProject\PythonProject\airi-voice-module

# 运行模式
python -m src.main --test-vad         # VAD 测试
python -m src.main --test-stt         # STT 测试
python -m src.main --test-tts         # TTS 交互测试
python -m src.main                    # 全链路（需 AIRI 在线）
```

## 环境陷阱

- **PYTHONPATH 污染**：Hermes 注入 Python 3.11 的 site-packages，会导致 numpy C 扩展不兼容。项目 `tests/conftest.py` 已在测试时自动修复。运行 `src/main.py` 时需 `PYTHONPATH="" .venv/bin/python3 -m src.main`。

## 依赖

```
sounddevice, numpy, scipy, silero-vad, onnxruntime, websockets, pyyaml, loguru
faster-whisper (Phase 2), CosyVoice 2 (Phase 3, 需从 GitHub 源码安装)
```
