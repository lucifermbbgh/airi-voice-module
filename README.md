# AIRI Voice Module

实时语音对话后端模块，为 [Project AIRI](https://github.com/moeru-ai/airi) 添加语音交互能力（「耳朵」VAD→STT + 「嘴巴」TTS）。

## 架构概览

```
[麦克风] → [VAD] → [STT] → [AIRI WebSocket] → [LLM] → [TTS] → [扬声器]
   P1        P1       P2          P4            P4       P3       P1
```

Voice Module 作为 AIRI 插件运行，通过 `ws://localhost:10443` 与 AIRI 通信，不直接调用 LLM。

## 开发阶段

| Phase | 内容 | 状态 | 平台 |
|-------|------|------|------|
| **Phase 1** | VAD 语音检测 | ✅ 代码完成 | Windows Realtek DSP 阻塞 |
| **Phase 2** | STT 语音识别（Faster-Whisper） | ✅ 完成 | Linux + Windows 双平台 67/67 |
| **Phase 3** | TTS 语音合成（CosyVoice 2） | ✅ 完成 | Linux 67/67，Windows 全链路闭环 |
| **Phase 4** | LLM 对话集成（AIRI WebSocket） | ✅ 代码完成 | Linux 201 测试，Windows 单元验证通过 |
| **Phase 5** | 打断机制 | 🚧 已预埋 `tts_mgr.stop()` | 待实现 |
| **Phase 6** | 产品级体验 | 📋 待规划 | — |

**当前阻塞项**：Phase 1 Windows 端 Realtek 声卡 DSP 降噪导致 VAD 概率归零，需禁用音频增强或外接 USB 麦克风。全链路端到端测试依赖此阻塞解决。

## 快速开始

### 环境要求

- Python 3.11+（Windows 目标机 3.13，Linux 开发机 3.14）
- Windows 11（目标运行平台）
- NVIDIA GPU (CUDA 12.x) 加速 STT/TTS（可选，CPU 也可运行）

### 安装

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/macOS

# 基础依赖（VAD/STT/播放/配置）
pip install -r requirements.txt

# CosyVoice 2 TTS（需从 GitHub 源码 + 依赖清单，详见 docs/PHASE-3-TTS-DESIGN.md）
# 关键依赖版本锁定：
#   transformers==4.51.3  （4.55.x 会导致 LLM 重复生成乱码）
#   x-transformers==2.11.24
```

### 运行

```bash
python -m src.main --test-vad      # VAD 测试
python -m src.main --test-stt      # STT 测试
python -m src.main --test-tts      # TTS 交互测试
python -m src.main --test-tts-no-play  # TTS 测试（不播放）
python -m src.main                 # 全链路（需 AIRI 在线）
```

### 配置

编辑 `config/default.yaml` 或通过环境变量覆盖：

```bash
AIRI_HOST=192.168.1.100 python -m src.main
TTS_MODEL_DIR="D:\...\CosyVoice2-0.5B" python -m src.main
```

## 项目结构

```
src/
├── main.py                  # 入口：CLI 模式分发 + _run_full() 全链路
├── config.py                # 配置加载（YAML + 环境变量覆盖）
├── logger.py                # 日志
├── audio/
│   ├── capture.py           # 麦克风捕获（sounddevice）
│   ├── playback.py          # 扬声器播放（含采样率重采样 + 线程安全）
│   └── resampler.py         # 48kHz→16kHz 重采样
├── vad/
│   └── silero_vad.py        # Silero VAD + 状态机
├── stt/
│   ├── faster_whisper_stt.py    # Faster-Whisper 引擎
│   └── post_processor.py        # 文本后处理
├── tts/
│   ├── tts_engine.py        # TTS 接口抽象
│   ├── cosyvoice_tts.py     # CosyVoice 2 引擎（零样本合成）
│   └── tts_manager.py       # 合成→缓存→播放编排
├── airi/
│   ├── websocket_client.py  # AIRI WS 客户端
│   └── conversation.py      # 对话上下文管理（Phase 4）
└── pipeline/
    ├── audio_pipeline.py    # 三协程编排
    └── ring_buffer.py       # 环形缓冲

tests/
├── test_vad.py / test_stt.py / test_tts.py ...  # 各阶段单元测试
├── diagnose_cosyvoice.py    # TTS 诊断脚本（官方 API 二分定位）
└── transcribe_wavs.py       # 本地 STT 转写脚本
```

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/ARCHITECTURE.md` | 总体架构 |
| `docs/PHASE-1-VAD-DESIGN.md` + `PHASE-1-VAD-DETAILED-DESIGN.md` + `PHASE-1-VAD-TEST-REPORT.md` | Phase 1 VAD 总体/详细设计 + 测试 |
| `docs/PHASE-2-STT-DESIGN.md` + `PHASE-2-STT-DETAILED-DESIGN.md` + `PHASE-2-STT-TEST-REPORT.md` | Phase 2 STT 总体/详细设计 + 测试 |
| `docs/PHASE-3-TTS-DESIGN.md` + `PHASE-3-TTS-DETAILED-DESIGN.md` + `PHASE-3-TTS-TEST-REPORT.md` | Phase 3 TTS 总体/详细设计 + 测试 |
| `docs/PHASE-4-LLM-DESIGN.md` + `PHASE-4-LLM-DETAILED-DESIGN.md` + `PHASE-4-LLM-TEST-REPORT.md` | Phase 4 LLM 总体/详细设计 + 测试 |
| `docs/PHASE-3-WINDOWS-ISSUES.md` | Phase 3 Windows 部署问题记录 |
| `docs/archive/` | 历史快照（会话报告、进度快照、中间设计） |

## 许可证

MIT
