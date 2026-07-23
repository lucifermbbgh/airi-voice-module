# AIRI Voice Module

实时语音对话后端模块，为 [Project AIRI](https://github.com/moeru-ai/airi) 添加语音交互能力。

## 架构概览

```
[麦克风] → [VAD] → [STT] → [WebSocket] → [AIRI LLM] → [TTS] → [扬声器]
   P1         P1        P2                     P4           P3        P1
```

## 开发阶段

| Phase | 内容 | 状态 |
|-------|------|------|
| **Phase 1** | 基础音频管道（Capture→VAD→Playback） | 🚧 进行中 |
| Phase 2 | STT 集成（Faster-Whisper） | 📋 待规划 |
| Phase 3 | TTS 集成（CosyVoice 2） | 📋 待规划 |
| Phase 4 | LLM 对话集成（AIRI WebSocket） | 📋 待规划 |
| Phase 5 | 打断机制 | 📋 待规划 |
| Phase 6 | 产品级体验 | 📋 待规划 |

## 快速开始

### 环境要求

- Python 3.12+
- Windows 11（目标运行平台）
- 可选: NVIDIA GPU (CUDA 12.x) 加速 STT/TTS

### 安装

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt

# 下载 Silero VAD 模型
python -c "import silero_vad; silero_vad.fetch_model()"
# 或手动下载到 models/ 目录
```

### 运行

```bash
# Phase 1 测试: Capture → VAD → 语音事件输出
python -m src.main --test-vad

# 完整管线
python -m src.main
```

## 配置

编辑 `config/default.yaml` 或通过环境变量覆盖：

```bash
AIRI_HOST=192.168.1.100 python -m src.main
```

## 项目结构

```
src/
├── main.py                  # 入口
├── config.py                # 配置加载
├── logger.py                # 日志
├── audio/
│   ├── capture.py           # 麦克风捕获
│   ├── playback.py          # 扬声器播放
│   └── resampler.py         # 重采样
├── vad/
│   └── silero_vad.py        # VAD 检测
├── pipeline/
│   ├── audio_pipeline.py    # 三协程编排
│   └── ring_buffer.py       # 环形缓冲
└── airi/
    └── websocket_client.py  # AIRI WebSocket
```

## 许可证

MIT
