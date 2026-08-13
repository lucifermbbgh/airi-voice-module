# AIRI 语音对话模块 — Phase 1 方案设计

> **日期**: 2026-07-21
> **目标平台**: Windows 11 (AIRI Electron 应用主机)
> **项目路径**: `airi-voice-module/`

---

## 一、系统架构总览

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       Windows 11 主机                            │
│                                                                  │
│  ┌─────────────────────────┐     ┌───────────────────────────┐  │
│  │    AIRI (Electron App)  │     │  AIRI Voice Module        │  │
│  │                         │     │  (Python Backend)         │  │
│  │  ┌───────────────────┐  │     │                           │  │
│  │  │  LLM Orchestrator │  │     │  ┌─────────────────────┐ │  │
│  │  │  TTS Pipeline     │  │◄────┼──┤ WebSocket Client    │ │  │
│  │  │  Plugin Protocol  │  │     │  └─────────┬───────────┘ │  │
│  │  └───────────────────┘  │     │            │             │  │
│  └─────────────────────────┘     │  ┌─────────▼───────────┐ │  │
│                                  │  │ Audio Stream Router │ │  │
│                                  │  │  (Asyncio Pipeline) │ │  │
│                                  │  └──┬──────────────┬───┘ │  │
│                                  │     │              │     │  │
│                                  │  ┌──▼──┐      ┌────▼──┐ │  │
│                                  │  │ VAD │      │Sound- │ │  │
│                                  │  │     │      │device │ │  │
│                                  │  └─────┘      └───┬───┘ │  │
│                                  └────────────────────┼─────┘  │
│                                                       │        │
│                                              ┌────────▼──────┐ │
│                                              │  Mic / Speaker│ │
│                                              │  (WASAPI)      │ │
│                                              └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 全链路数据流（6 个 Phase 总览）

```
[ 麦克风 ] ─→ [ VAD ] ─→ [ STT ] ─→ [ LLM ] ─→ [ TTS ] ─→ [ 扬声器 ]
   Phase 1     Phase 1     Phase 2      Phase 4      Phase 3      Phase 1
                                          │
                                     [ SillyTavern /
                                       Ollama / API ]
                                        Phase 4
```

### Phase 1 范围（绿色部分）

```
[ 麦克风 ] ─→ [ 16kHz重采样 ] ─→ [ 环形缓冲区 ] ─→ [ Silero VAD ] ─→ [ 语音事件 ]
     │                              Phase 1                               │
     │                                                                     │
     └─────────────────────────────────────────────────────────────────────┘
                               [ 扬声器播放 ]
```

---

## 二、技术选型

### 核心依赖

| 组件 | 库 | 版本 | 选型理由 |
|------|----|------|---------|
| 音频 I/O | `sounddevice` | ≥0.5.0 | 跨平台、Pythonic API、底层 PortAudio、低延迟 WASAPI 支持 |
| VAD | `silero-vad` | ≥4.0 | 业界最佳流式 VAD、ONNX Runtime、支持中文/英文 |
| 音频处理 | `numpy` | ≥1.26 | 音频缓冲区操作的标配 |
| 重采样 | `scipy.signal` | ≥1.14 | 48kHz→16kHz 高质量降采样 |
| 异步 | `asyncio` (内置) | Python 3.12+ | 三协程流水线架构 |
| 日志 | `loguru` | ≥0.7 | 结构化日志、自动轮转 |
| WebSocket | `websockets` | ≥12.0 | 连接 AIRI 插件协议 |
| 配置 | `pyyaml` | ≥6.0 | YAML 配置文件 |
| 音频编码 | `struct` (内置) | — | WAV/PCM 编码 |

### VAD 选型对比

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **Silero VAD v5** ✅ | 最佳准确率、流式支持、ONNX 推理快、支持中英文 | 需 ONNX Runtime (~200MB) | **首选** |
| WebRTC VAD | 轻量、无依赖延迟低 | 准确率差、不支持中文 | 备选降级 |
| Python `webrtcvad` | 安装简单 | 维护少、效果一般 | 不推荐 |

### 音频格式标准

| 属性 | 值 | 说明 |
|------|-----|------|
| 采样率 | 16 kHz | VAD/STT/TTS 通用标准 |
| 位深 | 16-bit | PCM 有符号整数 |
| 声道 | 单声道 (Mono) | 语音输入输出标准 |
| 帧长 | 512 samples (32ms) | VAD 处理单元 |
| VAD 步长 | 256 samples (16ms) | Silero VAD 滑动窗口 |

---

## 三、模块设计

### 3.1 音频捕获模块 (`src/audio/capture.py`)

```
sounddevice.InputStream
        │
        ▼
┌───────────────────────┐
│   回调函数 (callback)  │  ← 每帧被 sounddevice 调用
│  frames: 512 @ 48kHz  │
└───────┬───────────────┘
        │ 原始音频帧
        ▼
┌───────────────────────┐
│  音频捕获器 (Capture)  │
│  1. 接收 sounddevice  │
│     回调数据           │
│  2. 转换为 float32     │
│  3. 推入 asyncio.Queue │
└───────┬───────────────┘
        │ 48kHz float32 numpy array
        ▼
```

**接口设计**:
```python
class AudioCapture:
    """麦克风音频捕获"""
    
    def __init__(self, device_id: int | None, 
                 sample_rate: int = 48000,
                 frames_per_buffer: int = 512):
        ...
    
    async def start(self) -> AsyncIterator[np.ndarray]:
        """启动捕获，返回音频帧异步迭代器"""
    
    async def stop(self):
        """停止捕获"""
    
    @staticmethod
    def list_devices() -> list[dict]:
        """列出可用音频设备"""
```

### 3.2 音频播放模块 (`src/audio/playback.py`)

```
┌───────────────────────┐
│  播放器 (Playback)     │
│  1. 接收 AudioSegment  │  ← 来自 TTS/测试音频
│  2. 推入播放队列       │
│  3. sounddevice 播放   │
│  4. 支持打断           │
└───────┬───────────────┘
        │
sounddevice.OutputStream
        │
        ▼
     扬声器
```

**接口设计**:
```python
class AudioPlayback:
    """音频播放引擎"""
    
    async def play(self, audio: np.ndarray, sample_rate: int):
        """播放音频片段（非阻塞）"""
    
    async def stop(self):
        """停止当前播放"""
    
    async def wait_for_completion(self):
        """等待播放完毕"""
```

### 3.3 重采样器 (`src/audio/resampler.py`)

```
┌───────────────────────┐
│     Resampler         │
│  48kHz int16/float32  │
│       ↓               │
│  16kHz float32        │  ← VAD 需要的格式
└───────────────────────┘
```

使用 `scipy.signal.resample` 或 `sounddevice` 内置重采样。

### 3.4 环形缓冲区 (`src/pipeline/ring_buffer.py`)

```
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  F0  │  F1  │  F2  │  F3  │  F4  │  F5  │  F6  │  F7  │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
   ^               ^
   │               │
 write_idx      read_idx (滑动窗口)
 
- 生产者: 麦克风捕获 (48kHz, 512样本/帧)
- 消费者: VAD 处理 (16kHz, 512样本/帧)
- 使用 asyncio.Queue 实现线程安全
- 两阶段缓冲:
  Stage 1: 原始 48kHz 环形缓冲区（防止回调溢出）
  Stage 2: 重采样后 16kHz asyncio.Queue（VAD 消费）
```

### 3.5 Silero VAD 模块 (`src/vad/silero_vad.py`)

```
┌──────────────────────────────────────┐
│         VAD 检测器 (VADDetector)      │
│                                      │
│  16kHz float32 ─→ SileroVAD ONNX ─→  │
│                  每 512 样本帧        │
│                                      │
│  输出:                                │
│  - speech_prob: float (0.0~1.0)      │
│  - is_speech: bool (≥threshold)      │
│                                      │
│  状态机:                              │
│  ┌────────┐  prob≥thresh  ┌────────┐ │
│  │ SILENCE │─────────────→│ SPEECH │ │
│  │         │←────────────│        │ │
│  └────────┘  prob<thresh  └────────┘ │
│     │  dur≥min_silence      │        │
│     │  → SpeechEndEvent     │        │
│     └───────────────────────┘        │
│                            │         │
│                     dur≥min_speech   │
│                     → SpeechStart    │
└──────────────────────────────────────┘
```

**VAD 参数**:

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `threshold` | 0.5 | 语音/非语音判定阈值 |
| `min_speech_duration` | 0.25s | 最小语音段长度（防误触发） |
| `min_silence_duration` | 0.5s | 最小静音段长度（判定说话结束） |
| `frame_size` | 512 | 每帧样本数 @16kHz |

**语音事件**:

```python
@dataclass
class SpeechEvent:
    class Type(Enum):
        SPEECH_START = "speech_start"     # 开始说话
        SPEECH_END = "speech_end"         # 结束说话
        SPEECH_CHUNK = "speech_chunk"     # 语音中间帧（可选）
    
    type: Type
    audio_chunk: np.ndarray | None       # 语音音频数据
    timestamp: float                     # 事件时间戳
    duration: float                      # 语音段长度（仅 END）
```

### 3.6 流水线编排 (`src/pipeline/audio_pipeline.py`)

中心化的 asyncio 三协程架构：

```python
class AudioPipeline:
    """音频处理管道（三协程流水线）"""
    
    async def run(self):
        """主入口，启动三个并发协程"""
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._capture_loop())
            tg.create_task(self._vad_loop())
            tg.create_task(self._playback_loop())
    
    async def _capture_loop(self):
        """1️⃣ 捕获循环：
           从麦克风读取音频帧 → 原始缓冲
        """
    
    async def _vad_loop(self):
        """2️⃣ VAD 循环：
           从重采样队列读取 → VAD 检测 → 语音事件
        """
    
    async def _playback_loop(self):
        """3️⃣ 播放循环：
           监听播放队列 → 输出到扬声器
        """
```

### 3.7 AIRI WebSocket 客户端 (`src/airi/websocket_client.py`)

Phase 1 实现基础连接和心跳，为后续 Phase 配置通信通道。

```python
class AIRIClient:
    """AIRI 插件协议 WebSocket 客户端"""
    
    async def connect(self, url: str, token: str):
        """连接到 AIRI WebSocket 服务器"""
    
    async def send_input_text(self, text: str):
        """发送 input:text 事件（Phase 4 使用）"""
    
    async def send_input_text_voice(self, text: str, 
                                     audio: bytes | None = None):
        """发送 input:text:voice 事件（Phase 2+/4 使用）"""
    
    async def listen(self) -> AsyncIterator[dict]:
        """监听 AIRI 事件（TTS 音频等）"""
```

---

## 四、项目结构

```
airi-voice-module/
├── README.md                         # 项目简介 + 安装说明
├── requirements.txt                  # pip 依赖清单
├── pyproject.toml                    # 项目元数据 + 打包配置
├── config/
│   └── default.yaml                  # 默认配置（所有 Phase）
├── docs/
│   ├── PHASE-1-DESIGN.md             # 本文件 - Phase 1 设计
│   ├── PHASE-2-STT.md                # Phase 2 设计（后续）
│   ├── PHASE-3-TTS.md                # Phase 3 设计（后续）
│   ├── PHASE-4-LLM.md                # Phase 4 设计（后续）
│   ├── PHASE-5-INTERRUPT.md          # Phase 5 设计（后续）
│   └── PHASE-6-PRODUCTION.md         # Phase 6 设计（后续）
├── src/
│   ├── __init__.py
│   ├── main.py                       # 入口：CLI + 启动管道
│   ├── config.py                     # 配置加载器
│   ├── logger.py                     # 日志配置
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py                # 麦克风输入
│   │   ├── playback.py               # 扬声器输出
│   │   └── resampler.py              # 音频重采样
│   ├── vad/
│   │   ├── __init__.py
│   │   └── silero_vad.py             # Silero VAD 集成
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── audio_pipeline.py         # asyncio 管道编排
│   │   └── ring_buffer.py            # 音频环形缓冲
│   └── airi/
│       ├── __init__.py
│       └── websocket_client.py       # AIRI WebSocket 连接
└── tests/
    ├── __init__.py
    ├── test_capture.py
    ├── test_vad.py
    └── test_pipeline.py
```

---

## 五、配置文件 (`config/default.yaml`)

```yaml
# AIRI Voice Module - 默认配置

audio:
  # 麦克风
  input_device: null           # null = 系统默认
  sample_rate: 48000           # 麦克风原始采样率 (Hz)
  frames_per_buffer: 512       # 每帧样本数
  channels: 1                  # 单声道
  
  # 扬声器
  output_device: null          # null = 系统默认
  output_sample_rate: 24000    # TTS 输出采样率 (Hz)
  
  # 重采样
  target_sample_rate: 16000    # VAD/STT 目标采样率

vad:
  model_path: "models/silero_vad.onnx"  # 模型路径
  threshold: 0.5                         # VAD 阈值
  min_speech_duration: 0.25             # 最短语音段 (秒)
  min_silence_duration: 0.5             # 最短静音段 (秒)
  frame_size: 512                        # 帧大小 @16kHz

airi:
  host: "localhost"
  port: 10443                    # AIRI WebSocket 端口 (默认)
  token: ""                      # 认证令牌
  reconnect_interval: 5          # 重连间隔 (秒)
  max_reconnect_attempts: 0      # 0 = 无限重试

logging:
  level: "DEBUG"
  format: "{time:HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} | {message}"
  file: "logs/voice-module.log"
  rotation: "10 MB"

pipeline:
  speech_buffer_max_duration: 10.0  # 语音缓冲最大时长 (秒)
```

---

## 六、Phase 1 实施步骤

| 步骤 | 模块 | 预计工时 | 说明 |
|------|------|---------|------|
| **Step 1** | 项目脚手架 | 0.5h | 创建目录结构、requirements.txt、pyproject.toml |
| **Step 2** | 配置加载 | 0.5h | config.py 实现 YAML 加载 + 类型校验 |
| **Step 3** | 音频捕获 | 1.5h | AudioCapture: sounddevice InputStream → asyncio.Queue |
| **Step 4** | 重采样器 | 0.5h | 48kHz→16kHz 降采样 (scipy.signal.resample_poly) |
| **Step 5** | 环形缓冲区 | 0.5h | 线程安全的 asyncio 音频缓冲 |
| **Step 6** | Silero VAD | 2h | 模型下载 + ONNX Runtime + 语音事件检测 |
| **Step 7** | 音频播放 | 1h | AudioPlayback: sounddevice OutputStream |
| **Step 8** | 管道编排 | 1.5h | AudioPipeline: 三协程编排 + 事件路由 |
| **Step 9** | AIRI WebSocket | 1h | 基础连接 + 心跳 + 认证 |
| **Step 10** | 集成测试 | 1h | 端到端验证：Capture→VAD→事件输出 |
| **Step 11** | 配置调优 | 0.5h | VAD 参数调优、延迟优化 |
| | **合计** | **~10.5h** | |

---

## 七、AIRI 集成接口

### WebSocket 事件协议

Voice Module 作为 AIRI 插件运行，使用以下事件：

| 事件方向 | 事件类型 | 说明 | 使用 Phase |
|---------|---------|------|-----------|
| Module→AIRI | `input:text:voice` | 语音转文字后发送 | Phase 2+ |
| Module→AIRI | `input:voice` | 原始音频发送 | Phase 5 打断 |
| AIRI→Module | `output:gen-ai:chat:message` | LLM 流式文字输出 | Phase 3+ |
| AIRI→Module | `output:gen-ai:chat:complete` | LLM 完成事件 | Phase 3+ |

### 生命周期流程

```
Phase 1 实现:
┌────────┐   ┌──────────┐   ┌────────┐
│ 连接AIRI│──→│ 音频捕获 │──→│ VAD检测│
│ (WS)   │   │ (Mic)    │   │        │
└────────┘   └──────────┘   └──┬─────┘
                               │ speech_start/speech_end
                               ▼
                         ┌──────────────┐
                         │ 控制台输出/   │
                         │ 日志记录      │
                         └──────────────┘
```

---

## 八、性能指标 (Phase 1 目标)

| 指标 | 目标 | 测量方式 |
|------|------|---------|
| 端到端延迟 | <100ms | 麦克风输入 → VAD 事件输出 |
| VAD 推理延迟 | <5ms/帧 | 每 512 样本帧推理时间 |
| CPU 占用 | <15% (单核) | Windows 任务管理器 |
| 内存占用 | <300MB | Python 进程 (含模型) |
| 音频丢帧率 | <0.1% | 声卡回调成功/失败比 |
| VAD 准确率 | >90% | 在测试语音集上的 F1 分数 |

---

## 九、风险 & 缓解措施

| 风险 | 影响 | 概率 | 缓解 |
|------|------|------|------|
| ONNX Runtime 对 CPU 要求高 | VAD 延迟上升 | 低 | 使用 `int8` 量化模型，或降级到 WebRTC VAD |
| 声卡独占 | 无法同时捕获 | 中 | 使用共享模式，避免 WASAPI Exclusive |
| Python GIL 造成音频中断 | 音频丢失 | 低 | 使用 sounddevice 原生回调（C level） |
| 模型下载失败 | VAD 无法使用 | 低 | 预下载到 `models/` 目录，提供离线安装包 |

---

## 十、代码规范

1. **类型注解**: 所有函数使用 Python type hints
2. **异步优先**: 所有 IO 操作用 `async/await`
3. **错误处理**: `try/except` 包围所有外部调用，日志记录异常
4. **配置化**: 所有可变参数来自 YAML 配置，无硬编码
5. **文档字符串**: NumPy 风格 docstring
6. **测试覆盖**: `pytest` + `pytest-asyncio`

---

> **下一步**: 完成设计文档审查后，开始 Step 1 项目脚手架搭建。
