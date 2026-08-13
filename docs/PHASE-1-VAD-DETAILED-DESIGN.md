# AIRI 语音对话模块 — Phase 1 VAD 详细实现设计

> **日期**: 2026-08-13
> **版本**: 1.0 (实现完成，离线测试通过；Windows 实时 VAD 受 Realtek DSP 阻塞)
> **对应代码**: `src/audio/capture.py`、`src/audio/resampler.py`、`src/pipeline/ring_buffer.py`、`src/vad/silero_vad.py`、`src/pipeline/audio_pipeline.py`
> **单元测试**: `tests/test_vad.py` (9 项)
> **总体设计**: 见 `PHASE-1-VAD-DESIGN.md`

---

## 目录

1. [总体架构](#一总体架构)
2. [核心数据模型](#二核心数据模型)
3. [音频捕获模块](#三音频捕获模块)
4. [重采样器](#四重采样器)
5. [环形缓冲区](#五环形缓冲区)
6. [Silero VAD 状态机](#六silero-vad-状态机)
7. [三协程编排](#七三协程编排)
8. [测试设计](#八测试设计)
9. [已知问题](#九已知问题)

---

## 一、总体架构

Phase 1 负责语音链路的最前端：**从麦克风采集音频 → VAD 检测语音活动 → 产出语音事件**。

```
[麦克风] → [AudioCapture] → [Resampler 48k→16k] → [AudioRingBuffer] → [SileroVAD] → SpeechEvent
                                                                              │
                                                                    SPEECH_START / SPEECH_END
```

### 1.1 模块分层

| 模块 | 文件 | 职责 |
|:-----|:-----|:-----|
| 音频捕获 | `src/audio/capture.py` (200 行) | sounddevice InputStream 采集麦克风音频 |
| 重采样器 | `src/audio/resampler.py` (133 行) | 48kHz → 16kHz（VAD 输入标准） |
| 环形缓冲 | `src/pipeline/ring_buffer.py` (126 行) | 线程安全的音频帧缓冲（历史回看） |
| VAD 检测 | `src/vad/silero_vad.py` (423 行) | Silero ONNX 模型 + 语音状态机 |
| 编排 | `src/pipeline/audio_pipeline.py` (245 行) | capture/vad/playback 三协程并发 |

---

## 二、核心数据模型

### 2.1 枚举

```python
class SpeechState(Enum):
    SILENCE = "silence"              # 静音
    SPEECH = "speech"                # 确认的语音
    PENDING_START = "pending_start"  # 等待 min_speech_duration 确认
    PENDING_END = "pending_end"      # 等待 min_silence_duration 确认（预埋）

class SpeechEventType(Enum):
    SPEECH_START = "speech_start"    # 语音开始
    SPEECH_END = "speech_end"        # 语音结束（含完整音频）
    UTTERANCE = "utterance"          # 完整话语（含完整音频）
```

### 2.2 `SpeechEvent` — 语音事件

```python
@dataclass
class SpeechEvent:
    type: SpeechEventType
    audio: np.ndarray | None = None   # 语音片段（仅 END/UTTERANCE）
    timestamp: float = 0.0
    duration: float = 0.0             # 片段时长（仅 END）
    num_frames: int = 0               # VAD 帧数
    max_prob: float = 0.0             # 片段内最大语音概率
    sample_rate: int = 16000
```

---

## 三、音频捕获模块

### 3.1 `AudioCapture` 设计

```python
class AudioCapture:
    def __init__(self, device=None, sample_rate=48000, channels=1,
                 block_size=4800, ring_buffer_max_history=100):
        ...
```

- 用 `sounddevice.InputStream` 后台线程采集
- 采集帧推入 `AudioRingBuffer`
- 通过 `read_frames()` 异步迭代器供消费方读取

### 3.2 采集参数

| 参数 | 值 | 说明 |
|:-----|:--:|:-----|
| sample_rate | 48000 | 麦克风原生采样率 |
| channels | 1 | 单声道 |
| block_size | 4800 | 100ms 块（48000 × 0.1） |

---

## 四、重采样器

### 4.1 `Resampler` 设计

```python
class Resampler:
    def __init__(self, source_rate=48000, target_rate=16000):
        ...
    def resample(self, data) -> np.ndarray:      # 单帧重采样
    def resample_buffer(self, data) -> np.ndarray:  # 缓冲重采样
    @staticmethod
    def to_float32(data) -> np.ndarray:          # int16 → float32
    @staticmethod
    def to_int16(data) -> np.ndarray:            # float32 → int16
```

重采样基于线性插值，将麦克风的 48kHz 降到 VAD 需要的 16kHz。

---

## 五、环形缓冲区

### 5.1 `AudioRingBuffer` 设计

```python
class AudioRingBuffer:
    def __init__(self, max_history=100):   # 保留最近 100 帧
        ...
    def write_raw(self, frame, sample_rate)     # 写入原始帧
    async def write_processed(self, frame)      # 写入处理后的帧
    async def read(self) -> AsyncIterator       # 异步读取迭代器
    def get_raw_history(self) -> list           # 获取历史帧列表
    def get_raw_concatenated(self) -> np.ndarray  # 拼接历史帧
```

### 5.2 设计意图

环形缓冲的关键价值是**历史回看**：当 VAD 确认"语音开始"时，之前处于
PENDING_START 的帧已经累积在缓冲里，可以拼接回完整话语，避免丢失语音开头。

---

## 六、Silero VAD 状态机

### 6.1 构造参数

```python
SileroVAD(
    model_path="models/silero_vad.onnx",
    threshold=0.5,               # 语音概率阈值
    min_speech_duration=0.25,    # 最短语音确认时长（s）
    min_silence_duration=0.5,    # 结束话语的静音时长（s）
    frame_size=512,              # 帧大小（样本数）
    sample_rate=16000,           # 输入采样率
)
```

内部换算：`min_speech_frames = min_speech_duration * sample_rate / frame_size`
（0.25 × 16000 / 512 ≈ 8 帧），`min_silence_frames` 同理（0.5 × 16000 / 512 ≈ 16 帧）。

### 6.2 状态机转换

```
            ┌────────────────────────────────────────────┐
            │                  SILENCE                    │
            └──────────────┬─────────────────────────────┘
                           │ is_speech（prob > threshold）
                           ▼
            ┌────────────────────────────────────────────┐
            │               PENDING_START                 │
            │        累积 speech_frames 计数              │
            └──────┬──────────────────────┬──────────────┘
        ≥ min_speech_frames │              │ is_speech=False（噪音尖峰）
                           ▼              ▼
            ┌────────────────────┐   ┌──────────┐
            │       SPEECH       │   │ SILENCE  │（false start 取消）
            │ 持续累积 + 静音计数 │   └──────────┘
            └──────────┬─────────┘
        ≥ min_silence_frames 连续静音
                       ▼
              ┌────────────────┐
              │ SPEECH_END 事件 │ → _reset() → SILENCE
              └────────────────┘
```

### 6.3 关键设计

| 设计点 | 说明 |
|:-------|:-----|
| **false start 抑制** | PENDING_START 阶段若出现静音帧，判定为噪音尖峰，回退 SILENCE |
| **语音开头保留** | PENDING_START 即开始累积 `_speech_frames`，确认 SPEECH 后无缝拼接 |
| **尾部静音容忍** | SPEECH 阶段连续 `min_silence_frames`（16 帧 ≈ 0.5s）静音才判定结束，容忍句中停顿 |
| **状态化 ONNX** | `_hn` 保存 LSTM 隐状态，跨帧传递，提升连续帧判定准确性 |

### 6.4 概率计算

`_get_speech_prob(audio_frame)` 用 Silero ONNX 模型对单帧推理，输出语音概率。
Silero 模型输入要求 16kHz 单声道 float32（帧大小 512 样本）。

---

## 七、三协程编排

### 7.1 `AudioPipeline` 设计

```python
class AudioPipeline:
    def __init__(self, config): ...
    def on_speech_event(self, callback): ...   # 注册语音事件回调
    async def _capture_loop(self): ...         # 采集 → 重采样 → 缓冲
    async def _playback_loop(self): ...        # 播放队列消费
    async def start(self): ...                 # 启动所有协程
    async def stop(self): ...                  # 停止
    async def play_audio(self, audio, ...): ...  # 播放（TTS 输出）
```

### 7.2 协程协作

```
_capture_loop ──► AudioRingBuffer.write ──► VAD.process_frame ──► SpeechEvent
                                                                       │
                                                        on_speech_event 回调
                                                                       │
                                                        SPEECH_START → 打断预备
                                                        SPEECH_END   → STT 转写（Phase 2）
```

VAD 检测在 capture_loop 内联执行（逐帧），语音事件通过回调分发。

---

## 八、测试设计

| 文件 | 测试数 | 说明 |
|:-----|:------:|:-----|
| `tests/test_vad.py` | 9 | VAD 状态机单元测试（离线合成音频） |

覆盖点：状态机转换（SILENCE→PENDING_START→SPEECH→END）、false start 抑制、
事件字段完整性、ONNX 模型加载。

**平台结果**：Linux 9/9 ✅；Windows 离线 9/9 ✅（真实麦克风受 Realtek DSP 阻塞）。

---

## 九、已知问题

### 9.1 Realtek DSP 降噪（阻塞项）

**现象**：Windows 端真实麦克风输入时，Silero VAD 概率归零（仅 7.87%，远低于 0.5 阈值）。

**根因**：Realtek(R) Audio 声卡驱动的 DSP 增强功能滤除高频语音成分，导致 Silero 模型
无法识别语音。离线合成音频测试 9/9 通过，证明代码正确，问题在硬件驱动的信号预处理。

**解决方案（三选一）**：
1. 禁用音频增强：`mmsys.cpl` → 录制 → 麦克风属性 → 增强 → 全部禁用
2. 外接 USB 麦克风（绕过 Realtek 声卡，推荐）
3. VAD 前加高通滤波（代码层，效果有限，DSP 已造成信息丢失）
