# AIRI 语音对话模块 — Phase 2 STT 详细实现设计

> **日期**: 2026-07-23
> **版本**: 1.0 (实现完成)
> **对应代码**: `src/stt/faster_whisper_stt.py` (483 行)
> **单元测试**: `tests/test_stt.py` (203 行)
> **提交**: `93e9577`

---

## 目录

1. [总体架构](#一总体架构)
2. [核心数据模型](#二核心数据模型)
3. [STT 引擎设计](#三stt-引擎设计)
4. [transcribe() 执行流水线](#四transcribe-执行流水线)
5. [音频预处理与校验](#五音频预处理与校验)
6. [异步线程模型](#六异步线程模型)
7. [模型生命周期管理](#七模型生命周期管理)
8. [流式处理](#八流式处理)
9. [测试设计](#九测试设计)
10. [Pipeline 集成方案](#十pipeline-集成方案)
11. [性能指标](#十一性能指标)
12. [设计权衡记录](#十二设计权衡记录)

---

## 一、总体架构

### 1.1 Phase 2 在全链路中的位置

```
[ 麦克风 ] ─→ [ 重采样 ] ─→ [ VAD ] ─→ [ STT ] ─→ [ WebSocket ] ─→ [ AIRI ]
   Phase 1        Phase 1        Phase 1      Phase 2      Phase 1
                                               ↑
                                          您在这里 👈
```

### 1.2 与 Phase 1 的集成关系

```
Phase 1 已有代码                         Phase 2 新增
┌──────────────────────────┐            ┌──────────────────────────────┐
│ AudioPipeline            │            │ src/stt/                     │
│  ├─ AudioCapture         │            │  ├─ __init__.py              │
│  ├─ Resampler            │            │  └─ faster_whisper_stt.py    │
│  ├─ SileroVAD            │            │     ├─ STTResult (dataclass) │
│  └─ on_speech_event()───┼────────────┼──→ │  ├─ FasterWhisperSTT     │
│     callback            │            │    │  │  ├─ transcribe()      │
│                         │            │    │  │  ├─ _infer_sync()     │
│                         │            │    │  │  ├─ _validate_audio() │
│                         │            │    │  │  ├─ _is_silence()     │
│                         │            │    │  │  └─ load_model()      │
│                         │            │    │  └─ transcribe_stream() │
└──────────────────────────┘            └──────────────────────────────┘
         │                                          │
         │ Pipeline.on_speech_event() 绑定          │
         │ handle_speech → stt.transcribe           │
         │                                          │
         ▼                                          ▼
   ┌──────────────────────────────────────────────────────────┐
   │ main.py (入口编排)                                       │
   │  pipeline = AudioPipeline(config)                        │
   │  pipeline.on_speech_event(on_speech)  ← 挂载点          │
   │  stt = FasterWhisperSTT(model_size="small")              │
   │  airi = AIRIClient(...)                                  │
   └──────────────────────────────────────────────────────────┘
```

### 1.3 核心设计原则

| 原则 | 说明 |
|------|------|
| **零侵入集成** | Phase 2 是"插在" Phase 1 已有回调机制上的新增模块，不修改任何 Phase 1 已有代码 |
| **异步非阻塞** | CPU 密集型推理通过 `loop.run_in_executor()` 隔离到线程池，不阻塞 asyncio 事件循环 |
| **懒加载** | 模型在构造时不加载，首次 `transcribe()` 调用时自动下载并加载 |
| **错误隔离** | 所有异常在模块内部捕获并返回空 `STTResult`，管道不会被 STT 错误中断 |
| **快速通道** | 静音 / 空音频在模型推理前直接返回空结果，节省 CPU 资源 |

---

## 二、核心数据模型

### 2.1 `STTResult` — 统一结果容器

**位置**: `src/stt/faster_whisper_stt.py:44-63`

```python
@dataclass
class STTResult:
    """STT inference result.

    Attributes:
        text: Recognized text.
        confidence: Overall confidence score (0.0 to 1.0).
        language: Detected or specified language code.
        language_probability: Confidence in language detection.
        duration: Duration of input audio in seconds.
        inference_time: Time taken for inference in seconds.
        segments: Optional list of segment dicts with timestamps.
    """
    text: str
    confidence: float
    language: str
    language_probability: float
    duration: float
    inference_time: float
    segments: list[dict] | None = None
```

**设计理由**:
- 统一输出格式，下游消费方（回调函数、日志、AIRI）不依赖 Faster-Whisper 内部类型
- `segments` 为可选字段，包含每段的 `start` / `end` / `text` / `avg_logprob` / `no_speech_prob`
- 空结果（静音/错误）统一表示为 `text=""`, `confidence=0.0`

### 2.2 常量定义

**位置**: `src/stt/faster_whisper_stt.py:35-38`

| 常量 | 值 | 说明 |
|------|-----|------|
| `_STANDARD_SAMPLE_RATE` | 16000 | Whisper 标准输入采样率 |
| `_MIN_AUDIO_DURATION` | 0.1s | 最短音频时长（低于此视为噪声） |
| `_MAX_AUDIO_DURATION` | 30.0s | 最长音频时长（实时语音段不应超此值） |
| `_SILENCE_THRESHOLD_RMS` | 0.01 | 静音 RMS 阈值 |

---

## 三、STT 引擎设计

### 3.1 类定义

**位置**: `src/stt/faster_whisper_stt.py:69-93`

```python
class FasterWhisperSTT:
    """Faster-Whisper STT engine with async wrapper.

    Provides speech-to-text conversion using CTranslate2-optimized
    Whisper models. Designed for integration with the VAD pipeline's
    SpeechEvent.SPEECH_END callback.
    """

    # Predefined model specifications
    MODELS: dict[str, dict] = {
        "tiny":     {"ram_mb": 400,  "rtf": 0.3},
        "base":     {"ram_mb": 600,  "rtf": 0.2},
        "small":    {"ram_mb": 1200, "rtf": 0.1},    # ← 默认推荐
        "medium":   {"ram_mb": 2500, "rtf": 0.08},
        "large-v3": {"ram_mb": 3200, "rtf": 0.05},
    }
```

### 3.2 构造参数

**位置**: `src/stt/faster_whisper_stt.py:95-147`

```python
def __init__(
    self,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    model_dir: str | None = None,
    language: str | None = "zh",
    beam_size: int = 5,
    vad_filter: bool = True,
    hotwords: list[str] | None = None,
):
```

| 参数 | 默认值 | 说明 | 可选值 |
|------|--------|------|--------|
| `model_size` | `"small"` | 模型规格 | tiny / base / **small** / medium / large-v3 |
| `device` | `"cpu"` | 计算设备 | cpu / cuda |
| `compute_type` | `"int8"` | 量化类型 | **int8**(CPU) / float16 / float32(GPU) |
| `model_dir` | `None` | 模型缓存目录 | None(=Hub默认) / 自定义路径 |
| `language` | `"zh"` | 语言偏好 | **zh** / en / None(=自动检测) |
| `beam_size` | `5` | 束搜索宽度 | 1~10 (越大越准越慢) |
| `vad_filter` | `True` | Whisper 内置 VAD 过滤 | True / False |
| `hotwords` | `None` | 热词列表 | ["Claude", "AIRI"] |

**初始化行为**:
- 校验 `model_size` 合法性，非法值立即抛 `ValueError`
- 构造时**不加载模型**（`self._model = None`）
- 构造时**不创建线程池**（`self._executor = None`）
- 记录初始化的日志（含模型规格、预估内存、RTF），方便调试

---

## 四、transcribe() 执行流水线

### 4.1 四阶段处理流

**位置**: `src/stt/faster_whisper_stt.py:291-384`

```
调用 transcribe(audio, 16000)
        │
        ▼
┌──────────────────────────────────────────────┐
│ ① 快速通道 (Fast Path)                       │
│                                              │
│ len(audio)==0? ────→ 是 ──→ 空结果           │
│ _is_silence(audio)? ─→ 是 ──→ 空结果         │
│ (RMS < 0.01)                                 │
└──────────────────────┬───────────────────────┘
                       │ 非空 + 非静音
                       ▼
┌──────────────────────────────────────────────┐
│ ② 格式验证 (Validation)                      │
│                                              │
│ dtype == float32? ──── 否 ──→ 返回空结果     │
│ sample_rate == 16000? ─ 否 ──→ 返回空结果    │
│ 时长 0.1s~30.0s? ────── 否 ──→ 返回空结果   │
│                                              │
│ 所有不通过都走日志警告但不抛异常              │
└──────────────────────┬───────────────────────┘
                       │ 格式通过
                       ▼
┌──────────────────────────────────────────────┐
│ ③ 懒加载模型 (Lazy Load)                     │
│                                              │
│ self._model is None?                         │
│   └─→ load_model()                            │
│       ├─ 从 huggingface_hub 下载 (首次)       │
│       ├─ 加载 CTranslate2 WhisperModel        │
│       └─ 耗时 ~2s (small int8 CPU)            │
│ 已加载 → 跳过                                 │
└──────────────────────┬───────────────────────┘
                       │ 模型就绪
                       ▼
┌──────────────────────────────────────────────┐
│ ④ 异步推理 (Async Inference)                 │
│                                              │
│ 获取 asyncio 事件循环: asyncio.get_running() │
│ 创建 ThreadPoolExecutor(1, "stt")            │
│ result = await loop.run_in_executor(         │
│     executor,                                │
│     self._infer_sync,                        │
│     audio, sample_rate,                      │
│ )                                            │
│                                              │
│ 事件循环在推理期间继续处理其他协程           │
└──────────────────────┬───────────────────────┘
                       │ STTResult
                       ▼
         返回结果 + 日志输出 ("STT: 今天天气怎么样 (conf=0.92, ...)")
```

### 4.2 快速通道逻辑

```python
# 行 311-319
if len(audio) == 0 or self._is_silence(audio):
    return STTResult(
        text="", confidence=0.0,
        language=self.language or "unknown",
        language_probability=0.0,
        duration=len(audio) / sample_rate,
        inference_time=0.0,
    )
```

**静音检测实现** (行 274-287):
```python
@staticmethod
def _is_silence(audio: np.ndarray) -> bool:
    if len(audio) == 0:
        return True
    rms = np.sqrt(np.mean(audio ** 2))
    return rms < _SILENCE_THRESHOLD_RMS  # 0.01
```

### 4.3 格式验证逻辑

```python
# 行 237-272
def _validate_audio(self, audio, sample_rate) -> bool:
    # 检查项：
    # 1. dtype 必须是 float32
    # 2. sample_rate 必须是 16000
    # 3. 时长必须 >= 0.1 秒
    # 4. 时长必须 <= 30.0 秒
    return True  # 全部通过
```

### 4.4 异步推理逻辑

```python
# 行 337-384
loop = asyncio.get_running_loop()
if self._executor is None:
    from concurrent.futures import ThreadPoolExecutor
    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

infer_start = time.monotonic()
try:
    result = await loop.run_in_executor(
        self._executor, self._infer_sync, audio, sample_rate,
    )
except Exception as e:
    logger.error("STT inference error: {}", e)
    # 返回空结果，不让异常传播到管道
    return STTResult(text="", confidence=0.0, ...)
```

**异常处理策略**: 任何异常都不会传播到外部 — 返回空 `STTResult`，VAD 管道不受影响。

### 4.5 同步推理引擎 `_infer_sync()`

**位置**: `src/stt/faster_whisper_stt.py:386-447`

```python
def _infer_sync(self, audio, sample_rate) -> STTResult:
    segments, info = self._model.transcribe(
        audio=audio,
        language=self.language,              # "zh" → 中文优先
        beam_size=self.beam_size,            # 5
        vad_filter=self.vad_filter,          # True
        hotwords=" ".join(self.hotwords),    # "Claude AIRI"
        condition_on_previous_text=True,     # 利用上下文
    )

    text_parts = []
    all_segments = []
    total_confidence = 0.0
    segment_count = 0

    for seg in segments:
        text_parts.append(seg.text)
        all_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
        })
        seg_duration = seg.end - seg.start
        if seg_duration > 0:
            # avg_logprob → 0~1 置信度, 按段时长加权
            seg_confidence = max(0.0, min(1.0, 1.0 + seg.avg_logprob))
            total_confidence += seg_confidence * seg_duration
            segment_count += seg_duration

    text = " ".join(text_parts).strip()
    avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0

    return STTResult(
        text=text,
        confidence=avg_confidence,
        language=info.language,
        language_probability=info.language_probability,
        duration=duration,
        inference_time=0.0,  # 由调用者设置
        segments=all_segments if all_segments else None,
    )
```

**置信度计算**:
- Faster-Whisper 输出 `avg_logprob`（典型值 -0.5 ~ 0.0，越高越好）
- 转换公式: `confidence = max(0.0, min(1.0, 1.0 + avg_logprob))`
- 按段时长加权平均，长句占比更大

---

## 五、音频预处理与校验

### 5.1 校验矩阵

| 检查项 | 条件 | 日志级别 | 返回结果 |
|--------|------|---------|---------|
| `dtype` | `== float32` | `WARNING` | 空 `STTResult` |
| `sample_rate` | `== 16000` | `WARNING` | 空 `STTResult` |
| 最短时长 | `>= 0.1s` | `DEBUG` | 空 `STTResult` |
| 最长时长 | `<= 30.0s` | `WARNING` | 空 `STTResult` |
| 静音检测 | `RMS < 0.01` | — | 空 `STTResult` |

### 5.2 设计意图

- **dtype 检查**: Whisper 模型输入要求 float32，不转换直接抛错会崩溃
- **采样率检查**: 16kHz 是 Whisper 训练标准，非标准采样率导致识别率大幅下降
- **最短时长**: 太短的音频（<0.1s）不可能是有效语音，通常是麦克风噪声
- **最长时长**: 实时对话中 VAD 切段不应超过 30s，超长音频通常表示 VAD 异常
- **静音检测**: 在模型推理前快速过滤，节省 CPU 资源

---

## 六、异步线程模型

### 6.1 架构

```
主事件循环 (asyncio)                STT 线程池 (ThreadPoolExecutor)
┌──────────────────────────┐       ┌───────────────────────────┐
│ AudioPipeline            │       │  stt (max_workers=1)      │
│  ├─ _capture_loop (协程) │       │                           │
│  ├─ _playback_loop (协程)│       │  _infer_sync()            │
│  └─ on_speech()    (协程)│──────→│  ├─ model.transcribe()    │
│     └─ stt.transcribe()  │       │  ├─ CTranslate2 推理     │
│        └─ transcribe()   │       │  ├─ segments 解析        │
│           └─run_in_exec  │       │  └─ STTResult 构造       │
│                 utor()  ←│───────┤                           │
└──────────────────────────┘       └───────────────────────────┘
```

### 6.2 设计决策

| 决策 | 值 | 理由 |
|------|----|------|
| 线程数 | `max_workers=1` | CTranslate2 内部已使用 OpenMP 多线程，外层不再需要多线程 |
| 线程名 | `"stt"` | 方便调试和 CPU profiling |
| 创建时机 | 首次推理时（懒创建） | 避免构造时创建无用的线程 |
| 销毁时机 | `cleanup()` 调用时 | 显示生命周期管理 |

### 6.3 为什么不用 `asyncio.to_thread()`

- 需要持有一个专用的 `ThreadPoolExecutor` 以便在 `cleanup()` 时显式关闭
- `run_in_executor` 提供了同样的 API，但拥有更多控制权

---

## 七、模型生命周期管理

### 7.1 生命周期状态图

```
  ┌──────────────┐
  │   未初始化   │
  │ _model=None  │
  └──────┬───────┘
         │ transcribe() 被调用
         ▼
  ┌──────────────┐   首次     ┌──────────────────┐
  │   加载中     │──────────→│  已加载 (Ready)   │
  │ load_model() │ 下载+加载  │  _model=Whisper   │
  └──────────────┘           └──────────────────┘
                                    │
                                    │ cleanup()
                                    ▼
                            ┌──────────────┐
                            │   已卸载      │
                            │ _model=None   │
                            │ executor.stop │
                            └──────────────┘
```

### 7.2 方法说明

| 方法 | 位置 | 说明 |
|------|------|------|
| `load_model()` | 行 151-188 | 从 HuggingFace Hub 下载并加载模型（首次），或从缓存加载 |
| `unload_model()` | 行 190-193 | 释放模型内存（`self._model = None`） |
| `is_loaded` (property) | 行 195-198 | 检查模型是否加载 |
| `model_info` (property) | 行 200-210 | 返回模型元信息（大小/设备/量化/是否加载） |
| `cleanup()` | 行 477-483 | 卸载模型 + 关闭线程池 |

### 7.3 懒加载的优缺点

| 优点 | 缺点 |
|------|------|
| 构造时零开销 | 首次 `transcribe()` 会延迟 ~2s |
| 不使用的配置不会浪费内存 | 延迟发生在对话中，可能影响首次体验 |
| 便于单元测试（不加载模型） | 依赖网络下载（可预下载到 `models/` 目录缓解） |

---

## 八、流式处理

### 8.1 `transcribe_stream()` — 多段处理

**位置**: `src/stt/faster_whisper_stt.py:449-473`

```python
async def transcribe_stream(
    self,
    audio_chunks: AsyncIterator[tuple[np.ndarray, int]],
) -> AsyncIterator[STTResult]:
    async for audio, sample_rate in audio_chunks:
        result = await self.transcribe(audio, sample_rate)
        if result.text:  # Only yield non-empty results
            yield result
```

### 8.2 使用场景

```
VAD 输出:  SPEECH_END(段1)  →  SPEECH_END(段2)  →  SPEECH_END(段3)
               │                   │                   │
               ▼                   ▼                   ▼
transcribe_stream:  文字1     →    文字2         →     文字3
```

- 用户连续说多句话时，VAD 会输出多个 `SPEECH_END` 事件
- 每段独立推理，逐段输出结果
- 静音段自动过滤（空文字不 yield）

---

## 九、测试设计

### 9.1 测试架构

**位置**: `tests/test_stt.py` (203 行)

```
test_stt.py
├── Fixtures
│   ├── stt()             → FasterWhisperSTT(tiny, model_dir=临时目录)
│   └── sample_audio()    → 1秒 440Hz 正弦波 + 噪声 (模拟语音)
│
├── TestSTTResult                     (集成测试前验)
│   ├── test_minimal_result           → dataclass 基础构造
│   ├── test_result_with_segments     → 带时间戳分段的构造
│   └── test_result_empty_text        → 空文本/空结果表示
│
├── TestFasterWhisperSTTInit          (构造测试)
│   ├── test_default_initialization   → 默认参数验证
│   ├── test_model_not_loaded_by_default → 懒加载验证
│   ├── test_valid_model_sizes        → 所有合法模型大小
│   ├── test_invalid_model_size       → 非法大小 → ValueError
│   └── test_hotwords                 → 热词存储
│
├── TestAudioPreprocessing            (预处理测试)
│   ├── test_valid_audio_format       → 合法音频
│   ├── test_invalid_sample_rate      → 48000Hz → 拒绝
│   ├── test_invalid_dtype            → int16 → 拒绝
│   ├── test_empty_audio              → 空数组 → 拒绝
│   ├── test_too_short_audio          → 0.1s 以下 → 拒绝
│   ├── test_too_long_audio           → 30s 以上 → 拒绝
│   ├── test_silence_detection        → 全零 → 静音 True
│   └── test_speech_detection         → 正弦波 → 静音 False
│
└── TestTranscribe                    (推理测试, 不加载真实模型)
    ├── test_transcribe_raises_before_load → 未加载→RuntimeError
    ├── test_transcribe_empty_audio   → 空音频→空结果
    └── test_transcribe_silence       → 静音→空结果
```

### 9.2 测试策略

| 策略 | 说明 |
|------|------|
| **无真实模型依赖** | 所有测试不加载真实的 Faster-Whisper 模型 |
| **快速执行** | 纯 CPU + numpy，毫秒级完成 |
| **边界覆盖** | 空数组、极短、极长、静音、非 16kHz、非 float32 |
| **接口契约** | dataclass 构造、属性访问、初始化参数校验 |

---

## 十、Pipeline 集成方案

### 10.1 挂载代码

当 Phase 1 和 Phase 2 联调时，在 `main.py` 中添加：

```python
# Pipeline + STT + AIRI 集成
stt = FasterWhisperSTT(
    model_size="small",
    device="cpu",
    compute_type="int8",
    language="zh",
)

airi = AIRIClient(
    host=config.airi.host,
    port=config.airi.port,
    token=config.airi.token,
)

async def on_speech(event: SpeechEvent) -> None:
    """语音事件回调 → STT → AIRI"""
    if event.type == SpeechEventType.SPEECH_END:
        # STT 推理
        result = await stt.transcribe(event.audio)

        # 低置信度过滤
        if result.confidence < 0.3:
            logger.warning("Low confidence ({:.2f}), dropping", result.confidence)
            return

        # 发送到 AIRI
        await airi.send_input_text_voice(
            text=result.text,
            language=result.language,
        )

# 挂载到 Pipeline
pipeline = AudioPipeline(config)
pipeline.on_speech_event(on_speech)
```

### 10.2 数据流完整路径

```
麦克风 → [sounddevice] → AudioCapture → [48kHz float32]
    → Resampler → [16kHz float32, 512 samples/frame]
    → SileroVAD.process_frame()
        → SpeechEvent.SPEECH_START (可选)
        → SpeechEvent.SPEECH_END (含完整音频段)
            → on_speech() 回调
                → FasterWhisperSTT.transcribe(audio)
                    → [ThreadPoolExecutor] _infer_sync()
                        → STTResult(text, confidence, language)
                → 置信度 >= 0.3?
                    → 是 → AIRIClient.send_input_text_voice(text)
                    → 否 → 丢弃 + 日志
```

---

## 十一、性能指标

### 11.1 预期性能

| 模型 | 量化 | 1s 语音推理 | 5s 语音推理 | 峰值内存 |
|------|------|------------|------------|---------|
| tiny | int8 | ~30ms | ~150ms | ~400MB |
| **small** | **int8** | **~100ms** | **~500ms** | **~1.2GB** |
| base | int8 | ~60ms | ~300ms | ~600MB |
| medium | int8 | ~300ms | ~1.5s | ~2.5GB |

### 11.2 延迟预算

```
VAD 语音结束 (SPEECH_END)
        │
        │ 0ms
        ▼
快速通道检查 (静音? 格式?)
        │ ~0.1ms
        ▼
模型懒加载 (仅首次)
        │ ~2000ms (首次) / 0ms (后续)
        ▼
run_in_executor 调度
        │ ~0.5ms
        ▼
CTranslate2 推理 (small + int8 + CPU)
        │ ~100ms (1s音频) / ~500ms (5s音频)
        ▼
STTResult 构造 + 返回
        │ ~0.1ms
        ▼
回调消费 (AIRI WebSocket 发送)
        │ ~1ms (本地) / ~50ms (网络)
        ▼
端到端总延迟: ~150ms~600ms (不含首次模型加载)
```

### 11.3 目标

| 指标 | 目标 | 说明 |
|------|------|------|
| 端到端延迟 (语音结束→文字) | <1s | 包含 VAD 切段 + STT 推理 |
| 推理 RTF | <0.2 | 5s 语音推理 <1s |
| 中文 CER | <10% | small 模型在测试集上的字错率 |
| 内存占用 | <1.5GB | 含模型常驻内存 |
| CPU 占用 | <30% | 推理时单核 |

---

## 十二、设计权衡记录

### 12.1 技术选型

| 决策 | 选择 | 备选 | 选择理由 |
|------|------|------|---------|
| STT 引擎 | Faster-Whisper | SenseVoice / FunASR | 生态成熟、pip 即装、MIT 许可 |
| 模型大小 | small | tiny / base / medium | tiny CER 太高(15%), medium 太大(2.5GB) |
| 量化 | int8 | float32 / float16 | CPU 可用, 显存减半, 速度翻倍 |
| 默认设备 | cpu | cuda | 无 GPU 也可用, int8 速度足够 |
| 默认语言 | zh | en / auto | 目标用户主中文, 固定语言提升准确率 |
| 线程模型 | ThreadPoolExecutor(1) | asyncio.to_thread | 需要显式控制生命周期 |

### 12.2 接口设计

| 决策 | 选择 | 理由 |
|------|------|------|
| 错误处理 | 吞异常 + 返回空结果 | 管道必须弹性, 不因 STT 错误阻断 VAD 循环 |
| 模型加载 | 懒加载 (Lazy) | 构造零开销, 单元测试不需 mock 模型 |
| 输入校验 | 严格校验 + 快速失败 | 防止无效音频浪费 CPU 推理 |
| 输出格式 | 自定义 dataclass | 不暴露 Faster-Whisper 内部类型, 便于测试和 mock |
| 置信度计算 | avg_logprob → 0~1 | 统一度量, 方便下游做阈值判断 |

### 12.3 已知风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 模型首次下载延迟 | 🟡 中 | 首次 transcribe 卡 2s | 预下载到 `models/` 目录 |
| CPU 推理长句延迟 | 🟡 中 | >10s 句子推理 >2s | VAD 配置短句切分 (<5s) |
| 中文专有名词识别差 | 🟢 低 | 品牌/人名出错 | 热词功能 + 动态添加 |
| CUDA 不可用 | 🟡 中 | GPU 加速不可得 | int8 CPU 默认, RTF~0.1 已足够 |

### 12.4 未来优化方向

| 优化项 | 说明 | 预计 Phase |
|--------|------|-----------|
| 流式推理 | 边说话边出文字, 不等 VAD 结束 | Phase 2.2 |
| Whisper 内置 VAD 调参 | 适应不同噪声环境 | Phase 2.1 |
| 多模型热切换 | 中文/英文动态切换 | Phase 2.1 |
| batched inference | 多个短句批量推理 | Phase 2.1 |
| 标点恢复 ML 模型 | 替代简单规则 | Phase 2.2 |

---

> **文档版本**: 1.0
> **对应提交**: `93e9577` — Phase 2: STT 模块骨架 + 设计文档
> **下一阶段**: 测试验证 → Phase 2 Step 2 模型下载与缓存策略
