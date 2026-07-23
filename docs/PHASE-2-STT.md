# AIRI 语音对话模块 — Phase 2 设计：STT (Speech-to-Text)

> **日期**: 2026-07-23
> **状态**: 设计阶段（待实现）
> **依赖**: Phase 1 VAD 事件（`SpeechEvent.SPEECH_END`）
> **目标平台**: Windows 11 (AIRI 主机) / Linux (开发测试)

---

## 一、架构概览

### Phase 2 在全链路中的位置

```
[ 麦克风 ] ─→ [ VAD ] ─→ [ STT ] ─→ [ LLM ] ─→ [ TTS ] ─→ [ 扬声器 ]
   Phase 1      Phase 1     Phase 2      Phase 4      Phase 3      Phase 1
                              ↑
                        您在这里 👈
```

### 数据流

```
VAD SpeechEvent.SPEECH_END
    │
    │  audio: float32[样本数] @ 16kHz
    │  duration: float (秒)
    ▼
┌──────────────────────────────────────────────────────────┐
│                     STT Engine                           │
│                                                          │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────┐ │
│  │ SpeechBuffer │───→│ Faster-Whisper   │───→│ 后处理 │ │
│  │ (可选拼接)   │    │ int8 量化推理    │    │ 标点   │ │
│  └──────────────┘    └──────────────────┘    └───┬────┘ │
│                                                   │     │
└───────────────────────────────────────────────────┼─────┘
                                                     │ text: str
                                                     ▼
┌──────────────────────────────────────────────────────────┐
│                    AIRIClient                            │
│  send_input_text_voice(text, language)                   │
└──────────────────────────┬───────────────────────────────┘
                           │ WebSocket
                           ▼
                     ┌──────────┐
                     │   AIRI   │
                     └──────────┘
```

### 与 Phase 1 的集成关系

```
Phase 1 已有代码                    Phase 2 新增
┌────────────────────────┐        ┌──────────────────────┐
│ AudioPipeline          │        │ src/stt/             │
│  ├─ AudioCapture       │        │  ├─ __init__.py      │
│  ├─ Resampler          │        │  └─ faster_whisper_  │
│  ├─ SileroVAD          │        │     stt.py           │
│  └─ on_speech_event()──┼────────┼──→ 回调入口         │
│     callback           │        │                      │
└────────────────────────┘        └──────────────────────┘
         │                                    │
         │ AudioPipeline.on_speech_event()    │
         │ 绑定 handle_speech → stt.transcribe│
         │                                    │
         ▼                                    ▼
   ┌──────────────────────────────────────────────────┐
   │ main.py (入口编排)                                │
   │  pipeline.on_speech_event(handle_speech) ← 新增 │
   └──────────────────────────────────────────────────┘
```

**核心原则**：Phase 2 是"插在" Phase 1 已有回调机制上的新增模块，**不修改任何 Phase 1 已有代码**。

---

## 二、技术选型

### STT 引擎对比

| 方案 | 模型大小 | 首次加载 | 实时因子(RTF) | 中文准确率(CER) | 内存占用 | License |
|------|---------|---------|--------------|----------------|---------|---------|
| **Faster-Whisper small** 🏆 | ~460MB | ~2s (int8) | 0.1-0.3x | ~5% | ~1.2GB | MIT |
| **Faster-Whisper base** | ~140MB | ~1s | 0.2-0.5x | ~8% | ~600MB | MIT |
| **Faster-Whisper tiny** | ~75MB | ~0.5s | 0.3-0.8x | ~15% | ~400MB | MIT |
| **Whisper.cpp** (GGML) | 75-1500MB | ~0.3s | 0.1-0.4x | ~5% | ~300MB | MIT |
| **SenseVoice** (阿里) | ~120MB | ~1s | 0.05x | ~3% 🥇 | ~500MB | MIT |
| **FunASR** (阿里) | ~200MB | ~1s | 0.03x 🥇 | ~2% 🥇 | ~800MB | MIT |

### 推荐方案：Faster-Whisper (首选) + SenseVoice (备用)

**选择 Faster-Whisper 的理由：**

| 维度 | 评价 |
|------|------|
| 🏗️ **生态成熟度** | pip install 即用，社区活跃，文档完善 |
| 📦 **模型弹性** | tiny(75MB) → large-v3(3GB)，按需选择 |
| ⚡ **推理速度** | int8 量化后 CPU 可用，RTF < 0.1 |
| 🗣️ **中文支持** | large-v3 中文 CER ~5%，日常对话可用 |
| 🔧 **工程化** | CTranslate2 后端，batch 推理优化 |
| 📜 **许可证** | MIT，无商用限制 |

**备选 SenseVoice 的理由：**
- 中文识别准确率优于 Whisper (CER ~3% vs ~5%)
- 极小模型 (120MB) + 极低延迟
- 内置情感识别、语种识别、事件检测
- 但社区和文档不如 Whisper 成熟

### 最终选型

```
Phase 2 默认: Faster-Whisper small (int8, CPU)
备用方案:     SenseVoice (中文场景需要更高精度时)
排除:         Whisper.cpp (C++ 集成成本 > Python 方案的收益)
              FunASR (阿里内部依赖太重，文档以中文为主)
```

### 依赖规格

```yaml
# requirements.txt 新增
faster-whisper>=1.1.0:
  上游依赖:
    - ctranslate2>=4.0        # int8 量化推理引擎
    - huggingface_hub>=0.23   # 模型自动下载
    - tokenizers>=0.15        # 分词器
    - numpy>=1.26             # 与现有依赖共享

# 模型缓存 (自动下载到 ~/.cache/huggingface/hub/)
# 或手动下载到 models/ 目录离线加载
models:
  - faster-whisper-small: ~460MB
  - faster-whisper-base:  ~140MB   # 降级选项
```

---

## 三、模块设计

### 3.1 核心 STT 类 (`src/stt/faster_whisper_stt.py`)

```python
@dataclass
class STTResult:
    """STT 推理结果"""
    text: str                       # 识别文本
    confidence: float               # 置信度 (0.0~1.0)
    language: str                   # 检测到的语言 (zh/en/...)
    language_probability: float     # 语言检测置信度
    duration: float                 # 语音段时长 (秒)
    inference_time: float           # 推理耗时 (秒)
    segments: list[Segment] | None  # 详细时间戳分段 (可选)


class FasterWhisperSTT:
    """Faster-Whisper 流式 STT 引擎。

    将 VAD 输出的语音音频段识别为文字。
    使用 int8 量化 + CPU 推理作为默认配置，
    支持 CUDA GPU 加速作为高级选项。

    用法:
        stt = FasterWhisperSTT(model_size="small", device="cpu")
        result = await stt.transcribe(audio_ndarray)
        print(result.text)  # "今天天气怎么样"
    """

    # 预定义模型规格
    MODELS = {
        "tiny":   {"size": "tiny",   "ram_mb": 400,  "rtf": 0.3},
        "base":   {"size": "base",   "ram_mb": 600,  "rtf": 0.2},
        "small":  {"size": "small",  "ram_mb": 1200, "rtf": 0.1},
        "medium": {"size": "medium", "ram_mb": 2500, "rtf": 0.08},
        "large-v3": {"size": "large-v3", "ram_mb": 3200, "rtf": 0.05},
    }

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
        """初始化 STT 引擎。

        Args:
            model_size: 模型大小 (tiny/base/small/medium/large-v3)
            device: 计算设备 (cpu/cuda)
            compute_type: 量化类型 (int8/float16/float32)
            model_dir: 模型缓存目录 (None=默认缓存)
            language: 语言偏好 (None=自动检测)
            beam_size: 束搜索宽度 (越大越准越慢)
            vad_filter: 启用 VAD 预过滤 (去除静音段)
            hotwords: 热词列表 (提升特定词汇准确率)
        """

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> STTResult:
        """将语音音频识别为文字。

        Args:
            audio: float32 numpy 数组，VAD 输出的语音段。
            sample_rate: 音频采样率 (应为 16000)。

        Returns:
            STTResult 包含识别文字和元数据。

        性能预期 (small + int8 + CPU):
            1秒语音 → ~100ms 推理 (RTF=0.1)
            5秒语音 → ~500ms 推理

        线程模型:
            Faster-Whisper 内部使用 CTranslate2 + OpenMP
            多线程 CPU 推理。asyncio 层通过
            loop.run_in_executor() 封装，不阻塞主事件循环。
        """

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[STTResult]:
        """流式处理多个语音段。

        逐个处理 VAD 按顺序输出的语音段，
        每次输出一段的识别结果。

        Args:
            audio_chunks: 音频帧异步迭代器。

        Yields:
            每段的 STTResult。
        """

    def set_language(self, language: str | None) -> None:
        """设置或清除语言偏好。"""

    def add_hotwords(self, hotwords: list[str]) -> None:
        """追加热词。"""

    def unload_model(self) -> None:
        """卸载模型以释放内存。

        在长时间不使用或切换模型时调用。
        """

    @property
    def model_info(self) -> dict:
        """获取当前模型信息。"""

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载。"""
```

### 3.2 文本后处理器 (`src/stt/post_processor.py`，可选)

```python
class TextPostProcessor:
    """STT 文本后处理。

    功能:
    - 标点恢复（去除无标点输出）
    - 中英文混排空格规范化
    - 数字格式化（如果有）
    - 专有名词纠错（热词优先）
    """

    def process(self, text: str, hotwords: list[str] | None = None) -> str:
        ...
```

**设计决策**：后处理器设计为**轻量规则引擎**而非额外模型。标点恢复如果需要，可以用 `pypinyin` 或简单规则实现。

### 3.3 Pipeline 集成

```python
# 在 main.py 中（仅需 ~15 行新增代码）

# 初始化 STT 引擎
stt = FasterWhisperSTT(
    model_size="small",
    device="cpu",
    compute_type="int8",
    language="zh",
)

# 初始化 AIRI 客户端
airi = AIRIClient(
    host=config.airi.host,
    port=config.airi.port,
    token=config.airi.token,
)

# 注册语音事件回调
async def on_speech(event: SpeechEvent) -> None:
    """语音事件处理：VAD 输出 → STT → AIRI"""
    if event.type == SpeechEventType.SPEECH_END:
        logger.info(
            "Speech segment: {:.2f}s, max_prob={:.3f}",
            event.duration, event.max_prob,
        )

        # STT 推理
        result = await stt.transcribe(event.audio)

        # 低置信度过滤（防止噪声误触发）
        if result.confidence < 0.3:
            logger.warning(
                "Low confidence ({:.2f}), dropping: {}",
                result.confidence, result.text[:50],
            )
            return

        # 发送到 AIRI
        await airi.send_input_text_voice(
            text=result.text,
            language=result.language,
        )

        logger.info("STT → AIRI: {} (conf={:.2f})", result.text, result.confidence)

pipeline.on_speech_event(on_speech)
```

---

## 四、项目结构变更

```
airi-voice-module/
├── docs/
│   ├── PHASE-1-DESIGN.md
│   └── PHASE-2-STT.md                ← 本文件 (新增)
├── src/
│   ├── __init__.py
│   ├── main.py                       ← 回调集成 (新增 ~15 行)
│   ├── config.py                     ← STT 配置段 (新增)
│   ├── stt/                          ← 新增目录
│   │   ├── __init__.py
│   │   ├── faster_whisper_stt.py     ← STT 引擎核心
│   │   └── post_processor.py         ← 文本后处理 (可选)
│   ├── audio/                        ← 不变
│   ├── vad/                          ← 不变
│   ├── pipeline/                     ← 不变
│   └── airi/                         ← 不变
├── models/
│   ├── silero_vad.onnx               ← 已有
│   └── whisper-small/                ← 新增 (预下载模型)
├── tests/
│   ├── test_stt.py                   ← STT 单元测试 (新增)
│   ├── test_stt_integration.py       ← STT 集成测试 (新增)
│   ├── test_vad.py                   ← 不变
│   └── test_capture.py               ← 不变
└── requirements.txt                  ← faster-whisper 依赖 (新增)
```

### 文件职责矩阵

| 文件 | 职责 | 行数估算 |
|------|------|---------|
| `stt/__init__.py` | 导出公共 API | ~15 |
| `stt/faster_whisper_stt.py` | STT 引擎（模型加载/推理/线程管理） | ~200 |
| `stt/post_processor.py` | 文本后处理（可选） | ~80 |
| `tests/test_stt.py` | 单元测试（mock 模型加载） | ~120 |
| `tests/test_stt_integration.py` | 集成测试（真实 WAV + 可选真实模型） | ~80 |

---

## 五、配置方案

### `config/default.yaml` 新增 STT 段

```yaml
stt:
  # 引擎配置
  engine: "faster_whisper"        # faster_whisper / sensevoice
  model_size: "small"             # tiny / base / small / medium / large-v3
  model_dir: "models/"            # 模型缓存目录

  # 推理配置
  device: "cpu"                   # cpu / cuda
  compute_type: "int8"            # int8 / float16 / float32
  language: "zh"                  # zh / en / auto (自动检测)
  beam_size: 5                    # 束搜索宽度
  vad_filter: true                # Whisper 内置 VAD 过滤

  # 文本后处理
  enable_post_processing: true    # 启用标点恢复等后处理
  min_confidence: 0.3             # 最低置信度过滤

  # 热词（提升特定词汇识别率）
  hotwords: []
```

### `src/config.py` 新增配置模型

```python
@dataclass
class STTConfig:
    engine: str = "faster_whisper"
    model_size: str = "small"
    model_dir: str = "models/"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "zh"
    beam_size: int = 5
    vad_filter: bool = True
    enable_post_processing: bool = True
    min_confidence: float = 0.3
    hotwords: list[str] = field(default_factory=list)
```

---

## 六、实施步骤

| 步骤 | 模块 | 预计工时 | 依赖 | 可独立于 Phase 1？ |
|------|------|---------|------|:---:|
| **Step 1** | `stt/faster_whisper_stt.py` 核心实现 | 2h | 无 | ✅ 完全独立 |
| **Step 2** | 模型下载 + 离线缓存 `models/` | 0.5h | Step 1 | ✅ |
| **Step 3** | 文本后处理器 `post_processor.py` | 0.5h | Step 1 | ✅ |
| **Step 4** | `config.py` + `default.yaml` 配置扩展 | 0.5h | Step 1 | ✅ |
| **Step 5** | Pipeline 回调集成 (`main.py`) | 0.5h | Phase 1 VAD + Step 1 | ⚠️ 需 VAD 事件 |
| **Step 6** | 单元测试 (`test_stt.py`) | 1h | Step 1 | ✅ |
| **Step 7** | Linux 端到端集成测试 | 1h | Steps 1-6 | ✅ (Linux VAD 正常) |
| **Step 8** | 性能基准测试 + 调优 | 1h | Step 1 | ✅ |
| **Step 9** | Windows 集成验证 | 0.5h | Phase 1 Realtek 修复 | ❌ 被阻塞 |
| | **合计** | **~7.5h** | | **~80% 可独立推进** |

---

## 七、性能指标目标

| 指标 | 目标 | 测量方式 |
|------|------|---------|
| 端到端延迟（语音结束→文字输出） | <1s (含 VAD 切段) | 计时从 VAD SPEECH_END 到 STT 回调 |
| 推理延迟 (RTF) | <0.2 (5秒语音 <1秒) | `inference_time / audio_duration` |
| 中文识别准确率 (CER) | <10% | 测试集上字错率 |
| 内存占用 | <500MB (模型常驻) | Python 进程内存 |
| CPU 占用 | <30% (单核) | htop / 任务管理器 |

---

## 八、风险 & 缓解

| 风险 | 概率 | 影响 | 缓解方案 |
|------|------|------|---------|
| Realtek DSP 问题传导到 STT | 🔴 高 | 被 DSP 过滤的音频 STT 识别率极低 | Linux 开发测试；备选 USB 麦克风绕过板载声卡 |
| Faster-Whisper 模型下载失败 | 🟡 中 | 首次启动无法运行 | 预下载到 `models/`；提供离线安装包脚本 |
| CPU 推理延迟过高（长句 >10s） | 🟡 中 | 对话卡顿 | VAD 配置 `min_silence_duration=0.3` 切更短句 |
| CUDA 不可用导致性能不足 | 🟡 中 | GPU 加速不可用 | int8 + CPU 默认方案，RTF<0.2 已足够 |
| 多语言混合识别偏差 | 🟢 低 | 中英混说时识别错乱 | 默认 `language="zh"`；支持动态切换 |

---

## 九、代码规范

与 Phase 1 保持一致：

1. **类型注解**: 所有函数使用 Python type hints
2. **异步优先**: IO 操作用 `async/await`，CPU 密集用 `run_in_executor`
3. **错误处理**: `try/except` 包围外部调用，`loguru` 记录异常
4. **配置化**: 无硬编码参数，所有可变参数来自 YAML
5. **文档字符串**: NumPy 风格 docstring
6. **测试覆盖**: `pytest` + `pytest-asyncio`，mock 外部依赖

---

## 十、Future Work（Phase 2 范围外）

| 功能 | 预计 Phase | 说明 |
|------|-----------|------|
| 热词自定义 | Phase 2.1 | 运行时动态热词更新 |
| 说话人分离 (SD) | Phase 5+ | 区分多说话人 |
| 流式 STT（非完整段） | Phase 2.2 | VAD 结束前就输出中间文字 |
| 情感识别 | Phase 4+ | 从语音中提取情感特征 |
| 标点恢复 ML 模型 | Phase 2.2 | 当前用规则，后续可升级为模型 |

---

> **下一步**: 完成设计评审后，开始 Step 1 — 实现 `FasterWhisperSTT` 核心类。
