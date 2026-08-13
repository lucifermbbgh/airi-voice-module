# AIRI 语音对话模块 — Phase 3 TTS 详细实现设计

> **日期**: 2026-08-13
> **版本**: 1.0 (实现完成，Windows 端全链路验证通过)
> **对应代码**: `src/tts/cosyvoice_tts.py`、`src/tts/tts_manager.py`、`src/audio/playback.py`
> **单元测试**: `tests/test_tts.py` (39 项) + `tests/test_tts_integration.py` (28 项)
> **提交**: `2e2e6ae`
> **总体设计**: 见 `PHASE-3-TTS.md`

---

## 目录

1. [总体架构](#一总体架构)
2. [核心数据模型](#二核心数据模型)
3. [CosyVoice 引擎设计（零样本合成）](#三cosyvoice-引擎设计零样本合成)
4. [synthesize() 执行流水线](#四synthesize-执行流水线)
5. [采样率处理与播放引擎](#五采样率处理与播放引擎)
6. [异步线程模型](#六异步线程模型)
7. [模型生命周期管理](#七模型生命周期管理)
8. [流式处理](#八流式处理)
9. [测试设计](#九测试设计)
10. [Pipeline 集成方案](#十pipeline-集成方案)
11. [性能指标](#十一性能指标)
12. [设计权衡记录](#十二设计权衡记录)

---

## 一、总体架构

### 1.1 Phase 3 在全链路中的位置

```
Mic → VAD → STT → AIRI WebSocket → LLM → TTS → Speaker
 P1     P1    P2        P4          P4     P3     P3
```

Phase 3 负责链路的最后一段：**把 AIRI（LLM）返回的文字回复合成为语音，并通过扬声器播放**。

### 1.2 模块分层

```
src/tts/
├── tts_engine.py       # TTS 接口抽象（TTSBase + TTSResult + TTSConfig）
├── cosyvoice_tts.py    # CosyVoice 2 引擎实现（零样本合成）
└── tts_manager.py      # 编排：合成 → 缓存 → 播放 → 回调

src/audio/
└── playback.py         # 扬声器播放引擎（AudioPlayback，采样率重采样）
```

### 1.3 核心设计原则

1. **懒加载**：模型在首次 `synthesize()` 时才加载，避免启动慢、内存占用高
2. **零样本**：CosyVoice2 是零样本模型，音色由参考音频（prompt_wav）决定，不内置固定音色
3. **采样率解耦**：TTS 输出 24000Hz，播放设备 44100Hz，播放层负责重采样
4. **线程安全**：播放回调在 PortAudio 线程，asyncio 事件循环在另一线程，跨线程必须用 `call_soon_threadsafe`

---

## 二、核心数据模型

### 2.1 `TTSResult` — 统一结果容器

```python
@dataclass
class TTSResult:
    audio: np.ndarray      # 合成音频（float32，1-D，范围约 [-1, 1]）
    sample_rate: int       # 音频采样率（CosyVoice2 = 24000）
    duration: float        # 时长（秒）
    text: str              # 合成文本
    synthesis_time: float  # 合成耗时（秒）
```

### 2.2 `TTSConfig` — 配置模型

```python
@dataclass
class TTSConfig:
    engine: str = "cosyvoice"      # 引擎名
    model_size: str = "base"       # 模型规格
    device: str = "cpu"            # 推理设备（cuda/cpu）
    model_dir: str | None = None   # 模型目录（None 走环境变量/默认相对路径）
    sample_rate: int = 24000       # 输出采样率
    voice_id: str = "default"      # 音色（零样本模式下保留字段，暂未用于映射）
    speed: float = 1.0             # 语速
```

### 2.3 `PlaybackSegment` — 播放片段

```python
@dataclass
class PlaybackSegment:
    audio: np.ndarray    # 音频数据（已重采样到播放流采样率）
    sample_rate: int     # 播放流采样率
    text: str = ""       # 关联文本
    sequence: int = 0    # 序号
```

---

## 三、CosyVoice 引擎设计（零样本合成）

### 3.1 关键事实：CosyVoice2 是零样本模型

CosyVoice2 **不能只传文本合成**。它需要：

| 输入 | 说明 |
|:-----|:-----|
| `tts_text` | 要合成的目标文本 |
| `prompt_text` | 参考音频对应的文字（必须与 prompt_wav 内容一致） |
| `prompt_wav` | 参考音频路径（决定音色） |

官方 API 是生成器：

```python
for output in model.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False, speed=1.0):
    speech = output["tts_speech"]   # torch.Tensor, shape [1, T], 24000Hz
```

### 3.2 `_synthesize_sync()` — 同步合成核心

```python
def _synthesize_sync(self, text, voice_id="default", speed=1.0) -> np.ndarray:
    prompt_wav = self._resolve_prompt_wav()
    prompt_text = "希望你以后能够做的比我还好呦。"

    chunks = []
    for output in self._model.inference_zero_shot(text, prompt_text=prompt_text,
                                                  prompt_wav=prompt_wav, stream=False, speed=speed):
        speech = output["tts_speech"]
        if hasattr(speech, "cpu"):           # torch.Tensor（可能在 GPU）
            speech = speech.cpu().numpy()
        chunks.append(np.asarray(speech, dtype=np.float32))

    audio = np.concatenate(chunks, axis=-1) if chunks else np.array([], dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    return self.normalize_volume(audio)
```

### 3.3 `_resolve_prompt_wav()` — 参考音频定位

按优先级解析参考音频路径：

| 优先级 | 来源 |
|:------:|:-----|
| 1 | `COSYVOICE_PROMPT_WAV` 环境变量 |
| 2 | 从 `model_dir` 推导 `<cosyvoice_root>/asset/zero_shot_prompt.wav` |
| 3 | `COSYVOICE_ROOT` 环境变量的 `asset/zero_shot_prompt.wav` |

默认使用 CosyVoice 仓库自带的 `asset/zero_shot_prompt.wav`（内容即 prompt_text "希望你以后能够做的比我还好呦。"）。

### 3.4 模型加载（懒加载）

```python
async def load_model(self):
    from cosyvoice.cli.cosyvoice import CosyVoice2  # 注意：是 CosyVoice2，不是基类 CosyVoice
    self._model = CosyVoice2(model_path)
```

**坑 1**：必须 import `CosyVoice2`（找 `cosyvoice2.yaml`），不是基类 `CosyVoice`（找 `cosyvoice.yaml`）。
**坑 2**：模型名是 `CosyVoice2-0.5B`（`CosyVoice2` 无中间横杠），不是 `CosyVoice-2-0.5B`。

---

## 四、synthesize() 执行流水线

```
async def synthesize(text, voice_id, speed)
  ├─ 1. 懒加载模型（首次）
  ├─ 2. 文本 strip + 空文本早退
  ├─ 3. 提交 _synthesize_sync 到线程池（CosyVoice 推理是阻塞的）
  ├─ 4. 构造 TTSResult（含 synthesis_time）
  └─ 5. 返回 TTSResult
```

### 4.1 为什么推理放线程池

CosyVoice2 的 `inference_zero_shot` 是**阻塞的同步生成器**，若直接在 asyncio 事件循环
调用会阻塞整个循环（包括 VAD 采集和播放）。因此用 `ThreadPoolExecutor` 隔离。

---

## 五、采样率处理与播放引擎

### 5.1 采样率不匹配问题（本会话核心 Bug）

| 组件 | 采样率 |
|:-----|:------:|
| CosyVoice2 输出（`cosyvoice2.yaml` 的 `sample_rate`） | 24000 Hz |
| Windows 默认输出设备（Realtek 扬声器） | 44100 Hz |

早期 `AudioPlayback` 用 `samplerate=24000` 打开 44100 设备，依赖 PortAudio 自动重采样，
但 Realtek 驱动 + Nahimic 音效导致重采样未生效，音频变调 1.84 倍（"花栗鼠音"）。

### 5.2 修复方案

1. **播放流用设备原生采样率**（动态检测 `sd.query_devices(kind="output")["default_samplerate"]`，44100）
2. **`play()` 内做线性插值重采样**：

```python
async def play(self, audio, sample_rate=None, text=""):
    src_rate = sample_rate or self.sample_rate
    if src_rate != self.sample_rate:
        audio = self._resample(audio, src_rate, self.sample_rate)  # 24000 → 44100
    ...

@staticmethod
def _resample(audio, src_rate, dst_rate):
    n_out = int(round(len(audio) * dst_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)
```

### 5.3 播放引擎线程安全（本会话核心 Bug）

`AudioPlayback._callback` 在 **PortAudio 回调线程**执行，而 `asyncio.Event.set()` 不是
线程安全的——从回调线程直接调用会导致 `wait_for_completion` 的 waiter 不被唤醒，程序死锁
（实测卡 12 分钟）。

**修复**：

```python
def __init__(self, ...):
    self._loop: asyncio.AbstractEventLoop | None = None   # 保存事件循环引用

async def start(self):
    self._loop = asyncio.get_running_loop()

def _advance_to_next(self):
    # 播放完最后一段时，从回调线程安全地唤醒等待者
    if self._loop is not None and not self._loop.is_closed():
        self._loop.call_soon_threadsafe(self._event.set)
    else:
        self._event.set()
```

同时 `wait_for_completion` 在 `clear()` 后重新检查状态，消除 clear/set 竞态。

---

## 六、异步线程模型

```
asyncio 事件循环线程           ThreadPoolExecutor 工作线程        PortAudio 回调线程
─────────────────────          ─────────────────────────          ────────────────────
synthesize()  ────────────►   _synthesize_sync()（阻塞推理）
  ▲                                    │
  └────────── TTSResult ───────────────┘

playback.play()  ──────────►  queue.append(segment)
                                   │
wait_for_completion() ◄───── call_soon_threadsafe(event.set) ◄── _advance_to_next()
```

三条线程通过 `asyncio.Queue` / `deque` / `loop.call_soon_threadsafe` 协同。

---

## 七、模型生命周期管理

| 状态 | 方法 | 说明 |
|:-----|:-----|:-----|
| 未加载 | `__init__` | 只存配置，不加载模型 |
| 加载中 | `load_model()` | 首次 synthesize 触发，约 15-30s（CUDA） |
| 就绪 | `is_loaded()` | `self._model is not None` |
| 卸载 | `cleanup()` | 释放模型 + 线程池 |

懒加载的权衡：首次合成延迟 15-30s（可接受，换来启动零等待和低内存占用）。

---

## 八、流式处理

### 8.1 `synthesize_stream()` — 分句流式合成

```python
async def synthesize_stream(self, text, voice_id="default", speed=1.0):
    sentences = self._split_sentences(text)   # 按标点分句
    for sentence in sentences:
        result = await self.synthesize(sentence, voice_id, speed)
        if len(result.audio) > 0:
            yield result.audio
```

分句边界：`_SENTENCE_END` 标点集合（。！？.!? 等）。用于降低首字延迟——第一句合成完即可播放，
不必等全文。

---

## 九、测试设计

### 9.1 测试架构

| 文件 | 测试数 | 说明 |
|:-----|:------:|:-----|
| `tests/test_tts.py` | 39 | 单元测试（mock 引擎，不加载真实模型） |
| `tests/test_tts_integration.py` | 28 | 集成测试（TTSManager + 播放编排） |
| `tests/test_tts_windows.py`（scripts） | — | Windows 端验证脚本（synthesize/play 模式） |
| `tests/diagnose_cosyvoice.py` | — | 诊断脚本（官方 API 二分定位 + 依赖版本检查） |

### 9.2 关键测试策略

- **单元测试不加载真实模型**：用 mock 引擎隔离 CosyVoice 依赖，保证 CI 可在无 GPU 环境跑
- **Windows 验证分离**：真实模型合成/播放验证用独立脚本（`test_tts_windows.py`），不在 pytest 套件内
- **诊断脚本**：`diagnose_cosyvoice.py` 绕过项目 TTS 模块，直接用官方 API 合成，用于二分定位"是代码问题还是环境问题"

---

## 十、Pipeline 集成方案

### 10.1 全链路数据流（`_run_full()`）

```
SPEECH_START → STT 文字 → AIRI WebSocket → output:gen-ai:chat:message
                                                    ↓
                                          TTSManager.say(text)
                                                    ↓
                                          synthesize → cache → playback.play
```

### 10.2 打断预备（Phase 5）

`SPEECH_START` 事件触发 `tts_mgr.stop()`（已预埋），打断当前播放，Phase 5 实现完整打断机制。

---

## 十一、性能指标

| 指标 | 目标 | 实测（Windows RTX 3070 Ti） |
|:-----|:----:|:---------------------------|
| 合成 RTF | <0.3 | ≈1.0（onnxruntime CPU 版，见下） |
| 20 字合成时长 | ~5s | 3.52s（transformers 4.51.3 修复后） |
| 模型加载 | <30s | 15-16s |
| 显存占用 | <4GB | ~1.5GB |

**RTF 未达标的根因**：onnxruntime 是 CPU 版（缺 `CUDAExecutionProvider`），speech_tokenizer
和 campplus 跑在 CPU。已尝试 `onnxruntime-gpu==1.28.0`，但要求 CUDA 13 + cuDNN 9（本机
CUDA 12.4）加载失败回退 CPU。**不影响正确性，仅影响速度**，留待后续装匹配 CUDA 12 的
onnxruntime-gpu 版本优化。

---

## 十二、设计权衡记录

### 12.1 依赖版本锁定（本会话核心教训）

| 依赖 | 版本 | 原因 |
|:-----|:-----|:-----|
| transformers | ==4.51.3（锁死） | 4.55.x 的 SDPA 与 CosyVoice `forward_one_step` 自定义逐步生成不兼容，导致 LLM 重复生成（"呜呜冬/哦哦哦"乱码） |
| x-transformers | ==2.11.24（锁死） | flow DiT 的 RotaryEmbedding 版本敏感 |
| torchvision | ==0.21.0+cu124 | 与 torch cu124 匹配（cu118 会在导入时触发 CUDA 版本检查报错） |

### 12.2 零样本 vs 固定音色

选择零样本（`inference_zero_shot`）而非固定音色（`inference_sft`），因为 CosyVoice2-0.5B
是零样本模型，音色由参考音频决定，灵活性高。代价是每次合成需提取参考音频的音色嵌入
（campplus + speech_tokenizer，耗时约 1-2s）。

### 12.3 线性插值 vs scipy 重采样

`_resample` 用 numpy 线性插值而非 scipy.signal.resample_poly，避免 playback.py 引入 scipy
硬依赖。线性插值对 24000→44100 的语音播放质量足够，产品化阶段可换高质量重采样。
