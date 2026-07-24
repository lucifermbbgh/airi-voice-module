# AIRI 语音对话模块 — Phase 3 设计：TTS (Text-to-Speech)

> **日期**: 2026-07-24
> **状态**: 设计阶段（待实现）
> **依赖**: Phase 1 (AudioPlayback) + Phase 2 (STT)
> **目标平台**: Windows 11 (AIRI 主机) / Linux (开发测试)

---

## 一、架构概览

### Phase 3 在全链路中的位置

```
[ 麦克风 ] ─→ [ VAD ] ─→ [ STT ] ─→ [ LLM ] ─→ [ TTS ] ─→ [ 扬声器 ]
   Phase 1      Phase 1     Phase 2      Phase 4      Phase 3      Phase 1
                                                       ↑
                                                  您在这里 👈
```

### 数据流

```
AIRI (LLM 回复)                     TTS 模块
      │                                │
      │ WebSocket 收到回复              │
      ▼                                ▼
┌──────────────────────────────────────────────────────────────┐
│                       TTS Engine                             │
│                                                              │
│  输入: text(str) + voice_id(str) + speed(float)              │
│                                                              │
│  ┌─────────────┐   ┌────────────────┐   ┌────────────────┐  │
│  │ Text Normal │ → │  Synthesizer   │ → │ Audio PostProc │  │
│  │ (SSML/格式) │   │  (CosyVoice 2) │   │ (音量/格式)    │  │
│  └─────────────┘   └────────────────┘   └────────────────┘  │
│                           │                                  │
│                           ▼ float32 audio                    │
│                     ┌────────────────┐                       │
│                     │  AudioPlayback │                       │
│                     │  (队列播放)    │                       │
│                     └────────────────┘                       │
│                           │                                  │
│                           ▼                                  │
│                      扬声器输出                                │
└──────────────────────────────────────────────────────────────┘
```

### 与已有模块的集成关系

```
已有代码                              Phase 3 新增
┌────────────────────────┐          ┌──────────────────────┐
│ AudioPlayback          │          │ src/tts/             │
│  ├─ play(audio, rate)  │◄─────────┼──┐                   │
│  ├─ stop_current()     │  tts→play│  ├─ __init__.py      │
│  ├─ pause/resume()     │          │  ├─ tts_engine.py    │
│  └─ wait_for_completion│          │  │  (接口抽象)        │
│                        │          │  ├─ cosyvoice_tts.py │
│ src/pipeline/          │          │  │  (CosyVoice 2)    │
│  AudioPipeline         │          │  └─ tts_manager.py   │
│  ├─ _playback_loop ────│──────────┼──┐ (语音管理)        │
│  └─ play_audio()       │          │                      │
│                        │          │ tests/               │
│ src/airi/              │          │  ├─ test_tts.py      │
│  AIRIClient            │          │  └─ test_tts_int.py  │
│  └─ on() → 注册回调 ──┼──────────┼──┐                   │
│                        │          │                      │
│ src/config.py          │          │ docs/                │
│  Config ───────────────┼──────────┼──┐ PHASE-3-TTS.md    │
│                        │          │                      │
└────────────────────────┘          └──────────────────────┘
```

**核心设计原则**:

| 原则 | 说明 |
|------|------|
| **引擎抽象** | 通过 `TTSBase` 接口支持多引擎切换（CosyVoice 2 / Edge-TTS / ChatTTS） |
| **流式输出** | 支持边合成边播放，减少首字延迟 |
| **非阻塞** | 合成在独立线程中运行，不阻塞 asyncio 事件循环 |
| **可中断** | 支持打断当前 TTS 播放（Phase 5 打断机制的基础） |
| **缓存优化** | 对重复文本做音频缓存，减少合成次数 |

---

## 二、技术选型

### TTS 引擎对比

| 方案 | 中文质量 | 首次加载 | 实时因子 | 离线 | 内存占用 | 许可证 |
|:----|:--------:|:--------:|:--------:|:----:|:--------:|:------:|
| **CosyVoice 2** 🏆 | 🥇 极优 | ~3s | 0.1-0.3x | ✅ 是 | ~2GB | Apache 2.0 |
| **ChatTTS** | 🥇 优秀 | ~5s | 0.2-0.5x | ✅ 是 | ~2.5GB | Apache 2.0 |
| **Edge-TTS** | ✅ 良好 | ~1s | 0.5-2x | ❌ 需联网 | ~100MB | 免费 API |
| **GPT-SoVITS** | 🥇 极优 | ~10s | 0.3-0.8x | ✅ 是 | ~3GB | MIT |

### 推荐方案：CosyVoice 2（首选）+ Edge-TTS（备用）

**选择 CosyVoice 2 的理由**:

| 维度 | 评价 |
|:----|:------|
| 🗣️ **中文语音质量** | 业界顶尖，自然度接近真人，零样本语音克隆 |
| ⚡ **推理速度** | 流式支持，首字延迟 <500ms，可实时合成 |
| 📦 **模型弹性** | 提供多个预训练模型（Base / Small 等），按需选择 |
| 🔧 **pip 安装** | `pip install cosyvoice` 即可使用 |
| 🚫 **完全离线** | 模型本地加载，无需互联网连接 |
| 📜 **许可证** | Apache 2.0，无商用限制 |
| 🆕 **活跃度** | 阿里达摩院持续更新，社区活跃 |

**备选 Edge-TTS 的理由**:
- 零安装（调用微软在线 API）
- 超小内存（仅 HTTP 请求缓存）
- 适用于快速验证场景
- 但依赖网络，有延迟波动风险

### 最终选型

```
Phase 3 默认:     CosyVoice 2 (流式模式, CPU int8)
轻量模式 (备选):    Edge-TTS (在线, 零模型加载)
语音克隆 (扩展):    CosyVoice 2 zero-shot (少量样本即可克隆)
```

### 依赖规格

```yaml
# requirements.txt 新增
cosyvoice>=1.0.0:
  上游依赖:
    - torch>=2.0               # PyTorch 推理框架
    - numpy>=1.26              # 与现有依赖共享
    - soundfile>=0.12          # 音频文件读写

# 模型下载 (首次自动下载 ~1.5GB)
# 缓存目录: ~/.cache/modelscope/hub/
# 或手动下载到 models/cosyvoice/ 目录
```

---

## 三、模块设计

### 3.1 TTS 接口抽象 (`src/tts/tts_engine.py`)

```python
from __future__ import annotations

import abc
from dataclasses import dataclass
import numpy as np


@dataclass
class TTSResult:
    """TTS 合成结果.

    Attributes:
        audio: 合成的音频数据 (float32 numpy array).
        sample_rate: 采样率 (Hz).
        duration: 音频时长 (秒).
        text: 合成的文本.
        synthesis_time: 合成耗时 (秒).
    """
    audio: np.ndarray
    sample_rate: int
    duration: float
    text: str
    synthesis_time: float


class TTSBase(abc.ABC):
    """TTS 引擎抽象基类.

    所有 TTS 实现必须继承此类，实现合成方法。
    """

    @abc.abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str = "default",
        speed: float = 1.0,
    ) -> TTSResult:
        """合成单段文本为语音.

        Args:
            text: 待合成的文本.
            voice_id: 音色 ID (引擎具体实现定义可用音色).
            speed: 语速 (0.5~2.0, 1.0=正常).

        Returns:
            TTSResult 包含合成的音频数据.
        """
        ...

    @abc.abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice_id: str = "default",
        speed: float = 1.0,
    ) -> "AsyncIterator[np.ndarray]":
        """流式合成文本，逐段输出音频.

        适用于长文本，边合成边播放，减少首字延迟.

        Args:
            text: 待合成的文本.
            voice_id: 音色 ID.
            speed: 语速.

        Yields:
            不定长的音频片段 (float32 np.ndarray), 每段 ~200ms.
        """
        ...

    @abc.abstractmethod
    async def load_model(self) -> None:
        """加载 TTS 模型 (懒加载)."""
        ...

    @abc.abstractmethod
    async def unload_model(self) -> None:
        """卸载 TTS 模型，释放内存."""
        ...

    @abc.abstractmethod
    async def cleanup(self) -> None:
        """释放所有资源."""
        ...

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """模型是否已加载."""
        ...

    @property
    @abc.abstractmethod
    def voices(self) -> list[dict]:
        """获取可用音色列表.

        Returns:
            [{"id": "...", "name": "...", "description": "..."}, ...]
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """引擎名称 (如 'cosyvoice', 'edge_tts')."""
        ...
```

### 3.2 CosyVoice 2 引擎 (`src/tts/cosyvoice_tts.py`)

```python
class CosyVoiceTTS(TTSBase):
    """CosyVoice 2 TTS 引擎实现.

    支持的音色:
        - "default": 默认中文女声
        - "中文男声": 男声
        - 自定义: 零样本语音克隆 (需要提供参考音频)
    """

    MODEL_SIZES = {
        "base": {"ram_mb": 1500, "rtf": 0.2},
        "small": {"ram_mb": 1000, "rtf": 0.15},
    }

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        model_dir: str | None = None,
        sample_rate: int = 24000,
    ):
        """构造函数行为:
        - 校验 model_size
        - 构造时不加载模型 (懒加载)
        """
        ...

    async def synthesize(self, text, voice_id="default", speed=1.0) -> TTSResult:
        """文本合成 — 完整音频.
        1. 懒加载模型
        2. 文本正则化 (数字/日期/符号)
        3. 调用 CosyVoice 2 inference
        4. 后处理 (音量归一化)
        5. 返回 TTSResult
        """
        ...

    async def synthesize_stream(self, text, voice_id="default", speed=1.0):
        """流式合成 — 逐段输出.
        1. 文本分句 (按标点断开)
        2. 逐句合成
        3. yield 每句音频片段
        """
        ...
```

### 3.3 TTS 管理器 (`src/tts/tts_manager.py`)

```python
class TTSManager:
    """TTS 语音管理器.

    职责:
    1. 管理 TTS 引擎生命周期
    2. 语音缓存 (避免重复合成)
    3. 播放队列管理 (与 AudioPlayback 对接)
    4. 根据事件自动触发 TTS

    用法:
        tts = TTSManager(engine, playback, config)
        await tts.say("你好，我是 AIRI")
        await tts.say_stream("这是一段长文本...")
        await tts.stop()  # 打断当前播放
    """

    def __init__(
        self,
        engine: TTSBase,
        playback: AudioPlayback,
        cache_size: int = 128,
    ):
        ...
```

### 3.4 全链路数据流

```
AIRI WebSocket 收到 output:gen-ai:chat:message
        │
        ▼
on_airi_message() 回调
        │
        ▼
提取 AI 回复文本
        │
        ▼
┌─────────────────────────────────────────────────┐
│ TTSManager.say(text, voice_id)                   │
│                                                  │
│ 1. 检查缓存 → 命中 → 直接播放                   │
│ 2. 文本正则化 (数字/日期 → 读音)                │
│ 3. TTS 引擎合成 → 获取 audio float32             │
│ 4. 音量归一化                                    │
│ 5. 写入缓存                                       │
│ 6. AudioPlayback.play(audio, sample_rate)         │
│ 7. 返回                                            │
└─────────────────────────────────────────────────┘
        │
        ▼
AudioPlayback._callback() → sounddevice OutputStream
        │
        ▼
        扬声器 🔊
```

---

## 四、项目结构变更

```
airi-voice-module/
├── src/
│   ├── tts/                          ← 新增目录
│   │   ├── __init__.py               ← 模块导出
│   │   ├── tts_engine.py             ← 接口抽象 (TTSBase, TTSResult)
│   │   ├── cosyvoice_tts.py          ← CosyVoice 2 实现
│   │   └── tts_manager.py            ← TTS 管理器 (缓存/队列/生命周期)
│   ├── audio/
│   │   └── playback.py               ← 已有，不变
│   ├── config.py                     ← 添加 TTSConfig
│   ├── main.py                       ← 添加 TTS 集成
│   └── ... (已有文件不变)
├── models/
│   ├── silero_vad.onnx               ← 已有
│   └── CosyVoice-2/                  ← 新增 (TTS 模型缓存)
├── tests/
│   ├── test_tts.py                   ← TTS 单元测试 (mock 引擎)
│   └── test_tts_integration.py       ← TTS 集成测试
├── docs/
│   └── PHASE-3-TTS.md                ← 本设计文档
└── config/
    └── default.yaml                  ← 添加 TTS 配置段
```

### 文件职责矩阵

| 文件 | 职责 | 行数估算 |
|:----|:-----|:--------:|
| `tts/__init__.py` | 导出公共 API | ~15 |
| `tts/tts_engine.py` | TTS 接口抽象 + 数据类 | ~100 |
| `tts/cosyvoice_tts.py` | CosyVoice 2 引擎实现 | ~300 |
| `tts/tts_manager.py` | TTS 管理器 (缓存/队列) | ~200 |
| `tests/test_tts.py` | 单元测试 (mock 引擎) | ~150 |
| `tests/test_tts_integration.py` | 集成测试 (WAV 对比) | ~100 |

---

## 五、配置方案

### `config/default.yaml` 新增 TTS 段

```yaml
tts:
  # 引擎配置
  engine: "cosyvoice"              # cosyvoice / edge_tts
  model_size: "base"               # base / small
  model_dir: "models/CosyVoice-2"  # 模型缓存目录

  # 语音配置
  voice_id: "default"              # 音色 ID
  speed: 1.0                       # 语速 (0.5 - 2.0)
  sample_rate: 24000               # 输出采样率 (Hz)

  # 合成配置
  device: "cpu"                    # cpu / cuda
  streaming: true                  # 流式合成 (边合成边播放)
  max_text_length: 500             # 单次合成最大文本长度

  # 缓存
  enable_cache: true               # 启用语音缓存
  cache_size: 128                  # 缓存条目数上限
```

### `src/config.py` 新增配置模型

```python
@dataclass
class TTSConfig:
    engine: str = "cosyvoice"
    model_size: str = "base"
    model_dir: str | None = None
    voice_id: str = "default"
    speed: float = 1.0
    sample_rate: int = 24000
    device: str = "cpu"
    streaming: bool = True
    max_text_length: int = 500
    enable_cache: bool = True
    cache_size: int = 128
```

---

## 六、实施步骤

| 步骤 | 模块 | 内容 | 预计工时 | 可独立验证？ |
|:----|:-----|:-----|:--------:|:-----------:|
| **Step 1** | `tts_engine.py` | 接口抽象 + TTSResult 数据类 | 1h | ✅ |
| **Step 2** | `cosyvoice_tts.py` | CosyVoice 2 引擎实现 (合成 + 流式) | 3h | ✅ (可合成测试) |
| **Step 3** | `tts_manager.py` | 缓存 + 队列 + 对接 AudioPlayback | 2h | ⚠️ 需 Playback |
| **Step 4** | `config.py` + `default.yaml` | TTS 配置段 | 0.5h | ✅ |
| **Step 5** | `main.py` 集成 | AIRI 回调 → TTS → 播放 | 1h | ⚠️ 需 AIRI 连接 |
| **Step 6** | 单元测试 | `test_tts.py` (mock 引擎) | 1h | ✅ |
| **Step 7** | Windows 验证 | 真实 TTS 合成 + 播放测试 | 1h | ⚠️ 需声卡 |
| | **合计** | | **~9.5h** | **~70% 可独立验证** |

---

## 七、性能指标目标

| 指标 | 目标 | 测量方式 |
|:----|:----|:---------|
| 首字延迟 (TTFB) | <500ms | 从 `synthesize()` 调用到播放第一帧 |
| 端到端延迟 (文本→语音) | <2s (50字以内) | 从收到文本到播放完成 |
| 实时因子 (RTF) | <0.3 | `synthesis_time / audio_duration` |
| 语音自然度 MOS | >4.0 | 主观听感评分 |
| 内存占用 (模型常驻) | <2GB | Python 进程内存 |
| 缓存命中率 | >30% (常见场景) | `cache_hits / total_synthesize` |

---

## 八、风险 & 缓解

| 风险 | 概率 | 影响 | 缓解方案 |
|:----|:----:|:----:|:---------|
| CosyVoice 2 模型体积大 (~1.5GB) | 🟡 中 | 首次下载慢 | 提供预下载脚本 + 断点续传 |
| CPU 合成延迟高 | 🟡 中 | 首字延迟 >1s | int8 量化 + 流式输出, RTF 目标 <0.3 |
| 语音克隆精度不足 | 🟢 低 | 特定音色不像 | 先用预设音色, 克隆作为扩展功能 |
| 长篇对话缓存占用过大 | 🟢 低 | 内存增长 | LRU 缓存 + 限制缓存大小 (128条) |
| CosyVoice 2 API 变动 | 🟢 低 | 接口不兼容 | 接口抽象层隔离变动 |

---

## 九、测试策略

### 单元测试 (无需模型)

```
test_tts.py
├── TestTTSResult          — 数据类构造/序列化
├── TestTTSEngineBase      — 抽象接口契约
├── TestTTSManager         — 缓存/队列/生命周期
│   ├── test_say_with_cache
│   ├── test_say_no_cache
│   ├── test_stop_interrupt
│   └── test_playback_integration
└── TestTTSConfig          — 配置加载/环境变量
```

### 集成测试 (需要真实模型)

```
test_tts_integration.py
├── test_cosyvoice_synthesize    — 真实合成 + WAV 文件验证
├── test_synthesize_stream       — 流式合成逐段输出
├── test_voice_switching         — 音色切换
└── test_speed_control           — 语速调节
```

---

## 十、Future Work（Phase 3 范围外）

| 功能 | 预计 Phase | 说明 |
|:----|:----------|:-----|
| 语音克隆 (zero-shot) | Phase 3.1 | 用 3-5 秒参考音频克隆用户声线 |
| 情感语音 (Emotional TTS) | Phase 3.2 | 根据 LLM 输出情感调整语气 |
| 多语言混读 | Phase 3.2 | 中英混说时自动切换发音人 |
| 说话人分离 (SD) | Phase 5+ | 多条 TTS 流区分不同角色 |
| 语音活动感知播放 | Phase 5 | VAD 检测到用户说话时自动暂停 TTS |

---

> **文档版本**: 1.0 (2026-07-24)
> **下一步**: 设计评审 → Step 1: 实现 TTS 接口抽象
