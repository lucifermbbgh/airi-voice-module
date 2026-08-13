# AIRI Voice Module — Step 5 设计文档: main.py TTS 集成

> **日期**: 2026-07-27
> **版本**: 1.0
> **关联设计**: [PHASE-3-TTS-DESIGN.md](../PHASE-3-TTS-DESIGN.md)
> **关联提交**: `ceef273`, `fd0b01b`

---

## 1. 设计目标

将 Phase 3 TTS 模块（CosyVoice 2 引擎 + TTSManager + AudioPlayback）接入 `src/main.py`，形成完整的语音交互闭环：

```
[麦克风] → [VAD] → [STT] → [AIRI API] → [TTS] → [扬声器]
   Phase 1    Phase 1   Phase 2    Phase 4    Phase 3   Phase 1
```

---

## 2. 架构设计

### 2.1 分层验证架构

为了支持独立排障和渐进式测试，设计了三层 CLI 模式：

```
CLI 入口 (python -m src.main)
    │
    ├── --test-vad        Layer 1: 仅验证 VAD (无 STT/TTS/AIRI)
    ├── --test-stt        Layer 2: VAD + STT (无 AIRI/TTS)
    ├── --test-tts        Layer 3: 纯 TTS (打字→语音, 无硬件依赖)
    ├── --test-tts-no-play Layer 3a: TTS 合成→WAV (无声卡)
    └── (无参数)            Layer 4: 全链路 VAD→STT→AIRI→TTS
```

### 2.2 AIRI → TTS 事件流

```
AIRI WebSocket
    │
    ├── output:gen-ai:chat:message  ← 监听此事件
    │   └── data.text / .message / .content
    │       └── → TTSManager.say(text)
    │           ├── 缓存命中 → 直接播放
    │           ├── 缓存未命中 → CosyVoice 2 合成 → 缓存 → 播放
    │           └── → AudioPlayback (sounddevice OutputStream)
    │               └── 扬声器 🔊
    │
    └── transport:connection:heartbeat  ← 忽略
```

`text` 字段支持三种来源以保证兼容性：

```python
text = data.get("text") or data.get("message") or data.get("content") or ""
```

### 2.3 TTS 容错设计

TTS 初始化包裹在 `try/except` 中，TTS 失败不会阻塞 STT→AIRI 链路：

```python
try:
    tts_engine = CosyVoiceTTS(...)
    tts_mgr = TTSManager(engine=tts_engine, playback=playback)
except ImportError:
    # TTS 禁用，其他功能正常运行
    logger.warning("TTS disabled: engine not available")
    tts_mgr = None
```

---

## 3. 新增 CLI 参数

| 参数 | 类型 | 默认 | 说明 |
|:----|:----|:----:|:------|
| `--test-tts` | `store_true` | `False` | 交互式 TTS 测试模式 |
| `--test-tts-no-play` | `str` / `const` | `output/tts_test.wav` | TTS 合成→保存 WAV |

### 3.1 `--test-tts` 交互模式

```
运行流程:
1. 初始化 AudioPlayback + TTSManager
2. 进入 input() 循环
3. 用户输入文本 → 判断长度 → say() 或 say_stream()
4. 输入 exit/quit → 退出
```

长文本（>20 字符）自动选择 `say_stream()` 流式合成，短文本用 `say()` 一次合成。

### 3.2 `--test-tts-no-play` 无播放模式

```
运行流程:
1. 初始化 CosyVoiceTTS (不需要 AudioPlayback)
2. 3 段预设中文/英文文本自动合成
3. scipy.wavfile.write() 保存为 WAV
4. 进入交互模式: 用户可输入自定义文本合成
5. 输出: output/tts_test_{1..3}.wav, output/tts_custom.wav
```

设计目标：在**无音频硬件**的环境下验证 TTS 引擎是否正常工作。

---

## 4. Windows 验证脚本设计

`scripts/test_tts_windows.py` 是为 Step 7 设计的独立验证脚本。

### 4.1 四模式架构

```
check      → 环境检查: Python/CUDA/PyTorch/基础依赖/TTS 依赖/声卡
synthesize → 合成验证: 加载模型 → 合成3段文本 → 保存WAV → 流式测试 → 音色切换 → 性能记录
play       → 播放验证: 合成 → 扬声器播放 → pause/resume 控制测试
all        → 渐进式执行: check → synthesize → play
```

### 4.2 防御性设计

| 设计模式 | 说明 |
|:---------|:------|
| 基础依赖前置检查 | 在 import 项目模块前检查 loguru/numpy/websockets/pyyaml/onnxruntime |
| 延迟导入 | `from src.tts import CosyVoiceTTS` 在函数体内部，外层 try/except |
| CUDA→CPU 降级 | CUDA 模型加载失败自动用 CPU 重试 |
| scipy 可选 | scipy 缺失时不保存 WAV，不影响合成测试 |
| 渐进式停止 | `--mode all` 时 synthesize 失败则跳过 play |

### 4.3 依赖检查树

```
check_environment()
    ├── Python 版本
    ├── PyTorch / CUDA (GPU 型号/数量)
    ├── 基础项目依赖
    │   ├── loguru         ← 必须 (src/logger)
    │   ├── numpy          ← 必须 (音频处理)
    │   ├── websockets     ← 必须 (AIRI 连接)
    │   ├── pyyaml         ← 必须 (配置加载)
    │   └── onnxruntime    ← 必须 (VAD/STT)
    ├── 音频设备
    │   └── sounddevice    ← play 模式必须
    └── TTS 依赖
        ├── cosyvoice      ← 必须 (主引擎)
        ├── scipy          ← 建议 (WAV 保存)
        └── edge_tts       ← 可选 (备用引擎)
```

---

## 5. CosyVoice 2 安装问题

### 问题

```bash
pip install cosyvoice
# → KeyError: '__version__'
```

PyPI 上的 `cosyvoice-0.0.8` 是社区上传的老版本，其 `setup.py` 尝试读取 `__version__` 但源码包中未定义。

### 解决方案

```bash
pip install git+https://github.com/FunAudioLLM/CosyVoice.git
```

或：

```bash
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -e .
```

---

## 6. 文件变更清单

| 文件 | 变更类型 | 变更行数 | 说明 |
|:----|:--------:|:--------:|:------|
| `src/main.py` | 修改 | +299/-14 | Step 5 核心集成 |
| `scripts/test_tts_windows.py` | 新增 | +302 | Windows 验证脚本 |
| `docs/PHASE-3-TTS-TEST-REPORT.md` | 修改 | +68/-14 | 测试报告更新 |
| `docs/STEP-5-DESIGN.md` | 新增 | - | 本文档 |

---

## 7. 测试状态

| 测试项 | 结果 | 备注 |
|:------|:----:|:------|
| TTS 单元测试 (39项) | ✅ 全部通过 | mock 引擎, 无硬件依赖 |
| TTS 集成测试 (28项) | ✅ 全部通过 | mock 引擎, 无硬件依赖 |
| main.py 编译 | ✅ 通过 | `py_compile` 检查 |
| `--test-tts` argparse | ✅ 通过 | 参数解析验证 |
| `--test-tts-no-play` argparse | ✅ 通过 | 参数解析验证 |
| Windows 环境检查 | ⏳ 待执行 | 需安装 CosyVoice 2 |
| Windows 全链路验证 | ⏳ 待执行 | Step 7 |
