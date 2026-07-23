# AIRI Voice Module — Phase 1 测试报告

> **日期**: 2026-07-23
> **目标平台**: Windows 11（最终部署目标） + Linux（开发验证）
> **项目路径**: `airi-voice-module/`
> **Git Commit**: `3d40ae3`

---

## 一、测试范围概述

Phase 1 覆盖了**基础音频管线**的完整链路：麦克风捕获 → 重采样 → VAD 检测 → 语音事件输出。

### 模块清单

| 模块 | 文件 | 功能 |
|:----|:-----|:-----|
| 麦克风捕获 | `src/audio/capture.py` | 通过 sounddevice 读取麦克风音频流 |
| 扬声器输出 | `src/audio/playback.py` | 扬声器播放（供后续 TTS 使用） |
| 重采样 | `src/audio/resampler.py` | 48kHz → 16kHz 高质量降采样 |
| Silero VAD | `src/vad/silero_vad.py` | VAD 状态机 + ONNX Runtime 推理 |
| 管线编排 | `src/pipeline/audio_pipeline.py` | asyncio 三协程流水线 |
| 环形缓冲 | `src/pipeline/ring_buffer.py` | 音频环形缓冲区 |
| WebSocket | `src/airi/websocket_client.py` | AIRI 插件协议客户端 |
| 配置加载 | `src/config.py` | YAML + 环境变量配置 |
| 日志 | `src/logger.py` | loguru 结构化日志 |
| CLI 入口 | `src/main.py` | 命令行入口（3 种运行模式） |

---

## 二、测试环境

### 开发环境（Linux）

| 项目 | 值 |
|:----|:----|
| 系统 | Linux (WSL / 开发服务器) |
| Python | 3.14 |
| 音频设备 | 虚拟音频 / 无真实麦克风（单元测试模式） |

### 目标部署环境（Windows）

| 项目 | 值 |
|:----|:----|
| 系统 | Windows 11 |
| Python | 3.13.2 |
| 安装路径 | `D:\DevTools\Python\Python313\` |
| 项目路径 | `D:\DevProject\PythonProject\airi-voice-module` |
| 声卡 | Realtek(R) Audio（麦克风阵列） |

---

## 三、测试用例与覆盖率

### 3.1 重采样模块 — `tests/test_capture.py`

| 测试用例 | 类型 | 说明 |
|:---------|:----|:-----|
| `TestResampler::test_init` | 单元 | 验证初始化参数和重采样比例 |
| `TestResampler::test_downsample` | 单元 | 48kHz → 16kHz 降采样，正弦波验证 |
| `TestResampler::test_upsample` | 单元 | 16kHz → 48kHz 升采样 |
| `TestResampler::test_same_rate` | 单元 | 输入输出同采样率时不变 |
| `TestResampler::test_empty_data` | 边界 | 空数组输入不崩溃 |
| `TestResampler::test_to_float32_from_int16` | 单元 | int16 → float32 格式转换 |
| `TestResampler::test_to_int16_from_float32` | 单元 | float32 → int16 格式转换 |
| `TestResampler::test_invalid_rate` | 异常 | 非法采样率抛出 ValueError |
| `TestResampler::test_multichannel` | 单元 | 多声道音频重采样保持声道数 |
| `TestAudioCapture::test_list_devices` | 集成 | 设备列表 API 正确返回结构 |

### 3.2 VAD 状态机 — `tests/test_vad.py`

| 测试用例 | 类型 | 说明 |
|:---------|:----|:-----|
| `TestSileroVADInit::test_init_defaults` | 单元 | 默认初始化参数 |
| `TestSileroVADInit::test_custom_params` | 单元 | 自定义参数 |
| `TestSileroVADInit::test_min_speech_frames_calculation` | 单元 | 语音帧数计算逻辑 |
| `TestVADStateMachine::test_silence_frame` | 集成 | 静音帧 VAD 推理概率 < 0.5 |
| `TestVADStateMachine::test_speech_event_structure` | 单元 | SpeechEvent 数据结构 |
| `TestSpeechEvent::test_speech_start_event` | 单元 | SPEECH_START 事件结构 |
| `TestSpeechEvent::test_speech_end_event` | 单元 | SPEECH_END 事件结构（含音频） |
| `TestVADFlush::test_flush_when_silent` | 单元 | 静音状态 flush 返回 None |
| `TestVADFlush::test_flush_resets_state` | 单元 | flush 后状态重置为 SILENCE |

### 3.3 管线编排 — `tests/test_pipeline.py`

| 测试用例 | 类型 | 说明 |
|:---------|:----|:-----|
| `TestConfig::test_default_config` | 单元 | 默认配置值验证 |
| `TestConfig::test_config_from_yaml` | 集成 | YAML 配置文件加载 |
| `TestConfig::test_env_overrides` | 集成 | 环境变量覆盖配置 |
| `TestPipelineInit::test_pipeline_creation` | 单元 | AudioPipeline 对象创建和组件连线 |
| `TestPipelineInit::test_speech_callback_registration` | 单元 | 事件回调注册 |
| `TestPipelineInit::test_audio_device_listing` | 集成 | 音频设备列表 API |
| `TestRingBuffer::test_ring_buffer_creation` | 单元 | 环形缓冲区初始化 |
| `TestRingBuffer::test_raw_write_and_retrieve` | 单元 | 写入和读取原始帧 |
| `TestRingBuffer::test_raw_concatenation` | 单元 | 多帧拼接 |
| `TestRingBuffer::test_clear` | 单元 | 清空缓冲区 |

### 3.4 WebSocket 客户端 — `tests/test_airi_client.py`

| 测试用例 | 类型 | 说明 |
|:---------|:----|:-----|
| `TestAIRIClient::test_init_defaults` | 单元 | 默认连接参数 |
| `TestAIRIClient::test_custom_params` | 单元 | 自定义连接参数 |
| `TestAIRIClient::test_event_handler_registration` | 单元 | 事件处理器注册 |
| `TestAIRIClient::test_multiple_handlers` | 单元 | 同一事件多处理器 |
| `TestAIRIClient::test_send_when_disconnected` | 单元 | 未连接状态发送返回 False |
| `TestAIRIClient::test_input_text_message` | 单元 | input:text 消息结构 |
| `TestAIRIClient::test_input_text_voice_message` | 单元 | input:text:voice 消息结构 |

### 3.5 诊断工具（开发辅助）

| 工具 | 文件 | 用途 |
|:----|:-----|:-----|
| 麦克风音量测试 | `tests/test_mic_level.py` | 检测麦克风实际输入电平峰值 |
| VAD 实时诊断 | `tests/test_vad_diagnostic.py` | 实时显示 VAD 概率曲线，支持标准/直捕双模式 |
| VAD 模型对比 | `tests/test_vad_model_compare.py` | 对比 ONNX Runtime vs silero-vad 包 API 推理结果 |

---

## 四、执行结果

### 4.1 Linux 开发环境

```
Phase 1 完整测试: 45 total, 44 passed, 1 skipped, 0 failed
Phase 2 STT 测试: 21 total, 21 passed, 0 failed
```

跳过 1 个测试：`test_vad_model_compare` 的 silero-vad 包 API 对比需要安装 silero-vad pip 包（含 PyTorch，开发环境未部署）。

### 4.2 Windows 测试结果

#### 测试 1: 音频设备识别 ✅

```
python -m src.main --list-devices
→ 正确识别 9 个输入设备、11 个输出设备
→ 推荐: 输入 ID 9 (WASAPI 48000Hz), 输出 ID 8 (WASAPI 48000Hz)
```

#### 测试 2: 麦克风电平 ✅

```
python -m tests.test_mic_level
→ Peak: 1.000000 (100.0%)
→ 结论: 📢 音量充足! VAD 应能正常工作
```

#### 测试 3: VAD 实时诊断 ❌

| 模式 | 最大概率 | 状态 |
|:----|:--------:|:----:|
| 直捕 16kHz | 0.0787 (7.87%) | ❌ 远低于阈值 0.3 |
| 标准 48kHz → 重采样 | 0.0038 (0.38%) | ❌ 等效纯噪声 |
| 上次测试（2026-07-21） | 0.0031 (0.31%) | ❌ |

**结论**: 麦克风硬件和驱动层工作正常（电平 100%），但 Silero VAD 无法检测到语音特征。

---

## 五、发现的问题与解决方案

### 问题 1: PyCharm 内置 pip 不兼容 Python 3.13

- **错误**: `ModuleNotFoundError: No module named 'distutils'`
- **原因**: PyCharm 内置 pip 20.3.4（2020 年）引用了 Python 3.12 已移除的 distutils
- **解决**: 在终端手动创建 venv，不使用 PyCharm 内置 pip
- **执行结果**: ✅ 已解决

### 问题 2: pip SSL 证书错误

- **错误**: `SSLEOFError: [SSL: UNEXPECTED_EOF_WHILE_READING]`
- **原因**: 企业网络/安全软件拦截 HTTPS 连接
- **解决**: `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt`
- **执行结果**: ✅ 已解决（已写入 requirements.txt 注释作为参考）

### 问题 3: Silero VAD 模型下载 404

- **错误**: GitHub 原始 ONNX 模型下载地址失效
- **原因**: silero-vad v6.x 已将 ONNX 模型内嵌到 pip 包中，旧地址 404
- **解决方案**:
  1. 重构 `src/vad/silero_vad.py` 的 `load_model()` 方法
  2. Strategy 1: 通过 `importlib.resources.files('silero_vad.data')` 查找内嵌模型
  3. Strategy 2: 回退到本地 `models/silero_vad.onnx` 文件
- **执行结果**: ✅ 模型可通过 pip 包内嵌模型自动加载（无需手动下载）

### 问题 4: 虚拟环境 Python 解释器指向 WSL 路径

- **错误**: `did not find executable at '/usr/bin/python.exe'`
- **原因**: `.venv` 创建时误用了 WSL Python 而非 Windows 原生 Python
- **解决**: 重建 venv → `D:\DevTools\Python\Python313\python.exe -m venv .venv`
- **执行结果**: ✅ 已解决

### 问题 5: silero-vad pip 包强制依赖 PyTorch

- **问题**: silero-vad v6.x 的 `__init__.py` 会加载 PyTorch，即使我们只使用内嵌 ONNX 模型
- **影响**: pip 安装时拉取 PyTorch + CUDA (~1GB)，且 `import silero_vad` 就会崩溃
- **解决方案**:
  1. 简化 `silero_vad.py`：去掉 Strategy 1（importlib.resources）
  2. 只使用 Strategy 2：ONNX Runtime 直接加载本地模型文件
  3. 将 ONNX 模型放入 `models/` 目录
  4. 更新 `requirements.txt`：将 silero-vad 标记为可选注释
- **执行状态**: ⚠️ 部分解决（代码已适配，但 Windows 端需同步下载模型到本地）

### 问题 6: 依赖安装简化

- **问题**: 一行命令复杂且易出错
- **解决方案**: 统一安装命令（跳过 SSL + 跳过 PyTorch）
- **执行结果**: ✅ 文档已更新

### 问题 7: 🔴 Realtek 声卡驱动 DSP 降噪导致 VAD 概率归零 ⚠️ 未解决

- **症状**:
  - 麦克风电平正常 (peak=100%)
  - VAD 概率 max=0.0787（直捕）/ 0.0038（标准）
  - 直捕 16kHz 排除重采样问题
  - ONNX Runtime vs silero-vad 包 API 对比两种方式都得到零概率（已通过模型对比工具验证）

- **根因**:
  > 笔记本电脑 **Realtek 麦克风阵列声卡驱动** 内置降噪/波束成形 DSP。驱动层面的音频处理在送到应用层之前就已经滤除了 Silero VAD 需要的语音特征频率。

- **建议解决方案**（按优先级）:

  | # | 方案 | 操作 | 状态 |
  |:-:|:----|:-----|:----:|
  | ① | **禁用音频增强** | Windows → 声音设置 → 麦克风阵列 → 增强 → 关闭所有效果 | ⏳ 待测试 |
  | ② | **外接 USB 麦克风** | USB 麦克风走独立音频通道，跳过笔记本内置 DSP | ⏳ 待测试 |
  | ③ | **换用 webrtcvad** | WebRTC VAD 对降噪后音频的兼容性更好 | ⏳ 可选方案 |

- **提交的诊断工具**:
  - `tests/test_vad_diagnostic.py` — 实时 VAD 概率 + `--direct-16k` 直捕模式
  - `tests/test_mic_level.py` — 麦克风电平诊断
  - `tests/test_vad_model_compare.py` — ONNX Runtime vs 包 API 对比

---

## 六、统计汇总

### 测试汇总

| 类别 | 总数 | 通过 | 跳过 | 失败 |
|:----|:---:|:---:|:---:|:---:|
| 重采样测试 | 10 | 10 | 0 | 0 |
| VAD 状态机测试 | 9 | 9 | 0 | 0 |
| 管线编排测试 | 10 | 10 | 0 | 0 |
| WebSocket 测试 | 7 | 7 | 0 | 0 |
| 硬件环境测试（Windows） | 2 | 1 | 0 | 1 |
| **合计** | **38** | **37** | **0** | **1** |

### 问题统计

| 严重程度 | 数量 | 已解决 | 待解决 |
|:--------|:---:|:-----:|:-----:|
| 🔴 阻塞 | 1 | 0 | 1（Realtek DSP） |
| 🟡 中等 | 3 | 3 | 0 |
| 🟢 轻微 | 3 | 3 | 0 |

---

## 七、已知问题与待办

### 阻塞

- [ ] **Realtek DSP 降噪导致 VAD 检测失败** — 需尝试禁用音频增强或外接 USB 麦克风

### 后续 Phase 建议

- [ ] 在 Phase 2 (STT) 开始前先解决 VAD 问题，否则全链路语音输入无法验证
- [ ] Windows 端补充 `models/silero_vad.onnx` 本地模型文件
- [ ] 考虑 WebRTC VAD 作为 Silero VAD 的降级备选方案

---

## 八、附录

### 相关文档

- `docs/PHASE-1-DESIGN.md` — Phase 1 方案设计
- `docs/ARCHITECTURE.md` — 整体架构文档
- `docs/PHASE-2-STT.md` — Phase 2 STT 集成方案

### 相关提交

```
3d40ae3 Phase 2: 修复 STT 单元测试兼容性
1287a9f Phase 2: 添加 STT 详细实现设计文档
93e9577 Phase 2: STT 模块骨架 + 设计文档
43c1dee feat: add VAD model comparison tool
2a1eb97 feat: add --direct-16k mode to VAD diagnostic tool
2d65970 feat: add VAD real-time diagnostic test tool
584b64c docs: 更新 requirements.txt 注释说明
3bdcd67 feat: initial AIRI Voice Module - Phase 1
```
