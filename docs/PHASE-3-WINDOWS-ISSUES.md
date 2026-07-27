# AIRI Voice Module — Phase 3 问题记录: Windows 部署与验证

> **日期**: 2026-07-27
> **关联**: Windows 10/11, CosyVoice 2, pip 安装
> **项目路径**: `D:\DevProject\PythonProject\airi-voice-module`

---

## 问题 1: CosyVoice 2 PyPI 安装失败

### 现象

```powershell
(.venv) PS> pip install cosyvoice
```
```
× Getting requirements to build wheel did not run successfully.
│ exit code: 1
│ ...
│ KeyError: '__version__'
```

### 根因分析

- PyPI 上 `cosyvoice-0.0.8.tar.gz` (25KB) 是社区上传的陈旧版本
- `setup.py` 通过 `exec(open('cosyvoice/version.py').read())` 或类似方式读取版本号
- 但 `.tar.gz` 源码包中 `cosyvoice/__init__.py` 未定义 `__version__` 变量 → `KeyError`
- 官方 CosyVoice 2 团队（阿里达摩院 FunAudioLLM）**未在 PyPI 发布官方包**

### 复现条件

| 条件 | 值 |
|:----|:----|
| 操作系统 | Windows 11 |
| Python | 3.13.2 |
| pip | 24.3.1 |
| cosyvoice 版本 | 0.0.8 (PyPI) |

### 解决方案

```powershell
# 方案 A: 从 GitHub 源码安装（推荐）
pip install git+https://github.com/FunAudioLLM/CosyVoice.git

# 方案 B: 本地克隆后安装
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -e .
```

### 验证

安装成功后运行：
```powershell
python -c "from cosyvoice.cli.cosyvoice import CosyVoice; print('OK')"
```

---

## 问题 2: Windows 验证脚本因缺失依赖崩溃

### 现象

```powershell
(.venv) PS> python scripts\test_tts_windows.py --mode all
```
```
  ❌ sounddevice: Not installed
  ❌ CosyVoice 2: Not installed
  ...
Traceback ...
  File "src\tts\cosyvoice_tts.py", line 34, in <module>
    from src.logger import get_logger
  File "src\logger.py", line 12, in <module>
    from loguru import logger
ModuleNotFoundError: No module named 'loguru'
```

### 根因分析

1. **验证脚本的设计缺陷**: `test_synthesize()` 在模块顶层执行 `from src.tts import CosyVoiceTTS`，但 `cosyvoice_tts.py` 的 import chain 会触发 `from src.logger import get_logger` → `from loguru import logger`
2. **Check 模式检查不完整**: 只覆盖了 `cosyvoice`/`edge-tts`/`scipy`/`sounddevice`，但 `loguru`、`websockets`、`pyyaml`、`onnxruntime` 等基础依赖未检查

### 修复

在 `scripts/test_tts_windows.py v2` (commit `fd0b01b`) 中:

| 修复项 | 说明 |
|:-------|:------|
| 新增基础依赖区块 | 在 check 模式中检查 loguru/numpy/websockets/pyyaml/onnxruntime |
| 延迟导入 | 将 `from src.tts` / `from src.audio` 移入函数内部 try/except |
| 前置检查 | synthesize/play 模式执行前先确认核心依赖齐全 |
| 错误提示 | 缺失依赖时显示 `pip install` 命令而不是 traceback |

### 状态

✅ 已修复 (commit `fd0b01b`)

---

## 问题 3: Windows 基础依赖缺失

### 现象

`pip install loguru numpy websockets pyyaml onnxruntime scipy sounddevice` 一次性全部成功。但首次执行时这些包均未安装在 `.venv` 中。

### 实际已安装环境

| 包 | 版本 | 来源 |
|:---|:----:|:-----|
| numpy (已全局) | 2.3.2 | `--system-site-packages` 继承 |
| pyyaml (已全局) | 6.0.2 | `--system-site-packages` 继承 |
| PyTorch (已全局) | 2.6.0+cu124 | 手动安装 CUDA 版 |
| loguru | 0.7.3 | 新装 |
| websockets | 16.1.1 | 新装 |
| onnxruntime | 1.28.0 | 新装 |
| scipy | 1.18.0 | 新装 |
| sounddevice | 0.5.5 | 新装 |
| cosyvoice | ❌ 待装 | 需从 GitHub 源码安装 |

### 说明

项目的 `.venv` 使用 `--system-site-packages` 模式创建，因此 PyTorch CUDA 版等全局安装的包可以被 venv 继承。但 `pyproject.toml` 中声明的项目依赖需要确保已在任意一处安装。

---

## 问题 4: CUDA 模型加载失败兜底

### 现象（预期）

如果配置 `device="cuda"` 但模型加载出错（如显存不足、驱动版本不匹配），脚本应自动降级到 CPU。

### 修复

```python
try:
    await tts.load_model()
except Exception:
    if cfg.device == "cuda":
        print("CUDA model load failed, retrying on CPU...")
        cfg.device = "cpu"
        tts = CosyVoiceTTS(device="cpu")
        await tts.load_model()
    else:
        raise
```

### 状态

✅ 已修复 (commit `fd0b01b`)

---

## 待解决问题汇总

| # | 问题 | 优先级 | 状态 | 解决方案 |
|:-:|:-----|:------:|:----:|:---------|
| 1 | CosyVoice 2 PyPI 安装失败 | 🔴 高 | 🟡 待修复 | 改从 GitHub 源码安装 |
| 2 | 验证脚本缺失基础依赖检查 | 🟡 中 | ✅ 已修复 | 脚本 v2 已添加 |
| 3 | Windows 基础依赖预装 | 🟢 低 | ✅ 已安装 | `pip install` 完成 |
| 4 | CUDA→CPU fallback | 🟢 低 | ✅ 已实现 | 脚本 v2 已添加 |
| 5 | Windows 全链路验证 (Step 7) | 🔴 高 | ⏳ 待执行 | 安装 CosyVoice 2 后运行 |

---

## 验证回归清单

安装 CosyVoice 2 后执行的测试顺序：

```powershell
# 1. 环境检查
python scripts\test_tts_windows.py --mode check

# 2. 合成验证 (无播放)
python scripts\test_tts_windows.py --mode synthesize

# 3. 全链路验证 (含播放)
python scripts\test_tts_windows.py --mode all
```
