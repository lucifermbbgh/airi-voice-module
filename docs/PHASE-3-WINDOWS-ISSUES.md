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
pip install git+https://github.com/QwenAudio/CosyVoice.git

# 方案 B: 本地克隆后安装
git clone --recursive https://github.com/QwenAudio/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive
pip install -r requirements.txt
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

---

## 问题 5: grpcio / grpcio-tools 编译失败 — pkg_resources 缺失

### 现象

```powershell
pip install -r requirements.txt
# ...
# × Getting requirements to build wheel did not run successfully.
# ModuleNotFoundError: No module named 'pkg_resources'
```

### 根因分析

| 因素 | 说明 |
|:----|:------|
| 直接原因 | `grpcio==1.57.0` 和 `grpcio-tools==1.57.0` 的 `setup.py` 在构建时调用 `import pkg_resources` |
| 深层原因 | `setuptools>=70.0.0` 已移除 `pkg_resources` 模块 |
| Python 版本 | Python 3.13 + setuptools 83.0 → `pkg_resources` 不复存在 |
| 软件版本 | grpcio 1.57.0 (2023) 太老，不兼容 Python 3.13 |

### 修复

修改 `CosyVoice/requirements.txt`，放宽版本 pin 以使用预编译 wheel：

```diff
- grpcio==1.57.0
+ grpcio>=1.57.0

- grpcio-tools==1.57.0
+ grpcio-tools>=1.57.0
```

放宽后 pip 检测到现有的 `grpcio 1.83.0` / `grpcio-tools 1.83.0` 满足条件，不再尝试从源码编译。新版提供了 `cp313-win_amd64` 的预编译 wheel，无需 C 编译器。

### 验证

```powershell
# 先安装新版
pip install "grpcio>=1.62.0"
pip install "grpcio-tools>=1.62.0"

# 验证
python -c "import grpc; import grpc_tools; print(f'grpcio: {grpc.__version__}')"
```

### 状态

✅ 已修复（版本 pin 放宽 + 预编译 wheel）

---

## 问题 6: numpy 1.26.4 源码编译失败 — 无 C 编译器

### 现象

```powershell
pip install -r requirements.txt
# ...
# ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ['clang'], ...]
# WARNING: Failed to activate VS environment: Could not find vswhere.exe
# numpy 1.26.4 尝试用 Meson 从源码编译失败
```

### 根因分析

| 因素 | 说明 |
|:----|:------|
| numpy 版本 | `numpy==1.26.4` (2023) 没有 Python 3.13 的预编译 wheel |
| 构建系统 | numpy 使用 Meson 构建系统，需要 C 编译器 |
| 环境 | Windows 未安装 Microsoft C++ Build Tools / MSVC 编译器 |
| 已有版本 | 用户已通过 `--system-site-packages` 继承全局 `numpy 2.3.2` |

### 修复

同样放宽版本 pin：

```diff
- numpy==1.26.4
+ numpy>=1.26.4
```

用户已有的 `numpy 2.3.2 >= 1.26.4`，pip 直接跳过。

**注意**: 此处放宽可能引入 API 兼容性问题（numpy 2.x API 向后兼容但少数函数行为有变）。若后续 CosyVoice 运行时报 numpy 相关错误，可考虑安装 Microsoft C++ Build Tools 后编译 exact 版本。

### 状态

✅ 已修复（版本 pin 放宽，使用已有 2.3.2）

---

## 问题 7: pydantic-core==2.18.1 源码编译失败（Python 3.13 无 wheel）

### 现象

```powershell
pip install -r requirements.txt
# ...
# pydantic-core==2.18.1 显示为 .tar.gz (非 wheel)
# 卡在 "Preparing metadata (pyproject.toml)" 很久
# 最终需要 Rust 编译器从源码构建
```

### 根因分析

| 因素 | 说明 |
|:----|:------|
| 直接原因 | `pydantic==2.7.0` → 依赖 `pydantic-core==2.18.1`，**无 cp313 wheel** |
| 深层原因 | pydantic-core 是 Rust 实现，预编译 wheel 需要在 PyPI 上存在对应版本 |
| Python 3.13 | 2025 年 10 月发布，pydantic-core 2.18.x 发布时尚未提供 cp313 构建 |
| 危害 | 卡在 metadata 准备阶段，用户误以为 pip 死机 |

### 修复

升级 pydantic 到有 cp313 wheel 的版本：

```diff
- pydantic==2.7.0
+ pydantic>=2.10.0
```

```powershell
pip install "pydantic>=2.10.0" "pydantic-core>=2.27.0"
# → pydantic 2.13.4 + pydantic-core 2.46.4 (cp313 wheel, 瞬间完成)
```

### 状态

✅ 已修复（升级至 pydantic 2.13.4 + pydantic-core 2.46.4）

---

## 问题 8: pip 缓存权限错误（MarkupSafe wheel 损坏）

### 现象

```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied:
  'C:\users\lenovo\appdata\local\pip\cache\wheels\...\MarkupSafe-2.1.5-py3-none-any.whl'
```

### 根因分析

前序 pip 安装被中断导致缓存文件权限锁定。

### 修复

```powershell
pip cache purge
# Files removed: 1134 (6376.5 MB)
```

### 状态

✅ 已修复

---

## 问题 9: grpcio-tools 依赖回溯 — protobuf 版本冲突

### 现象

即使 `grpcio==1.83.0` 和 `grpcio-tools==1.83.0` 已安装，pip install -r 仍回溯：

```
INFO: pip is looking at multiple versions of grpcio-tools...
# 从 1.83.0 → 1.82.1 → ... → 1.62.3 (回溯 20+ 个版本)
# 最终 1.62.3 源码构建失败: ModuleNotFoundError: No module named 'pkg_resources'
```

### 根因分析

| 因素 | 说明 |
|:----|:------|
| protobuf 冲突 | `grpcio-tools==1.83.0` 自动升级了 `protobuf==4.25` → **`protobuf==7.35.1`** |
| requirements.txt | 仍保留 `protobuf==4.25`，与已安装的 7.35.1 冲突 |
| pip 行为 | pip 检测到冲突后回溯 grpcio-tools 版本，试图找到兼容 protobuf 4.25 的版本 |
| 旧版无 cp313 | grpcio-tools < 1.66 无 cp313 wheel → 源码编译 → 缺 pkg_resources → 崩溃 |

### 修复

删除 requirements.txt 中的 `protobuf==4.25` 行（由 grpcio-tools 自动管理版本）：

```powershell
Get-Content requirements.txt | Where-Object {$_ -notmatch 'protobuf==4.25'} | Set-Content requirements.txt
```

先预装新版 grpcio/grpcio-tools，让 pip 跳过回溯：

```powershell
pip install "grpcio==1.83.0" "grpcio-tools==1.83.0"
# → 自动安装 protobuf 7.35.1 (符合 >=4.25 的约束)
# → 后续 pip install -r 时 grpcio 已满足，跳过回溯
```

### 状态

✅ 已修复（删除 protobuf pin + 预装新版 grpcio-tools）

---

## 问题 10: pyworld==0.3.4 编译失败（无 MSVC Build Tools）

### 现象

```
Building wheel for pyworld (pyproject.toml) ... error
  error: Microsoft Visual C++ 14.0 or greater is required.
  Get it with "Microsoft C++ Build Tools"
```

### 根因分析

| 因素 | 说明 |
|:----|:------|
| pyworld 0.3.4 | 无 Python 3.13 的预编译 wheel |
| 构建依赖 | pyworld 使用 Cython 编写 C 扩展，需要 MSVC 编译器 |
| 用户环境 | Windows 11 未安装 Microsoft C++ Build Tools |

### 影响评估

pyworld 仅用于训练阶段（`cosyvoice/dataset/processor.py`），**推理不需要**：

```python
# 唯一引用 — cosyvoice/dataset/processor.py (训练数据预处理)
import pyworld as pw

# 推理入口 — cosyvoice/cli/cosyvoice.py (不引用 pyworld)
from cosyvoice.cli.cosyvoice import CosyVoice
```

### 解决方案

**方案 A**: 跳过，推理不受影响（推荐）

```powershell
# 从 requirements.txt 删除 pyworld 行
Get-Content requirements.txt | Where-Object {$_ -notmatch 'pyworld'} | Set-Content requirements.txt
```

**方案 B**: 安装 MSVC Build Tools 后编译（仅训练需要）
- 下载: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- 安装 "Desktop development with C++" workload

### 状态

⚠️ 已跳过（推理不受影响，训练需要时再装 MSVC）

---

## 问题 11: --no-deps 导致 20+ 传递依赖缺失

### 现象

```powershell
pip install -r requirements.txt --no-deps  # 跳过 pyworld
# 29 个主包安装成功 ✅
# 但 python -c "from cosyvoice.cli.cosyvoice import AutoModel" 失败
```

```
ModuleNotFoundError: No module named 'tqdm'           # 第一个缺失
# 之后还会缺: PyYAML, requests, huggingface-hub, tokenizers,
#             safetensors, filelock, packaging, regex, numba,
#             pytorch-lightning, torchmetrics, urllib3, ...
```

### 根因分析

--no-deps 只装 requirements.txt 中的 29 个包，所有传递依赖全部跳过。

安装后 pip 的依赖冲突报告显示 20+ 缺失项。

### 修复

重新运行完整依赖解析（不再用 --no-deps）：

```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org `
    --trusted-host download.pytorch.org `
    --trusted-host aiinfra.pkgs.visualstudio.com `
    -r requirements.txt
```

这次所有回溯问题已解决（grpcio 预装、protobuf 冲突解除、pydantic 升级），pip 应能正常完成。

### 状态

⏳ 待执行

---

## 问题 12: CosyVoice 仓库无 setup.py，无需 pip install -e .

### 现象

CosyVoice 仓库根目录没有 `setup.py`, `setup.cfg`, `pyproject.toml`。

```
CosyVoice/
├── cosyvoice/          ← 直接 import 即可
│   └── cli/
│       └── cosyvoice.py
├── requirements.txt
└── example.py
```

### 说明

CosyVoice 设计为直接 clone 后通过 PYTHONPATH 使用，无需 `pip install -e .`：

```python
# 在 CosyVoice 仓库根目录运行即可
from cosyvoice.cli.cosyvoice import AutoModel
```

### 状态

✅ 无需修复，直接使用

---

## 待解决问题汇总

| # | 问题 | 优先级 | 状态 | 解决方案 |
|:-:|:-----|:------:|:----:|:---------|
| 1 | CosyVoice 2 PyPI 安装失败 | 🔴 高 | 🟡 待修复 | 改从 GitHub 源码安装 (`QwenAudio/CosyVoice`) |
| 2 | 验证脚本缺失基础依赖检查 | 🟡 中 | ✅ 已修复 | 脚本 v2 已添加 |
| 3 | Windows 基础依赖预装 | 🟢 低 | ✅ 已安装 | `pip install` 完成 |
| 4 | CUDA→CPU fallback | 🟢 低 | ✅ 已实现 | 脚本 v2 已添加 |
| 5 | grpcio / grpcio-tools 编译失败 (pkg_resources) | 🔴 高 | ✅ 已修复 | 放宽版本 pin → 预编译 wheel |
| 6 | numpy 1.26.4 编译失败 (无 C 编译器) | 🟡 中 | ✅ 已修复 | 放宽版本 pin → 使用已有 2.3.2 |
| 7 | pydantic-core 2.18.1 编译失败 (缺 Rust) | 🔴 高 | ✅ 已修复 | 升级 pydantic>=2.10.0 |
| 8 | pip 缓存权限错误 (MarkupSafe) | 🟡 中 | ✅ 已修复 | pip cache purge |
| 9 | protobuf 冲突导致 grpcio-tools 回溯 | 🔴 高 | ✅ 已修复 | 删除 protobuf pin + 预装新版 |
| 10 | pyworld 编译失败 (无 MSVC) | 🟢 低 | ⚠️ 已跳过 | 推理不需要，跳过 |
| 11 | --no-deps 传递依赖缺失 (20+ 包) | 🔴 高 | ⏳ 待执行 | 重跑无 --no-deps |
| 12 | CosyVoice 无 setup.py | 🟢 低 | ✅ 已确认 | 无需 pip install -e . |
| 13 | Windows 全链路验证 (Step 7) | 🔴 高 | ⏳ 进行中 (~85%) | 补传递依赖 → 模型下载 → 测试 |

---

## 验证回归清单

```powershell
# 1. 基础依赖检查
pip install loguru numpy websockets pyyaml onnxruntime scipy sounddevice

# 2. 克隆 CosyVoice 2
cd D:\DevProject\PythonProject
git clone --recursive https://github.com/QwenAudio/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# 3. 修改 requirements.txt 放宽 grpcio/grpcio-tools/numpy 版本 pin
notepad requirements.txt
# grpcio==1.57.0 → grpcio>=1.57.0
# grpcio-tools==1.57.0 → grpcio-tools>=1.57.0
# numpy==1.26.4 → numpy>=1.26.4

# 4. 安装依赖
pip install -r requirements.txt

# 5. 安装 CosyVoice 本体
pip install -e .

# 6. 验证安装
python -c "from cosyvoice.cli.cosyvoice import CosyVoice; print('OK')"

# 7. 下载模型
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice-2-0.5B')"

# 8. 环境检查
cd D:\DevProject\PythonProject\airi-voice-module
python scripts\test_tts_windows.py --mode check

# 9. 合成验证 (无播放)
python scripts\test_tts_windows.py --mode synthesize

# 10. 全链路验证 (含播放)
python scripts\test_tts_windows.py --mode all
```
