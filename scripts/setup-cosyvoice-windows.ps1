<#=============================================================================
   AIRI Voice Module — CosyVoice 2 Windows 安装脚本
   =============================================================================
   说明: 在 Windows 11 + Python 3.13 环境中安装 CosyVoice 2 推理依赖
   用法: PowerShell 右键 "以管理员身份运行" 或:
         Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
         .\setup-cosyvoice-windows.ps1
   项目路径: D:\DevProject\PythonProject\airi-voice-module
   CosyVoice: D:\DevProject\PythonProject\CosyVoice
   ===========================================================================#>

$ErrorActionPreference = "Stop"
$ProgressPreference = 'Continue'

# ─── 颜色输出辅助 ──────────────────────────────────────────────────────────
function Write-Step($msg)  { Write-Host "`n═══════════════════════════════════════════════" -ForegroundColor Cyan; Write-Host "  $msg" -ForegroundColor Cyan; Write-Host "═══════════════════════════════════════════════" }
function Write-OK($msg)    { Write-Host "  ✅ $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "  ⚠️  $msg" -ForegroundColor Yellow }
function Write-Info($msg)  { Write-Host "  ℹ️  $msg" -ForegroundColor Gray }

# ─── 路径配置 ───────────────────────────────────────────────────────────────
$COSYVOICE_DIR = "D:\DevProject\PythonProject\CosyVoice"
$AIRI_DIR      = "D:\DevProject\PythonProject\airi-voice-module"

Write-Step "AIRI CosyVoice 2 — Windows 安装脚本 (Python 3.13)"

# ============================================================================
# 第一步: 检查环境
# ============================================================================
Write-Host "`n[Step 1/7] 环境检查..." -ForegroundColor Yellow

# 1.1 检查 Python 版本
$pyVer = python --version 2>&1
Write-Info "Python: $pyVer"
if ($pyVer -notmatch "3\.13") {
    Write-Warn "推荐 Python 3.13，当前版本: $pyVer"
}

# 1.2 检查目录是否存在
if (-not (Test-Path $COSYVOICE_DIR)) {
    Write-Warn "CosyVoice 目录不存在，请先克隆:"
    Write-Info "  git clone --recursive https://github.com/QwenAudio/CosyVoice.git"
    Write-Info "  cd CosyVoice && git submodule update --init --recursive"
    $choice = Read-Host "是否继续? (y/n, 默认 n)"
    if ($choice -ne 'y') { exit 1 }
} else {
    Write-OK "CosyVoice 目录: $COSYVOICE_DIR"
}

# 1.3 检查 PyTorch CUDA
try {
    $cudaTest = python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>&1
    Write-Info "  $cudaTest"
} catch {
    Write-Warn "PyTorch 未安装或不可用，后续可能需要单独安装"
}

Write-OK "环境检查完成"

# ============================================================================
# 第二步: 备份原始 requirements.txt
# ============================================================================
Write-Host "`n[Step 2/7] 备份 requirements.txt..." -ForegroundColor Yellow

$reqFile = Join-Path $COSYVOICE_DIR "requirements.txt"
$bakFile = Join-Path $COSYVOICE_DIR "requirements.txt.backup"

if (Test-Path $reqFile) {
    Copy-Item $reqFile $bakFile -Force
    Write-OK "已备份到 requirements.txt.backup"
} else {
    Write-Warn "requirements.txt 不存在于 $COSYVOICE_DIR"
}

# ============================================================================
# 第三步: 创建推理最小依赖清单 (requirements_infer.txt)
# ============================================================================
Write-Host "`n[Step 3/7] 创建推理最小依赖清单..." -ForegroundColor Yellow

$inferReq = @"
--extra-index-url https://download.pytorch.org/whl/cu121
# CosyVoice 2 推理最小依赖 (2026-07-28)
# 去除了训练/WebUI/ASR/可视化 等无关包

# 模型下载与加载
modelscope>=1.20.0
soundfile>=0.12.0
tqdm>=4.0

# 配置框架
omegaconf>=2.3.0
hydra-core>=1.3.0
hyperpyyaml>=1.2.0

# 数据处理
numpy>=1.26.0
protobuf>=4.25
pyarrow>=18.0.0

# gRPC (已放宽 pin 适配 Python 3.13)
grpcio>=1.57.0
grpcio-tools>=1.57.0

# 文本处理
inflect>=7.3.1
wetext>=0.0.4
x-transformers>=2.11.24

# 工具
rich>=13.7.1
pydantic>=2.7.0
"@

$inferReq | Out-File -FilePath (Join-Path $COSYVOICE_DIR "requirements_infer.txt") -Encoding utf8
Write-OK "已创建 requirements_infer.txt"

# ============================================================================
# 第四步: 安装推理依赖
# ============================================================================
Write-Host "`n[Step 4/7] 安装推理依赖 (pip install -r requirements_infer.txt)..." -ForegroundColor Yellow
Write-Host "      可能耗时 5-15 分钟，取决于网络速度" -ForegroundColor Gray

Set-Location $COSYVOICE_DIR

pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org `
    --trusted-host download.pytorch.org `
    --trusted-host aiinfra.pkgs.visualstudio.com `
    -r requirements_infer.txt

if ($LASTEXITCODE -eq 0) {
    Write-OK "推理依赖安装完成"
} else {
    Write-Warn "部分依赖安装失败，请检查上方错误信息"
}

# ============================================================================
# 第五步: 安装 CosyVoice 本体
# ============================================================================
Write-Host "`n[Step 5/7] 安装 CosyVoice 本体 (pip install -e .)..." -ForegroundColor Yellow

Set-Location $COSYVOICE_DIR
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e .

if ($LASTEXITCODE -eq 0) {
    Write-OK "CosyVoice 本体安装成功"
} else {
    Write-Warn "CosyVoice 本体安装失败，请检查上方错误信息"
}

# ============================================================================
# 第六步: 验证安装
# ============================================================================
Write-Host "`n[Step 6/7] 验证安装..." -ForegroundColor Yellow

try {
    $result = python -c "from cosyvoice.cli.cosyvoice import CosyVoice; print('OK')" 2>&1
    if ($result -eq "OK") {
        Write-OK "CosyVoice 验证通过!"
    } else {
        Write-Warn "验证输出: $result"
    }
} catch {
    Write-Warn "验证失败: $_"
}

# ============================================================================
# 第七步: 下载模型
# ============================================================================
Write-Host "`n[Step 7/7] 下载 CosyVoice-2-0.5B 模型 (~1.5GB)..." -ForegroundColor Yellow
Write-Host "      是否现在下载？(可能需要 5-15 分钟)" -ForegroundColor Gray

$choice = Read-Host "立即下载? (y/n, 默认 n)"
if ($choice -eq 'y') {
    Set-Location $COSYVOICE_DIR

    if (-not (Test-Path "pretrained_models")) {
        New-Item -ItemType Directory -Path "pretrained_models" -Force | Out-Null
    }

    python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice-2-0.5B')"

    if ($LASTEXITCODE -eq 0) {
        Write-OK "模型下载完成: pretrained_models/CosyVoice-2-0.5B"
    } else {
        Write-Warn "模型下载失败，稍后可用以下命令单独下载:"
        Write-Info "  python -c `"from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice-2-0.5B')`""
    }
} else {
    Write-Info "跳过模型下载，稍后可用以下命令下载:"
    Write-Info "  cd $COSYVOICE_DIR"
    Write-Info "  python -c `"from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice-2-0.5B')`""
}

# ============================================================================
# 完成
# ============================================================================
Write-Step "安装完成!"

Write-Host "  接下来可以运行验证脚本:" -ForegroundColor Green
Write-Host ""
Write-Host "  cd $AIRI_DIR" -ForegroundColor White
Write-Host "  python scripts\test_tts_windows.py --mode check" -ForegroundColor White
Write-Host "  python scripts\test_tts_windows.py --mode synthesize" -ForegroundColor White
Write-Host "  python scripts\test_tts_windows.py --mode play" -ForegroundColor White

Write-Host ""
Write-Host "  如果模型还没下载:" -ForegroundColor Green
Write-Host "  cd $COSYVOICE_DIR" -ForegroundColor White
Write-Host "  python -c `"from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice-2-0.5B')`"" -ForegroundColor White

Write-Host ""
Write-Host "  💡 如果遇到任何错误，请截图发给我" -ForegroundColor Cyan
