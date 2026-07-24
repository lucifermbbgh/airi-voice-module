"""
Environment check test — verifies that all project dependencies
are installed and the runtime environment is properly configured.

Can be run in three ways:

1. As pytest:
       python -m pytest tests/test_env_check.py -v

2. Standalone (detailed report):
       python tests/test_env_check.py

3. Import and reuse EnvironmentChecker in other tests:
       from tests.test_env_check import EnvironmentChecker
       env = EnvironmentChecker()
       if env.pytorch_available:
           print(env.pytorch_version)
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass, field

import pytest


# ── Dependency specifications ───────────────────────────────────────

DEPENDENCIES: list[dict] = [
    {"name": "numpy", "module": "numpy", "pip": "numpy", "required": True},
    {"name": "sounddevice", "module": "sounddevice", "pip": "sounddevice", "required": True,
     "note": "Requires PortAudio (usually pre-installed)"},
    {"name": "scipy", "module": "scipy", "pip": "scipy", "required": True},
    {"name": "onnxruntime", "module": "onnxruntime", "pip": "onnxruntime", "required": True},
    {"name": "websockets", "module": "websockets", "pip": "websockets", "required": True},
    {"name": "pyyaml", "module": "yaml", "pip": "pyyaml", "required": True},
    {"name": "loguru", "module": "loguru", "pip": "loguru", "required": True},
    {"name": "pytest", "module": "pytest", "pip": "pytest", "required": True},
    {"name": "pytest-asyncio", "module": "pytest_asyncio", "pip": "pytest-asyncio", "required": True},
    {"name": "faster-whisper", "module": "faster_whisper", "pip": "faster-whisper", "required": False,
     "note": "Required for STT (Phase 2)"},
    {"name": "huggingface-hub", "module": "huggingface_hub", "pip": "huggingface-hub", "required": False,
     "note": "Required for model downloads"},
    {"name": "torch", "module": "torch", "pip": "torch", "required": False,
     "note": "Required for TTS (Phase 3) / CUDA acceleration"},
    {"name": "cosyvoice", "module": "cosyvoice", "pip": "cosyvoice", "required": False,
     "note": "Required for CosyVoice 2 TTS (Phase 3)"},
    {"name": "edge-tts", "module": "edge_tts", "pip": "edge-tts", "required": False,
     "note": "Optional TTS fallback (online)"},
]


@dataclass
class EnvCheckResult:
    """Result of a single environment check."""
    name: str
    available: bool = False
    version: str = ""
    error: str = ""
    note: str = ""


@dataclass
class EnvReport:
    """Full environment check report."""
    python_version: str = ""
    platform: str = ""
    architecture: str = ""
    checks: list[EnvCheckResult] = field(default_factory=list)
    cuda_available: bool = False
    cuda_version: str = ""
    cuda_device_count: int = 0
    cuda_device_name: str = ""

    @property
    def passed(self) -> int:
        """Number of passed checks."""
        return sum(1 for c in self.checks if c.available)

    @property
    def failed(self) -> int:
        """Number of failed checks."""
        return sum(1 for c in self.checks if not c.available)

    @property
    def total(self) -> int:
        """Total number of checks."""
        return len(self.checks)

    @property
    def missing_required(self) -> list[EnvCheckResult]:
        """List of missing required dependencies."""
        return [c for c in self.checks
                if not c.available and self._is_required(c.name)]

    @staticmethod
    def _is_required(name: str) -> bool:
        for dep in DEPENDENCIES:
            if dep["name"] == name:
                return dep["required"]
        return False


class EnvironmentChecker:
    """Reusable environment checker for AIRI Voice Module.

    Usage:
        checker = EnvironmentChecker()
        report = checker.check_all()

        # Print summary
        print(f"PyTorch: {report.checks[12].available}")
        print(f"CUDA: {report.cuda_available}")

        # Use in tests
        if checker.pytorch_available:
            import torch
    """

    def __init__(self):
        self._report: EnvReport | None = None

    # ── Public Properties ───────────────────────────────────────────

    @property
    def pytorch_available(self) -> bool:
        """Check if PyTorch is available."""
        return self._get_check("torch").available if self._report else False

    @property
    def pytorch_version(self) -> str:
        """Get PyTorch version."""
        return self._get_check("torch").version if self._report else ""

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._report.cuda_available if self._report else False

    @property
    def cuda_version(self) -> str:
        """Get CUDA version."""
        return self._report.cuda_version if self._report else ""

    @property
    def faster_whisper_available(self) -> bool:
        """Check if faster-whisper is available."""
        return self._get_check("faster-whisper").available if self._report else False

    @property
    def cosyvoice_available(self) -> bool:
        """Check if cosyvoice is available."""
        return self._get_check("cosyvoice").available if self._report else False

    # ── Main Check ──────────────────────────────────────────────────

    def check_all(self) -> EnvReport:
        """Run all environment checks.

        Returns:
            EnvReport with all check results.
        """
        report = EnvReport()
        report.python_version = sys.version
        report.platform = platform.system()
        report.architecture = platform.machine()

        # Check each dependency
        for dep in DEPENDENCIES:
            result = self._check_module(
                name=dep["name"],
                module_name=dep["module"],
                note=dep.get("note", ""),
            )
            report.checks.append(result)

        # Check CUDA (requires PyTorch)
        self._check_cuda(report)

        self._report = report
        return report

    def _check_module(self, name: str, module_name: str,
                      note: str = "") -> EnvCheckResult:
        """Check if a Python module is installed and get its version.

        Args:
            name: Display name.
            module_name: Python import name.
            note: Optional note about the dependency.

        Returns:
            EnvCheckResult.
        """
        result = EnvCheckResult(name=name, note=note)
        try:
            mod = importlib.import_module(module_name)
            result.available = True
            result.version = getattr(mod, "__version__", "")
        except ImportError as e:
            result.available = False
            result.error = str(e)
        except Exception as e:
            result.available = False
            result.error = f"Unexpected error: {e}"
        return result

    def _check_cuda(self, report: EnvReport) -> None:
        """Check CUDA availability via PyTorch.

        Args:
            report: Report to update with CUDA info.
        """
        # Only check CUDA if PyTorch is available
        torch_check = self._get_check_from_report(report, "torch")
        if not torch_check.available:
            return

        try:
            import torch
            report.cuda_available = torch.cuda.is_available()
            report.cuda_version = torch.version.cuda or ""

            if report.cuda_available:
                report.cuda_device_count = torch.cuda.device_count()
                report.cuda_device_name = torch.cuda.get_device_name(0) \
                    if report.cuda_device_count > 0 else ""
        except Exception as e:
            report.cuda_available = False
            report.cuda_version = f"Check error: {e}"

    def _get_check(self, name: str) -> EnvCheckResult:
        """Get check result by name from current report."""
        if not self._report:
            return EnvCheckResult(name=name, available=False)
        return self._get_check_from_report(self._report, name)

    @staticmethod
    def _get_check_from_report(report: EnvReport, name: str) -> EnvCheckResult:
        """Get check result by name from a report."""
        for c in report.checks:
            if c.name == name:
                return c
        return EnvCheckResult(name=name, available=False)


# ── Report Formatting ───────────────────────────────────────────────

def format_report(report: EnvReport) -> str:
    """Format environment report as a readable string.

    Args:
        report: EnvReport from EnvironmentChecker.

    Returns:
        Formatted report string.
    """
    lines = []
    lines.append("=" * 55)
    lines.append("  AIRI Voice Module — Environment Check")
    lines.append("=" * 55)
    lines.append(f"  Python:    {report.python_version.split()[0]}")
    lines.append(f"  Platform:  {report.platform} ({report.architecture})")
    lines.append(f"  Checks:    {report.passed}/{report.total} passed")
    lines.append("")

    # Dependencies
    lines.append("  ── Dependencies ──")
    for check in report.checks:
        status = "✅" if check.available else "❌"
        version = f" ({check.version})" if check.version else ""
        note = f"  — {check.note}" if check.note else ""
        lines.append(f"  {status} {check.name:<18}{version}{note}")
        if check.error:
            lines.append(f"       error: {check.error}")

    # CUDA (if PyTorch available)
    if report.cuda_available:
        lines.append("")
        lines.append("  ── CUDA ──")
        lines.append(f"  ✅ CUDA {report.cuda_version}")
        lines.append(f"     Device count: {report.cuda_device_count}")
        lines.append(f"     Device name:  {report.cuda_device_name}")
    elif report.cuda_version:
        lines.append("")
        lines.append("  ── CUDA ──")
        lines.append(f"  ❌ CUDA not available (PyTorch: {report.cuda_version})")

    # Missing required
    missing = report.missing_required
    if missing:
        lines.append("")
        lines.append("  ── Missing Required ──")
        for m in missing:
            lines.append(f"  ❌ {m.name}: pip install <package>")

    lines.append("")
    lines.append("=" * 55)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Pytest Tests
# ══════════════════════════════════════════════════════════════════════


class TestEnvironment:
    """Environment check tests (runs with pytest)."""

    @pytest.fixture(scope="module")
    def report(self) -> EnvReport:
        """Run environment check once per test class."""
        checker = EnvironmentChecker()
        return checker.check_all()

    def test_python_version(self, report):
        """Python version is 3.10+."""
        major, minor = sys.version_info[:2]
        assert major >= 3, f"Python 3 required, got {major}"
        assert minor >= 10, f"Python 3.10+ required, got {minor}.{major}"

    def test_required_deps(self, report):
        """All required dependencies are installed."""
        missing = report.missing_required
        assert len(missing) == 0, \
            f"Missing required deps: {[m.name for m in missing]}"

    def test_numpy(self, report):
        """numpy is installed."""
        self._assert_dep(report, "numpy")

    def test_sounddevice(self, report):
        """sounddevice is installed."""
        self._assert_dep(report, "sounddevice")

    def test_scipy(self, report):
        """scipy is installed."""
        self._assert_dep(report, "scipy")

    def test_onnxruntime(self, report):
        """onnxruntime is installed."""
        self._assert_dep(report, "onnxruntime")

    def test_websockets(self, report):
        """websockets is installed."""
        self._assert_dep(report, "websockets")

    def test_pyyaml(self, report):
        """pyyaml is installed."""
        self._assert_dep(report, "pyyaml")

    def test_loguru(self, report):
        """loguru is installed."""
        self._assert_dep(report, "loguru")

    def test_pytest(self, report):
        """pytest is installed."""
        self._assert_dep(report, "pytest")

    def test_pytorch_stub(self, report):
        """Check PyTorch availability (optional)."""
        check = self._get_check(report, "torch")
        if not check.available:
            pytest.skip("PyTorch not installed (optional for Phase 3 TTS)")

    def test_cuda_stub(self, report):
        """Check CUDA availability (optional)."""
        if not report.cuda_available:
            pytest.skip("CUDA not available (GPU acceleration not required)")

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _assert_dep(report: EnvReport, name: str):
        """Assert a dependency is installed."""
        check = EnvironmentChecker._get_check_from_report(report, name)
        assert check.available, \
            f"{name} not installed. Install: pip install {_get_pip_name(name)}"

    @staticmethod
    def _get_check(report: EnvReport, name: str) -> EnvCheckResult:
        return EnvironmentChecker._get_check_from_report(report, name)


def _get_pip_name(name: str) -> str:
    """Get pip package name for a dependency."""
    for dep in DEPENDENCIES:
        if dep["name"] == name:
            return dep["pip"]
    return name


# ══════════════════════════════════════════════════════════════════════
# Standalone Entry Point
# ══════════════════════════════════════════════════════════════════════


def main():
    """Standalone entry point — prints detailed report."""
    checker = EnvironmentChecker()
    report = checker.check_all()
    print(format_report(report))

    # Exit with code 1 if missing required deps
    if report.missing_required:
        sys.exit(1)


if __name__ == "__main__":
    main()


