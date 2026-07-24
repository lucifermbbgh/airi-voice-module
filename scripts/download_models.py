#!/usr/bin/env python3
"""
Model download script for AIRI Voice Module.

Downloads Faster-Whisper models from HuggingFace Hub to the local
models/ directory for offline use. This avoids the 9-minute anonymous
download delay during first inference.

Usage:
    # Download default (tiny) model
    python scripts/download_models.py

    # Download specific model
    python scripts/download_models.py --model-size small

    # Download and verify inference
    python scripts/download_models.py --model-size small --verify

    # Download to custom directory
    python scripts/download_models.py --model-size small --dir D:/models/whisper

    # List available models
    python scripts/download_models.py --list-models

    # Force re-download
    python scripts/download_models.py --model-size small --force

    # Set HuggingFace token (avoids rate limiting)
    #   export HF_TOKEN=hf_your_token_here
    #   python scripts/download_models.py --model-size small
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from pathlib import Path

# Ensure project root is in path for imports
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Available Models ────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "tiny": {
        "repo": "Systran/faster-whisper-tiny",
        "size_mb": 75,
        "description": "Fastest, lowest accuracy (~15% CER)",
        "recommended": False,
    },
    "base": {
        "repo": "Systran/faster-whisper-base",
        "size_mb": 140,
        "description": "Good balance for quick testing",
        "recommended": False,
    },
    "small": {
        "repo": "Systran/faster-whisper-small",
        "size_mb": 460,
        "description": "Best for real use (~5% CER on Chinese) ✅",
        "recommended": True,
    },
    "medium": {
        "repo": "Systran/faster-whisper-medium",
        "size_mb": 1200,
        "description": "Higher accuracy, more memory",
        "recommended": False,
    },
    "large-v3": {
        "repo": "Systran/faster-whisper-large-v3",
        "size_mb": 3000,
        "description": "Best accuracy, heavy (~3GB RAM)",
        "recommended": False,
    },
}

_DEFAULT_MODEL = "tiny"


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download Faster-Whisper models for AIRI Voice Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "__doc__" in dir() else "",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=_DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Model size to download (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Download directory (default: models/faster-whisper-{size})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run inference test after download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists locally",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    return parser.parse_args()


def _list_models() -> None:
    """Print available models with info."""
    print("\n" + "=" * 65)
    print("  Available Faster-Whisper Models")
    print("=" * 65)
    print(f"  {'Name':<12} {'Size':<8} {'Recommended':<14}  Description")
    print(f"  {'─'*12:12} {'─'*8:8} {'─'*14:14}  {'─'*30}")
    for name, info in MODELS.items():
        rec = "✅" if info["recommended"] else ""
        print(f"  {name:<12} {info['size_mb']:<4}MB   {rec:<14}  {info['description']}")
    print()


def _get_download_dir(model_size: str, custom_dir: str | None) -> Path:
    """Get the download directory for the model.

    Args:
        model_size: Model size identifier.
        custom_dir: Custom directory override.

    Returns:
        Path to download directory.
    """
    if custom_dir:
        return Path(custom_dir)
    return _PROJECT_ROOT / "models" / f"faster-whisper-{model_size}"


def _check_existing(download_dir: Path, force: bool) -> bool:
    """Check if model files already exist locally.

    Args:
        download_dir: Target download directory.
        force: Force re-download flag.

    Returns:
        True if model already exists and we can skip download.
    """
    if force:
        return False

    # Check for the model.bin file which is the core model weight
    if download_dir.exists():
        # Count files — typical model has at least 5-10 files (model.bin, config.json, tokenizer, etc.)
        files = list(download_dir.iterdir())
        if len(files) >= 3:
            # Check for at least one model file
            has_model_file = any(
                f.name.startswith("model") for f in files
                if f.is_file()
            )
            if has_model_file:
                return True
    return False


def download_model(
    model_size: str,
    download_dir: Path,
    force: bool = False,
) -> bool:
    """Download a Faster-Whisper model from HuggingFace Hub.

    Args:
        model_size: Model size identifier.
        download_dir: Directory to save model files.
        force: Force re-download.

    Returns:
        True if download succeeded.
    """
    model_info = MODELS[model_size]
    repo_id = model_info["repo"]

    # Check if already downloaded
    if _check_existing(download_dir, force):
        print(f"  ✅ Model already exists at: {download_dir}")
        print(f"     (use --force to re-download)")
        return True

    # Ensure parent directory exists
    download_dir.mkdir(parents=True, exist_ok=True)

    # Get HF token (optional — avoids rate limiting)
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        print(f"  🔑 Using HF_TOKEN (authenticated — faster downloads)")
    else:
        print(f"  ⚠️  No HF_TOKEN set. Anonymous downloads are rate-limited.")
        print(f"     Set HF_TOKEN for faster downloads:")
        print(f"     export HF_TOKEN=hf_your_token_here")

    model_size_mb = model_info["size_mb"]
    estimated_minutes = math.ceil(model_size_mb / 100)  # ~100MB/min anonymous
    print(f"  📦 Downloading {repo_id} ({model_size} ≈ {model_size_mb}MB)...")
    print(f"     Estimated: ~{max(1, estimated_minutes)} min (anonymous)")
    print(f"     Saving to: {download_dir}")
    print()

    # Download via huggingface_hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "  ❌ huggingface_hub not installed.\n"
            "     Install: pip install huggingface_hub"
        )
        return False

    start = time.monotonic()

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,  # Windows compatibility
            token=hf_token,
            ignore_patterns=["*.h5", "*.ot"],  # Skip non-essential files
        )
    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        # Clean up partial download
        if download_dir.exists():
            import shutil
            shutil.rmtree(download_dir)
        return False

    elapsed = time.monotonic() - start
    elapsed_min = elapsed / 60

    print(f"\n  ✅ Download complete in {elapsed:.0f}s ({elapsed_min:.1f} min)")
    print(f"     Location: {download_dir}")

    # Count downloaded files
    file_count = len(list(download_dir.rglob("*")))
    total_size_mb = sum(
        f.stat().st_size for f in download_dir.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    print(f"     Files: {file_count}, Total size: {total_size_mb:.0f}MB")

    return True


def verify_model(model_size: str, model_dir: Path) -> bool:
    """Verify model by running a short inference test.

    Creates synthetic audio, loads the model, and transcribes it
    to confirm the model works correctly.

    Args:
        model_size: Model size identifier.
        model_dir: Directory containing model files.

    Returns:
        True if verification passed.
    """
    print(f"\n  🔍 Verifying {model_size} model...")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "  ❌ faster-whisper not installed.\n"
            "     Install: pip install faster-whisper"
        )
        return False

    # Create synthetic audio (1 second of 440Hz tone)
    import numpy as np

    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)

    try:
        load_start = time.monotonic()
        model = WhisperModel(
            model_size_or_path=str(model_dir),
            device="cpu",
            compute_type="int8",
        )
        load_time = time.monotonic() - load_start
        print(f"     Load time: {load_time:.1f}s")

        infer_start = time.monotonic()
        segments, info = model.transcribe(audio=audio, beam_size=1)
        infer_time = time.monotonic() - infer_start

        # Collect text (should be empty or noise — that's fine for synthetic audio)
        text = ""
        for seg in segments:
            text += seg.text

        print(f"     Inference: {infer_time:.3f}s")
        print(f"     Detected language: {info.language} (prob={info.language_probability:.2f})")
        print(f"     Output: \"{text[:80]}\"")

        # Cleanup
        del model

        print(f"  ✅ Verification passed!")
        return True

    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False


def main() -> None:
    """Main entry point."""
    args = _parse_args()

    if args.list_models:
        _list_models()
        return

    model_size = args.model_size
    download_dir = _get_download_dir(model_size, args.dir)

    print(f"\n  {'='*55}")
    print(f"    AIRI Voice Module — Model Downloader")
    print(f"  {'='*55}")
    print()

    # Download model
    success = download_model(
        model_size=model_size,
        download_dir=download_dir,
        force=args.force,
    )

    if not success:
        print("\n  ❌ Download failed. Please try again.")
        sys.exit(1)

    # Verify if requested
    if args.verify:
        success = verify_model(model_size, download_dir)
        if not success:
            print("\n  ⚠️  Download succeeded but verification failed.")
            print("     The model may still work — try running the inference test manually.")
            sys.exit(1)

    print(f"\n  ✅ Done! Model '{model_size}' is ready for use.")
    print(f"     Configure: model_size={model_size}, model_dir={download_dir}")
    print()


if __name__ == "__main__":
    main()
