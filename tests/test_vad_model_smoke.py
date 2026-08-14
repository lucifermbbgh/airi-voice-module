"""
Silero VAD 真实模型 smoke test

验证 `_get_speech_prob` 对真实语音输出正常概率，防止 context 前缀缺失等问题
再次发生（该问题曾因 tests/test_vad.py mock 模型加载而被掩盖）。

背景：2026-08-15 发现 Silero VAD v5 模型的输入需要 576 样本
（64 样本 context + 512 样本 chunk），原代码只传 512 样本，
导致正常语音的概率只有 0.002（几乎静音），VAD 无法触发。
修复后概率恢复正常（语音段 > 0.9）。

用法:
    python -m tests.test_vad_model_smoke
    pytest tests/test_vad_model_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

MODEL_PATH = Path("models/silero_vad.onnx")
FIXTURE_PATH = Path("tests/fixtures/speech_sample.npy")


@pytest.fixture(scope="module")
def vad():
    """加载真实 Silero VAD 模型（依赖/文件缺失时 skip 而非 fail）。"""
    pytest.importorskip("onnxruntime")
    if not MODEL_PATH.exists():
        pytest.skip(f"模型文件不存在: {MODEL_PATH}")
    from src.vad.silero_vad import SileroVAD

    v = SileroVAD(model_path=str(MODEL_PATH), threshold=0.3)
    v.load_model()
    return v


@pytest.fixture(scope="module")
def speech_audio():
    """加载真实语音 fixture（TTS 合成，缺失时 skip）。"""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"语音 fixture 不存在: {FIXTURE_PATH}")
    return np.load(FIXTURE_PATH)


def test_speech_prob_above_threshold(vad, speech_audio):
    """真实语音的 VAD 概率应显著高于阈值，证明模型正确识别语音。

    修复前的 bug：`_get_speech_prob` 缺少 64 样本 context 前缀（感受野），
    导致正常语音概率只有 0.002。本测试断言 >0.5 的帧占比 > 30%，
    防止该 bug 回归。
    """
    probs = []
    for i in range(0, len(speech_audio) - 512 + 1, 512):
        frame = speech_audio[i : i + 512].astype(np.float32)
        probs.append(vad._get_speech_prob(frame))

    probs = np.array(probs)
    high_ratio = (probs > 0.5).sum() / len(probs)

    assert high_ratio > 0.3, (
        f"真实语音的 VAD 概率异常低：>0.5 帧占比 {high_ratio:.1%}，"
        f"最高概率 {probs.max():.4f}。"
        f"可能缺失 context 前缀（感受野）或模型文件有问题。"
    )
