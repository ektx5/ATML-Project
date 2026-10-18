"""
test_audio_utils.py — Unit tests for audio preprocessing utilities.

Run with: pytest tests/test_audio_utils.py -v
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.audio_utils import (
    pad_or_trim,
    vad_trim,
    compute_mel_spectrogram,
    apply_spec_augment,
    add_gaussian_noise,
    resample_waveform,
)


SAMPLE_RATE = 16000
N_MELS = 80

AUDIO_CFG = {
    "sample_rate": 16000,
    "max_duration_s": 2.0,
    "n_mels": 80,
    "n_fft": 512,
    "hop_length": 160,
    "win_length": 400,
    "fmin": 20.0,
    "fmax": 7600.0,
}


class TestPadOrTrim:
    def test_pads_short_waveform(self):
        # 1 second of audio at 16kHz = 16000 samples
        waveform = torch.randn(1, 16000)
        padded = pad_or_trim(waveform, max_duration_s=2.0, sample_rate=SAMPLE_RATE)
        assert padded.shape == (1, 32000)
        # Check padding on the right is zero
        assert torch.all(padded[..., 16000:] == 0.0)

    def test_trims_long_waveform(self):
        # 3 seconds of audio = 48000 samples
        waveform = torch.randn(1, 48000)
        trimmed = pad_or_trim(waveform, max_duration_s=2.0, sample_rate=SAMPLE_RATE, random_crop=False)
        assert trimmed.shape == (1, 32000)

    def test_preserves_exact_length(self):
        waveform = torch.randn(1, 32000)
        result = pad_or_trim(waveform, max_duration_s=2.0, sample_rate=SAMPLE_RATE)
        assert result.shape == (1, 32000)
        assert torch.all(result == waveform)


class TestVADTrim:
    def test_trims_silence(self):
        # 1000 zeros, 5000 active noise, 1000 zeros
        active = torch.randn(1, 5000) * 0.5
        silence = torch.zeros(1, 1000)
        waveform = torch.cat([silence, active, silence], dim=-1)
        trimmed = vad_trim(waveform, sample_rate=SAMPLE_RATE, top_db=20.0)
        assert trimmed.shape[-1] <= waveform.shape[-1]
        assert trimmed.shape[-1] > 0


class TestComputeMelSpectrogram:
    def test_output_shape(self):
        # 2 seconds at 16kHz
        waveform = torch.randn(1, 32000)
        mel_spec = compute_mel_spectrogram(
            waveform,
            sample_rate=AUDIO_CFG["sample_rate"],
            n_mels=AUDIO_CFG["n_mels"],
            n_fft=AUDIO_CFG["n_fft"],
            hop_length=AUDIO_CFG["hop_length"],
            win_length=AUDIO_CFG["win_length"],
            fmin=AUDIO_CFG["fmin"],
            fmax=AUDIO_CFG["fmax"],
        )
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] == N_MELS
        # Expected frames ~ (32000 // 160) + 1 = 201
        assert abs(mel_spec.shape[1] - 201) <= 2

    def test_finite_values(self):
        waveform = torch.randn(1, 16000) * 0.1
        mel_spec = compute_mel_spectrogram(waveform, sample_rate=16000, n_mels=80)
        assert torch.all(torch.isfinite(mel_spec))


class TestSpecAugment:
    def test_output_shape_unchanged(self):
        mel_spec = torch.randn(80, 200)
        augmented = apply_spec_augment(mel_spec, freq_mask_param=10, time_mask_param=20)
        assert augmented.shape == (80, 200)

    def test_contains_zeros_after_mask(self):
        mel_spec = torch.ones(80, 200)
        augmented = apply_spec_augment(mel_spec, freq_mask_param=20, time_mask_param=40)
        assert (augmented == 0.0).any()


class TestResample:
    def test_resample_ratio(self):
        waveform = torch.randn(1, 8000)  # 0.5s at 16kHz
        resampled = resample_waveform(waveform, orig_sr=16000, target_sr=8000)
        assert resampled.shape[-1] == 4000
