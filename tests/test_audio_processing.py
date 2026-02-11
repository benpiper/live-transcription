"""
Tests for audio_processing module.

Tests voice activity detection, noise reduction, and normalization.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_processing import (
    is_speech,
    reduce_noise,
    normalize_rms,
    apply_audio_enhancements
)


class TestVAD(unittest.TestCase):
    """Test Voice Activity Detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 1.0
        self.num_samples = int(self.sample_rate * self.duration)

    def test_vad_with_silence(self):
        """VAD should detect silence as non-speech."""
        # Generate silent audio
        silent_audio = np.zeros(self.num_samples, dtype=np.float32)

        result, confidence = is_speech(
            silent_audio,
            sample_rate=self.sample_rate,
            confidence_threshold=0.5
        )

        # Should either detect no speech or return None if VAD unavailable
        if result is not None:
            self.assertFalse(result, "VAD should detect silence as non-speech")
            self.assertLess(confidence, 0.5, "Confidence should be below threshold for silence")

    def test_vad_with_noise(self):
        """VAD should handle noise appropriately."""
        # Generate white noise
        noise = np.random.normal(0, 0.01, self.num_samples).astype(np.float32)

        result, confidence = is_speech(
            noise,
            sample_rate=self.sample_rate,
            confidence_threshold=0.5
        )

        # Should detect noise as not speech (or unavailable)
        if result is not None:
            # Confidence should be reasonably low for pure noise
            self.assertLess(confidence, 0.7, "Noise should not be high confidence speech")

    def test_vad_returns_confidence(self):
        """VAD should return confidence score in valid range."""
        # Generate arbitrary audio
        audio = np.random.normal(0, 0.05, self.num_samples).astype(np.float32)

        result, confidence = is_speech(
            audio,
            sample_rate=self.sample_rate,
            confidence_threshold=0.5
        )

        # If VAD returns result, confidence should be in [0, 1]
        if result is not None:
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)


class TestNoiseReduction(unittest.TestCase):
    """Test spectral noise reduction."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 0.5
        self.num_samples = int(self.sample_rate * self.duration)

    def test_noise_reduction_returns_array(self):
        """Noise reduction should return numpy array."""
        audio = np.random.normal(0, 0.05, self.num_samples).astype(np.float32)

        result = reduce_noise(
            audio,
            sample_rate=self.sample_rate,
            stationary=True
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, audio.shape)

    def test_noise_reduction_preserves_dtype(self):
        """Noise reduction should preserve float32 dtype."""
        audio = np.random.normal(0, 0.05, self.num_samples).astype(np.float32)

        result = reduce_noise(audio, sample_rate=self.sample_rate)

        self.assertEqual(result.dtype, np.float32)

    def test_noise_reduction_handles_silence(self):
        """Noise reduction should handle silent audio."""
        silent_audio = np.zeros(self.num_samples, dtype=np.float32)

        result = reduce_noise(silent_audio, sample_rate=self.sample_rate)

        # Should remain silent
        self.assertLess(np.max(np.abs(result)), 1e-4)

    def test_noise_reduction_reduces_energy(self):
        """Noise reduction should generally reduce audio energy (if available)."""
        audio = np.random.normal(0, 0.05, self.num_samples).astype(np.float32)

        result = reduce_noise(audio, sample_rate=self.sample_rate)

        original_energy = np.sum(audio ** 2)
        reduced_energy = np.sum(result ** 2)

        # Noise reduction should reduce energy (most noise is removed)
        # If noisereduce is unavailable, result equals original (acceptable fallback)
        self.assertLessEqual(reduced_energy, original_energy + 1e-6)


class TestRMSNormalization(unittest.TestCase):
    """Test RMS-based normalization."""

    def test_rms_normalization_to_target(self):
        """RMS normalization should achieve target RMS level."""
        # Create audio with known RMS
        audio = np.random.normal(0, 0.5, 16000).astype(np.float32)
        target_rms = 0.1

        result = normalize_rms(audio, target_rms=target_rms)

        # Calculate actual RMS of result
        actual_rms = np.sqrt(np.mean(result ** 2))

        # Should be close to target (allowing for rounding/clipping)
        self.assertAlmostEqual(actual_rms, target_rms, places=2)

    def test_rms_normalization_preserves_shape(self):
        """RMS normalization should preserve audio shape."""
        audio = np.random.normal(0, 0.05, 16000).astype(np.float32)

        result = normalize_rms(audio, target_rms=0.1)

        self.assertEqual(result.shape, audio.shape)

    def test_rms_normalization_handles_silence(self):
        """RMS normalization should handle silent audio."""
        silent_audio = np.zeros(16000, dtype=np.float32)

        result = normalize_rms(silent_audio, target_rms=0.1)

        # Should remain silent
        self.assertEqual(np.max(np.abs(result)), 0.0)

    def test_rms_normalization_low_target(self):
        """RMS normalization should work with low target levels."""
        audio = np.random.normal(0, 0.5, 16000).astype(np.float32)

        result = normalize_rms(audio, target_rms=0.01)

        actual_rms = np.sqrt(np.mean(result ** 2))
        self.assertAlmostEqual(actual_rms, 0.01, places=2)

    def test_rms_normalization_high_target(self):
        """RMS normalization should work with high target levels."""
        audio = np.random.normal(0, 0.05, 16000).astype(np.float32)

        result = normalize_rms(audio, target_rms=0.3)

        actual_rms = np.sqrt(np.mean(result ** 2))
        # High targets may be clipped by soft clipping, allow more tolerance
        self.assertAlmostEqual(actual_rms, 0.3, places=1)


class TestCombinedEnhancements(unittest.TestCase):
    """Test combined audio enhancements."""

    def test_enhancements_with_peak_normalization(self):
        """Combined enhancements should work with peak normalization."""
        audio = np.random.normal(0, 0.1, 16000).astype(np.float32)

        result = apply_audio_enhancements(
            audio,
            sample_rate=16000,
            enable_vad=False,
            enable_noise_reduction=False,
            normalization_method="peak"
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, audio.shape)

    def test_enhancements_with_rms_normalization(self):
        """Combined enhancements should work with RMS normalization."""
        audio = np.random.normal(0, 0.1, 16000).astype(np.float32)

        result = apply_audio_enhancements(
            audio,
            sample_rate=16000,
            enable_vad=False,
            enable_noise_reduction=False,
            normalization_method="rms",
            rms_target_level=0.1
        )

        self.assertIsInstance(result, np.ndarray)

    def test_enhancements_with_noise_reduction(self):
        """Combined enhancements should apply noise reduction."""
        audio = np.random.normal(0, 0.1, 16000).astype(np.float32)

        result = apply_audio_enhancements(
            audio,
            sample_rate=16000,
            enable_vad=False,
            enable_noise_reduction=True,
            normalization_method="peak"
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, audio.shape)


if __name__ == "__main__":
    unittest.main()
